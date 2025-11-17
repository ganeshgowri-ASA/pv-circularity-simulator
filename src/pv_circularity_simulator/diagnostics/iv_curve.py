"""
IV Curve Analysis and Electrical Diagnostics for PV modules.

This module provides comprehensive IV curve analysis capabilities including:
- Curve tracing and parameter extraction (Voc, Isc, Vmp, Imp, FF, Rs, Rsh)
- Degradation analysis and trend detection
- Cell mismatch detection
- Electrical fault diagnostics
- Baseline comparison and anomaly detection
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

from pv_circularity_simulator.core.constants import (
    BOLTZMANN_CONSTANT,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    CONFIDENCE_THRESHOLD_MEDIUM,
    DEGRADATION_CRITICAL_THRESHOLD,
    DEGRADATION_WARNING_THRESHOLD,
    ELECTRON_CHARGE,
    IDEAL_FILL_FACTOR,
    MIN_FILL_FACTOR,
    SERIES_RESISTANCE_THRESHOLD,
    SHUNT_RESISTANCE_THRESHOLD,
    STANDARD_IRRADIANCE,
    STANDARD_TEMPERATURE,
    TYPICAL_FILL_FACTOR,
)
from pv_circularity_simulator.core.exceptions import (
    AnalysisError,
    InsufficientDataError,
    InvalidIVCurveError,
    ModelFittingError,
)
from pv_circularity_simulator.core.models import (
    DegradationAnalysis,
    ElectricalParameters,
    IVAnalysisResult,
    IVCurveData,
    SeverityLevel,
)
from pv_circularity_simulator.core.utils import (
    calculate_curve_quality,
    detect_outliers_iqr,
    interpolate_curve,
)


class IVCurveAnalyzer:
    """
    IV curve analysis with parameter extraction and diagnostics.

    This class provides comprehensive IV curve analysis including curve tracing,
    parameter extraction, and quality assessment.
    """

    def __init__(self, num_cells: int = 60, cell_area_cm2: float = 243.36):
        """
        Initialize IV curve analyzer.

        Args:
            num_cells: Number of cells in series in the module
            cell_area_cm2: Area of each cell in cm²
        """
        self.num_cells = num_cells
        self.cell_area_cm2 = cell_area_cm2
        self.module_area_m2 = num_cells * cell_area_cm2 / 10000  # Convert to m²

    def curve_tracing(
        self, iv_data: IVCurveData, smooth: bool = True, remove_outliers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trace and process IV curve data.

        Performs smoothing, outlier removal, and interpolation to create
        a clean, analyzable IV curve.

        Args:
            iv_data: Raw IV curve measurement data
            smooth: Apply smoothing to reduce noise
            remove_outliers: Remove outlier points

        Returns:
            Tuple of (voltage, current, power) arrays

        Raises:
            InvalidIVCurveError: If curve data is invalid

        Examples:
            >>> analyzer = IVCurveAnalyzer()
            >>> from datetime import datetime
            >>> v = np.linspace(0, 30, 100)
            >>> i = 5 * (1 - v/30) + np.random.normal(0, 0.01, 100)
            >>> iv_data = IVCurveData(
            ...     voltage=v, current=i, temperature=25.0,
            ...     irradiance=1000.0, timestamp=datetime.now()
            ... )
            >>> v_clean, i_clean, p = analyzer.curve_tracing(iv_data)
            >>> len(v_clean) > 0
            True
        """
        try:
            voltage = iv_data.voltage.copy()
            current = iv_data.current.copy()

            # Validate data
            if len(voltage) != len(current):
                raise InvalidIVCurveError("Voltage and current arrays must have same length")

            if len(voltage) < 10:
                raise InvalidIVCurveError("Insufficient data points (need at least 10)")

            # Sort by voltage
            sort_idx = np.argsort(voltage)
            voltage = voltage[sort_idx]
            current = current[sort_idx]

            # Remove negative currents (measurement errors)
            valid_mask = current >= 0
            voltage = voltage[valid_mask]
            current = current[valid_mask]

            if len(voltage) < 10:
                raise InvalidIVCurveError("Too few valid points after filtering")

            # Remove outliers
            if remove_outliers:
                current_outliers = detect_outliers_iqr(current)
                valid_mask = ~current_outliers
                voltage = voltage[valid_mask]
                current = current[valid_mask]

            # Smooth the curve
            if smooth and len(current) >= 11:
                # Savitzky-Golay filter for smoothing while preserving features
                window_length = min(11, len(current) if len(current) % 2 == 1 else len(current) - 1)
                current = savgol_filter(current, window_length, 3)

            # Calculate power
            power = voltage * current

            return voltage, current, power

        except Exception as e:
            if isinstance(e, InvalidIVCurveError):
                raise
            raise InvalidIVCurveError(f"Curve tracing failed: {str(e)}")

    def parameter_extraction(
        self, iv_data: IVCurveData, extract_resistances: bool = True
    ) -> ElectricalParameters:
        """
        Extract electrical parameters from IV curve.

        Extracts: Voc, Isc, Vmp, Imp, Pmp, FF, and optionally Rs and Rsh.

        Args:
            iv_data: IV curve measurement data
            extract_resistances: If True, extract series and shunt resistances

        Returns:
            Extracted electrical parameters

        Raises:
            InvalidIVCurveError: If parameter extraction fails

        Examples:
            >>> analyzer = IVCurveAnalyzer()
            >>> # params = analyzer.parameter_extraction(iv_data)
            >>> # params.fill_factor > 0
            True
        """
        try:
            # Trace and clean the curve
            voltage, current, power = self.curve_tracing(iv_data)

            # Extract basic parameters
            voc = self._extract_voc(voltage, current)
            isc = self._extract_isc(voltage, current)
            vmp, imp, pmp = self._extract_mpp(voltage, current, power)

            # Calculate fill factor
            fill_factor = pmp / (voc * isc) if (voc * isc) > 0 else 0.0

            # Calculate efficiency if irradiance is available
            efficiency = None
            if iv_data.irradiance > 0:
                efficiency = pmp / (iv_data.irradiance * self.module_area_m2)

            # Extract resistances if requested
            rs, rsh, ideality = None, None, None
            if extract_resistances:
                try:
                    rs = self._extract_series_resistance(voltage, current, voc, isc)
                    rsh = self._extract_shunt_resistance(voltage, current)
                    ideality = self._extract_ideality_factor(
                        voltage, current, iv_data.temperature
                    )
                except Exception:
                    # Resistance extraction is optional, continue without it
                    pass

            params = ElectricalParameters(
                voc=float(voc),
                isc=float(isc),
                vmp=float(vmp),
                imp=float(imp),
                pmp=float(pmp),
                fill_factor=float(fill_factor),
                efficiency=float(efficiency) if efficiency is not None else None,
                rs=float(rs) if rs is not None else None,
                rsh=float(rsh) if rsh is not None else None,
                ideality_factor=float(ideality) if ideality is not None else None,
            )

            return params

        except Exception as e:
            if isinstance(e, InvalidIVCurveError):
                raise
            raise InvalidIVCurveError(f"Parameter extraction failed: {str(e)}")

    def _extract_voc(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Extract open circuit voltage (Voc)."""
        # Find where current crosses zero
        if current[0] < 1e-6:  # First point is already at Voc
            return voltage[0]

        # Interpolate to find exact Voc
        f = interp1d(current[::-1], voltage[::-1], kind="linear", fill_value="extrapolate")
        voc = float(f(0))

        # Validate
        if voc <= 0 or voc > voltage[-1] * 1.1:
            # Fallback: use last voltage value
            voc = voltage[-1]

        return voc

    def _extract_isc(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Extract short circuit current (Isc)."""
        # Find current at V=0
        if voltage[0] < 1e-6:  # First point is at V=0
            return current[0]

        # Interpolate to find exact Isc
        f = interp1d(voltage, current, kind="linear", fill_value="extrapolate")
        isc = float(f(0))

        # Validate
        if isc <= 0:
            # Fallback: use first current value
            isc = current[0]

        return isc

    def _extract_mpp(
        self, voltage: np.ndarray, current: np.ndarray, power: np.ndarray
    ) -> Tuple[float, float, float]:
        """Extract maximum power point (Vmp, Imp, Pmp)."""
        # Find point with maximum power
        max_idx = np.argmax(power)
        vmp = voltage[max_idx]
        imp = current[max_idx]
        pmp = power[max_idx]

        # Refine using parabolic fit around the maximum
        if max_idx > 0 and max_idx < len(power) - 1:
            # Use 3-point parabolic interpolation
            v_points = voltage[max_idx - 1 : max_idx + 2]
            p_points = power[max_idx - 1 : max_idx + 2]

            # Fit parabola: P = a*V^2 + b*V + c
            coeffs = np.polyfit(v_points, p_points, 2)
            if coeffs[0] < 0:  # Ensure it's a maximum (negative leading coefficient)
                # Vertex of parabola: V = -b/(2a)
                vmp_refined = -coeffs[1] / (2 * coeffs[0])

                # Ensure refined value is within bounds
                if v_points[0] <= vmp_refined <= v_points[-1]:
                    vmp = vmp_refined
                    # Interpolate current at refined Vmp
                    f_i = interp1d(voltage, current, kind="cubic")
                    imp = float(f_i(vmp))
                    pmp = vmp * imp

        return float(vmp), float(imp), float(pmp)

    def _extract_series_resistance(
        self, voltage: np.ndarray, current: np.ndarray, voc: float, isc: float
    ) -> float:
        """
        Extract series resistance using slope at Voc.

        Rs ≈ -dV/dI at I=0 (near Voc)
        """
        # Use points near Voc (last 10% of voltage range)
        voc_region = voltage > (0.9 * voc)
        if np.sum(voc_region) < 3:
            return 0.0

        v_region = voltage[voc_region]
        i_region = current[voc_region]

        # Calculate slope dV/dI
        if len(v_region) >= 2:
            slope, _, _, _, _ = linregress(i_region, v_region)
            rs = abs(slope)
            return float(rs)

        return 0.0

    def _extract_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """
        Extract shunt resistance using slope near Isc.

        Rsh ≈ dV/dI at V=0 (near Isc)
        """
        # Use points near Isc (first 10% of voltage range)
        isc_region = voltage < (0.1 * voltage[-1])
        if np.sum(isc_region) < 3:
            return 1000.0  # Default high value

        v_region = voltage[isc_region]
        i_region = current[isc_region]

        # Calculate slope dV/dI
        if len(v_region) >= 2:
            slope, _, _, _, _ = linregress(i_region, v_region)
            rsh = abs(slope)
            return float(rsh)

        return 1000.0

    def _extract_ideality_factor(
        self, voltage: np.ndarray, current: np.ndarray, temperature: float
    ) -> float:
        """
        Extract diode ideality factor from exponential region.

        Uses the relationship: I = Iph - I0*[exp(q*V/(n*k*T)) - 1]
        """
        # Convert temperature to Kelvin
        temp_kelvin = temperature + 273.15

        # Thermal voltage: Vt = kT/q
        vt = (BOLTZMANN_CONSTANT * temp_kelvin) / ELECTRON_CHARGE

        # Use middle region of curve (exponential region)
        mid_start = len(voltage) // 3
        mid_end = 2 * len(voltage) // 3

        v_region = voltage[mid_start:mid_end]
        i_region = current[mid_start:mid_end]

        if len(v_region) < 3:
            return 1.3  # Default typical value

        # Linearize: ln(Iph - I) ≈ ln(I0) + V/(n*Vt)
        iph = current[0]  # Approximate photocurrent as Isc
        i_diff = iph - i_region

        # Filter out non-positive values
        valid = i_diff > 0
        if np.sum(valid) < 3:
            return 1.3

        v_valid = v_region[valid]
        ln_i = np.log(i_diff[valid])

        # Fit line
        slope, _, _, _, _ = linregress(v_valid, ln_i)

        # Extract ideality factor: n = 1/(slope * Vt)
        if slope > 0:
            ideality = 1.0 / (slope * vt)
            # Typical range is 1.0 to 2.0
            ideality = np.clip(ideality, 1.0, 2.0)
            return float(ideality)

        return 1.3

    def degradation_analysis(
        self, current_params: ElectricalParameters, baseline_params: ElectricalParameters
    ) -> DegradationAnalysis:
        """
        Analyze degradation by comparing current parameters to baseline.

        Args:
            current_params: Current electrical parameters
            baseline_params: Baseline (original or expected) parameters

        Returns:
            Degradation analysis results

        Examples:
            >>> analyzer = IVCurveAnalyzer()
            >>> # Create baseline and current params
            >>> # degradation = analyzer.degradation_analysis(current, baseline)
            >>> # degradation.severity in SeverityLevel
            True
        """
        # Calculate degradation percentages
        power_deg = (baseline_params.pmp - current_params.pmp) / baseline_params.pmp * 100
        voc_deg = (baseline_params.voc - current_params.voc) / baseline_params.voc * 100
        isc_deg = (baseline_params.isc - current_params.isc) / baseline_params.isc * 100
        ff_deg = (
            (baseline_params.fill_factor - current_params.fill_factor)
            / baseline_params.fill_factor
            * 100
        )

        # Determine severity
        abs_power_deg = abs(power_deg) / 100
        if abs_power_deg < DEGRADATION_WARNING_THRESHOLD:
            severity = SeverityLevel.NORMAL
        elif abs_power_deg < DEGRADATION_CRITICAL_THRESHOLD:
            severity = SeverityLevel.WARNING
        else:
            severity = SeverityLevel.SEVERE

        degradation = DegradationAnalysis(
            power_degradation_percent=float(power_deg),
            voc_degradation_percent=float(voc_deg),
            isc_degradation_percent=float(isc_deg),
            ff_degradation_percent=float(ff_deg),
            degradation_rate_per_year=None,  # Requires time-series data
            estimated_remaining_life_years=None,
            severity=severity,
        )

        return degradation

    def mismatch_detection(self, iv_data: IVCurveData) -> Dict[str, any]:
        """
        Detect cell mismatch from IV curve characteristics.

        Cell mismatch often appears as steps or irregularities in the IV curve.

        Args:
            iv_data: IV curve measurement data

        Returns:
            Dictionary with mismatch detection results

        Examples:
            >>> analyzer = IVCurveAnalyzer()
            >>> # result = analyzer.mismatch_detection(iv_data)
            >>> # 'mismatch_detected' in result
            True
        """
        voltage, current, power = self.curve_tracing(iv_data)

        # Calculate second derivative to detect steps
        if len(current) > 4:
            second_derivative = np.diff(current, 2)
            # Normalize by current range
            current_range = np.max(current) - np.min(current)
            if current_range > 0:
                normalized_derivative = second_derivative / current_range
                max_step = np.max(np.abs(normalized_derivative))

                # Threshold for mismatch detection
                mismatch_detected = max_step > 0.1

                return {
                    "mismatch_detected": bool(mismatch_detected),
                    "max_step_magnitude": float(max_step),
                    "confidence": float(min(max_step / 0.2, 1.0)),
                    "description": "Cell mismatch detected - irregular current steps"
                    if mismatch_detected
                    else "No significant mismatch detected",
                }

        return {
            "mismatch_detected": False,
            "max_step_magnitude": 0.0,
            "confidence": 0.0,
            "description": "Insufficient data for mismatch detection",
        }


class ElectricalDiagnostics:
    """
    Electrical diagnostics for fault detection and characterization.

    This class provides methods to diagnose common electrical faults in PV
    systems including string underperformance, cell failures, and bypass diode issues.
    """

    def __init__(self, analyzer: Optional[IVCurveAnalyzer] = None):
        """
        Initialize electrical diagnostics.

        Args:
            analyzer: IV curve analyzer instance (creates default if None)
        """
        self.analyzer = analyzer or IVCurveAnalyzer()

    def string_underperformance(
        self, string_data: List[IVCurveData], expected_params: ElectricalParameters
    ) -> Dict[str, any]:
        """
        Detect and diagnose string underperformance.

        Args:
            string_data: List of IV curve data for modules in string
            expected_params: Expected electrical parameters

        Returns:
            Dictionary with underperformance analysis

        Examples:
            >>> diagnostics = ElectricalDiagnostics()
            >>> # result = diagnostics.string_underperformance(string_data, expected)
            >>> # 'underperforming_modules' in result
            True
        """
        if len(string_data) == 0:
            raise InsufficientDataError("No string data provided")

        underperforming = []
        results = []

        for idx, iv_data in enumerate(string_data):
            try:
                params = self.analyzer.parameter_extraction(iv_data)

                # Compare to expected
                power_ratio = params.pmp / expected_params.pmp
                voc_ratio = params.voc / expected_params.voc
                isc_ratio = params.isc / expected_params.isc

                is_underperforming = power_ratio < 0.90  # More than 10% below expected

                module_result = {
                    "module_index": idx,
                    "power_ratio": float(power_ratio),
                    "voc_ratio": float(voc_ratio),
                    "isc_ratio": float(isc_ratio),
                    "underperforming": bool(is_underperforming),
                    "severity": (
                        SeverityLevel.SEVERE
                        if power_ratio < 0.80
                        else SeverityLevel.WARNING if is_underperforming else SeverityLevel.NORMAL
                    ).value,
                }

                results.append(module_result)

                if is_underperforming:
                    underperforming.append(idx)

            except Exception as e:
                results.append(
                    {
                        "module_index": idx,
                        "error": str(e),
                        "underperforming": None,
                    }
                )

        return {
            "total_modules": len(string_data),
            "underperforming_count": len(underperforming),
            "underperforming_modules": underperforming,
            "module_results": results,
            "string_health": (
                SeverityLevel.SEVERE
                if len(underperforming) > len(string_data) / 2
                else SeverityLevel.WARNING
                if len(underperforming) > 0
                else SeverityLevel.NORMAL
            ).value,
        }

    def cell_failures(self, iv_data: IVCurveData) -> Dict[str, any]:
        """
        Detect cell failures from IV curve signatures.

        Cell failures typically manifest as:
        - Reduced Isc (cell cracking, disconnection)
        - Reduced Voc (shunting)
        - Steps in IV curve (bypass diode activation)

        Args:
            iv_data: IV curve measurement data

        Returns:
            Dictionary with cell failure analysis

        Examples:
            >>> diagnostics = ElectricalDiagnostics()
            >>> # result = diagnostics.cell_failures(iv_data)
            >>> # 'failure_detected' in result
            True
        """
        try:
            params = self.analyzer.parameter_extraction(iv_data)
            mismatch = self.analyzer.mismatch_detection(iv_data)

            failures = []

            # Check for low fill factor (indicates shunting or high resistance)
            if params.fill_factor < MIN_FILL_FACTOR:
                failures.append(
                    {
                        "type": "low_fill_factor",
                        "value": params.fill_factor,
                        "severity": SeverityLevel.SEVERE.value,
                        "description": "Low fill factor indicates cell shunting or high series resistance",
                    }
                )

            # Check for high series resistance
            if params.rs is not None and params.rs > SERIES_RESISTANCE_THRESHOLD:
                failures.append(
                    {
                        "type": "high_series_resistance",
                        "value": params.rs,
                        "severity": SeverityLevel.WARNING.value,
                        "description": "High series resistance detected - possible connection issues",
                    }
                )

            # Check for low shunt resistance
            if params.rsh is not None and params.rsh < SHUNT_RESISTANCE_THRESHOLD:
                failures.append(
                    {
                        "type": "low_shunt_resistance",
                        "value": params.rsh,
                        "severity": SeverityLevel.SEVERE.value,
                        "description": "Low shunt resistance indicates cell damage or defects",
                    }
                )

            # Check for cell mismatch
            if mismatch["mismatch_detected"]:
                failures.append(
                    {
                        "type": "cell_mismatch",
                        "severity": SeverityLevel.WARNING.value,
                        "description": mismatch["description"],
                    }
                )

            overall_severity = (
                max((f.get("severity", "normal") for f in failures), default=SeverityLevel.NORMAL.value)
                if failures
                else SeverityLevel.NORMAL.value
            )

            return {
                "failure_detected": len(failures) > 0,
                "failure_count": len(failures),
                "failures": failures,
                "overall_severity": overall_severity,
                "electrical_parameters": params.model_dump(),
            }

        except Exception as e:
            raise AnalysisError(f"Cell failure analysis failed: {str(e)}")

    def bypass_diode_issues(self, iv_data: IVCurveData) -> Dict[str, any]:
        """
        Detect bypass diode issues from IV curve characteristics.

        Bypass diode issues manifest as:
        - Steps in IV curve (diode activated under normal conditions)
        - Voltage loss (shorted diode)
        - Overheating patterns

        Args:
            iv_data: IV curve measurement data

        Returns:
            Dictionary with bypass diode analysis

        Examples:
            >>> diagnostics = ElectricalDiagnostics()
            >>> # result = diagnostics.bypass_diode_issues(iv_data)
            >>> # 'diode_issues_detected' in result
            True
        """
        try:
            voltage, current, power = self.analyzer.curve_tracing(iv_data)

            issues = []

            # Detect voltage steps (indicates bypass diode activation)
            if len(voltage) > 10:
                voltage_diff = np.diff(voltage)
                # Normalize by expected voltage step between points
                expected_step = voltage_diff.mean()

                # Look for abnormally large steps
                large_steps = voltage_diff > (expected_step * 3)

                if np.any(large_steps):
                    step_indices = np.where(large_steps)[0]
                    issues.append(
                        {
                            "type": "bypass_diode_activation",
                            "severity": SeverityLevel.WARNING.value,
                            "step_count": int(np.sum(large_steps)),
                            "step_locations": step_indices.tolist(),
                            "description": "Bypass diode activation detected - possible cell shading or damage",
                        }
                    )

            # Check for abnormally low Voc (shorted bypass diode)
            params = self.analyzer.parameter_extraction(iv_data)
            expected_voc_per_cell = 0.6  # Typical for silicon
            expected_voc = expected_voc_per_cell * self.analyzer.num_cells

            if params.voc < (expected_voc * 0.7):  # More than 30% below expected
                issues.append(
                    {
                        "type": "possible_shorted_diode",
                        "severity": SeverityLevel.SEVERE.value,
                        "measured_voc": params.voc,
                        "expected_voc": expected_voc,
                        "description": "Abnormally low Voc - possible shorted bypass diode",
                    }
                )

            return {
                "diode_issues_detected": len(issues) > 0,
                "issue_count": len(issues),
                "issues": issues,
                "overall_severity": (
                    max((i.get("severity", "normal") for i in issues), default=SeverityLevel.NORMAL.value)
                    if issues
                    else SeverityLevel.NORMAL.value
                ),
            }

        except Exception as e:
            raise AnalysisError(f"Bypass diode analysis failed: {str(e)}")


class CurveComparison:
    """
    Compare IV curves for baseline comparison, trend analysis, and anomaly detection.

    This class provides methods to compare IV curves over time to detect degradation
    trends and identify anomalous behavior.
    """

    def __init__(self, analyzer: Optional[IVCurveAnalyzer] = None):
        """
        Initialize curve comparison.

        Args:
            analyzer: IV curve analyzer instance (creates default if None)
        """
        self.analyzer = analyzer or IVCurveAnalyzer()

    def baseline_comparison(
        self, current_data: IVCurveData, baseline_data: IVCurveData
    ) -> Dict[str, any]:
        """
        Compare current IV curve to baseline.

        Args:
            current_data: Current IV curve measurement
            baseline_data: Baseline IV curve measurement

        Returns:
            Dictionary with comparison results including degradation analysis

        Examples:
            >>> comparison = CurveComparison()
            >>> # result = comparison.baseline_comparison(current, baseline)
            >>> # 'degradation_analysis' in result
            True
        """
        try:
            # Extract parameters
            current_params = self.analyzer.parameter_extraction(current_data)
            baseline_params = self.analyzer.parameter_extraction(baseline_data)

            # Perform degradation analysis
            degradation = self.analyzer.degradation_analysis(current_params, baseline_params)

            # Calculate curve similarity
            v_current, i_current, _ = self.analyzer.curve_tracing(current_data)
            v_baseline, i_baseline, _ = self.analyzer.curve_tracing(baseline_data)

            # Interpolate to common voltage grid
            v_interp, i_current_interp = interpolate_curve(v_current, i_current, 100)
            _, i_baseline_interp = interpolate_curve(v_baseline, i_baseline, 100)

            # Calculate normalized root mean square deviation
            nrmsd = np.sqrt(np.mean((i_current_interp - i_baseline_interp) ** 2)) / np.mean(
                i_baseline_interp
            )

            similarity_score = max(0, 1 - nrmsd)

            return {
                "current_parameters": current_params.model_dump(),
                "baseline_parameters": baseline_params.model_dump(),
                "degradation_analysis": degradation.model_dump(),
                "curve_similarity_score": float(similarity_score),
                "nrmsd": float(nrmsd),
                "overall_health": degradation.severity.value,
            }

        except Exception as e:
            raise AnalysisError(f"Baseline comparison failed: {str(e)}")

    def trend_analysis(self, historical_data: List[Tuple[float, IVCurveData]]) -> Dict[str, any]:
        """
        Analyze trends in IV curves over time.

        Args:
            historical_data: List of (timestamp, IV curve data) tuples, sorted by time

        Returns:
            Dictionary with trend analysis results including degradation rate

        Raises:
            InsufficientDataError: If not enough historical data

        Examples:
            >>> comparison = CurveComparison()
            >>> # result = comparison.trend_analysis(historical)
            >>> # 'degradation_rate_per_year' in result
            True
        """
        if len(historical_data) < 2:
            raise InsufficientDataError("Need at least 2 data points for trend analysis")

        try:
            # Extract parameters for all curves
            timestamps = []
            powers = []
            vocs = []
            iscs = []
            ffs = []

            for timestamp, iv_data in historical_data:
                params = self.analyzer.parameter_extraction(iv_data)
                timestamps.append(timestamp)
                powers.append(params.pmp)
                vocs.append(params.voc)
                iscs.append(params.isc)
                ffs.append(params.fill_factor)

            timestamps = np.array(timestamps)
            powers = np.array(powers)
            vocs = np.array(vocs)
            iscs = np.array(iscs)
            ffs = np.array(ffs)

            # Normalize time to years from first measurement
            time_years = (timestamps - timestamps[0]) / (365.25 * 24 * 3600)  # Convert to years

            # Fit linear trends
            power_trend = linregress(time_years, powers)
            voc_trend = linregress(time_years, vocs)
            isc_trend = linregress(time_years, iscs)
            ff_trend = linregress(time_years, ffs)

            # Calculate degradation rates (% per year)
            power_deg_rate = (power_trend.slope / powers[0]) * 100
            voc_deg_rate = (voc_trend.slope / vocs[0]) * 100
            isc_deg_rate = (isc_trend.slope / iscs[0]) * 100
            ff_deg_rate = (ff_trend.slope / ffs[0]) * 100

            # Estimate remaining life (when power drops below 80% of initial)
            if power_trend.slope < 0:
                years_to_80_percent = (0.80 * powers[0] - powers[0]) / power_trend.slope
                remaining_life_years = max(0, years_to_80_percent)
            else:
                remaining_life_years = None  # No degradation or improving

            # Determine severity based on degradation rate
            abs_power_deg = abs(power_deg_rate)
            if abs_power_deg < 0.5:  # < 0.5% per year
                severity = SeverityLevel.NORMAL
            elif abs_power_deg < 1.0:  # < 1% per year
                severity = SeverityLevel.WARNING
            else:
                severity = SeverityLevel.SEVERE

            return {
                "num_measurements": len(historical_data),
                "time_span_years": float(time_years[-1]),
                "power_degradation_rate_per_year": float(power_deg_rate),
                "voc_degradation_rate_per_year": float(voc_deg_rate),
                "isc_degradation_rate_per_year": float(isc_deg_rate),
                "ff_degradation_rate_per_year": float(ff_deg_rate),
                "estimated_remaining_life_years": (
                    float(remaining_life_years) if remaining_life_years is not None else None
                ),
                "trend_confidence": float(power_trend.rvalue**2),  # R²
                "severity": severity.value,
                "trends": {
                    "power": {
                        "slope": float(power_trend.slope),
                        "r_squared": float(power_trend.rvalue**2),
                    },
                    "voc": {"slope": float(voc_trend.slope), "r_squared": float(voc_trend.rvalue**2)},
                    "isc": {"slope": float(isc_trend.slope), "r_squared": float(isc_trend.rvalue**2)},
                    "ff": {"slope": float(ff_trend.slope), "r_squared": float(ff_trend.rvalue**2)},
                },
            }

        except Exception as e:
            raise AnalysisError(f"Trend analysis failed: {str(e)}")

    def anomaly_detection(
        self, current_data: IVCurveData, historical_data: List[IVCurveData]
    ) -> Dict[str, any]:
        """
        Detect anomalies by comparing current curve to historical distribution.

        Args:
            current_data: Current IV curve measurement
            historical_data: List of historical IV curve measurements

        Returns:
            Dictionary with anomaly detection results

        Raises:
            InsufficientDataError: If not enough historical data

        Examples:
            >>> comparison = CurveComparison()
            >>> # result = comparison.anomaly_detection(current, historical)
            >>> # 'is_anomaly' in result
            True
        """
        if len(historical_data) < 3:
            raise InsufficientDataError("Need at least 3 historical curves for anomaly detection")

        try:
            # Extract current parameters
            current_params = self.analyzer.parameter_extraction(current_data)

            # Extract historical parameters
            historical_powers = []
            historical_vocs = []
            historical_iscs = []
            historical_ffs = []

            for iv_data in historical_data:
                params = self.analyzer.parameter_extraction(iv_data)
                historical_powers.append(params.pmp)
                historical_vocs.append(params.voc)
                historical_iscs.append(params.isc)
                historical_ffs.append(params.fill_factor)

            # Calculate Z-scores
            power_z = self._calculate_zscore(current_params.pmp, historical_powers)
            voc_z = self._calculate_zscore(current_params.voc, historical_vocs)
            isc_z = self._calculate_zscore(current_params.isc, historical_iscs)
            ff_z = self._calculate_zscore(current_params.fill_factor, historical_ffs)

            # Determine if anomaly (Z-score > 3)
            anomaly_threshold = 3.0
            is_power_anomaly = abs(power_z) > anomaly_threshold
            is_voc_anomaly = abs(voc_z) > anomaly_threshold
            is_isc_anomaly = abs(isc_z) > anomaly_threshold
            is_ff_anomaly = abs(ff_z) > anomaly_threshold

            is_anomaly = any([is_power_anomaly, is_voc_anomaly, is_isc_anomaly, is_ff_anomaly])

            anomalies = []
            if is_power_anomaly:
                anomalies.append({"parameter": "power", "z_score": power_z})
            if is_voc_anomaly:
                anomalies.append({"parameter": "voc", "z_score": voc_z})
            if is_isc_anomaly:
                anomalies.append({"parameter": "isc", "z_score": isc_z})
            if is_ff_anomaly:
                anomalies.append({"parameter": "fill_factor", "z_score": ff_z})

            # Calculate overall anomaly score
            max_z_score = max(abs(power_z), abs(voc_z), abs(isc_z), abs(ff_z))
            anomaly_score = min(max_z_score / anomaly_threshold, 1.0)

            severity = (
                SeverityLevel.SEVERE
                if anomaly_score > 0.8
                else SeverityLevel.WARNING if is_anomaly else SeverityLevel.NORMAL
            )

            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "anomalies": [
                    {"parameter": a["parameter"], "z_score": float(a["z_score"])} for a in anomalies
                ],
                "z_scores": {
                    "power": float(power_z),
                    "voc": float(voc_z),
                    "isc": float(isc_z),
                    "fill_factor": float(ff_z),
                },
                "severity": severity.value,
                "current_parameters": current_params.model_dump(),
            }

        except Exception as e:
            raise AnalysisError(f"Anomaly detection failed: {str(e)}")

    def _calculate_zscore(self, value: float, historical_values: List[float]) -> float:
        """Calculate Z-score for a value compared to historical distribution."""
        historical_array = np.array(historical_values)
        mean = np.mean(historical_array)
        std = np.std(historical_array)

        if std == 0:
            return 0.0

        z_score = (value - mean) / std
        return float(z_score)
