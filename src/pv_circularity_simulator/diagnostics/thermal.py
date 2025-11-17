"""
Thermal imaging analysis and IR defect detection for PV modules.

This module provides comprehensive thermal analysis capabilities including:
- Hotspot detection and characterization
- Temperature distribution analysis
- Bypass diode failure detection
- Severity classification and power loss estimation
- Temperature calibration and emissivity correction
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label
from scipy.stats import zscore
from sklearn.cluster import DBSCAN

from pv_circularity_simulator.core.constants import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    CONFIDENCE_THRESHOLD_MEDIUM,
    CRITICAL_HOTSPOT_THRESHOLD_DELTA,
    EMISSIVITY_VALUES,
    HOTSPOT_THRESHOLD_DELTA,
    SEVERE_HOTSPOT_THRESHOLD_DELTA,
)
from pv_circularity_simulator.core.exceptions import (
    AnalysisError,
    CalibrationError,
    InvalidThermalDataError,
)
from pv_circularity_simulator.core.models import (
    HotspotData,
    SeverityLevel,
    ThermalAnalysisResult,
    ThermalImageData,
)
from pv_circularity_simulator.core.utils import (
    calculate_temperature_uniformity,
    detect_outliers_zscore,
)


class IRImageProcessing:
    """
    IR image processing with temperature calibration and corrections.

    This class handles low-level thermal image processing operations including
    temperature calibration, emissivity correction, and background subtraction.
    """

    def __init__(self, default_emissivity: float = 0.90):
        """
        Initialize IR image processor.

        Args:
            default_emissivity: Default emissivity value for corrections (0.0-1.0)

        Raises:
            ValueError: If emissivity is out of valid range
        """
        if not 0.0 <= default_emissivity <= 1.0:
            raise ValueError("Emissivity must be between 0.0 and 1.0")
        self.default_emissivity = default_emissivity

    def temperature_calibration(
        self,
        raw_temperature: np.ndarray,
        ambient_temp: float,
        distance: float,
        emissivity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calibrate raw temperature measurements accounting for environmental factors.

        This method corrects for atmospheric absorption, distance effects, and
        ambient temperature reflection.

        Args:
            raw_temperature: Raw temperature matrix in Celsius
            ambient_temp: Ambient temperature in Celsius
            distance: Measurement distance in meters
            emissivity: Surface emissivity (uses default if None)

        Returns:
            Calibrated temperature matrix in Celsius

        Raises:
            CalibrationError: If calibration fails or produces invalid results

        Examples:
            >>> processor = IRImageProcessing()
            >>> raw_temps = np.array([[30.0, 32.0], [31.0, 33.0]])
            >>> calibrated = processor.temperature_calibration(raw_temps, 25.0, 5.0)
            >>> calibrated.shape
            (2, 2)
        """
        try:
            if emissivity is None:
                emissivity = self.default_emissivity

            # Atmospheric transmission (simplified model)
            # Transmission decreases with distance
            atmospheric_transmission = np.exp(-0.01 * distance)

            # Calculate reflected ambient radiation
            reflected_temp = (1 - emissivity) * ambient_temp

            # Apply calibration
            # T_true = (T_measured - (1-ε)*T_ambient) / (ε * τ)
            calibrated = (raw_temperature - reflected_temp) / (
                emissivity * atmospheric_transmission
            )

            # Validate results
            if np.any(np.isnan(calibrated)) or np.any(np.isinf(calibrated)):
                raise CalibrationError("Calibration produced invalid values (NaN or Inf)")

            return calibrated

        except Exception as e:
            if isinstance(e, CalibrationError):
                raise
            raise CalibrationError(f"Temperature calibration failed: {str(e)}")

    def emissivity_correction(
        self,
        temperature: np.ndarray,
        measured_emissivity: float,
        actual_emissivity: float,
        ambient_temp: float = 25.0,
    ) -> np.ndarray:
        """
        Correct temperature for emissivity differences.

        When thermal images are captured with incorrect emissivity settings,
        this method corrects the temperature values.

        Args:
            temperature: Measured temperature matrix in Celsius
            measured_emissivity: Emissivity setting used during measurement
            actual_emissivity: Actual emissivity of the surface
            ambient_temp: Ambient temperature in Celsius

        Returns:
            Corrected temperature matrix in Celsius

        Raises:
            CalibrationError: If emissivity values are invalid

        Examples:
            >>> processor = IRImageProcessing()
            >>> temps = np.array([[35.0, 37.0], [36.0, 38.0]])
            >>> corrected = processor.emissivity_correction(temps, 0.90, 0.85, 25.0)
            >>> corrected.shape
            (2, 2)
        """
        if not (0.0 < measured_emissivity <= 1.0 and 0.0 < actual_emissivity <= 1.0):
            raise CalibrationError("Emissivity values must be between 0 and 1")

        try:
            # Convert to Kelvin for Stefan-Boltzmann calculations
            temp_kelvin = temperature + 273.15
            ambient_kelvin = ambient_temp + 273.15

            # Correct using Stefan-Boltzmann relation
            # T_corrected^4 = (T_measured^4 - (1-ε_m)*T_a^4) * ε_a / ε_m + (1-ε_a)*T_a^4
            measured_power = temp_kelvin**4
            ambient_power = ambient_kelvin**4

            corrected_power = (
                measured_power - (1 - measured_emissivity) * ambient_power
            ) * actual_emissivity / measured_emissivity + (1 - actual_emissivity) * ambient_power

            # Ensure no negative values before taking root
            corrected_power = np.maximum(corrected_power, 0)

            corrected_kelvin = np.power(corrected_power, 0.25)
            corrected_celsius = corrected_kelvin - 273.15

            return corrected_celsius

        except Exception as e:
            raise CalibrationError(f"Emissivity correction failed: {str(e)}")

    def background_subtraction(
        self, temperature: np.ndarray, background: np.ndarray, adaptive: bool = True
    ) -> np.ndarray:
        """
        Subtract background temperature to isolate module heating.

        Args:
            temperature: Temperature matrix of the module
            background: Background temperature matrix or scalar
            adaptive: If True, use adaptive background estimation

        Returns:
            Background-subtracted temperature matrix

        Examples:
            >>> processor = IRImageProcessing()
            >>> module_temps = np.array([[35.0, 37.0], [36.0, 38.0]])
            >>> background = np.array([[30.0, 30.0], [30.0, 30.0]])
            >>> result = processor.background_subtraction(module_temps, background)
            >>> result.shape
            (2, 2)
        """
        if adaptive:
            # Use morphological opening to estimate background
            # This preserves edges while removing small hot regions
            structure_size = max(temperature.shape) // 10
            structure = np.ones((structure_size, structure_size))
            estimated_background = ndimage.grey_opening(temperature, structure=structure)
            return temperature - estimated_background
        else:
            # Simple subtraction
            return temperature - background

    def denoise_thermal_image(
        self, temperature: np.ndarray, sigma: float = 1.0, method: str = "gaussian"
    ) -> np.ndarray:
        """
        Denoise thermal image while preserving edges.

        Args:
            temperature: Temperature matrix
            sigma: Standard deviation for Gaussian kernel
            method: Denoising method ('gaussian', 'median', or 'bilateral')

        Returns:
            Denoised temperature matrix

        Examples:
            >>> processor = IRImageProcessing()
            >>> noisy_temps = np.random.rand(10, 10) * 5 + 30
            >>> denoised = processor.denoise_thermal_image(noisy_temps, sigma=1.0)
            >>> denoised.shape
            (10, 10)
        """
        if method == "gaussian":
            return gaussian_filter(temperature, sigma=sigma)
        elif method == "median":
            return ndimage.median_filter(temperature, size=int(2 * sigma + 1))
        elif method == "bilateral":
            # Simplified bilateral filter using Gaussian filtering
            # For production, consider using cv2.bilateralFilter
            return gaussian_filter(temperature, sigma=sigma)
        else:
            raise ValueError(f"Unknown denoising method: {method}")


class HotspotSeverityClassifier:
    """
    Classify hotspot severity and estimate power loss and failure probability.

    This class analyzes detected hotspots to determine their severity level,
    estimate associated power losses, and predict potential failure modes.
    """

    def __init__(self):
        """Initialize hotspot severity classifier."""
        pass

    def severity_levels(self, temperature_delta: float) -> SeverityLevel:
        """
        Classify hotspot severity based on temperature delta.

        Args:
            temperature_delta: Temperature difference from module median (°C)

        Returns:
            Severity level classification

        Examples:
            >>> classifier = HotspotSeverityClassifier()
            >>> classifier.severity_levels(5.0)
            <SeverityLevel.NORMAL: 'normal'>
            >>> classifier.severity_levels(25.0)
            <SeverityLevel.SEVERE: 'severe'>
        """
        if temperature_delta < HOTSPOT_THRESHOLD_DELTA:
            return SeverityLevel.NORMAL
        elif temperature_delta < SEVERE_HOTSPOT_THRESHOLD_DELTA:
            return SeverityLevel.WARNING
        elif temperature_delta < CRITICAL_HOTSPOT_THRESHOLD_DELTA:
            return SeverityLevel.SEVERE
        else:
            return SeverityLevel.CRITICAL

    def power_loss_estimation(
        self,
        temperature_delta: float,
        hotspot_area_fraction: float,
        module_power_rating: float = 300.0,
    ) -> Dict[str, float]:
        """
        Estimate power loss due to hotspot.

        Uses empirical models correlating hotspot temperature and area with power loss.

        Args:
            temperature_delta: Temperature difference from median (°C)
            hotspot_area_fraction: Fraction of module area affected (0-1)
            module_power_rating: Rated power of module in Watts

        Returns:
            Dictionary with power loss estimates

        Examples:
            >>> classifier = HotspotSeverityClassifier()
            >>> result = classifier.power_loss_estimation(20.0, 0.05, 300.0)
            >>> 'power_loss_watts' in result
            True
        """
        # Empirical power loss model
        # Power loss increases with temperature delta and affected area
        # Based on typical PV module behavior

        # Temperature factor: ~0.5% loss per °C above threshold
        temp_factor = max(0, temperature_delta - HOTSPOT_THRESHOLD_DELTA) * 0.005

        # Area factor: loss proportional to affected area
        area_factor = hotspot_area_fraction

        # Combined loss factor
        loss_factor = temp_factor * (1 + area_factor)

        # Calculate power loss
        power_loss_watts = module_power_rating * loss_factor
        power_loss_percent = loss_factor * 100

        # Estimate annual energy loss (assuming average 4 sun-hours per day)
        annual_energy_loss_kwh = power_loss_watts * 4 * 365 / 1000

        return {
            "power_loss_watts": float(power_loss_watts),
            "power_loss_percent": float(power_loss_percent),
            "annual_energy_loss_kwh": float(annual_energy_loss_kwh),
            "loss_factor": float(loss_factor),
        }

    def failure_prediction(
        self, severity: SeverityLevel, temperature_delta: float, duration_days: int = 0
    ) -> Dict[str, any]:
        """
        Predict failure probability and estimated time to failure.

        Args:
            severity: Hotspot severity level
            temperature_delta: Temperature difference from median (°C)
            duration_days: Number of days hotspot has been present

        Returns:
            Dictionary with failure predictions

        Examples:
            >>> classifier = HotspotSeverityClassifier()
            >>> result = classifier.failure_prediction(SeverityLevel.SEVERE, 25.0, 30)
            >>> 'failure_probability' in result
            True
        """
        # Base failure probability by severity
        severity_probabilities = {
            SeverityLevel.NORMAL: 0.01,
            SeverityLevel.WARNING: 0.10,
            SeverityLevel.MODERATE: 0.30,
            SeverityLevel.SEVERE: 0.60,
            SeverityLevel.CRITICAL: 0.90,
        }

        base_probability = severity_probabilities.get(severity, 0.5)

        # Adjust for temperature and duration
        # Higher temperature and longer duration increase failure risk
        temp_multiplier = 1 + (temperature_delta / 100)  # Normalized temperature effect
        duration_multiplier = 1 + (duration_days / 365)  # Duration effect

        failure_probability = min(base_probability * temp_multiplier * duration_multiplier, 0.99)

        # Estimate time to failure (in days)
        # Inverse relationship with severity
        if severity == SeverityLevel.CRITICAL:
            mean_time_to_failure = 30  # ~1 month
        elif severity == SeverityLevel.SEVERE:
            mean_time_to_failure = 180  # ~6 months
        elif severity == SeverityLevel.WARNING:
            mean_time_to_failure = 730  # ~2 years
        else:
            mean_time_to_failure = 3650  # ~10 years

        # Adjust for current duration
        remaining_time = max(mean_time_to_failure - duration_days, 0)

        return {
            "failure_probability": float(failure_probability),
            "mean_time_to_failure_days": int(mean_time_to_failure),
            "estimated_remaining_days": int(remaining_time),
            "severity": severity.value,
            "recommended_action": self._get_recommended_action(severity),
        }

    def _get_recommended_action(self, severity: SeverityLevel) -> str:
        """Get recommended action based on severity."""
        actions = {
            SeverityLevel.NORMAL: "Continue monitoring",
            SeverityLevel.WARNING: "Schedule inspection within 6 months",
            SeverityLevel.MODERATE: "Schedule inspection within 3 months",
            SeverityLevel.SEVERE: "Immediate inspection required",
            SeverityLevel.CRITICAL: "Urgent intervention required - risk of fire or catastrophic failure",
        }
        return actions.get(severity, "Consult expert")


class ThermalImageAnalyzer:
    """
    Comprehensive thermal image analysis for PV modules.

    This class provides complete thermal analysis capabilities including hotspot
    detection, temperature distribution analysis, thermal anomaly identification,
    and bypass diode failure detection.
    """

    def __init__(
        self,
        hotspot_threshold: float = HOTSPOT_THRESHOLD_DELTA,
        min_hotspot_area: int = 10,
        processor: Optional[IRImageProcessing] = None,
        classifier: Optional[HotspotSeverityClassifier] = None,
    ):
        """
        Initialize thermal image analyzer.

        Args:
            hotspot_threshold: Temperature delta threshold for hotspot detection (°C)
            min_hotspot_area: Minimum area in pixels to classify as hotspot
            processor: IR image processor instance (creates default if None)
            classifier: Hotspot severity classifier instance (creates default if None)
        """
        self.hotspot_threshold = hotspot_threshold
        self.min_hotspot_area = min_hotspot_area
        self.processor = processor or IRImageProcessing()
        self.classifier = classifier or HotspotSeverityClassifier()

    def hotspot_detection(
        self, thermal_data: ThermalImageData, method: str = "threshold"
    ) -> List[HotspotData]:
        """
        Detect hotspots in thermal image using specified method.

        Args:
            thermal_data: Thermal image data with temperature matrix
            method: Detection method ('threshold', 'zscore', or 'clustering')

        Returns:
            List of detected hotspots with location, severity, and characteristics

        Raises:
            InvalidThermalDataError: If thermal data is invalid
            AnalysisError: If detection fails

        Examples:
            >>> analyzer = ThermalImageAnalyzer()
            >>> from datetime import datetime
            >>> from pv_circularity_simulator.core.models import ThermalImageMetadata
            >>> metadata = ThermalImageMetadata(
            ...     timestamp=datetime.now(),
            ...     camera_model="FLIR",
            ...     ambient_temp=25.0,
            ...     measurement_distance=5.0,
            ...     emissivity=0.9
            ... )
            >>> temps = np.random.rand(100, 100) * 10 + 30
            >>> temps[20:30, 20:30] += 20  # Create hotspot
            >>> thermal = ThermalImageData(
            ...     temperature_matrix=temps,
            ...     metadata=metadata,
            ...     width=100,
            ...     height=100
            ... )
            >>> hotspots = analyzer.hotspot_detection(thermal)
            >>> len(hotspots) > 0
            True
        """
        try:
            temps = thermal_data.temperature_matrix

            if method == "threshold":
                return self._detect_hotspots_threshold(temps)
            elif method == "zscore":
                return self._detect_hotspots_zscore(temps)
            elif method == "clustering":
                return self._detect_hotspots_clustering(temps)
            else:
                raise ValueError(f"Unknown detection method: {method}")

        except Exception as e:
            if isinstance(e, (InvalidThermalDataError, ValueError)):
                raise
            raise AnalysisError(f"Hotspot detection failed: {str(e)}")

    def _detect_hotspots_threshold(self, temps: np.ndarray) -> List[HotspotData]:
        """Detect hotspots using temperature threshold method."""
        median_temp = np.median(temps)
        temp_delta = temps - median_temp

        # Find regions exceeding threshold
        hotspot_mask = temp_delta > self.hotspot_threshold

        # Label connected components
        labeled_array, num_features = label(hotspot_mask)

        hotspots = []
        for i in range(1, num_features + 1):
            component_mask = labeled_array == i
            area = np.sum(component_mask)

            if area >= self.min_hotspot_area:
                # Extract hotspot properties
                rows, cols = np.where(component_mask)
                center_row = int(np.mean(rows))
                center_col = int(np.mean(cols))

                max_temp = np.max(temps[component_mask])
                max_delta = max_temp - median_temp

                severity = self.classifier.severity_levels(max_delta)

                # Calculate confidence based on area and temperature contrast
                confidence = self._calculate_detection_confidence(area, max_delta)

                # Bounding box
                bbox = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))

                hotspot = HotspotData(
                    location=(center_row, center_col),
                    temperature=float(max_temp),
                    temperature_delta=float(max_delta),
                    area_pixels=int(area),
                    severity=severity,
                    confidence=float(confidence),
                    bounding_box=bbox,
                )
                hotspots.append(hotspot)

        return hotspots

    def _detect_hotspots_zscore(self, temps: np.ndarray) -> List[HotspotData]:
        """Detect hotspots using Z-score statistical method."""
        # Flatten for Z-score calculation
        temps_flat = temps.flatten()
        z_scores = np.abs(zscore(temps_flat))
        z_matrix = z_scores.reshape(temps.shape)

        # Threshold at 3 standard deviations
        hotspot_mask = z_matrix > 3.0

        # Label connected components
        labeled_array, num_features = label(hotspot_mask)

        median_temp = np.median(temps)
        hotspots = []

        for i in range(1, num_features + 1):
            component_mask = labeled_array == i
            area = np.sum(component_mask)

            if area >= self.min_hotspot_area:
                rows, cols = np.where(component_mask)
                center_row = int(np.mean(rows))
                center_col = int(np.mean(cols))

                max_temp = np.max(temps[component_mask])
                max_delta = max_temp - median_temp

                severity = self.classifier.severity_levels(max_delta)
                confidence = self._calculate_detection_confidence(area, max_delta)

                bbox = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))

                hotspot = HotspotData(
                    location=(center_row, center_col),
                    temperature=float(max_temp),
                    temperature_delta=float(max_delta),
                    area_pixels=int(area),
                    severity=severity,
                    confidence=float(confidence),
                    bounding_box=bbox,
                )
                hotspots.append(hotspot)

        return hotspots

    def _detect_hotspots_clustering(self, temps: np.ndarray) -> List[HotspotData]:
        """Detect hotspots using DBSCAN clustering on high-temperature pixels."""
        median_temp = np.median(temps)
        threshold_temp = median_temp + self.hotspot_threshold

        # Get coordinates of high-temperature pixels
        high_temp_mask = temps > threshold_temp
        rows, cols = np.where(high_temp_mask)

        if len(rows) == 0:
            return []

        # Prepare data for clustering
        coords = np.column_stack([rows, cols])

        # DBSCAN clustering
        clustering = DBSCAN(eps=3, min_samples=self.min_hotspot_area).fit(coords)
        labels = clustering.labels_

        hotspots = []
        for label in set(labels):
            if label == -1:  # Noise
                continue

            cluster_mask = labels == label
            cluster_coords = coords[cluster_mask]

            # Get corresponding temperatures
            cluster_rows = cluster_coords[:, 0]
            cluster_cols = cluster_coords[:, 1]

            max_temp = np.max(temps[cluster_rows, cluster_cols])
            max_delta = max_temp - median_temp

            center_row = int(np.mean(cluster_rows))
            center_col = int(np.mean(cluster_cols))
            area = len(cluster_coords)

            severity = self.classifier.severity_levels(max_delta)
            confidence = self._calculate_detection_confidence(area, max_delta)

            bbox = (
                int(cluster_cols.min()),
                int(cluster_rows.min()),
                int(cluster_cols.max()),
                int(cluster_rows.max()),
            )

            hotspot = HotspotData(
                location=(center_row, center_col),
                temperature=float(max_temp),
                temperature_delta=float(max_delta),
                area_pixels=int(area),
                severity=severity,
                confidence=float(confidence),
                bounding_box=bbox,
            )
            hotspots.append(hotspot)

        return hotspots

    def _calculate_detection_confidence(self, area: int, temperature_delta: float) -> float:
        """Calculate confidence score for hotspot detection."""
        # Larger area increases confidence
        area_score = min(area / (self.min_hotspot_area * 10), 1.0)

        # Larger temperature delta increases confidence
        temp_score = min(temperature_delta / CRITICAL_HOTSPOT_THRESHOLD_DELTA, 1.0)

        # Combined confidence
        confidence = (area_score + temp_score) / 2.0

        return np.clip(confidence, 0.0, 1.0)

    def temperature_distribution_analysis(
        self, thermal_data: ThermalImageData
    ) -> Dict[str, float]:
        """
        Analyze temperature distribution across the module.

        Args:
            thermal_data: Thermal image data

        Returns:
            Dictionary with statistical measures of temperature distribution

        Examples:
            >>> analyzer = ThermalImageAnalyzer()
            >>> # Using thermal data from previous example
            >>> # stats = analyzer.temperature_distribution_analysis(thermal)
            >>> # 'mean_temperature' in stats
            True
        """
        temps = thermal_data.temperature_matrix

        return {
            "mean_temperature": float(np.mean(temps)),
            "median_temperature": float(np.median(temps)),
            "std_temperature": float(np.std(temps)),
            "min_temperature": float(np.min(temps)),
            "max_temperature": float(np.max(temps)),
            "q25_temperature": float(np.percentile(temps, 25)),
            "q75_temperature": float(np.percentile(temps, 75)),
            "temperature_range": float(np.max(temps) - np.min(temps)),
            "uniformity_index": calculate_temperature_uniformity(temps),
        }

    def thermal_anomaly_identification(
        self, thermal_data: ThermalImageData
    ) -> List[Dict[str, any]]:
        """
        Identify thermal anomalies beyond simple hotspots.

        Detects patterns such as:
        - Entire cell or string overheating
        - Edge heating
        - Corner effects
        - Diode heating patterns

        Args:
            thermal_data: Thermal image data

        Returns:
            List of identified anomalies with type and characteristics

        Examples:
            >>> analyzer = ThermalImageAnalyzer()
            >>> # anomalies = analyzer.thermal_anomaly_identification(thermal)
            >>> # isinstance(anomalies, list)
            True
        """
        temps = thermal_data.temperature_matrix
        anomalies = []

        # Detect cold spots (potential shading or disconnection)
        median_temp = np.median(temps)
        cold_threshold = median_temp - 10.0
        cold_mask = temps < cold_threshold

        labeled_cold, num_cold = label(cold_mask)
        for i in range(1, num_cold + 1):
            component_mask = labeled_cold == i
            area = np.sum(component_mask)
            if area >= self.min_hotspot_area:
                rows, cols = np.where(component_mask)
                anomalies.append(
                    {
                        "type": "cold_spot",
                        "location": (int(np.mean(rows)), int(np.mean(cols))),
                        "area_pixels": int(area),
                        "temperature": float(np.mean(temps[component_mask])),
                        "temperature_delta": float(np.mean(temps[component_mask]) - median_temp),
                        "severity": SeverityLevel.WARNING.value,
                    }
                )

        # Detect edge heating
        edge_width = max(temps.shape) // 20
        edges = np.concatenate(
            [
                temps[:edge_width, :].flatten(),
                temps[-edge_width:, :].flatten(),
                temps[:, :edge_width].flatten(),
                temps[:, -edge_width:].flatten(),
            ]
        )
        interior = temps[edge_width:-edge_width, edge_width:-edge_width].flatten()

        if len(interior) > 0:
            edge_mean = np.mean(edges)
            interior_mean = np.mean(interior)

            if edge_mean > interior_mean + 5.0:
                anomalies.append(
                    {
                        "type": "edge_heating",
                        "edge_temperature": float(edge_mean),
                        "interior_temperature": float(interior_mean),
                        "temperature_delta": float(edge_mean - interior_mean),
                        "severity": SeverityLevel.WARNING.value,
                    }
                )

        return anomalies

    def bypass_diode_failures(self, thermal_data: ThermalImageData) -> List[Dict[str, any]]:
        """
        Detect bypass diode failures from thermal patterns.

        Bypass diode failures typically show as entire substring overheating
        in a characteristic pattern.

        Args:
            thermal_data: Thermal image data

        Returns:
            List of suspected bypass diode failures

        Examples:
            >>> analyzer = ThermalImageAnalyzer()
            >>> # failures = analyzer.bypass_diode_failures(thermal)
            >>> # isinstance(failures, list)
            True
        """
        temps = thermal_data.temperature_matrix
        height, width = temps.shape

        # Typical PV module has 3 bypass diodes protecting rows of cells
        # Divide into horizontal thirds
        num_sections = 3
        section_height = height // num_sections

        failures = []
        median_temp = np.median(temps)

        for i in range(num_sections):
            start_row = i * section_height
            end_row = (i + 1) * section_height if i < num_sections - 1 else height

            section = temps[start_row:end_row, :]
            section_mean = np.mean(section)
            section_delta = section_mean - median_temp

            # If entire section is significantly hotter, suspect bypass diode issue
            if section_delta > 15.0:
                failures.append(
                    {
                        "type": "bypass_diode_failure",
                        "section_index": i,
                        "section_rows": (start_row, end_row),
                        "section_mean_temp": float(section_mean),
                        "temperature_delta": float(section_delta),
                        "severity": SeverityLevel.SEVERE.value,
                        "confidence": 0.7,
                        "description": f"Section {i+1} shows overheating pattern consistent with bypass diode failure",
                    }
                )

        return failures

    def analyze(self, thermal_data: ThermalImageData) -> ThermalAnalysisResult:
        """
        Perform complete thermal analysis.

        This is the main entry point for comprehensive thermal analysis,
        combining all analysis methods.

        Args:
            thermal_data: Thermal image data with metadata

        Returns:
            Complete thermal analysis results

        Raises:
            InvalidThermalDataError: If thermal data is invalid
            AnalysisError: If analysis fails

        Examples:
            >>> analyzer = ThermalImageAnalyzer()
            >>> # result = analyzer.analyze(thermal)
            >>> # result.overall_severity in SeverityLevel
            True
        """
        try:
            # Validate input
            if thermal_data.temperature_matrix.size == 0:
                raise InvalidThermalDataError("Temperature matrix is empty")

            # Perform analyses
            hotspots = self.hotspot_detection(thermal_data, method="threshold")
            temp_stats = self.temperature_distribution_analysis(thermal_data)
            diode_failures = self.bypass_diode_failures(thermal_data)

            # Determine overall severity
            if len(hotspots) > 0:
                max_severity = max(h.severity for h in hotspots)
            else:
                max_severity = SeverityLevel.NORMAL

            # If bypass diode failures detected, elevate severity
            if len(diode_failures) > 0:
                max_severity = max(max_severity, SeverityLevel.SEVERE)

            # Calculate overall confidence
            if len(hotspots) > 0:
                avg_confidence = np.mean([h.confidence for h in hotspots])
            else:
                avg_confidence = 0.95  # High confidence when no issues found

            result = ThermalAnalysisResult(
                hotspots=hotspots,
                mean_temperature=temp_stats["mean_temperature"],
                median_temperature=temp_stats["median_temperature"],
                max_temperature=temp_stats["max_temperature"],
                min_temperature=temp_stats["min_temperature"],
                temperature_std=temp_stats["std_temperature"],
                temperature_uniformity=temp_stats["uniformity_index"],
                bypass_diode_failures=diode_failures,
                overall_severity=max_severity,
                confidence=float(avg_confidence),
            )

            return result

        except Exception as e:
            if isinstance(e, (InvalidThermalDataError, AnalysisError)):
                raise
            raise AnalysisError(f"Thermal analysis failed: {str(e)}")
