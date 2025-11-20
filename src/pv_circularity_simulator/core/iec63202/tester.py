"""
IEC 63202 CTM Testing implementation.

This module implements the IEC 63202 standard for Cell-to-Module (CTM) power ratio
testing, including reference cell measurements, module flash testing, and comprehensive
CTM validation procedures.
"""

from datetime import datetime
from typing import List, Tuple, Optional
import logging

import numpy as np
from scipy import interpolate, optimize

from pv_circularity_simulator.core.iec63202.models import (
    CTMTestConfig,
    CTMTestResult,
    IVCurveData,
    CTMLossComponents,
    TestStatus,
    CTMCertificate,
)
from pv_circularity_simulator.core.utils.constants import (
    STC_IRRADIANCE,
    STC_TEMPERATURE,
    IEC_63202_COMPLIANCE,
    MEASUREMENT_UNCERTAINTY,
)

logger = logging.getLogger(__name__)


class IEC63202CTMTester:
    """
    IEC 63202 Cell-to-Module (CTM) testing and validation.

    This class implements the complete IEC 63202 CTM testing procedure including:
    - Reference cell measurement under STC
    - Module flash testing under STC
    - CTM power ratio calculation
    - Loss component analysis
    - Compliance validation against IEC 63202 requirements
    - Certificate generation

    Attributes:
        config: Test configuration
        reference_calibration: Reference device calibration handler
        power_loss_analyzer: CTM power loss analyzer
    """

    def __init__(
        self,
        config: CTMTestConfig,
        reference_calibration: Optional['ReferenceDeviceCalibration'] = None,
        power_loss_analyzer: Optional['CTMPowerLossAnalyzer'] = None
    ) -> None:
        """
        Initialize IEC 63202 CTM tester.

        Args:
            config: Test configuration per IEC 63202
            reference_calibration: Reference device calibration handler
            power_loss_analyzer: CTM power loss analyzer
        """
        self.config = config
        self.reference_calibration = reference_calibration
        self.power_loss_analyzer = power_loss_analyzer
        self.test_result: Optional[CTMTestResult] = None

        logger.info(
            f"IEC 63202 CTM Tester initialized for test {config.test_id}"
        )

    def reference_cell_measurement(
        self,
        voltage: List[float],
        current: List[float],
        temperature: float,
        irradiance: float
    ) -> IVCurveData:
        """
        Perform calibrated reference cell measurement under STC.

        This method measures the IV characteristic of reference cells under
        Standard Test Conditions (STC: 1000 W/m², 25°C, AM1.5 spectrum).
        Temperature and irradiance corrections are applied to normalize
        measurements to STC.

        Args:
            voltage: Measured voltage points (V)
            current: Measured current points (A)
            temperature: Cell temperature during measurement (°C)
            irradiance: Irradiance during measurement (W/m²)

        Returns:
            Calibrated and corrected IV curve data at STC

        Raises:
            ValueError: If measurement data is invalid

        Example:
            >>> tester = IEC63202CTMTester(config)
            >>> iv_curve = tester.reference_cell_measurement(
            ...     voltage=[0.0, 0.2, 0.4, 0.6, 0.65],
            ...     current=[8.5, 8.4, 8.0, 6.0, 0.0],
            ...     temperature=26.5,
            ...     irradiance=1005.0
            ... )
            >>> print(f"Pmax at STC: {iv_curve.pmax:.2f} W")
        """
        logger.info(
            f"Reference cell measurement: T={temperature}°C, G={irradiance}W/m²"
        )

        # Create initial IV curve
        iv_curve = IVCurveData(
            voltage=voltage,
            current=current,
            temperature=temperature,
            irradiance=irradiance,
        )

        # Apply corrections to STC
        corrected_iv = self._correct_to_stc(
            iv_curve,
            self.config.cell_properties.technology.value
        )

        # Apply reference device calibration if available
        if self.reference_calibration:
            corrected_iv = self.reference_calibration.apply_calibration(corrected_iv)

        logger.info(
            f"Cell measurement corrected to STC: Pmax={corrected_iv.pmax:.3f}W"
        )

        return corrected_iv

    def module_flash_test(
        self,
        voltage: List[float],
        current: List[float],
        temperature: float,
        irradiance: float
    ) -> IVCurveData:
        """
        Perform module IV curve measurement using flash simulator under STC.

        Module flash testing is performed using a solar simulator (Xenon or LED)
        calibrated to AM1.5 spectrum. The flash duration is typically 5-20 ms
        to minimize thermal effects. Measurements are corrected to STC conditions.

        Args:
            voltage: Measured voltage points (V)
            current: Measured current points (A)
            temperature: Module temperature during flash (°C)
            irradiance: Flash irradiance (W/m²)

        Returns:
            Module IV curve data corrected to STC

        Raises:
            ValueError: If measurement data is invalid or flash conditions
                       deviate too far from STC requirements

        Example:
            >>> tester = IEC63202CTMTester(config)
            >>> module_iv = tester.module_flash_test(
            ...     voltage=[0.0, 10.0, 20.0, 30.0, 35.0],
            ...     current=[8.5, 8.4, 8.0, 6.0, 0.0],
            ...     temperature=25.5,
            ...     irradiance=998.0
            ... )
            >>> print(f"Module Pmax: {module_iv.pmax:.2f} W")
        """
        logger.info(
            f"Module flash test: T={temperature}°C, G={irradiance}W/m²"
        )

        # Validate flash simulator conditions
        self._validate_flash_conditions(temperature, irradiance)

        # Create IV curve
        iv_curve = IVCurveData(
            voltage=voltage,
            current=current,
            temperature=temperature,
            irradiance=irradiance,
        )

        # Correct to STC
        corrected_iv = self._correct_to_stc(
            iv_curve,
            self.config.cell_properties.technology.value
        )

        # Apply spectral mismatch correction
        if self.power_loss_analyzer:
            spectral_factor = self.power_loss_analyzer.spectral_mismatch_factor(
                self.config.flash_simulator.spectral_distribution
            )
            corrected_iv = self._apply_spectral_correction(corrected_iv, spectral_factor)

        logger.info(
            f"Module measurement corrected to STC: Pmax={corrected_iv.pmax:.3f}W"
        )

        return corrected_iv

    def ctm_power_ratio_test(
        self,
        cell_measurements: List[IVCurveData],
        module_measurements: List[IVCurveData]
    ) -> CTMTestResult:
        """
        Measure CTM power ratio (Pmodule/Pcells) per IEC 63202.

        The CTM ratio is calculated as:
        CTM = P_module / (N_cells × P_cell_avg)

        where:
        - P_module: Measured module power at STC
        - N_cells: Total number of cells in module
        - P_cell_avg: Average power of reference cells at STC

        Args:
            cell_measurements: List of reference cell IV curves (min 3-5 cells)
            module_measurements: List of module IV curves (min 3 modules)

        Returns:
            Complete CTM test result with ratio, losses, and compliance status

        Raises:
            ValueError: If insufficient measurements provided

        Example:
            >>> tester = IEC63202CTMTester(config)
            >>> result = tester.ctm_power_ratio_test(
            ...     cell_measurements=[cell1_iv, cell2_iv, cell3_iv],
            ...     module_measurements=[mod1_iv, mod2_iv, mod3_iv]
            ... )
            >>> print(f"CTM Ratio: {result.ctm_ratio:.2f}%")
            >>> print(f"Compliant: {result.compliance_status}")
        """
        logger.info(
            f"CTM ratio test: {len(cell_measurements)} cells, "
            f"{len(module_measurements)} modules"
        )

        # Validate measurement counts
        if len(cell_measurements) < 3:
            raise ValueError("Minimum 3 cell measurements required per IEC 63202")
        if len(module_measurements) < 1:
            raise ValueError("Minimum 1 module measurement required")

        # Calculate cell statistics
        cell_powers = [iv.pmax for iv in cell_measurements]
        cell_power_avg = np.mean(cell_powers)
        cell_power_std = np.std(cell_powers, ddof=1)

        # Calculate module statistics
        module_powers = [iv.pmax for iv in module_measurements]
        module_power_avg = np.mean(module_powers)
        module_power_std = np.std(module_powers, ddof=1)

        # Calculate CTM ratio
        num_cells = self.config.module_config.total_cells
        expected_module_power = cell_power_avg * num_cells
        ctm_ratio = (module_power_avg / expected_module_power) * 100

        # Calculate measurement uncertainty
        ctm_uncertainty = self._calculate_ctm_uncertainty(
            cell_power_std / cell_power_avg,
            module_power_std / module_power_avg,
            len(cell_measurements),
            len(module_measurements)
        )

        # Calculate loss components
        loss_components = self.calculate_ctm_losses(
            cell_power_avg,
            module_power_avg,
            num_cells
        )

        # Validate against IEC 63202 compliance
        compliance_status = self.validate_ctm_ratio(ctm_ratio, ctm_uncertainty)

        # Create test result
        self.test_result = CTMTestResult(
            config=self.config,
            cell_measurements=cell_measurements,
            module_measurements=module_measurements,
            cell_power_avg=cell_power_avg,
            cell_power_std=cell_power_std,
            module_power_avg=module_power_avg,
            module_power_std=module_power_std,
            ctm_ratio=ctm_ratio,
            ctm_ratio_uncertainty=ctm_uncertainty,
            loss_components=loss_components,
            compliance_status=compliance_status,
            test_status=TestStatus.COMPLETED,
        )

        logger.info(
            f"CTM test complete: Ratio={ctm_ratio:.2f}±{ctm_uncertainty:.2f}%, "
            f"Compliant={compliance_status}"
        )

        return self.test_result

    def calculate_ctm_losses(
        self,
        cell_power: float,
        module_power: float,
        num_cells: int
    ) -> CTMLossComponents:
        """
        Break down CTM losses into optical, electrical, and thermal components.

        Args:
            cell_power: Average cell power (W)
            module_power: Average module power (W)
            num_cells: Number of cells in module

        Returns:
            Detailed breakdown of CTM loss components

        Example:
            >>> losses = tester.calculate_ctm_losses(5.0, 290.0, 60)
            >>> print(f"Optical: {losses.total_optical_loss:.2f}%")
            >>> print(f"Electrical: {losses.total_electrical_loss:.2f}%")
        """
        if self.power_loss_analyzer:
            return self.power_loss_analyzer.total_ctm_loss_budget(
                self.config.cell_properties,
                self.config.module_config
            )

        # Simplified loss estimation without analyzer
        expected_power = cell_power * num_cells
        total_loss = ((expected_power - module_power) / expected_power) * 100

        # Estimate typical loss distribution
        return CTMLossComponents(
            optical_reflection=total_loss * 0.30,  # ~30% of total
            optical_absorption=total_loss * 0.20,  # ~20% of total
            optical_shading=total_loss * 0.25,     # ~25% of total
            electrical_series_resistance=total_loss * 0.15,  # ~15% of total
            electrical_mismatch=total_loss * 0.10,  # ~10% of total
        )

    def validate_ctm_ratio(
        self,
        ctm_ratio: float,
        uncertainty: float
    ) -> bool:
        """
        Validate CTM ratio against manufacturer specifications and IEC 63202.

        Typical acceptable CTM ratios are 95-102% depending on module technology
        and manufacturing quality. IEC 63202 requires uncertainty < 3%.

        Args:
            ctm_ratio: Measured CTM ratio (%)
            uncertainty: Measurement uncertainty (%)

        Returns:
            True if CTM ratio meets acceptance criteria

        Example:
            >>> is_compliant = tester.validate_ctm_ratio(97.5, 1.8)
            >>> print(f"Compliant: {is_compliant}")
            Compliant: True
        """
        min_ratio = self.config.acceptance_criteria_min
        max_ratio = self.config.acceptance_criteria_max
        max_uncertainty = IEC_63202_COMPLIANCE["max_uncertainty"]

        ratio_acceptable = min_ratio <= ctm_ratio <= max_ratio
        uncertainty_acceptable = uncertainty <= max_uncertainty

        compliant = ratio_acceptable and uncertainty_acceptable

        logger.info(
            f"CTM validation: Ratio {ctm_ratio:.2f}% "
            f"{'PASS' if ratio_acceptable else 'FAIL'} "
            f"({min_ratio}-{max_ratio}%), "
            f"Uncertainty {uncertainty:.2f}% "
            f"{'PASS' if uncertainty_acceptable else 'FAIL'} "
            f"(≤{max_uncertainty}%)"
        )

        return compliant

    def generate_ctm_certificate(
        self,
        certified_by: str = "IEC 63202 Testing Laboratory",
        validity_months: int = 12
    ) -> CTMCertificate:
        """
        Produce IEC 63202 compliance certificate.

        Generates an official CTM test certificate documenting compliance
        with IEC 63202 standard, including test results, uncertainty analysis,
        and certification validity period.

        Args:
            certified_by: Name of certifying authority
            validity_months: Certificate validity period in months

        Returns:
            IEC 63202 CTM compliance certificate

        Raises:
            ValueError: If no test result available

        Example:
            >>> certificate = tester.generate_ctm_certificate(
            ...     certified_by="NREL PV Testing Lab",
            ...     validity_months=24
            ... )
            >>> print(f"Certificate: {certificate.certificate_number}")
            >>> print(f"Valid until: {certificate.expiry_date}")
        """
        if not self.test_result:
            raise ValueError("No test result available. Run CTM test first.")

        if not self.test_result.compliance_status:
            logger.warning(
                "Generating certificate for non-compliant test result"
            )

        # Generate certificate number
        cert_number = self._generate_certificate_number()

        certificate = CTMCertificate(
            certificate_number=cert_number,
            issue_date=datetime.now(),
            test_result=self.test_result,
            certified_by=certified_by,
            validity_period_months=validity_months,
        )

        logger.info(
            f"Certificate generated: {cert_number}, "
            f"valid until {certificate.expiry_date}"
        )

        return certificate

    def _correct_to_stc(
        self,
        iv_curve: IVCurveData,
        technology: str
    ) -> IVCurveData:
        """
        Correct IV curve measurements to STC conditions.

        Args:
            iv_curve: Measured IV curve
            technology: Cell technology type

        Returns:
            IV curve corrected to STC (1000 W/m², 25°C)
        """
        from pv_circularity_simulator.core.utils.constants import (
            get_temperature_correction,
            TEMPERATURE_COEFFICIENTS
        )

        # Temperature correction
        temp_correction = get_temperature_correction(
            technology,
            iv_curve.temperature,
            STC_TEMPERATURE
        )

        # Irradiance correction (linear approximation)
        irrad_correction = STC_IRRADIANCE / iv_curve.irradiance

        # Apply corrections to current (irradiance)
        corrected_current = [i * irrad_correction for i in iv_curve.current]

        # Apply corrections to voltage (temperature)
        temp_coeff_v = TEMPERATURE_COEFFICIENTS[technology]["voc"]
        temp_diff = iv_curve.temperature - STC_TEMPERATURE
        voltage_correction = temp_diff * temp_coeff_v * len(iv_curve.voltage) / iv_curve.voc

        corrected_voltage = [v - voltage_correction for v in iv_curve.voltage]

        return IVCurveData(
            voltage=corrected_voltage,
            current=corrected_current,
            temperature=STC_TEMPERATURE,
            irradiance=STC_IRRADIANCE,
        )

    def _validate_flash_conditions(
        self,
        temperature: float,
        irradiance: float
    ) -> None:
        """
        Validate flash simulator conditions per IEC 63202.

        Args:
            temperature: Measured temperature (°C)
            irradiance: Measured irradiance (W/m²)

        Raises:
            ValueError: If conditions deviate too far from STC
        """
        max_temp_dev = IEC_63202_COMPLIANCE["max_temperature_deviation"]
        max_irrad_dev = IEC_63202_COMPLIANCE["max_irradiance_deviation"]

        temp_deviation = abs(temperature - STC_TEMPERATURE)
        irrad_deviation = abs(irradiance - STC_IRRADIANCE)

        if temp_deviation > max_temp_dev:
            raise ValueError(
                f"Temperature deviation {temp_deviation:.1f}°C exceeds "
                f"maximum {max_temp_dev}°C per IEC 63202"
            )

        if irrad_deviation > max_irrad_dev:
            raise ValueError(
                f"Irradiance deviation {irrad_deviation:.1f}W/m² exceeds "
                f"maximum {max_irrad_dev}W/m² per IEC 63202"
            )

    def _apply_spectral_correction(
        self,
        iv_curve: IVCurveData,
        spectral_factor: float
    ) -> IVCurveData:
        """
        Apply spectral mismatch correction to IV curve.

        Args:
            iv_curve: IV curve to correct
            spectral_factor: Spectral mismatch correction factor

        Returns:
            Spectrally corrected IV curve
        """
        corrected_current = [i * spectral_factor for i in iv_curve.current]

        return IVCurveData(
            voltage=iv_curve.voltage,
            current=corrected_current,
            temperature=iv_curve.temperature,
            irradiance=iv_curve.irradiance,
        )

    def _calculate_ctm_uncertainty(
        self,
        cell_cv: float,  # Coefficient of variation
        module_cv: float,
        n_cells: int,
        n_modules: int
    ) -> float:
        """
        Calculate CTM ratio uncertainty per GUM methodology.

        Args:
            cell_cv: Cell power coefficient of variation
            module_cv: Module power coefficient of variation
            n_cells: Number of cell measurements
            n_modules: Number of module measurements

        Returns:
            Combined CTM uncertainty (%)
        """
        # Type A uncertainties (statistical)
        u_cell = cell_cv / np.sqrt(n_cells)
        u_module = module_cv / np.sqrt(n_modules)

        # Type B uncertainties (systematic)
        u_ref_device = MEASUREMENT_UNCERTAINTY["reference_device_calibration"]
        u_flash_stability = MEASUREMENT_UNCERTAINTY["flash_simulator_stability"]
        u_temperature = MEASUREMENT_UNCERTAINTY["temperature_measurement"]
        u_spatial = MEASUREMENT_UNCERTAINTY["spatial_non_uniformity"]

        # Combined uncertainty (RSS - Root Sum Square)
        combined_uncertainty = np.sqrt(
            u_cell**2 +
            u_module**2 +
            (u_ref_device / 100)**2 +
            (u_flash_stability / 100)**2 +
            (u_spatial / 100)**2
        ) * 100

        return combined_uncertainty

    def _generate_certificate_number(self) -> str:
        """
        Generate unique certificate number.

        Returns:
            Certificate number in format: IEC63202-YYYYMMDD-XXXXXX
        """
        date_str = datetime.now().strftime("%Y%m%d")
        test_id_short = self.config.test_id[:6] if self.config.test_id else "000000"
        return f"IEC63202-{date_str}-{test_id_short.upper()}"
