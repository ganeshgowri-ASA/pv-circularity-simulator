"""
Reference Device Calibration for IEC 63202 CTM Testing.

This module manages calibration of reference devices (cells/photodiodes) used
in CTM testing, including traceability to primary standards, temperature and
spectral corrections, and uniformity validation.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging

import numpy as np
from scipy import interpolate

from pv_circularity_simulator.core.iec63202.models import (
    ReferenceDeviceData,
    IVCurveData,
    FlashSimulatorData,
)
from pv_circularity_simulator.core.utils.constants import (
    STC_TEMPERATURE,
    STC_IRRADIANCE,
    AM15_SPECTRUM,
)

logger = logging.getLogger(__name__)


class ReferenceDeviceCalibration:
    """
    Reference device calibration and correction management.

    This class handles calibration of reference solar cells and photodiodes
    used for CTM testing per IEC 60904-2 and IEC 60904-7 standards. It provides:
    - Calibration against primary standards
    - Temperature correction to STC
    - Spectral mismatch correction
    - Flash simulator spatial uniformity validation
    - Traceability documentation to SI units

    Attributes:
        reference_device: Reference device calibration data
        calibration_history: Historical calibration records
    """

    def __init__(self, reference_device: ReferenceDeviceData) -> None:
        """
        Initialize reference device calibration handler.

        Args:
            reference_device: Reference device calibration data
        """
        self.reference_device = reference_device
        self.calibration_history: List[ReferenceDeviceData] = [reference_device]
        self._validate_calibration()

        logger.info(
            f"Reference device calibration initialized: {reference_device.device_id}, "
            f"calibrated {reference_device.calibration_date}"
        )

    def calibrate_reference_cell(
        self,
        measured_isc: float,
        measured_temperature: float,
        measured_irradiance: float,
        primary_standard_isc: float,
        primary_standard_uncertainty: float = 1.0
    ) -> ReferenceDeviceData:
        """
        Calibrate reference device against primary standard.

        Calibration process per IEC 60904-2:
        1. Measure reference cell Isc under controlled conditions
        2. Compare with primary standard measurement
        3. Calculate calibration factor and uncertainty
        4. Update reference device calibration data

        Args:
            measured_isc: Measured short-circuit current (A)
            measured_temperature: Cell temperature during calibration (°C)
            measured_irradiance: Irradiance during calibration (W/m²)
            primary_standard_isc: Primary standard Isc measurement (A)
            primary_standard_uncertainty: Primary standard uncertainty (%)

        Returns:
            Updated reference device calibration data

        Example:
            >>> calibrator = ReferenceDeviceCalibration(ref_device)
            >>> new_calibration = calibrator.calibrate_reference_cell(
            ...     measured_isc=8.456,
            ...     measured_temperature=25.0,
            ...     measured_irradiance=1000.0,
            ...     primary_standard_isc=8.450
            ... )
            >>> print(f"New Isc: {new_calibration.short_circuit_current:.4f} A")
        """
        logger.info(
            f"Calibrating reference cell: Isc={measured_isc:.4f}A, "
            f"T={measured_temperature}°C, G={measured_irradiance}W/m²"
        )

        # Temperature correction to STC
        temp_corrected_isc = self._apply_temperature_correction(
            measured_isc,
            measured_temperature,
            STC_TEMPERATURE,
            self.reference_device.temperature_coefficient
        )

        # Irradiance correction to STC
        irrad_corrected_isc = temp_corrected_isc * (STC_IRRADIANCE / measured_irradiance)

        # Calculate calibration factor
        calibration_factor = primary_standard_isc / irrad_corrected_isc

        # Apply calibration factor
        calibrated_isc = irrad_corrected_isc * calibration_factor

        # Calculate combined uncertainty
        measurement_uncertainty = 0.5  # Typical Type A uncertainty (%)
        combined_uncertainty = np.sqrt(
            primary_standard_uncertainty**2 +
            measurement_uncertainty**2 +
            self.reference_device.uncertainty_isc**2
        )

        # Update reference device data
        new_calibration = ReferenceDeviceData(
            device_id=self.reference_device.device_id,
            calibration_date=datetime.now(),
            calibration_lab=self.reference_device.calibration_lab,
            calibration_certificate=self._generate_certificate_number(),
            short_circuit_current=calibrated_isc,
            responsivity=calibrated_isc / STC_IRRADIANCE,
            temperature_coefficient=self.reference_device.temperature_coefficient,
            spectral_response=self.reference_device.spectral_response,
            uncertainty_isc=combined_uncertainty,
            uncertainty_temperature=self.reference_device.uncertainty_temperature,
            traceability_chain=self._update_traceability(),
            next_calibration_due=datetime.now() + timedelta(days=365),
        )

        # Add to calibration history
        self.calibration_history.append(new_calibration)
        self.reference_device = new_calibration

        logger.info(
            f"Calibration complete: Isc={calibrated_isc:.4f}A, "
            f"Uncertainty=±{combined_uncertainty:.2f}%"
        )

        return new_calibration

    def temperature_correction(
        self,
        measured_current: float,
        measured_temperature: float,
        target_temperature: float = STC_TEMPERATURE
    ) -> float:
        """
        Correct measurement for temperature variations from STC.

        Temperature correction per IEC 60891:
        I_corrected = I_measured × [1 + α(T_target - T_measured)]

        where α is the temperature coefficient of Isc.

        Args:
            measured_current: Measured current (A)
            measured_temperature: Measurement temperature (°C)
            target_temperature: Target temperature, typically STC (°C)

        Returns:
            Temperature-corrected current (A)

        Example:
            >>> corrected_i = calibrator.temperature_correction(
            ...     measured_current=8.5,
            ...     measured_temperature=30.0,
            ...     target_temperature=25.0
            ... )
            >>> print(f"Corrected current: {corrected_i:.4f} A")
        """
        corrected = self._apply_temperature_correction(
            measured_current,
            measured_temperature,
            target_temperature,
            self.reference_device.temperature_coefficient
        )

        logger.debug(
            f"Temperature correction: {measured_current:.4f}A @ {measured_temperature}°C "
            f"→ {corrected:.4f}A @ {target_temperature}°C"
        )

        return corrected

    def spectral_correction(
        self,
        simulator_spectrum: Dict[float, float],
        reference_spectrum: Optional[Dict[float, float]] = None
    ) -> float:
        """
        Calculate spectral mismatch correction factor per IEC 60904-7.

        The spectral mismatch factor M accounts for differences between
        the flash simulator spectrum and the AM1.5 reference spectrum,
        weighted by the reference device spectral response.

        Args:
            simulator_spectrum: Flash simulator spectral distribution
                               (wavelength nm: irradiance W/m²/nm)
            reference_spectrum: Reference spectrum (default: AM1.5)

        Returns:
            Spectral mismatch correction factor

        Example:
            >>> factor = calibrator.spectral_correction(
            ...     simulator_spectrum={400: 0.8, 600: 1.5, 800: 1.0, ...}
            ... )
            >>> print(f"Spectral correction: {factor:.4f}")
        """
        if reference_spectrum is None:
            reference_spectrum = AM15_SPECTRUM

        if not self.reference_device.spectral_response:
            logger.warning(
                "No spectral response data available, using default correction"
            )
            return 1.0

        # Get common wavelength range
        wavelengths = sorted(
            set(simulator_spectrum.keys()) &
            set(reference_spectrum.keys()) &
            set(self.reference_device.spectral_response.keys())
        )

        if len(wavelengths) < 5:
            logger.warning("Insufficient spectral data, using default correction")
            return 1.0

        # Create wavelength grid
        wl_grid = np.linspace(min(wavelengths), max(wavelengths), 100)

        # Interpolate spectra
        e_sim = np.interp(
            wl_grid,
            list(simulator_spectrum.keys()),
            list(simulator_spectrum.values())
        )
        e_ref = np.interp(
            wl_grid,
            list(reference_spectrum.keys()),
            list(reference_spectrum.values())
        )
        sr = np.interp(
            wl_grid,
            list(self.reference_device.spectral_response.keys()),
            list(self.reference_device.spectral_response.values())
        )

        # Calculate spectral mismatch factor
        integral_ref_sr = np.trapz(e_ref * sr, wl_grid)
        integral_sim_sr = np.trapz(e_sim * sr, wl_grid)
        integral_sim = np.trapz(e_sim, wl_grid)
        integral_ref = np.trapz(e_ref, wl_grid)

        if integral_sim_sr == 0 or integral_ref == 0:
            logger.error("Invalid spectral integrals")
            return 1.0

        mismatch_factor = (integral_ref_sr / integral_sim_sr) * (integral_sim / integral_ref)

        logger.info(f"Spectral mismatch factor: {mismatch_factor:.4f}")

        return mismatch_factor

    def spatial_uniformity_check(
        self,
        irradiance_map: Dict[Tuple[float, float], float],
        threshold_percent: float = 2.0
    ) -> Tuple[bool, float]:
        """
        Validate flash simulator spatial uniformity per IEC 60904-9.

        IEC 60904-9 Class A simulators require spatial non-uniformity ≤2%
        across the test plane. This method analyzes the irradiance distribution
        and validates compliance.

        Args:
            irradiance_map: Spatial irradiance map {(x, y): irradiance}
                           Coordinates in meters, irradiance in W/m²
            threshold_percent: Maximum acceptable non-uniformity (%)

        Returns:
            Tuple of (is_compliant, measured_non_uniformity_percent)

        Raises:
            ValueError: If irradiance map has insufficient data points

        Example:
            >>> irradiance_map = {
            ...     (0.0, 0.0): 1000.0,
            ...     (0.1, 0.0): 998.0,
            ...     (0.0, 0.1): 1002.0,
            ...     (0.1, 0.1): 999.0,
            ... }
            >>> is_ok, non_uniformity = calibrator.spatial_uniformity_check(
            ...     irradiance_map
            ... )
            >>> print(f"Uniformity: {non_uniformity:.2f}% ({'PASS' if is_ok else 'FAIL'})")
        """
        if len(irradiance_map) < 9:
            raise ValueError(
                "Minimum 9 measurement points required (3×3 grid) "
                "for uniformity validation"
            )

        irradiances = np.array(list(irradiance_map.values()))

        # Calculate mean and standard deviation
        mean_irradiance = np.mean(irradiances)
        std_irradiance = np.std(irradiances)

        # Calculate non-uniformity as percentage
        non_uniformity = (std_irradiance / mean_irradiance) * 100

        # Check compliance
        is_compliant = non_uniformity <= threshold_percent

        logger.info(
            f"Spatial uniformity check: Non-uniformity={non_uniformity:.2f}% "
            f"({'PASS' if is_compliant else 'FAIL'}, threshold={threshold_percent}%)"
        )

        if not is_compliant:
            logger.warning(
                f"Flash simulator non-uniformity {non_uniformity:.2f}% "
                f"exceeds threshold {threshold_percent}%"
            )

        return is_compliant, non_uniformity

    def traceability_chain(self) -> str:
        """
        Document calibration traceability to SI units.

        Generates comprehensive traceability documentation linking the
        reference device calibration through intermediate standards to
        the primary radiometric standard (SI traceable).

        Returns:
            Traceability chain documentation string

        Example:
            >>> traceability = calibrator.traceability_chain()
            >>> print(traceability)
        """
        chain = []

        chain.append("=== CALIBRATION TRACEABILITY CHAIN ===\n")
        chain.append(f"Reference Device: {self.reference_device.device_id}")
        chain.append(f"Current Calibration: {self.reference_device.calibration_date}")
        chain.append(f"Certificate: {self.reference_device.calibration_certificate}")
        chain.append(f"Calibrated Isc: {self.reference_device.short_circuit_current:.4f} A")
        chain.append(f"Uncertainty: ±{self.reference_device.uncertainty_isc:.2f}%\n")

        chain.append("Traceability Path:")
        chain.append("1. Reference Device (Working Standard)")
        chain.append(f"   Lab: {self.reference_device.calibration_lab}")
        chain.append("2. Secondary Standard")
        chain.append("   Calibrated against NREL/WPVS primary reference")
        chain.append("3. Primary Standard (WPVS)")
        chain.append("   World Photovoltaic Scale maintained by NREL")
        chain.append("4. SI Units")
        chain.append("   Traceable to radiometric standards (NIST)\n")

        chain.append("Calibration History:")
        for i, cal in enumerate(self.calibration_history[-5:], 1):  # Last 5 calibrations
            chain.append(
                f"{i}. {cal.calibration_date.strftime('%Y-%m-%d')}: "
                f"Isc={cal.short_circuit_current:.4f}A, "
                f"Cert={cal.calibration_certificate}"
            )

        chain.append(f"\nNext Calibration Due: {self.reference_device.next_calibration_due}")

        traceability_doc = "\n".join(chain)

        logger.info("Traceability chain documentation generated")

        return traceability_doc

    def apply_calibration(self, iv_curve: IVCurveData) -> IVCurveData:
        """
        Apply reference device calibration to measured IV curve.

        Args:
            iv_curve: Measured IV curve data

        Returns:
            Calibrated IV curve

        Example:
            >>> calibrated_iv = calibrator.apply_calibration(measured_iv)
        """
        # Calculate calibration factor from responsivity
        expected_isc = self.reference_device.short_circuit_current
        measured_isc = iv_curve.isc

        if measured_isc == 0:
            logger.warning("Zero Isc measured, cannot apply calibration")
            return iv_curve

        calibration_factor = expected_isc / measured_isc

        # Apply calibration to current
        calibrated_current = [i * calibration_factor for i in iv_curve.current]

        return IVCurveData(
            voltage=iv_curve.voltage,
            current=calibrated_current,
            temperature=iv_curve.temperature,
            irradiance=iv_curve.irradiance,
            timestamp=iv_curve.timestamp,
        )

    def is_calibration_valid(self) -> bool:
        """
        Check if current calibration is still valid.

        Returns:
            True if calibration is within validity period

        Example:
            >>> if not calibrator.is_calibration_valid():
            ...     print("Recalibration required!")
        """
        is_valid = datetime.now() < self.reference_device.next_calibration_due

        if not is_valid:
            logger.warning(
                f"Reference device calibration expired on "
                f"{self.reference_device.next_calibration_due}"
            )

        return is_valid

    def _validate_calibration(self) -> None:
        """
        Validate reference device calibration data.

        Raises:
            ValueError: If calibration data is invalid
        """
        if not self.is_calibration_valid():
            logger.warning(
                "Reference device calibration has expired. "
                "Recalibration recommended."
            )

        if self.reference_device.uncertainty_isc > 5.0:
            logger.warning(
                f"High calibration uncertainty: "
                f"{self.reference_device.uncertainty_isc:.2f}%"
            )

    def _apply_temperature_correction(
        self,
        current: float,
        measured_temp: float,
        target_temp: float,
        temp_coefficient: float
    ) -> float:
        """
        Apply temperature correction to current measurement.

        Args:
            current: Measured current (A)
            measured_temp: Measurement temperature (°C)
            target_temp: Target temperature (°C)
            temp_coefficient: Temperature coefficient (A/°C)

        Returns:
            Temperature-corrected current (A)
        """
        temp_diff = target_temp - measured_temp
        corrected = current * (1.0 + temp_coefficient * temp_diff / current)
        return corrected

    def _generate_certificate_number(self) -> str:
        """
        Generate calibration certificate number.

        Returns:
            Certificate number in format: CAL-YYYYMMDD-NNNN
        """
        date_str = datetime.now().strftime("%Y%m%d")
        sequence = len(self.calibration_history) + 1
        return f"CAL-{date_str}-{sequence:04d}"

    def _update_traceability(self) -> str:
        """
        Update traceability documentation.

        Returns:
            Updated traceability information
        """
        return (
            f"Traceable to WPVS via {self.reference_device.calibration_lab} "
            f"secondary standard, calibrated {datetime.now().strftime('%Y-%m-%d')}"
        )
