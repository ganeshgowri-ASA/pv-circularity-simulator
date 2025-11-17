"""String sizing calculator with NEC 690 and IEC 60364 compliance."""

import logging
from typing import Dict, Optional, Tuple
import numpy as np

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    StringConfiguration,
)

logger = logging.getLogger(__name__)


class StringSizingCalculator:
    """
    Calculate PV string configurations based on module and inverter parameters.

    Complies with NEC 690 (2023) and IEC 60364 standards for voltage and current limits.
    """

    def __init__(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        site_temp_min: float = -10.0,
        site_temp_max: float = 70.0,
        safety_factor: float = 1.25,
    ):
        """
        Initialize string sizing calculator.

        Args:
            module: Module parameters including Voc, Vmp, Isc, Imp, and temperature coefficients
            inverter: Inverter parameters including MPPT voltage range and current limits
            site_temp_min: Minimum expected site temperature in °C (default: -10°C)
            site_temp_max: Maximum expected site temperature in °C (default: 70°C)
            safety_factor: NEC 690.7 safety factor for continuous current (default: 1.25)
        """
        self.module = module
        self.inverter = inverter
        self.site_temp_min = site_temp_min
        self.site_temp_max = site_temp_max
        self.safety_factor = safety_factor

        logger.info(
            f"Initialized StringSizingCalculator for {module.manufacturer} {module.model} "
            f"with {inverter.manufacturer} {inverter.model}"
        )

    def temperature_voltage_correction(
        self, voltage_stc: float, temp_celsius: float, temp_coeff_percent: float
    ) -> float:
        """
        Calculate voltage corrected for temperature.

        Args:
            voltage_stc: Voltage at STC (25°C) in volts
            temp_celsius: Operating temperature in °C
            temp_coeff_percent: Temperature coefficient in %/°C

        Returns:
            Temperature-corrected voltage in volts
        """
        temp_diff = temp_celsius - 25.0
        voltage_corrected = voltage_stc * (1 + (temp_coeff_percent / 100.0) * temp_diff)
        return voltage_corrected

    def calculate_max_string_length(self) -> int:
        """
        Calculate maximum string length based on Voc at minimum temperature.

        Per NEC 690.7, the maximum string voltage must not exceed the inverter's
        maximum DC voltage rating at the lowest expected site temperature.

        Returns:
            Maximum number of modules per string

        Raises:
            ValueError: If configuration results in less than 1 module per string
        """
        # Calculate Voc at minimum temperature
        voc_min_temp = self.temperature_voltage_correction(
            self.module.voc, self.site_temp_min, self.module.temp_coeff_voc
        )

        logger.debug(
            f"Voc at {self.site_temp_min}°C: {voc_min_temp:.2f}V "
            f"(STC: {self.module.voc:.2f}V, coeff: {self.module.temp_coeff_voc}%/°C)"
        )

        # Calculate max modules based on inverter Vdc_max
        max_modules_voc = int(np.floor(self.inverter.vdc_max / voc_min_temp))

        if max_modules_voc < 1:
            raise ValueError(
                f"Module Voc at minimum temperature ({voc_min_temp:.2f}V) "
                f"exceeds inverter maximum voltage ({self.inverter.vdc_max}V)"
            )

        logger.info(f"Maximum string length (Voc constraint): {max_modules_voc} modules")
        return max_modules_voc

    def calculate_min_string_length(self) -> int:
        """
        Calculate minimum string length based on Vmp for MPPT operation.

        The string voltage at maximum operating temperature must be within the
        inverter's MPPT voltage range for optimal operation.

        Returns:
            Minimum number of modules per string

        Raises:
            ValueError: If MPPT range cannot be satisfied
        """
        # Calculate Vmp at maximum temperature
        vmp_max_temp = self.temperature_voltage_correction(
            self.module.vmp, self.site_temp_max, self.module.temp_coeff_voc
        )

        logger.debug(
            f"Vmp at {self.site_temp_max}°C: {vmp_max_temp:.2f}V "
            f"(STC: {self.module.vmp:.2f}V, coeff: {self.module.temp_coeff_voc}%/°C)"
        )

        # Calculate min modules based on MPPT minimum voltage
        min_modules_mppt = int(np.ceil(self.inverter.mppt_vmin / vmp_max_temp))

        # Also check against inverter minimum DC voltage
        min_modules_vdc = int(np.ceil(self.inverter.vdc_min / vmp_max_temp))

        min_modules = max(min_modules_mppt, min_modules_vdc)

        # Verify this is achievable
        vmp_min_temp = self.temperature_voltage_correction(
            self.module.vmp, self.site_temp_min, self.module.temp_coeff_voc
        )
        max_voltage_at_min_modules = min_modules * vmp_min_temp

        if max_voltage_at_min_modules > self.inverter.mppt_vmax:
            raise ValueError(
                f"Cannot satisfy MPPT voltage range. "
                f"Minimum {min_modules} modules gives {max_voltage_at_min_modules:.1f}V "
                f"which exceeds MPPT max ({self.inverter.mppt_vmax}V)"
            )

        logger.info(f"Minimum string length (MPPT constraint): {min_modules} modules")
        return min_modules

    def calculate_max_strings_per_mppt(self) -> int:
        """
        Calculate maximum strings per MPPT input based on current limits.

        Per NEC 690.8, the continuous current rating must be at least 125% of the
        short-circuit current (Isc).

        Returns:
            Maximum number of strings per MPPT input

        Raises:
            ValueError: If current requirements cannot be met
        """
        # Apply safety factor to Isc per NEC 690.8
        isc_continuous = self.module.isc * self.safety_factor

        # Calculate max strings based on inverter DC current limit
        max_strings_idc = int(np.floor(self.inverter.idc_max / isc_continuous))

        # Also check against inverter's specified strings per MPPT
        max_strings = min(max_strings_idc, self.inverter.strings_per_mppt)

        if max_strings < 1:
            raise ValueError(
                f"Module Isc with safety factor ({isc_continuous:.2f}A) "
                f"exceeds inverter maximum current ({self.inverter.idc_max}A)"
            )

        logger.info(
            f"Maximum strings per MPPT: {max_strings} "
            f"(Isc: {self.module.isc:.2f}A, with safety factor: {isc_continuous:.2f}A)"
        )
        return max_strings

    def calculate_string_voltage_range(
        self, modules_per_string: int
    ) -> Dict[str, float]:
        """
        Calculate voltage range for a string configuration.

        Args:
            modules_per_string: Number of modules in the string

        Returns:
            Dictionary with voltage parameters:
                - voc_stc: Voc at STC (V)
                - voc_min_temp: Voc at minimum temperature (V)
                - voc_max_temp: Voc at maximum temperature (V)
                - vmp_stc: Vmp at STC (V)
                - vmp_min_temp: Vmp at minimum temperature (V)
                - vmp_max_temp: Vmp at maximum temperature (V)
        """
        voc_stc = self.module.voc * modules_per_string
        vmp_stc = self.module.vmp * modules_per_string

        voc_min_temp = (
            self.temperature_voltage_correction(
                self.module.voc, self.site_temp_min, self.module.temp_coeff_voc
            )
            * modules_per_string
        )
        voc_max_temp = (
            self.temperature_voltage_correction(
                self.module.voc, self.site_temp_max, self.module.temp_coeff_voc
            )
            * modules_per_string
        )

        vmp_min_temp = (
            self.temperature_voltage_correction(
                self.module.vmp, self.site_temp_min, self.module.temp_coeff_voc
            )
            * modules_per_string
        )
        vmp_max_temp = (
            self.temperature_voltage_correction(
                self.module.vmp, self.site_temp_max, self.module.temp_coeff_voc
            )
            * modules_per_string
        )

        return {
            "voc_stc": voc_stc,
            "voc_min_temp": voc_min_temp,
            "voc_max_temp": voc_max_temp,
            "vmp_stc": vmp_stc,
            "vmp_min_temp": vmp_min_temp,
            "vmp_max_temp": vmp_max_temp,
        }

    def validate_string_configuration(
        self, modules_per_string: int, strings_per_mppt: int
    ) -> Tuple[bool, str]:
        """
        Validate that a string configuration meets all requirements.

        Args:
            modules_per_string: Number of modules per string
            strings_per_mppt: Number of strings per MPPT

        Returns:
            Tuple of (is_valid, message) where is_valid is True if configuration is valid
        """
        try:
            # Check string length limits
            min_modules = self.calculate_min_string_length()
            max_modules = self.calculate_max_string_length()

            if modules_per_string < min_modules:
                return (
                    False,
                    f"String too short: {modules_per_string} < {min_modules} minimum",
                )

            if modules_per_string > max_modules:
                return (
                    False,
                    f"String too long: {modules_per_string} > {max_modules} maximum",
                )

            # Check strings per MPPT limit
            max_strings = self.calculate_max_strings_per_mppt()

            if strings_per_mppt > max_strings:
                return (
                    False,
                    f"Too many strings per MPPT: {strings_per_mppt} > {max_strings} maximum",
                )

            # Check voltage ranges
            voltages = self.calculate_string_voltage_range(modules_per_string)

            if voltages["voc_min_temp"] > self.inverter.vdc_max:
                return (
                    False,
                    f"Voc at min temp ({voltages['voc_min_temp']:.1f}V) "
                    f"exceeds inverter max ({self.inverter.vdc_max}V)",
                )

            if voltages["vmp_max_temp"] < self.inverter.mppt_vmin:
                return (
                    False,
                    f"Vmp at max temp ({voltages['vmp_max_temp']:.1f}V) "
                    f"below MPPT min ({self.inverter.mppt_vmin}V)",
                )

            if voltages["vmp_min_temp"] > self.inverter.mppt_vmax:
                return (
                    False,
                    f"Vmp at min temp ({voltages['vmp_min_temp']:.1f}V) "
                    f"exceeds MPPT max ({self.inverter.mppt_vmax}V)",
                )

            # Check current limits
            total_current = self.module.isc * self.safety_factor * strings_per_mppt

            if total_current > self.inverter.idc_max:
                return (
                    False,
                    f"Total current ({total_current:.1f}A) "
                    f"exceeds inverter max ({self.inverter.idc_max}A)",
                )

            return (True, "Configuration valid")

        except Exception as e:
            return (False, f"Validation error: {str(e)}")

    def design_optimal_string(self) -> StringConfiguration:
        """
        Design optimal string configuration for maximum energy harvest.

        Selects string length near the middle of the MPPT voltage range at STC
        for optimal year-round performance.

        Returns:
            Optimized StringConfiguration

        Raises:
            ValueError: If no valid configuration can be found
        """
        min_modules = self.calculate_min_string_length()
        max_modules = self.calculate_max_string_length()

        if min_modules > max_modules:
            raise ValueError(
                f"No valid string configuration: min ({min_modules}) > max ({max_modules})"
            )

        # Target voltage near middle of MPPT range at STC
        mppt_middle = (self.inverter.mppt_vmin + self.inverter.mppt_vmax) / 2
        optimal_modules = int(round(mppt_middle / self.module.vmp))

        # Constrain to valid range
        optimal_modules = max(min_modules, min(optimal_modules, max_modules))

        # Calculate max strings per MPPT
        max_strings = self.calculate_max_strings_per_mppt()

        # Validate configuration
        is_valid, message = self.validate_string_configuration(optimal_modules, max_strings)

        if not is_valid:
            raise ValueError(f"Optimal configuration invalid: {message}")

        # Calculate voltage parameters
        voltages = self.calculate_string_voltage_range(optimal_modules)

        string_config = StringConfiguration(
            modules_per_string=optimal_modules,
            strings_per_mppt=max_strings,
            orientation_azimuth=180.0,  # Default south-facing
            tilt_angle=30.0,  # Default tilt
            voc_stc=voltages["voc_stc"],
            vmp_stc=voltages["vmp_stc"],
            isc_stc=self.module.isc,
            imp_stc=self.module.imp,
            voc_min_temp=voltages["voc_min_temp"],
            voc_max_temp=voltages["voc_max_temp"],
            vmp_min_temp=voltages["vmp_min_temp"],
            vmp_max_temp=voltages["vmp_max_temp"],
        )

        logger.info(
            f"Optimal string design: {optimal_modules} modules/string, "
            f"{max_strings} strings/MPPT, "
            f"Vmp @ STC: {voltages['vmp_stc']:.1f}V"
        )

        return string_config

    def fuse_sizing(self, strings_per_mppt: int) -> Dict[str, float]:
        """
        Calculate fuse sizing per NEC 690.9.

        Args:
            strings_per_mppt: Number of parallel strings

        Returns:
            Dictionary with fuse parameters:
                - min_fuse_rating: Minimum fuse current rating (A)
                - recommended_fuse_rating: Recommended standard fuse rating (A)
                - string_isc: String short-circuit current (A)
        """
        # Per NEC 690.9, fuse rating must be >= 1.56 * Isc
        string_isc = self.module.isc
        min_fuse_rating = string_isc * 1.56

        # Round up to next standard fuse size
        # Standard fuse sizes: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 125, 150, 200A
        standard_sizes = [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 125, 150, 200
        ]
        recommended_fuse_rating = next(
            (size for size in standard_sizes if size >= min_fuse_rating), 200
        )

        logger.info(
            f"Fuse sizing: Isc={string_isc:.2f}A, "
            f"min rating={min_fuse_rating:.2f}A, "
            f"recommended={recommended_fuse_rating}A"
        )

        return {
            "min_fuse_rating": min_fuse_rating,
            "recommended_fuse_rating": recommended_fuse_rating,
            "string_isc": string_isc,
        }

    def string_mismatch_analysis(
        self,
        azimuth_diff: float = 0.0,
        tilt_diff: float = 0.0,
        shading_factor: float = 0.0,
    ) -> float:
        """
        Analyze string mismatch losses for non-uniform conditions.

        Args:
            azimuth_diff: Azimuth difference from optimal (degrees)
            tilt_diff: Tilt difference from optimal (degrees)
            shading_factor: Partial shading factor (0-1, where 0=no shading)

        Returns:
            Estimated mismatch loss percentage (%)
        """
        # Base mismatch loss from manufacturing tolerance (~1%)
        base_mismatch = 1.0

        # Orientation mismatch (approximately 0.1% per 10° azimuth difference)
        azimuth_loss = abs(azimuth_diff) * 0.01

        # Tilt mismatch (approximately 0.05% per 5° tilt difference)
        tilt_loss = abs(tilt_diff) * 0.01

        # Shading mismatch (can be significant, up to 10% for heavy partial shading)
        shading_loss = shading_factor * 10.0

        total_mismatch = base_mismatch + azimuth_loss + tilt_loss + shading_loss

        logger.debug(
            f"Mismatch analysis: base={base_mismatch:.1f}%, "
            f"azimuth={azimuth_loss:.1f}%, tilt={tilt_loss:.1f}%, "
            f"shading={shading_loss:.1f}%, total={total_mismatch:.1f}%"
        )

        return total_mismatch
