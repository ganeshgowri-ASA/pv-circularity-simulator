"""
Constants and reference values for PV simulations and CTM testing.

This module contains physical constants, standard test condition parameters,
and CTM loss factors (k1-k24) per IEC 63202 and industry standards.
"""

from typing import Dict, Final
import numpy as np

# Physical Constants
PLANCK_CONSTANT: Final[float] = 6.62607015e-34  # J·s
SPEED_OF_LIGHT: Final[float] = 2.99792458e8  # m/s
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23  # J/K
ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19  # C
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8  # W/(m²·K⁴)

# Standard Test Conditions (STC)
STC_IRRADIANCE: Final[float] = 1000.0  # W/m²
STC_TEMPERATURE: Final[float] = 25.0  # °C
STC_AIR_MASS: Final[float] = 1.5  # AM 1.5 spectrum
STC_TEMPERATURE_KELVIN: Final[float] = 298.15  # K

# Nominal Operating Cell Temperature (NOCT) Conditions
NOCT_IRRADIANCE: Final[float] = 800.0  # W/m²
NOCT_AMBIENT_TEMP: Final[float] = 20.0  # °C
NOCT_WIND_SPEED: Final[float] = 1.0  # m/s

# IEC 63202 CTM Loss Factors (k1-k24)
# All factors are multiplicative (1.0 = no loss, <1.0 = loss, >1.0 = gain)

CTM_LOSS_FACTORS: Dict[str, Dict[str, float]] = {
    # k1-k5: Cell-level losses
    "k1_cell_binning": {
        "tight": 0.995,  # ±2% binning
        "medium": 0.990,  # ±5% binning
        "loose": 0.985,  # ±10% binning
    },
    "k2_cell_degradation_storage": {
        "excellent": 0.999,  # <1 month storage
        "good": 0.997,  # 1-3 months
        "fair": 0.995,  # 3-6 months
        "poor": 0.990,  # >6 months
    },
    "k3_cell_breakage": {
        "zero_breakage": 1.000,
        "minor_microcracks": 0.998,
        "moderate_microcracks": 0.995,
        "severe_cracks": 0.990,
    },
    "k4_measurement_uncertainty_cell": {
        "class_a": 0.985,  # ±1.5% uncertainty
        "class_b": 0.980,  # ±2% uncertainty
        "class_c": 0.975,  # ±2.5% uncertainty
    },
    "k5_cell_temperature_variation": {
        "excellent": 0.998,  # ±0.5°C
        "good": 0.995,  # ±1°C
        "fair": 0.992,  # ±2°C
    },

    # k6-k10: Interconnection losses
    "k6_ribbon_resistance": {
        "low_resistance": 0.995,  # Multi-busbar (MBB), low-R ribbon
        "medium_resistance": 0.990,  # Standard ribbon
        "high_resistance": 0.985,  # Thick ribbon, long paths
    },
    "k7_solder_joint_quality": {
        "excellent": 0.998,  # Automated soldering, high quality
        "good": 0.995,  # Standard soldering
        "poor": 0.990,  # Manual, inconsistent joints
    },
    "k8_busbar_resistance": {
        "low": 0.997,  # Wide busbar, low resistivity
        "medium": 0.994,  # Standard busbar
        "high": 0.990,  # Narrow/thin busbar
    },
    "k9_cell_mismatch": {
        "tight_binning": 0.998,  # <1% power variation
        "medium_binning": 0.995,  # 1-2% variation
        "loose_binning": 0.990,  # 2-5% variation
    },
    "k10_interconnect_shading": {
        "mbb_5bb": 0.980,  # Multi-busbar (5+ BB)
        "standard_3bb": 0.975,  # 3 busbar
        "conventional_2bb": 0.970,  # 2 busbar
    },

    # k11-k15: Encapsulation and lamination losses
    "k11_glass_transmission": {
        "ar_coated_high": 0.975,  # AR-coated, high transmission
        "ar_coated_standard": 0.970,  # Standard AR coating
        "uncoated": 0.960,  # No AR coating
    },
    "k12_encapsulant_transmission": {
        "eva_high_quality": 0.985,  # High-quality EVA
        "poe_premium": 0.987,  # POE encapsulant
        "eva_standard": 0.980,  # Standard EVA
    },
    "k13_encapsulant_absorption": {
        "low_absorption": 0.995,  # Premium materials
        "medium_absorption": 0.990,  # Standard materials
        "high_absorption": 0.985,  # Lower quality materials
    },
    "k14_backsheet_reflectance": {
        "white_high_reflectance": 1.020,  # White backsheet gain
        "white_standard": 1.015,  # Standard white
        "transparent": 1.000,  # No reflectance gain
        "black": 0.995,  # Black backsheet (aesthetic)
    },
    "k15_lamination_bubbles_delamination": {
        "perfect": 1.000,  # No defects
        "minor_defects": 0.998,  # Small bubbles
        "moderate_defects": 0.995,  # Some delamination
        "severe_defects": 0.985,  # Significant issues
    },

    # k16-k20: Module assembly and framing
    "k16_junction_box_diode_loss": {
        "low_voltage_drop": 0.995,  # Schottky diodes
        "medium_voltage_drop": 0.992,  # Standard diodes
        "high_voltage_drop": 0.988,  # High Vf diodes
    },
    "k17_frame_shading": {
        "frameless": 1.000,  # No frame
        "thin_frame": 0.998,  # Thin frame
        "standard_frame": 0.995,  # Standard frame
        "thick_frame": 0.992,  # Thick frame
    },
    "k18_module_edge_effects": {
        "negligible": 0.999,
        "minor": 0.997,
        "moderate": 0.995,
    },
    "k19_thermal_stress_assembly": {
        "low_stress": 0.998,  # Controlled lamination
        "medium_stress": 0.995,  # Standard process
        "high_stress": 0.990,  # Aggressive conditions
    },
    "k20_quality_control_process": {
        "stringent": 1.005,  # Strict QC, rejection of defects
        "standard": 1.000,  # Normal QC
        "minimal": 0.995,  # Limited QC
    },

    # k21-k24: Measurement and environmental factors
    "k21_flash_simulator_spectrum": {
        "class_aaa": 0.995,  # AAA simulator
        "class_aba": 0.990,  # ABA simulator
        "class_bbb": 0.985,  # BBB simulator
    },
    "k22_spatial_uniformity": {
        "excellent": 0.998,  # ≤1% non-uniformity
        "good": 0.995,  # ≤2% non-uniformity
        "fair": 0.990,  # ≤5% non-uniformity
    },
    "k23_measurement_uncertainty_module": {
        "class_a": 0.985,  # ±1.5% uncertainty
        "class_b": 0.980,  # ±2% uncertainty
        "class_c": 0.975,  # ±2.5% uncertainty
    },
    "k24_module_temperature_variation": {
        "excellent": 0.998,  # ±0.5°C from STC
        "good": 0.995,  # ±1°C
        "fair": 0.992,  # ±2°C
    },
}

# Default CTM loss factor selections for different quality scenarios
DEFAULT_CTM_SCENARIOS: Dict[str, Dict[str, str]] = {
    "premium_quality": {
        "k1_cell_binning": "tight",
        "k2_cell_degradation_storage": "excellent",
        "k3_cell_breakage": "zero_breakage",
        "k4_measurement_uncertainty_cell": "class_a",
        "k5_cell_temperature_variation": "excellent",
        "k6_ribbon_resistance": "low_resistance",
        "k7_solder_joint_quality": "excellent",
        "k8_busbar_resistance": "low",
        "k9_cell_mismatch": "tight_binning",
        "k10_interconnect_shading": "mbb_5bb",
        "k11_glass_transmission": "ar_coated_high",
        "k12_encapsulant_transmission": "poe_premium",
        "k13_encapsulant_absorption": "low_absorption",
        "k14_backsheet_reflectance": "white_high_reflectance",
        "k15_lamination_bubbles_delamination": "perfect",
        "k16_junction_box_diode_loss": "low_voltage_drop",
        "k17_frame_shading": "thin_frame",
        "k18_module_edge_effects": "negligible",
        "k19_thermal_stress_assembly": "low_stress",
        "k20_quality_control_process": "stringent",
        "k21_flash_simulator_spectrum": "class_aaa",
        "k22_spatial_uniformity": "excellent",
        "k23_measurement_uncertainty_module": "class_a",
        "k24_module_temperature_variation": "excellent",
    },
    "standard_quality": {
        "k1_cell_binning": "medium",
        "k2_cell_degradation_storage": "good",
        "k3_cell_breakage": "minor_microcracks",
        "k4_measurement_uncertainty_cell": "class_b",
        "k5_cell_temperature_variation": "good",
        "k6_ribbon_resistance": "medium_resistance",
        "k7_solder_joint_quality": "good",
        "k8_busbar_resistance": "medium",
        "k9_cell_mismatch": "medium_binning",
        "k10_interconnect_shading": "standard_3bb",
        "k11_glass_transmission": "ar_coated_standard",
        "k12_encapsulant_transmission": "eva_high_quality",
        "k13_encapsulant_absorption": "medium_absorption",
        "k14_backsheet_reflectance": "white_standard",
        "k15_lamination_bubbles_delamination": "minor_defects",
        "k16_junction_box_diode_loss": "medium_voltage_drop",
        "k17_frame_shading": "standard_frame",
        "k18_module_edge_effects": "minor",
        "k19_thermal_stress_assembly": "medium_stress",
        "k20_quality_control_process": "standard",
        "k21_flash_simulator_spectrum": "class_aba",
        "k22_spatial_uniformity": "good",
        "k23_measurement_uncertainty_module": "class_b",
        "k24_module_temperature_variation": "good",
    },
    "economy_quality": {
        "k1_cell_binning": "loose",
        "k2_cell_degradation_storage": "fair",
        "k3_cell_breakage": "moderate_microcracks",
        "k4_measurement_uncertainty_cell": "class_c",
        "k5_cell_temperature_variation": "fair",
        "k6_ribbon_resistance": "high_resistance",
        "k7_solder_joint_quality": "poor",
        "k8_busbar_resistance": "high",
        "k9_cell_mismatch": "loose_binning",
        "k10_interconnect_shading": "conventional_2bb",
        "k11_glass_transmission": "uncoated",
        "k12_encapsulant_transmission": "eva_standard",
        "k13_encapsulant_absorption": "high_absorption",
        "k14_backsheet_reflectance": "transparent",
        "k15_lamination_bubbles_delamination": "moderate_defects",
        "k16_junction_box_diode_loss": "high_voltage_drop",
        "k17_frame_shading": "thick_frame",
        "k18_module_edge_effects": "moderate",
        "k19_thermal_stress_assembly": "high_stress",
        "k20_quality_control_process": "minimal",
        "k21_flash_simulator_spectrum": "class_bbb",
        "k22_spatial_uniformity": "fair",
        "k23_measurement_uncertainty_module": "class_c",
        "k24_module_temperature_variation": "fair",
    },
}

# Temperature coefficients by technology (per °C)
TEMPERATURE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "mono-Si": {
        "pmax": -0.45,  # %/°C
        "voc": -0.0033,  # V/°C (absolute)
        "isc": 0.0005,  # %/°C
    },
    "poly-Si": {
        "pmax": -0.43,
        "voc": -0.0031,
        "isc": 0.0006,
    },
    "PERC": {
        "pmax": -0.39,
        "voc": -0.0029,
        "isc": 0.0005,
    },
    "TOPCon": {
        "pmax": -0.36,
        "voc": -0.0027,
        "isc": 0.0004,
    },
    "HJT": {
        "pmax": -0.30,
        "voc": -0.0024,
        "isc": 0.0003,
    },
    "IBC": {
        "pmax": -0.35,
        "voc": -0.0026,
        "isc": 0.0004,
    },
    "CIGS": {
        "pmax": -0.32,
        "voc": -0.0028,
        "isc": 0.0002,
    },
    "CdTe": {
        "pmax": -0.25,
        "voc": -0.0023,
        "isc": 0.0001,
    },
}

# AM1.5 Solar Spectrum Reference (wavelength in nm, irradiance in W/m²/nm)
AM15_SPECTRUM: Dict[float, float] = {
    300: 0.0,
    350: 0.35,
    400: 0.95,
    450: 1.45,
    500: 1.75,
    550: 1.80,
    600: 1.65,
    650: 1.50,
    700: 1.35,
    750: 1.20,
    800: 1.05,
    850: 0.90,
    900: 0.75,
    950: 0.60,
    1000: 0.50,
    1050: 0.40,
    1100: 0.30,
    1150: 0.22,
    1200: 0.15,
}

# IEC 63202 compliance thresholds
IEC_63202_COMPLIANCE: Dict[str, float] = {
    "min_ctm_ratio": 95.0,  # Minimum acceptable CTM ratio (%)
    "max_ctm_ratio": 102.0,  # Maximum acceptable CTM ratio (%)
    "max_uncertainty": 3.0,  # Maximum measurement uncertainty (%)
    "min_spatial_uniformity": 98.0,  # Minimum flash uniformity (%)
    "max_temperature_deviation": 2.0,  # Max temp deviation from STC (°C)
    "max_irradiance_deviation": 20.0,  # Max irradiance deviation from STC (W/m²)
}

# Measurement uncertainty components (per GUM - Guide to Uncertainty in Measurement)
MEASUREMENT_UNCERTAINTY: Dict[str, float] = {
    "reference_device_calibration": 1.0,  # %
    "flash_simulator_stability": 0.5,  # %
    "temperature_measurement": 0.3,  # °C
    "irradiance_measurement": 1.0,  # %
    "iv_curve_measurement": 0.5,  # %
    "spatial_non_uniformity": 1.0,  # %
    "spectral_mismatch": 2.0,  # %
}


def calculate_total_ctm_factor(scenario: str = "standard_quality") -> float:
    """
    Calculate total CTM multiplication factor from k1-k24 factors.

    Args:
        scenario: Quality scenario ("premium_quality", "standard_quality", "economy_quality")

    Returns:
        Total CTM factor (product of all k factors)

    Example:
        >>> calculate_total_ctm_factor("premium_quality")
        0.9825  # Typical premium quality CTM ratio
    """
    if scenario not in DEFAULT_CTM_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(DEFAULT_CTM_SCENARIOS.keys())}")

    scenario_config = DEFAULT_CTM_SCENARIOS[scenario]
    total_factor = 1.0

    for factor_name, quality_level in scenario_config.items():
        if factor_name in CTM_LOSS_FACTORS:
            factor_value = CTM_LOSS_FACTORS[factor_name][quality_level]
            total_factor *= factor_value

    return total_factor


def get_temperature_correction(
    technology: str,
    measured_temp: float,
    target_temp: float = STC_TEMPERATURE
) -> float:
    """
    Calculate temperature correction factor.

    Args:
        technology: Cell technology type
        measured_temp: Measured temperature (°C)
        target_temp: Target temperature, typically STC (°C)

    Returns:
        Temperature correction factor

    Example:
        >>> get_temperature_correction("PERC", 30.0, 25.0)
        0.9805  # 5°C above STC causes ~2% power reduction
    """
    if technology not in TEMPERATURE_COEFFICIENTS:
        raise ValueError(f"Unknown technology: {technology}")

    temp_diff = measured_temp - target_temp
    temp_coeff = TEMPERATURE_COEFFICIENTS[technology]["pmax"]

    # Convert %/°C to factor
    correction_factor = 1.0 + (temp_coeff / 100.0) * temp_diff

    return correction_factor
