"""
Physical constants and default values for PV thermal modeling.

This module contains standard constants used in temperature and thermal calculations
for photovoltaic systems.
"""

from typing import Final

# Physical constants
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8  # W/(m²·K⁴) - Stefan-Boltzmann constant
STANDARD_ATMOSPHERE: Final[float] = 101325.0  # Pa - Standard atmospheric pressure
GRAVITY: Final[float] = 9.80665  # m/s² - Standard gravity
AIR_DENSITY_STP: Final[float] = 1.225  # kg/m³ - Air density at STP
SPECIFIC_HEAT_AIR: Final[float] = 1005.0  # J/(kg·K) - Specific heat of air at constant pressure
THERMAL_CONDUCTIVITY_AIR: Final[float] = 0.026  # W/(m·K) - Thermal conductivity of air at 20°C
KINEMATIC_VISCOSITY_AIR: Final[float] = 1.5e-5  # m²/s - Kinematic viscosity of air at 20°C

# Standard test conditions (STC)
STC_IRRADIANCE: Final[float] = 1000.0  # W/m² - Standard test irradiance
STC_TEMPERATURE: Final[float] = 25.0  # °C - Standard test cell temperature
STC_AIR_MASS: Final[float] = 1.5  # AM1.5 - Standard air mass

# NOCT (Nominal Operating Cell Temperature) test conditions
NOCT_IRRADIANCE: Final[float] = 800.0  # W/m² - NOCT test irradiance
NOCT_AMBIENT_TEMP: Final[float] = 20.0  # °C - NOCT test ambient temperature
NOCT_WIND_SPEED: Final[float] = 1.0  # m/s - NOCT test wind speed
NOCT_TILT: Final[float] = 45.0  # degrees - NOCT test tilt angle

# Default thermal parameters
DEFAULT_MODULE_HEAT_CAPACITY: Final[float] = 11000.0  # J/(m²·K) - Typical module heat capacity
DEFAULT_ABSORPTIVITY: Final[float] = 0.9  # Dimensionless - Typical solar absorptivity
DEFAULT_EMISSIVITY: Final[float] = 0.85  # Dimensionless - Typical thermal emissivity
DEFAULT_SKY_EMISSIVITY: Final[float] = 0.9  # Dimensionless - Typical sky emissivity

# Temperature coefficient ranges (typical values per °C)
TYPICAL_TEMP_COEFF_POWER_MIN: Final[float] = -0.005  # /°C - Minimum typical power temp coeff
TYPICAL_TEMP_COEFF_POWER_MAX: Final[float] = -0.003  # /°C - Maximum typical power temp coeff
TYPICAL_TEMP_COEFF_VOC_MIN: Final[float] = -0.004  # /°C - Minimum typical Voc temp coeff
TYPICAL_TEMP_COEFF_VOC_MAX: Final[float] = -0.002  # /°C - Maximum typical Voc temp coeff
TYPICAL_TEMP_COEFF_ISC: Final[float] = 0.0005  # /°C - Typical Isc temp coeff (positive)

# Mounting configuration parameters
MOUNTING_CONFIGS: Final[dict[str, dict[str, float]]] = {
    "open_rack": {
        "convection_coeff": 29.0,  # W/(m²·K) - Natural convection coefficient
        "wind_coeff": 0.0,  # Baseline for wind effect
        "thermal_resistance": 0.034,  # K·m²/W - Thermal resistance
    },
    "roof_mounted": {
        "convection_coeff": 15.0,  # W/(m²·K) - Reduced convection
        "wind_coeff": -5.0,  # Reduced wind effect
        "thermal_resistance": 0.067,  # K·m²/W - Higher thermal resistance
    },
    "ground_mounted": {
        "convection_coeff": 32.0,  # W/(m²·K) - Enhanced convection
        "wind_coeff": 3.0,  # Enhanced wind effect
        "thermal_resistance": 0.025,  # K·m²/W - Lower thermal resistance
    },
    "building_integrated": {
        "convection_coeff": 10.0,  # W/(m²·K) - Minimal convection
        "wind_coeff": -10.0,  # Minimal wind effect
        "thermal_resistance": 0.1,  # K·m²/W - Highest thermal resistance
    },
}

# Default model parameters (from pvlib)
# Sandia model default parameters (King et al. 2004)
SANDIA_DEFAULT_A: Final[float] = -3.47  # Dimensionless - Empirical coefficient
SANDIA_DEFAULT_B: Final[float] = -0.0594  # s/m - Empirical coefficient
SANDIA_DEFAULT_DELTA_T: Final[float] = 3.0  # °C - Temperature difference at 1000 W/m²

# Faiman model default parameters
FAIMAN_DEFAULT_U0: Final[float] = 25.0  # W/(m²·K) - Constant heat transfer coefficient
FAIMAN_DEFAULT_U1: Final[float] = 6.84  # W/(m²·K)/(m/s) - Wind-dependent coefficient

# PVsyst model default parameters
PVSYST_DEFAULT_U_C: Final[float] = 29.0  # W/(m²·K) - Constant heat loss factor
PVSYST_DEFAULT_U_V: Final[float] = 0.0  # W/(m²·K)/(m/s) - Wind-related heat loss factor
