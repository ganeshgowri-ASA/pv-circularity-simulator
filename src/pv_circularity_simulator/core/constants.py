"""
Physical constants and standard test conditions for PV systems.
"""

from typing import Dict

# Physical Constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K - Boltzmann constant
ELECTRON_CHARGE = 1.602176634e-19  # C - Elementary charge

# Standard Test Conditions (STC) - IEC 61215
STANDARD_IRRADIANCE = 1000.0  # W/m² - Standard irradiance
STANDARD_TEMPERATURE = 25.0  # °C - Standard cell temperature
STANDARD_AM = 1.5  # - Air Mass coefficient

# Temperature Coefficients (typical for silicon PV)
# These are typical values; actual values should be obtained from module datasheets
TEMP_COEFF_VOC = -0.0033  # V/°C per cell - Open circuit voltage temperature coefficient
TEMP_COEFF_ISC = 0.0005  # A/°C - Short circuit current temperature coefficient
TEMP_COEFF_POWER = -0.0045  # %/°C - Power temperature coefficient

# Thermal Imaging Constants
MIN_OPERATING_TEMP = -40.0  # °C - Minimum operating temperature
MAX_OPERATING_TEMP = 85.0  # °C - Maximum operating temperature
HOTSPOT_THRESHOLD_DELTA = 10.0  # °C - Temperature difference to classify as hotspot
SEVERE_HOTSPOT_THRESHOLD_DELTA = 20.0  # °C - Severe hotspot temperature difference
CRITICAL_HOTSPOT_THRESHOLD_DELTA = 30.0  # °C - Critical hotspot temperature difference

# Emissivity values for common PV materials
EMISSIVITY_VALUES: Dict[str, float] = {
    "glass": 0.94,  # Front glass
    "eva": 0.85,  # EVA encapsulant
    "silicon": 0.65,  # Silicon cells
    "aluminum": 0.10,  # Aluminum frame (polished)
    "aluminum_anodized": 0.77,  # Anodized aluminum
    "tedlar": 0.91,  # Tedlar backsheet
    "default": 0.90,  # Default conservative value
}

# IV Curve Analysis Constants
MIN_FILL_FACTOR = 0.60  # - Minimum acceptable fill factor
TYPICAL_FILL_FACTOR = 0.75  # - Typical fill factor for healthy modules
IDEAL_FILL_FACTOR = 0.85  # - Ideal fill factor
SERIES_RESISTANCE_THRESHOLD = 1.0  # Ω - Threshold for high series resistance
SHUNT_RESISTANCE_THRESHOLD = 500.0  # Ω - Threshold for low shunt resistance

# Degradation thresholds
DEGRADATION_WARNING_THRESHOLD = 0.05  # 5% power degradation
DEGRADATION_CRITICAL_THRESHOLD = 0.20  # 20% power degradation

# Statistical thresholds for anomaly detection
ANOMALY_ZSCORE_THRESHOLD = 3.0  # Standard deviations for anomaly detection
CONFIDENCE_THRESHOLD_HIGH = 0.90  # High confidence threshold
CONFIDENCE_THRESHOLD_MEDIUM = 0.70  # Medium confidence threshold
CONFIDENCE_THRESHOLD_LOW = 0.50  # Low confidence threshold
