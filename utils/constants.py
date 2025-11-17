"""
Constants Module
================
Application-wide constants for PV Circularity Simulator.
"""

from typing import Dict, List
from enum import Enum


# ============================================================================
# APPLICATION CONSTANTS
# ============================================================================

APP_NAME = "PV Circularity Simulator"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "End-to-end Solar PV Lifecycle Management Platform"
APP_AUTHOR = "PV Circularity Team"
APP_LICENSE = "MIT"

# Session tracking
TOTAL_SESSIONS = 71
TOTAL_BRANCHES = 15


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Solar constants
SOLAR_CONSTANT = 1361  # W/m² (extraterrestrial solar irradiance)
STANDARD_TEST_CONDITIONS_IRRADIANCE = 1000  # W/m²
STANDARD_TEST_CONDITIONS_TEMP = 25  # °C
STANDARD_TEST_CONDITIONS_SPECTRUM = "AM1.5G"

# Standard reference irradiance levels
REFERENCE_IRRADIANCES = {
    'STC': 1000,      # W/m² - Standard Test Conditions
    'NOCT': 800,      # W/m² - Nominal Operating Cell Temperature
    'LOW': 200,       # W/m² - Low light conditions
    'MEDIUM': 500     # W/m² - Medium irradiance
}

# Temperature coefficients (typical ranges)
TEMP_COEFF_RANGES = {
    'c_Si': (-0.45, -0.35),         # %/°C
    'Perovskite': (-0.30, -0.20),   # %/°C
    'CIGS': (-0.36, -0.28),         # %/°C
    'CdTe': (-0.28, -0.21)          # %/°C
}


# ============================================================================
# MODULE & SYSTEM CONSTANTS
# ============================================================================

# Standard module configurations
STANDARD_CELL_COUNTS = [36, 60, 72, 96, 120, 144]

# Standard module dimensions (mm)
STANDARD_MODULE_SIZES = {
    '60cell': {'width': 1000, 'height': 1650, 'thickness': 40},
    '72cell': {'width': 1000, 'height': 2000, 'thickness': 40},
    '144cell': {'width': 1134, 'height': 2278, 'thickness': 35}
}

# Module weight (kg)
TYPICAL_MODULE_WEIGHT_KG = {
    '60cell': 18.5,
    '72cell': 22.0,
    '144cell': 30.0
}

# System voltage ranges
DC_VOLTAGE_RANGES = {
    'residential': (300, 600),      # V
    'commercial': (600, 1000),      # V
    'utility': (800, 1500)          # V
}

# Inverter efficiency by type
INVERTER_EFFICIENCY = {
    'micro': 96.5,          # %
    'string': 97.5,         # %
    'central': 98.5,        # %
    'hybrid': 96.0          # %
}


# ============================================================================
# DEGRADATION & LIFETIME
# ============================================================================

# Annual degradation rates (%/year)
DEGRADATION_RATES = {
    'c_Si': 0.5,
    'Perovskite': 2.0,
    'CIGS': 1.2,
    'CdTe': 0.8,
    'HJT': 0.25,
    'TOPCon': 0.30
}

# Expected lifetimes (years)
COMPONENT_LIFETIMES = {
    'module': 25,
    'inverter_string': 15,
    'inverter_central': 20,
    'mounting_structure': 30,
    'cables': 25,
    'combiner_box': 20,
    'monitoring_system': 10
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'end_of_life': 80,          # % of original efficiency
    'warranty_guarantee': 80,    # % at 25 years
    'alarm_threshold': 85,       # % below expected triggers alarm
    'critical_threshold': 75     # % requires immediate action
}


# ============================================================================
# IEC TESTING STANDARDS
# ============================================================================

IEC_STANDARDS = {
    'IEC_61215': 'Design qualification and type approval',
    'IEC_61730': 'Safety qualification',
    'IEC_61853': 'PV module performance testing',
    'IEC_62804': 'Potential Induced Degradation (PID)',
    'IEC_61701': 'Salt mist corrosion',
    'IEC_60891': 'I-V curve procedures',
    'IEC_61724': 'System performance monitoring'
}

# Test duration requirements (hours)
IEC_TEST_DURATIONS = {
    'damp_heat': 1000,
    'thermal_cycling': 200,
    'humidity_freeze': 80,
    'uv_preconditioning': 120,
    'outdoor_exposure': 720,
    'pid_test': 96
}

# Test temperature ranges
IEC_TEST_TEMPERATURES = {
    'low_temp': -40,    # °C
    'high_temp': 85,    # °C
    'damp_heat': 85,    # °C with 85% RH
    'noct_ambient': 20  # °C
}


# ============================================================================
# CTM LOSS CATEGORIES
# ============================================================================

CTM_LOSS_CATEGORIES = [
    'Optical',
    'Electrical',
    'Thermal',
    'Geometric',
    'Environmental',
    'Manufacturing',
    'Degradation'
]

# Typical CTM ratio ranges
CTM_RATIO_RANGES = {
    'excellent': (0.95, 1.00),
    'good': (0.90, 0.95),
    'average': (0.85, 0.90),
    'poor': (0.80, 0.85)
}


# ============================================================================
# FINANCIAL CONSTANTS
# ============================================================================

# Default financial parameters
DEFAULT_DISCOUNT_RATE = 8.0         # %
DEFAULT_INFLATION_RATE = 2.0        # %
DEFAULT_TAX_RATE = 21.0             # %
DEFAULT_LOAN_TERM = 15              # years
DEFAULT_PROJECT_LIFETIME = 25       # years

# Cost ranges ($/W)
CAPEX_RANGES = {
    'residential': (2.5, 3.5),
    'commercial': (1.8, 2.5),
    'utility': (0.9, 1.5)
}

# O&M costs ($/kW/year)
OPEX_RANGES = {
    'residential': (15, 25),
    'commercial': (12, 20),
    'utility': (8, 15)
}

# Electricity prices ($/kWh) - US averages
ELECTRICITY_PRICES = {
    'residential': 0.13,
    'commercial': 0.11,
    'industrial': 0.07,
    'utility_ppa': 0.04
}

# Bankability thresholds
BANKABILITY_THRESHOLDS = {
    'min_dscr': 1.20,               # Debt Service Coverage Ratio
    'min_irr': 8.0,                 # %
    'max_payback': 12,              # years
    'min_equity_irr': 10.0          # %
}


# ============================================================================
# CIRCULARITY CONSTANTS
# ============================================================================

# Material recovery rates (%)
MATERIAL_RECOVERY_RATES = {
    'silicon': 95,
    'glass': 98,
    'aluminum': 96,
    'copper': 98,
    'silver': 90,
    'plastic': 60
}

# Material weights in typical 60-cell c-Si module (kg)
MODULE_MATERIAL_COMPOSITION = {
    'glass': 12.0,
    'aluminum_frame': 2.5,
    'silicon_cells': 3.5,
    'copper': 0.5,
    'silver': 0.03,
    'eva_encapsulant': 1.2,
    'backsheet': 0.8,
    'junction_box': 0.3,
    'other': 0.5
}

# Material prices ($/kg)
MATERIAL_PRICES = {
    'silicon': 2.0,
    'glass': 0.05,
    'aluminum': 2.5,
    'copper': 8.0,
    'silver': 600.0,
    'plastic': 0.5,
    'eva': 2.0
}

# Recycling costs ($/module)
RECYCLING_COSTS = {
    'mechanical': 2.0,
    'thermal': 3.5,
    'chemical': 5.0
}

# Reuse value depreciation (%/year)
REUSE_DEPRECIATION_RATE = 5.0


# ============================================================================
# ENERGY STORAGE CONSTANTS
# ============================================================================

# Battery technology specifications
BATTERY_SPECS = {
    'lithium_ion': {
        'energy_density': 250,      # Wh/kg
        'power_density': 1000,      # W/kg
        'efficiency': 95,           # %
        'cycle_life': 6000,
        'calendar_life': 15,        # years
        'cost_per_kwh': 400         # $/kWh
    },
    'lead_acid': {
        'energy_density': 40,
        'power_density': 180,
        'efficiency': 80,
        'cycle_life': 1500,
        'calendar_life': 8,
        'cost_per_kwh': 200
    },
    'flow_battery': {
        'energy_density': 30,
        'power_density': 20,
        'efficiency': 75,
        'cycle_life': 10000,
        'calendar_life': 20,
        'cost_per_kwh': 500
    }
}

# C-rates for different applications
C_RATES = {
    'residential_backup': 0.5,      # 2-hour discharge
    'frequency_regulation': 2.0,    # 30-minute discharge
    'peak_shaving': 0.25,           # 4-hour discharge
    'energy_arbitrage': 0.2         # 5-hour discharge
}


# ============================================================================
# WEATHER & LOCATION CONSTANTS
# ============================================================================

# Typical meteorological year (TMY) sources
TMY_SOURCES = [
    'PVGIS',
    'NASA_SSE',
    'Meteonorm',
    'NREL_NSRDB',
    'SOLCAST'
]

# Climate zones
CLIMATE_ZONES = {
    'tropical': {'latitude_range': (-23.5, 23.5)},
    'subtropical': {'latitude_range': (23.5, 35.0)},
    'temperate': {'latitude_range': (35.0, 50.0)},
    'cold': {'latitude_range': (50.0, 70.0)}
}

# Soiling rates by environment (%/month)
SOILING_RATES = {
    'clean': 0.2,
    'rural': 0.5,
    'suburban': 1.0,
    'urban': 1.5,
    'industrial': 2.5,
    'desert': 3.0
}


# ============================================================================
# FAULT DETECTION CONSTANTS
# ============================================================================

# Fault severity multipliers (power loss factor)
FAULT_SEVERITY_MULTIPLIERS = {
    'hotspot': 0.15,            # 15% power loss
    'cell_crack': 0.05,         # 5% power loss
    'diode_failure': 0.30,      # 30% power loss
    'soiling': 0.10,            # 10% power loss
    'shading': 0.20,            # 20% power loss
    'delamination': 0.08        # 8% power loss
}

# Thermal imaging thresholds
THERMAL_THRESHOLDS = {
    'normal_temp_diff': 10,         # °C above ambient
    'hotspot_warning': 15,          # °C above module average
    'hotspot_critical': 25,         # °C above module average
    'max_operating_temp': 85        # °C
}

# I-V curve analysis thresholds
IV_CURVE_THRESHOLDS = {
    'min_fill_factor': 0.70,
    'normal_fill_factor': 0.75,
    'good_fill_factor': 0.78,
    'excellent_fill_factor': 0.82
}


# ============================================================================
# DATA QUALITY CONSTANTS
# ============================================================================

# Data quality score thresholds
DATA_QUALITY_THRESHOLDS = {
    'excellent': 95,
    'good': 85,
    'acceptable': 75,
    'poor': 60,
    'unacceptable': 0
}

# Maximum acceptable data gaps
MAX_DATA_GAPS = {
    'real_time': 5,         # minutes
    'hourly': 2,            # hours
    'daily': 1,             # days
    'monthly': 3            # days
}


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

UNIT_CONVERSIONS = {
    # Power
    'kw_to_w': 1000,
    'mw_to_kw': 1000,
    'gw_to_mw': 1000,

    # Energy
    'kwh_to_wh': 1000,
    'mwh_to_kwh': 1000,
    'gwh_to_mwh': 1000,

    # Area
    'm2_to_cm2': 10000,
    'km2_to_m2': 1000000,

    # Temperature
    'c_to_k_offset': 273.15,
    'c_to_f_factor': 1.8,
    'c_to_f_offset': 32,

    # Pressure
    'pa_to_bar': 0.00001,
    'psi_to_pa': 6894.76
}


# ============================================================================
# COLOR SCHEMES (for visualization)
# ============================================================================

COLOR_SCHEMES = {
    'performance': {
        'excellent': '#2ecc71',     # Green
        'good': '#3498db',          # Blue
        'warning': '#f39c12',       # Orange
        'critical': '#e74c3c',      # Red
        'offline': '#95a5a6'        # Gray
    },
    'efficiency': {
        'high': '#27ae60',
        'medium': '#f1c40f',
        'low': '#e67e22'
    },
    'circularity': {
        'reduce': '#1abc9c',
        'reuse': '#3498db',
        'recycle': '#9b59b6'
    }
}


# ============================================================================
# API & INTEGRATION CONSTANTS
# ============================================================================

# API timeout settings (seconds)
API_TIMEOUTS = {
    'weather_api': 30,
    'scada_api': 10,
    'database': 60,
    'external_services': 45
}

# Rate limits (requests per minute)
API_RATE_LIMITS = {
    'weather_api': 60,
    'pvgis': 30,
    'nasa_sse': 100
}


# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALIDATION_RANGES = {
    'efficiency': {'min': 0, 'max': 50},                    # %
    'performance_ratio': {'min': 0, 'max': 100},            # %
    'irradiance': {'min': 0, 'max': 1500},                  # W/m²
    'ambient_temp': {'min': -50, 'max': 60},                # °C
    'module_temp': {'min': -40, 'max': 100},                # °C
    'voltage': {'min': 0, 'max': 1500},                     # V
    'current': {'min': 0, 'max': 20},                       # A
    'power': {'min': 0, 'max': 1000000},                    # kW
    'wind_speed': {'min': 0, 'max': 50},                    # m/s
    'humidity': {'min': 0, 'max': 100},                     # %
}


# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'invalid_efficiency': "Efficiency must be between 0% and 50%",
    'invalid_irradiance': "Irradiance must be between 0 and 1500 W/m²",
    'invalid_temperature': "Temperature out of valid range",
    'missing_data': "Required data not provided",
    'database_error': "Database connection error",
    'api_timeout': "API request timed out",
    'invalid_configuration': "Invalid system configuration"
}


# Export all constants
__all__ = [
    'APP_NAME', 'APP_VERSION', 'TOTAL_SESSIONS', 'TOTAL_BRANCHES',
    'SOLAR_CONSTANT', 'STANDARD_TEST_CONDITIONS_IRRADIANCE',
    'DEGRADATION_RATES', 'COMPONENT_LIFETIMES', 'IEC_STANDARDS',
    'CTM_LOSS_CATEGORIES', 'DEFAULT_DISCOUNT_RATE', 'MATERIAL_RECOVERY_RATES',
    'BATTERY_SPECS', 'VALIDATION_RANGES', 'ERROR_MESSAGES'
]
