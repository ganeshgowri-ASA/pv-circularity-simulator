"""
Shared constants for PV Circularity Simulator.

This module contains all constants used across the application including:
- Physical constants
- Material properties
- IEC standard parameters
- Financial parameters
- System configuration defaults
"""

from typing import Dict, List
import numpy as np

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

PLANCK_CONSTANT: float = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K
ELEMENTARY_CHARGE: float = 1.602176634e-19  # C
SPEED_OF_LIGHT: float = 299792458  # m/s
STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/(m²·K⁴)
STANDARD_TEMPERATURE: float = 298.15  # K (25°C)
STANDARD_IRRADIANCE: float = 1000.0  # W/m²
AIR_MASS_COEFFICIENT: float = 1.5  # AM1.5G

# ============================================================================
# MATERIAL PROPERTIES DATABASE
# ============================================================================

MATERIAL_PROPERTIES: Dict[str, Dict] = {
    "c-Si": {
        "name": "Crystalline Silicon",
        "type": "semiconductor",
        "bandgap": 1.12,  # eV
        "efficiency_range": (20.0, 26.7),  # %
        "cost_per_wp": 0.45,  # $/Wp
        "degradation_rate": 0.5,  # %/year
        "recyclability": 95,  # %
        "density": 2329,  # kg/m³
        "thermal_conductivity": 148,  # W/(m·K)
        "temp_coefficient": -0.45,  # %/°C
        "color": "#4A90E2"
    },
    "perovskite": {
        "name": "Perovskite",
        "type": "semiconductor",
        "bandgap": 1.55,  # eV
        "efficiency_range": (22.0, 25.8),  # %
        "cost_per_wp": 0.38,  # $/Wp
        "degradation_rate": 2.0,  # %/year
        "recyclability": 65,  # %
        "density": 4000,  # kg/m³
        "thermal_conductivity": 0.5,  # W/(m·K)
        "temp_coefficient": -0.30,  # %/°C
        "color": "#E94B3C"
    },
    "CIGS": {
        "name": "Copper Indium Gallium Selenide",
        "type": "thin-film",
        "bandgap": 1.15,  # eV
        "efficiency_range": (17.0, 22.9),  # %
        "cost_per_wp": 0.52,  # $/Wp
        "degradation_rate": 1.2,  # %/year
        "recyclability": 75,  # %
        "density": 5760,  # kg/m³
        "thermal_conductivity": 8.4,  # W/(m·K)
        "temp_coefficient": -0.36,  # %/°C
        "color": "#6BCB77"
    },
    "CdTe": {
        "name": "Cadmium Telluride",
        "type": "thin-film",
        "bandgap": 1.45,  # eV
        "efficiency_range": (18.0, 21.0),  # %
        "cost_per_wp": 0.40,  # $/Wp
        "degradation_rate": 0.8,  # %/year
        "recyclability": 90,  # %
        "density": 5850,  # kg/m³
        "thermal_conductivity": 6.2,  # W/(m·K)
        "temp_coefficient": -0.25,  # %/°C
        "color": "#FFD93D"
    },
    "tandem_perovskite_si": {
        "name": "Perovskite/Silicon Tandem",
        "type": "tandem",
        "bandgap": (1.68, 1.12),  # eV (top, bottom)
        "efficiency_range": (28.0, 33.7),  # %
        "cost_per_wp": 0.58,  # $/Wp
        "degradation_rate": 1.0,  # %/year
        "recyclability": 80,  # %
        "density": 3000,  # kg/m³
        "thermal_conductivity": 50,  # W/(m·K)
        "temp_coefficient": -0.35,  # %/°C
        "color": "#9B59B6"
    },
    "bifacial_si": {
        "name": "Bifacial Silicon",
        "type": "advanced_silicon",
        "bandgap": 1.12,  # eV
        "efficiency_range": (21.0, 24.0),  # %
        "cost_per_wp": 0.48,  # $/Wp
        "degradation_rate": 0.6,  # %/year
        "recyclability": 96,  # %
        "density": 2329,  # kg/m³
        "thermal_conductivity": 148,  # W/(m·K)
        "temp_coefficient": -0.40,  # %/°C
        "bifaciality_factor": 0.70,  # 70% rear efficiency
        "color": "#3498DB"
    }
}

# ============================================================================
# CTM (CELL-TO-MODULE) LOSS FACTORS (Fraunhofer ISE)
# ============================================================================

CTM_LOSS_FACTORS: Dict[str, Dict] = {
    "k1": {"name": "Reflection losses", "typical_range": (1.5, 3.5), "unit": "%"},
    "k2": {"name": "Soiling losses", "typical_range": (1.0, 5.0), "unit": "%"},
    "k3": {"name": "Temperature losses", "typical_range": (2.0, 8.0), "unit": "%"},
    "k4": {"name": "Series resistance losses", "typical_range": (1.5, 3.0), "unit": "%"},
    "k5": {"name": "Mismatch losses", "typical_range": (1.0, 3.0), "unit": "%"},
    "k6": {"name": "Wiring losses", "typical_range": (0.5, 2.0), "unit": "%"},
    "k7": {"name": "Shading losses", "typical_range": (0.0, 5.0), "unit": "%"},
    "k8": {"name": "Spectral losses", "typical_range": (1.0, 3.0), "unit": "%"},
    "k9": {"name": "IAM losses", "typical_range": (1.0, 4.0), "unit": "%"},
    "k10": {"name": "Cell encapsulation", "typical_range": (0.5, 2.0), "unit": "%"},
    "k11": {"name": "Front glass absorption", "typical_range": (1.0, 2.5), "unit": "%"},
    "k12": {"name": "Back sheet reflection", "typical_range": (0.5, 1.5), "unit": "%"},
    "k13": {"name": "Cell spacing losses", "typical_range": (2.0, 4.0), "unit": "%"},
    "k14": {"name": "Ribbon shading", "typical_range": (1.5, 3.0), "unit": "%"},
    "k15": {"name": "Busbar shading", "typical_range": (2.0, 4.0), "unit": "%"},
    "k16": {"name": "Temperature non-uniformity", "typical_range": (0.5, 2.0), "unit": "%"},
    "k17": {"name": "Degradation losses", "typical_range": (0.5, 2.5), "unit": "%"},
    "k18": {"name": "Light-induced degradation", "typical_range": (1.0, 3.0), "unit": "%"},
    "k19": {"name": "Potential-induced degradation", "typical_range": (0.0, 5.0), "unit": "%"},
    "k20": {"name": "Quality losses", "typical_range": (0.5, 2.0), "unit": "%"},
    "k21": {"name": "Manufacturing tolerances", "typical_range": (0.5, 1.5), "unit": "%"},
    "k22": {"name": "Optical losses", "typical_range": (1.0, 3.0), "unit": "%"},
    "k23": {"name": "Interconnection losses", "typical_range": (0.5, 2.0), "unit": "%"},
    "k24": {"name": "Other module losses", "typical_range": (0.5, 2.0), "unit": "%"}
}

# ============================================================================
# IEC STANDARD PARAMETERS
# ============================================================================

IEC_STANDARDS: Dict[str, Dict] = {
    "IEC_61215": {
        "name": "Crystalline Silicon Terrestrial PV Modules - Design Qualification and Type Approval",
        "test_categories": [
            "Visual Inspection",
            "Maximum Power Determination",
            "Insulation Test",
            "Temperature Coefficients",
            "NOCT Measurement",
            "Low Irradiance Performance",
            "Outdoor Exposure",
            "Hot-Spot Endurance",
            "UV Preconditioning",
            "Thermal Cycling",
            "Humidity-Freeze",
            "Damp Heat",
            "Robustness of Terminations",
            "Wet Leakage Current",
            "Mechanical Load",
            "Hail Impact",
            "Bypass Diode Thermal"
        ],
        "thermal_cycling_count": 200,
        "damp_heat_duration": 1000,  # hours
        "humidity_freeze_cycles": 10
    },
    "IEC_61730": {
        "name": "PV Module Safety Qualification",
        "parts": ["Part 1: Requirements for Construction", "Part 2: Requirements for Testing"],
        "test_categories": [
            "Fire Test",
            "Electrical Shock Protection",
            "Mechanical Stress",
            "Environmental Stress"
        ]
    },
    "IEC_63202": {
        "name": "Photovoltaic Cells - Measurement of Light-Induced Degradation",
        "degradation_types": ["LID", "LETID"],
        "test_duration": 100,  # hours minimum
        "irradiance": 1000  # W/m²
    },
    "IEC_63209": {
        "name": "Photovoltaic Modules - Extended-Stress Testing",
        "enhanced_tests": [
            "Dynamic Mechanical Load",
            "Extended Thermal Cycling",
            "Damp Heat plus High Voltage Bias"
        ]
    },
    "IEC_TS_63279": {
        "name": "Photovoltaic Modules - Modeling for Thermal Characteristics",
        "parameters": ["U-value", "Thermal capacitance", "Wind effects"]
    }
}

# ============================================================================
# SYSTEM DESIGN PARAMETERS
# ============================================================================

INVERTER_TYPES: Dict[str, Dict] = {
    "string": {
        "name": "String Inverter",
        "power_range": (1, 100),  # kW
        "efficiency": 0.98,
        "cost_per_kw": 150,  # $/kW
        "mppt_channels": (1, 4),
        "voltage_range": (150, 1000)  # V
    },
    "central": {
        "name": "Central Inverter",
        "power_range": (100, 5000),  # kW
        "efficiency": 0.985,
        "cost_per_kw": 100,  # $/kW
        "mppt_channels": (4, 20),
        "voltage_range": (600, 1500)  # V
    },
    "micro": {
        "name": "Microinverter",
        "power_range": (0.25, 0.5),  # kW
        "efficiency": 0.96,
        "cost_per_kw": 300,  # $/kW
        "mppt_channels": (1, 1),
        "voltage_range": (25, 60)  # V
    },
    "hybrid": {
        "name": "Hybrid Inverter",
        "power_range": (3, 100),  # kW
        "efficiency": 0.975,
        "cost_per_kw": 200,  # $/kW
        "mppt_channels": (2, 6),
        "voltage_range": (150, 1000),  # V
        "battery_compatible": True
    }
}

MOUNTING_TYPES: Dict[str, Dict] = {
    "fixed_tilt": {
        "name": "Fixed Tilt",
        "cost_per_kw": 120,  # $/kW
        "maintenance_factor": 1.0,
        "performance_boost": 1.0
    },
    "single_axis": {
        "name": "Single-Axis Tracking",
        "cost_per_kw": 250,  # $/kW
        "maintenance_factor": 1.2,
        "performance_boost": 1.25
    },
    "dual_axis": {
        "name": "Dual-Axis Tracking",
        "cost_per_kw": 400,  # $/kW
        "maintenance_factor": 1.5,
        "performance_boost": 1.35
    },
    "rooftop": {
        "name": "Rooftop Mounted",
        "cost_per_kw": 100,  # $/kW
        "maintenance_factor": 0.8,
        "performance_boost": 0.95
    }
}

# ============================================================================
# PERFORMANCE MONITORING KPIs
# ============================================================================

PERFORMANCE_KPIS: List[str] = [
    "Performance Ratio (PR)",
    "Capacity Factor",
    "Specific Yield (kWh/kWp/day)",
    "System Efficiency",
    "Inverter Efficiency",
    "Temperature-Corrected PR",
    "Availability (%)",
    "Energy Yield (kWh)",
    "Peak Power (kW)",
    "DC/AC Ratio"
]

FAULT_TYPES: Dict[str, Dict] = {
    "hotspot": {
        "name": "Hot Spot",
        "severity": "high",
        "detection_method": "thermal_imaging",
        "typical_temp_delta": 15  # °C above normal
    },
    "cell_crack": {
        "name": "Cell Crack",
        "severity": "medium",
        "detection_method": "el_imaging",
        "power_loss_range": (5, 20)  # %
    },
    "bypass_diode": {
        "name": "Bypass Diode Failure",
        "severity": "high",
        "detection_method": "iv_curve",
        "power_loss_range": (10, 30)  # %
    },
    "soiling": {
        "name": "Soiling",
        "severity": "low",
        "detection_method": "visual_inspection",
        "power_loss_range": (1, 10)  # %
    },
    "delamination": {
        "name": "Delamination",
        "severity": "high",
        "detection_method": "visual_inspection",
        "power_loss_range": (5, 25)  # %
    },
    "pid": {
        "name": "Potential-Induced Degradation",
        "severity": "high",
        "detection_method": "iv_curve",
        "power_loss_range": (10, 50)  # %
    }
}

# ============================================================================
# FINANCIAL PARAMETERS
# ============================================================================

FINANCIAL_DEFAULTS: Dict[str, float] = {
    "discount_rate": 0.08,  # 8%
    "inflation_rate": 0.03,  # 3%
    "electricity_price": 0.12,  # $/kWh
    "electricity_escalation": 0.025,  # 2.5% annual
    "project_lifetime": 25,  # years
    "o_and_m_cost_per_kw": 15,  # $/kW/year
    "insurance_rate": 0.005,  # 0.5% of capital cost
    "property_tax_rate": 0.01,  # 1% of capital cost
    "federal_itc": 0.30,  # 30% Investment Tax Credit
    "depreciation_schedule": "MACRS_5"  # Modified Accelerated Cost Recovery System
}

MACRS_SCHEDULES: Dict[str, List[float]] = {
    "MACRS_5": [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576],
    "MACRS_7": [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
}

# ============================================================================
# CIRCULARITY PARAMETERS
# ============================================================================

CIRCULARITY_METRICS: Dict[str, Dict] = {
    "reuse": {
        "name": "Reuse Potential",
        "criteria": ["Remaining capacity > 80%", "No physical damage", "Age < 15 years"],
        "value_retention": 0.60  # 60% of original value
    },
    "repair": {
        "name": "Repair Value",
        "common_repairs": ["Bypass diode replacement", "Junction box replacement", "Cable replacement"],
        "cost_range": (50, 200),  # $/module
        "life_extension": 5  # years
    },
    "recycle": {
        "name": "Recycling Revenue",
        "materials_recovered": {
            "silicon": {"recovery_rate": 0.85, "price_per_kg": 2.5},
            "aluminum": {"recovery_rate": 0.95, "price_per_kg": 1.8},
            "copper": {"recovery_rate": 0.90, "price_per_kg": 6.5},
            "silver": {"recovery_rate": 0.80, "price_per_kg": 650},
            "glass": {"recovery_rate": 0.95, "price_per_kg": 0.05}
        },
        "recycling_cost_per_kg": 0.30  # $/kg
    }
}

MODULE_COMPOSITION: Dict[str, float] = {
    "glass": 0.70,  # 70% by weight
    "aluminum": 0.10,
    "silicon": 0.05,
    "copper": 0.01,
    "silver": 0.0005,
    "eva_polymer": 0.10,
    "backsheet": 0.035,
    "other": 0.0045
}

# ============================================================================
# WEATHER AND LOCATION PARAMETERS
# ============================================================================

CLIMATE_ZONES: Dict[str, Dict] = {
    "tropical": {"avg_ghi": 2000, "temp_range": (20, 35), "humidity": 0.80},
    "arid": {"avg_ghi": 2400, "temp_range": (15, 40), "humidity": 0.30},
    "temperate": {"avg_ghi": 1600, "temp_range": (5, 25), "humidity": 0.60},
    "continental": {"avg_ghi": 1400, "temp_range": (-10, 30), "humidity": 0.65},
    "polar": {"avg_ghi": 800, "temp_range": (-30, 10), "humidity": 0.70}
}

# ============================================================================
# BATTERY STORAGE PARAMETERS
# ============================================================================

BATTERY_TYPES: Dict[str, Dict] = {
    "lithium_ion": {
        "name": "Lithium-Ion",
        "energy_density": 250,  # Wh/kg
        "efficiency": 0.95,
        "cycle_life": 6000,
        "cost_per_kwh": 300,  # $/kWh
        "degradation_rate": 0.02  # per year
    },
    "lead_acid": {
        "name": "Lead-Acid",
        "energy_density": 40,  # Wh/kg
        "efficiency": 0.85,
        "cycle_life": 1500,
        "cost_per_kwh": 150,  # $/kWh
        "degradation_rate": 0.05  # per year
    },
    "flow_battery": {
        "name": "Vanadium Flow Battery",
        "energy_density": 35,  # Wh/kg
        "efficiency": 0.80,
        "cycle_life": 20000,
        "cost_per_kwh": 500,  # $/kWh
        "degradation_rate": 0.01  # per year
    }
}

# ============================================================================
# VISUALIZATION COLORS
# ============================================================================

COLOR_PALETTE: Dict[str, str] = {
    "primary": "#2ECC71",
    "secondary": "#3498DB",
    "accent": "#E74C3C",
    "warning": "#F39C12",
    "success": "#27AE60",
    "info": "#3498DB",
    "dark": "#2C3E50",
    "light": "#ECF0F1"
}

MATERIAL_COLORS: Dict[str, str] = {
    "c-Si": "#4A90E2",
    "perovskite": "#E94B3C",
    "CIGS": "#6BCB77",
    "CdTe": "#FFD93D",
    "tandem": "#9B59B6",
    "bifacial": "#3498DB"
}

# ============================================================================
# APPLICATION METADATA
# ============================================================================

APP_NAME: str = "PV Circularity Simulator"
APP_VERSION: str = "2.0.0"
TOTAL_SESSIONS: int = 71
TOTAL_BRANCHES: int = 15
