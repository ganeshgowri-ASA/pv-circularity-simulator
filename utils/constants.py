"""
Constants and Configuration for PV Circularity Simulator

This module contains all standards, configurations, presets, and reference data
used across the entire PV lifecycle simulation platform.

Author: PV Circularity Simulator Team
Version: 1.0 (71 Sessions Integrated)
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
import numpy as np

# ============================================================================
# MATERIAL CONSTANTS
# ============================================================================

class MaterialType(Enum):
    """PV material types."""
    SILICON_MONO = "Monocrystalline Silicon"
    SILICON_POLY = "Polycrystalline Silicon"
    PEROVSKITE = "Perovskite"
    CIGS = "CIGS"
    CDTE = "CdTe"
    BIFACIAL = "Bifacial Silicon"
    HIT = "HIT (Heterojunction)"
    PERC = "PERC"
    TOPCON = "TOPCon"
    IBC = "IBC (Interdigitated Back Contact)"

MATERIALS_DATABASE: Dict[str, Dict[str, Any]] = {
    "Silicon (c-Si)": {
        "efficiency": 21.5,
        "cost_per_wp": 0.45,
        "degradation_rate": 0.5,
        "recyclability": 95,
        "bandgap_ev": 1.12,
        "temp_coefficient": -0.45,
        "density": 2.33,
        "thermal_conductivity": 148,
        "lifespan_years": 25,
        "carbon_footprint": 45,
    },
    "Perovskite": {
        "efficiency": 24.2,
        "cost_per_wp": 0.38,
        "degradation_rate": 2.0,
        "recyclability": 65,
        "bandgap_ev": 1.55,
        "temp_coefficient": -0.30,
        "density": 4.0,
        "thermal_conductivity": 0.5,
        "lifespan_years": 15,
        "carbon_footprint": 30,
    },
    "CIGS": {
        "efficiency": 18.8,
        "cost_per_wp": 0.52,
        "degradation_rate": 1.2,
        "recyclability": 75,
        "bandgap_ev": 1.15,
        "temp_coefficient": -0.36,
        "density": 5.75,
        "thermal_conductivity": 10,
        "lifespan_years": 25,
        "carbon_footprint": 38,
    },
    "CdTe": {
        "efficiency": 20.5,
        "cost_per_wp": 0.40,
        "degradation_rate": 0.8,
        "recyclability": 90,
        "bandgap_ev": 1.45,
        "temp_coefficient": -0.25,
        "density": 5.85,
        "thermal_conductivity": 6.2,
        "lifespan_years": 25,
        "carbon_footprint": 25,
    },
    "Bifacial Si": {
        "efficiency": 22.1,
        "cost_per_wp": 0.48,
        "degradation_rate": 0.6,
        "recyclability": 96,
        "bandgap_ev": 1.12,
        "temp_coefficient": -0.40,
        "density": 2.33,
        "thermal_conductivity": 148,
        "lifespan_years": 30,
        "carbon_footprint": 50,
    },
}

# ============================================================================
# CTM LOSS FACTORS (Fraunhofer ISE Standard - 24 factors)
# ============================================================================

CTM_LOSS_FACTORS: Dict[str, Dict[str, Any]] = {
    "k1_reflection": {"loss_pct": 2.5, "description": "Front glass reflection loss"},
    "k2_absorption": {"loss_pct": 1.8, "description": "Glass absorption"},
    "k3_transmission": {"loss_pct": 0.5, "description": "Transmission losses"},
    "k4_soiling": {"loss_pct": 1.2, "description": "Soiling and dust"},
    "k5_temperature": {"loss_pct": 3.2, "description": "Temperature coefficient"},
    "k6_low_irradiance": {"loss_pct": 1.5, "description": "Low irradiance losses"},
    "k7_spectral": {"loss_pct": 1.0, "description": "Spectral mismatch"},
    "k8_shading": {"loss_pct": 0.8, "description": "Self-shading"},
    "k9_mismatch": {"loss_pct": 1.5, "description": "Cell mismatch"},
    "k10_wiring": {"loss_pct": 0.8, "description": "Internal wiring resistance"},
    "k11_connection": {"loss_pct": 0.5, "description": "Connection losses"},
    "k12_lid": {"loss_pct": 2.0, "description": "Light-induced degradation"},
    "k13_pid": {"loss_pct": 1.0, "description": "Potential-induced degradation"},
    "k14_encapsulation": {"loss_pct": 0.7, "description": "Encapsulation losses"},
    "k15_backsheet": {"loss_pct": 0.3, "description": "Backsheet reflection"},
    "k16_edge_delete": {"loss_pct": 0.4, "description": "Edge deletion area"},
    "k17_bus_bar": {"loss_pct": 2.5, "description": "Bus bar shading"},
    "k18_junction_box": {"loss_pct": 0.2, "description": "Junction box shading"},
    "k19_cell_gap": {"loss_pct": 3.0, "description": "Cell gap losses"},
    "k20_lamination": {"loss_pct": 0.5, "description": "Lamination stress"},
    "k21_quality": {"loss_pct": 1.0, "description": "Manufacturing quality"},
    "k22_sorting": {"loss_pct": 0.8, "description": "Cell sorting binning"},
    "k23_flash_test": {"loss_pct": 0.3, "description": "Flash test uncertainty"},
    "k24_outdoor": {"loss_pct": 1.5, "description": "Indoor-outdoor spectral"},
}

# ============================================================================
# IEC TESTING STANDARDS
# ============================================================================

IEC_STANDARDS: Dict[str, Dict[str, Any]] = {
    "IEC_61215": {
        "name": "Crystalline Silicon Terrestrial PV Modules",
        "tests": [
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
            "Bypass Diode",
        ],
        "duration_hours": 1200,
    },
    "IEC_61730": {
        "name": "PV Module Safety Qualification",
        "tests": [
            "Construction Requirements",
            "Marking and Instructions",
            "Environmental Requirements",
            "Electrical Requirements",
            "Mechanical Requirements",
        ],
        "duration_hours": 480,
    },
    "IEC_62804": {
        "name": "PID Testing",
        "tests": ["PID Stress Test", "Recovery Test"],
        "duration_hours": 192,
    },
    "IEC_61853": {
        "name": "PV Module Performance Testing",
        "tests": [
            "Irradiance and Temperature Performance",
            "Spectral Responsivity",
            "Incidence Angle and Module Temperature",
        ],
        "duration_hours": 240,
    },
}

# ============================================================================
# WEATHER AND ENVIRONMENTAL DATA
# ============================================================================

WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "Desert": {
        "annual_ghi": 2400,
        "avg_temp": 35,
        "humidity": 20,
        "wind_speed": 4.5,
        "soiling_rate": 0.3,
        "rainfall_mm": 100,
    },
    "Tropical": {
        "annual_ghi": 1800,
        "avg_temp": 28,
        "humidity": 80,
        "wind_speed": 3.2,
        "soiling_rate": 0.1,
        "rainfall_mm": 2500,
    },
    "Temperate": {
        "annual_ghi": 1400,
        "avg_temp": 15,
        "humidity": 65,
        "wind_speed": 4.0,
        "soiling_rate": 0.15,
        "rainfall_mm": 800,
    },
    "Coastal": {
        "annual_ghi": 1900,
        "avg_temp": 22,
        "humidity": 75,
        "wind_speed": 5.5,
        "soiling_rate": 0.12,
        "rainfall_mm": 1200,
    },
    "Mountain": {
        "annual_ghi": 1600,
        "avg_temp": 10,
        "humidity": 60,
        "wind_speed": 6.0,
        "soiling_rate": 0.08,
        "rainfall_mm": 1000,
    },
}

# ============================================================================
# SYSTEM DESIGN PARAMETERS
# ============================================================================

INVERTER_TYPES: Dict[str, Dict[str, Any]] = {
    "String": {
        "efficiency": 97.5,
        "cost_per_kw": 150,
        "lifespan_years": 12,
        "power_range": (1, 100),
        "max_dc_voltage": 1000,
        "mppt_efficiency": 99.5,
    },
    "Central": {
        "efficiency": 98.5,
        "cost_per_kw": 120,
        "lifespan_years": 15,
        "power_range": (100, 5000),
        "max_dc_voltage": 1500,
        "mppt_efficiency": 99.8,
    },
    "Micro": {
        "efficiency": 96.0,
        "cost_per_kw": 200,
        "lifespan_years": 25,
        "power_range": (0.2, 1),
        "max_dc_voltage": 60,
        "mppt_efficiency": 98.5,
    },
    "Hybrid": {
        "efficiency": 97.0,
        "cost_per_kw": 180,
        "lifespan_years": 12,
        "power_range": (3, 50),
        "max_dc_voltage": 1000,
        "mppt_efficiency": 99.0,
        "battery_compatible": True,
    },
}

MOUNTING_TYPES: Dict[str, Dict[str, Any]] = {
    "Fixed Tilt": {
        "cost_per_kw": 50,
        "energy_gain": 0,
        "maintenance": "Low",
    },
    "Single-Axis Tracking": {
        "cost_per_kw": 120,
        "energy_gain": 25,
        "maintenance": "Medium",
    },
    "Dual-Axis Tracking": {
        "cost_per_kw": 200,
        "energy_gain": 40,
        "maintenance": "High",
    },
    "Rooftop": {
        "cost_per_kw": 80,
        "energy_gain": -5,
        "maintenance": "Low",
    },
}

# ============================================================================
# MONITORING AND DIAGNOSTICS
# ============================================================================

FAULT_TYPES: Dict[str, Dict[str, Any]] = {
    "Hot Spot": {
        "severity": "High",
        "detection_method": "IR Imaging",
        "action": "Immediate inspection",
        "power_loss_pct": 15,
    },
    "Bypass Diode Failure": {
        "severity": "Medium",
        "detection_method": "IV Curve",
        "action": "Schedule replacement",
        "power_loss_pct": 33,
    },
    "Soiling": {
        "severity": "Low",
        "detection_method": "Performance Ratio",
        "action": "Cleaning",
        "power_loss_pct": 5,
    },
    "Shading": {
        "severity": "Medium",
        "detection_method": "String Monitoring",
        "action": "Trim vegetation",
        "power_loss_pct": 20,
    },
    "PID": {
        "severity": "High",
        "detection_method": "EL Imaging",
        "action": "Voltage correction",
        "power_loss_pct": 30,
    },
    "Cell Crack": {
        "severity": "Medium",
        "detection_method": "EL Imaging",
        "action": "Monitor degradation",
        "power_loss_pct": 10,
    },
    "Delamination": {
        "severity": "High",
        "detection_method": "Visual + IR",
        "action": "Module replacement",
        "power_loss_pct": 25,
    },
    "Corrosion": {
        "severity": "Medium",
        "detection_method": "Visual Inspection",
        "action": "Replace affected",
        "power_loss_pct": 12,
    },
}

SENSOR_TYPES: List[str] = [
    "Irradiance (POA)",
    "Module Temperature",
    "Ambient Temperature",
    "Wind Speed",
    "Humidity",
    "String Current",
    "String Voltage",
    "Inverter Power",
    "Grid Frequency",
    "Energy Meter",
]

# ============================================================================
# CIRCULARITY AND RECYCLING
# ============================================================================

RECYCLING_PROCESSES: Dict[str, Dict[str, Any]] = {
    "Mechanical": {
        "efficiency": 85,
        "cost_per_module": 5,
        "recovered_materials": ["Glass", "Aluminum", "Copper"],
        "energy_consumption": 50,
    },
    "Thermal": {
        "efficiency": 92,
        "cost_per_module": 8,
        "recovered_materials": ["Silicon", "Silver", "Glass"],
        "energy_consumption": 120,
    },
    "Chemical": {
        "efficiency": 95,
        "cost_per_module": 12,
        "recovered_materials": ["Silicon", "Silver", "Copper", "Lead"],
        "energy_consumption": 80,
    },
}

MATERIAL_RECOVERY_RATES: Dict[str, float] = {
    "Glass": 95,
    "Aluminum": 98,
    "Silicon": 85,
    "Silver": 92,
    "Copper": 96,
    "Lead": 88,
    "EVA": 70,
    "Backsheet": 65,
}

# ============================================================================
# FINANCIAL PARAMETERS
# ============================================================================

FINANCIAL_DEFAULTS: Dict[str, Any] = {
    "discount_rate": 6.0,
    "inflation_rate": 2.5,
    "electricity_price": 0.12,
    "electricity_escalation": 3.0,
    "system_lifetime": 25,
    "inverter_replacement_year": 12,
    "o_and_m_annual": 15,  # $/kW/year
    "insurance_annual": 0.25,  # % of capex
    "tax_rate": 21,
    "depreciation_period": 5,
    "itc_percent": 30,  # Investment Tax Credit
}

ELECTRICITY_TARIFFS: Dict[str, float] = {
    "Residential": 0.13,
    "Commercial": 0.11,
    "Industrial": 0.08,
    "Utility": 0.06,
}

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

PERFORMANCE_THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    # (critical, warning, good)
    "Performance Ratio": (0.75, 0.80, 0.85),
    "System Efficiency": (0.12, 0.14, 0.16),
    "Capacity Factor": (0.15, 0.18, 0.22),
    "Availability": (0.95, 0.97, 0.99),
    "Inverter Efficiency": (0.95, 0.97, 0.98),
}

# ============================================================================
# STC AND NOCT CONDITIONS
# ============================================================================

STC_CONDITIONS: Dict[str, float] = {
    "irradiance": 1000,  # W/m²
    "cell_temperature": 25,  # °C
    "air_mass": 1.5,
}

NOCT_CONDITIONS: Dict[str, float] = {
    "irradiance": 800,  # W/m²
    "ambient_temperature": 20,  # °C
    "wind_speed": 1,  # m/s
    "mounting": 0,  # open rack
}

# ============================================================================
# DEGRADATION MODELS
# ============================================================================

DEGRADATION_MODES: Dict[str, Dict[str, Any]] = {
    "Linear": {
        "formula": "P(t) = P0 * (1 - rate * t)",
        "typical_rate": 0.5,
    },
    "Exponential": {
        "formula": "P(t) = P0 * exp(-rate * t)",
        "typical_rate": 0.006,
    },
    "Piecewise": {
        "formula": "LID + Linear",
        "lid_year1": 2.0,
        "linear_rate": 0.5,
    },
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

SIMULATION_DEFAULTS: Dict[str, Any] = {
    "time_step_hours": 1,
    "simulation_years": 25,
    "monte_carlo_iterations": 1000,
    "confidence_interval": 0.95,
    "weather_year_type": "TMY3",  # Typical Meteorological Year
}

# ============================================================================
# SCAPS-1D SIMULATION PARAMETERS
# ============================================================================

SCAPS_DEFAULTS: Dict[str, Any] = {
    "substrate_types": ["Glass", "Plastic", "Metal", "Flexible"],
    "device_thickness_um": (0.1, 10.0),
    "simulation_temperature": 300,  # K
    "illumination": "AM1.5G",
    "voltage_points": 100,
    "convergence_criteria": 1e-6,
}

# ============================================================================
# HYBRID ENERGY STORAGE
# ============================================================================

BATTERY_TYPES: Dict[str, Dict[str, Any]] = {
    "Lithium-Ion": {
        "efficiency": 95,
        "cycle_life": 5000,
        "cost_per_kwh": 400,
        "degradation_per_cycle": 0.01,
        "depth_of_discharge": 90,
    },
    "Lead-Acid": {
        "efficiency": 85,
        "cycle_life": 1500,
        "cost_per_kwh": 200,
        "degradation_per_cycle": 0.03,
        "depth_of_discharge": 50,
    },
    "Flow Battery": {
        "efficiency": 80,
        "cycle_life": 10000,
        "cost_per_kwh": 500,
        "degradation_per_cycle": 0.005,
        "depth_of_discharge": 100,
    },
}

# ============================================================================
# REVAMP AND RETROFIT OPTIONS
# ============================================================================

REVAMP_OPTIONS: Dict[str, Dict[str, Any]] = {
    "Module Replacement": {
        "cost_per_kw": 600,
        "efficiency_gain": 25,
        "lifespan_extension": 20,
    },
    "Inverter Upgrade": {
        "cost_per_kw": 150,
        "efficiency_gain": 2,
        "lifespan_extension": 12,
    },
    "Tracker Retrofit": {
        "cost_per_kw": 120,
        "efficiency_gain": 20,
        "lifespan_extension": 0,
    },
    "BESS Integration": {
        "cost_per_kw": 800,
        "efficiency_gain": 0,
        "value_enhancement": 40,
    },
}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

COLOR_PALETTE: Dict[str, str] = {
    "primary": "#2ecc71",
    "secondary": "#3498db",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "success": "#27ae60",
    "info": "#16a085",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
}

# ============================================================================
# EXPORT FORMATS
# ============================================================================

EXPORT_FORMATS: List[str] = [
    "CSV",
    "Excel",
    "JSON",
    "PDF Report",
    "HTML Dashboard",
    "PowerPoint",
]

# ============================================================================
# VERSION INFORMATION
# ============================================================================

VERSION_INFO: Dict[str, str] = {
    "version": "1.0.0",
    "release_date": "2025-11-17",
    "sessions_integrated": "71",
    "branches": "15",
    "status": "Production-Ready",
}
