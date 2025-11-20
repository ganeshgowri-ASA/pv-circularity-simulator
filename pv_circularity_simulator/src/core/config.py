"""
Configuration
=============

Application configuration and constants.
"""

import os
from pathlib import Path
from typing import Dict, Any


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Application settings
APP_NAME = "PV Circularity Simulator"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = """
A comprehensive tool for simulating the complete lifecycle of photovoltaic systems,
from material selection and module design through system operation to end-of-life
circularity strategies.
"""

# Module configuration
MODULES = {
    'dashboard': {
        'name': 'Dashboard',
        'icon': '📊',
        'description': 'Overview and project management'
    },
    'material_selection': {
        'name': 'Material Selection',
        'icon': '🔬',
        'description': 'Select and compare PV materials'
    },
    'module_design': {
        'name': 'Module Design',
        'icon': '⚡',
        'description': 'Design PV module specifications'
    },
    'ctm_loss_analysis': {
        'name': 'CTM Loss Analysis',
        'icon': '📉',
        'description': 'Analyze cell-to-module losses'
    },
    'system_design': {
        'name': 'System Design',
        'icon': '🏗️',
        'description': 'Design complete PV system'
    },
    'eya_simulation': {
        'name': 'EYA Simulation',
        'icon': '☀️',
        'description': 'Energy yield assessment'
    },
    'performance_monitoring': {
        'name': 'Performance Monitoring',
        'icon': '📈',
        'description': 'Monitor system performance'
    },
    'fault_diagnostics': {
        'name': 'Fault Diagnostics',
        'icon': '🔍',
        'description': 'Diagnose system faults'
    },
    'hya_simulation': {
        'name': 'HYA Simulation',
        'icon': '📅',
        'description': 'Historical yield analysis'
    },
    'energy_forecasting': {
        'name': 'Energy Forecasting',
        'icon': '🔮',
        'description': 'Forecast energy production'
    },
    'revamp_repower': {
        'name': 'Revamp & Repower',
        'icon': '🔄',
        'description': 'Plan system upgrades'
    },
    'circularity_3r': {
        'name': 'Circularity (3R)',
        'icon': '♻️',
        'description': 'Reduce, Reuse, Recycle analysis'
    }
}

# CTM Loss factor defaults and ranges
CTM_LOSS_FACTORS = {
    'k1_reflection': {'default': 0.98, 'min': 0.90, 'max': 0.99, 'description': 'Reflection losses'},
    'k2_shading': {'default': 0.97, 'min': 0.90, 'max': 0.99, 'description': 'Shading losses'},
    'k3_absorption': {'default': 0.99, 'min': 0.95, 'max': 0.995, 'description': 'Absorption losses'},
    'k4_resistive': {'default': 0.98, 'min': 0.95, 'max': 0.99, 'description': 'Resistive losses'},
    'k5_mismatch': {'default': 0.98, 'min': 0.95, 'max': 0.99, 'description': 'Mismatch losses'},
    'k6_junction_box': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Junction box losses'},
    'k7_temperature': {'default': 0.96, 'min': 0.90, 'max': 0.98, 'description': 'Temperature coefficient'},
    'k8_hotspot': {'default': 0.99, 'min': 0.95, 'max': 0.995, 'description': 'Hotspot losses'},
    'k9_encapsulation': {'default': 0.99, 'min': 0.97, 'max': 0.995, 'description': 'Encapsulation losses'},
    'k10_lamination': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Lamination losses'},
    'k11_lid': {'default': 0.98, 'min': 0.95, 'max': 0.99, 'description': 'Light-induced degradation'},
    'k12_pid': {'default': 0.99, 'min': 0.95, 'max': 0.995, 'description': 'Potential-induced degradation'},
    'k13_mechanical': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Mechanical stress losses'},
    'k14_cell_degradation': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Cell degradation'},
    'k15_interconnect': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Interconnect losses'},
    'k21_humidity': {'default': 0.99, 'min': 0.95, 'max': 0.995, 'description': 'Humidity degradation'},
    'k22_uv_exposure': {'default': 0.99, 'min': 0.95, 'max': 0.995, 'description': 'UV exposure degradation'},
    'k23_thermal_cycling': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Thermal cycling stress'},
    'k24_corrosion': {'default': 0.995, 'min': 0.99, 'max': 0.999, 'description': 'Corrosion losses'},
}

# Standard test conditions
STC = {
    'irradiance': 1000,  # W/m²
    'temperature': 25,   # °C
    'air_mass': 1.5
}

# Data file paths
MATERIALS_DB_PATH = DATA_DIR / "materials_db.json"
CELL_TYPES_PATH = DATA_DIR / "cell_types.json"
STANDARDS_PATH = DATA_DIR / "standards.json"

# Streamlit page configuration
PAGE_CONFIG = {
    'page_title': APP_NAME,
    'page_icon': '♻️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Chart styling
CHART_THEME = {
    'template': 'plotly_white',
    'color_scheme': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
}

# Environmental constants
ENVIRONMENTAL_DEFAULTS = {
    'ambient_temp': 25,  # °C
    'wind_speed': 1,     # m/s
    'albedo': 0.2,       # ground reflectance
    'altitude': 0        # meters
}

# Circularity metrics
CIRCULARITY_METRICS = {
    'recyclability_threshold': 70,  # % recyclable content
    'material_recovery_target': 95,  # % material recovery
    'carbon_reduction_target': 50,   # % carbon reduction vs baseline
    'circular_economy_score_weights': {
        'material_efficiency': 0.25,
        'recyclability': 0.25,
        'reusability': 0.25,
        'carbon_footprint': 0.25
    }
}


def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return globals().get(key, default)
