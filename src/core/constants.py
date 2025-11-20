"""
Physical constants and default simulation parameters for PV cell modeling.

This module defines fundamental physical constants, standard test conditions,
and default parameters used throughout the simulator.
"""

from typing import Dict, Any
import numpy as np

# Physical constants
Q = 1.602176634e-19  # Elementary charge (C)
K_B = 1.380649e-23   # Boltzmann constant (J/K)
H = 6.62607015e-34   # Planck constant (J·s)
C = 2.99792458e8     # Speed of light (m/s)
EPS_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
M_E = 9.1093837015e-31    # Electron mass (kg)

# Standard test conditions (STC)
STC_TEMPERATURE = 298.15  # K (25°C)
STC_IRRADIANCE = 1000.0   # W/m²
STC_SPECTRUM = "AM1.5G"   # Air mass 1.5 global

# Default simulation parameters
DEFAULT_MESH_POINTS = 100
DEFAULT_VOLTAGE_POINTS = 201
DEFAULT_VOLTAGE_RANGE = (-0.2, 1.2)  # V
DEFAULT_WAVELENGTH_RANGE = (300, 1200)  # nm
DEFAULT_WAVELENGTH_POINTS = 200

# Cell architecture types
CELL_ARCHITECTURES = {
    "Al-BSF": "Aluminum Back Surface Field",
    "PERC": "Passivated Emitter and Rear Cell",
    "TOPCon": "Tunnel Oxide Passivated Contact",
    "HJT": "Heterojunction Technology",
    "IBC": "Interdigitated Back Contact"
}

# Layer types
LAYER_TYPES = {
    "substrate": "Substrate (bulk semiconductor)",
    "emitter": "Emitter (n or p-type)",
    "bsf": "Back Surface Field",
    "passivation": "Passivation layer",
    "contact": "Contact layer",
    "tcO": "Transparent Conductive Oxide",
    "arc": "Anti-Reflection Coating",
    "metal": "Metal contact"
}

# Material categories
MATERIAL_CATEGORIES = {
    "semiconductor": ["Si", "GaAs", "CIGS", "CdTe", "Perovskite"],
    "tco": ["ITO", "AZO", "FTO", "SnO2"],
    "passivation": ["SiO2", "Al2O3", "SiNx", "a-Si:H(i)"],
    "contact": ["a-Si:H(n)", "a-Si:H(p)", "Poly-Si(n)", "Poly-Si(p)"],
    "metal": ["Al", "Ag", "Cu"]
}

# Default material properties template
DEFAULT_MATERIAL_PROPERTIES = {
    "bandgap": 1.12,           # eV
    "electron_affinity": 4.05,  # eV
    "dielectric_constant": 11.9,
    "electron_mobility": 1400,  # cm²/V·s
    "hole_mobility": 450,       # cm²/V·s
    "electron_lifetime": 1e-6,  # s
    "hole_lifetime": 1e-6,      # s
    "nc": 2.8e19,              # cm⁻³ (effective density of states in CB)
    "nv": 1.04e19,             # cm⁻³ (effective density of states in VB)
    "thermal_velocity": 1e7,    # cm/s
}

# Default layer properties template
DEFAULT_LAYER_PROPERTIES = {
    "thickness": 1.0,           # µm
    "doping_type": "n",         # n or p
    "doping_concentration": 1e16,  # cm⁻³
    "defect_density": 1e10,     # cm⁻³
}

# Optimization constraints
OPTIMIZATION_CONSTRAINTS = {
    "thickness": {"min": 0.001, "max": 500},      # µm
    "doping": {"min": 1e13, "max": 1e21},         # cm⁻³
    "temperature": {"min": 223, "max": 373},      # K
    "irradiance": {"min": 0, "max": 1500},        # W/m²
}

# Color scheme for layer visualization
LAYER_COLORS = {
    "Si(p+)": "#FF6B6B",        # Red
    "Si(p)": "#FFB6B6",         # Light red
    "Si(n)": "#6B9FFF",         # Light blue
    "Si(n+)": "#4169E1",        # Blue
    "a-Si:H(i)": "#FFE66D",     # Yellow
    "a-Si:H(n)": "#8ECAE6",     # Cyan
    "a-Si:H(p)": "#FFB4A2",     # Pink
    "ITO": "#95E1D3",           # Teal
    "AZO": "#A8DADC",           # Light teal
    "SiO2": "#E5E5E5",          # Light gray
    "Al2O3": "#D3D3D3",         # Gray
    "SiNx": "#B8B8FF",          # Lavender
    "Poly-Si(n+)": "#3D5A80",   # Dark blue
    "Poly-Si(p+)": "#C1121F",   # Dark red
    "Al": "#C0C0C0",            # Silver
    "Ag": "#E8E8E8",            # Bright silver
    "default": "#CCCCCC",       # Default gray
}

# Loss mechanisms for waterfall chart
LOSS_MECHANISMS = [
    "Thermalization",
    "Transmission",
    "Reflection",
    "Recombination (bulk)",
    "Recombination (surface)",
    "Contact resistance",
    "Shading",
    "Grid resistance",
    "Fill factor",
]

# Standard efficiency metrics
EFFICIENCY_METRICS = [
    "Voc",   # Open-circuit voltage (V)
    "Jsc",   # Short-circuit current density (mA/cm²)
    "FF",    # Fill factor (%)
    "Eff",   # Efficiency (%)
    "Vmpp",  # Voltage at maximum power point (V)
    "Jmpp",  # Current density at maximum power point (mA/cm²)
    "Pmpp",  # Power at maximum power point (mW/cm²)
]

def get_thermal_voltage(temperature: float = STC_TEMPERATURE) -> float:
    """
    Calculate thermal voltage Vt = kT/q.

    Args:
        temperature: Temperature in Kelvin

    Returns:
        Thermal voltage in Volts
    """
    return K_B * temperature / Q


def get_intrinsic_carrier_concentration(
    temperature: float = STC_TEMPERATURE,
    bandgap: float = 1.12
) -> float:
    """
    Calculate intrinsic carrier concentration for silicon.

    Args:
        temperature: Temperature in Kelvin
        bandgap: Bandgap energy in eV

    Returns:
        Intrinsic carrier concentration in cm⁻³
    """
    # Simplified model for Si
    ni_300k = 1.0e10  # cm⁻³ at 300K
    return ni_300k * (temperature / 300) ** 1.5 * np.exp(
        -bandgap * Q / (2 * K_B) * (1/temperature - 1/300)
    )
