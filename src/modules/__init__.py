"""
PV Modules Package

This package contains modules for PV system component modeling including:
- Module temperature calculations
- NOCT (Nominal Operating Cell Temperature) modeling
- Thermal effects and mounting configurations
- Temperature coefficient losses

Author: PV Circularity Simulator Team
"""

from src.modules.module_temperature import (
    ModuleTemperatureModel,
    MountingType,
    TemperatureModelType,
    ModuleTechnology,
    NOCTCalculationInput,
    ModuleTemperatureInput,
    TemperatureCoefficientInput,
    ModuleSpecification,
    TemperatureCalculationResult,
    get_default_temp_coefficient,
    estimate_noct_from_mounting,
    calculate_power_at_temperature,
)

__all__ = [
    "ModuleTemperatureModel",
    "MountingType",
    "TemperatureModelType",
    "ModuleTechnology",
    "NOCTCalculationInput",
    "ModuleTemperatureInput",
    "TemperatureCoefficientInput",
    "ModuleSpecification",
    "TemperatureCalculationResult",
    "get_default_temp_coefficient",
    "estimate_noct_from_mounting",
    "calculate_power_at_temperature",
]
