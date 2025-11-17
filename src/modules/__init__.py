"""
PV Module Models

Module-level models for degradation, performance, and lifetime prediction.
"""

from .module_degradation import (
    ModuleDegradationModel,
    ModuleType,
    DegradationMode,
    LifetimeModelType,
    EnvironmentalStressFactors,
    TechnologyDegradationRates,
    WarrantySpecification,
    DegradationResult,
    MonteCarloConfig,
    create_typical_desert_environment,
    create_typical_coastal_environment,
    create_typical_continental_environment,
)

__all__ = [
    "ModuleDegradationModel",
    "ModuleType",
    "DegradationMode",
    "LifetimeModelType",
    "EnvironmentalStressFactors",
    "TechnologyDegradationRates",
    "WarrantySpecification",
    "DegradationResult",
    "MonteCarloConfig",
    "create_typical_desert_environment",
    "create_typical_coastal_environment",
    "create_typical_continental_environment",
]
