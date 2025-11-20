"""
Pydantic data models for PV circularity simulator.

This module contains all the core Pydantic v2 models used throughout the
PV circularity simulator, including models for materials, cells, modules,
systems, performance metrics, and financial analysis.

All models include:
- Comprehensive validation using Pydantic v2 validators
- Full serialization/deserialization support
- Type hints for all fields
- Detailed docstrings for production use
- Custom validators for domain-specific constraints
"""

from pv_circularity_simulator.models.cells import (
    CellArchitecture,
    CellDesign,
    CellElectricalCharacteristics,
    CellGeometry,
    CellModel,
    CellType,
)
from pv_circularity_simulator.models.core import BaseModel, TimestampedModel, UUIDModel
from pv_circularity_simulator.models.financial import (
    CapitalCost,
    CircularityMetrics,
    EndOfLifeCost,
    EndOfLifeScenario,
    FinancialAnalysis,
    FinancialModel,
    MaterialRecoveryData,
    OperatingCost,
)
from pv_circularity_simulator.models.materials import (
    ContactMaterial,
    MaterialModel,
    MaterialProperties,
    MaterialType,
    PassivationMaterial,
    SiliconMaterial,
)
from pv_circularity_simulator.models.modules import (
    ElectricalParameters,
    MechanicalProperties,
    ModuleConfiguration,
    ModuleModel,
    ThermalProperties,
)
from pv_circularity_simulator.models.performance import (
    DegradationModel,
    LossAnalysis,
    PerformanceMetrics,
    PerformanceModel,
    TemperatureCoefficients,
)
from pv_circularity_simulator.models.systems import (
    ElectricalProtection,
    InverterConfiguration,
    LocationCoordinates,
    MountingStructure,
    Orientation,
    SystemConfiguration,
    SystemModel,
)

__all__ = [
    # Core base models
    "BaseModel",
    "TimestampedModel",
    "UUIDModel",
    # Material models
    "MaterialModel",
    "MaterialType",
    "MaterialProperties",
    "SiliconMaterial",
    "PassivationMaterial",
    "ContactMaterial",
    # Cell models
    "CellModel",
    "CellType",
    "CellArchitecture",
    "CellGeometry",
    "CellElectricalCharacteristics",
    "CellDesign",
    # Module models
    "ModuleModel",
    "ModuleConfiguration",
    "ElectricalParameters",
    "MechanicalProperties",
    "ThermalProperties",
    # System models
    "SystemModel",
    "SystemConfiguration",
    "LocationCoordinates",
    "Orientation",
    "InverterConfiguration",
    "MountingStructure",
    "ElectricalProtection",
    # Performance models
    "PerformanceModel",
    "PerformanceMetrics",
    "TemperatureCoefficients",
    "DegradationModel",
    "LossAnalysis",
    # Financial models
    "FinancialModel",
    "CapitalCost",
    "OperatingCost",
    "FinancialAnalysis",
    "EndOfLifeCost",
    "EndOfLifeScenario",
    "MaterialRecoveryData",
    "CircularityMetrics",
]
