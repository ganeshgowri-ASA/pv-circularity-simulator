"""Core module for PV Circularity Simulator."""

from .data_models import (
    # Enums
    CellTechnology,
    MountingType,
    MaterialType,
    # Material models
    Material,
    MaterialProperties,
    CircularityMetrics,
    # Cell models
    Cell,
    TemperatureCoefficients,
    # Module models
    Module,
    CuttingPattern,
    ModuleLayout,
    # System models
    PVSystem,
    Location,
    # Performance & Financial models
    PerformanceData,
    FinancialModel,
    # Helper functions
    create_default_monocrystalline_cell,
    create_example_silicon_material,
)

from .session_manager import (
    SessionManager,
    SessionState,
    ProjectMetadata,
    ModuleCompletionStatus,
    SimulationModule,
    ActivityEntry,
)

__all__ = [
    # Enums
    "CellTechnology",
    "MountingType",
    "MaterialType",
    "SimulationModule",
    # Material models
    "Material",
    "MaterialProperties",
    "CircularityMetrics",
    # Cell models
    "Cell",
    "TemperatureCoefficients",
    # Module models
    "Module",
    "CuttingPattern",
    "ModuleLayout",
    # System models
    "PVSystem",
    "Location",
    # Performance & Financial models
    "PerformanceData",
    "FinancialModel",
    # Session management
    "SessionManager",
    "SessionState",
    "ProjectMetadata",
    "ModuleCompletionStatus",
    "ActivityEntry",
    # Helper functions
    "create_default_monocrystalline_cell",
    "create_example_silicon_material",
]
