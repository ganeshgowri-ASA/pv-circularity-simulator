"""
PV Module Engineering Package

This package provides comprehensive tools for PV module design, configuration,
and analysis.
"""

from .module_builder import (
    # Enums
    CellType,
    LayoutType,
    ConnectionType,
    ValidationLevel,

    # Models
    CellDesign,
    ModuleLayout,
    ModuleConfig,
    ModuleSpecs,
    ValidationIssue,
    ValidationReport,
    OptimalLayout,

    # Main class
    ModuleConfigBuilder,

    # Convenience functions
    create_standard_module,
)

__all__ = [
    # Enums
    'CellType',
    'LayoutType',
    'ConnectionType',
    'ValidationLevel',

    # Models
    'CellDesign',
    'ModuleLayout',
    'ModuleConfig',
    'ModuleSpecs',
    'ValidationIssue',
    'ValidationReport',
    'OptimalLayout',

    # Main class
    'ModuleConfigBuilder',

    # Convenience functions
    'create_standard_module',
]
