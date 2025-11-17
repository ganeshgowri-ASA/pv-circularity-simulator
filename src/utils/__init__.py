"""
Utilities for PV Circularity Simulator

This package contains utility modules for material loading,
analysis, and circularity calculations.
"""

from .material_loader import (
    # Models
    Material,
    MaterialCondition,
    EnvironmentalImpact,
    MaterialMetadata,

    # Exceptions
    MaterialLoaderError,
    DatabaseNotFoundError,
    MaterialNotFoundError,
    InvalidFilterError,
    UnitConversionError,

    # Core functions
    load_materials_database,
    get_materials_by_category,
    search_materials,
    calculate_circularity_score,
    compare_materials,
    get_material_carbon_footprint,
    estimate_eol_value,

    # Unit conversion
    convert_mass,
    convert_volume,
    convert_energy,
    convert_unit,

    # Helper functions
    get_all_material_ids,
    get_all_categories,
    get_material_by_id,
)

__all__ = [
    # Models
    'Material',
    'MaterialCondition',
    'EnvironmentalImpact',
    'MaterialMetadata',

    # Exceptions
    'MaterialLoaderError',
    'DatabaseNotFoundError',
    'MaterialNotFoundError',
    'InvalidFilterError',
    'UnitConversionError',

    # Core functions
    'load_materials_database',
    'get_materials_by_category',
    'search_materials',
    'calculate_circularity_score',
    'compare_materials',
    'get_material_carbon_footprint',
    'estimate_eol_value',

    # Unit conversion
    'convert_mass',
    'convert_volume',
    'convert_energy',
    'convert_unit',

    # Helper functions
    'get_all_material_ids',
    'get_all_categories',
    'get_material_by_id',
]
