"""
Material Properties & Circularity Calculator

This module provides comprehensive functionality for loading, searching, and analyzing
material properties for PV circularity simulations, including circularity scoring,
carbon footprint calculations, and end-of-life value estimations.

Author: PV Circularity Simulator
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Default path to materials database
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "materials_db.json"

# Condition mapping for EOL value estimation
class MaterialCondition(str, Enum):
    """Enumeration of material condition states"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DEGRADED = "degraded"


# Unit conversion constants
UNIT_CONVERSIONS = {
    "mass": {
        "kg_to_g": 1000,
        "kg_to_mg": 1_000_000,
        "kg_to_ton": 0.001,
        "g_to_kg": 0.001,
        "ton_to_kg": 1000,
        "lb_to_kg": 0.453592,
        "kg_to_lb": 2.20462
    },
    "volume": {
        "m3_to_l": 1000,
        "m3_to_cm3": 1_000_000,
        "l_to_m3": 0.001,
        "gal_to_l": 3.78541,
        "l_to_gal": 0.264172
    },
    "energy": {
        "kwh_to_j": 3_600_000,
        "kwh_to_mj": 3.6,
        "j_to_kwh": 2.77778e-7,
        "mj_to_kwh": 0.277778
    }
}


# ============================================================================
# Exceptions
# ============================================================================

class MaterialLoaderError(Exception):
    """Base exception for material loader errors"""
    pass


class DatabaseNotFoundError(MaterialLoaderError):
    """Raised when materials database file is not found"""
    pass


class MaterialNotFoundError(MaterialLoaderError):
    """Raised when a specific material is not found"""
    pass


class InvalidFilterError(MaterialLoaderError):
    """Raised when invalid filter criteria are provided"""
    pass


class UnitConversionError(MaterialLoaderError):
    """Raised when unit conversion fails"""
    pass


# ============================================================================
# Pydantic Models
# ============================================================================

class EnvironmentalImpact(BaseModel):
    """Environmental impact metrics"""
    air_pollution_score: int = Field(ge=0, le=10, description="Air pollution score (0-10)")
    water_pollution_score: int = Field(ge=0, le=10, description="Water pollution score (0-10)")
    soil_pollution_score: int = Field(ge=0, le=10, description="Soil pollution score (0-10)")


class MaterialMetadata(BaseModel):
    """Material metadata information"""
    last_updated: str
    data_source: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class Material(BaseModel):
    """
    Comprehensive material model with validation

    This model represents a material used in PV modules with all relevant
    properties for circularity analysis, carbon footprint calculation,
    and end-of-life value estimation.
    """

    # Core identification
    id: str = Field(..., min_length=1, description="Unique material identifier")
    name: str = Field(..., min_length=1, description="Material name")
    category: str = Field(..., min_length=1, description="Primary category")
    sub_category: Optional[str] = Field(None, description="Sub-category")

    # Physical properties
    density_kg_m3: float = Field(gt=0, description="Density in kg/m³")
    cost_per_kg_usd: float = Field(ge=0, description="Cost per kg in USD")

    # Circularity metrics
    recyclability_rate: float = Field(ge=0.0, le=1.0, description="Recyclability rate (0-1)")
    recycled_content_rate: float = Field(ge=0.0, le=1.0, description="Recycled content rate (0-1)")
    virgin_material_rate: float = Field(ge=0.0, le=1.0, description="Virgin material rate (0-1)")
    durability_years: int = Field(gt=0, description="Expected durability in years")
    repairability_index: float = Field(ge=0.0, le=1.0, description="Repairability index (0-1)")
    reusability_potential: float = Field(ge=0.0, le=1.0, description="Reusability potential (0-1)")
    renewable_content: float = Field(ge=0.0, le=1.0, description="Renewable content (0-1)")
    biodegradable: bool = Field(default=False, description="Is biodegradable")
    compostable: bool = Field(default=False, description="Is compostable")

    # Environmental metrics
    carbon_footprint_kg_co2_per_kg: float = Field(ge=0, description="Carbon footprint in kg CO2/kg")
    recycled_carbon_footprint_kg_co2_per_kg: float = Field(ge=0, description="Recycled carbon footprint in kg CO2/kg")
    energy_intensity_kwh_per_kg: float = Field(ge=0, description="Energy intensity in kWh/kg")
    water_usage_l_per_kg: float = Field(ge=0, description="Water usage in L/kg")
    toxicity_score: int = Field(ge=0, le=10, description="Toxicity score (0-10)")
    environmental_impact: EnvironmentalImpact

    # Economic metrics
    eol_recovery_value_per_kg: float = Field(ge=0, description="End-of-life recovery value per kg")
    condition_factors: Dict[str, float] = Field(..., description="Condition multipliers")

    # Supply chain
    certifications: List[str] = Field(default_factory=list, description="Certifications")
    suppliers: List[str] = Field(default_factory=list, description="Suppliers")
    sourcing_region: Optional[str] = Field(None, description="Sourcing region")
    traceability_level: Optional[str] = Field(None, description="Traceability level")

    # Metadata
    metadata: MaterialMetadata

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow"  # Allow extra fields for future extensibility
    )

    @field_validator('virgin_material_rate')
    @classmethod
    def validate_virgin_rate(cls, v, info):
        """Validate that virgin_material_rate + recycled_content_rate ≈ 1.0"""
        if info.data.get('recycled_content_rate') is not None:
            total = v + info.data['recycled_content_rate']
            if not (0.98 <= total <= 1.02):  # Allow 2% tolerance
                raise ValueError(
                    f"Virgin material rate ({v}) + Recycled content rate "
                    f"({info.data['recycled_content_rate']}) must sum to ~1.0, got {total}"
                )
        return v

    @field_validator('condition_factors')
    @classmethod
    def validate_condition_factors(cls, v):
        """Validate that all condition factors are present and valid"""
        required_conditions = {e.value for e in MaterialCondition}
        provided_conditions = set(v.keys())

        if required_conditions != provided_conditions:
            missing = required_conditions - provided_conditions
            extra = provided_conditions - required_conditions
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing conditions: {missing}")
            if extra:
                msg_parts.append(f"Unknown conditions: {extra}")
            raise ValueError("; ".join(msg_parts))

        # Validate all factors are between 0 and 1
        for condition, factor in v.items():
            if not (0.0 <= factor <= 1.0):
                raise ValueError(
                    f"Condition factor for '{condition}' must be between 0 and 1, "
                    f"got {factor}"
                )

        return v


# ============================================================================
# Core Functions
# ============================================================================

@lru_cache(maxsize=1)
def load_materials_database(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load materials database from JSON file with caching

    This function loads the materials database and caches the result for
    improved performance on subsequent calls.

    Args:
        db_path: Path to materials database JSON file. If None, uses default path.

    Returns:
        Dictionary containing materials database

    Raises:
        DatabaseNotFoundError: If database file is not found
        MaterialLoaderError: If database loading or parsing fails

    Example:
        >>> db = load_materials_database()
        >>> len(db['materials'])
        10
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = Path(db_path)

    try:
        if not db_path.exists():
            raise DatabaseNotFoundError(
                f"Materials database not found at {db_path}. "
                f"Please ensure the file exists."
            )

        logger.info(f"Loading materials database from {db_path}")

        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate database structure
        if 'materials' not in data:
            raise MaterialLoaderError(
                "Invalid database format: 'materials' key not found"
            )

        # Validate each material against Pydantic model
        validated_materials = []
        for idx, material_data in enumerate(data['materials']):
            try:
                material = Material(**material_data)
                validated_materials.append(material.model_dump())
            except Exception as e:
                logger.warning(
                    f"Material at index {idx} (ID: {material_data.get('id', 'unknown')}) "
                    f"failed validation: {e}. Skipping."
                )

        data['materials'] = validated_materials

        logger.info(
            f"Successfully loaded {len(validated_materials)} materials from database"
        )

        return data

    except DatabaseNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise MaterialLoaderError(f"Failed to parse JSON database: {e}")
    except Exception as e:
        raise MaterialLoaderError(f"Failed to load materials database: {e}")


def get_materials_by_category(
    category: str,
    db_path: Optional[Path] = None,
    sub_category: Optional[str] = None
) -> List[Material]:
    """
    Get all materials in a specific category

    Args:
        category: Primary category to filter by
        db_path: Optional custom database path
        sub_category: Optional sub-category to filter by

    Returns:
        List of Material objects matching the category

    Raises:
        MaterialLoaderError: If database loading fails

    Example:
        >>> materials = get_materials_by_category("semiconductor")
        >>> len(materials)
        2
        >>> materials[0].name
        'Polycrystalline Silicon'
    """
    try:
        db = load_materials_database(db_path)
        materials = []

        for material_data in db['materials']:
            if material_data['category'].lower() == category.lower():
                if sub_category is None or \
                   material_data.get('sub_category', '').lower() == sub_category.lower():
                    materials.append(Material(**material_data))

        logger.info(
            f"Found {len(materials)} materials in category '{category}'"
            + (f", sub-category '{sub_category}'" if sub_category else "")
        )

        return materials

    except Exception as e:
        raise MaterialLoaderError(f"Failed to get materials by category: {e}")


def search_materials(
    filters: Dict[str, Any],
    db_path: Optional[Path] = None
) -> List[Material]:
    """
    Search materials using flexible filter criteria

    Supports filtering by any material property using comparison operators.

    Filter format:
        - Exact match: {"category": "semiconductor"}
        - Range: {"cost_per_kg_usd__gte": 10, "cost_per_kg_usd__lte": 50}
        - Contains: {"name__contains": "Silicon"}
        - In list: {"id__in": ["si_poly", "si_mono"]}

    Operators:
        - __gte: Greater than or equal
        - __lte: Less than or equal
        - __gt: Greater than
        - __lt: Less than
        - __contains: String contains (case-insensitive)
        - __in: Value in list
        - No operator: Exact match

    Args:
        filters: Dictionary of filter criteria
        db_path: Optional custom database path

    Returns:
        List of Material objects matching all filter criteria

    Raises:
        InvalidFilterError: If filter criteria are invalid
        MaterialLoaderError: If database loading fails

    Example:
        >>> # Find high-recyclability, low-cost materials
        >>> filters = {
        ...     "recyclability_rate__gte": 0.9,
        ...     "cost_per_kg_usd__lte": 10
        ... }
        >>> materials = search_materials(filters)
    """
    try:
        db = load_materials_database(db_path)
        materials = []

        for material_data in db['materials']:
            material = Material(**material_data)
            matches = True

            for filter_key, filter_value in filters.items():
                # Parse operator
                if '__' in filter_key:
                    field_name, operator = filter_key.rsplit('__', 1)
                else:
                    field_name = filter_key
                    operator = 'eq'

                # Get field value
                try:
                    field_value = getattr(material, field_name)
                except AttributeError:
                    # Try nested fields
                    if '.' in field_name:
                        parts = field_name.split('.')
                        field_value = material
                        for part in parts:
                            field_value = getattr(field_value, part)
                    else:
                        raise InvalidFilterError(
                            f"Invalid filter field: {field_name}"
                        )

                # Apply operator
                if operator == 'eq':
                    if field_value != filter_value:
                        matches = False
                        break
                elif operator == 'gte':
                    if field_value < filter_value:
                        matches = False
                        break
                elif operator == 'lte':
                    if field_value > filter_value:
                        matches = False
                        break
                elif operator == 'gt':
                    if field_value <= filter_value:
                        matches = False
                        break
                elif operator == 'lt':
                    if field_value >= filter_value:
                        matches = False
                        break
                elif operator == 'contains':
                    if isinstance(field_value, str):
                        if filter_value.lower() not in field_value.lower():
                            matches = False
                            break
                    else:
                        raise InvalidFilterError(
                            f"'contains' operator only works with string fields"
                        )
                elif operator == 'in':
                    if field_value not in filter_value:
                        matches = False
                        break
                else:
                    raise InvalidFilterError(f"Unknown operator: {operator}")

            if matches:
                materials.append(material)

        logger.info(f"Found {len(materials)} materials matching filters")
        return materials

    except InvalidFilterError:
        raise
    except Exception as e:
        raise MaterialLoaderError(f"Failed to search materials: {e}")


def calculate_circularity_score(
    material: Material,
    weights: Optional[Dict[str, float]] = None,
    db_path: Optional[Path] = None
) -> float:
    """
    Calculate circularity score for a material

    The circularity score is a weighted combination of various circularity
    metrics, normalized to a 0-100 scale.

    Default weights (can be overridden):
        - recyclability_rate: 0.25
        - recycled_content_rate: 0.20
        - durability_years: 0.15 (normalized to 0-1 scale)
        - reusability_potential: 0.15
        - repairability_index: 0.10
        - renewable_content: 0.10
        - biodegradable: 0.05 (1 if True, 0 if False)

    Args:
        material: Material object to score
        weights: Optional custom weights for scoring factors
        db_path: Optional custom database path for loading default weights

    Returns:
        Circularity score (0-100)

    Example:
        >>> material = Material(**material_data)
        >>> score = calculate_circularity_score(material)
        >>> print(f"Circularity score: {score:.2f}")
        Circularity score: 78.45
    """
    try:
        # Load default weights from database if not provided
        if weights is None:
            db = load_materials_database(db_path)
            weights = db.get('circularity_weights', {
                'recyclability_rate': 0.25,
                'recycled_content_rate': 0.20,
                'durability_years': 0.15,
                'reusability_potential': 0.15,
                'repairability_index': 0.10,
                'renewable_content': 0.10,
                'biodegradable': 0.05
            })

        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not (0.98 <= total_weight <= 1.02):
            logger.warning(
                f"Weights sum to {total_weight}, expected ~1.0. "
                f"Normalizing weights..."
            )
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate normalized scores
        scores = {
            'recyclability_rate': material.recyclability_rate,
            'recycled_content_rate': material.recycled_content_rate,
            'durability_years': min(material.durability_years / 30.0, 1.0),  # Normalize to 30 years max
            'reusability_potential': material.reusability_potential,
            'repairability_index': material.repairability_index,
            'renewable_content': material.renewable_content,
            'biodegradable': 1.0 if material.biodegradable else 0.0
        }

        # Calculate weighted score
        circularity_score = sum(
            scores.get(factor, 0.0) * weight
            for factor, weight in weights.items()
        )

        # Convert to 0-100 scale
        circularity_score *= 100

        logger.debug(
            f"Calculated circularity score for {material.name}: {circularity_score:.2f}"
        )

        return circularity_score

    except Exception as e:
        raise MaterialLoaderError(
            f"Failed to calculate circularity score for {material.name}: {e}"
        )


def compare_materials(
    material_ids: List[str],
    db_path: Optional[Path] = None,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple materials across key metrics

    Args:
        material_ids: List of material IDs to compare
        db_path: Optional custom database path
        metrics: Optional list of specific metrics to compare. If None, uses default set.

    Returns:
        DataFrame with materials as rows and metrics as columns

    Raises:
        MaterialNotFoundError: If any material ID is not found
        MaterialLoaderError: If comparison fails

    Example:
        >>> df = compare_materials(["si_poly", "si_mono", "glass_tempered"])
        >>> print(df[['name', 'circularity_score', 'carbon_footprint_kg_co2_per_kg']])
    """
    try:
        db = load_materials_database(db_path)

        # Find materials
        materials = []
        for material_id in material_ids:
            material_data = next(
                (m for m in db['materials'] if m['id'] == material_id),
                None
            )
            if material_data is None:
                raise MaterialNotFoundError(
                    f"Material with ID '{material_id}' not found in database"
                )
            materials.append(Material(**material_data))

        # Default metrics if not specified
        if metrics is None:
            metrics = [
                'id', 'name', 'category', 'cost_per_kg_usd',
                'recyclability_rate', 'recycled_content_rate',
                'carbon_footprint_kg_co2_per_kg', 'energy_intensity_kwh_per_kg',
                'durability_years', 'reusability_potential',
                'eol_recovery_value_per_kg'
            ]

        # Build comparison data
        comparison_data = []
        for material in materials:
            row = {}
            for metric in metrics:
                try:
                    value = getattr(material, metric)
                    row[metric] = value
                except AttributeError:
                    # Handle nested fields
                    if '.' in metric:
                        parts = metric.split('.')
                        value = material
                        for part in parts:
                            value = getattr(value, part)
                        row[metric] = value
                    else:
                        row[metric] = None

            # Add circularity score
            row['circularity_score'] = calculate_circularity_score(material)

            comparison_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Add circularity_score to metrics if not already there
        if 'circularity_score' not in metrics:
            # Reorder columns to put circularity_score after basic info
            basic_cols = ['id', 'name', 'category']
            other_cols = [c for c in df.columns if c not in basic_cols and c != 'circularity_score']
            df = df[basic_cols + ['circularity_score'] + other_cols]

        logger.info(f"Successfully compared {len(materials)} materials")

        return df

    except MaterialNotFoundError:
        raise
    except Exception as e:
        raise MaterialLoaderError(f"Failed to compare materials: {e}")


def get_material_carbon_footprint(
    material_id: str,
    quantity_kg: float,
    db_path: Optional[Path] = None,
    use_recycled: bool = False
) -> float:
    """
    Calculate carbon footprint for a given quantity of material

    Args:
        material_id: Material identifier
        quantity_kg: Quantity in kilograms
        db_path: Optional custom database path
        use_recycled: If True, use recycled carbon footprint instead of virgin

    Returns:
        Total carbon footprint in kg CO2

    Raises:
        MaterialNotFoundError: If material is not found
        ValueError: If quantity is negative

    Example:
        >>> # Calculate footprint for 10kg of polycrystalline silicon
        >>> footprint = get_material_carbon_footprint("si_poly", 10.0)
        >>> print(f"Carbon footprint: {footprint:.2f} kg CO2")
        Carbon footprint: 450.00 kg CO2
    """
    if quantity_kg < 0:
        raise ValueError(f"Quantity must be non-negative, got {quantity_kg}")

    try:
        db = load_materials_database(db_path)

        # Find material
        material_data = next(
            (m for m in db['materials'] if m['id'] == material_id),
            None
        )
        if material_data is None:
            raise MaterialNotFoundError(
                f"Material with ID '{material_id}' not found in database"
            )

        material = Material(**material_data)

        # Select appropriate carbon footprint
        if use_recycled:
            footprint_per_kg = material.recycled_carbon_footprint_kg_co2_per_kg
            logger.debug(f"Using recycled carbon footprint for {material.name}")
        else:
            footprint_per_kg = material.carbon_footprint_kg_co2_per_kg
            logger.debug(f"Using virgin carbon footprint for {material.name}")

        # Calculate total footprint
        total_footprint = footprint_per_kg * quantity_kg

        logger.info(
            f"Carbon footprint for {quantity_kg} kg of {material.name}: "
            f"{total_footprint:.2f} kg CO2"
        )

        return total_footprint

    except MaterialNotFoundError:
        raise
    except Exception as e:
        raise MaterialLoaderError(
            f"Failed to calculate carbon footprint for material '{material_id}': {e}"
        )


def estimate_eol_value(
    material_id: str,
    quantity_kg: float,
    condition: Union[str, MaterialCondition],
    db_path: Optional[Path] = None
) -> float:
    """
    Estimate end-of-life recovery value for material

    The EOL value is calculated based on the base recovery value per kg,
    adjusted by the material condition factor.

    Args:
        material_id: Material identifier
        quantity_kg: Quantity in kilograms
        condition: Material condition (excellent, good, fair, poor, degraded)
        db_path: Optional custom database path

    Returns:
        Estimated end-of-life recovery value in USD

    Raises:
        MaterialNotFoundError: If material is not found
        ValueError: If quantity is negative or condition is invalid

    Example:
        >>> # Estimate EOL value for 10kg of aluminum in good condition
        >>> value = estimate_eol_value("al_frame", 10.0, "good")
        >>> print(f"EOL recovery value: ${value:.2f}")
        EOL recovery value: $17.02
    """
    if quantity_kg < 0:
        raise ValueError(f"Quantity must be non-negative, got {quantity_kg}")

    # Normalize condition
    if isinstance(condition, str):
        try:
            condition = MaterialCondition(condition.lower())
        except ValueError:
            valid_conditions = [c.value for c in MaterialCondition]
            raise ValueError(
                f"Invalid condition '{condition}'. "
                f"Valid conditions: {valid_conditions}"
            )

    try:
        db = load_materials_database(db_path)

        # Find material
        material_data = next(
            (m for m in db['materials'] if m['id'] == material_id),
            None
        )
        if material_data is None:
            raise MaterialNotFoundError(
                f"Material with ID '{material_id}' not found in database"
            )

        material = Material(**material_data)

        # Get condition factor
        condition_factor = material.condition_factors.get(condition.value)
        if condition_factor is None:
            raise ValueError(
                f"Condition factor for '{condition.value}' not found for material '{material_id}'"
            )

        # Calculate EOL value
        base_value = material.eol_recovery_value_per_kg * quantity_kg
        adjusted_value = base_value * condition_factor

        logger.info(
            f"EOL value for {quantity_kg} kg of {material.name} "
            f"in {condition.value} condition: ${adjusted_value:.2f} "
            f"(base: ${base_value:.2f}, factor: {condition_factor})"
        )

        return adjusted_value

    except (MaterialNotFoundError, ValueError):
        raise
    except Exception as e:
        raise MaterialLoaderError(
            f"Failed to estimate EOL value for material '{material_id}': {e}"
        )


# ============================================================================
# Unit Conversion Utilities
# ============================================================================

def convert_mass(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert mass between different units

    Supported units: kg, g, mg, ton, lb

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value

    Raises:
        UnitConversionError: If conversion is not supported

    Example:
        >>> convert_mass(1000, "g", "kg")
        1.0
        >>> convert_mass(2.5, "kg", "lb")
        5.51155
    """
    try:
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit == to_unit:
            return value

        # Convert to kg first (base unit)
        conversion_key = f"{from_unit}_to_kg"
        if conversion_key in UNIT_CONVERSIONS['mass']:
            value_kg = value * UNIT_CONVERSIONS['mass'][conversion_key]
        elif from_unit == 'kg':
            value_kg = value
        else:
            raise UnitConversionError(
                f"Unsupported mass unit: {from_unit}"
            )

        # Convert from kg to target unit
        conversion_key = f"kg_to_{to_unit}"
        if conversion_key in UNIT_CONVERSIONS['mass']:
            result = value_kg * UNIT_CONVERSIONS['mass'][conversion_key]
        elif to_unit == 'kg':
            result = value_kg
        else:
            raise UnitConversionError(
                f"Unsupported mass unit: {to_unit}"
            )

        logger.debug(f"Converted {value} {from_unit} to {result} {to_unit}")
        return result

    except UnitConversionError:
        raise
    except Exception as e:
        raise UnitConversionError(
            f"Failed to convert {value} {from_unit} to {to_unit}: {e}"
        )


def convert_volume(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert volume between different units

    Supported units: m3, l, cm3, gal

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value

    Raises:
        UnitConversionError: If conversion is not supported

    Example:
        >>> convert_volume(1, "m3", "l")
        1000.0
        >>> convert_volume(5, "gal", "l")
        18.9271
    """
    try:
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit == to_unit:
            return value

        # Convert to m3 first (base unit)
        conversion_key = f"{from_unit}_to_m3"
        if conversion_key in UNIT_CONVERSIONS['volume']:
            value_m3 = value * UNIT_CONVERSIONS['volume'][conversion_key]
        elif from_unit == 'm3':
            value_m3 = value
        else:
            raise UnitConversionError(
                f"Unsupported volume unit: {from_unit}"
            )

        # Convert from m3 to target unit
        conversion_key = f"m3_to_{to_unit}"
        if conversion_key in UNIT_CONVERSIONS['volume']:
            result = value_m3 * UNIT_CONVERSIONS['volume'][conversion_key]
        elif to_unit == 'm3':
            result = value_m3
        else:
            raise UnitConversionError(
                f"Unsupported volume unit: {to_unit}"
            )

        logger.debug(f"Converted {value} {from_unit} to {result} {to_unit}")
        return result

    except UnitConversionError:
        raise
    except Exception as e:
        raise UnitConversionError(
            f"Failed to convert {value} {from_unit} to {to_unit}: {e}"
        )


def convert_energy(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert energy between different units

    Supported units: kwh, j, mj

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value

    Raises:
        UnitConversionError: If conversion is not supported

    Example:
        >>> convert_energy(1, "kwh", "mj")
        3.6
        >>> convert_energy(3600000, "j", "kwh")
        1.0
    """
    try:
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit == to_unit:
            return value

        # Convert to kWh first (base unit)
        conversion_key = f"{from_unit}_to_kwh"
        if conversion_key in UNIT_CONVERSIONS['energy']:
            value_kwh = value * UNIT_CONVERSIONS['energy'][conversion_key]
        elif from_unit == 'kwh':
            value_kwh = value
        else:
            raise UnitConversionError(
                f"Unsupported energy unit: {from_unit}"
            )

        # Convert from kWh to target unit
        conversion_key = f"kwh_to_{to_unit}"
        if conversion_key in UNIT_CONVERSIONS['energy']:
            result = value_kwh * UNIT_CONVERSIONS['energy'][conversion_key]
        elif to_unit == 'kwh':
            result = value_kwh
        else:
            raise UnitConversionError(
                f"Unsupported energy unit: {to_unit}"
            )

        logger.debug(f"Converted {value} {from_unit} to {result} {to_unit}")
        return result

    except UnitConversionError:
        raise
    except Exception as e:
        raise UnitConversionError(
            f"Failed to convert {value} {from_unit} to {to_unit}: {e}"
        )


def convert_unit(
    value: float,
    from_unit: str,
    to_unit: str,
    unit_type: Optional[str] = None
) -> float:
    """
    Generic unit conversion function

    Automatically detects unit type or uses provided type.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        unit_type: Optional unit type (mass, volume, energy). Auto-detected if None.

    Returns:
        Converted value

    Raises:
        UnitConversionError: If conversion fails or unit type cannot be determined

    Example:
        >>> convert_unit(1000, "g", "kg")
        1.0
        >>> convert_unit(1, "m3", "l", unit_type="volume")
        1000.0
    """
    try:
        # Auto-detect unit type if not provided
        if unit_type is None:
            from_unit_lower = from_unit.lower()

            # Check mass units
            if any(from_unit_lower == u for u in ['kg', 'g', 'mg', 'ton', 'lb']):
                unit_type = 'mass'
            # Check volume units
            elif any(from_unit_lower == u for u in ['m3', 'l', 'cm3', 'gal']):
                unit_type = 'volume'
            # Check energy units
            elif any(from_unit_lower == u for u in ['kwh', 'j', 'mj']):
                unit_type = 'energy'
            else:
                raise UnitConversionError(
                    f"Cannot auto-detect unit type for '{from_unit}'. "
                    f"Please specify unit_type parameter."
                )

        # Route to appropriate conversion function
        if unit_type == 'mass':
            return convert_mass(value, from_unit, to_unit)
        elif unit_type == 'volume':
            return convert_volume(value, from_unit, to_unit)
        elif unit_type == 'energy':
            return convert_energy(value, from_unit, to_unit)
        else:
            raise UnitConversionError(
                f"Unknown unit type: {unit_type}. "
                f"Supported types: mass, volume, energy"
            )

    except UnitConversionError:
        raise
    except Exception as e:
        raise UnitConversionError(
            f"Failed to convert {value} {from_unit} to {to_unit}: {e}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_material_ids(db_path: Optional[Path] = None) -> List[str]:
    """
    Get list of all material IDs in the database

    Args:
        db_path: Optional custom database path

    Returns:
        List of material IDs
    """
    db = load_materials_database(db_path)
    return [m['id'] for m in db['materials']]


def get_all_categories(db_path: Optional[Path] = None) -> List[str]:
    """
    Get list of all unique categories in the database

    Args:
        db_path: Optional custom database path

    Returns:
        List of unique categories
    """
    db = load_materials_database(db_path)
    categories = list(set(m['category'] for m in db['materials']))
    return sorted(categories)


def get_material_by_id(
    material_id: str,
    db_path: Optional[Path] = None
) -> Material:
    """
    Get a single material by ID

    Args:
        material_id: Material identifier
        db_path: Optional custom database path

    Returns:
        Material object

    Raises:
        MaterialNotFoundError: If material is not found
    """
    db = load_materials_database(db_path)
    material_data = next(
        (m for m in db['materials'] if m['id'] == material_id),
        None
    )
    if material_data is None:
        raise MaterialNotFoundError(
            f"Material with ID '{material_id}' not found in database"
        )
    return Material(**material_data)


# ============================================================================
# Module Info
# ============================================================================

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


if __name__ == '__main__':
    # Demo usage
    print("=" * 80)
    print("Material Properties & Circularity Calculator Demo")
    print("=" * 80)

    try:
        # Load database
        print("\n1. Loading materials database...")
        db = load_materials_database()
        print(f"   ✓ Loaded {len(db['materials'])} materials")

        # Get materials by category
        print("\n2. Getting semiconductor materials...")
        semiconductors = get_materials_by_category("semiconductor")
        for mat in semiconductors:
            print(f"   - {mat.name} (ID: {mat.id})")

        # Search materials
        print("\n3. Searching for high-recyclability materials (>0.9)...")
        high_recyclability = search_materials({
            "recyclability_rate__gte": 0.9,
            "cost_per_kg_usd__lte": 10
        })
        for mat in high_recyclability:
            print(f"   - {mat.name}: {mat.recyclability_rate:.1%} recyclable, ${mat.cost_per_kg_usd:.2f}/kg")

        # Calculate circularity score
        print("\n4. Calculating circularity scores...")
        for material_id in ["si_poly", "al_frame", "eva"]:
            material = get_material_by_id(material_id)
            score = calculate_circularity_score(material)
            print(f"   - {material.name}: {score:.2f}/100")

        # Compare materials
        print("\n5. Comparing materials...")
        comparison = compare_materials(["si_poly", "si_mono", "glass_tempered"])
        print(comparison[['name', 'circularity_score', 'carbon_footprint_kg_co2_per_kg', 'cost_per_kg_usd']].to_string(index=False))

        # Carbon footprint
        print("\n6. Calculating carbon footprint...")
        footprint = get_material_carbon_footprint("si_poly", 10.0)
        print(f"   10 kg of Polycrystalline Silicon: {footprint:.2f} kg CO2")

        # EOL value
        print("\n7. Estimating end-of-life value...")
        eol_value = estimate_eol_value("al_frame", 5.0, "good")
        print(f"   5 kg of Aluminum Frame in good condition: ${eol_value:.2f}")

        # Unit conversion
        print("\n8. Unit conversions...")
        print(f"   1000 g = {convert_mass(1000, 'g', 'kg')} kg")
        print(f"   1 m³ = {convert_volume(1, 'm3', 'l')} L")
        print(f"   1 kWh = {convert_energy(1, 'kwh', 'mj')} MJ")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
