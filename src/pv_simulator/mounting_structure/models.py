"""Pydantic models for mounting structure design and validation."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class MountingType(str, Enum):
    """Types of PV mounting systems."""
    GROUND_FIXED_TILT = "ground_fixed_tilt"
    GROUND_SINGLE_AXIS = "ground_single_axis"
    GROUND_DUAL_AXIS = "ground_dual_axis"
    ROOFTOP_FLAT = "rooftop_flat"
    ROOFTOP_PITCHED = "rooftop_pitched"
    CARPORT = "carport"
    CANOPY = "canopy"
    FLOATING = "floating"
    AGRIVOLTAIC = "agrivoltaic"
    BIPV_FACADE = "bipv_facade"
    BIPV_SKYLIGHT = "bipv_skylight"
    BIPV_CURTAIN_WALL = "bipv_curtain_wall"


class ModuleOrientation(str, Enum):
    """Module orientation on racking."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class RackingConfiguration(str, Enum):
    """Racking configuration types."""
    ONE_PORTRAIT = "1P"
    TWO_PORTRAIT = "2P"
    THREE_PORTRAIT = "3P"
    FOUR_PORTRAIT = "4P"
    ONE_LANDSCAPE = "1L"
    TWO_LANDSCAPE = "2L"
    THREE_LANDSCAPE = "3L"


class FoundationType(str, Enum):
    """Foundation types for mounting structures."""
    DRIVEN_PILE = "driven_pile"
    HELICAL_PILE = "helical_pile"
    GROUND_SCREW = "ground_screw"
    BALLASTED = "ballasted"
    CONCRETE_PAD = "concrete_pad"
    SPREAD_FOOTING = "spread_footing"
    ROOF_ATTACHMENT = "roof_attachment"
    PONTOON = "pontoon"


class SoilType(str, Enum):
    """Soil classification types."""
    CLAY = "clay"
    SAND = "sand"
    SILT = "silt"
    GRAVEL = "gravel"
    ROCK = "rock"
    MIXED = "mixed"


class ExposureCategory(str, Enum):
    """ASCE 7 exposure categories for wind load."""
    B = "B"  # Urban/suburban
    C = "C"  # Open terrain
    D = "D"  # Flat, unobstructed coastal areas


class SeismicDesignCategory(str, Enum):
    """Seismic design categories."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class MaterialType(str, Enum):
    """Structural material types."""
    STEEL_GALVANIZED = "steel_galvanized"
    ALUMINUM = "aluminum"
    STAINLESS_STEEL = "stainless_steel"
    CONCRETE = "concrete"
    TIMBER = "timber"
    HDPE = "hdpe"
    COMPOSITE = "composite"


class SiteParameters(BaseModel):
    """Site-specific parameters for structural design."""
    model_config = ConfigDict(extra='forbid')

    latitude: float = Field(..., ge=-90, le=90, description="Site latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Site longitude in degrees")
    elevation: float = Field(..., ge=0, description="Site elevation above sea level (m)")
    wind_speed: float = Field(..., gt=0, description="Design wind speed (m/s)")
    exposure_category: ExposureCategory = Field(..., description="ASCE 7 exposure category")
    ground_snow_load: float = Field(..., ge=0, description="Ground snow load (kN/m²)")
    seismic_category: SeismicDesignCategory = Field(default=SeismicDesignCategory.A, description="Seismic design category")
    soil_type: SoilType = Field(..., description="Primary soil type")
    bearing_capacity: Optional[float] = Field(None, ge=0, description="Allowable soil bearing capacity (kN/m²)")
    frost_depth: float = Field(default=0.0, ge=0, description="Frost depth (m)")
    corrosion_environment: str = Field(default="normal", description="Corrosion environment classification")


class ModuleDimensions(BaseModel):
    """PV module physical dimensions."""
    model_config = ConfigDict(extra='forbid')

    length: float = Field(..., gt=0, description="Module length (m)")
    width: float = Field(..., gt=0, description="Module width (m)")
    thickness: float = Field(..., gt=0, description="Module thickness (m)")
    weight: float = Field(..., gt=0, description="Module weight (kg)")
    frame_width: float = Field(default=0.035, gt=0, description="Frame width (m)")
    glass_thickness: float = Field(default=0.0032, gt=0, description="Front glass thickness (m)")


class LoadAnalysis(BaseModel):
    """Structural load analysis results."""
    model_config = ConfigDict(extra='forbid')

    dead_load: float = Field(..., description="Dead load (kN/m²)")
    live_load: float = Field(..., description="Live load (kN/m²)")
    wind_load_uplift: float = Field(..., description="Wind uplift load (kN/m²)")
    wind_load_downward: float = Field(..., description="Wind downward load (kN/m²)")
    snow_load: float = Field(..., description="Snow load (kN/m²)")
    seismic_load: Optional[float] = Field(None, description="Seismic load (kN)")
    total_load_combination: float = Field(..., description="Critical load combination (kN/m²)")
    safety_factor: float = Field(..., gt=1.0, description="Applied safety factor")


class FoundationDesign(BaseModel):
    """Foundation design specifications."""
    model_config = ConfigDict(extra='forbid')

    foundation_type: FoundationType = Field(..., description="Type of foundation")
    depth: float = Field(..., gt=0, description="Foundation depth (m)")
    diameter: Optional[float] = Field(None, gt=0, description="Pile/pier diameter (m)")
    length: Optional[float] = Field(None, gt=0, description="Foundation length (m)")
    width: Optional[float] = Field(None, gt=0, description="Foundation width (m)")
    capacity: float = Field(..., gt=0, description="Foundation capacity (kN)")
    spacing: float = Field(..., gt=0, description="Foundation spacing (m)")
    quantity: int = Field(..., gt=0, description="Number of foundations required")
    material: MaterialType = Field(..., description="Foundation material")
    embedment_depth: float = Field(..., ge=0, description="Embedment depth below grade (m)")
    concrete_volume: Optional[float] = Field(None, ge=0, description="Concrete volume per foundation (m³)")
    reinforcement: Optional[str] = Field(None, description="Reinforcement specification")


class StructuralMember(BaseModel):
    """Structural member specification."""
    model_config = ConfigDict(extra='forbid')

    member_type: str = Field(..., description="Member type (beam, column, purlin, rail)")
    material: MaterialType = Field(..., description="Member material")
    profile: str = Field(..., description="Profile designation (W8x24, HSS4x4x1/4, etc.)")
    length: float = Field(..., gt=0, description="Member length (m)")
    spacing: float = Field(..., gt=0, description="Member spacing (m)")
    quantity: int = Field(..., gt=0, description="Number of members")
    capacity: float = Field(..., gt=0, description="Design capacity (kN or kN-m)")
    utilization: float = Field(..., ge=0, le=1.5, description="Utilization ratio (demand/capacity)")


class BillOfMaterials(BaseModel):
    """Bill of materials for mounting structure."""
    model_config = ConfigDict(extra='forbid')

    item_number: str = Field(..., description="Item number/SKU")
    description: str = Field(..., description="Item description")
    material: MaterialType = Field(..., description="Material type")
    specification: str = Field(..., description="Detailed specification")
    quantity: float = Field(..., gt=0, description="Quantity required")
    unit: str = Field(..., description="Unit of measure")
    unit_weight: Optional[float] = Field(None, ge=0, description="Unit weight (kg)")
    total_weight: Optional[float] = Field(None, ge=0, description="Total weight (kg)")
    unit_cost: Optional[float] = Field(None, ge=0, description="Unit cost ($)")
    total_cost: Optional[float] = Field(None, ge=0, description="Total cost ($)")
    supplier: Optional[str] = Field(None, description="Supplier name")
    notes: Optional[str] = Field(None, description="Additional notes")


class StructuralAnalysisResult(BaseModel):
    """Complete structural analysis results."""
    model_config = ConfigDict(extra='forbid')

    mounting_type: MountingType = Field(..., description="Mounting system type")
    load_analysis: LoadAnalysis = Field(..., description="Load analysis results")
    foundation_design: FoundationDesign = Field(..., description="Foundation design")
    structural_members: List[StructuralMember] = Field(..., description="Structural members")
    bill_of_materials: List[BillOfMaterials] = Field(..., description="Complete BOM")
    max_deflection: float = Field(..., description="Maximum deflection (m)")
    deflection_limit: float = Field(..., description="Allowable deflection limit (m)")
    connection_details: Dict[str, Any] = Field(default_factory=dict, description="Connection design details")
    compliance_notes: List[str] = Field(default_factory=list, description="Code compliance notes")
    total_steel_weight: float = Field(..., ge=0, description="Total steel weight (kg)")
    total_cost_estimate: Optional[float] = Field(None, ge=0, description="Total cost estimate ($)")


class MountingConfig(BaseModel):
    """Base mounting configuration."""
    model_config = ConfigDict(extra='forbid')

    mounting_type: MountingType = Field(..., description="Type of mounting system")
    site_parameters: SiteParameters = Field(..., description="Site-specific parameters")
    module_dimensions: ModuleDimensions = Field(..., description="PV module dimensions")
    num_modules: int = Field(..., gt=0, description="Total number of modules")
    tilt_angle: float = Field(..., ge=0, le=90, description="Module tilt angle (degrees)")
    azimuth: float = Field(default=180.0, ge=0, le=360, description="Azimuth angle (degrees, 180=south)")


class GroundMountConfig(MountingConfig):
    """Ground-mount specific configuration."""
    model_config = ConfigDict(extra='forbid')

    orientation: ModuleOrientation = Field(..., description="Module orientation")
    racking_config: RackingConfiguration = Field(..., description="Racking configuration")
    row_spacing: Optional[float] = Field(None, gt=0, description="Row-to-row spacing (m)")
    gcr: Optional[float] = Field(None, gt=0, le=1.0, description="Ground coverage ratio")
    post_spacing: float = Field(default=3.0, gt=0, description="Post spacing along row (m)")
    foundation_type: FoundationType = Field(..., description="Foundation type")
    tracker_type: Optional[str] = Field(None, description="Tracker type if applicable")
    backtracking_enabled: bool = Field(default=False, description="Backtracking enabled for single-axis")
    max_tracking_angle: Optional[float] = Field(None, ge=0, le=90, description="Maximum tracking angle (degrees)")


class RooftopMountConfig(MountingConfig):
    """Rooftop-mount specific configuration."""
    model_config = ConfigDict(extra='forbid')

    roof_type: str = Field(..., description="Roof type (flat, pitched, metal)")
    roof_pitch: float = Field(default=0.0, ge=0, le=90, description="Roof pitch (degrees)")
    roof_material: str = Field(..., description="Roof surface material")
    attachment_type: str = Field(..., description="Attachment method")
    rail_type: str = Field(default="shared", description="Rail type (shared, dedicated)")
    fire_setback: float = Field(default=1.0, ge=0, description="Fire setback distance (m)")
    wind_zone: int = Field(default=1, ge=1, le=4, description="Wind zone (1-4)")
    roof_dead_load_capacity: float = Field(..., gt=0, description="Existing roof dead load capacity (kN/m²)")
    roof_live_load_capacity: float = Field(..., gt=0, description="Existing roof live load capacity (kN/m²)")


class CarportConfig(MountingConfig):
    """Carport/canopy specific configuration."""
    model_config = ConfigDict(extra='forbid')

    carport_type: str = Field(..., description="Carport type (single_cantilever, double_cantilever, four_post)")
    span_length: float = Field(..., gt=0, description="Span length (m)")
    cantilever_length: Optional[float] = Field(None, ge=0, description="Cantilever length (m)")
    clearance_height: float = Field(default=2.5, gt=0, description="Vehicle clearance height (m)")
    column_spacing: float = Field(..., gt=0, description="Column spacing (m)")
    beam_material: MaterialType = Field(default=MaterialType.STEEL_GALVANIZED, description="Beam material")
    drainage_slope: float = Field(default=0.02, gt=0, le=0.1, description="Drainage slope (m/m)")
    ada_compliance: bool = Field(default=False, description="ADA compliance required")


class FloatingPVConfig(MountingConfig):
    """Floating PV specific configuration."""
    model_config = ConfigDict(extra='forbid')

    water_body_type: str = Field(..., description="Water body type (lake, reservoir, pond)")
    water_depth: float = Field(..., gt=0, description="Water depth (m)")
    max_wave_height: float = Field(..., ge=0, description="Maximum wave height (m)")
    water_level_variation: float = Field(..., ge=0, description="Water level variation (m)")
    pontoon_material: MaterialType = Field(default=MaterialType.HDPE, description="Pontoon material")
    pontoon_spacing: float = Field(default=1.0, gt=0, description="Pontoon spacing (m)")
    anchoring_type: str = Field(..., description="Anchoring type (mooring, pile)")
    coverage_ratio: float = Field(..., gt=0, le=0.4, description="Water surface coverage ratio")
    cooling_benefit: bool = Field(default=True, description="Account for evaporative cooling")


class AgrivoltaicConfig(MountingConfig):
    """Agrivoltaic specific configuration."""
    model_config = ConfigDict(extra='forbid')

    clearance_height: float = Field(..., gt=2.0, description="Clearance height for equipment (m)")
    crop_type: str = Field(..., description="Crop type to be grown")
    row_spacing_for_crops: float = Field(..., gt=0, description="Row spacing for crop requirements (m)")
    bifacial_modules: bool = Field(default=False, description="Using bifacial modules")
    adjustable_tilt: bool = Field(default=False, description="Seasonal tilt adjustment capability")
    irrigation_integration: bool = Field(default=False, description="Integrate with irrigation system")
    equipment_access_width: float = Field(default=4.0, gt=0, description="Equipment access width (m)")


class BIPVConfig(MountingConfig):
    """Building-integrated PV specific configuration."""
    model_config = ConfigDict(extra='forbid')

    integration_type: str = Field(..., description="Integration type (facade, skylight, curtain_wall)")
    building_height: float = Field(..., gt=0, description="Building height (m)")
    vertical_installation: bool = Field(default=False, description="Vertical installation")
    translucent_modules: bool = Field(default=False, description="Using translucent modules")
    structural_glazing: bool = Field(default=False, description="Structural glazing system")
    thermal_break: bool = Field(default=False, description="Thermal break required")
    junction_box_location: str = Field(default="back", description="Junction box location")
    conduit_routing: str = Field(..., description="Conduit routing method")
