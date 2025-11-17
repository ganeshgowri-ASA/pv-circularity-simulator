"""
Core Pydantic models for PV system components.

This module defines comprehensive data models for photovoltaic system components
including modules, inverters, mounting structures, and complete system designs.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, computed_field
import numpy as np


class ModuleTechnology(str, Enum):
    """PV module technology types."""
    MONO_SI = "mono-Si"
    POLY_SI = "poly-Si"
    THIN_FILM = "thin-film"
    PERC = "PERC"
    TOPCON = "TOPCon"
    HJT = "HJT"
    BIFACIAL = "bifacial"


class InverterType(str, Enum):
    """Inverter topology types."""
    STRING = "string"
    CENTRAL = "central"
    MICROINVERTER = "microinverter"
    OPTIMIZER = "optimizer"


class MountingType(str, Enum):
    """Mounting structure types."""
    FIXED_TILT = "fixed-tilt"
    SINGLE_AXIS_TRACKER = "single-axis-tracker"
    DUAL_AXIS_TRACKER = "dual-axis-tracker"
    ROOFTOP = "rooftop"
    GROUND_MOUNT = "ground-mount"
    CARPORT = "carport"


class OrientationType(str, Enum):
    """System orientation types."""
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class PVModule(BaseModel):
    """
    Comprehensive PV module model.

    Represents a photovoltaic module with electrical, physical, and thermal characteristics.
    """

    # Identification
    manufacturer: str = Field(..., description="Module manufacturer name")
    model: str = Field(..., description="Module model number")
    technology: ModuleTechnology = Field(..., description="Module technology type")

    # Electrical characteristics (STC)
    p_max: float = Field(..., gt=0, description="Maximum power at STC (W)")
    v_mp: float = Field(..., gt=0, description="Voltage at maximum power point (V)")
    i_mp: float = Field(..., gt=0, description="Current at maximum power point (A)")
    v_oc: float = Field(..., gt=0, description="Open circuit voltage (V)")
    i_sc: float = Field(..., gt=0, description="Short circuit current (A)")

    # Temperature coefficients (%/°C)
    temp_coeff_p_max: float = Field(default=-0.4, description="Power temperature coefficient (%/°C)")
    temp_coeff_v_oc: float = Field(default=-0.3, description="Voc temperature coefficient (%/°C)")
    temp_coeff_i_sc: float = Field(default=0.05, description="Isc temperature coefficient (%/°C)")

    # Physical dimensions
    length: float = Field(..., gt=0, description="Module length (m)")
    width: float = Field(..., gt=0, description="Module width (m)")
    thickness: float = Field(default=0.04, gt=0, description="Module thickness (m)")
    weight: float = Field(..., gt=0, description="Module weight (kg)")

    # Performance characteristics
    efficiency: float = Field(..., gt=0, le=100, description="Module efficiency (%)")
    noct: float = Field(default=45.0, description="Nominal operating cell temperature (°C)")
    max_system_voltage: float = Field(default=1500.0, description="Maximum system voltage (V)")
    series_fuse_rating: float = Field(default=20.0, description="Series fuse rating (A)")

    # Bifacial characteristics (if applicable)
    is_bifacial: bool = Field(default=False, description="Whether module is bifacial")
    bifaciality: Optional[float] = Field(default=None, ge=0, le=1, description="Bifaciality factor (0-1)")

    # Degradation
    degradation_rate: float = Field(default=0.5, description="Annual degradation rate (%/year)")

    # Additional metadata
    warranty_years: int = Field(default=25, gt=0, description="Warranty period (years)")
    certification: List[str] = Field(default_factory=lambda: ["IEC 61215", "IEC 61730"], description="Certifications")

    @computed_field
    @property
    def area(self) -> float:
        """Module area in m²."""
        return self.length * self.width

    @computed_field
    @property
    def cells_count(self) -> int:
        """Estimate number of cells based on voltage."""
        return int(self.v_oc / 0.6)  # Approximate 0.6V per cell

    @field_validator('bifaciality')
    @classmethod
    def validate_bifaciality(cls, v, info):
        """Validate bifaciality is only set for bifacial modules."""
        if v is not None and not info.data.get('is_bifacial'):
            raise ValueError("bifaciality can only be set for bifacial modules")
        return v


class Inverter(BaseModel):
    """
    Comprehensive inverter model.

    Represents a solar inverter with electrical characteristics and MPPT capabilities.
    """

    # Identification
    manufacturer: str = Field(..., description="Inverter manufacturer name")
    model: str = Field(..., description="Inverter model number")
    inverter_type: InverterType = Field(..., description="Inverter topology type")

    # Power ratings
    p_ac_rated: float = Field(..., gt=0, description="Rated AC output power (W)")
    p_dc_max: float = Field(..., gt=0, description="Maximum DC input power (W)")

    # Voltage ranges
    v_dc_min: float = Field(..., gt=0, description="Minimum DC operating voltage (V)")
    v_dc_max: float = Field(..., gt=0, description="Maximum DC operating voltage (V)")
    v_mpp_min: float = Field(..., gt=0, description="Minimum MPPT voltage (V)")
    v_mpp_max: float = Field(..., gt=0, description="Maximum MPPT voltage (V)")

    # Current limits
    i_dc_max: float = Field(..., gt=0, description="Maximum DC input current (A)")
    i_ac_max: float = Field(..., gt=0, description="Maximum AC output current (A)")

    # MPPT configuration
    num_mppt: int = Field(default=2, gt=0, description="Number of MPPT inputs")
    num_strings_per_mppt: int = Field(default=2, gt=0, description="Maximum strings per MPPT")

    # Efficiency
    max_efficiency: float = Field(default=98.0, gt=0, le=100, description="Maximum efficiency (%)")
    euro_efficiency: float = Field(default=97.5, gt=0, le=100, description="European efficiency (%)")
    cec_efficiency: float = Field(default=97.0, gt=0, le=100, description="CEC weighted efficiency (%)")

    # Physical characteristics
    weight: float = Field(..., gt=0, description="Inverter weight (kg)")
    dimensions: Tuple[float, float, float] = Field(..., description="Dimensions (H, W, D) in meters")

    # Protection
    ip_rating: str = Field(default="IP65", description="Ingress protection rating")

    # Grid characteristics
    v_ac_nominal: float = Field(default=240.0, description="Nominal AC voltage (V)")
    frequency: float = Field(default=60.0, description="Nominal frequency (Hz)")
    phases: int = Field(default=1, ge=1, le=3, description="Number of phases")

    # Additional features
    has_ground_fault_protection: bool = Field(default=True, description="Has ground fault protection")
    has_arc_fault_protection: bool = Field(default=True, description="Has arc fault protection")
    has_rapid_shutdown: bool = Field(default=True, description="Has rapid shutdown capability")

    @computed_field
    @property
    def dc_ac_ratio_max(self) -> float:
        """Maximum DC/AC ratio."""
        return self.p_dc_max / self.p_ac_rated

    @field_validator('v_mpp_max')
    @classmethod
    def validate_mppt_range(cls, v, info):
        """Validate MPPT voltage range is within DC operating range."""
        if v > info.data.get('v_dc_max', float('inf')):
            raise ValueError("v_mpp_max must be <= v_dc_max")
        return v


class MountingStructure(BaseModel):
    """
    Mounting structure configuration.

    Defines the physical mounting and orientation of PV modules.
    """

    mounting_type: MountingType = Field(..., description="Type of mounting structure")
    tilt_angle: float = Field(..., ge=0, le=90, description="Tilt angle from horizontal (degrees)")
    azimuth: float = Field(..., ge=0, lt=360, description="Azimuth angle (degrees, 0=North, 90=East)")

    # Tracking parameters (if applicable)
    is_tracking: bool = Field(default=False, description="Whether system uses tracking")
    tracking_type: Optional[str] = Field(default=None, description="Type of tracking system")
    max_tracking_angle: Optional[float] = Field(default=None, description="Maximum tracking angle (degrees)")
    backtracking_enabled: bool = Field(default=False, description="Whether backtracking is enabled")

    # Row spacing (for ground mount)
    row_spacing: Optional[float] = Field(default=None, gt=0, description="Row spacing (m)")
    gcr: Optional[float] = Field(default=None, gt=0, le=1, description="Ground coverage ratio")

    # Height parameters
    height_min: float = Field(default=0.5, gt=0, description="Minimum height above ground (m)")
    height_max: Optional[float] = Field(default=None, gt=0, description="Maximum height above ground (m)")

    # Wind and snow loading
    wind_load_rating: float = Field(default=50.0, description="Wind load rating (m/s)")
    snow_load_rating: float = Field(default=2000.0, description="Snow load rating (Pa)")

    # Material and foundation
    material: str = Field(default="aluminum", description="Primary structural material")
    foundation_type: str = Field(default="driven-pile", description="Foundation type")

    @field_validator('gcr')
    @classmethod
    def validate_gcr(cls, v, info):
        """Validate GCR is consistent with row spacing."""
        if v is not None and v > 1.0:
            raise ValueError("GCR must be <= 1.0")
        return v


class StringConfiguration(BaseModel):
    """
    String configuration for a PV array.

    Defines how modules are connected in series and parallel.
    """

    modules_per_string: int = Field(..., gt=0, description="Number of modules in series per string")
    num_strings: int = Field(..., gt=0, description="Number of parallel strings")
    module: PVModule = Field(..., description="Module used in this string")

    @computed_field
    @property
    def total_modules(self) -> int:
        """Total number of modules in configuration."""
        return self.modules_per_string * self.num_strings

    @computed_field
    @property
    def string_v_oc(self) -> float:
        """String open circuit voltage (V)."""
        return self.module.v_oc * self.modules_per_string

    @computed_field
    @property
    def string_v_mp(self) -> float:
        """String maximum power voltage (V)."""
        return self.module.v_mp * self.modules_per_string

    @computed_field
    @property
    def string_i_sc(self) -> float:
        """String short circuit current (A)."""
        return self.module.i_sc

    @computed_field
    @property
    def total_power(self) -> float:
        """Total configuration power at STC (W)."""
        return self.module.p_max * self.total_modules


class SiteLocation(BaseModel):
    """
    Site location and environmental parameters.
    """

    # Geographic location
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (degrees)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (degrees)")
    elevation: float = Field(default=0.0, description="Elevation above sea level (m)")
    timezone: str = Field(default="UTC", description="Timezone string")

    # Site identification
    name: str = Field(default="", description="Site name")
    address: str = Field(default="", description="Site address")

    # Environmental parameters
    albedo: float = Field(default=0.2, ge=0, le=1, description="Ground albedo (reflectance)")
    average_wind_speed: float = Field(default=1.0, ge=0, description="Average wind speed (m/s)")

    # Climate data
    annual_ghi: Optional[float] = Field(default=None, description="Annual GHI (kWh/m²/year)")
    annual_dni: Optional[float] = Field(default=None, description="Annual DNI (kWh/m²/year)")
    avg_ambient_temp: Optional[float] = Field(default=None, description="Average ambient temperature (°C)")


class SystemDesign(BaseModel):
    """
    Complete PV system design.

    Integrates all components into a complete system design with validation.
    """

    # Design identification
    design_id: str = Field(..., description="Unique design identifier")
    design_name: str = Field(..., description="Design name")
    created_at: datetime = Field(default_factory=datetime.now, description="Design creation timestamp")
    version: str = Field(default="1.0", description="Design version")

    # Site and location
    site: SiteLocation = Field(..., description="Site location and parameters")

    # System components
    modules: List[StringConfiguration] = Field(..., min_length=1, description="Module string configurations")
    inverters: List[Inverter] = Field(..., min_length=1, description="Inverter configurations")
    mounting: MountingStructure = Field(..., description="Mounting structure configuration")

    # System configuration
    dc_ac_ratio: float = Field(..., gt=0, description="DC/AC ratio (oversizing)")
    num_arrays: int = Field(default=1, gt=0, description="Number of separate arrays")

    # Design constraints
    max_voltage: float = Field(default=1500.0, description="Maximum system voltage (V)")
    min_voltage: float = Field(default=300.0, description="Minimum operating voltage (V)")

    # Performance parameters
    system_losses: Dict[str, float] = Field(
        default_factory=lambda: {
            "soiling": 2.0,
            "shading": 3.0,
            "snow": 0.0,
            "mismatch": 2.0,
            "wiring": 2.0,
            "connections": 0.5,
            "lid": 1.5,  # Light-induced degradation
            "nameplate": 1.0,
            "availability": 1.0
        },
        description="System loss percentages"
    )

    # Design status
    is_validated: bool = Field(default=False, description="Whether design has been validated")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warning messages")

    # Design metadata
    designer: str = Field(default="", description="Designer name")
    notes: str = Field(default="", description="Design notes")
    tags: List[str] = Field(default_factory=list, description="Design tags")

    @computed_field
    @property
    def total_dc_power(self) -> float:
        """Total DC power at STC (kW)."""
        return sum(config.total_power for config in self.modules) / 1000.0

    @computed_field
    @property
    def total_ac_power(self) -> float:
        """Total AC power (kW)."""
        return sum(inv.p_ac_rated for inv in self.inverters) / 1000.0

    @computed_field
    @property
    def total_modules_count(self) -> int:
        """Total number of modules in system."""
        return sum(config.total_modules for config in self.modules)

    @computed_field
    @property
    def total_system_losses(self) -> float:
        """Total system losses (%)."""
        # Calculate compound losses: 1 - product of (1 - loss/100) for each loss
        losses_decimal = [loss / 100.0 for loss in self.system_losses.values()]
        total_loss = 1.0 - np.prod([1.0 - loss for loss in losses_decimal])
        return total_loss * 100.0


class DesignComparison(BaseModel):
    """
    Model for comparing multiple system designs.
    """

    designs: List[SystemDesign] = Field(..., min_length=2, description="Designs to compare")
    comparison_metrics: List[str] = Field(
        default_factory=lambda: [
            "total_dc_power",
            "total_ac_power",
            "dc_ac_ratio",
            "total_modules_count",
            "total_system_losses"
        ],
        description="Metrics to compare"
    )

    @computed_field
    @property
    def comparison_summary(self) -> Dict[str, Any]:
        """Generate comparison summary."""
        summary = {}
        for metric in self.comparison_metrics:
            values = []
            for design in self.designs:
                if hasattr(design, metric):
                    values.append(getattr(design, metric))
            if values:
                summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "values": dict(zip([d.design_name for d in self.designs], values))
                }
        return summary


class ValidationResult(BaseModel):
    """
    Design validation result.
    """

    is_valid: bool = Field(..., description="Whether design is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    checks_performed: List[str] = Field(default_factory=list, description="Validation checks performed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")

    @computed_field
    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)

    @computed_field
    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)
