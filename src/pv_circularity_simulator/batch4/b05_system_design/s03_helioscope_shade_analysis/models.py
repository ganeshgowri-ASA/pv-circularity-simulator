"""
Pydantic models for shade analysis data validation and configuration.

This module provides comprehensive data models for all shade analysis components,
ensuring type safety and data validation throughout the system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TranspositionModel(str, Enum):
    """Transposition models for converting GHI to POA irradiance."""

    ISOTROPIC = "isotropic"
    PEREZ = "perez"
    HAY_DAVIES = "hay_davies"
    REINDL = "reindl"
    KLUCHER = "klucher"


class AOIModel(str, Enum):
    """Angle of incidence correction models."""

    ASHRAE = "ashrae"
    PHYSICAL = "physical"
    SANDIA = "sandia"
    MARTIN_RUIZ = "martin_ruiz"


class FileFormat(str, Enum):
    """Supported 3D model file formats."""

    SKETCHUP = "skp"
    OBJ = "obj"
    STL = "stl"
    DXF = "dxf"
    PLY = "ply"


class TrackerType(str, Enum):
    """Types of tracking systems."""

    FIXED_TILT = "fixed_tilt"
    SINGLE_AXIS = "single_axis"
    DUAL_AXIS = "dual_axis"


class Location(BaseModel):
    """Geographic location data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    elevation: float = Field(default=0.0, description="Elevation above sea level in meters")
    timezone: str = Field(default="UTC", description="IANA timezone identifier")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within valid range."""
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within valid range."""
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v


class TerrainData(BaseModel):
    """Digital terrain model data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_coordinates: List[float] = Field(..., description="X coordinates of terrain points (meters)")
    y_coordinates: List[float] = Field(..., description="Y coordinates of terrain points (meters)")
    elevations: List[List[float]] = Field(..., description="Elevation grid (meters)")
    resolution: float = Field(default=1.0, description="Spatial resolution in meters")
    datum: str = Field(default="WGS84", description="Geodetic datum")


class Obstacle(BaseModel):
    """3D obstacle definition (buildings, trees, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Obstacle identifier")
    vertices: List[Tuple[float, float, float]] = Field(..., description="3D vertices (x, y, z) in meters")
    faces: List[List[int]] = Field(..., description="Face definitions as vertex indices")
    height: float = Field(..., gt=0, description="Maximum height in meters")
    obstacle_type: str = Field(default="building", description="Type of obstacle")
    reflectance: float = Field(default=0.2, ge=0, le=1, description="Surface reflectance (albedo)")


class HorizonProfile(BaseModel):
    """Horizon profile data for far shading analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    azimuths: List[float] = Field(..., description="Azimuth angles in degrees (0-360)")
    elevations: List[float] = Field(..., description="Horizon elevation angles in degrees")
    source: str = Field(default="manual", description="Data source (survey, google_earth, photo, etc.)")

    @field_validator("azimuths")
    @classmethod
    def validate_azimuths(cls, v: List[float]) -> List[float]:
        """Validate azimuth angles are in valid range."""
        if not all(0 <= az <= 360 for az in v):
            raise ValueError("All azimuth angles must be between 0 and 360 degrees")
        return v

    @field_validator("elevations")
    @classmethod
    def validate_elevations(cls, v: List[float]) -> List[float]:
        """Validate elevation angles are in valid range."""
        if not all(-90 <= el <= 90 for el in v):
            raise ValueError("All elevation angles must be between -90 and 90 degrees")
        return v


class SiteModel(BaseModel):
    """Comprehensive site model with terrain, obstacles, and horizon data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    location: Location = Field(..., description="Geographic location")
    terrain: Optional[TerrainData] = Field(default=None, description="Digital terrain model")
    obstacles: List[Obstacle] = Field(default_factory=list, description="Site obstacles")
    horizon_profile: Optional[HorizonProfile] = Field(default=None, description="Horizon profile")
    albedo: float = Field(default=0.2, ge=0, le=1, description="Ground reflectance")
    site_boundary: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Site boundary polygon (x, y) coordinates"
    )


class ArrayGeometry(BaseModel):
    """PV array geometry configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tilt: float = Field(..., ge=0, le=90, description="Tilt angle in degrees")
    azimuth: float = Field(..., ge=0, le=360, description="Azimuth angle in degrees (0=North)")
    gcr: float = Field(..., gt=0, le=1, description="Ground coverage ratio")
    module_width: float = Field(..., gt=0, description="Module width in meters")
    module_height: float = Field(..., gt=0, description="Module height in meters")
    modules_per_string: int = Field(..., gt=0, description="Number of modules per string")
    strings_per_row: int = Field(default=1, gt=0, description="Number of strings per row")
    row_spacing: float = Field(..., gt=0, description="Row-to-row spacing in meters")
    tracker_type: TrackerType = Field(default=TrackerType.FIXED_TILT, description="Tracker type")
    tracker_max_angle: float = Field(default=60.0, ge=0, le=90, description="Max tracker rotation angle")
    enable_backtracking: bool = Field(default=True, description="Enable backtracking for trackers")


class ModuleElectricalParams(BaseModel):
    """Module electrical parameters for shading analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells_in_series: int = Field(..., gt=0, description="Number of cells in series")
    cell_rows: int = Field(..., gt=0, description="Number of cell rows")
    cell_columns: int = Field(..., gt=0, description="Number of cell columns")
    bypass_diodes: int = Field(..., gt=0, description="Number of bypass diodes")
    cells_per_diode: int = Field(..., gt=0, description="Cells per bypass diode")
    v_oc: float = Field(..., gt=0, description="Open circuit voltage (V)")
    i_sc: float = Field(..., gt=0, description="Short circuit current (A)")
    v_mp: float = Field(..., gt=0, description="Max power voltage (V)")
    i_mp: float = Field(..., gt=0, description="Max power current (A)")
    p_max: float = Field(..., gt=0, description="Maximum power (W)")
    temp_coeff_p: float = Field(default=-0.004, description="Power temperature coefficient (%/°C)")
    temp_coeff_v: float = Field(default=-0.003, description="Voltage temperature coefficient (V/°C)")


class ShadeAnalysisConfig(BaseModel):
    """Configuration for shade analysis execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_date: datetime = Field(..., description="Analysis start date and time")
    end_date: datetime = Field(..., description="Analysis end date and time")
    timestep_minutes: int = Field(default=60, gt=0, le=60, description="Timestep in minutes")

    enable_near_shading: bool = Field(default=True, description="Enable near shading analysis")
    enable_far_shading: bool = Field(default=True, description="Enable far shading from horizon")
    enable_electrical_model: bool = Field(default=True, description="Enable electrical shading effects")

    transposition_model: TranspositionModel = Field(
        default=TranspositionModel.PEREZ,
        description="Transposition model for POA irradiance"
    )
    aoi_model: AOIModel = Field(default=AOIModel.ASHRAE, description="AOI correction model")

    soiling_loss: float = Field(default=0.02, ge=0, le=1, description="Soiling loss fraction")
    spectral_mismatch: float = Field(default=0.0, ge=-0.1, le=0.1, description="Spectral mismatch correction")

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: datetime, info) -> datetime:
        """Validate end date is after start date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class SunPosition(BaseModel):
    """Solar position at a specific time and location."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Date and time")
    azimuth: float = Field(..., ge=0, le=360, description="Solar azimuth angle (degrees, 0=North)")
    elevation: float = Field(..., ge=-90, le=90, description="Solar elevation angle (degrees)")
    zenith: float = Field(..., ge=0, le=180, description="Solar zenith angle (degrees)")
    declination: float = Field(..., ge=-23.45, le=23.45, description="Solar declination (degrees)")
    hour_angle: float = Field(..., ge=-180, le=180, description="Hour angle (degrees)")
    equation_of_time: float = Field(..., description="Equation of time (minutes)")


class IrradianceComponents(BaseModel):
    """Irradiance components for POA calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Date and time")
    ghi: float = Field(..., ge=0, description="Global horizontal irradiance (W/m²)")
    dni: float = Field(..., ge=0, description="Direct normal irradiance (W/m²)")
    dhi: float = Field(..., ge=0, description="Diffuse horizontal irradiance (W/m²)")

    poa_direct: float = Field(default=0.0, ge=0, description="POA direct irradiance (W/m²)")
    poa_diffuse: float = Field(default=0.0, ge=0, description="POA diffuse irradiance (W/m²)")
    poa_ground: float = Field(default=0.0, ge=0, description="POA ground-reflected irradiance (W/m²)")
    poa_global: float = Field(default=0.0, ge=0, description="POA global irradiance (W/m²)")

    aoi: float = Field(default=0.0, ge=0, le=180, description="Angle of incidence (degrees)")
    aoi_modifier: float = Field(default=1.0, ge=0, le=1, description="AOI loss modifier")


class ShadingLoss(BaseModel):
    """Shading loss details for a specific time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Date and time")
    near_shading_loss: float = Field(default=0.0, ge=0, le=1, description="Near shading loss fraction")
    far_shading_loss: float = Field(default=0.0, ge=0, le=1, description="Far shading loss fraction")
    total_shading_loss: float = Field(default=0.0, ge=0, le=1, description="Total shading loss fraction")
    shaded_modules: int = Field(default=0, ge=0, description="Number of fully shaded modules")
    partially_shaded_modules: int = Field(default=0, ge=0, description="Number of partially shaded modules")


class ShadeAnalysisResult(BaseModel):
    """Comprehensive shade analysis results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_model: SiteModel = Field(..., description="Site model used for analysis")
    array_geometry: ArrayGeometry = Field(..., description="Array geometry configuration")
    config: ShadeAnalysisConfig = Field(..., description="Analysis configuration")

    shading_losses: List[ShadingLoss] = Field(default_factory=list, description="Time-series shading losses")
    monthly_losses: Dict[int, float] = Field(
        default_factory=dict,
        description="Monthly average shading losses (month: loss_fraction)"
    )
    annual_shading_loss: float = Field(default=0.0, ge=0, le=1, description="Annual average shading loss")

    worst_shaded_modules: List[int] = Field(
        default_factory=list,
        description="Module indices with highest shading losses"
    )

    irradiance_data: List[IrradianceComponents] = Field(
        default_factory=list,
        description="Time-series POA irradiance data"
    )


class ElectricalShadeResult(BaseModel):
    """Electrical shading analysis results with bypass diode effects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Date and time")
    module_id: int = Field(..., ge=0, description="Module identifier")

    shaded_cells: List[int] = Field(default_factory=list, description="Indices of shaded cells")
    active_bypass_diodes: List[int] = Field(
        default_factory=list,
        description="Indices of active bypass diodes"
    )

    voltage_loss: float = Field(default=0.0, ge=0, description="Voltage loss due to bypass diodes (V)")
    power_loss: float = Field(default=0.0, ge=0, le=1, description="Power loss fraction")
    current_mismatch_loss: float = Field(default=0.0, ge=0, le=1, description="Current mismatch loss")

    hotspot_risk: bool = Field(default=False, description="Hotspot risk detected")
    hotspot_cells: List[int] = Field(default_factory=list, description="Cells at risk of hotspot")


class LayoutOptimizationResult(BaseModel):
    """Results from layout optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimal_tilt: float = Field(..., ge=0, le=90, description="Optimal tilt angle (degrees)")
    optimal_azimuth: float = Field(..., ge=0, le=360, description="Optimal azimuth (degrees)")
    optimal_gcr: float = Field(..., gt=0, le=1, description="Optimal ground coverage ratio")
    optimal_row_spacing: float = Field(..., gt=0, description="Optimal row spacing (meters)")

    annual_energy_yield: float = Field(..., ge=0, description="Annual energy yield (kWh)")
    capacity_factor: float = Field(..., ge=0, le=1, description="Capacity factor")

    shading_loss: float = Field(default=0.0, ge=0, le=1, description="Annual shading loss")
    aoi_loss: float = Field(default=0.0, ge=0, le=1, description="Annual AOI loss")
    soiling_loss: float = Field(default=0.0, ge=0, le=1, description="Annual soiling loss")

    land_use_efficiency: float = Field(..., gt=0, description="Energy per unit area (kWh/m²)")

    optimization_iterations: int = Field(..., ge=0, description="Number of optimization iterations")
    convergence_achieved: bool = Field(..., description="Whether optimization converged")


class SunPathPoint(BaseModel):
    """Point on the sun path for visualization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Date and time")
    azimuth: float = Field(..., ge=0, le=360, description="Solar azimuth (degrees)")
    elevation: float = Field(..., ge=-90, le=90, description="Solar elevation (degrees)")
    is_daylight: bool = Field(..., description="Whether sun is above horizon")
