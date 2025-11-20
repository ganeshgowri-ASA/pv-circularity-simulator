"""Pydantic schemas for validation and serialization."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class AssetStatus(str, Enum):
    """Asset lifecycle status."""

    PLANNED = "planned"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    DECOMMISSIONED = "decommissioned"
    RECYCLED = "recycled"


class EquipmentType(str, Enum):
    """Equipment type classification."""

    SOLAR_PANEL = "solar_panel"
    INVERTER = "inverter"
    BATTERY = "battery"
    MOUNTING_STRUCTURE = "mounting_structure"
    MONITORING_DEVICE = "monitoring_device"
    OTHER = "other"


# Site Schemas


class SiteBase(BaseModel):
    """Base site schema with common fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Site name")
    location: str = Field(..., min_length=1, max_length=500, description="Site location")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    capacity_kw: float = Field(..., gt=0, description="Site capacity in kW")
    installation_date: datetime = Field(..., description="Installation date")
    status: AssetStatus = Field(default=AssetStatus.ACTIVE, description="Site status")
    description: Optional[str] = Field(None, description="Site description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional site data")


class SiteCreate(SiteBase):
    """Schema for creating a new site."""

    pass


class SiteUpdate(BaseModel):
    """Schema for updating a site."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    location: Optional[str] = Field(None, min_length=1, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    capacity_kw: Optional[float] = Field(None, gt=0)
    installation_date: Optional[datetime] = None
    status: Optional[AssetStatus] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SiteResponse(SiteBase):
    """Schema for site response."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Asset Schemas


class AssetBase(BaseModel):
    """Base asset schema with common fields."""

    asset_id: str = Field(..., min_length=1, max_length=100, description="Unique asset ID")
    name: str = Field(..., min_length=1, max_length=255, description="Asset name")
    asset_type: str = Field(..., min_length=1, max_length=100, description="Asset type")
    manufacturer: Optional[str] = Field(None, max_length=255, description="Manufacturer")
    model: Optional[str] = Field(None, max_length=255, description="Model")
    serial_number: Optional[str] = Field(None, max_length=255, description="Serial number")
    status: AssetStatus = Field(default=AssetStatus.ACTIVE, description="Asset status")
    acquisition_date: datetime = Field(..., description="Acquisition date")
    installation_date: Optional[datetime] = Field(None, description="Installation date")
    warranty_expiry: Optional[datetime] = Field(None, description="Warranty expiry date")
    expected_lifetime_years: Optional[float] = Field(None, gt=0, description="Expected lifetime")
    purchase_cost: Optional[float] = Field(None, ge=0, description="Purchase cost")
    current_value: Optional[float] = Field(None, ge=0, description="Current value")
    description: Optional[str] = Field(None, description="Asset description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional asset data")


class AssetCreate(AssetBase):
    """Schema for creating a new asset."""

    pass


class AssetUpdate(BaseModel):
    """Schema for updating an asset."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    asset_type: Optional[str] = Field(None, min_length=1, max_length=100)
    manufacturer: Optional[str] = Field(None, max_length=255)
    model: Optional[str] = Field(None, max_length=255)
    serial_number: Optional[str] = Field(None, max_length=255)
    status: Optional[AssetStatus] = None
    acquisition_date: Optional[datetime] = None
    installation_date: Optional[datetime] = None
    warranty_expiry: Optional[datetime] = None
    expected_lifetime_years: Optional[float] = Field(None, gt=0)
    purchase_cost: Optional[float] = Field(None, ge=0)
    current_value: Optional[float] = Field(None, ge=0)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AssetResponse(AssetBase):
    """Schema for asset response."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Equipment Schemas


class EquipmentBase(BaseModel):
    """Base equipment schema with common fields."""

    equipment_id: str = Field(..., min_length=1, max_length=100, description="Unique equipment ID")
    site_id: int = Field(..., description="Site ID")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    name: str = Field(..., min_length=1, max_length=255, description="Equipment name")
    manufacturer: Optional[str] = Field(None, max_length=255, description="Manufacturer")
    model: Optional[str] = Field(None, max_length=255, description="Model")
    serial_number: Optional[str] = Field(None, max_length=255, description="Serial number")
    status: AssetStatus = Field(default=AssetStatus.ACTIVE, description="Equipment status")

    # Technical specifications
    rated_power_w: Optional[float] = Field(None, gt=0, description="Rated power in watts")
    efficiency_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Efficiency percentage"
    )
    degradation_rate_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Annual degradation rate"
    )
    temperature_coefficient: Optional[float] = Field(
        None, description="Temperature coefficient (%/°C)"
    )

    # Lifecycle tracking
    manufacturing_date: Optional[datetime] = Field(None, description="Manufacturing date")
    installation_date: datetime = Field(..., description="Installation date")
    warranty_expiry: Optional[datetime] = Field(None, description="Warranty expiry date")
    expected_lifetime_years: Optional[float] = Field(None, gt=0, description="Expected lifetime")
    last_maintenance_date: Optional[datetime] = Field(None, description="Last maintenance date")
    next_maintenance_date: Optional[datetime] = Field(None, description="Next maintenance date")

    # Financial
    purchase_cost: Optional[float] = Field(None, ge=0, description="Purchase cost")
    current_value: Optional[float] = Field(None, ge=0, description="Current value")

    # Circular economy
    recyclable: bool = Field(default=True, description="Is recyclable")
    material_composition: Optional[Dict[str, float]] = Field(
        None, description="Material composition"
    )
    recycling_value: Optional[float] = Field(None, ge=0, description="Recycling value")

    # Additional
    description: Optional[str] = Field(None, description="Equipment description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional equipment data")

    @field_validator("material_composition")
    @classmethod
    def validate_material_composition(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate that material composition percentages sum to approximately 1.0."""
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating-point errors
                raise ValueError(f"Material composition must sum to 1.0, got {total}")
        return v


class EquipmentCreate(EquipmentBase):
    """Schema for creating new equipment."""

    pass


class EquipmentUpdate(BaseModel):
    """Schema for updating equipment."""

    site_id: Optional[int] = None
    equipment_type: Optional[EquipmentType] = None
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    manufacturer: Optional[str] = Field(None, max_length=255)
    model: Optional[str] = Field(None, max_length=255)
    serial_number: Optional[str] = Field(None, max_length=255)
    status: Optional[AssetStatus] = None
    rated_power_w: Optional[float] = Field(None, gt=0)
    efficiency_percent: Optional[float] = Field(None, ge=0, le=100)
    degradation_rate_percent: Optional[float] = Field(None, ge=0, le=100)
    temperature_coefficient: Optional[float] = None
    manufacturing_date: Optional[datetime] = None
    installation_date: Optional[datetime] = None
    warranty_expiry: Optional[datetime] = None
    expected_lifetime_years: Optional[float] = Field(None, gt=0)
    last_maintenance_date: Optional[datetime] = None
    next_maintenance_date: Optional[datetime] = None
    purchase_cost: Optional[float] = Field(None, ge=0)
    current_value: Optional[float] = Field(None, ge=0)
    recyclable: Optional[bool] = None
    material_composition: Optional[Dict[str, float]] = None
    recycling_value: Optional[float] = Field(None, ge=0)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EquipmentResponse(EquipmentBase):
    """Schema for equipment response."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Performance Record Schemas


class PerformanceRecordBase(BaseModel):
    """Base performance record schema."""

    site_id: int = Field(..., description="Site ID")
    equipment_id: Optional[int] = Field(None, description="Equipment ID (optional)")
    timestamp: datetime = Field(..., description="Record timestamp")

    # Performance metrics
    energy_generated_kwh: Optional[float] = Field(None, ge=0, description="Energy generated (kWh)")
    power_output_kw: Optional[float] = Field(None, ge=0, description="Power output (kW)")
    efficiency_percent: Optional[float] = Field(None, ge=0, le=100, description="Efficiency (%)")
    capacity_factor_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Capacity factor (%)"
    )
    performance_ratio: Optional[float] = Field(None, ge=0, le=1, description="Performance ratio")

    # Environmental conditions
    irradiance_w_m2: Optional[float] = Field(None, ge=0, description="Irradiance (W/m²)")
    temperature_c: Optional[float] = Field(None, description="Temperature (°C)")
    wind_speed_ms: Optional[float] = Field(None, ge=0, description="Wind speed (m/s)")

    # System health
    availability_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Availability (%)"
    )
    downtime_hours: Optional[float] = Field(None, ge=0, description="Downtime (hours)")
    fault_codes: Optional[List[str]] = Field(None, description="Fault codes")

    # Additional metrics
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metrics")


class PerformanceRecordCreate(PerformanceRecordBase):
    """Schema for creating a performance record."""

    pass


class PerformanceRecordResponse(PerformanceRecordBase):
    """Schema for performance record response."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Portfolio and Summary Schemas


class SiteInventorySummary(BaseModel):
    """Summary of site inventory."""

    total_sites: int = Field(..., description="Total number of sites")
    total_capacity_kw: float = Field(..., description="Total capacity in kW")
    sites_by_status: Dict[str, int] = Field(..., description="Site count by status")
    average_capacity_kw: float = Field(..., description="Average site capacity")


class EquipmentInventorySummary(BaseModel):
    """Summary of equipment inventory."""

    total_equipment: int = Field(..., description="Total equipment count")
    equipment_by_type: Dict[str, int] = Field(..., description="Equipment count by type")
    equipment_by_status: Dict[str, int] = Field(..., description="Equipment count by status")
    total_rated_power_kw: float = Field(..., description="Total rated power in kW")
    average_efficiency_percent: Optional[float] = Field(
        None, description="Average efficiency percentage"
    )


class PerformanceHistorySummary(BaseModel):
    """Summary of performance history."""

    total_records: int = Field(..., description="Total number of records")
    date_range: Dict[str, datetime] = Field(..., description="Date range (start, end)")
    total_energy_kwh: float = Field(..., description="Total energy generated (kWh)")
    average_power_kw: float = Field(..., description="Average power output (kW)")
    average_efficiency_percent: Optional[float] = Field(
        None, description="Average efficiency percentage"
    )
    average_capacity_factor_percent: Optional[float] = Field(
        None, description="Average capacity factor"
    )
