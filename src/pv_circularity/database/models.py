"""SQLAlchemy database models for asset management and portfolio tracking."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Enum as SQLEnum,
    Boolean,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class AssetStatus(str, enum.Enum):
    """Asset lifecycle status."""

    PLANNED = "planned"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    DECOMMISSIONED = "decommissioned"
    RECYCLED = "recycled"


class EquipmentType(str, enum.Enum):
    """Equipment type classification."""

    SOLAR_PANEL = "solar_panel"
    INVERTER = "inverter"
    BATTERY = "battery"
    MOUNTING_STRUCTURE = "mounting_structure"
    MONITORING_DEVICE = "monitoring_device"
    OTHER = "other"


class Site(Base):
    """Site/Installation model representing a PV installation location."""

    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    location = Column(String(500), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    capacity_kw = Column(Float, nullable=False)
    installation_date = Column(DateTime, nullable=False)
    status = Column(SQLEnum(AssetStatus), default=AssetStatus.ACTIVE, nullable=False)
    description = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional site-specific data
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    equipment = relationship("Equipment", back_populates="site", cascade="all, delete-orphan")
    performance_records = relationship(
        "PerformanceRecord", back_populates="site", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Site(id={self.id}, name='{self.name}', capacity={self.capacity_kw}kW)>"


class Asset(Base):
    """Base asset model for tracking PV system components."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    asset_type = Column(String(100), nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True)
    status = Column(SQLEnum(AssetStatus), default=AssetStatus.ACTIVE, nullable=False)
    acquisition_date = Column(DateTime, nullable=False)
    installation_date = Column(DateTime, nullable=True)
    warranty_expiry = Column(DateTime, nullable=True)
    expected_lifetime_years = Column(Float, nullable=True)
    purchase_cost = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Asset(id={self.id}, asset_id='{self.asset_id}', type='{self.asset_type}')>"


class Equipment(Base):
    """Equipment model for tracking specific PV components."""

    __tablename__ = "equipment"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String(100), unique=True, nullable=False, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    equipment_type = Column(SQLEnum(EquipmentType), nullable=False)
    name = Column(String(255), nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True, unique=True)
    status = Column(SQLEnum(AssetStatus), default=AssetStatus.ACTIVE, nullable=False)

    # Technical specifications
    rated_power_w = Column(Float, nullable=True)
    efficiency_percent = Column(Float, nullable=True)
    degradation_rate_percent = Column(Float, nullable=True)
    temperature_coefficient = Column(Float, nullable=True)

    # Lifecycle tracking
    manufacturing_date = Column(DateTime, nullable=True)
    installation_date = Column(DateTime, nullable=False)
    warranty_expiry = Column(DateTime, nullable=True)
    expected_lifetime_years = Column(Float, nullable=True)
    last_maintenance_date = Column(DateTime, nullable=True)
    next_maintenance_date = Column(DateTime, nullable=True)

    # Financial
    purchase_cost = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)

    # Circular economy attributes
    recyclable = Column(Boolean, default=True, nullable=False)
    material_composition = Column(JSON, nullable=True)  # e.g., {"silicon": 0.4, "glass": 0.3}
    recycling_value = Column(Float, nullable=True)

    # Additional data
    description = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    site = relationship("Site", back_populates="equipment")
    performance_records = relationship(
        "PerformanceRecord", back_populates="equipment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Equipment(id={self.id}, equipment_id='{self.equipment_id}', "
            f"type='{self.equipment_type}')>"
        )


class PerformanceRecord(Base):
    """Performance tracking records for sites and equipment."""

    __tablename__ = "performance_records"

    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Performance metrics
    energy_generated_kwh = Column(Float, nullable=True)
    power_output_kw = Column(Float, nullable=True)
    efficiency_percent = Column(Float, nullable=True)
    capacity_factor_percent = Column(Float, nullable=True)
    performance_ratio = Column(Float, nullable=True)

    # Environmental conditions
    irradiance_w_m2 = Column(Float, nullable=True)
    temperature_c = Column(Float, nullable=True)
    wind_speed_ms = Column(Float, nullable=True)

    # System health
    availability_percent = Column(Float, nullable=True)
    downtime_hours = Column(Float, nullable=True)
    fault_codes = Column(JSON, nullable=True)

    # Additional metrics
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    site = relationship("Site", back_populates="performance_records")
    equipment = relationship("Equipment", back_populates="performance_records")

    def __repr__(self) -> str:
        return (
            f"<PerformanceRecord(id={self.id}, site_id={self.site_id}, "
            f"timestamp='{self.timestamp}')>"
        )
