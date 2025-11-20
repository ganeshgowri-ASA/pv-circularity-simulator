"""Pytest configuration and fixtures."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.pv_circularity.database.models import Base, AssetStatus, EquipmentType
from src.pv_circularity.managers.asset_manager import AssetManager
from src.pv_circularity.models.schemas import (
    SiteCreate,
    EquipmentCreate,
    AssetCreate,
    PerformanceRecordCreate,
)


@pytest.fixture(scope="function")
def db_session() -> Session:
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def asset_manager(db_session: Session) -> AssetManager:
    """Create an AssetManager instance with test database session."""
    return AssetManager(db_session)


@pytest.fixture
def sample_site_data() -> SiteCreate:
    """Sample site data for testing."""
    return SiteCreate(
        name="Test Solar Farm 1",
        location="California, USA",
        latitude=37.7749,
        longitude=-122.4194,
        capacity_kw=1000.0,
        installation_date=datetime(2020, 1, 15),
        status=AssetStatus.ACTIVE,
        description="Test solar installation",
        metadata={"region": "West Coast", "grid_connection": "active"},
    )


@pytest.fixture
def sample_equipment_data() -> EquipmentCreate:
    """Sample equipment data for testing."""
    return EquipmentCreate(
        equipment_id="PANEL-001",
        site_id=1,
        equipment_type=EquipmentType.SOLAR_PANEL,
        name="High Efficiency Solar Panel",
        manufacturer="SolarTech Inc.",
        model="ST-500W",
        serial_number="SN123456789",
        status=AssetStatus.ACTIVE,
        rated_power_w=500.0,
        efficiency_percent=22.5,
        degradation_rate_percent=0.5,
        temperature_coefficient=-0.35,
        manufacturing_date=datetime(2019, 11, 1),
        installation_date=datetime(2020, 1, 15),
        warranty_expiry=datetime(2030, 1, 15),
        expected_lifetime_years=25.0,
        purchase_cost=500.0,
        current_value=450.0,
        recyclable=True,
        material_composition={"silicon": 0.4, "glass": 0.3, "aluminum": 0.2, "other": 0.1},
        recycling_value=50.0,
        description="High efficiency monocrystalline panel",
    )


@pytest.fixture
def sample_asset_data() -> AssetCreate:
    """Sample asset data for testing."""
    return AssetCreate(
        asset_id="ASSET-001",
        name="Test Asset",
        asset_type="solar_equipment",
        manufacturer="TestCorp",
        model="TC-1000",
        serial_number="TC123456",
        status=AssetStatus.ACTIVE,
        acquisition_date=datetime(2019, 12, 1),
        installation_date=datetime(2020, 1, 15),
        warranty_expiry=datetime(2029, 12, 1),
        expected_lifetime_years=20.0,
        purchase_cost=10000.0,
        current_value=8500.0,
        description="Test asset for inventory",
    )


@pytest.fixture
def sample_performance_data() -> PerformanceRecordCreate:
    """Sample performance record data for testing."""
    return PerformanceRecordCreate(
        site_id=1,
        equipment_id=1,
        timestamp=datetime(2023, 6, 15, 12, 0, 0),
        energy_generated_kwh=450.0,
        power_output_kw=500.0,
        efficiency_percent=22.0,
        capacity_factor_percent=85.0,
        performance_ratio=0.95,
        irradiance_w_m2=1000.0,
        temperature_c=25.0,
        wind_speed_ms=3.5,
        availability_percent=99.5,
        downtime_hours=0.1,
        fault_codes=None,
    )


@pytest.fixture
def populated_db(
    asset_manager: AssetManager,
    sample_site_data: SiteCreate,
    sample_equipment_data: EquipmentCreate,
    sample_performance_data: PerformanceRecordCreate,
) -> AssetManager:
    """Create a populated database with sample data."""
    # Create site
    site = asset_manager.create_site(sample_site_data)

    # Update equipment data with correct site_id
    sample_equipment_data.site_id = site.id

    # Create equipment
    equipment = asset_manager.create_equipment(sample_equipment_data)

    # Update performance data with correct IDs
    sample_performance_data.site_id = site.id
    sample_performance_data.equipment_id = equipment.id

    # Create performance records
    for i in range(10):
        perf_data = sample_performance_data.model_copy()
        perf_data.timestamp = datetime(2023, 6, 15 + i, 12, 0, 0)
        perf_data.energy_generated_kwh = 450.0 + i * 10
        asset_manager.create_performance_record(perf_data)

    return asset_manager
