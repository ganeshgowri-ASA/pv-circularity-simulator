"""Unit tests for AssetManager."""

import pytest
from datetime import datetime, timedelta

from src.pv_circularity.managers.asset_manager import AssetManager
from src.pv_circularity.database.models import AssetStatus, EquipmentType
from src.pv_circularity.models.schemas import (
    SiteCreate,
    SiteUpdate,
    EquipmentCreate,
    EquipmentUpdate,
    AssetCreate,
    AssetUpdate,
    PerformanceRecordCreate,
)


class TestSiteManagement:
    """Test site management functionality."""

    def test_create_site(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test creating a new site."""
        site = asset_manager.create_site(sample_site_data)

        assert site.id is not None
        assert site.name == sample_site_data.name
        assert site.location == sample_site_data.location
        assert site.capacity_kw == sample_site_data.capacity_kw
        assert site.status == sample_site_data.status
        assert site.created_at is not None
        assert site.updated_at is not None

    def test_create_duplicate_site_name(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate
    ):
        """Test that creating a site with duplicate name raises error."""
        asset_manager.create_site(sample_site_data)

        with pytest.raises(ValueError, match="already exists"):
            asset_manager.create_site(sample_site_data)

    def test_get_site(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test retrieving a site by ID."""
        created_site = asset_manager.create_site(sample_site_data)
        retrieved_site = asset_manager.get_site(created_site.id)

        assert retrieved_site is not None
        assert retrieved_site.id == created_site.id
        assert retrieved_site.name == created_site.name

    def test_get_site_by_name(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test retrieving a site by name."""
        created_site = asset_manager.create_site(sample_site_data)
        retrieved_site = asset_manager.get_site_by_name(sample_site_data.name)

        assert retrieved_site is not None
        assert retrieved_site.id == created_site.id
        assert retrieved_site.name == created_site.name

    def test_get_nonexistent_site(self, asset_manager: AssetManager):
        """Test retrieving a non-existent site returns None."""
        site = asset_manager.get_site(999)
        assert site is None

    def test_update_site(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test updating a site."""
        created_site = asset_manager.create_site(sample_site_data)

        update_data = SiteUpdate(
            capacity_kw=1500.0,
            status=AssetStatus.MAINTENANCE,
        )
        updated_site = asset_manager.update_site(created_site.id, update_data)

        assert updated_site is not None
        assert updated_site.capacity_kw == 1500.0
        assert updated_site.status == AssetStatus.MAINTENANCE
        assert updated_site.name == sample_site_data.name  # Unchanged

    def test_delete_site(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test deleting a site."""
        created_site = asset_manager.create_site(sample_site_data)

        result = asset_manager.delete_site(created_site.id)
        assert result is True

        # Verify site is deleted
        retrieved_site = asset_manager.get_site(created_site.id)
        assert retrieved_site is None

    def test_delete_nonexistent_site(self, asset_manager: AssetManager):
        """Test deleting a non-existent site returns False."""
        result = asset_manager.delete_site(999)
        assert result is False

    def test_list_sites(self, asset_manager: AssetManager):
        """Test listing sites."""
        # Create multiple sites
        for i in range(5):
            site_data = SiteCreate(
                name=f"Site {i}",
                location=f"Location {i}",
                capacity_kw=1000.0 + i * 100,
                installation_date=datetime(2020, 1, i + 1),
                status=AssetStatus.ACTIVE if i % 2 == 0 else AssetStatus.MAINTENANCE,
            )
            asset_manager.create_site(site_data)

        # List all sites
        all_sites = asset_manager.list_sites()
        assert len(all_sites) == 5

        # List active sites only
        active_sites = asset_manager.list_sites(status=AssetStatus.ACTIVE)
        assert len(active_sites) == 3

        # List with limit
        limited_sites = asset_manager.list_sites(limit=2)
        assert len(limited_sites) == 2


class TestEquipmentManagement:
    """Test equipment management functionality."""

    def test_create_equipment(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test creating equipment."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id

        equipment = asset_manager.create_equipment(sample_equipment_data)

        assert equipment.id is not None
        assert equipment.equipment_id == sample_equipment_data.equipment_id
        assert equipment.site_id == site.id
        assert equipment.equipment_type == sample_equipment_data.equipment_type
        assert equipment.rated_power_w == sample_equipment_data.rated_power_w

    def test_create_equipment_invalid_site(
        self, asset_manager: AssetManager, sample_equipment_data: EquipmentCreate
    ):
        """Test creating equipment with invalid site raises error."""
        sample_equipment_data.site_id = 999

        with pytest.raises(ValueError, match="Site with ID 999 not found"):
            asset_manager.create_equipment(sample_equipment_data)

    def test_create_duplicate_equipment_id(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test creating equipment with duplicate equipment_id raises error."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id

        asset_manager.create_equipment(sample_equipment_data)

        with pytest.raises(ValueError, match="already exists"):
            asset_manager.create_equipment(sample_equipment_data)

    def test_get_equipment(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test retrieving equipment by ID."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id

        created_equipment = asset_manager.create_equipment(sample_equipment_data)
        retrieved_equipment = asset_manager.get_equipment(created_equipment.id)

        assert retrieved_equipment is not None
        assert retrieved_equipment.id == created_equipment.id
        assert retrieved_equipment.equipment_id == created_equipment.equipment_id

    def test_get_equipment_by_equipment_id(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test retrieving equipment by equipment_id."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id

        created_equipment = asset_manager.create_equipment(sample_equipment_data)
        retrieved_equipment = asset_manager.get_equipment_by_equipment_id(
            sample_equipment_data.equipment_id
        )

        assert retrieved_equipment is not None
        assert retrieved_equipment.id == created_equipment.id

    def test_update_equipment(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test updating equipment."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id
        created_equipment = asset_manager.create_equipment(sample_equipment_data)

        update_data = EquipmentUpdate(
            status=AssetStatus.MAINTENANCE,
            current_value=400.0,
        )
        updated_equipment = asset_manager.update_equipment(created_equipment.id, update_data)

        assert updated_equipment is not None
        assert updated_equipment.status == AssetStatus.MAINTENANCE
        assert updated_equipment.current_value == 400.0

    def test_delete_equipment(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate, sample_equipment_data: EquipmentCreate
    ):
        """Test deleting equipment."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id
        created_equipment = asset_manager.create_equipment(sample_equipment_data)

        result = asset_manager.delete_equipment(created_equipment.id)
        assert result is True

        retrieved_equipment = asset_manager.get_equipment(created_equipment.id)
        assert retrieved_equipment is None

    def test_list_equipment(self, asset_manager: AssetManager, sample_site_data: SiteCreate):
        """Test listing equipment with filters."""
        site = asset_manager.create_site(sample_site_data)

        # Create multiple equipment items
        for i in range(5):
            equipment_data = EquipmentCreate(
                equipment_id=f"PANEL-{i:03d}",
                site_id=site.id,
                equipment_type=EquipmentType.SOLAR_PANEL if i % 2 == 0 else EquipmentType.INVERTER,
                name=f"Equipment {i}",
                installation_date=datetime(2020, 1, 15),
                status=AssetStatus.ACTIVE if i % 2 == 0 else AssetStatus.MAINTENANCE,
            )
            asset_manager.create_equipment(equipment_data)

        # List all equipment
        all_equipment = asset_manager.list_equipment()
        assert len(all_equipment) == 5

        # Filter by site
        site_equipment = asset_manager.list_equipment(site_id=site.id)
        assert len(site_equipment) == 5

        # Filter by type
        panels = asset_manager.list_equipment(equipment_type=EquipmentType.SOLAR_PANEL)
        assert len(panels) == 3

        # Filter by status
        active = asset_manager.list_equipment(status=AssetStatus.ACTIVE)
        assert len(active) == 3


class TestAssetManagement:
    """Test asset management functionality."""

    def test_create_asset(self, asset_manager: AssetManager, sample_asset_data: AssetCreate):
        """Test creating an asset."""
        asset = asset_manager.create_asset(sample_asset_data)

        assert asset.id is not None
        assert asset.asset_id == sample_asset_data.asset_id
        assert asset.name == sample_asset_data.name
        assert asset.asset_type == sample_asset_data.asset_type

    def test_create_duplicate_asset_id(
        self, asset_manager: AssetManager, sample_asset_data: AssetCreate
    ):
        """Test creating asset with duplicate asset_id raises error."""
        asset_manager.create_asset(sample_asset_data)

        with pytest.raises(ValueError, match="already exists"):
            asset_manager.create_asset(sample_asset_data)

    def test_get_asset(self, asset_manager: AssetManager, sample_asset_data: AssetCreate):
        """Test retrieving an asset."""
        created_asset = asset_manager.create_asset(sample_asset_data)
        retrieved_asset = asset_manager.get_asset(created_asset.id)

        assert retrieved_asset is not None
        assert retrieved_asset.id == created_asset.id

    def test_get_asset_by_asset_id(
        self, asset_manager: AssetManager, sample_asset_data: AssetCreate
    ):
        """Test retrieving an asset by asset_id."""
        created_asset = asset_manager.create_asset(sample_asset_data)
        retrieved_asset = asset_manager.get_asset_by_asset_id(sample_asset_data.asset_id)

        assert retrieved_asset is not None
        assert retrieved_asset.id == created_asset.id

    def test_update_asset(self, asset_manager: AssetManager, sample_asset_data: AssetCreate):
        """Test updating an asset."""
        created_asset = asset_manager.create_asset(sample_asset_data)

        update_data = AssetUpdate(status=AssetStatus.DECOMMISSIONED, current_value=7000.0)
        updated_asset = asset_manager.update_asset(created_asset.id, update_data)

        assert updated_asset is not None
        assert updated_asset.status == AssetStatus.DECOMMISSIONED
        assert updated_asset.current_value == 7000.0

    def test_delete_asset(self, asset_manager: AssetManager, sample_asset_data: AssetCreate):
        """Test deleting an asset."""
        created_asset = asset_manager.create_asset(sample_asset_data)

        result = asset_manager.delete_asset(created_asset.id)
        assert result is True

        retrieved_asset = asset_manager.get_asset(created_asset.id)
        assert retrieved_asset is None

    def test_list_assets(self, asset_manager: AssetManager):
        """Test listing assets with filters."""
        for i in range(5):
            asset_data = AssetCreate(
                asset_id=f"ASSET-{i:03d}",
                name=f"Asset {i}",
                asset_type="solar_equipment" if i % 2 == 0 else "monitoring_device",
                acquisition_date=datetime(2020, 1, 1),
                status=AssetStatus.ACTIVE if i % 2 == 0 else AssetStatus.MAINTENANCE,
            )
            asset_manager.create_asset(asset_data)

        all_assets = asset_manager.list_assets()
        assert len(all_assets) == 5

        solar_assets = asset_manager.list_assets(asset_type="solar_equipment")
        assert len(solar_assets) == 3

        active_assets = asset_manager.list_assets(status=AssetStatus.ACTIVE)
        assert len(active_assets) == 3


class TestPerformanceTracking:
    """Test performance tracking functionality."""

    def test_create_performance_record(
        self,
        asset_manager: AssetManager,
        sample_site_data: SiteCreate,
        sample_equipment_data: EquipmentCreate,
        sample_performance_data: PerformanceRecordCreate,
    ):
        """Test creating a performance record."""
        site = asset_manager.create_site(sample_site_data)
        sample_equipment_data.site_id = site.id
        equipment = asset_manager.create_equipment(sample_equipment_data)

        sample_performance_data.site_id = site.id
        sample_performance_data.equipment_id = equipment.id

        record = asset_manager.create_performance_record(sample_performance_data)

        assert record.id is not None
        assert record.site_id == site.id
        assert record.equipment_id == equipment.id
        assert record.energy_generated_kwh == sample_performance_data.energy_generated_kwh

    def test_create_performance_record_invalid_site(
        self, asset_manager: AssetManager, sample_performance_data: PerformanceRecordCreate
    ):
        """Test creating performance record with invalid site raises error."""
        sample_performance_data.site_id = 999

        with pytest.raises(ValueError, match="Site with ID 999 not found"):
            asset_manager.create_performance_record(sample_performance_data)

    def test_create_performance_record_invalid_equipment(
        self,
        asset_manager: AssetManager,
        sample_site_data: SiteCreate,
        sample_performance_data: PerformanceRecordCreate,
    ):
        """Test creating performance record with invalid equipment raises error."""
        site = asset_manager.create_site(sample_site_data)
        sample_performance_data.site_id = site.id
        sample_performance_data.equipment_id = 999

        with pytest.raises(ValueError, match="Equipment with ID 999 not found"):
            asset_manager.create_performance_record(sample_performance_data)

    def test_list_performance_records(self, populated_db: AssetManager):
        """Test listing performance records."""
        records = populated_db.list_performance_records()
        assert len(records) == 10

    def test_list_performance_records_with_date_filter(self, populated_db: AssetManager):
        """Test listing performance records with date filters."""
        start_date = datetime(2023, 6, 18)
        end_date = datetime(2023, 6, 22)

        records = populated_db.list_performance_records(
            start_date=start_date, end_date=end_date
        )
        assert len(records) == 5  # Records from 18th to 22nd


class TestCoreInventoryMethods:
    """Test core inventory methods: site_inventory, equipment_tracking, performance_history."""

    def test_site_inventory(self, populated_db: AssetManager):
        """Test site_inventory method."""
        # Create additional sites
        for i in range(3):
            site_data = SiteCreate(
                name=f"Additional Site {i}",
                location=f"Location {i}",
                capacity_kw=500.0 + i * 100,
                installation_date=datetime(2021, 1, 1),
                status=AssetStatus.ACTIVE if i % 2 == 0 else AssetStatus.MAINTENANCE,
            )
            populated_db.create_site(site_data)

        result = populated_db.site_inventory(include_summary=True)

        assert "sites" in result
        assert "summary" in result
        assert len(result["sites"]) == 4  # 1 from populated_db + 3 new

        summary = result["summary"]
        assert summary.total_sites == 4
        assert summary.total_capacity_kw > 0
        assert "active" in summary.sites_by_status

    def test_site_inventory_with_status_filter(self, populated_db: AssetManager):
        """Test site_inventory with status filter."""
        for i in range(3):
            site_data = SiteCreate(
                name=f"Site {i}",
                location=f"Location {i}",
                capacity_kw=500.0,
                installation_date=datetime(2021, 1, 1),
                status=AssetStatus.ACTIVE if i % 2 == 0 else AssetStatus.MAINTENANCE,
            )
            populated_db.create_site(site_data)

        result = populated_db.site_inventory(status=AssetStatus.ACTIVE, include_summary=True)

        assert "sites" in result
        active_count = len(result["sites"])
        assert active_count >= 2  # At least 2 active sites

    def test_equipment_tracking(self, populated_db: AssetManager):
        """Test equipment_tracking method."""
        result = populated_db.equipment_tracking(include_summary=True)

        assert "equipment" in result
        assert "summary" in result
        assert len(result["equipment"]) == 1  # 1 from populated_db

        summary = result["summary"]
        assert summary.total_equipment == 1
        assert summary.total_rated_power_kw > 0
        assert "solar_panel" in summary.equipment_by_type

    def test_equipment_tracking_with_filters(self, populated_db: AssetManager):
        """Test equipment_tracking with filters."""
        # Get site from populated_db
        sites = populated_db.list_sites()
        site_id = sites[0].id

        # Create additional equipment
        for i in range(3):
            equipment_data = EquipmentCreate(
                equipment_id=f"INV-{i:03d}",
                site_id=site_id,
                equipment_type=EquipmentType.INVERTER,
                name=f"Inverter {i}",
                installation_date=datetime(2020, 1, 15),
                rated_power_w=5000.0,
            )
            populated_db.create_equipment(equipment_data)

        result = populated_db.equipment_tracking(
            equipment_type=EquipmentType.INVERTER, include_summary=True
        )

        assert len(result["equipment"]) == 3
        summary = result["summary"]
        assert summary.equipment_by_type["inverter"] == 3

    def test_performance_history(self, populated_db: AssetManager):
        """Test performance_history method."""
        result = populated_db.performance_history(include_summary=True)

        assert "records" in result
        assert "summary" in result
        assert len(result["records"]) == 10

        summary = result["summary"]
        assert summary.total_records == 10
        assert summary.total_energy_kwh > 0
        assert summary.average_power_kw > 0
        assert "start" in summary.date_range
        assert "end" in summary.date_range

    def test_performance_history_with_date_filter(self, populated_db: AssetManager):
        """Test performance_history with date filters."""
        start_date = datetime(2023, 6, 18)
        end_date = datetime(2023, 6, 22)

        result = populated_db.performance_history(
            start_date=start_date, end_date=end_date, include_summary=True
        )

        assert len(result["records"]) == 5
        summary = result["summary"]
        assert summary.total_records == 5

    def test_performance_history_without_summary(self, populated_db: AssetManager):
        """Test performance_history without summary."""
        result = populated_db.performance_history(include_summary=False)

        assert "records" in result
        assert "summary" not in result


class TestCircularEconomyFeatures:
    """Test circular economy specific features."""

    def test_equipment_material_composition(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate
    ):
        """Test equipment with material composition for circular economy."""
        site = asset_manager.create_site(sample_site_data)

        equipment_data = EquipmentCreate(
            equipment_id="RECYCLABLE-001",
            site_id=site.id,
            equipment_type=EquipmentType.SOLAR_PANEL,
            name="Recyclable Panel",
            installation_date=datetime(2020, 1, 15),
            recyclable=True,
            material_composition={"silicon": 0.4, "glass": 0.3, "aluminum": 0.2, "other": 0.1},
            recycling_value=75.0,
        )

        equipment = asset_manager.create_equipment(equipment_data)

        assert equipment.recyclable is True
        assert equipment.material_composition is not None
        assert equipment.material_composition["silicon"] == 0.4
        assert equipment.recycling_value == 75.0

    def test_equipment_lifecycle_tracking(
        self, asset_manager: AssetManager, sample_site_data: SiteCreate
    ):
        """Test equipment lifecycle tracking."""
        site = asset_manager.create_site(sample_site_data)

        equipment_data = EquipmentCreate(
            equipment_id="LIFECYCLE-001",
            site_id=site.id,
            equipment_type=EquipmentType.SOLAR_PANEL,
            name="Lifecycle Panel",
            manufacturing_date=datetime(2019, 6, 1),
            installation_date=datetime(2020, 1, 15),
            expected_lifetime_years=25.0,
            last_maintenance_date=datetime(2023, 1, 15),
            next_maintenance_date=datetime(2024, 1, 15),
            status=AssetStatus.ACTIVE,
        )

        equipment = asset_manager.create_equipment(equipment_data)

        assert equipment.manufacturing_date is not None
        assert equipment.expected_lifetime_years == 25.0
        assert equipment.last_maintenance_date is not None
        assert equipment.next_maintenance_date is not None
