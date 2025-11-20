"""Asset Manager for PV Circularity Simulator - Portfolio Tracking and Management."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database.models import Asset, Equipment, Site, PerformanceRecord, AssetStatus, EquipmentType
from ..models.schemas import (
    AssetCreate,
    AssetUpdate,
    AssetResponse,
    EquipmentCreate,
    EquipmentUpdate,
    EquipmentResponse,
    SiteCreate,
    SiteUpdate,
    SiteResponse,
    PerformanceRecordCreate,
    PerformanceRecordResponse,
    SiteInventorySummary,
    EquipmentInventorySummary,
    PerformanceHistorySummary,
)


class AssetManager:
    """
    Comprehensive Asset Manager for PV installations.

    Manages sites, equipment, assets, and performance tracking for solar installations
    with support for circular economy principles.
    """

    def __init__(self, db_session: Session):
        """
        Initialize AssetManager.

        Args:
            db_session: SQLAlchemy database session.
        """
        self.db = db_session

    # ========== SITE MANAGEMENT ==========

    def create_site(self, site_data: SiteCreate) -> SiteResponse:
        """
        Create a new site.

        Args:
            site_data: Site creation data.

        Returns:
            Created site response.

        Raises:
            ValueError: If site with the same name already exists.
        """
        # Check if site with name already exists
        existing = self.db.query(Site).filter(Site.name == site_data.name).first()
        if existing:
            raise ValueError(f"Site with name '{site_data.name}' already exists")

        site = Site(**site_data.model_dump())
        self.db.add(site)
        self.db.commit()
        self.db.refresh(site)
        return SiteResponse.model_validate(site)

    def get_site(self, site_id: int) -> Optional[SiteResponse]:
        """
        Get a site by ID.

        Args:
            site_id: Site ID.

        Returns:
            Site response or None if not found.
        """
        site = self.db.query(Site).filter(Site.id == site_id).first()
        return SiteResponse.model_validate(site) if site else None

    def get_site_by_name(self, name: str) -> Optional[SiteResponse]:
        """
        Get a site by name.

        Args:
            name: Site name.

        Returns:
            Site response or None if not found.
        """
        site = self.db.query(Site).filter(Site.name == name).first()
        return SiteResponse.model_validate(site) if site else None

    def update_site(self, site_id: int, site_data: SiteUpdate) -> Optional[SiteResponse]:
        """
        Update a site.

        Args:
            site_id: Site ID.
            site_data: Site update data.

        Returns:
            Updated site response or None if not found.
        """
        site = self.db.query(Site).filter(Site.id == site_id).first()
        if not site:
            return None

        update_dict = site_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(site, key, value)

        self.db.commit()
        self.db.refresh(site)
        return SiteResponse.model_validate(site)

    def delete_site(self, site_id: int) -> bool:
        """
        Delete a site.

        Args:
            site_id: Site ID.

        Returns:
            True if deleted, False if not found.
        """
        site = self.db.query(Site).filter(Site.id == site_id).first()
        if not site:
            return False

        self.db.delete(site)
        self.db.commit()
        return True

    def list_sites(
        self,
        status: Optional[AssetStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[SiteResponse]:
        """
        List sites with optional filtering.

        Args:
            status: Filter by status.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of site responses.
        """
        query = self.db.query(Site)

        if status:
            query = query.filter(Site.status == status)

        sites = query.offset(skip).limit(limit).all()
        return [SiteResponse.model_validate(site) for site in sites]

    # ========== EQUIPMENT MANAGEMENT ==========

    def create_equipment(self, equipment_data: EquipmentCreate) -> EquipmentResponse:
        """
        Create new equipment.

        Args:
            equipment_data: Equipment creation data.

        Returns:
            Created equipment response.

        Raises:
            ValueError: If equipment_id already exists or site not found.
        """
        # Validate site exists
        site = self.db.query(Site).filter(Site.id == equipment_data.site_id).first()
        if not site:
            raise ValueError(f"Site with ID {equipment_data.site_id} not found")

        # Check if equipment_id already exists
        existing = (
            self.db.query(Equipment)
            .filter(Equipment.equipment_id == equipment_data.equipment_id)
            .first()
        )
        if existing:
            raise ValueError(f"Equipment with ID '{equipment_data.equipment_id}' already exists")

        equipment = Equipment(**equipment_data.model_dump())
        self.db.add(equipment)
        self.db.commit()
        self.db.refresh(equipment)
        return EquipmentResponse.model_validate(equipment)

    def get_equipment(self, equipment_id: int) -> Optional[EquipmentResponse]:
        """
        Get equipment by ID.

        Args:
            equipment_id: Equipment database ID.

        Returns:
            Equipment response or None if not found.
        """
        equipment = self.db.query(Equipment).filter(Equipment.id == equipment_id).first()
        return EquipmentResponse.model_validate(equipment) if equipment else None

    def get_equipment_by_equipment_id(self, equipment_id: str) -> Optional[EquipmentResponse]:
        """
        Get equipment by equipment_id.

        Args:
            equipment_id: Equipment ID string.

        Returns:
            Equipment response or None if not found.
        """
        equipment = (
            self.db.query(Equipment).filter(Equipment.equipment_id == equipment_id).first()
        )
        return EquipmentResponse.model_validate(equipment) if equipment else None

    def update_equipment(
        self, equipment_id: int, equipment_data: EquipmentUpdate
    ) -> Optional[EquipmentResponse]:
        """
        Update equipment.

        Args:
            equipment_id: Equipment database ID.
            equipment_data: Equipment update data.

        Returns:
            Updated equipment response or None if not found.
        """
        equipment = self.db.query(Equipment).filter(Equipment.id == equipment_id).first()
        if not equipment:
            return None

        update_dict = equipment_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(equipment, key, value)

        self.db.commit()
        self.db.refresh(equipment)
        return EquipmentResponse.model_validate(equipment)

    def delete_equipment(self, equipment_id: int) -> bool:
        """
        Delete equipment.

        Args:
            equipment_id: Equipment database ID.

        Returns:
            True if deleted, False if not found.
        """
        equipment = self.db.query(Equipment).filter(Equipment.id == equipment_id).first()
        if not equipment:
            return False

        self.db.delete(equipment)
        self.db.commit()
        return True

    def list_equipment(
        self,
        site_id: Optional[int] = None,
        equipment_type: Optional[EquipmentType] = None,
        status: Optional[AssetStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[EquipmentResponse]:
        """
        List equipment with optional filtering.

        Args:
            site_id: Filter by site ID.
            equipment_type: Filter by equipment type.
            status: Filter by status.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of equipment responses.
        """
        query = self.db.query(Equipment)

        if site_id:
            query = query.filter(Equipment.site_id == site_id)
        if equipment_type:
            query = query.filter(Equipment.equipment_type == equipment_type)
        if status:
            query = query.filter(Equipment.status == status)

        equipment_list = query.offset(skip).limit(limit).all()
        return [EquipmentResponse.model_validate(eq) for eq in equipment_list]

    # ========== ASSET MANAGEMENT ==========

    def create_asset(self, asset_data: AssetCreate) -> AssetResponse:
        """
        Create a new asset.

        Args:
            asset_data: Asset creation data.

        Returns:
            Created asset response.

        Raises:
            ValueError: If asset_id already exists.
        """
        # Check if asset_id already exists
        existing = self.db.query(Asset).filter(Asset.asset_id == asset_data.asset_id).first()
        if existing:
            raise ValueError(f"Asset with ID '{asset_data.asset_id}' already exists")

        asset = Asset(**asset_data.model_dump())
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return AssetResponse.model_validate(asset)

    def get_asset(self, asset_id: int) -> Optional[AssetResponse]:
        """
        Get an asset by database ID.

        Args:
            asset_id: Asset database ID.

        Returns:
            Asset response or None if not found.
        """
        asset = self.db.query(Asset).filter(Asset.id == asset_id).first()
        return AssetResponse.model_validate(asset) if asset else None

    def get_asset_by_asset_id(self, asset_id: str) -> Optional[AssetResponse]:
        """
        Get an asset by asset_id.

        Args:
            asset_id: Asset ID string.

        Returns:
            Asset response or None if not found.
        """
        asset = self.db.query(Asset).filter(Asset.asset_id == asset_id).first()
        return AssetResponse.model_validate(asset) if asset else None

    def update_asset(self, asset_id: int, asset_data: AssetUpdate) -> Optional[AssetResponse]:
        """
        Update an asset.

        Args:
            asset_id: Asset database ID.
            asset_data: Asset update data.

        Returns:
            Updated asset response or None if not found.
        """
        asset = self.db.query(Asset).filter(Asset.id == asset_id).first()
        if not asset:
            return None

        update_dict = asset_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(asset, key, value)

        self.db.commit()
        self.db.refresh(asset)
        return AssetResponse.model_validate(asset)

    def delete_asset(self, asset_id: int) -> bool:
        """
        Delete an asset.

        Args:
            asset_id: Asset database ID.

        Returns:
            True if deleted, False if not found.
        """
        asset = self.db.query(Asset).filter(Asset.id == asset_id).first()
        if not asset:
            return False

        self.db.delete(asset)
        self.db.commit()
        return True

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        status: Optional[AssetStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AssetResponse]:
        """
        List assets with optional filtering.

        Args:
            asset_type: Filter by asset type.
            status: Filter by status.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of asset responses.
        """
        query = self.db.query(Asset)

        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)
        if status:
            query = query.filter(Asset.status == status)

        assets = query.offset(skip).limit(limit).all()
        return [AssetResponse.model_validate(asset) for asset in assets]

    # ========== PERFORMANCE TRACKING ==========

    def create_performance_record(
        self, record_data: PerformanceRecordCreate
    ) -> PerformanceRecordResponse:
        """
        Create a performance record.

        Args:
            record_data: Performance record creation data.

        Returns:
            Created performance record response.

        Raises:
            ValueError: If site or equipment not found.
        """
        # Validate site exists
        site = self.db.query(Site).filter(Site.id == record_data.site_id).first()
        if not site:
            raise ValueError(f"Site with ID {record_data.site_id} not found")

        # Validate equipment if provided
        if record_data.equipment_id:
            equipment = (
                self.db.query(Equipment).filter(Equipment.id == record_data.equipment_id).first()
            )
            if not equipment:
                raise ValueError(f"Equipment with ID {record_data.equipment_id} not found")

        record = PerformanceRecord(**record_data.model_dump())
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return PerformanceRecordResponse.model_validate(record)

    def get_performance_record(self, record_id: int) -> Optional[PerformanceRecordResponse]:
        """
        Get a performance record by ID.

        Args:
            record_id: Performance record ID.

        Returns:
            Performance record response or None if not found.
        """
        record = self.db.query(PerformanceRecord).filter(PerformanceRecord.id == record_id).first()
        return PerformanceRecordResponse.model_validate(record) if record else None

    def list_performance_records(
        self,
        site_id: Optional[int] = None,
        equipment_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PerformanceRecordResponse]:
        """
        List performance records with optional filtering.

        Args:
            site_id: Filter by site ID.
            equipment_id: Filter by equipment ID.
            start_date: Filter by start date.
            end_date: Filter by end date.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of performance record responses.
        """
        query = self.db.query(PerformanceRecord)

        if site_id:
            query = query.filter(PerformanceRecord.site_id == site_id)
        if equipment_id:
            query = query.filter(PerformanceRecord.equipment_id == equipment_id)
        if start_date:
            query = query.filter(PerformanceRecord.timestamp >= start_date)
        if end_date:
            query = query.filter(PerformanceRecord.timestamp <= end_date)

        query = query.order_by(PerformanceRecord.timestamp.desc())
        records = query.offset(skip).limit(limit).all()
        return [PerformanceRecordResponse.model_validate(record) for record in records]

    # ========== CORE METHODS: SITE INVENTORY, EQUIPMENT TRACKING, PERFORMANCE HISTORY ==========

    def site_inventory(
        self,
        status: Optional[AssetStatus] = None,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive site inventory with summary statistics.

        Args:
            status: Filter by status (None for all).
            include_summary: Include summary statistics.

        Returns:
            Dictionary containing sites list and optional summary.
        """
        sites = self.list_sites(status=status, limit=1000)

        result: Dict[str, Any] = {"sites": sites}

        if include_summary:
            total_sites = len(sites)
            total_capacity = sum(site.capacity_kw for site in sites)
            avg_capacity = total_capacity / total_sites if total_sites > 0 else 0

            # Group by status
            status_counts: Dict[str, int] = {}
            for site in sites:
                status_str = site.status.value
                status_counts[status_str] = status_counts.get(status_str, 0) + 1

            summary = SiteInventorySummary(
                total_sites=total_sites,
                total_capacity_kw=total_capacity,
                sites_by_status=status_counts,
                average_capacity_kw=avg_capacity,
            )
            result["summary"] = summary

        return result

    def equipment_tracking(
        self,
        site_id: Optional[int] = None,
        equipment_type: Optional[EquipmentType] = None,
        status: Optional[AssetStatus] = None,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive equipment tracking with summary statistics.

        Args:
            site_id: Filter by site ID.
            equipment_type: Filter by equipment type.
            status: Filter by status.
            include_summary: Include summary statistics.

        Returns:
            Dictionary containing equipment list and optional summary.
        """
        equipment_list = self.list_equipment(
            site_id=site_id,
            equipment_type=equipment_type,
            status=status,
            limit=1000,
        )

        result: Dict[str, Any] = {"equipment": equipment_list}

        if include_summary:
            total_equipment = len(equipment_list)

            # Group by type
            type_counts: Dict[str, int] = {}
            for eq in equipment_list:
                type_str = eq.equipment_type.value
                type_counts[type_str] = type_counts.get(type_str, 0) + 1

            # Group by status
            status_counts: Dict[str, int] = {}
            for eq in equipment_list:
                status_str = eq.status.value
                status_counts[status_str] = status_counts.get(status_str, 0) + 1

            # Calculate total rated power
            total_power_w = sum(
                eq.rated_power_w for eq in equipment_list if eq.rated_power_w is not None
            )
            total_power_kw = total_power_w / 1000

            # Calculate average efficiency
            efficiencies = [
                eq.efficiency_percent
                for eq in equipment_list
                if eq.efficiency_percent is not None
            ]
            avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else None

            summary = EquipmentInventorySummary(
                total_equipment=total_equipment,
                equipment_by_type=type_counts,
                equipment_by_status=status_counts,
                total_rated_power_kw=total_power_kw,
                average_efficiency_percent=avg_efficiency,
            )
            result["summary"] = summary

        return result

    def performance_history(
        self,
        site_id: Optional[int] = None,
        equipment_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Get performance history with summary statistics.

        Args:
            site_id: Filter by site ID.
            equipment_id: Filter by equipment ID.
            start_date: Filter by start date.
            end_date: Filter by end date.
            include_summary: Include summary statistics.

        Returns:
            Dictionary containing performance records and optional summary.
        """
        records = self.list_performance_records(
            site_id=site_id,
            equipment_id=equipment_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        result: Dict[str, Any] = {"records": records}

        if include_summary and records:
            total_records = len(records)

            # Calculate date range
            timestamps = [r.timestamp for r in records]
            date_range = {"start": min(timestamps), "end": max(timestamps)}

            # Calculate totals and averages
            total_energy = sum(
                r.energy_generated_kwh for r in records if r.energy_generated_kwh is not None
            )

            power_outputs = [r.power_output_kw for r in records if r.power_output_kw is not None]
            avg_power = sum(power_outputs) / len(power_outputs) if power_outputs else 0

            efficiencies = [
                r.efficiency_percent for r in records if r.efficiency_percent is not None
            ]
            avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else None

            capacity_factors = [
                r.capacity_factor_percent for r in records if r.capacity_factor_percent is not None
            ]
            avg_capacity_factor = (
                sum(capacity_factors) / len(capacity_factors) if capacity_factors else None
            )

            summary = PerformanceHistorySummary(
                total_records=total_records,
                date_range=date_range,
                total_energy_kwh=total_energy,
                average_power_kw=avg_power,
                average_efficiency_percent=avg_efficiency,
                average_capacity_factor_percent=avg_capacity_factor,
            )
            result["summary"] = summary

        return result
