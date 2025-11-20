"""
Pytest configuration and shared fixtures for PV Circularity Simulator tests.
"""

from datetime import datetime, date, timedelta
from typing import List
import pytest

from src.pv_circularity.models import (
    Defect,
    DefectType,
    DefectSeverity,
    Coordinates,
    DiagnosticResult,
    RecommendedAction,
    MaintenanceSchedule,
    MaintenanceType,
    MaintenancePriority,
    Technician,
    WorkOrder,
    WorkOrderStatus,
    SparePart,
    GeoLocation,
)


@pytest.fixture
def sample_coordinates() -> Coordinates:
    """Sample coordinates for testing."""
    return Coordinates(x=100.0, y=200.0, width=50.0, height=50.0)


@pytest.fixture
def sample_defect(sample_coordinates: Coordinates) -> Defect:
    """Sample defect for testing."""
    return Defect(
        type=DefectType.CRACK,
        severity=DefectSeverity.MEDIUM,
        location=sample_coordinates,
        confidence=0.85,
        panel_id="PANEL-001",
        module_id="MODULE-001",
        description="Micro-crack detected in cell",
        estimated_power_loss=5.0,
    )


@pytest.fixture
def sample_defects(sample_coordinates: Coordinates) -> List[Defect]:
    """Multiple sample defects for testing."""
    defects = []

    # Create defects of various types and severities
    types_severities = [
        (DefectType.CRACK, DefectSeverity.HIGH, 8.0),
        (DefectType.HOTSPOT, DefectSeverity.CRITICAL, 15.0),
        (DefectType.DELAMINATION, DefectSeverity.MEDIUM, 6.0),
        (DefectType.SOILING, DefectSeverity.LOW, 2.0),
        (DefectType.PID, DefectSeverity.HIGH, 12.0),
    ]

    for idx, (defect_type, severity, power_loss) in enumerate(types_severities):
        defect = Defect(
            type=defect_type,
            severity=severity,
            location=Coordinates(
                x=100.0 + idx * 50,
                y=200.0 + idx * 30,
                width=50.0,
                height=50.0,
            ),
            confidence=0.75 + idx * 0.05,
            panel_id=f"PANEL-{idx:03d}",
            description=f"Test {defect_type.value}",
            estimated_power_loss=power_loss,
        )
        defects.append(defect)

    return defects


@pytest.fixture
def sample_diagnostic_result(sample_defect: Defect) -> DiagnosticResult:
    """Sample diagnostic result for testing."""
    return DiagnosticResult(
        defect_id=sample_defect.id,
        defect=sample_defect,
        root_cause="Mechanical stress during installation",
        root_cause_confidence=0.80,
        recommended_action=RecommendedAction.SCHEDULE_INSPECTION,
        priority=2,
        estimated_impact=5.0,
        estimated_cost=250.0,
        time_to_failure=60,
        analysis_notes="Recommend monitoring for progression",
    )


@pytest.fixture
def sample_maintenance_schedule() -> MaintenanceSchedule:
    """Sample maintenance schedule for testing."""
    return MaintenanceSchedule(
        schedule_name="Quarterly Inspection",
        site_id="SITE-001",
        maintenance_type=MaintenanceType.INSPECTION,
        priority=MaintenancePriority.MEDIUM,
        scheduled_date=date.today() + timedelta(days=30),
        estimated_duration_hours=4.0,
        required_parts=["test_kit", "cleaning_solution"],
        required_skills=["electrical_inspection", "visual_inspection"],
        description="Quarterly comprehensive inspection",
    )


@pytest.fixture
def sample_technician() -> Technician:
    """Sample technician for testing."""
    return Technician(
        technician_id="TECH-001",
        name="John Doe",
        skills=["electrical_maintenance", "mechanical_maintenance", "safety_certified"],
        availability=True,
        contact_info={"email": "john.doe@example.com", "phone": "555-0100"},
        hourly_rate=75.0,
    )


@pytest.fixture
def sample_work_order() -> WorkOrder:
    """Sample work order for testing."""
    scheduled_start = datetime.utcnow() + timedelta(days=7)
    return WorkOrder(
        work_order_number="WO-001",
        title="Panel Replacement",
        site_id="SITE-001",
        maintenance_type=MaintenanceType.CORRECTIVE,
        priority=MaintenancePriority.HIGH,
        status=WorkOrderStatus.DRAFT,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_start + timedelta(hours=4),
        description="Replace damaged panel PANEL-001",
        required_parts={"panel_300w": 1, "connectors": 4},
        estimated_cost=500.0,
    )


@pytest.fixture
def sample_spare_part() -> SparePart:
    """Sample spare part for testing."""
    return SparePart(
        part_number="PANEL-300W",
        part_name="300W Solar Panel",
        description="Standard 300W monocrystalline panel",
        category="panel",
        quantity_available=10,
        quantity_reserved=2,
        reorder_level=5,
        unit_cost=250.0,
        supplier="Solar Supplier Inc.",
        lead_time_days=14,
        location="Warehouse A, Bay 3",
    )


@pytest.fixture
def sample_geo_location() -> GeoLocation:
    """Sample geographic location for testing."""
    return GeoLocation(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=100.0,
        site_name="San Francisco Test Site",
    )
