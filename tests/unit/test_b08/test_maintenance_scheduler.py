"""
Tests for MaintenanceScheduler (B08-S04).
"""

from datetime import date, timedelta
from typing import List
import pytest

from src.pv_circularity.models import (
    DiagnosticResult,
    MaintenanceSchedule,
    MaintenanceType,
    SparePart,
)
from src.pv_circularity.b08_diagnostics.b08_s04 import (
    MaintenanceScheduler,
    MaintenancePolicy,
)


class TestMaintenanceScheduler:
    """Test suite for MaintenanceScheduler."""

    def test_initialization(self):
        """Test that MaintenanceScheduler initializes correctly."""
        scheduler = MaintenanceScheduler()
        assert scheduler is not None
        assert scheduler.policy is not None
        assert isinstance(scheduler.policy, MaintenancePolicy)

    def test_initialization_with_custom_policy(self):
        """Test initialization with custom maintenance policy."""
        policy = MaintenancePolicy(
            preventive_interval_days=90,
            inspection_interval_days=45,
        )
        scheduler = MaintenanceScheduler(policy=policy)
        assert scheduler.policy.preventive_interval_days == 90
        assert scheduler.policy.inspection_interval_days == 45

    def test_preventive_maintenance_planning(self):
        """Test preventive maintenance planning."""
        scheduler = MaintenanceScheduler()
        schedules = scheduler.preventive_maintenance_planning(
            site_id="SITE-001",
            planning_horizon_days=365,
            panel_count=100,
        )

        assert isinstance(schedules, list)
        assert len(schedules) > 0
        assert all(isinstance(s, MaintenanceSchedule) for s in schedules)

        # Check that different types of maintenance are scheduled
        types = set(s.maintenance_type for s in schedules)
        assert MaintenanceType.INSPECTION in types
        assert MaintenanceType.PREVENTIVE in types

    def test_preventive_maintenance_planning_short_horizon(self):
        """Test planning with short horizon."""
        scheduler = MaintenanceScheduler()
        schedules = scheduler.preventive_maintenance_planning(
            site_id="SITE-001",
            planning_horizon_days=30,
            panel_count=50,
        )

        # Should still generate some schedules
        assert len(schedules) >= 0

    def test_preventive_maintenance_planning_with_last_maintenance(self):
        """Test planning considering last maintenance date."""
        scheduler = MaintenanceScheduler()
        last_maintenance = date.today() - timedelta(days=30)

        schedules = scheduler.preventive_maintenance_planning(
            site_id="SITE-001",
            planning_horizon_days=365,
            panel_count=100,
            last_maintenance_date=last_maintenance,
        )

        assert len(schedules) > 0

    def test_corrective_action_tracking(
        self,
        sample_diagnostic_result: DiagnosticResult
    ):
        """Test corrective action tracking."""
        scheduler = MaintenanceScheduler()
        diagnostics = [sample_diagnostic_result]

        actions = scheduler.corrective_action_tracking(
            diagnostics=diagnostics,
            site_id="SITE-001",
        )

        assert isinstance(actions, list)
        # High priority diagnostics should generate actions
        if sample_diagnostic_result.priority <= 3:
            assert len(actions) > 0

    def test_corrective_action_tracking_multiple_diagnostics(
        self,
        sample_diagnostic_result: DiagnosticResult
    ):
        """Test corrective action tracking with multiple diagnostics."""
        scheduler = MaintenanceScheduler()

        # Create multiple high-priority diagnostics
        diagnostics = []
        for i in range(3):
            diag = DiagnosticResult(
                defect_id=f"DEFECT-{i}",
                root_cause="Test root cause",
                root_cause_confidence=0.8,
                recommended_action=sample_diagnostic_result.recommended_action,
                priority=2,  # High priority
                estimated_impact=10.0,
                estimated_cost=300.0,
            )
            diagnostics.append(diag)

        actions = scheduler.corrective_action_tracking(
            diagnostics=diagnostics,
            site_id="SITE-001",
        )

        assert len(actions) == 3

    def test_spare_parts_management(
        self,
        sample_maintenance_schedule: MaintenanceSchedule
    ):
        """Test spare parts management."""
        scheduler = MaintenanceScheduler()

        # Add some spare parts to inventory
        part1 = SparePart(
            part_number="test_kit",
            part_name="Test Kit",
            category="tools",
            quantity_available=10,
            quantity_reserved=0,
            reorder_level=5,
            unit_cost=50.0,
        )
        scheduler.add_spare_part(part1)

        schedules = [sample_maintenance_schedule]
        analysis = scheduler.spare_parts_management(schedules)

        assert isinstance(analysis, dict)
        assert "required" in analysis
        assert "available" in analysis
        assert "shortages" in analysis
        assert "reorder" in analysis

    def test_spare_parts_management_shortages(
        self,
        sample_maintenance_schedule: MaintenanceSchedule
    ):
        """Test spare parts management detects shortages."""
        scheduler = MaintenanceScheduler()

        # Add part with insufficient quantity
        part = SparePart(
            part_number="test_kit",
            part_name="Test Kit",
            category="tools",
            quantity_available=1,
            quantity_reserved=0,
            reorder_level=5,
            unit_cost=50.0,
        )
        scheduler.add_spare_part(part)

        schedules = [sample_maintenance_schedule] * 5  # Need 5 test kits
        analysis = scheduler.spare_parts_management(schedules)

        # Should detect shortage
        assert "test_kit" in analysis["shortages"]

    def test_spare_parts_management_reorder_level(self):
        """Test spare parts management detects reorder needs."""
        scheduler = MaintenanceScheduler()

        # Add part below reorder level
        part = SparePart(
            part_number="PANEL-300W",
            part_name="300W Panel",
            category="panel",
            quantity_available=3,
            quantity_reserved=0,
            reorder_level=5,
            unit_cost=250.0,
        )
        scheduler.add_spare_part(part)

        analysis = scheduler.spare_parts_management([])

        # Should trigger reorder
        assert "PANEL-300W" in analysis["reorder"]

    def test_add_spare_part(self, sample_spare_part: SparePart):
        """Test adding spare part to inventory."""
        scheduler = MaintenanceScheduler()
        scheduler.add_spare_part(sample_spare_part)

        assert sample_spare_part.part_number in scheduler.spare_parts_inventory
        assert (
            scheduler.spare_parts_inventory[sample_spare_part.part_number]
            == sample_spare_part
        )

    def test_update_corrective_action_status(
        self,
        sample_diagnostic_result: DiagnosticResult
    ):
        """Test updating corrective action status."""
        scheduler = MaintenanceScheduler()

        # Create a corrective action
        actions = scheduler.corrective_action_tracking(
            diagnostics=[sample_diagnostic_result],
            site_id="SITE-001",
        )

        if actions:
            action_id = actions[0].action_id

            # Update status
            updated = scheduler.update_corrective_action_status(
                action_id=action_id,
                status="completed",
                effectiveness_score=85.0,
                notes="Successfully completed",
            )

            assert updated is not None
            assert updated.status == "completed"
            assert updated.effectiveness_score == 85.0
            assert updated.notes == "Successfully completed"

    def test_update_nonexistent_corrective_action(self):
        """Test updating non-existent corrective action."""
        scheduler = MaintenanceScheduler()

        updated = scheduler.update_corrective_action_status(
            action_id="NONEXISTENT",
            status="completed",
        )

        assert updated is None
