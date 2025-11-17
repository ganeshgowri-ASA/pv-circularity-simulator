"""Integration tests for RepairOptimizer end-to-end workflows.

These tests demonstrate complete workflows combining multiple RepairOptimizer
features to simulate real-world usage scenarios.
"""

from datetime import datetime, timedelta

import pytest

from pv_simulator.managers.repair_optimizer import RepairOptimizer
from pv_simulator.models.maintenance import (
    ComponentHealth,
    ComponentStatus,
    MaintenancePriority,
    MaintenanceType,
    RepairTask,
    SparePart,
)


class TestCompleteMaintenanceWorkflow:
    """Test complete maintenance workflow from diagnosis to scheduling."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with pre-configured spare parts inventory."""
        opt = RepairOptimizer(labor_rate=75.0, overhead_rate=0.15)

        # Add spare parts inventory
        spare_parts = [
            SparePart(
                part_id="BP-DIODE-001",
                part_name="bypass_diode",
                part_number="BD-12V-10A",
                category="electrical",
                quantity_available=100,
                unit_cost=15.0,
                lead_time_days=7,
                reorder_point=30,
                reorder_quantity=100,
                supplier="ElectroSupply Inc",
            ),
            SparePart(
                part_id="JB-001",
                part_name="junction_box",
                part_number="JB-PV-STANDARD",
                category="electrical",
                quantity_available=50,
                unit_cost=25.0,
                lead_time_days=10,
                reorder_point=20,
                reorder_quantity=50,
                supplier="ElectroSupply Inc",
            ),
            SparePart(
                part_id="CAP-001",
                part_name="capacitor",
                part_number="CAP-450V-100uF",
                category="electrical",
                quantity_available=200,
                unit_cost=50.0,
                lead_time_days=5,
                reorder_point=50,
                reorder_quantity=200,
                supplier="ComponentWorld",
            ),
            SparePart(
                part_id="FAN-001",
                part_name="cooling_fan",
                part_number="CF-120MM-12V",
                category="thermal",
                quantity_available=25,
                unit_cost=40.0,
                lead_time_days=7,
                reorder_point=15,
                reorder_quantity=30,
                supplier="ThermalParts Ltd",
            ),
        ]

        for part in spare_parts:
            opt.add_spare_part(part)

        return opt

    def test_full_diagnosis_to_repair_workflow(self, optimizer):
        """Test complete workflow: diagnosis -> cost estimation -> scheduling."""
        # Step 1: Diagnose faults in multiple components
        panel_fault = optimizer.fault_diagnosis(
            component_id="PANEL-A12",
            component_type="panel",
            performance_data={
                "voltage": 25.0,
                "current": 7.0,
                "efficiency": 0.14,
                "temperature": 48.0,
            },
            baseline_data={
                "voltage": 30.0,
                "current": 8.0,
                "efficiency": 0.18,
                "temperature": 45.0,
            },
        )

        inverter_fault = optimizer.fault_diagnosis(
            component_id="INV-001",
            component_type="inverter",
            performance_data={
                "efficiency": 0.88,
                "temperature": 78.0,
                "voltage_in": 600.0,
                "voltage_out": 240.0,
            },
            baseline_data={
                "efficiency": 0.96,
                "temperature": 50.0,
                "voltage_in": 600.0,
                "voltage_out": 240.0,
            },
        )

        assert panel_fault is not None
        assert inverter_fault is not None

        # Step 2: Generate cost estimates
        panel_estimate = optimizer.repair_cost_estimation(panel_fault)
        inverter_estimate = optimizer.repair_cost_estimation(inverter_fault)

        assert panel_estimate.total_cost > 0
        assert inverter_estimate.total_cost > 0

        # Step 3: Create repair tasks
        tasks = [
            RepairTask(
                fault_id=panel_fault.fault_id,
                component_id=panel_fault.component_id,
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.HIGH,
                estimated_duration_hours=panel_estimate.labor_hours,
                estimated_cost=panel_estimate.total_cost,
                required_skills=["electrical", "pv_systems"],
            ),
            RepairTask(
                fault_id=inverter_fault.fault_id,
                component_id=inverter_fault.component_id,
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.MEDIUM,
                estimated_duration_hours=inverter_estimate.labor_hours,
                estimated_cost=inverter_estimate.total_cost,
                required_skills=["electrical", "inverters"],
            ),
        ]

        # Step 4: Create optimized schedule
        start_date = datetime.now()
        end_date = start_date + timedelta(days=14)

        schedule = optimizer.maintenance_scheduling(
            tasks=tasks,
            start_date=start_date,
            end_date=end_date,
            optimization_objective="maximize_priority",
        )

        assert len(schedule.tasks) == 2
        assert schedule.total_estimated_cost > 0
        assert schedule.constraints_satisfied

        # Step 5: Verify all tasks are scheduled
        for task in schedule.tasks:
            assert task.scheduled_start is not None
            assert task.scheduled_end is not None

    def test_spare_parts_integration_with_repairs(self, optimizer):
        """Test spare parts management in context of multiple repairs."""
        # Diagnose multiple faults that will require spare parts
        faults = []
        for i in range(5):
            fault = optimizer.fault_diagnosis(
                component_id=f"PANEL-{i:03d}",
                component_type="panel",
                performance_data={
                    "voltage": 24.0,
                    "current": 7.0,
                    "efficiency": 0.15,
                },
                baseline_data={
                    "voltage": 30.0,
                    "current": 8.0,
                    "efficiency": 0.18,
                },
            )
            if fault:
                faults.append(fault)

        assert len(faults) > 0

        # Check spare parts availability
        initial_inventory = optimizer.spare_parts_management()
        initial_status = initial_inventory["inventory_status"]

        # Simulate parts being reserved for repairs
        for fault in faults:
            estimate = optimizer.repair_cost_estimation(fault)
            # Parts would be reserved here in a real system

        # Check if reordering is needed
        parts_check = optimizer.spare_parts_management(forecast_days=60)

        assert "reorder_recommendations" in parts_check
        assert "inventory_status" in parts_check

    def test_multiple_component_health_tracking(self, optimizer):
        """Test tracking health of multiple components over time."""
        # Add component health records
        components = [
            ComponentHealth(
                component_id="PANEL-001",
                component_type="panel",
                status=ComponentStatus.OPERATIONAL,
                health_score=0.95,
                degradation_rate=0.005,
                performance_metrics={"efficiency": 0.18, "voltage": 30.0},
            ),
            ComponentHealth(
                component_id="PANEL-002",
                component_type="panel",
                status=ComponentStatus.DEGRADED,
                health_score=0.75,
                degradation_rate=0.015,
                performance_metrics={"efficiency": 0.15, "voltage": 28.0},
            ),
            ComponentHealth(
                component_id="INV-001",
                component_type="inverter",
                status=ComponentStatus.OPERATIONAL,
                health_score=0.92,
                degradation_rate=0.008,
                performance_metrics={"efficiency": 0.94, "temperature": 52.0},
            ),
        ]

        for health in components:
            optimizer.update_component_health(health)

        assert len(optimizer.component_health) == 3

        # Diagnose fault in degraded panel
        fault = optimizer.fault_diagnosis(
            component_id="PANEL-002",
            component_type="panel",
            performance_data={"efficiency": 0.14, "voltage": 27.0},
            baseline_data={"efficiency": 0.18, "voltage": 30.0},
        )

        if fault:
            # Verify fault is linked to component health
            component = optimizer.component_health["PANEL-002"]
            assert fault.fault_id in component.current_faults

    def test_preventive_maintenance_scheduling(self, optimizer):
        """Test scheduling preventive maintenance alongside corrective repairs."""
        # Create mix of corrective and preventive tasks
        tasks = [
            # Corrective repair (high priority)
            RepairTask(
                component_id="INV-001",
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.EMERGENCY,
                estimated_duration_hours=3.0,
                estimated_cost=400.0,
            ),
            # Preventive maintenance (lower priority)
            RepairTask(
                component_id="PANEL-ARRAY-A",
                task_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.ROUTINE,
                estimated_duration_hours=6.0,
                estimated_cost=300.0,
            ),
            # Inspection (medium priority)
            RepairTask(
                component_id="ALL-PANELS",
                task_type=MaintenanceType.INSPECTION,
                priority=MaintenancePriority.MEDIUM,
                estimated_duration_hours=4.0,
                estimated_cost=200.0,
            ),
        ]

        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)

        schedule = optimizer.maintenance_scheduling(
            tasks=tasks,
            start_date=start_date,
            end_date=end_date,
            optimization_objective="maximize_priority",
        )

        # Emergency task should be scheduled first
        assert schedule.tasks[0].priority == MaintenancePriority.EMERGENCY
        assert schedule.tasks[0].task_type == MaintenanceType.CORRECTIVE


class TestScalabilityScenarios:
    """Test optimizer performance with larger datasets."""

    def test_large_fault_diagnosis_batch(self):
        """Test diagnosing faults across many components."""
        optimizer = RepairOptimizer()

        faults = []
        for i in range(50):
            fault = optimizer.fault_diagnosis(
                component_id=f"PANEL-{i:03d}",
                component_type="panel",
                performance_data={
                    "voltage": 25.0 + (i % 5),
                    "current": 7.0,
                    "efficiency": 0.15,
                    "temperature": 45.0 + (i % 10),
                },
            )
            if fault:
                faults.append(fault)

        assert len(optimizer.active_faults) > 0

    def test_large_maintenance_schedule(self):
        """Test scheduling many maintenance tasks."""
        optimizer = RepairOptimizer()

        # Create 30 repair tasks
        tasks = []
        for i in range(30):
            priority_cycle = [
                MaintenancePriority.EMERGENCY,
                MaintenancePriority.HIGH,
                MaintenancePriority.MEDIUM,
                MaintenancePriority.LOW,
                MaintenancePriority.ROUTINE,
            ]

            task = RepairTask(
                component_id=f"COMP-{i:03d}",
                task_type=MaintenanceType.CORRECTIVE,
                priority=priority_cycle[i % 5],
                estimated_duration_hours=2.0 + (i % 4),
                estimated_cost=150.0 + (i * 10),
            )
            tasks.append(task)

        start_date = datetime.now()
        end_date = start_date + timedelta(days=60)

        schedule = optimizer.maintenance_scheduling(
            tasks=tasks,
            start_date=start_date,
            end_date=end_date,
            max_daily_hours=8.0,
        )

        assert len(schedule.tasks) > 0
        assert schedule.total_estimated_hours > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_repair_with_no_spare_parts(self):
        """Test cost estimation when no spare parts are available."""
        optimizer = RepairOptimizer()

        fault = optimizer.fault_diagnosis(
            component_id="PANEL-999",
            component_type="panel",
            performance_data={"temperature": 95.0},
        )

        assert fault is not None

        # Estimate without parts
        estimate = optimizer.repair_cost_estimation(fault, include_parts=False)
        assert estimate.parts_cost == 0.0

    def test_scheduling_with_tight_deadline(self):
        """Test scheduling when deadline is very tight."""
        optimizer = RepairOptimizer()

        tasks = [
            RepairTask(
                component_id="URGENT-001",
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.EMERGENCY,
                estimated_duration_hours=6.0,
                estimated_cost=500.0,
            ),
            RepairTask(
                component_id="URGENT-002",
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.EMERGENCY,
                estimated_duration_hours=5.0,
                estimated_cost=450.0,
            ),
        ]

        # Only one day to complete 11 hours of work (with 8 hour days)
        start_date = datetime.now()
        end_date = start_date + timedelta(days=2)

        schedule = optimizer.maintenance_scheduling(
            tasks=tasks, start_date=start_date, end_date=end_date, max_daily_hours=8.0
        )

        # Should schedule across 2 days
        assert len(schedule.tasks) <= len(tasks)

    def test_fault_clearing_workflow(self):
        """Test clearing faults after repair completion."""
        optimizer = RepairOptimizer()

        # Diagnose fault
        fault = optimizer.fault_diagnosis(
            component_id="TEST-001",
            component_type="panel",
            performance_data={"temperature": 88.0},
        )

        assert fault is not None
        assert fault in optimizer.active_faults

        # Clear fault after repair
        cleared = optimizer.clear_fault(fault.fault_id)
        assert cleared is True
        assert fault not in optimizer.active_faults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
