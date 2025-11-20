"""Comprehensive unit tests for the RepairOptimizer class.

Tests cover all major functionality including fault diagnosis, cost estimation,
maintenance scheduling, and spare parts management.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pv_simulator.managers.repair_optimizer import RepairOptimizer
from pv_simulator.models.maintenance import (
    ComponentHealth,
    ComponentStatus,
    Fault,
    FaultSeverity,
    FaultType,
    MaintenancePriority,
    MaintenanceType,
    RepairTask,
    SparePart,
)


class TestRepairOptimizerInitialization:
    """Test RepairOptimizer initialization and configuration."""

    def test_default_initialization(self):
        """Test creating optimizer with default parameters."""
        optimizer = RepairOptimizer()
        assert optimizer.labor_rate == 75.0
        assert optimizer.overhead_rate == 0.15
        assert optimizer.fault_detection_threshold == 0.6
        assert len(optimizer.spare_parts) == 0
        assert len(optimizer.active_faults) == 0

    def test_custom_initialization(self):
        """Test creating optimizer with custom parameters."""
        optimizer = RepairOptimizer(
            labor_rate=100.0, overhead_rate=0.20, fault_detection_threshold=0.7
        )
        assert optimizer.labor_rate == 100.0
        assert optimizer.overhead_rate == 0.20
        assert optimizer.fault_detection_threshold == 0.7

    def test_invalid_labor_rate(self):
        """Test that negative labor rate raises error."""
        with pytest.raises(ValueError, match="Labor rate must be non-negative"):
            RepairOptimizer(labor_rate=-10.0)

    def test_invalid_overhead_rate(self):
        """Test that invalid overhead rate raises error."""
        with pytest.raises(ValueError, match="Overhead rate must be between"):
            RepairOptimizer(overhead_rate=1.5)

    def test_invalid_threshold(self):
        """Test that invalid detection threshold raises error."""
        with pytest.raises(ValueError, match="Fault detection threshold must be between"):
            RepairOptimizer(fault_detection_threshold=1.5)


class TestFaultDiagnosis:
    """Test fault diagnosis functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return RepairOptimizer(fault_detection_threshold=0.5)

    def test_diagnose_efficiency_degradation(self, optimizer):
        """Test detection of efficiency degradation fault."""
        performance_data = {
            "voltage": 28.0,
            "current": 7.5,
            "efficiency": 0.15,
            "temperature": 45.0,
        }
        baseline_data = {
            "voltage": 30.0,
            "current": 8.0,
            "efficiency": 0.18,
            "temperature": 45.0,
        }

        fault = optimizer.fault_diagnosis(
            component_id="PANEL-001",
            component_type="panel",
            performance_data=performance_data,
            baseline_data=baseline_data,
        )

        assert fault is not None
        assert fault.component_id == "PANEL-001"
        assert fault.fault_type == FaultType.DEGRADATION
        assert fault.diagnosis_confidence >= 0.5
        assert "efficiency_drop" in fault.symptoms

    def test_diagnose_thermal_fault(self, optimizer):
        """Test detection of thermal fault."""
        performance_data = {
            "voltage": 30.0,
            "current": 8.0,
            "efficiency": 0.18,
            "temperature": 85.0,  # Very high temperature
        }

        fault = optimizer.fault_diagnosis(
            component_id="INV-001",
            component_type="inverter",
            performance_data=performance_data,
        )

        assert fault is not None
        assert fault.fault_type == FaultType.THERMAL
        assert "temperature_high" in fault.symptoms
        assert fault.symptoms["temperature_high"] == 85.0

    def test_diagnose_electrical_fault(self, optimizer):
        """Test detection of electrical fault (voltage deviation)."""
        performance_data = {
            "voltage": 25.0,  # Significant voltage drop
            "current": 8.0,
            "efficiency": 0.17,
            "temperature": 45.0,
        }
        baseline_data = {
            "voltage": 30.0,
            "current": 8.0,
            "efficiency": 0.18,
            "temperature": 45.0,
        }

        fault = optimizer.fault_diagnosis(
            component_id="PANEL-002",
            component_type="panel",
            performance_data=performance_data,
            baseline_data=baseline_data,
        )

        assert fault is not None
        assert fault.fault_type == FaultType.ELECTRICAL
        assert "voltage_deviation" in fault.symptoms

    def test_no_fault_detected(self, optimizer):
        """Test that no fault is returned when performance is normal."""
        performance_data = {
            "voltage": 29.8,
            "current": 8.1,
            "efficiency": 0.179,
            "temperature": 46.0,
        }
        baseline_data = {
            "voltage": 30.0,
            "current": 8.0,
            "efficiency": 0.18,
            "temperature": 45.0,
        }

        fault = optimizer.fault_diagnosis(
            component_id="PANEL-003",
            component_type="panel",
            performance_data=performance_data,
            baseline_data=baseline_data,
        )

        # Should return None or a fault with very low confidence
        if fault is not None:
            assert fault.diagnosis_confidence < optimizer.fault_detection_threshold

    def test_fault_with_historical_data(self, optimizer):
        """Test fault diagnosis using historical trend data."""
        performance_data = {"efficiency": 0.15}
        historical_data = [
            {"efficiency": 0.18},
            {"efficiency": 0.175},
            {"efficiency": 0.17},
            {"efficiency": 0.165},
            {"efficiency": 0.16},
            {"efficiency": 0.155},
        ]

        fault = optimizer.fault_diagnosis(
            component_id="PANEL-004",
            component_type="panel",
            performance_data=performance_data,
            historical_data=historical_data,
        )

        assert fault is not None
        assert "degradation_rate" in fault.symptoms

    def test_empty_component_id_raises_error(self, optimizer):
        """Test that empty component ID raises ValueError."""
        with pytest.raises(ValueError, match="component_id cannot be empty"):
            optimizer.fault_diagnosis(
                component_id="",
                component_type="panel",
                performance_data={"voltage": 30.0},
            )

    def test_empty_performance_data_raises_error(self, optimizer):
        """Test that empty performance data raises ValueError."""
        with pytest.raises(ValueError, match="performance_data cannot be empty"):
            optimizer.fault_diagnosis(
                component_id="PANEL-001", component_type="panel", performance_data={}
            )

    def test_fault_added_to_active_faults(self, optimizer):
        """Test that diagnosed fault is added to active faults list."""
        initial_count = len(optimizer.active_faults)

        performance_data = {"temperature": 90.0}
        fault = optimizer.fault_diagnosis(
            component_id="INV-002",
            component_type="inverter",
            performance_data=performance_data,
        )

        assert fault is not None
        assert len(optimizer.active_faults) == initial_count + 1
        assert fault in optimizer.active_faults


class TestRepairCostEstimation:
    """Test repair cost estimation functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return RepairOptimizer(labor_rate=80.0, overhead_rate=0.15)

    @pytest.fixture
    def sample_fault(self):
        """Create a sample fault for testing."""
        return Fault(
            component_id="PANEL-001",
            component_type="panel",
            fault_type=FaultType.ELECTRICAL,
            severity=FaultSeverity.MEDIUM,
            description="Electrical fault in panel",
            symptoms={"voltage_deviation": 0.15},
            diagnosis_confidence=0.85,
        )

    def test_basic_cost_estimation(self, optimizer, sample_fault):
        """Test basic cost estimation for a fault."""
        estimate = optimizer.repair_cost_estimation(sample_fault)

        assert estimate.fault_id == sample_fault.fault_id
        assert estimate.component_id == sample_fault.component_id
        assert estimate.labor_hours > 0
        assert estimate.labor_cost > 0
        assert estimate.total_cost > 0
        assert estimate.confidence_level > 0

    def test_cost_estimation_without_parts(self, optimizer, sample_fault):
        """Test cost estimation excluding parts."""
        estimate = optimizer.repair_cost_estimation(sample_fault, include_parts=False)

        assert estimate.parts_cost == 0.0
        assert len(estimate.parts_breakdown) == 0
        assert estimate.total_cost == estimate.labor_cost + estimate.overhead_cost

    def test_cost_estimation_with_rush_service(self, optimizer, sample_fault):
        """Test that rush service increases labor cost."""
        normal_estimate = optimizer.repair_cost_estimation(sample_fault, rush_service=False)
        rush_estimate = optimizer.repair_cost_estimation(sample_fault, rush_service=True)

        assert rush_estimate.labor_rate == normal_estimate.labor_rate * 1.5
        assert rush_estimate.labor_cost > normal_estimate.labor_cost
        assert "RUSH SERVICE" in rush_estimate.notes

    def test_cost_estimation_with_custom_hours(self, optimizer, sample_fault):
        """Test cost estimation with custom labor hours."""
        custom_hours = 5.0
        estimate = optimizer.repair_cost_estimation(
            sample_fault, custom_labor_hours=custom_hours
        )

        assert estimate.labor_hours == custom_hours
        assert estimate.labor_cost == custom_hours * optimizer.labor_rate

    def test_critical_fault_higher_cost(self, optimizer):
        """Test that critical faults have higher estimated costs."""
        medium_fault = Fault(
            component_id="INV-001",
            component_type="inverter",
            fault_type=FaultType.ELECTRICAL,
            severity=FaultSeverity.MEDIUM,
            description="Medium severity fault",
            symptoms={},
            diagnosis_confidence=0.8,
        )

        critical_fault = Fault(
            component_id="INV-002",
            component_type="inverter",
            fault_type=FaultType.ELECTRICAL,
            severity=FaultSeverity.CRITICAL,
            description="Critical severity fault",
            symptoms={},
            diagnosis_confidence=0.8,
        )

        medium_estimate = optimizer.repair_cost_estimation(medium_fault)
        critical_estimate = optimizer.repair_cost_estimation(critical_fault)

        assert critical_estimate.labor_hours > medium_estimate.labor_hours

    def test_estimate_includes_overhead(self, optimizer, sample_fault):
        """Test that estimate includes overhead costs."""
        estimate = optimizer.repair_cost_estimation(sample_fault)

        expected_overhead = (estimate.labor_cost + estimate.parts_cost) * optimizer.overhead_rate
        assert abs(estimate.overhead_cost - expected_overhead) < 0.01

    def test_estimate_total_cost_calculation(self, optimizer, sample_fault):
        """Test that total cost is correctly calculated."""
        estimate = optimizer.repair_cost_estimation(sample_fault)

        expected_total = estimate.labor_cost + estimate.parts_cost + estimate.overhead_cost
        assert abs(estimate.total_cost - expected_total) < 0.01


class TestMaintenanceScheduling:
    """Test maintenance scheduling functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return RepairOptimizer()

    @pytest.fixture
    def sample_tasks(self):
        """Create sample repair tasks for testing."""
        return [
            RepairTask(
                component_id="PANEL-001",
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.HIGH,
                estimated_duration_hours=2.0,
                estimated_cost=200.0,
            ),
            RepairTask(
                component_id="PANEL-002",
                task_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.MEDIUM,
                estimated_duration_hours=1.5,
                estimated_cost=150.0,
            ),
            RepairTask(
                component_id="INV-001",
                task_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.EMERGENCY,
                estimated_duration_hours=4.0,
                estimated_cost=500.0,
            ),
        ]

    def test_basic_scheduling(self, optimizer, sample_tasks):
        """Test basic maintenance scheduling."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks, start_date=start_date, end_date=end_date
        )

        assert schedule.valid_from == start_date
        assert schedule.valid_until == end_date
        assert len(schedule.tasks) > 0
        assert schedule.total_estimated_cost > 0
        assert schedule.total_estimated_hours > 0

    def test_schedule_all_tasks_have_times(self, optimizer, sample_tasks):
        """Test that all scheduled tasks have start and end times."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks, start_date=start_date, end_date=end_date
        )

        for task in schedule.tasks:
            assert task.scheduled_start is not None
            assert task.scheduled_end is not None
            assert task.scheduled_end > task.scheduled_start

    def test_minimize_cost_objective(self, optimizer, sample_tasks):
        """Test scheduling with minimize_cost objective."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks,
            start_date=start_date,
            end_date=end_date,
            optimization_objective="minimize_cost",
        )

        assert schedule.optimization_objective == "minimize_cost"
        assert len(schedule.tasks) > 0

    def test_maximize_priority_objective(self, optimizer, sample_tasks):
        """Test scheduling with maximize_priority objective."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks,
            start_date=start_date,
            end_date=end_date,
            optimization_objective="maximize_priority",
        )

        assert schedule.optimization_objective == "maximize_priority"
        # First task should be emergency priority
        assert schedule.tasks[0].priority == MaintenancePriority.EMERGENCY

    def test_daily_hours_constraint(self, optimizer, sample_tasks):
        """Test that daily hours constraint is respected."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)
        max_daily_hours = 8.0

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks,
            start_date=start_date,
            end_date=end_date,
            max_daily_hours=max_daily_hours,
        )

        # Verify no day exceeds max hours
        tasks_by_date = {}
        for task in schedule.tasks:
            if task.scheduled_start:
                date_key = task.scheduled_start.date()
                tasks_by_date[date_key] = (
                    tasks_by_date.get(date_key, 0.0) + task.estimated_duration_hours
                )

        for date_key, hours in tasks_by_date.items():
            assert hours <= max_daily_hours

    def test_empty_task_list_raises_error(self, optimizer):
        """Test that empty task list raises ValueError."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        with pytest.raises(ValueError, match="At least one task must be provided"):
            optimizer.maintenance_scheduling(tasks=[], start_date=start_date, end_date=end_date)

    def test_invalid_date_range_raises_error(self, optimizer, sample_tasks):
        """Test that invalid date range raises ValueError."""
        start_date = datetime.now()
        end_date = start_date - timedelta(days=1)  # End before start

        with pytest.raises(ValueError, match="end_date must be after start_date"):
            optimizer.maintenance_scheduling(tasks=sample_tasks, start_date=start_date, end_date=end_date)

    def test_schedule_total_cost_calculation(self, optimizer, sample_tasks):
        """Test that schedule total cost is correctly calculated."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)

        schedule = optimizer.maintenance_scheduling(
            tasks=sample_tasks, start_date=start_date, end_date=end_date
        )

        expected_cost = sum(task.estimated_cost for task in schedule.tasks)
        assert abs(schedule.total_estimated_cost - expected_cost) < 0.01


class TestSparePartsManagement:
    """Test spare parts management functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with sample spare parts."""
        opt = RepairOptimizer()

        # Add sample spare parts
        opt.add_spare_part(
            SparePart(
                part_id="PART-001",
                part_name="Bypass Diode",
                part_number="BD-12V-10A",
                category="electrical",
                quantity_available=50,
                quantity_reserved=5,
                unit_cost=15.0,
                lead_time_days=7,
                reorder_point=20,
                reorder_quantity=50,
                supplier="ElectroSupply Inc",
            )
        )

        opt.add_spare_part(
            SparePart(
                part_id="PART-002",
                part_name="Junction Box",
                part_number="JB-PV-001",
                category="electrical",
                quantity_available=15,  # Below reorder point
                quantity_reserved=2,
                unit_cost=25.0,
                lead_time_days=14,
                reorder_point=20,
                reorder_quantity=30,
                supplier="ElectroSupply Inc",
            )
        )

        opt.add_spare_part(
            SparePart(
                part_id="PART-003",
                part_name="Cooling Fan",
                part_number="CF-120MM",
                category="thermal",
                quantity_available=0,  # Critical shortage
                quantity_reserved=0,
                unit_cost=40.0,
                lead_time_days=10,
                reorder_point=10,
                reorder_quantity=20,
                supplier="ThermalParts Ltd",
            )
        )

        return opt

    def test_inventory_status_check(self, optimizer):
        """Test checking inventory status."""
        result = optimizer.spare_parts_management(
            check_inventory=True, generate_reorder_list=False
        )

        assert "inventory_status" in result
        assert len(result["inventory_status"]) == 3
        assert "PART-001" in result["inventory_status"]

    def test_identify_parts_needing_reorder(self, optimizer):
        """Test identification of parts needing reorder."""
        result = optimizer.spare_parts_management()

        assert "reorder_recommendations" in result
        # PART-002 and PART-003 should need reorder
        reorder_ids = [item["part_id"] for item in result["reorder_recommendations"]]
        assert "PART-002" in reorder_ids
        assert "PART-003" in reorder_ids

    def test_critical_shortages_detection(self, optimizer):
        """Test detection of critical shortages."""
        result = optimizer.spare_parts_management()

        assert "critical_shortages" in result
        # PART-003 has zero quantity
        shortage_ids = [item["part_id"] for item in result["critical_shortages"]]
        assert "PART-003" in shortage_ids

    def test_reorder_cost_calculation(self, optimizer):
        """Test calculation of total reorder cost."""
        result = optimizer.spare_parts_management()

        assert "total_reorder_cost" in result
        assert result["total_reorder_cost"] > 0

        # Verify cost calculation
        expected_cost = sum(item["total_cost"] for item in result["reorder_recommendations"])
        assert abs(result["total_reorder_cost"] - expected_cost) < 0.01

    def test_forecasted_demand(self, optimizer):
        """Test demand forecasting."""
        # Add some active faults to generate demand
        optimizer.fault_diagnosis(
            component_id="PANEL-001",
            component_type="panel",
            performance_data={"temperature": 90.0},
        )

        result = optimizer.spare_parts_management(forecast_days=60)

        assert "forecasted_demand" in result

    def test_empty_inventory(self):
        """Test spare parts management with empty inventory."""
        optimizer = RepairOptimizer()
        result = optimizer.spare_parts_management()

        assert result["inventory_status"] == {}
        assert result["reorder_recommendations"] == []
        assert result["critical_shortages"] == []

    def test_spare_part_quantity_on_hand(self):
        """Test calculation of quantity on hand."""
        part = SparePart(
            part_id="PART-TEST",
            part_name="Test Part",
            part_number="TP-001",
            category="test",
            quantity_available=100,
            quantity_reserved=30,
            unit_cost=10.0,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=50,
            supplier="Test Supplier",
        )

        assert part.quantity_on_hand == 70

    def test_spare_part_needs_reorder(self):
        """Test needs_reorder property."""
        part_needs_reorder = SparePart(
            part_id="PART-LOW",
            part_name="Low Stock Part",
            part_number="LSP-001",
            category="test",
            quantity_available=15,
            quantity_reserved=0,
            unit_cost=10.0,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=50,
            supplier="Test Supplier",
        )

        assert part_needs_reorder.needs_reorder is True

        part_ok = SparePart(
            part_id="PART-OK",
            part_name="OK Stock Part",
            part_number="OSP-001",
            category="test",
            quantity_available=50,
            quantity_reserved=0,
            unit_cost=10.0,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=50,
            supplier="Test Supplier",
        )

        assert part_ok.needs_reorder is False


class TestUtilityMethods:
    """Test utility and helper methods."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return RepairOptimizer()

    def test_add_spare_part(self, optimizer):
        """Test adding spare part to inventory."""
        part = SparePart(
            part_id="PART-NEW",
            part_name="New Part",
            part_number="NP-001",
            category="test",
            quantity_available=100,
            unit_cost=20.0,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=50,
            supplier="Test Supplier",
        )

        optimizer.add_spare_part(part)
        assert "PART-NEW" in optimizer.spare_parts
        assert optimizer.spare_parts["PART-NEW"] == part

    def test_update_component_health(self, optimizer):
        """Test updating component health."""
        health = ComponentHealth(
            component_id="PANEL-001",
            component_type="panel",
            status=ComponentStatus.OPERATIONAL,
            health_score=0.95,
            degradation_rate=0.01,
        )

        optimizer.update_component_health(health)
        assert "PANEL-001" in optimizer.component_health
        assert optimizer.component_health["PANEL-001"].health_score == 0.95

    def test_get_active_faults_all(self, optimizer):
        """Test retrieving all active faults."""
        # Diagnose some faults
        optimizer.fault_diagnosis(
            component_id="PANEL-001",
            component_type="panel",
            performance_data={"temperature": 90.0},
        )
        optimizer.fault_diagnosis(
            component_id="INV-001",
            component_type="inverter",
            performance_data={"temperature": 85.0},
        )

        faults = optimizer.get_active_faults()
        assert len(faults) >= 2

    def test_get_active_faults_filtered(self, optimizer):
        """Test retrieving faults filtered by component."""
        # Diagnose faults for different components
        optimizer.fault_diagnosis(
            component_id="PANEL-001",
            component_type="panel",
            performance_data={"temperature": 90.0},
        )
        optimizer.fault_diagnosis(
            component_id="PANEL-002",
            component_type="panel",
            performance_data={"temperature": 88.0},
        )

        faults = optimizer.get_active_faults(component_id="PANEL-001")
        assert all(f.component_id == "PANEL-001" for f in faults)

    def test_clear_fault(self, optimizer):
        """Test clearing a fault from active list."""
        fault = optimizer.fault_diagnosis(
            component_id="PANEL-003",
            component_type="panel",
            performance_data={"temperature": 92.0},
        )

        assert fault is not None
        initial_count = len(optimizer.active_faults)

        result = optimizer.clear_fault(fault.fault_id)
        assert result is True
        assert len(optimizer.active_faults) == initial_count - 1

    def test_clear_nonexistent_fault(self, optimizer):
        """Test clearing a fault that doesn't exist."""
        fake_fault_id = uuid4()
        result = optimizer.clear_fault(fake_fault_id)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
