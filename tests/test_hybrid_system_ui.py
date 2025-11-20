"""
Tests for Hybrid Energy System UI module.

This test suite validates the core functionality of the HybridSystemUI class
and related components.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.config import (
    SystemConfiguration,
    ComponentConfig,
    OperationStrategy,
    ConfigManager,
)
from src.core.models import (
    HybridEnergySystem,
    PVArray,
    BatteryStorage,
    EnergyComponent,
)
from src.monitoring.metrics import PerformanceMetrics, SystemMetrics, MetricsTracker
from src.utils.helpers import (
    format_power,
    format_energy,
    format_percentage,
    sanitize_component_id,
)


class TestConfiguration:
    """Test suite for configuration management."""

    def test_component_config_creation(self):
        """Test creating a component configuration."""
        comp = ComponentConfig(
            component_id="test_001",
            component_type="pv_array",
            name="Test PV",
            capacity=10.0,
            efficiency=0.85,
        )

        assert comp.component_id == "test_001"
        assert comp.component_type == "pv_array"
        assert comp.capacity == 10.0
        assert comp.efficiency == 0.85

    def test_system_config_creation(self):
        """Test creating a system configuration."""
        config = SystemConfiguration(
            system_name="Test System",
            components=[
                ComponentConfig(
                    component_id="pv_001",
                    component_type="pv_array",
                    name="PV 1",
                    capacity=10.0,
                )
            ],
        )

        assert config.system_name == "Test System"
        assert len(config.components) == 1

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = ConfigManager.create_default_config()

        assert config.system_name == "Default Hybrid System"
        assert len(config.components) >= 2

    def test_get_component_by_id(self):
        """Test retrieving component by ID."""
        config = ConfigManager.create_default_config()
        comp = config.get_component_by_id("pv_001")

        assert comp is not None
        assert comp.component_id == "pv_001"

    def test_get_components_by_type(self):
        """Test retrieving components by type."""
        config = ConfigManager.create_default_config()
        pv_components = config.get_components_by_type("pv_array")

        assert len(pv_components) >= 1
        assert all(c.component_type == "pv_array" for c in pv_components)


class TestEnergyComponents:
    """Test suite for energy component models."""

    def test_pv_array_creation(self):
        """Test creating a PV array component."""
        pv = PVArray(
            component_id="pv_001",
            component_type="pv_array",
            name="Test PV",
            capacity=10.0,
            efficiency=0.85,
        )

        assert pv.component_id == "pv_001"
        assert pv.capacity == 10.0

    def test_pv_output_calculation(self):
        """Test PV power output calculation."""
        pv = PVArray(
            component_id="pv_001",
            component_type="pv_array",
            name="Test PV",
            capacity=10.0,
            efficiency=0.20,
            area_m2=50.0,
        )

        input_conditions = {"irradiance_w_m2": 1000.0, "temperature_c": 25.0}

        power = pv.calculate_output(input_conditions)

        assert power > 0
        assert power <= pv.capacity

    def test_battery_charge(self):
        """Test battery charging."""
        battery = BatteryStorage(
            component_id="bat_001",
            component_type="battery",
            name="Test Battery",
            capacity=20.0,
            state_of_charge=0.5,
        )

        initial_soc = battery.state_of_charge
        battery.charge(5.0, time_step_minutes=5)

        assert battery.state_of_charge > initial_soc

    def test_battery_discharge(self):
        """Test battery discharging."""
        battery = BatteryStorage(
            component_id="bat_001",
            component_type="battery",
            name="Test Battery",
            capacity=20.0,
            state_of_charge=0.5,
        )

        initial_soc = battery.state_of_charge
        battery.discharge(5.0, time_step_minutes=5)

        assert battery.state_of_charge < initial_soc

    def test_battery_soc_limits(self):
        """Test battery SOC limits."""
        battery = BatteryStorage(
            component_id="bat_001",
            component_type="battery",
            name="Test Battery",
            capacity=20.0,
            state_of_charge=0.9,
            max_soc=0.9,
        )

        # Try to charge beyond max SOC
        battery.charge(10.0, time_step_minutes=60)

        assert battery.state_of_charge <= battery.max_soc


class TestHybridEnergySystem:
    """Test suite for hybrid energy system."""

    def test_system_creation(self):
        """Test creating a hybrid energy system."""
        system = HybridEnergySystem("Test System")

        assert system.system_name == "Test System"
        assert len(system.components) == 0

    def test_add_component(self):
        """Test adding components to system."""
        system = HybridEnergySystem("Test System")

        pv = PVArray(
            component_id="pv_001",
            component_type="pv_array",
            name="PV 1",
            capacity=10.0,
        )

        system.add_component(pv)

        assert len(system.components) == 1
        assert "pv_001" in system.components

    def test_get_total_capacity(self):
        """Test calculating total system capacity."""
        system = HybridEnergySystem("Test System")

        pv = PVArray(
            component_id="pv_001",
            component_type="pv_array",
            name="PV 1",
            capacity=10.0,
        )

        system.add_component(pv)
        total_capacity = system.get_total_capacity()

        assert total_capacity == 10.0

    def test_simulate_step(self):
        """Test simulating one time step."""
        system = HybridEnergySystem("Test System")

        pv = PVArray(
            component_id="pv_001",
            component_type="pv_array",
            name="PV 1",
            capacity=10.0,
            efficiency=0.20,
            area_m2=50.0,
        )

        battery = BatteryStorage(
            component_id="bat_001",
            component_type="battery",
            name="Battery 1",
            capacity=20.0,
        )

        system.add_component(pv)
        system.add_component(battery)

        input_conditions = {"irradiance_w_m2": 1000.0, "temperature_c": 25.0}

        result = system.simulate_step(
            load_demand_kw=5.0, input_conditions=input_conditions, time_step_minutes=5
        )

        assert "pv_generation_kw" in result
        assert "battery_power_kw" in result
        assert result["load_demand_kw"] == 5.0


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_format_power(self):
        """Test power formatting."""
        assert "W" in format_power(0.5)
        assert "kW" in format_power(50.0)
        assert "MW" in format_power(5000.0)

    def test_format_energy(self):
        """Test energy formatting."""
        assert "Wh" in format_energy(0.5)
        assert "kWh" in format_energy(50.0)
        assert "MWh" in format_energy(5000.0)

    def test_format_percentage(self):
        """Test percentage formatting."""
        result = format_percentage(0.856)
        assert "%" in result
        assert "85" in result

    def test_sanitize_component_id(self):
        """Test component ID sanitization."""
        result = sanitize_component_id("PV Array #1!")
        assert result.isidentifier() or "_" in result

        # Test short ID
        result = sanitize_component_id("ab")
        assert len(result) >= 3


class TestMetrics:
    """Test suite for metrics tracking."""

    def test_system_metrics_creation(self):
        """Test creating system metrics."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            total_generation_kw=10.0,
            total_consumption_kw=8.0,
        )

        assert metrics.total_generation_kw == 10.0
        assert metrics.total_consumption_kw == 8.0

    def test_metrics_tracker(self):
        """Test metrics tracker."""
        tracker = MetricsTracker(max_history_points=100)

        metric = SystemMetrics(timestamp=datetime.now(), total_generation_kw=10.0)

        tracker.add_metric(metric)

        assert len(tracker.metrics_history) == 1

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        perf = PerformanceMetrics(start_time=datetime.now(), end_time=datetime.now())

        simulation_data = [
            {
                "pv_generation_kw": 10.0,
                "load_demand_kw": 8.0,
                "grid_power_kw": 0.0,
            }
            for _ in range(10)
        ]

        perf.calculate_metrics(simulation_data, system_capacity_kw=10.0)

        assert perf.total_energy_generated_kwh > 0
        assert perf.capacity_factor >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
