"""Pytest configuration and fixtures for PV Circularity Simulator tests."""

from datetime import datetime, timedelta
import pytest

from pv_circularity_simulator.core.enums import DegradationType
from pv_circularity_simulator.core.models import ModuleData, PerformanceMetrics
from pv_circularity_simulator.circularity.reuse_assessor import ReuseAssessor


@pytest.fixture
def assessor() -> ReuseAssessor:
    """Create a standard ReuseAssessor instance."""
    return ReuseAssessor(
        degradation_rate_per_year=0.005,
        expected_lifetime_years=25.0,
        minimum_performance_threshold=0.70,
        base_module_price_per_watt=0.50,
    )


@pytest.fixture
def new_module() -> ModuleData:
    """Create a new, pristine module (< 1 year old)."""
    return ModuleData(
        module_id="PV-NEW-001",
        manufacturer="SolarTech",
        model="ST-350",
        nameplate_power_w=350.0,
        manufacture_date=datetime.now() - timedelta(days=180),
        installation_date=datetime.now() - timedelta(days=90),
        age_years=0.5,
        visual_defects=[],
        degradation_types=[],
        location="Test Facility",
        environmental_conditions="Indoor testing",
    )


@pytest.fixture
def mid_life_module() -> ModuleData:
    """Create a mid-life module (10 years old) with minor defects."""
    return ModuleData(
        module_id="PV-MID-001",
        manufacturer="SunPower",
        model="SP-300",
        nameplate_power_w=300.0,
        manufacture_date=datetime.now() - timedelta(days=3650),
        installation_date=datetime.now() - timedelta(days=3650),
        age_years=10.0,
        visual_defects=["Minor discoloration"],
        degradation_types=[DegradationType.DISCOLORATION],
        location="Residential Rooftop",
        environmental_conditions="Moderate climate",
    )


@pytest.fixture
def old_module() -> ModuleData:
    """Create an old module (20 years) with multiple defects."""
    return ModuleData(
        module_id="PV-OLD-001",
        manufacturer="FirstSolar",
        model="FS-250",
        nameplate_power_w=250.0,
        manufacture_date=datetime.now() - timedelta(days=7300),
        installation_date=datetime.now() - timedelta(days=7300),
        age_years=20.0,
        visual_defects=["Discoloration", "Frame corrosion", "Minor delamination"],
        degradation_types=[
            DegradationType.DISCOLORATION,
            DegradationType.CORROSION,
            DegradationType.DELAMINATION,
        ],
        location="Desert Installation",
        environmental_conditions="Harsh desert climate",
    )


@pytest.fixture
def failed_module() -> ModuleData:
    """Create a failed module with critical defects."""
    return ModuleData(
        module_id="PV-FAIL-001",
        manufacturer="GenericSolar",
        model="GS-200",
        nameplate_power_w=200.0,
        manufacture_date=datetime.now() - timedelta(days=5475),
        age_years=15.0,
        visual_defects=["Severe delamination", "Hot spots", "Cell cracks", "Burned junction box"],
        degradation_types=[
            DegradationType.DELAMINATION,
            DegradationType.HOT_SPOT,
            DegradationType.CELL_CRACK,
            DegradationType.JUNCTION_BOX,
        ],
        location="Failed Field",
        environmental_conditions="Extreme conditions",
    )


@pytest.fixture
def high_performance() -> PerformanceMetrics:
    """Create high performance metrics (>90% of nameplate)."""
    return PerformanceMetrics(
        measured_power_w=340.0,  # 97% of 350W
        open_circuit_voltage_v=46.8,
        short_circuit_current_a=9.2,
        max_power_voltage_v=38.5,
        max_power_current_a=8.83,
        fill_factor=0.79,
        efficiency_percent=18.5,
        temperature_coefficient=-0.41,
        series_resistance_ohm=0.25,
        shunt_resistance_ohm=850.0,
        test_conditions="STC (1000 W/m², 25°C)",
    )


@pytest.fixture
def medium_performance() -> PerformanceMetrics:
    """Create medium performance metrics (70-90% of nameplate)."""
    return PerformanceMetrics(
        measured_power_w=240.0,  # 80% of 300W
        open_circuit_voltage_v=44.2,
        short_circuit_current_a=8.1,
        max_power_voltage_v=36.0,
        max_power_current_a=6.67,
        fill_factor=0.76,
        efficiency_percent=16.2,
        temperature_coefficient=-0.43,
        series_resistance_ohm=0.35,
        shunt_resistance_ohm=650.0,
        test_conditions="STC (1000 W/m², 25°C)",
    )


@pytest.fixture
def low_performance() -> PerformanceMetrics:
    """Create low performance metrics (50-70% of nameplate)."""
    return PerformanceMetrics(
        measured_power_w=150.0,  # 60% of 250W
        open_circuit_voltage_v=41.5,
        short_circuit_current_a=6.8,
        max_power_voltage_v=33.5,
        max_power_current_a=4.48,
        fill_factor=0.71,
        efficiency_percent=13.8,
        series_resistance_ohm=0.55,
        shunt_resistance_ohm=450.0,
        test_conditions="STC (1000 W/m², 25°C)",
    )


@pytest.fixture
def critical_performance() -> PerformanceMetrics:
    """Create critical performance metrics (<50% of nameplate)."""
    return PerformanceMetrics(
        measured_power_w=80.0,  # 40% of 200W
        open_circuit_voltage_v=38.2,
        short_circuit_current_a=5.2,
        max_power_voltage_v=29.8,
        max_power_current_a=2.68,
        fill_factor=0.63,
        efficiency_percent=9.5,
        series_resistance_ohm=1.2,
        shunt_resistance_ohm=250.0,
        test_conditions="STC (1000 W/m², 25°C)",
    )
