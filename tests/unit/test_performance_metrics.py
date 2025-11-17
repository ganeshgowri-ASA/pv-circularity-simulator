"""
Unit tests for PerformanceMetrics class.

Tests all performance metric calculations including PR, capacity factor,
specific yield, availability, and grid export monitoring.
"""

import pytest
from datetime import datetime
from config.settings import Settings
from src.monitoring.metrics.performance import PerformanceMetrics
from src.core.models.schemas import InverterData, SCADAData


@pytest.fixture
def settings():
    """Create test settings instance."""
    return Settings(
        site_id="TEST001",
        site_capacity_kw=1000.0
    )


@pytest.fixture
def metrics_calculator(settings):
    """Create PerformanceMetrics instance."""
    return PerformanceMetrics(settings)


@pytest.mark.asyncio
async def test_instantaneous_pr_normal_conditions(metrics_calculator):
    """Test PR calculation under normal operating conditions."""
    pr_data = await metrics_calculator.instantaneous_pr(
        actual_power=850.0,
        irradiance=950.0,
        module_temperature=45.0,
        rated_capacity=1000.0
    )

    assert pr_data.instantaneous_pr > 0
    assert pr_data.instantaneous_pr <= 100
    assert pr_data.actual_irradiance == 950.0
    assert pr_data.actual_temperature == 45.0


@pytest.mark.asyncio
async def test_instantaneous_pr_low_irradiance(metrics_calculator):
    """Test PR calculation with low irradiance."""
    pr_data = await metrics_calculator.instantaneous_pr(
        actual_power=100.0,
        irradiance=200.0,
        module_temperature=25.0,
        rated_capacity=1000.0
    )

    assert pr_data.instantaneous_pr >= 0
    assert pr_data.instantaneous_pr <= 100


@pytest.mark.asyncio
async def test_instantaneous_pr_high_temperature(metrics_calculator):
    """Test PR calculation with high module temperature."""
    pr_data = await metrics_calculator.instantaneous_pr(
        actual_power=800.0,
        irradiance=1000.0,
        module_temperature=65.0,  # High temperature
        rated_capacity=1000.0
    )

    # PR should be lower due to temperature losses
    assert pr_data.instantaneous_pr > 0
    assert pr_data.instantaneous_pr < 100


@pytest.mark.asyncio
async def test_capacity_factor_normal(metrics_calculator):
    """Test capacity factor calculation."""
    cf_data = await metrics_calculator.capacity_factor(
        actual_energy=18000.0,  # kWh
        time_period_hours=24.0,
        rated_capacity=1000.0
    )

    assert cf_data.capacity_factor == 75.0
    assert cf_data.time_period_hours == 24.0


@pytest.mark.asyncio
async def test_capacity_factor_low_production(metrics_calculator):
    """Test capacity factor with low energy production."""
    cf_data = await metrics_calculator.capacity_factor(
        actual_energy=5000.0,
        time_period_hours=24.0,
        rated_capacity=1000.0
    )

    assert cf_data.capacity_factor < 30.0
    assert cf_data.capacity_factor >= 0


@pytest.mark.asyncio
async def test_specific_yield_daily(metrics_calculator):
    """Test specific yield calculation for daily period."""
    sy_data = await metrics_calculator.specific_yield(
        energy_production=4250.0,
        installed_capacity=1000.0,
        period_type='daily'
    )

    assert sy_data.specific_yield == 4.25  # kWh/kWp
    assert sy_data.period_type == 'daily'


@pytest.mark.asyncio
async def test_specific_yield_monthly(metrics_calculator):
    """Test specific yield calculation for monthly period."""
    sy_data = await metrics_calculator.specific_yield(
        energy_production=120000.0,
        installed_capacity=1000.0,
        period_type='monthly'
    )

    assert sy_data.specific_yield == 120.0
    assert sy_data.period_type == 'monthly'


@pytest.mark.asyncio
async def test_availability_tracking_full(metrics_calculator):
    """Test availability tracking with full availability."""
    avail_data = await metrics_calculator.availability_tracking(
        uptime_hours=24.0,
        total_hours=24.0,
        available_components=100,
        total_components=100
    )

    assert avail_data.availability_percentage == 100.0
    assert avail_data.downtime_hours == 0.0


@pytest.mark.asyncio
async def test_availability_tracking_partial(metrics_calculator):
    """Test availability tracking with partial availability."""
    avail_data = await metrics_calculator.availability_tracking(
        uptime_hours=22.0,
        total_hours=24.0,
        available_components=95,
        total_components=100,
        planned_downtime_hours=1.0,
        unplanned_downtime_hours=1.0
    )

    expected_availability = (22.0 / 24.0) * 100
    assert abs(avail_data.availability_percentage - expected_availability) < 0.1
    assert avail_data.downtime_hours == 2.0


@pytest.mark.asyncio
async def test_grid_export_monitoring(metrics_calculator):
    """Test grid export monitoring."""
    grid_data = await metrics_calculator.grid_export_monitoring(
        export_power=950.0,
        export_energy=18500.0,
        grid_voltage=400.0,
        grid_frequency=50.0,
        power_factor=0.98
    )

    assert grid_data.export_power == 950.0
    assert grid_data.grid_frequency == 50.0
    assert grid_data.power_factor == 0.98
    assert grid_data.grid_connected is True


@pytest.mark.asyncio
async def test_grid_export_out_of_range_frequency(metrics_calculator):
    """Test grid monitoring with out-of-range frequency (should log warning)."""
    grid_data = await metrics_calculator.grid_export_monitoring(
        export_power=950.0,
        export_energy=18500.0,
        grid_voltage=400.0,
        grid_frequency=52.0,  # Out of normal range
        power_factor=0.98
    )

    # Should still return data but log warning
    assert grid_data.grid_frequency == 52.0


@pytest.mark.asyncio
async def test_calculate_batch_metrics(metrics_calculator):
    """Test batch metrics calculation."""
    # Create sample data
    inverter_data = [
        InverterData(
            inverter_id=f"INV{i:03d}",
            dc_power=48.0,
            ac_power=46.0,
            dc_voltage=600.0,
            dc_current=80.0,
            ac_voltage_l1=400.0,
            ac_current_l1=66.0,
            temperature=45.0,
            energy_daily=200.0,
            efficiency=95.8,
            status="online"
        )
        for i in range(1, 11)
    ]

    scada_data = [
        SCADAData(
            site_id="TEST001",
            total_dc_power=480.0,
            total_ac_power=460.0,
            irradiance=950.0,
            ambient_temperature=25.0,
            module_temperature=45.0,
            available_inverters=10,
            total_inverters=10
        )
    ]

    metrics = await metrics_calculator.calculate_batch_metrics(
        inverter_data=inverter_data,
        scada_data=scada_data,
        time_period_hours=24.0
    )

    assert 'performance_ratio' in metrics
    assert 'capacity_factor' in metrics
    assert 'specific_yield' in metrics
    assert 'availability' in metrics
    assert metrics['performance_ratio'].instantaneous_pr > 0


def test_performance_metrics_initialization(settings):
    """Test PerformanceMetrics initialization."""
    metrics = PerformanceMetrics(settings)
    assert metrics.settings == settings
    assert metrics._cache == {}
