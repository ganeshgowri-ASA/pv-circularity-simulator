"""
Pytest configuration and shared fixtures for PV Circularity Simulator tests.
"""

import pytest
from datetime import datetime
from typing import List

from pv_simulator.core.models import (
    WindResourceData,
    TurbineSpecifications,
    PVSystemConfig,
    HybridSystemConfig,
    TurbineType,
    CoordinationStrategy,
)


@pytest.fixture
def sample_wind_data() -> WindResourceData:
    """Create sample wind resource data for testing."""
    # Generate realistic wind speed data (8760 hours = 1 year)
    import numpy as np
    np.random.seed(42)

    # Generate wind speeds with mean ~7 m/s and some variation
    num_points = 8760
    base_speeds = np.random.weibull(2.0, num_points) * 7.0
    wind_speeds = base_speeds.tolist()

    # Generate wind directions (prevailing from west, 270°)
    wind_directions = (np.random.normal(270, 30, num_points) % 360).tolist()

    return WindResourceData(
        site_id="test_site_001",
        latitude=45.0,
        longitude=-95.0,
        elevation_m=300.0,
        wind_speeds_ms=wind_speeds,
        wind_directions_deg=wind_directions,
        air_density_kgm3=1.225,
        temperature_c=15.0,
        pressure_pa=101325.0,
        measurement_height_m=10.0,
        assessment_period_days=365,
        data_quality_score=0.95,
    )


@pytest.fixture
def sample_turbine_specs() -> TurbineSpecifications:
    """Create sample wind turbine specifications."""
    # Typical 3 MW turbine power curve
    power_curve_speeds = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    power_curve_kw = [0, 50, 250, 750, 1500, 2250, 2850, 3000, 3000, 3000, 3000, 3000, 0]

    return TurbineSpecifications(
        turbine_id="test_turbine_001",
        manufacturer="TestCorp",
        model="TC-3000",
        rated_power_kw=3000.0,
        rotor_diameter_m=112.0,
        hub_height_m=80.0,
        cut_in_speed_ms=3.0,
        rated_speed_ms=15.0,
        cut_out_speed_ms=25.0,
        power_curve_speeds_ms=power_curve_speeds,
        power_curve_kw=power_curve_kw,
        turbine_type=TurbineType.ONSHORE,
        efficiency=0.95,
    )


@pytest.fixture
def sample_pv_system_config() -> PVSystemConfig:
    """Create sample PV system configuration."""
    return PVSystemConfig(
        system_id="test_pv_001",
        capacity_mw=10.0,
        module_efficiency=0.20,
        inverter_efficiency=0.98,
        tilt_angle_deg=30.0,
        azimuth_deg=180.0,
        temperature_coefficient=-0.004,
    )


@pytest.fixture
def sample_hybrid_config(
    sample_turbine_specs: TurbineSpecifications,
    sample_pv_system_config: PVSystemConfig,
) -> HybridSystemConfig:
    """Create sample hybrid system configuration."""
    num_turbines = 5
    wind_capacity_mw = num_turbines * sample_turbine_specs.rated_power_kw / 1000

    return HybridSystemConfig(
        system_id="test_hybrid_001",
        site_name="Test Hybrid Site",
        pv_capacity_mw=10.0,
        wind_capacity_mw=wind_capacity_mw,
        num_turbines=num_turbines,
        pv_system=sample_pv_system_config,
        turbine_specs=sample_turbine_specs,
        shared_infrastructure=True,
        storage_capacity_mwh=20.0,
        grid_connection_capacity_mw=20.0,
    )


@pytest.fixture
def sample_coordination_strategy() -> CoordinationStrategy:
    """Create sample coordination strategy."""
    return CoordinationStrategy(
        strategy_name="test_strategy",
        dispatch_priority=["wind", "pv", "storage"],
        ramp_rate_limit_mw_per_min=5.0,
        forecast_horizon_hours=24,
        enable_storage_arbitrage=True,
        curtailment_strategy="proportional",
        grid_support_enabled=True,
    )


@pytest.fixture
def sample_generation_timeseries() -> tuple[List[float], List[float]]:
    """Create sample wind and PV generation timeseries."""
    import numpy as np
    np.random.seed(42)

    # 24 hours of generation data
    num_points = 288  # 5-minute intervals for 24 hours

    # Wind generation (more constant, with some variation)
    wind_gen = (np.random.normal(10, 2, num_points)).clip(0, 15).tolist()

    # PV generation (follows solar curve)
    hours = np.linspace(0, 24, num_points)
    solar_curve = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
    pv_gen = (solar_curve * 8 + np.random.normal(0, 0.5, num_points)).clip(0, 10).tolist()

    return wind_gen, pv_gen
