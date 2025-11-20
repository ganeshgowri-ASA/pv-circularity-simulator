"""
Unit tests for AlertEngine class.

Tests alert detection, management, and notification functionality.
"""

import pytest
from datetime import datetime
from config.settings import Settings
from src.monitoring.alerts.engine import AlertEngine
from src.core.models.schemas import (
    InverterData,
    SCADAData,
    PerformanceRatioData,
    AlertSeverity,
    AlertType
)


@pytest.fixture
def settings():
    """Create test settings instance."""
    return Settings(
        site_id="TEST001",
        site_capacity_kw=1000.0
    )


@pytest.fixture
def alert_engine(settings):
    """Create AlertEngine instance."""
    return AlertEngine(settings)


@pytest.mark.asyncio
async def test_underperformance_detection_triggered(alert_engine):
    """Test underperformance alert is triggered when PR is low."""
    pr_data = PerformanceRatioData(
        timestamp=datetime.utcnow(),
        site_id="TEST001",
        instantaneous_pr=70.0,  # Below 80% threshold
        actual_energy=700.0,
        expected_energy=1000.0,
        reference_irradiance=1000.0,
        actual_irradiance=950.0,
        actual_temperature=45.0
    )

    alert = await alert_engine.underperformance_detection(pr_data)

    assert alert is not None
    assert alert.alert_type == AlertType.UNDERPERFORMANCE
    assert alert.performance_ratio == 70.0
    assert alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]


@pytest.mark.asyncio
async def test_underperformance_detection_not_triggered(alert_engine):
    """Test no alert when PR is above threshold."""
    pr_data = PerformanceRatioData(
        timestamp=datetime.utcnow(),
        site_id="TEST001",
        instantaneous_pr=90.0,  # Above threshold
        actual_energy=900.0,
        expected_energy=1000.0,
        reference_irradiance=1000.0,
        actual_irradiance=950.0,
        actual_temperature=45.0
    )

    alert = await alert_engine.underperformance_detection(pr_data)

    assert alert is None


@pytest.mark.asyncio
async def test_equipment_fault_error_code(alert_engine):
    """Test equipment fault detection with error code."""
    inv_data = InverterData(
        inverter_id="INV001",
        dc_power=45.0,
        ac_power=43.0,
        dc_voltage=600.0,
        dc_current=75.0,
        ac_voltage_l1=400.0,
        ac_current_l1=62.0,
        temperature=50.0,
        error_code=10,  # Overtemperature error
        status="fault"
    )

    alert = await alert_engine.equipment_fault_alerts(inv_data)

    assert alert is not None
    assert alert.alert_type == AlertType.EQUIPMENT_FAULT
    assert alert.component_id == "INV001"
    assert alert.severity == AlertSeverity.HIGH


@pytest.mark.asyncio
async def test_equipment_fault_high_temperature(alert_engine):
    """Test equipment fault detection for high temperature."""
    inv_data = InverterData(
        inverter_id="INV002",
        dc_power=45.0,
        ac_power=43.0,
        dc_voltage=600.0,
        dc_current=75.0,
        ac_voltage_l1=400.0,
        ac_current_l1=62.0,
        temperature=90.0,  # Above 85°C threshold
        status="online"
    )

    alert = await alert_engine.equipment_fault_alerts(inv_data)

    assert alert is not None
    assert "temperature" in alert.description.lower()


@pytest.mark.asyncio
async def test_equipment_no_fault(alert_engine):
    """Test no alert when equipment is normal."""
    inv_data = InverterData(
        inverter_id="INV003",
        dc_power=45.0,
        ac_power=43.0,
        dc_voltage=600.0,
        dc_current=75.0,
        ac_voltage_l1=400.0,
        ac_current_l1=62.0,
        temperature=50.0,
        status="online"
    )

    alert = await alert_engine.equipment_fault_alerts(inv_data)

    assert alert is None


@pytest.mark.asyncio
async def test_grid_outage_zero_frequency(alert_engine):
    """Test grid outage detection with zero frequency."""
    scada_data = SCADAData(
        site_id="TEST001",
        total_dc_power=0.0,
        total_ac_power=0.0,
        irradiance=800.0,
        ambient_temperature=25.0,
        grid_frequency=0.0,  # Grid outage
        grid_voltage=0.0,
        available_inverters=0,
        total_inverters=10
    )

    alert = await alert_engine.grid_outage_detection(scada_data)

    assert alert is not None
    assert alert.alert_type == AlertType.GRID_OUTAGE
    assert alert.severity == AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_grid_outage_out_of_range_frequency(alert_engine):
    """Test grid outage detection with out-of-range frequency."""
    scada_data = SCADAData(
        site_id="TEST001",
        total_dc_power=500.0,
        total_ac_power=480.0,
        irradiance=950.0,
        ambient_temperature=25.0,
        grid_frequency=52.0,  # Above max threshold
        grid_voltage=400.0,
        available_inverters=10,
        total_inverters=10
    )

    alert = await alert_engine.grid_outage_detection(scada_data)

    assert alert is not None
    assert "frequency" in alert.description.lower()


@pytest.mark.asyncio
async def test_grid_normal(alert_engine):
    """Test no alert when grid is normal."""
    scada_data = SCADAData(
        site_id="TEST001",
        total_dc_power=500.0,
        total_ac_power=480.0,
        irradiance=950.0,
        ambient_temperature=25.0,
        grid_frequency=50.0,
        grid_voltage=400.0,
        available_inverters=10,
        total_inverters=10
    )

    alert = await alert_engine.grid_outage_detection(scada_data)

    assert alert is None


@pytest.mark.asyncio
async def test_acknowledge_alert(alert_engine):
    """Test alert acknowledgment."""
    # Generate an alert first
    pr_data = PerformanceRatioData(
        timestamp=datetime.utcnow(),
        site_id="TEST001",
        instantaneous_pr=70.0,
        actual_energy=700.0,
        expected_energy=1000.0,
        reference_irradiance=1000.0,
        actual_irradiance=950.0,
        actual_temperature=45.0
    )

    alert = await alert_engine.underperformance_detection(pr_data)
    assert alert is not None

    # Acknowledge the alert
    result = await alert_engine.acknowledge_alert(alert.alert_id, "test_user")

    assert result is True
    assert alert.acknowledged is True
    assert alert.acknowledged_by == "test_user"


@pytest.mark.asyncio
async def test_resolve_alert(alert_engine):
    """Test alert resolution."""
    # Generate an alert first
    pr_data = PerformanceRatioData(
        timestamp=datetime.utcnow(),
        site_id="TEST001",
        instantaneous_pr=70.0,
        actual_energy=700.0,
        expected_energy=1000.0,
        reference_irradiance=1000.0,
        actual_irradiance=950.0,
        actual_temperature=45.0
    )

    alert = await alert_engine.underperformance_detection(pr_data)
    assert alert is not None
    alert_id = alert.alert_id

    # Resolve the alert
    result = await alert_engine.resolve_alert(alert_id)

    assert result is True
    assert alert.resolved is True
    assert alert_id not in alert_engine._active_alerts


def test_get_active_alerts(alert_engine):
    """Test retrieving active alerts."""
    alerts = alert_engine.get_active_alerts()
    assert isinstance(alerts, list)


def test_get_alert_statistics(alert_engine):
    """Test retrieving alert statistics."""
    stats = alert_engine.get_alert_statistics()

    assert 'total_active' in stats
    assert 'total_history' in stats
    assert 'by_severity' in stats
    assert 'by_type' in stats
    assert 'unacknowledged' in stats


def test_alert_callback_registration(alert_engine):
    """Test callback registration."""
    async def test_callback(alert):
        pass

    alert_engine.register_callback(test_callback)
    assert test_callback in alert_engine._alert_callbacks
