"""
Tests for configuration module.
"""

import pytest

from pv_simulator.config import AlertSeverity, Environment, Settings, get_settings


def test_settings_initialization():
    """Test Settings initialization."""
    settings = get_settings()
    assert settings is not None
    assert isinstance(settings.app_name, str)
    assert isinstance(settings.app_env, Environment)


def test_settings_singleton():
    """Test Settings singleton pattern."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_alert_severity_levels():
    """Test AlertSeverity enum."""
    assert AlertSeverity.INFO.value == "INFO"
    assert AlertSeverity.WARNING.value == "WARNING"
    assert AlertSeverity.ERROR.value == "ERROR"
    assert AlertSeverity.CRITICAL.value == "CRITICAL"
    assert AlertSeverity.EMERGENCY.value == "EMERGENCY"


def test_enabled_notification_channels():
    """Test getting enabled notification channels."""
    settings = get_settings()
    channels = settings.get_enabled_notification_channels()
    assert isinstance(channels, list)


def test_environment_checks():
    """Test environment property checks."""
    settings = get_settings()
    assert isinstance(settings.is_production, bool)
    assert isinstance(settings.is_development, bool)


def test_settings_to_dict():
    """Test settings serialization to dict."""
    settings = get_settings()
    config_dict = settings.to_dict()
    assert isinstance(config_dict, dict)
    assert "app_name" in config_dict
