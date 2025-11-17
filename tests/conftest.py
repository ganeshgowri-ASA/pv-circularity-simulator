"""
Pytest configuration and fixtures for testing.
"""

import pytest
from datetime import datetime

from pv_simulator.config import Settings
from pv_simulator.models.weather import GeoLocation, WeatherDataPoint, WeatherProvider
from pv_simulator.weather.cache import CacheManager


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock API keys."""
    return Settings(
        openweathermap_api_key="test_openweather_key",
        visualcrossing_api_key="test_visualcrossing_key",
        meteomatics_username="test_user",
        meteomatics_password="test_pass",
        tomorrow_io_api_key="test_tomorrow_key",
        nrel_api_key="test_nrel_key",
        cache_type="memory",
        cache_ttl=300,
        log_level="DEBUG",
    )


@pytest.fixture
def cache_manager(test_settings: Settings) -> CacheManager:
    """Create cache manager for testing."""
    return CacheManager(test_settings)


@pytest.fixture
def test_location() -> GeoLocation:
    """Create test location (New York City)."""
    return GeoLocation(
        latitude=40.7128,
        longitude=-74.0060,
        city="New York",
        country="US",
        timezone="America/New_York",
    )


@pytest.fixture
def sample_weather_data(test_location: GeoLocation) -> WeatherDataPoint:
    """Create sample weather data point."""
    return WeatherDataPoint(
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        location=test_location,
        provider=WeatherProvider.OPENWEATHERMAP,
        temperature=15.5,
        feels_like=14.0,
        humidity=65.0,
        pressure=1013.0,
        wind_speed=5.5,
        wind_direction=180.0,
        cloud_cover=25.0,
        ghi=450.0,
        dni=600.0,
        dhi=150.0,
    )


@pytest.fixture
def sample_weather_series(test_location: GeoLocation) -> list[WeatherDataPoint]:
    """Create sample weather data series."""
    data_points = []

    for hour in range(24):
        point = WeatherDataPoint(
            timestamp=datetime(2024, 1, 15, hour, 0, 0),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            temperature=10.0 + hour * 0.5,
            humidity=60.0 + hour * 0.5,
            pressure=1013.0,
            wind_speed=3.0 + hour * 0.2,
            ghi=100.0 * (1 if 6 <= hour <= 18 else 0),
        )
        data_points.append(point)

    return data_points
