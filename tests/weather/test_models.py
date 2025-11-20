"""
Tests for weather data models.
"""

import pytest
from datetime import datetime

from pydantic import ValidationError

from pv_simulator.models.weather import (
    DataQualityMetrics,
    GeoLocation,
    WeatherDataPoint,
    WeatherProvider,
)


class TestGeoLocation:
    """Tests for GeoLocation model."""

    def test_valid_location(self):
        """Test creating valid location."""
        location = GeoLocation(latitude=40.7128, longitude=-74.0060)
        assert location.latitude == 40.7128
        assert location.longitude == -74.0060

    def test_invalid_latitude(self):
        """Test validation of latitude bounds."""
        with pytest.raises(ValidationError):
            GeoLocation(latitude=95.0, longitude=0.0)

        with pytest.raises(ValidationError):
            GeoLocation(latitude=-95.0, longitude=0.0)

    def test_invalid_longitude(self):
        """Test validation of longitude bounds."""
        with pytest.raises(ValidationError):
            GeoLocation(latitude=0.0, longitude=185.0)

        with pytest.raises(ValidationError):
            GeoLocation(latitude=0.0, longitude=-185.0)

    def test_location_string_representation(self):
        """Test string representation of location."""
        location = GeoLocation(
            latitude=40.7128,
            longitude=-74.0060,
            city="New York",
            country="US",
        )
        str_repr = str(location)
        assert "New York" in str_repr
        assert "US" in str_repr


class TestWeatherDataPoint:
    """Tests for WeatherDataPoint model."""

    def test_valid_weather_data(self, test_location):
        """Test creating valid weather data point."""
        data = WeatherDataPoint(
            timestamp=datetime.utcnow(),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            temperature=20.0,
            humidity=65.0,
        )
        assert data.temperature == 20.0
        assert data.humidity == 65.0

    def test_temperature_validation(self, test_location):
        """Test temperature bounds validation."""
        # Valid temperature
        data = WeatherDataPoint(
            timestamp=datetime.utcnow(),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            temperature=50.0,
        )
        assert data.temperature == 50.0

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            WeatherDataPoint(
                timestamp=datetime.utcnow(),
                location=test_location,
                provider=WeatherProvider.OPENWEATHERMAP,
                temperature=70.0,
            )

    def test_humidity_validation(self, test_location):
        """Test humidity bounds validation."""
        # Valid humidity
        data = WeatherDataPoint(
            timestamp=datetime.utcnow(),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            humidity=75.0,
        )
        assert data.humidity == 75.0

        # Invalid humidity
        with pytest.raises(ValidationError):
            WeatherDataPoint(
                timestamp=datetime.utcnow(),
                location=test_location,
                provider=WeatherProvider.OPENWEATHERMAP,
                humidity=150.0,
            )

    def test_has_solar_data(self, test_location):
        """Test checking for solar data presence."""
        # Data with solar irradiance
        data_with_solar = WeatherDataPoint(
            timestamp=datetime.utcnow(),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            ghi=500.0,
        )
        assert data_with_solar.has_solar_data() is True

        # Data without solar irradiance
        data_without_solar = WeatherDataPoint(
            timestamp=datetime.utcnow(),
            location=test_location,
            provider=WeatherProvider.OPENWEATHERMAP,
            temperature=20.0,
        )
        assert data_without_solar.has_solar_data() is False


class TestDataQualityMetrics:
    """Tests for DataQualityMetrics model."""

    def test_valid_metrics(self):
        """Test creating valid quality metrics."""
        metrics = DataQualityMetrics(
            total_points=100,
            valid_points=95,
            missing_points=5,
            temperature_completeness=0.95,
            solar_completeness=0.90,
            wind_completeness=0.92,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            quality_score=0.92,
        )
        assert metrics.total_points == 100
        assert metrics.valid_points == 95

    def test_completeness_ratio(self):
        """Test completeness ratio calculation."""
        metrics = DataQualityMetrics(
            total_points=100,
            valid_points=80,
            missing_points=20,
            temperature_completeness=0.8,
            solar_completeness=0.8,
            wind_completeness=0.8,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            quality_score=0.8,
        )
        assert metrics.completeness_ratio() == 0.8

    def test_has_high_quality(self):
        """Test high quality threshold check."""
        high_quality = DataQualityMetrics(
            total_points=100,
            valid_points=90,
            missing_points=10,
            temperature_completeness=0.9,
            solar_completeness=0.9,
            wind_completeness=0.9,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            quality_score=0.9,
        )
        assert high_quality.has_high_quality() is True

        low_quality = DataQualityMetrics(
            total_points=100,
            valid_points=60,
            missing_points=40,
            temperature_completeness=0.6,
            solar_completeness=0.6,
            wind_completeness=0.6,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            quality_score=0.6,
        )
        assert low_quality.has_high_quality() is False
