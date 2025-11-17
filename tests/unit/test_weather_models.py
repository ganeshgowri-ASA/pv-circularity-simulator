"""
Unit tests for weather data models.
"""

import pytest
from datetime import datetime

from pv_simulator.models.weather import (
    DataQuality,
    DataSource,
    GlobalLocation,
    TMYData,
    WeatherDataPoint,
)


class TestWeatherDataPoint:
    """Tests for WeatherDataPoint model."""

    def test_create_weather_point(self, sample_weather_point: WeatherDataPoint) -> None:
        """Test creating a weather data point."""
        assert sample_weather_point.temperature == 28.5
        assert sample_weather_point.irradiance_ghi == 950.0
        assert sample_weather_point.data_quality == DataQuality.EXCELLENT

    def test_irradiance_validation(self) -> None:
        """Test that GHI >= DHI validation works."""
        # This should raise a validation error
        with pytest.raises(ValueError, match="GHI must be greater than"):
            WeatherDataPoint(
                timestamp=datetime.now(),
                temperature=25.0,
                irradiance_ghi=100.0,  # GHI less than DHI
                irradiance_dni=50.0,
                irradiance_dhi=150.0,  # DHI greater than GHI
                wind_speed=3.0,
            )

    def test_temperature_range(self) -> None:
        """Test temperature range validation."""
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(),
                temperature=100.0,  # Too high
                irradiance_ghi=100.0,
                irradiance_dni=50.0,
                irradiance_dhi=50.0,
                wind_speed=3.0,
            )


class TestGlobalLocation:
    """Tests for GlobalLocation model."""

    def test_create_location(self, sample_location: GlobalLocation) -> None:
        """Test creating a location."""
        assert sample_location.name == "Denver"
        assert sample_location.country == "USA"
        assert sample_location.latitude == 39.7392
        assert sample_location.timezone == "America/Denver"

    def test_latitude_range(self) -> None:
        """Test latitude range validation."""
        with pytest.raises(ValueError):
            GlobalLocation(
                name="Test",
                country="Test",
                latitude=100.0,  # Invalid
                longitude=0.0,
                elevation=0.0,
                timezone="UTC",
            )

    def test_longitude_range(self) -> None:
        """Test longitude range validation."""
        with pytest.raises(ValueError):
            GlobalLocation(
                name="Test",
                country="Test",
                latitude=0.0,
                longitude=200.0,  # Invalid
                elevation=0.0,
                timezone="UTC",
            )


class TestTMYData:
    """Tests for TMYData model."""

    def test_create_tmy_data(self, sample_tmy_data: TMYData) -> None:
        """Test creating TMY data."""
        assert sample_tmy_data.location.name == "Denver"
        assert sample_tmy_data.data_source == DataSource.NSRDB
        assert len(sample_tmy_data.hourly_data) == 8760

    def test_get_annual_irradiation(self, sample_tmy_data: TMYData) -> None:
        """Test annual irradiation calculation."""
        annual_ghi = sample_tmy_data.get_annual_irradiation()
        assert annual_ghi > 0
        assert annual_ghi < 3000  # Reasonable range for most locations

    def test_get_average_temperature(self, sample_tmy_data: TMYData) -> None:
        """Test average temperature calculation."""
        avg_temp = sample_tmy_data.get_average_temperature()
        assert -50 < avg_temp < 50  # Reasonable range

    def test_year_range_validation(self, sample_location: GlobalLocation) -> None:
        """Test that end_year >= start_year."""
        with pytest.raises(ValueError):
            TMYData(
                location=sample_location,
                data_source=DataSource.NSRDB,
                start_year=2020,
                end_year=2010,  # Invalid: end < start
                hourly_data=[],
            )
