"""
Tests for weather data validator.
"""

import pytest
from datetime import datetime

from pv_simulator.models.weather import WeatherDataPoint
from pv_simulator.weather.validator import WeatherDataValidator


class TestWeatherDataValidator:
    """Tests for WeatherDataValidator."""

    @pytest.fixture
    def validator(self, test_settings):
        """Create validator instance."""
        return WeatherDataValidator(test_settings)

    def test_data_quality_checks(self, validator, sample_weather_series):
        """Test data quality checks."""
        metrics = validator.data_quality_checks(sample_weather_series)

        assert metrics.total_points == len(sample_weather_series)
        assert metrics.valid_points > 0
        assert 0.0 <= metrics.quality_score <= 1.0

    def test_outlier_detection_iqr(self, validator, sample_weather_series):
        """Test outlier detection using IQR method."""
        # Add an outlier
        outlier = sample_weather_series[0]
        outlier.temperature = 100.0  # Extreme temperature

        data_with_outliers = sample_weather_series + [outlier]

        result, count = validator.outlier_detection(
            data_with_outliers, method="iqr", threshold=1.5
        )

        assert count >= 0
        assert len(result) == len(data_with_outliers)

    def test_outlier_detection_zscore(self, validator, sample_weather_series):
        """Test outlier detection using Z-score method."""
        result, count = validator.outlier_detection(
            sample_weather_series, method="zscore", threshold=3.0
        )

        assert count >= 0
        assert len(result) == len(sample_weather_series)

    def test_gap_filling_linear(self, validator, sample_weather_series):
        """Test gap filling with linear interpolation."""
        # Create gaps by removing some data
        data_with_gaps = sample_weather_series.copy()
        data_with_gaps[5].temperature = None
        data_with_gaps[6].temperature = None

        result, count = validator.gap_filling(
            data_with_gaps, method="linear", max_gap_hours=3
        )

        assert len(result) == len(data_with_gaps)

    def test_unit_conversions_temperature(self, validator, sample_weather_data):
        """Test temperature unit conversions."""
        data = [sample_weather_data]

        # Convert to Fahrenheit
        converted = validator.unit_conversions(
            data, temp_unit="fahrenheit"
        )

        assert converted[0].temperature != sample_weather_data.temperature
        # Verify conversion (C to F formula)
        expected = sample_weather_data.temperature * 9/5 + 32
        assert abs(converted[0].temperature - expected) < 0.01

    def test_unit_conversions_wind_speed(self, validator, sample_weather_data):
        """Test wind speed unit conversions."""
        data = [sample_weather_data]

        # Convert to km/h
        converted = validator.unit_conversions(
            data, wind_unit="km_per_h"
        )

        # Verify conversion (m/s to km/h)
        expected = sample_weather_data.wind_speed * 3.6
        assert abs(converted[0].wind_speed - expected) < 0.01

    def test_timestamp_synchronization(self, validator, sample_weather_series):
        """Test timestamp synchronization."""
        result = validator.timestamp_synchronization(
            sample_weather_series,
            target_timezone="UTC",
            resample_frequency=None,
        )

        assert len(result) > 0

    def test_empty_data_handling(self, validator):
        """Test handling of empty data."""
        with pytest.raises(ValueError):
            validator.data_quality_checks([])
