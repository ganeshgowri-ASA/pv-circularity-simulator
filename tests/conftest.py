"""
Pytest configuration and fixtures for PV Simulator tests.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pv_simulator.models.weather import (
    DataQuality,
    DataSource,
    GlobalLocation,
    TemporalResolution,
    TMYData,
    TMYFormat,
    WeatherDataPoint,
)


@pytest.fixture
def sample_location() -> GlobalLocation:
    """Sample location for testing (Denver, CO)."""
    return GlobalLocation(
        name="Denver",
        country="USA",
        latitude=39.7392,
        longitude=-104.9903,
        elevation=1609.0,
        timezone="America/Denver",
        climate_zone="BSk",
    )


@pytest.fixture
def sample_weather_point() -> WeatherDataPoint:
    """Sample weather data point for testing."""
    return WeatherDataPoint(
        timestamp=datetime(2023, 7, 15, 12, 0, 0),
        temperature=28.5,
        irradiance_ghi=950.0,
        irradiance_dni=850.0,
        irradiance_dhi=150.0,
        wind_speed=3.5,
        wind_direction=180.0,
        relative_humidity=45.0,
        pressure=101325.0,
        albedo=0.2,
        precipitable_water=1.5,
        cloud_cover=0.2,
        data_quality=DataQuality.EXCELLENT,
    )


@pytest.fixture
def sample_hourly_data() -> list[WeatherDataPoint]:
    """Sample hourly data for one year (8760 hours)."""
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    data_points = []

    for hour in range(8760):
        timestamp = base_time + timedelta(hours=hour)

        # Simple sinusoidal pattern for realistic variation
        hour_of_day = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday

        # Temperature: varies by time of day and season
        temp = 15.0 + 10.0 * (1 + (day_of_year - 180) / 365) - 5.0 * abs(hour_of_day - 14) / 14

        # GHI: zero at night, peak at noon
        if 6 <= hour_of_day <= 18:
            ghi = 800.0 * (1 - abs(hour_of_day - 12) / 6) * (1 + 0.2 * (day_of_year - 180) / 365)
        else:
            ghi = 0.0

        # DNI and DHI
        dni = ghi * 0.8 if ghi > 0 else 0.0
        dhi = ghi * 0.2 if ghi > 0 else 0.0

        point = WeatherDataPoint(
            timestamp=timestamp,
            temperature=temp,
            irradiance_ghi=max(0, ghi),
            irradiance_dni=max(0, dni),
            irradiance_dhi=max(0, dhi),
            wind_speed=3.0 + 2.0 * (hour_of_day / 24),
            wind_direction=180.0,
            relative_humidity=50.0,
            pressure=101325.0,
        )
        data_points.append(point)

    return data_points


@pytest.fixture
def sample_tmy_data(sample_location: GlobalLocation, sample_hourly_data: list[WeatherDataPoint]) -> TMYData:
    """Sample TMY data for testing."""
    return TMYData(
        location=sample_location,
        data_source=DataSource.NSRDB,
        format_type=TMYFormat.TMY3,
        temporal_resolution=TemporalResolution.HOURLY,
        start_year=2007,
        end_year=2021,
        hourly_data=sample_hourly_data,
        metadata={"test": "data"},
        data_quality=DataQuality.EXCELLENT,
        completeness_percentage=100.0,
    )


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_tmy(temp_data_dir: Path, sample_tmy_data: TMYData) -> Path:
    """Create sample CSV TMY file for testing."""
    import pandas as pd

    csv_path = temp_data_dir / "sample_tmy.csv"

    # Convert TMY data to DataFrame
    df = pd.DataFrame([point.model_dump() for point in sample_tmy_data.hourly_data])

    # Save to CSV
    df.to_csv(csv_path, index=False)

    return csv_path
