"""Pytest configuration and fixtures."""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

from pv_simulator.core.schemas import TimeSeriesData, TimeSeriesFrequency


@pytest.fixture
def sample_time_series() -> TimeSeriesData:
    """Create sample time series data for testing."""
    # Generate 100 days of hourly data
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(100)]

    # Generate synthetic data with trend and seasonality
    t = np.arange(len(timestamps))
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
    noise = np.random.normal(0, 5, len(timestamps))
    values = (trend + seasonal + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency=TimeSeriesFrequency.HOURLY,
        name="test_series",
    )


@pytest.fixture
def sample_seasonal_data() -> TimeSeriesData:
    """Create sample time series with strong seasonality."""
    start_date = datetime(2020, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(365 * 2)]

    # Generate data with yearly seasonality
    t = np.arange(len(timestamps))
    trend = 1000 + 2 * t
    seasonal = 200 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 30, len(timestamps))
    values = (trend + seasonal + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency=TimeSeriesFrequency.DAILY,
        name="seasonal_series",
    )


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """Create sample weather data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.random.normal(25, 5, 100),
            "irradiance": np.random.uniform(0, 1000, 100),
            "wind_speed": np.random.uniform(0, 10, 100),
            "humidity": np.random.uniform(40, 80, 100),
        }
    )
