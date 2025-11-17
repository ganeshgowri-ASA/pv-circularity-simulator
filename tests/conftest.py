"""
Pytest configuration and shared fixtures for the test suite.

This module provides reusable fixtures for testing the PV Circularity Simulator,
including sample data generators, mock forecast data, and test utilities.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pytest

from pv_simulator.forecasting.models import (
    ForecastPoint,
    ForecastSeries,
    ForecastData,
    AccuracyMetrics,
)


@pytest.fixture
def sample_timestamps() -> List[datetime]:
    """
    Generate sample timestamps for testing.

    Returns:
        List[datetime]: List of 24 hourly timestamps starting from 2024-01-01
    """
    start = datetime(2024, 1, 1, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(24)]


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """
    Generate sample predicted values.

    Returns:
        np.ndarray: Array of 24 predicted values with a trend
    """
    np.random.seed(42)
    base = np.linspace(100, 120, 24)
    noise = np.random.normal(0, 2, 24)
    return base + noise


@pytest.fixture
def sample_actuals(sample_predictions: np.ndarray) -> np.ndarray:
    """
    Generate sample actual values based on predictions.

    Args:
        sample_predictions: Predicted values fixture

    Returns:
        np.ndarray: Array of actual values with realistic noise
    """
    np.random.seed(43)
    noise = np.random.normal(0, 3, len(sample_predictions))
    return sample_predictions + noise


@pytest.fixture
def forecast_points_with_actuals(
    sample_timestamps: List[datetime],
    sample_predictions: np.ndarray,
    sample_actuals: np.ndarray,
) -> List[ForecastPoint]:
    """
    Generate forecast points with both predictions and actuals.

    Args:
        sample_timestamps: Timestamps fixture
        sample_predictions: Predictions fixture
        sample_actuals: Actuals fixture

    Returns:
        List[ForecastPoint]: List of forecast points with actuals
    """
    return [
        ForecastPoint(
            timestamp=ts,
            predicted=pred,
            actual=act,
        )
        for ts, pred, act in zip(sample_timestamps, sample_predictions, sample_actuals)
    ]


@pytest.fixture
def forecast_points_without_actuals(
    sample_timestamps: List[datetime],
    sample_predictions: np.ndarray,
) -> List[ForecastPoint]:
    """
    Generate forecast points with predictions only (no actuals).

    Args:
        sample_timestamps: Timestamps fixture
        sample_predictions: Predictions fixture

    Returns:
        List[ForecastPoint]: List of forecast points without actuals
    """
    return [
        ForecastPoint(
            timestamp=ts,
            predicted=pred,
        )
        for ts, pred in zip(sample_timestamps, sample_predictions)
    ]


@pytest.fixture
def forecast_points_with_bounds(
    sample_timestamps: List[datetime],
    sample_predictions: np.ndarray,
    sample_actuals: np.ndarray,
) -> List[ForecastPoint]:
    """
    Generate forecast points with confidence bounds.

    Args:
        sample_timestamps: Timestamps fixture
        sample_predictions: Predictions fixture
        sample_actuals: Actuals fixture

    Returns:
        List[ForecastPoint]: List of forecast points with bounds
    """
    return [
        ForecastPoint(
            timestamp=ts,
            predicted=pred,
            actual=act,
            lower_bound=pred - 5,
            upper_bound=pred + 5,
            confidence_level=0.95,
        )
        for ts, pred, act in zip(sample_timestamps, sample_predictions, sample_actuals)
    ]


@pytest.fixture
def forecast_series_with_actuals(
    forecast_points_with_actuals: List[ForecastPoint],
) -> ForecastSeries:
    """
    Create a forecast series with actual values.

    Args:
        forecast_points_with_actuals: Forecast points fixture

    Returns:
        ForecastSeries: Forecast series with actuals
    """
    return ForecastSeries(
        id="test-series-001",
        name="Test Forecast Series",
        points=forecast_points_with_actuals,
        model_name="Test Model",
        parameters={"test_param": 42},
    )


@pytest.fixture
def forecast_series_without_actuals(
    forecast_points_without_actuals: List[ForecastPoint],
) -> ForecastSeries:
    """
    Create a forecast series without actual values.

    Args:
        forecast_points_without_actuals: Forecast points fixture

    Returns:
        ForecastSeries: Forecast series without actuals
    """
    return ForecastSeries(
        id="test-series-002",
        name="Test Forecast Series (No Actuals)",
        points=forecast_points_without_actuals,
        model_name="Test Model",
    )


@pytest.fixture
def forecast_data_with_actuals(
    forecast_series_with_actuals: ForecastSeries,
) -> ForecastData:
    """
    Create complete forecast data with actuals.

    Args:
        forecast_series_with_actuals: Forecast series fixture

    Returns:
        ForecastData: Complete forecast data
    """
    return ForecastData(
        id="test-forecast-001",
        name="Test Forecast",
        description="Test forecast data with actuals",
        series=forecast_series_with_actuals,
        metadata={"location": "Test Site", "units": "kWh"},
    )


@pytest.fixture
def forecast_data_without_actuals(
    forecast_series_without_actuals: ForecastSeries,
) -> ForecastData:
    """
    Create complete forecast data without actuals.

    Args:
        forecast_series_without_actuals: Forecast series fixture

    Returns:
        ForecastData: Complete forecast data without actuals
    """
    return ForecastData(
        id="test-forecast-002",
        name="Test Forecast (No Actuals)",
        series=forecast_series_without_actuals,
    )


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """
    Generate sample residuals for testing confidence intervals.

    Returns:
        np.ndarray: Array of residuals
    """
    np.random.seed(44)
    return np.random.normal(0, 5, 100)


@pytest.fixture
def sample_accuracy_metrics() -> AccuracyMetrics:
    """
    Create sample accuracy metrics for testing.

    Returns:
        AccuracyMetrics: Sample metrics
    """
    return AccuracyMetrics(
        mae=5.2,
        rmse=7.1,
        mse=50.41,
        mape=3.5,
        r2_score=0.92,
        bias=-1.2,
        n_samples=100,
    )


@pytest.fixture
def perfect_forecast_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate perfect forecast (predicted == actual) for edge case testing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (predictions, actuals) - identical values
    """
    values = np.array([100.0, 110.0, 105.0, 115.0, 120.0])
    return values, values.copy()


@pytest.fixture
def forecast_with_zeros() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate forecast data with zero values for testing edge cases.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (predictions, actuals) with zeros
    """
    predictions = np.array([0.0, 10.0, 20.0, 0.0, 30.0])
    actuals = np.array([5.0, 12.0, 18.0, 0.0, 32.0])
    return predictions, actuals
