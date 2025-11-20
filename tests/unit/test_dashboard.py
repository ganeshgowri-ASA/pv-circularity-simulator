"""
Unit tests for the Forecast Dashboard module.

Tests cover:
- MAE and RMSE calculations
- Accuracy metrics computation
- Confidence interval generation
- Dashboard visualization
- Error handling and edge cases
"""

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import pytest

from pv_simulator.dashboard.forecast_dashboard import (
    mae_rmse_calculation,
    accuracy_metrics,
    confidence_intervals,
    ForecastDashboard,
)
from pv_simulator.forecasting.models import (
    ForecastPoint,
    ForecastSeries,
    ForecastData,
    AccuracyMetrics,
)


class TestMAERMSECalculation:
    """Tests for mae_rmse_calculation function."""

    def test_basic_calculation(self):
        """Test basic MAE and RMSE calculation."""
        actual = np.array([100, 110, 105, 115])
        predicted = np.array([98, 112, 107, 113])

        mae, rmse, mse = mae_rmse_calculation(actual, predicted)

        assert mae > 0
        assert rmse > 0
        assert mse > 0
        assert rmse == pytest.approx(np.sqrt(mse))
        assert rmse >= mae  # RMSE is always >= MAE

    def test_perfect_prediction(self, perfect_forecast_data: Tuple[np.ndarray, np.ndarray]):
        """Test with perfect predictions (zero error)."""
        predicted, actual = perfect_forecast_data

        mae, rmse, mse = mae_rmse_calculation(actual, predicted)

        assert mae == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)
        assert mse == pytest.approx(0.0)

    def test_with_lists(self):
        """Test that function works with Python lists."""
        actual = [100, 110, 105, 115]
        predicted = [98, 112, 107, 113]

        mae, rmse, mse = mae_rmse_calculation(actual, predicted)

        assert mae > 0
        assert rmse > 0
        assert isinstance(mae, float)
        assert isinstance(rmse, float)

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mae_rmse_calculation([], [])

    def test_mismatched_lengths(self):
        """Test that arrays of different lengths raise ValueError."""
        actual = np.array([100, 110, 105])
        predicted = np.array([98, 112])

        with pytest.raises(ValueError, match="length mismatch"):
            mae_rmse_calculation(actual, predicted)

    def test_nan_values(self):
        """Test that NaN values raise ValueError."""
        actual = np.array([100, np.nan, 105])
        predicted = np.array([98, 112, 107])

        with pytest.raises(ValueError, match="NaN"):
            mae_rmse_calculation(actual, predicted)

    def test_inf_values(self):
        """Test that infinite values raise ValueError."""
        actual = np.array([100, np.inf, 105])
        predicted = np.array([98, 112, 107])

        with pytest.raises(ValueError, match="infinite"):
            mae_rmse_calculation(actual, predicted)

    def test_known_values(self):
        """Test with known values for verification."""
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        predicted = np.array([12.0, 18.0, 32.0, 38.0])

        # Expected: errors = [2, -2, 2, -2]
        # MAE = (2 + 2 + 2 + 2) / 4 = 2.0
        # MSE = (4 + 4 + 4 + 4) / 4 = 4.0
        # RMSE = sqrt(4) = 2.0

        mae, rmse, mse = mae_rmse_calculation(actual, predicted)

        assert mae == pytest.approx(2.0)
        assert mse == pytest.approx(4.0)
        assert rmse == pytest.approx(2.0)


class TestAccuracyMetrics:
    """Tests for accuracy_metrics function."""

    def test_basic_metrics_calculation(self):
        """Test basic accuracy metrics calculation."""
        actual = np.array([100, 110, 105, 115, 120])
        predicted = np.array([98, 112, 107, 113, 122])

        metrics = accuracy_metrics(actual, predicted)

        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mse > 0
        assert metrics.n_samples == 5
        assert metrics.mape is not None
        assert metrics.r2_score is not None

    def test_metrics_without_mape(self):
        """Test metrics calculation without MAPE."""
        actual = np.array([100, 110, 105, 115])
        predicted = np.array([98, 112, 107, 113])

        metrics = accuracy_metrics(actual, predicted, include_mape=False)

        assert metrics.mape is None
        assert metrics.mae > 0

    def test_metrics_without_r2(self):
        """Test metrics calculation without R²."""
        actual = np.array([100, 110, 105, 115])
        predicted = np.array([98, 112, 107, 113])

        metrics = accuracy_metrics(actual, predicted, include_r2=False)

        assert metrics.r2_score is None
        assert metrics.mae > 0

    def test_bias_calculation(self):
        """Test that bias is calculated correctly."""
        actual = np.array([100.0, 100.0, 100.0])
        predicted = np.array([105.0, 105.0, 105.0])

        metrics = accuracy_metrics(actual, predicted)

        # Bias should be 5.0 (systematic over-prediction)
        assert metrics.bias == pytest.approx(5.0)

    def test_mape_with_zeros(self, forecast_with_zeros: Tuple[np.ndarray, np.ndarray]):
        """Test MAPE calculation when actuals contain zeros."""
        predicted, actual = forecast_with_zeros

        metrics = accuracy_metrics(actual, predicted, include_mape=True)

        # MAPE should be calculated excluding zero values
        assert metrics.mape is not None
        assert metrics.mape >= 0

    def test_perfect_forecast_metrics(self, perfect_forecast_data: Tuple[np.ndarray, np.ndarray]):
        """Test metrics with perfect forecast."""
        predicted, actual = perfect_forecast_data

        metrics = accuracy_metrics(actual, predicted)

        assert metrics.mae == pytest.approx(0.0)
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.mse == pytest.approx(0.0)
        assert metrics.bias == pytest.approx(0.0)
        assert metrics.mape == pytest.approx(0.0)
        assert metrics.r2_score == pytest.approx(1.0)

    def test_metrics_to_summary_dict(self, sample_accuracy_metrics: AccuracyMetrics):
        """Test conversion of metrics to summary dictionary."""
        summary = sample_accuracy_metrics.to_summary_dict()

        assert isinstance(summary, dict)
        assert "MAE" in summary
        assert "RMSE" in summary
        assert "MSE" in summary
        assert "Samples" in summary
        assert summary["Samples"] == 100


class TestConfidenceIntervals:
    """Tests for confidence_intervals function."""

    def test_basic_intervals_normal_method(self):
        """Test basic confidence interval calculation with normal method."""
        predicted = np.array([100, 110, 105, 115])

        lower, upper = confidence_intervals(
            predicted=predicted,
            confidence_level=0.95,
            method="normal",
        )

        assert len(lower) == len(predicted)
        assert len(upper) == len(predicted)
        assert np.all(lower <= predicted)
        assert np.all(upper >= predicted)
        assert np.all(lower < upper)

    def test_intervals_with_residuals(self, sample_residuals: np.ndarray):
        """Test intervals with provided residuals."""
        predicted = np.array([100, 110, 105, 115])

        lower, upper = confidence_intervals(
            predicted=predicted,
            residuals=sample_residuals,
            confidence_level=0.95,
            method="normal",
        )

        assert len(lower) == len(predicted)
        assert len(upper) == len(predicted)
        assert np.all(lower < upper)

    def test_percentile_method(self, sample_residuals: np.ndarray):
        """Test confidence intervals with percentile method."""
        predicted = np.array([100, 110, 105, 115])

        lower, upper = confidence_intervals(
            predicted=predicted,
            residuals=sample_residuals,
            confidence_level=0.95,
            method="percentile",
        )

        assert len(lower) == len(predicted)
        assert len(upper) == len(predicted)
        assert np.all(lower < upper)

    def test_bootstrap_method(self, sample_residuals: np.ndarray):
        """Test confidence intervals with bootstrap method."""
        predicted = np.array([100, 110, 105, 115, 120])

        lower, upper = confidence_intervals(
            predicted=predicted,
            residuals=sample_residuals,
            confidence_level=0.95,
            method="bootstrap",
        )

        assert len(lower) == len(predicted)
        assert len(upper) == len(predicted)

    def test_different_confidence_levels(self):
        """Test that different confidence levels produce different intervals."""
        predicted = np.array([100, 110, 105, 115])

        lower_90, upper_90 = confidence_intervals(
            predicted=predicted,
            confidence_level=0.90,
        )

        lower_95, upper_95 = confidence_intervals(
            predicted=predicted,
            confidence_level=0.95,
        )

        # 95% intervals should be wider than 90%
        assert np.all((upper_95 - lower_95) >= (upper_90 - lower_90))

    def test_invalid_confidence_level(self):
        """Test that invalid confidence levels raise ValueError."""
        predicted = np.array([100, 110, 105])

        with pytest.raises(ValueError, match="between 0 and 1"):
            confidence_intervals(predicted, confidence_level=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            confidence_intervals(predicted, confidence_level=0.0)

    def test_invalid_method(self):
        """Test that invalid methods raise ValueError."""
        predicted = np.array([100, 110, 105])

        with pytest.raises(ValueError, match="Unsupported method"):
            confidence_intervals(predicted, method="invalid_method")

    def test_bootstrap_without_residuals(self):
        """Test that bootstrap method requires residuals."""
        predicted = np.array([100, 110, 105])

        with pytest.raises(ValueError, match="requires residuals"):
            confidence_intervals(predicted, method="bootstrap")

    def test_empty_predicted(self):
        """Test that empty predicted array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            confidence_intervals(np.array([]))

    def test_insufficient_residuals(self):
        """Test that insufficient residuals raise ValueError."""
        predicted = np.array([100, 110, 105])
        residuals = np.array([1.0])  # Only 1 residual

        with pytest.raises(ValueError, match="at least 2 residuals"):
            confidence_intervals(predicted, residuals=residuals)


class TestForecastDashboard:
    """Tests for ForecastDashboard class."""

    def test_dashboard_initialization(self, forecast_data_with_actuals: ForecastData):
        """Test dashboard initialization."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        assert dashboard.forecast_data == forecast_data_with_actuals
        assert dashboard.title == "Forecast Dashboard"
        assert dashboard.width == 1400
        assert dashboard.height == 1000

    def test_dashboard_custom_params(self, forecast_data_with_actuals: ForecastData):
        """Test dashboard initialization with custom parameters."""
        dashboard = ForecastDashboard(
            forecast_data_with_actuals,
            title="Custom Title",
            width=1200,
            height=800,
        )

        assert dashboard.title == "Custom Title"
        assert dashboard.width == 1200
        assert dashboard.height == 800

    def test_invalid_forecast_data(self):
        """Test that invalid forecast data raises ValueError."""
        with pytest.raises(ValueError, match="must be a ForecastData instance"):
            ForecastDashboard("not a forecast data")

    def test_metrics_auto_calculation(self, forecast_data_with_actuals: ForecastData):
        """Test that metrics are automatically calculated when actuals exist."""
        # Ensure metrics are not pre-calculated
        forecast_data_with_actuals.metrics = None

        dashboard = ForecastDashboard(forecast_data_with_actuals)

        assert dashboard.forecast_data.metrics is not None
        assert isinstance(dashboard.forecast_data.metrics, AccuracyMetrics)

    def test_forecast_visualization(self, forecast_data_with_actuals: ForecastData):
        """Test forecast visualization creation."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.forecast_visualization()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_visualization_without_actuals(self, forecast_data_without_actuals: ForecastData):
        """Test visualization when actuals are not available."""
        dashboard = ForecastDashboard(forecast_data_without_actuals)

        fig = dashboard.forecast_visualization(show_actuals=True)

        assert isinstance(fig, go.Figure)
        # Should still create visualization without actuals

    def test_visualization_with_confidence(self, forecast_data_with_actuals: ForecastData):
        """Test visualization with confidence intervals."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.forecast_visualization(
            show_confidence=True,
            confidence_level=0.95,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_visualization_without_confidence(self, forecast_data_with_actuals: ForecastData):
        """Test visualization without confidence intervals."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.forecast_visualization(show_confidence=False)

        assert isinstance(fig, go.Figure)

    def test_error_analysis(self, forecast_data_with_actuals: ForecastData):
        """Test error analysis visualization."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.create_error_analysis()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_error_analysis_without_actuals(self, forecast_data_without_actuals: ForecastData):
        """Test that error analysis returns None without actuals."""
        dashboard = ForecastDashboard(forecast_data_without_actuals)

        fig = dashboard.create_error_analysis()

        assert fig is None

    def test_metrics_table(self, forecast_data_with_actuals: ForecastData):
        """Test metrics table creation."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.create_metrics_table()

        assert isinstance(fig, go.Figure)

    def test_metrics_table_without_metrics(self, forecast_data_without_actuals: ForecastData):
        """Test that metrics table returns None without metrics."""
        dashboard = ForecastDashboard(forecast_data_without_actuals)

        fig = dashboard.create_metrics_table()

        assert fig is None

    def test_create_full_dashboard(self, forecast_data_with_actuals: ForecastData):
        """Test creation of full dashboard."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.create_dashboard(
            show_confidence=True,
            show_actuals=True,
            show_error_analysis=True,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_dashboard_without_error_analysis(self, forecast_data_with_actuals: ForecastData):
        """Test dashboard without error analysis."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        fig = dashboard.create_dashboard(show_error_analysis=False)

        assert isinstance(fig, go.Figure)

    def test_export_html(self, forecast_data_with_actuals: ForecastData, tmp_path):
        """Test HTML export functionality."""
        dashboard = ForecastDashboard(forecast_data_with_actuals)

        output_file = tmp_path / "test_dashboard.html"
        dashboard.export_html(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_dashboard_with_forecast_bounds(self):
        """Test dashboard with pre-calculated forecast bounds."""
        # Create forecast with bounds
        points = [
            ForecastPoint(
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                predicted=100 + i,
                actual=100 + i + np.random.randn(),
                lower_bound=95 + i,
                upper_bound=105 + i,
                confidence_level=0.95,
            )
            for i in range(24)
        ]

        series = ForecastSeries(
            points=points,
            model_name="Test Model",
        )

        forecast_data = ForecastData(series=series)
        dashboard = ForecastDashboard(forecast_data)

        fig = dashboard.forecast_visualization(show_confidence=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


@pytest.mark.integration
class TestDashboardIntegration:
    """Integration tests for the dashboard with real-world scenarios."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create forecast data
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(48)]
        np.random.seed(42)
        predicted = 100 + np.linspace(0, 20, 48) + np.random.normal(0, 2, 48)
        actual = predicted + np.random.normal(0, 3, 48)

        points = [
            ForecastPoint(timestamp=ts, predicted=pred, actual=act)
            for ts, pred, act in zip(timestamps, predicted, actual)
        ]

        series = ForecastSeries(
            id="integration-test",
            name="48-hour forecast",
            points=points,
            model_name="ARIMA",
        )

        forecast_data = ForecastData(
            id="integration-forecast",
            series=series,
            metadata={"test": True},
        )

        # Create dashboard
        dashboard = ForecastDashboard(forecast_data, title="Integration Test Dashboard")

        # Verify metrics were calculated
        assert dashboard.forecast_data.metrics is not None

        # Create all visualizations
        viz_fig = dashboard.forecast_visualization(show_confidence=True)
        error_fig = dashboard.create_error_analysis()
        metrics_fig = dashboard.create_metrics_table()
        full_dashboard = dashboard.create_dashboard()

        # Verify all figures were created
        assert isinstance(viz_fig, go.Figure)
        assert isinstance(error_fig, go.Figure)
        assert isinstance(metrics_fig, go.Figure)
        assert isinstance(full_dashboard, go.Figure)

        # Verify metrics are reasonable
        metrics = dashboard.forecast_data.metrics
        assert metrics.mae > 0
        assert metrics.rmse >= metrics.mae
        assert metrics.n_samples == 48

    def test_multiple_forecast_comparison(self):
        """Test dashboard with multiple forecast scenarios."""
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(24)]

        # Create two different forecasts
        np.random.seed(42)
        predicted1 = 100 + np.linspace(0, 10, 24) + np.random.normal(0, 1, 24)
        predicted2 = 100 + np.linspace(0, 10, 24) + np.random.normal(0, 2, 24)
        actual = 100 + np.linspace(0, 10, 24) + np.random.normal(0, 1.5, 24)

        # Create first forecast
        points1 = [
            ForecastPoint(timestamp=ts, predicted=pred, actual=act)
            for ts, pred, act in zip(timestamps, predicted1, actual)
        ]
        series1 = ForecastSeries(points=points1, model_name="Model A")
        forecast1 = ForecastData(series=series1)
        dashboard1 = ForecastDashboard(forecast1, title="Model A")

        # Create second forecast
        points2 = [
            ForecastPoint(timestamp=ts, predicted=pred, actual=act)
            for ts, pred, act in zip(timestamps, predicted2, actual)
        ]
        series2 = ForecastSeries(points=points2, model_name="Model B")
        forecast2 = ForecastData(series=series2)
        dashboard2 = ForecastDashboard(forecast2, title="Model B")

        # Both should have valid metrics
        assert dashboard1.forecast_data.metrics is not None
        assert dashboard2.forecast_data.metrics is not None

        # Model A should be more accurate (lower RMSE)
        assert dashboard1.forecast_data.metrics.rmse < dashboard2.forecast_data.metrics.rmse
