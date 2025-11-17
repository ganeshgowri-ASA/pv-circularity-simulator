"""Tests for statistical forecasting models."""

import pytest

from pv_simulator.core.schemas import TimeSeriesData
from pv_simulator.forecasting.statistical import (
    ARIMAModel,
    SARIMAModel,
    StatisticalAnalyzer,
    exponential_smoothing,
)


class TestARIMAModel:
    """Tests for ARIMA forecasting model."""

    def test_fit_predict(self, sample_time_series: TimeSeriesData) -> None:
        """Test ARIMA model fitting and prediction."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(sample_time_series)

        assert model.is_fitted
        assert model.model is not None

        # Generate forecast
        forecast = model.predict(horizon=24)

        assert len(forecast.predictions) == 24
        assert len(forecast.timestamps) == 24
        assert forecast.lower_bound is not None
        assert forecast.upper_bound is not None

    def test_evaluate(self, sample_time_series: TimeSeriesData) -> None:
        """Test model evaluation."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(sample_time_series)

        # Split data for evaluation
        test_data = TimeSeriesData(
            timestamps=sample_time_series.timestamps[-24:],
            values=sample_time_series.values[-24:],
            frequency=sample_time_series.frequency,
        )

        forecast = model.predict(horizon=24)
        metrics = model.evaluate(test_data, forecast)

        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0


class TestSARIMAModel:
    """Tests for SARIMA forecasting model."""

    def test_fit_predict(self, sample_seasonal_data: TimeSeriesData) -> None:
        """Test SARIMA model fitting and prediction."""
        model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        model.fit(sample_seasonal_data)

        assert model.is_fitted

        forecast = model.predict(horizon=30)
        assert len(forecast.predictions) == 30


class TestStatisticalAnalyzer:
    """Tests for statistical analysis tools."""

    def test_seasonality_decomposition(
        self, sample_seasonal_data: TimeSeriesData
    ) -> None:
        """Test seasonal decomposition."""
        analyzer = StatisticalAnalyzer()
        decomposition = analyzer.seasonality_decomposition(
            sample_seasonal_data, model="additive", period=365
        )

        assert len(decomposition.trend) == len(sample_seasonal_data.values)
        assert len(decomposition.seasonal) == len(sample_seasonal_data.values)
        assert 0 <= decomposition.seasonality_strength <= 1
        assert 0 <= decomposition.trend_strength <= 1

    def test_trend_analysis(self, sample_time_series: TimeSeriesData) -> None:
        """Test trend analysis."""
        analyzer = StatisticalAnalyzer()
        trend = analyzer.trend_analysis(sample_time_series)

        assert "slope" in trend
        assert "r_squared" in trend
        assert "trend_direction" in trend
        assert trend["trend_direction"] in ["increasing", "decreasing"]

    def test_autocorrelation(self, sample_time_series: TimeSeriesData) -> None:
        """Test autocorrelation calculation."""
        analyzer = StatisticalAnalyzer()
        acf_result = analyzer.autocorrelation(sample_time_series, nlags=40)

        assert "acf" in acf_result
        assert "pacf" in acf_result
        assert len(acf_result["acf"]) == 41  # nlags + 1
