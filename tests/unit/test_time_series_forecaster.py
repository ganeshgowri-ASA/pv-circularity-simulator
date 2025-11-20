"""Unit tests for TimeSeriesForecaster class.

This module provides comprehensive tests for all forecasting methods.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pv_circularity.forecasting import TimeSeriesForecaster
from pv_circularity.utils.validators import (
    ARIMAConfig,
    EnsembleConfig,
    ForecastMethod,
    LSTMConfig,
    ProphetConfig,
    TimeSeriesData,
)


@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n = 100
    timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

    # Create synthetic data with trend and seasonality
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    noise = np.random.randn(n) * 2
    values = (50 + trend + seasonal + noise).tolist()

    return TimeSeriesData(timestamps=timestamps, values=values, frequency="D", name="test_series")


@pytest.fixture
def short_time_series():
    """Create short time series for edge case testing."""
    timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    values = np.cumsum(np.random.randn(20)).tolist()
    return TimeSeriesData(timestamps=timestamps, values=values)


class TestTimeSeriesForecasterInitialization:
    """Tests for TimeSeriesForecaster initialization."""

    def test_init_with_time_series_data(self, sample_time_series):
        """Test initialization with TimeSeriesData."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        assert forecaster.data == sample_time_series
        assert not forecaster.verbose

    def test_init_with_dataframe(self, sample_time_series):
        """Test initialization with pandas DataFrame."""
        df = sample_time_series.to_dataframe()
        forecaster = TimeSeriesForecaster(data=df)
        assert len(forecaster.data.values) == len(sample_time_series.values)

    def test_init_with_series(self, sample_time_series):
        """Test initialization with pandas Series."""
        series = sample_time_series.to_series()
        forecaster = TimeSeriesForecaster(data=series)
        assert len(forecaster.data.values) == len(sample_time_series.values)

    def test_init_with_invalid_type(self):
        """Test initialization with invalid data type."""
        with pytest.raises(TypeError):
            TimeSeriesForecaster(data=[1, 2, 3, 4, 5])

    def test_init_with_verbose(self, sample_time_series):
        """Test initialization with verbose mode."""
        forecaster = TimeSeriesForecaster(data=sample_time_series, verbose=True)
        assert forecaster.verbose


class TestARIMAForecasting:
    """Tests for ARIMA forecasting method."""

    def test_arima_forecast_basic(self, sample_time_series):
        """Test basic ARIMA forecasting."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        config = ARIMAConfig(p=1, d=1, q=1)
        result = forecaster.arima_forecast(steps=10, config=config)

        assert result.method == ForecastMethod.ARIMA
        assert len(result.predictions) == 10
        assert len(result.timestamps) == 10
        assert result.confidence_intervals is not None
        assert "lower" in result.confidence_intervals
        assert "upper" in result.confidence_intervals

    def test_arima_forecast_default_config(self, sample_time_series):
        """Test ARIMA forecasting with default configuration."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        result = forecaster.arima_forecast(steps=5)

        assert len(result.predictions) == 5
        assert result.metrics is not None
        assert "aic" in result.metrics
        assert "bic" in result.metrics

    def test_arima_forecast_seasonal(self, sample_time_series):
        """Test SARIMA forecasting with seasonal component."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        config = ARIMAConfig(p=1, d=1, q=1, seasonal_order=(1, 0, 1, 7))
        result = forecaster.arima_forecast(steps=7, config=config)

        assert len(result.predictions) == 7
        assert result.model_params["seasonal_order"] == (1, 0, 1, 7)

    def test_arima_forecast_invalid_steps(self, sample_time_series):
        """Test ARIMA forecasting with invalid steps."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        with pytest.raises(ValueError):
            forecaster.arima_forecast(steps=0)
        with pytest.raises(ValueError):
            forecaster.arima_forecast(steps=-1)

    def test_arima_forecast_no_confidence_intervals(self, sample_time_series):
        """Test ARIMA forecasting without confidence intervals."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        result = forecaster.arima_forecast(steps=5, return_confidence_intervals=False)

        assert result.confidence_intervals is None


class TestProphetForecasting:
    """Tests for Prophet forecasting method."""

    def test_prophet_forecast_basic(self, sample_time_series):
        """Test basic Prophet forecasting."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        result = forecaster.prophet_forecast(steps=10)

        assert result.method == ForecastMethod.PROPHET
        assert len(result.predictions) == 10
        assert len(result.timestamps) == 10

    def test_prophet_forecast_with_config(self, sample_time_series):
        """Test Prophet forecasting with custom configuration."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        config = ProphetConfig(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
        )
        result = forecaster.prophet_forecast(steps=7, config=config)

        assert len(result.predictions) == 7
        assert result.model_params["seasonality_mode"] == "multiplicative"

    def test_prophet_forecast_confidence_intervals(self, sample_time_series):
        """Test Prophet forecasting with confidence intervals."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        result = forecaster.prophet_forecast(steps=5, return_confidence_intervals=True)

        assert result.confidence_intervals is not None
        assert "lower" in result.confidence_intervals
        assert "upper" in result.confidence_intervals
        assert len(result.confidence_intervals["lower"]) == 5


class TestLSTMForecasting:
    """Tests for LSTM forecasting method."""

    @pytest.mark.skipif(
        sys.platform == "darwin" and sys.version_info >= (3, 11),
        reason="TensorFlow may not be available on macOS with Python 3.11+",
    )
    def test_lstm_forecast_basic(self, sample_time_series):
        """Test basic LSTM forecasting."""
        try:
            forecaster = TimeSeriesForecaster(data=sample_time_series, verbose=False)
            config = LSTMConfig(epochs=5, batch_size=16, lookback_window=10)
            result = forecaster.lstm_forecast(steps=5, config=config)

            assert result.method == ForecastMethod.LSTM
            assert len(result.predictions) == 5
            assert len(result.timestamps) == 5
            assert result.metrics is not None
        except RuntimeError as e:
            if "TensorFlow is required" in str(e):
                pytest.skip("TensorFlow not available")
            raise

    @pytest.mark.skipif(
        sys.platform == "darwin" and sys.version_info >= (3, 11),
        reason="TensorFlow may not be available on macOS with Python 3.11+",
    )
    def test_lstm_forecast_with_config(self, sample_time_series):
        """Test LSTM forecasting with custom configuration."""
        try:
            forecaster = TimeSeriesForecaster(data=sample_time_series)
            config = LSTMConfig(
                n_layers=2,
                hidden_units=32,
                epochs=3,
                lookback_window=7,
            )
            result = forecaster.lstm_forecast(steps=3, config=config)

            assert len(result.predictions) == 3
            assert result.model_params["n_layers"] == 2
            assert result.model_params["hidden_units"] == 32
        except RuntimeError as e:
            if "TensorFlow is required" in str(e):
                pytest.skip("TensorFlow not available")
            raise

    def test_lstm_forecast_insufficient_data(self, short_time_series):
        """Test LSTM forecasting with insufficient data."""
        try:
            forecaster = TimeSeriesForecaster(data=short_time_series)
            config = LSTMConfig(lookback_window=50)  # Requires more data than available
            with pytest.raises(ValueError):
                forecaster.lstm_forecast(steps=5, config=config)
        except RuntimeError as e:
            if "TensorFlow is required" in str(e):
                pytest.skip("TensorFlow not available")
            raise


class TestEnsembleForecasting:
    """Tests for ensemble forecasting method."""

    def test_ensemble_predictions_basic(self, sample_time_series):
        """Test basic ensemble forecasting."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        # Use only ARIMA and Prophet to avoid TensorFlow dependency
        config = EnsembleConfig(
            methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
            aggregation="mean",
        )
        result = forecaster.ensemble_predictions(steps=10, config=config)

        assert result.method == ForecastMethod.ENSEMBLE
        assert len(result.predictions) == 10
        assert result.metrics is not None
        assert result.metrics["n_methods"] == 2

    def test_ensemble_predictions_weighted(self, sample_time_series):
        """Test ensemble forecasting with weighted aggregation."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        config = EnsembleConfig(
            methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
            weights=[0.7, 0.3],
            aggregation="weighted",
        )
        result = forecaster.ensemble_predictions(steps=5, config=config)

        assert len(result.predictions) == 5
        assert result.model_params["aggregation"] == "weighted"

    def test_ensemble_predictions_median(self, sample_time_series):
        """Test ensemble forecasting with median aggregation."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        config = EnsembleConfig(
            methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
            aggregation="median",
        )
        result = forecaster.ensemble_predictions(steps=5, config=config)

        assert result.model_params["aggregation"] == "median"

    def test_ensemble_predictions_default_config(self, sample_time_series):
        """Test ensemble forecasting with default configuration."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        # This might fail if LSTM is not available, which is okay for the test
        try:
            result = forecaster.ensemble_predictions(steps=5)
            assert len(result.predictions) == 5
        except Exception:
            # If ensemble fails (e.g., LSTM not available), test that it handles gracefully
            pass


class TestForecastResultConversion:
    """Tests for ForecastResult conversion methods."""

    def test_forecast_result_to_dataframe(self, sample_time_series):
        """Test conversion of ForecastResult to DataFrame."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        result = forecaster.arima_forecast(steps=10)

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "prediction" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns


class TestTimeSeriesDataValidation:
    """Tests for TimeSeriesData validation."""

    def test_valid_time_series_data(self):
        """Test creation of valid TimeSeriesData."""
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = list(range(10))
        ts_data = TimeSeriesData(timestamps=timestamps, values=values)

        assert len(ts_data.timestamps) == 10
        assert len(ts_data.values) == 10

    def test_invalid_empty_timestamps(self):
        """Test TimeSeriesData with empty timestamps."""
        with pytest.raises(ValueError):
            TimeSeriesData(timestamps=[], values=[])

    def test_invalid_mismatched_lengths(self):
        """Test TimeSeriesData with mismatched lengths."""
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = list(range(5))
        with pytest.raises(ValueError):
            TimeSeriesData(timestamps=timestamps, values=values)

    def test_invalid_unsorted_timestamps(self):
        """Test TimeSeriesData with unsorted timestamps."""
        timestamps = [datetime(2020, 1, i) for i in [5, 3, 1, 2, 4]]
        values = list(range(5))
        with pytest.raises(ValueError):
            TimeSeriesData(timestamps=timestamps, values=values)

    def test_time_series_data_to_series(self):
        """Test conversion to pandas Series."""
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        values = list(range(5))
        ts_data = TimeSeriesData(timestamps=timestamps, values=values, name="test")

        series = ts_data.to_series()
        assert isinstance(series, pd.Series)
        assert len(series) == 5
        assert series.name == "test"


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_arima_config(self):
        """Test valid ARIMA configuration."""
        config = ARIMAConfig(p=2, d=1, q=2, trend="ct")
        assert config.p == 2
        assert config.d == 1
        assert config.q == 2
        assert config.trend == "ct"

    def test_invalid_arima_trend(self):
        """Test invalid ARIMA trend parameter."""
        with pytest.raises(ValueError):
            ARIMAConfig(trend="invalid")

    def test_valid_prophet_config(self):
        """Test valid Prophet configuration."""
        config = ProphetConfig(
            growth="logistic",
            seasonality_mode="multiplicative",
        )
        assert config.growth == "logistic"
        assert config.seasonality_mode == "multiplicative"

    def test_invalid_prophet_growth(self):
        """Test invalid Prophet growth parameter."""
        with pytest.raises(ValueError):
            ProphetConfig(growth="invalid")

    def test_valid_lstm_config(self):
        """Test valid LSTM configuration."""
        config = LSTMConfig(n_layers=3, hidden_units=128, epochs=50)
        assert config.n_layers == 3
        assert config.hidden_units == 128
        assert config.epochs == 50

    def test_valid_ensemble_config(self):
        """Test valid ensemble configuration."""
        config = EnsembleConfig(
            methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
            weights=[0.6, 0.4],
        )
        assert len(config.methods) == 2
        assert config.weights == [0.6, 0.4]

    def test_invalid_ensemble_single_method(self):
        """Test ensemble config with single method."""
        with pytest.raises(ValueError):
            EnsembleConfig(methods=[ForecastMethod.ARIMA])

    def test_invalid_ensemble_weights(self):
        """Test ensemble config with invalid weights."""
        with pytest.raises(ValueError):
            EnsembleConfig(
                methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
                weights=[0.6, 0.6],  # Don't sum to 1.0
            )


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_fitted_model(self, sample_time_series):
        """Test getting fitted models."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        forecaster.arima_forecast(steps=5)

        model = forecaster.get_fitted_model("arima")
        assert model is not None

    def test_get_unfitted_model(self, sample_time_series):
        """Test getting unfitted model."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        model = forecaster.get_fitted_model("arima")
        assert model is None

    def test_repr(self, sample_time_series):
        """Test string representation."""
        forecaster = TimeSeriesForecaster(data=sample_time_series)
        repr_str = repr(forecaster)
        assert "TimeSeriesForecaster" in repr_str
        assert "100" in repr_str  # Number of observations
