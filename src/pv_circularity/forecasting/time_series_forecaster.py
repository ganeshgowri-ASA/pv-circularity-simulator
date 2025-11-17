"""Time-Series Forecasting Module.

This module provides a comprehensive TimeSeriesForecaster class with multiple
forecasting methods including ARIMA, Prophet, LSTM, and ensemble predictions.
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM forecasting will be disabled.")

from pv_circularity.utils.validators import (
    ARIMAConfig,
    EnsembleConfig,
    ForecastMethod,
    ForecastResult,
    LSTMConfig,
    ProphetConfig,
    TimeSeriesData,
)


class TimeSeriesForecaster:
    """Production-ready time-series forecaster with multiple methods.

    This class provides a unified interface for time-series forecasting using
    various methods including ARIMA, Prophet, LSTM, and ensemble predictions.

    Attributes:
        data: Time-series data for forecasting.
        verbose: Whether to print detailed progress information.

    Example:
        >>> from datetime import datetime, timedelta
        >>> import numpy as np
        >>> from pv_circularity.forecasting import TimeSeriesForecaster
        >>> from pv_circularity.utils.validators import TimeSeriesData, ARIMAConfig
        >>>
        >>> # Create sample data
        >>> timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        >>> values = np.cumsum(np.random.randn(100)) + 50
        >>> ts_data = TimeSeriesData(timestamps=timestamps, values=values.tolist())
        >>>
        >>> # Initialize forecaster
        >>> forecaster = TimeSeriesForecaster(data=ts_data)
        >>>
        >>> # ARIMA forecast
        >>> arima_result = forecaster.arima_forecast(steps=10, config=ARIMAConfig())
        >>>
        >>> # Prophet forecast
        >>> prophet_result = forecaster.prophet_forecast(steps=10)
        >>>
        >>> # Ensemble forecast
        >>> ensemble_result = forecaster.ensemble_predictions(steps=10)
    """

    def __init__(
        self,
        data: Union[TimeSeriesData, pd.DataFrame, pd.Series],
        verbose: bool = False,
    ):
        """Initialize the TimeSeriesForecaster.

        Args:
            data: Time-series data to forecast. Can be TimeSeriesData, DataFrame, or Series.
            verbose: Whether to print detailed progress information.

        Raises:
            ValueError: If data is invalid or empty.
            TypeError: If data type is not supported.
        """
        self.verbose = verbose
        self._original_data = data
        self._fitted_models: Dict[str, Any] = {}

        # Convert data to TimeSeriesData if needed
        if isinstance(data, TimeSeriesData):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            if "value" not in data.columns:
                raise ValueError("DataFrame must have a 'value' column")
            self.data = TimeSeriesData(
                timestamps=data.index.to_pydatetime().tolist(),
                values=data["value"].tolist(),
            )
        elif isinstance(data, pd.Series):
            self.data = TimeSeriesData(
                timestamps=data.index.to_pydatetime().tolist(),
                values=data.tolist(),
                name=data.name or "time_series",
            )
        else:
            raise TypeError(
                f"Data must be TimeSeriesData, DataFrame, or Series, got {type(data)}"
            )

        if self.verbose:
            print(f"Initialized TimeSeriesForecaster with {len(self.data.values)} observations")

    def arima_forecast(
        self,
        steps: int,
        config: Optional[ARIMAConfig] = None,
        return_confidence_intervals: bool = True,
        confidence_level: float = 0.95,
    ) -> ForecastResult:
        """Perform ARIMA (AutoRegressive Integrated Moving Average) forecasting.

        This method fits an ARIMA or SARIMA model to the time series and generates
        forecasts for the specified number of steps ahead.

        Args:
            steps: Number of steps ahead to forecast.
            config: ARIMA configuration. If None, uses default configuration.
            return_confidence_intervals: Whether to include confidence intervals.
            confidence_level: Confidence level for intervals (default: 0.95).

        Returns:
            ForecastResult containing predictions, timestamps, and optional
            confidence intervals.

        Raises:
            ValueError: If steps <= 0 or confidence_level is invalid.

        Example:
            >>> config = ARIMAConfig(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
            >>> result = forecaster.arima_forecast(steps=10, config=config)
            >>> print(f"Predictions: {result.predictions}")
        """
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        if not 0 < confidence_level < 1:
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")

        if config is None:
            config = ARIMAConfig()

        if self.verbose:
            print(f"Fitting ARIMA({config.p}, {config.d}, {config.q}) model...")

        # Prepare data
        series = self.data.to_series()

        # Fit model
        if config.seasonal_order == (0, 0, 0, 0):
            # Simple ARIMA
            model = ARIMA(
                series,
                order=(config.p, config.d, config.q),
                trend=config.trend,
                enforce_stationarity=config.enforce_stationarity,
                enforce_invertibility=config.enforce_invertibility,
            )
        else:
            # SARIMA
            model = SARIMAX(
                series,
                order=(config.p, config.d, config.q),
                seasonal_order=config.seasonal_order,
                trend=config.trend,
                enforce_stationarity=config.enforce_stationarity,
                enforce_invertibility=config.enforce_invertibility,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fitted_model = model.fit(disp=False)

        self._fitted_models["arima"] = fitted_model

        if self.verbose:
            print(f"Model fitted. AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

        # Generate forecast
        forecast_output = fitted_model.get_forecast(steps=steps, alpha=1 - confidence_level)
        predictions = forecast_output.predicted_mean.tolist()

        # Generate future timestamps
        last_timestamp = self.data.timestamps[-1]
        time_delta = self._infer_time_delta()
        future_timestamps = [last_timestamp + time_delta * (i + 1) for i in range(steps)]

        # Confidence intervals
        confidence_intervals = None
        if return_confidence_intervals:
            ci = forecast_output.conf_int()
            confidence_intervals = {
                "lower": ci.iloc[:, 0].tolist(),
                "upper": ci.iloc[:, 1].tolist(),
            }

        # Performance metrics
        metrics = {
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "loglikelihood": float(fitted_model.llf),
        }

        return ForecastResult(
            method=ForecastMethod.ARIMA,
            predictions=predictions,
            timestamps=future_timestamps,
            confidence_intervals=confidence_intervals,
            metrics=metrics,
            model_params={
                "order": (config.p, config.d, config.q),
                "seasonal_order": config.seasonal_order,
                "trend": config.trend,
            },
        )

    def prophet_forecast(
        self,
        steps: int,
        config: Optional[ProphetConfig] = None,
        return_confidence_intervals: bool = True,
    ) -> ForecastResult:
        """Perform Prophet forecasting.

        Prophet is a procedure for forecasting time series data based on an
        additive model where non-linear trends are fit with yearly, weekly,
        and daily seasonality.

        Args:
            steps: Number of steps ahead to forecast.
            config: Prophet configuration. If None, uses default configuration.
            return_confidence_intervals: Whether to include confidence intervals.

        Returns:
            ForecastResult containing predictions, timestamps, and optional
            confidence intervals.

        Raises:
            ValueError: If steps <= 0.

        Example:
            >>> config = ProphetConfig(seasonality_mode='multiplicative')
            >>> result = forecaster.prophet_forecast(steps=30, config=config)
            >>> print(f"Predictions: {result.predictions}")
        """
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")

        if config is None:
            config = ProphetConfig()

        if self.verbose:
            print("Fitting Prophet model...")

        # Prepare data in Prophet format
        df = pd.DataFrame(
            {
                "ds": self.data.timestamps,
                "y": self.data.values,
            }
        )

        # Initialize Prophet model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = Prophet(
                growth=config.growth,
                changepoint_prior_scale=config.changepoint_prior_scale,
                seasonality_prior_scale=config.seasonality_prior_scale,
                seasonality_mode=config.seasonality_mode,
                yearly_seasonality=config.yearly_seasonality,
                weekly_seasonality=config.weekly_seasonality,
                daily_seasonality=config.daily_seasonality,
                holidays_prior_scale=config.holidays_prior_scale,
            )

            # Fit model
            model.fit(df)

        self._fitted_models["prophet"] = model

        if self.verbose:
            print("Prophet model fitted successfully")

        # Create future dataframe
        future = model.make_future_dataframe(periods=steps, include_history=False)

        # Generate forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast = model.predict(future)

        predictions = forecast["yhat"].tolist()
        future_timestamps = future["ds"].dt.to_pydatetime().tolist()

        # Confidence intervals
        confidence_intervals = None
        if return_confidence_intervals:
            confidence_intervals = {
                "lower": forecast["yhat_lower"].tolist(),
                "upper": forecast["yhat_upper"].tolist(),
            }

        return ForecastResult(
            method=ForecastMethod.PROPHET,
            predictions=predictions,
            timestamps=future_timestamps,
            confidence_intervals=confidence_intervals,
            model_params={
                "growth": config.growth,
                "changepoint_prior_scale": config.changepoint_prior_scale,
                "seasonality_mode": config.seasonality_mode,
            },
        )

    def lstm_forecast(
        self,
        steps: int,
        config: Optional[LSTMConfig] = None,
        return_training_history: bool = False,
    ) -> ForecastResult:
        """Perform LSTM (Long Short-Term Memory) neural network forecasting.

        LSTM is a type of recurrent neural network capable of learning long-term
        dependencies in time-series data.

        Args:
            steps: Number of steps ahead to forecast.
            config: LSTM configuration. If None, uses default configuration.
            return_training_history: Whether to include training history in metrics.

        Returns:
            ForecastResult containing predictions and timestamps.

        Raises:
            ValueError: If steps <= 0.
            RuntimeError: If TensorFlow is not available.

        Example:
            >>> config = LSTMConfig(n_layers=2, hidden_units=64, epochs=50)
            >>> result = forecaster.lstm_forecast(steps=10, config=config)
            >>> print(f"Predictions: {result.predictions}")
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM forecasting but is not installed")

        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")

        if config is None:
            config = LSTMConfig()

        if self.verbose:
            print("Preparing LSTM model...")

        # Prepare data
        series = np.array(self.data.values)

        # Normalize data
        data_mean = series.mean()
        data_std = series.std()
        series_normalized = (series - data_mean) / data_std

        # Create sequences
        X, y = self._create_sequences(series_normalized, config.lookback_window)

        if len(X) < 10:
            raise ValueError(
                f"Insufficient data for LSTM training. Need at least "
                f"{config.lookback_window + 10} observations, got {len(series)}"
            )

        # Build LSTM model
        model = Sequential()

        # Add LSTM layers
        for i in range(config.n_layers):
            return_sequences = i < config.n_layers - 1
            model.add(
                LSTM(
                    config.hidden_units,
                    return_sequences=return_sequences,
                    input_shape=(config.lookback_window, 1) if i == 0 else None,
                )
            )
            if config.dropout_rate > 0:
                model.add(Dropout(config.dropout_rate))

        # Output layer
        model.add(Dense(1))

        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss="mse")

        if self.verbose:
            print(f"Training LSTM model for {config.epochs} epochs...")
            model.summary()

        # Early stopping
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(
            X,
            y,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=[early_stop],
            verbose=1 if self.verbose else 0,
        )

        self._fitted_models["lstm"] = model

        if self.verbose:
            final_loss = history.history["loss"][-1]
            final_val_loss = history.history["val_loss"][-1]
            print(f"Training complete. Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")

        # Generate predictions
        predictions = []
        current_sequence = series_normalized[-config.lookback_window :].reshape(1, -1, 1)

        for _ in range(steps):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)

            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred

        # Denormalize predictions
        predictions = np.array(predictions) * data_std + data_mean
        predictions = predictions.tolist()

        # Generate future timestamps
        last_timestamp = self.data.timestamps[-1]
        time_delta = self._infer_time_delta()
        future_timestamps = [last_timestamp + time_delta * (i + 1) for i in range(steps)]

        # Metrics
        metrics = {
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "epochs_trained": len(history.history["loss"]),
        }

        if return_training_history:
            metrics["train_loss_history"] = [float(x) for x in history.history["loss"]]
            metrics["val_loss_history"] = [float(x) for x in history.history["val_loss"]]

        return ForecastResult(
            method=ForecastMethod.LSTM,
            predictions=predictions,
            timestamps=future_timestamps,
            metrics=metrics,
            model_params={
                "n_layers": config.n_layers,
                "hidden_units": config.hidden_units,
                "lookback_window": config.lookback_window,
            },
        )

    def ensemble_predictions(
        self,
        steps: int,
        config: Optional[EnsembleConfig] = None,
        arima_config: Optional[ARIMAConfig] = None,
        prophet_config: Optional[ProphetConfig] = None,
        lstm_config: Optional[LSTMConfig] = None,
    ) -> ForecastResult:
        """Generate ensemble predictions combining multiple forecasting methods.

        This method combines predictions from multiple forecasting models to
        produce more robust forecasts through ensemble averaging.

        Args:
            steps: Number of steps ahead to forecast.
            config: Ensemble configuration. If None, uses default configuration.
            arima_config: Configuration for ARIMA model.
            prophet_config: Configuration for Prophet model.
            lstm_config: Configuration for LSTM model.

        Returns:
            ForecastResult containing ensemble predictions and timestamps.

        Raises:
            ValueError: If steps <= 0 or configuration is invalid.

        Example:
            >>> ensemble_config = EnsembleConfig(
            ...     methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
            ...     weights=[0.6, 0.4]
            ... )
            >>> result = forecaster.ensemble_predictions(steps=10, config=ensemble_config)
            >>> print(f"Ensemble predictions: {result.predictions}")
        """
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")

        if config is None:
            config = EnsembleConfig()

        if self.verbose:
            print(f"Generating ensemble predictions with {len(config.methods)} methods...")

        # Generate predictions from each method
        all_predictions: List[ForecastResult] = []
        method_names: List[str] = []

        for method in config.methods:
            if self.verbose:
                print(f"  - Generating {method.value} forecast...")

            try:
                if method == ForecastMethod.ARIMA:
                    result = self.arima_forecast(
                        steps=steps,
                        config=arima_config,
                        return_confidence_intervals=False,
                    )
                elif method == ForecastMethod.PROPHET:
                    result = self.prophet_forecast(
                        steps=steps,
                        config=prophet_config,
                        return_confidence_intervals=False,
                    )
                elif method == ForecastMethod.LSTM:
                    result = self.lstm_forecast(
                        steps=steps,
                        config=lstm_config,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                all_predictions.append(result)
                method_names.append(method.value)

            except Exception as e:
                if self.verbose:
                    print(f"    Warning: {method.value} failed: {e}")
                continue

        if not all_predictions:
            raise ValueError("All forecasting methods failed. Cannot generate ensemble.")

        # Combine predictions
        predictions_array = np.array([p.predictions for p in all_predictions])

        if config.weights is not None:
            # Filter weights for successful methods
            weights = np.array(config.weights[: len(all_predictions)])
            weights = weights / weights.sum()  # Renormalize
        else:
            weights = np.ones(len(all_predictions)) / len(all_predictions)

        # Aggregate predictions
        if config.aggregation == "mean":
            ensemble_predictions = predictions_array.mean(axis=0).tolist()
        elif config.aggregation == "median":
            ensemble_predictions = np.median(predictions_array, axis=0).tolist()
        elif config.aggregation == "weighted":
            ensemble_predictions = np.average(predictions_array, axis=0, weights=weights).tolist()
        else:
            raise ValueError(f"Unknown aggregation method: {config.aggregation}")

        # Use timestamps from first successful prediction
        future_timestamps = all_predictions[0].timestamps

        # Calculate prediction statistics
        prediction_std = predictions_array.std(axis=0).tolist()
        prediction_min = predictions_array.min(axis=0).tolist()
        prediction_max = predictions_array.max(axis=0).tolist()

        # Metrics
        metrics = {
            "n_methods": len(all_predictions),
            "methods_used": method_names,
            "aggregation": config.aggregation,
            "prediction_std": prediction_std,
        }

        # Confidence intervals based on ensemble spread
        confidence_intervals = {
            "lower": prediction_min,
            "upper": prediction_max,
            "std": prediction_std,
        }

        return ForecastResult(
            method=ForecastMethod.ENSEMBLE,
            predictions=ensemble_predictions,
            timestamps=future_timestamps,
            confidence_intervals=confidence_intervals,
            metrics=metrics,
            model_params={
                "methods": method_names,
                "weights": weights.tolist(),
                "aggregation": config.aggregation,
            },
        )

    def _create_sequences(
        self, data: np.ndarray, lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.

        Args:
            data: Input data array.
            lookback: Number of past timesteps to use.

        Returns:
            Tuple of (X, y) arrays for training.
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)

    def _infer_time_delta(self) -> timedelta:
        """Infer the time delta between observations.

        Returns:
            Inferred time delta.
        """
        if len(self.data.timestamps) < 2:
            return timedelta(days=1)

        # Calculate median time delta
        deltas = [
            self.data.timestamps[i + 1] - self.data.timestamps[i]
            for i in range(len(self.data.timestamps) - 1)
        ]
        return sorted(deltas)[len(deltas) // 2]

    def get_fitted_model(self, method: str) -> Optional[Any]:
        """Get a fitted model by method name.

        Args:
            method: Method name ('arima', 'prophet', or 'lstm').

        Returns:
            Fitted model if available, None otherwise.
        """
        return self._fitted_models.get(method)

    def __repr__(self) -> str:
        """Return string representation of the forecaster.

        Returns:
            String representation.
        """
        return (
            f"TimeSeriesForecaster(n_observations={len(self.data.values)}, "
            f"fitted_models={list(self._fitted_models.keys())})"
        )
