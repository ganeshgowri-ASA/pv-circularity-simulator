"""
Machine Learning forecasting models.

This module implements ML-based forecasting methods including LSTM neural networks,
Prophet, XGBoost, LightGBM, and ensemble methods for time series prediction.
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from pv_simulator.core.models import BaseForecaster
from pv_simulator.core.schemas import (
    ForecastResult,
    ModelMetrics,
    ModelType,
    TimeSeriesData,
)
from pv_simulator.forecasting.feature_engineering import FeatureEngineering


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecasting model.

    Prophet is designed for forecasting time series with strong seasonal patterns
    and multiple seasonality (daily, weekly, yearly).

    Args:
        seasonality_mode: 'additive' or 'multiplicative'
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
        **kwargs: Additional Prophet parameters
    """

    def __init__(
        self,
        seasonality_mode: str = "additive",
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize Prophet forecaster."""
        super().__init__(
            ModelType.PROPHET,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **kwargs,
        )
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "ProphetForecaster":
        """
        Fit Prophet model to time series data.

        Args:
            data: Time series data
            exogenous: Optional additional regressors
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        # Prepare data in Prophet format
        df = pd.DataFrame({"ds": data.timestamps, "y": data.values})

        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.params,
        )

        # Add exogenous regressors if provided
        if exogenous is not None:
            for col in exogenous.columns:
                self.model.add_regressor(col)
                df[col] = exogenous[col].values

        # Fit model
        self.model.fit(df, **kwargs)
        self.is_fitted = True

        return self

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts using Prophet.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables for forecast period
            confidence_level: Confidence level for intervals
            **kwargs: Additional parameters

        Returns:
            ForecastResult with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon)

        # Add exogenous variables if provided
        if exogenous is not None:
            for col in exogenous.columns:
                # Extend with future values
                future[col] = list(self.model.history[col]) + list(exogenous[col])

        # Generate forecast
        forecast = self.model.predict(future)

        # Extract only the forecast periods
        forecast = forecast.tail(horizon)

        predictions = forecast["yhat"].tolist()
        lower_bound = forecast["yhat_lower"].tolist()
        upper_bound = forecast["yhat_upper"].tolist()
        timestamps = forecast["ds"].dt.to_pydatetime().tolist()

        return ForecastResult(
            model_type=self.model_type,
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={"seasonality_mode": self.seasonality_mode},
        )

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate Prophet model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)

    def prophet_forecasting(
        self, data: TimeSeriesData, horizon: int, **kwargs: Any
    ) -> ForecastResult:
        """
        Convenience method for quick Prophet forecasting.

        Args:
            data: Time series data
            horizon: Forecast horizon
            **kwargs: Additional parameters

        Returns:
            ForecastResult
        """
        self.fit(data, **kwargs)
        return self.predict(horizon)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecasting model.

    XGBoost is a gradient boosting model that works well with engineered
    time series features.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        **kwargs: Additional XGBoost parameters
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        **kwargs: Any,
    ) -> None:
        """Initialize XGBoost forecaster."""
        super().__init__(
            ModelType.XGBOOST,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs,
        )
        self.feature_engineer = FeatureEngineering()
        self.feature_columns: List[str] = []

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "XGBoostForecaster":
        """
        Fit XGBoost model with engineered features.

        Args:
            data: Time series data
            exogenous: Optional exogenous variables
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        # Create features
        df = self.feature_engineer.create_all_features(
            data, target_col="value", weather_data=exogenous
        )

        # Prepare training data
        df = df.dropna()  # Remove rows with NaN from lag features

        # Separate features and target
        self.feature_columns = [
            col for col in df.columns if col not in ["timestamp", "value"]
        ]
        X = df[self.feature_columns].values
        y = df["value"].values

        # Initialize and fit XGBoost
        self.model = XGBRegressor(
            n_estimators=self.params.get("n_estimators", 100),
            learning_rate=self.params.get("learning_rate", 0.1),
            max_depth=self.params.get("max_depth", 6),
            **{k: v for k, v in self.params.items() if k not in ["n_estimators", "learning_rate", "max_depth"]},
        )
        self.model.fit(X, y, **kwargs)

        # Store last values for recursive prediction
        self.last_values = df
        self.is_fitted = True

        return self

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts using XGBoost.

        Uses recursive multi-step forecasting.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables
            confidence_level: Confidence level (not used for point forecasts)
            **kwargs: Additional parameters

        Returns:
            ForecastResult with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        current_data = self.last_values.copy()

        for step in range(horizon):
            # Get latest features
            latest_features = current_data[self.feature_columns].iloc[-1:].values

            # Predict next value
            next_pred = self.model.predict(latest_features)[0]
            predictions.append(next_pred)

            # Update data for next prediction
            last_timestamp = current_data["timestamp"].iloc[-1]
            freq = pd.infer_freq(current_data["timestamp"])
            next_timestamp = last_timestamp + pd.Timedelta(1, unit=freq or "H")

            # Create new row
            new_row = pd.DataFrame(
                {"timestamp": [next_timestamp], "value": [next_pred]}
            )

            # Recreate features for new row
            temp_data = pd.concat([current_data, new_row], ignore_index=True)
            temp_data = self.feature_engineer.create_all_features(
                TimeSeriesData(
                    timestamps=temp_data["timestamp"].tolist(),
                    values=temp_data["value"].tolist(),
                    frequency=TimeSeriesData(
                        timestamps=self.last_values["timestamp"].tolist()[:2],
                        values=self.last_values["value"].tolist()[:2],
                        frequency="H",
                    ).frequency,
                    name="temp",
                ),
                target_col="value",
            )

            current_data = temp_data

        # Generate timestamps
        last_timestamp = self.last_values["timestamp"].iloc[-1]
        freq = pd.infer_freq(self.last_values["timestamp"])
        forecast_index = pd.date_range(
            start=last_timestamp, periods=horizon + 1, freq=freq or "H"
        )[1:]
        timestamps = forecast_index.to_pydatetime().tolist()

        return ForecastResult(
            model_type=self.model_type,
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=None,  # XGBoost doesn't provide confidence intervals by default
            upper_bound=None,
            confidence_level=confidence_level,
            metadata={"n_estimators": self.params.get("n_estimators")},
        )

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate XGBoost model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)

    def gradient_boosting(
        self, data: TimeSeriesData, horizon: int, **kwargs: Any
    ) -> ForecastResult:
        """
        Convenience method for gradient boosting forecasting.

        Args:
            data: Time series data
            horizon: Forecast horizon
            **kwargs: Additional parameters

        Returns:
            ForecastResult
        """
        self.fit(data, **kwargs)
        return self.predict(horizon)


class LSTMForecaster(BaseForecaster):
    """
    LSTM (Long Short-Term Memory) neural network forecaster.

    LSTM networks are effective for capturing long-term dependencies
    in time series data.

    Args:
        lookback: Number of time steps to look back
        units: Number of LSTM units in each layer
        layers: Number of LSTM layers
        dropout: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        lookback: int = 24,
        units: int = 50,
        layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize LSTM forecaster."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecaster")

        super().__init__(
            ModelType.LSTM,
            lookback=lookback,
            units=units,
            layers=layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )
        self.lookback = lookback
        self.units = units
        self.num_layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = None

    def _prepare_sequences(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i : i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture."""
        model = keras.Sequential()

        # Add LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(
                layers.LSTM(
                    self.units,
                    return_sequences=return_sequences,
                    input_shape=input_shape if i == 0 else None,
                )
            )
            model.add(layers.Dropout(self.dropout))

        # Output layer
        model.add(layers.Dense(1))

        # Compile model
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "LSTMForecaster":
        """
        Fit LSTM model to time series data.

        Args:
            data: Time series data
            exogenous: Optional exogenous variables (not used in basic LSTM)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        normalized_data = self.scaler.fit_transform(
            np.array(data.values).reshape(-1, 1)
        ).flatten()

        # Prepare sequences
        X, y = self._prepare_sequences(normalized_data)

        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and fit model
        self.model = self._build_model(input_shape=(self.lookback, 1))

        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True
        )

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0,
            **kwargs,
        )

        # Store last values for prediction
        self.last_sequence = normalized_data[-self.lookback :]
        self.is_fitted = True

        return self

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts using LSTM.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables
            confidence_level: Confidence level (not used)
            **kwargs: Additional parameters

        Returns:
            ForecastResult with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(horizon):
            # Prepare input
            X = current_sequence.reshape(1, self.lookback, 1)

            # Predict next value
            next_pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(next_pred)

            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)

        # Denormalize predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten().tolist()

        # Generate timestamps (simplified - should be improved)
        timestamps = [None] * horizon  # Placeholder

        return ForecastResult(
            model_type=self.model_type,
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=None,
            upper_bound=None,
            confidence_level=confidence_level,
            metadata={
                "lookback": self.lookback,
                "units": self.units,
                "layers": self.num_layers,
            },
        )

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate LSTM model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)

    def lstm_models(
        self, data: TimeSeriesData, horizon: int, **kwargs: Any
    ) -> ForecastResult:
        """
        Convenience method for LSTM forecasting.

        Args:
            data: Time series data
            horizon: Forecast horizon
            **kwargs: Additional parameters

        Returns:
            ForecastResult
        """
        self.fit(data, **kwargs)
        return self.predict(horizon)


class MLForecaster:
    """
    Unified interface for machine learning forecasting models.

    Provides convenient methods for Prophet, XGBoost, LSTM, and ensemble methods.
    """

    @staticmethod
    def prophet_forecasting(
        data: TimeSeriesData, horizon: int, **kwargs: Any
    ) -> ForecastResult:
        """
        Create Prophet forecast.

        Args:
            data: Time series data
            horizon: Forecast horizon
            **kwargs: Prophet parameters

        Returns:
            ForecastResult
        """
        forecaster = ProphetForecaster(**kwargs)
        forecaster.fit(data)
        return forecaster.predict(horizon)

    @staticmethod
    def gradient_boosting(
        data: TimeSeriesData, horizon: int, model: str = "xgboost", **kwargs: Any
    ) -> ForecastResult:
        """
        Create gradient boosting forecast.

        Args:
            data: Time series data
            horizon: Forecast horizon
            model: Model type ('xgboost', 'lightgbm', 'randomforest')
            **kwargs: Model parameters

        Returns:
            ForecastResult
        """
        if model == "xgboost":
            forecaster = XGBoostForecaster(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model}")

        forecaster.fit(data)
        return forecaster.predict(horizon)

    @staticmethod
    def lstm_models(
        data: TimeSeriesData, horizon: int, **kwargs: Any
    ) -> ForecastResult:
        """
        Create LSTM forecast.

        Args:
            data: Time series data
            horizon: Forecast horizon
            **kwargs: LSTM parameters

        Returns:
            ForecastResult
        """
        forecaster = LSTMForecaster(**kwargs)
        forecaster.fit(data)
        return forecaster.predict(horizon)

    @staticmethod
    def ensemble_methods(
        data: TimeSeriesData,
        horizon: int,
        models: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> ForecastResult:
        """
        Create ensemble forecast from multiple models.

        Args:
            data: Time series data
            horizon: Forecast horizon
            models: List of models to ensemble ('prophet', 'xgboost', 'lstm')
            weights: Optional weights for weighted averaging

        Returns:
            ForecastResult with ensemble predictions
        """
        from pv_simulator.core.models import EnsembleForecaster

        models = models or ["prophet", "xgboost"]
        forecasters = []

        for model_name in models:
            if model_name == "prophet":
                forecasters.append(ProphetForecaster())
            elif model_name == "xgboost":
                forecasters.append(XGBoostForecaster())
            elif model_name == "lstm" and TENSORFLOW_AVAILABLE:
                forecasters.append(LSTMForecaster())

        ensemble = EnsembleForecaster(
            forecasters=forecasters,
            weights=weights,
            ensemble_method="weighted" if weights else "average",
        )

        ensemble.fit(data)
        return ensemble.predict(horizon)
