"""Pydantic models and validators for time-series data and configuration.

This module provides data validation and configuration models for the
PV Circularity Simulator forecasting and processing components.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class ForecastMethod(str, Enum):
    """Enumeration of available forecasting methods."""

    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class SeasonalPeriod(str, Enum):
    """Enumeration of seasonal periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class TimeSeriesData(BaseModel):
    """Validated time-series data model.

    Attributes:
        timestamps: List of datetime objects representing observation times.
        values: List of numeric values corresponding to timestamps.
        frequency: Optional frequency string (e.g., 'D', 'W', 'M', 'Q', 'Y').
        name: Optional name for the time series.
    """

    timestamps: List[datetime] = Field(..., description="Time series timestamps")
    values: List[float] = Field(..., description="Time series values")
    frequency: Optional[str] = Field(None, description="Time series frequency")
    name: Optional[str] = Field("time_series", description="Name of the time series")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(cls, v: List[datetime]) -> List[datetime]:
        """Validate that timestamps are sorted and non-empty.

        Args:
            v: List of timestamps to validate.

        Returns:
            Validated list of timestamps.

        Raises:
            ValueError: If timestamps are empty or not sorted.
        """
        if not v:
            raise ValueError("Timestamps cannot be empty")

        if len(v) < 2:
            raise ValueError("Time series must have at least 2 observations")

        # Check if sorted
        if v != sorted(v):
            raise ValueError("Timestamps must be sorted in ascending order")

        return v

    @model_validator(mode="after")
    def validate_lengths(self) -> "TimeSeriesData":
        """Validate that timestamps and values have the same length.

        Returns:
            Validated TimeSeriesData instance.

        Raises:
            ValueError: If lengths don't match.
        """
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"Timestamps ({len(self.timestamps)}) and values "
                f"({len(self.values)}) must have the same length"
            )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with timestamps as index and values as a column.
        """
        df = pd.DataFrame({"value": self.values}, index=self.timestamps)
        df.index.name = "timestamp"
        return df

    def to_series(self) -> pd.Series:
        """Convert to pandas Series.

        Returns:
            Series with timestamps as index.
        """
        return pd.Series(self.values, index=self.timestamps, name=self.name)


class ARIMAConfig(BaseModel):
    """Configuration for ARIMA forecasting.

    Attributes:
        p: Order of the autoregressive part.
        d: Degree of differencing.
        q: Order of the moving average part.
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        trend: Trend parameter ('n', 'c', 't', 'ct').
        enforce_stationarity: Whether to enforce stationarity.
        enforce_invertibility: Whether to enforce invertibility.
    """

    p: int = Field(1, ge=0, le=5, description="AR order")
    d: int = Field(1, ge=0, le=2, description="Differencing degree")
    q: int = Field(1, ge=0, le=5, description="MA order")
    seasonal_order: tuple[int, int, int, int] = Field(
        (0, 0, 0, 0), description="Seasonal ARIMA order"
    )
    trend: str = Field("c", description="Trend parameter")
    enforce_stationarity: bool = Field(True, description="Enforce stationarity")
    enforce_invertibility: bool = Field(True, description="Enforce invertibility")

    @field_validator("trend")
    @classmethod
    def validate_trend(cls, v: str) -> str:
        """Validate trend parameter.

        Args:
            v: Trend parameter to validate.

        Returns:
            Validated trend parameter.

        Raises:
            ValueError: If trend is invalid.
        """
        valid_trends = ["n", "c", "t", "ct"]
        if v not in valid_trends:
            raise ValueError(f"Trend must be one of {valid_trends}, got '{v}'")
        return v


class ProphetConfig(BaseModel):
    """Configuration for Prophet forecasting.

    Attributes:
        growth: Growth model ('linear' or 'logistic').
        changepoint_prior_scale: Flexibility of the trend.
        seasonality_prior_scale: Strength of the seasonality model.
        seasonality_mode: Type of seasonality ('additive' or 'multiplicative').
        yearly_seasonality: Whether to model yearly seasonality.
        weekly_seasonality: Whether to model weekly seasonality.
        daily_seasonality: Whether to model daily seasonality.
        holidays_prior_scale: Strength of the holiday components.
    """

    growth: str = Field("linear", description="Growth model")
    changepoint_prior_scale: float = Field(
        0.05, ge=0.001, le=0.5, description="Trend flexibility"
    )
    seasonality_prior_scale: float = Field(
        10.0, ge=0.01, le=100.0, description="Seasonality strength"
    )
    seasonality_mode: str = Field("additive", description="Seasonality mode")
    yearly_seasonality: Union[bool, int] = Field(True, description="Yearly seasonality")
    weekly_seasonality: Union[bool, int] = Field(True, description="Weekly seasonality")
    daily_seasonality: Union[bool, int] = Field(False, description="Daily seasonality")
    holidays_prior_scale: float = Field(10.0, ge=0.01, le=100.0, description="Holiday strength")

    @field_validator("growth")
    @classmethod
    def validate_growth(cls, v: str) -> str:
        """Validate growth model.

        Args:
            v: Growth model to validate.

        Returns:
            Validated growth model.

        Raises:
            ValueError: If growth model is invalid.
        """
        valid_growth = ["linear", "logistic"]
        if v not in valid_growth:
            raise ValueError(f"Growth must be one of {valid_growth}, got '{v}'")
        return v

    @field_validator("seasonality_mode")
    @classmethod
    def validate_seasonality_mode(cls, v: str) -> str:
        """Validate seasonality mode.

        Args:
            v: Seasonality mode to validate.

        Returns:
            Validated seasonality mode.

        Raises:
            ValueError: If seasonality mode is invalid.
        """
        valid_modes = ["additive", "multiplicative"]
        if v not in valid_modes:
            raise ValueError(f"Seasonality mode must be one of {valid_modes}, got '{v}'")
        return v


class LSTMConfig(BaseModel):
    """Configuration for LSTM forecasting.

    Attributes:
        n_layers: Number of LSTM layers.
        hidden_units: Number of hidden units per layer.
        dropout_rate: Dropout rate for regularization.
        learning_rate: Learning rate for optimization.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        lookback_window: Number of past timesteps to use.
        validation_split: Fraction of data for validation.
    """

    n_layers: int = Field(2, ge=1, le=5, description="Number of LSTM layers")
    hidden_units: int = Field(50, ge=10, le=500, description="Hidden units per layer")
    dropout_rate: float = Field(0.2, ge=0.0, le=0.5, description="Dropout rate")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size")
    epochs: int = Field(100, ge=1, le=1000, description="Training epochs")
    lookback_window: int = Field(10, ge=1, le=100, description="Lookback window")
    validation_split: float = Field(0.2, ge=0.0, le=0.5, description="Validation split")


class EnsembleConfig(BaseModel):
    """Configuration for ensemble forecasting.

    Attributes:
        methods: List of methods to include in ensemble.
        weights: Optional weights for each method.
        aggregation: Aggregation method ('mean', 'median', 'weighted').
    """

    methods: List[ForecastMethod] = Field(
        [ForecastMethod.ARIMA, ForecastMethod.PROPHET, ForecastMethod.LSTM],
        description="Methods to ensemble",
    )
    weights: Optional[List[float]] = Field(None, description="Method weights")
    aggregation: str = Field("mean", description="Aggregation method")

    @field_validator("methods")
    @classmethod
    def validate_methods(cls, v: List[ForecastMethod]) -> List[ForecastMethod]:
        """Validate that at least 2 methods are provided.

        Args:
            v: List of methods to validate.

        Returns:
            Validated list of methods.

        Raises:
            ValueError: If fewer than 2 methods are provided.
        """
        if len(v) < 2:
            raise ValueError("Ensemble requires at least 2 methods")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "EnsembleConfig":
        """Validate that weights match the number of methods.

        Returns:
            Validated EnsembleConfig instance.

        Raises:
            ValueError: If weights don't match methods.
        """
        if self.weights is not None:
            if len(self.weights) != len(self.methods):
                raise ValueError("Number of weights must match number of methods")
            if not np.isclose(sum(self.weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
        return self

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        """Validate aggregation method.

        Args:
            v: Aggregation method to validate.

        Returns:
            Validated aggregation method.

        Raises:
            ValueError: If aggregation method is invalid.
        """
        valid_methods = ["mean", "median", "weighted"]
        if v not in valid_methods:
            raise ValueError(f"Aggregation must be one of {valid_methods}, got '{v}'")
        return v


class ForecastResult(BaseModel):
    """Result from a forecasting operation.

    Attributes:
        method: Forecasting method used.
        predictions: Forecasted values.
        timestamps: Forecast timestamps.
        confidence_intervals: Optional confidence intervals.
        metrics: Optional performance metrics.
        model_params: Optional model parameters.
    """

    method: ForecastMethod = Field(..., description="Forecasting method")
    predictions: List[float] = Field(..., description="Forecasted values")
    timestamps: List[datetime] = Field(..., description="Forecast timestamps")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(
        None, description="Confidence intervals"
    )
    metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_lengths(self) -> "ForecastResult":
        """Validate that predictions and timestamps have the same length.

        Returns:
            Validated ForecastResult instance.

        Raises:
            ValueError: If lengths don't match.
        """
        if len(self.predictions) != len(self.timestamps):
            raise ValueError("Predictions and timestamps must have the same length")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with forecast results.
        """
        df = pd.DataFrame({"prediction": self.predictions}, index=self.timestamps)

        if self.confidence_intervals:
            for key, values in self.confidence_intervals.items():
                df[key] = values

        return df


class DecompositionResult(BaseModel):
    """Result from seasonal decomposition.

    Attributes:
        trend: Trend component.
        seasonal: Seasonal component.
        residual: Residual component.
        timestamps: Timestamps for components.
    """

    trend: List[float] = Field(..., description="Trend component")
    seasonal: List[float] = Field(..., description="Seasonal component")
    residual: List[float] = Field(..., description="Residual component")
    timestamps: List[datetime] = Field(..., description="Timestamps")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_lengths(self) -> "DecompositionResult":
        """Validate that all components have the same length.

        Returns:
            Validated DecompositionResult instance.

        Raises:
            ValueError: If lengths don't match.
        """
        lengths = [len(self.trend), len(self.seasonal), len(self.residual), len(self.timestamps)]
        if len(set(lengths)) > 1:
            raise ValueError("All components must have the same length")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with decomposition components.
        """
        return pd.DataFrame(
            {
                "trend": self.trend,
                "seasonal": self.seasonal,
                "residual": self.residual,
            },
            index=self.timestamps,
        )
