"""
Pydantic schemas for data validation and serialization.

This module defines the core data structures using Pydantic for type validation,
serialization, and configuration management.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TimeSeriesFrequency(str, Enum):
    """Supported time series frequencies."""

    MINUTE = "T"
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"


class ModelType(str, Enum):
    """Supported forecasting model types."""

    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    STATE_SPACE = "state_space"
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    GRU = "gru"
    ENSEMBLE = "ensemble"


class SeasonalityType(str, Enum):
    """Types of seasonality patterns."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class TimeSeriesData(BaseModel):
    """Schema for time series data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamps: List[datetime] = Field(..., description="List of timestamps")
    values: List[float] = Field(..., description="Time series values")
    frequency: TimeSeriesFrequency = Field(..., description="Data frequency")
    name: str = Field(default="time_series", description="Name of the time series")
    metadata: Optional[Dict[str, Union[str, float, int]]] = Field(
        default=None, description="Additional metadata"
    )

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(cls, v: List[datetime]) -> List[datetime]:
        """Validate that timestamps are sorted and unique."""
        if len(v) != len(set(v)):
            raise ValueError("Timestamps must be unique")
        if v != sorted(v):
            raise ValueError("Timestamps must be sorted")
        return v

    @field_validator("values")
    @classmethod
    def validate_values_length(cls, v: List[float], info) -> List[float]:
        """Validate that values have the same length as timestamps."""
        if "timestamps" in info.data and len(v) != len(info.data["timestamps"]):
            raise ValueError("Values must have the same length as timestamps")
        return v

    def to_numpy(self) -> np.ndarray:
        """Convert values to numpy array."""
        return np.array(self.values)

    def to_dict(self) -> Dict[str, Union[List[datetime], List[float]]]:
        """Convert to dictionary format."""
        return {"timestamps": self.timestamps, "values": self.values}


class ForecastConfig(BaseModel):
    """Configuration for forecasting models."""

    horizon: int = Field(..., gt=0, description="Forecast horizon (number of periods)")
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals"
    )
    seasonality_mode: SeasonalityType = Field(
        default=SeasonalityType.ADDITIVE, description="Seasonality mode"
    )
    seasonal_periods: Optional[int] = Field(
        default=None, gt=0, description="Number of periods in a season"
    )
    enable_holidays: bool = Field(default=False, description="Include holiday effects")
    enable_exogenous: bool = Field(default=False, description="Include exogenous variables")


class ForecastResult(BaseModel):
    """Schema for forecast results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_type: ModelType = Field(..., description="Type of model used")
    timestamps: List[datetime] = Field(..., description="Forecast timestamps")
    predictions: List[float] = Field(..., description="Point forecasts")
    lower_bound: Optional[List[float]] = Field(
        default=None, description="Lower confidence bound"
    )
    upper_bound: Optional[List[float]] = Field(
        default=None, description="Upper confidence bound"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level")
    metadata: Optional[Dict[str, Union[str, float, int]]] = Field(
        default=None, description="Additional metadata"
    )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        result = {"predictions": np.array(self.predictions)}
        if self.lower_bound is not None:
            result["lower_bound"] = np.array(self.lower_bound)
        if self.upper_bound is not None:
            result["upper_bound"] = np.array(self.upper_bound)
        return result


class ModelMetrics(BaseModel):
    """Schema for model evaluation metrics."""

    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")
    mse: Optional[float] = Field(default=None, description="Mean Squared Error")
    smape: Optional[float] = Field(
        default=None, description="Symmetric Mean Absolute Percentage Error"
    )
    additional_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Additional custom metrics"
    )

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        result = {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2,
        }
        if self.mse is not None:
            result["mse"] = self.mse
        if self.smape is not None:
            result["smape"] = self.smape
        if self.additional_metrics:
            result.update(self.additional_metrics)
        return result


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    lag_features: bool = Field(default=True, description="Generate lag features")
    lag_periods: List[int] = Field(
        default=[1, 7, 30], description="Lag periods to generate"
    )
    rolling_features: bool = Field(default=True, description="Generate rolling features")
    rolling_windows: List[int] = Field(
        default=[7, 14, 30], description="Rolling window sizes"
    )
    temporal_features: bool = Field(
        default=True, description="Generate temporal features (hour, day, month, etc.)"
    )
    weather_features: bool = Field(
        default=False, description="Include weather-related features"
    )
    cyclical_encoding: bool = Field(
        default=True, description="Use cyclical encoding for temporal features"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Test set size")
    validation_size: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Validation set size"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    cross_validation_folds: int = Field(
        default=5, gt=1, description="Number of CV folds"
    )
    hyperparameter_tuning: bool = Field(
        default=True, description="Enable hyperparameter tuning"
    )
    tuning_trials: int = Field(default=100, gt=0, description="Number of tuning trials")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(
        default=10, gt=0, description="Early stopping patience"
    )


class SeasonalDecomposition(BaseModel):
    """Schema for seasonal decomposition results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trend: List[float] = Field(..., description="Trend component")
    seasonal: List[float] = Field(..., description="Seasonal component")
    residual: List[float] = Field(..., description="Residual component")
    seasonality_strength: float = Field(
        ..., ge=0.0, le=1.0, description="Strength of seasonality"
    )
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of trend")

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        return {
            "trend": np.array(self.trend),
            "seasonal": np.array(self.seasonal),
            "residual": np.array(self.residual),
        }
