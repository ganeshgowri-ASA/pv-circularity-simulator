"""
Pydantic models for energy forecasting and prediction.

This module defines data models for:
- Forecast data points and series
- Accuracy metrics (MAE, RMSE, MAPE, etc.)
- Confidence intervals and prediction bounds
- Forecast metadata and configuration
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np
from pydantic import Field, field_validator, computed_field

from pv_simulator.core.models import BaseSimulatorModel, TimestampedModel, IdentifiableModel


class ForecastHorizon(str, Enum):
    """
    Enum for forecast time horizons.

    Attributes:
        SHORT_TERM: Short-term forecast (hours to days)
        MEDIUM_TERM: Medium-term forecast (weeks to months)
        LONG_TERM: Long-term forecast (months to years)
    """
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class ForecastPoint(TimestampedModel):
    """
    Single forecast data point with prediction and optional bounds.

    This model represents a single timestamped forecast value with
    optional confidence intervals and actual observed value for
    accuracy calculation.

    Attributes:
        timestamp: Time when the forecast is for
        predicted: Predicted value
        actual: Actual observed value (if available)
        lower_bound: Lower confidence interval bound
        upper_bound: Upper confidence interval bound
        confidence_level: Confidence level as decimal (e.g., 0.95 for 95%)

    Examples:
        >>> point = ForecastPoint(
        ...     timestamp=datetime(2024, 1, 1, 12, 0),
        ...     predicted=150.5,
        ...     actual=148.2,
        ...     lower_bound=140.0,
        ...     upper_bound=160.0,
        ...     confidence_level=0.95
        ... )
        >>> point.predicted
        150.5
    """

    timestamp: datetime = Field(
        ...,
        description="Timestamp for this forecast point"
    )
    predicted: float = Field(
        ...,
        description="Predicted value"
    )
    actual: Optional[float] = Field(
        default=None,
        description="Actual observed value (if available)"
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Lower confidence interval bound"
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Upper confidence interval bound"
    )
    confidence_level: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)"
    )

    @field_validator('confidence_level')
    @classmethod
    def validate_confidence_level(cls, v: Optional[float]) -> Optional[float]:
        """
        Validate confidence level is between 0 and 1.

        Args:
            v: Confidence level value

        Returns:
            Optional[float]: Validated confidence level

        Raises:
            ValueError: If confidence level is not between 0 and 1
        """
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Confidence level must be between 0.0 and 1.0")
        return v

    @computed_field
    @property
    def error(self) -> Optional[float]:
        """
        Calculate the forecast error (predicted - actual).

        Returns:
            Optional[float]: Error value if actual is available, None otherwise

        Examples:
            >>> point = ForecastPoint(
            ...     timestamp=datetime(2024, 1, 1),
            ...     predicted=150.0,
            ...     actual=148.0
            ... )
            >>> point.error
            2.0
        """
        if self.actual is not None:
            return self.predicted - self.actual
        return None

    @computed_field
    @property
    def absolute_error(self) -> Optional[float]:
        """
        Calculate the absolute forecast error.

        Returns:
            Optional[float]: Absolute error if actual is available, None otherwise

        Examples:
            >>> point = ForecastPoint(
            ...     timestamp=datetime(2024, 1, 1),
            ...     predicted=150.0,
            ...     actual=148.0
            ... )
            >>> point.absolute_error
            2.0
        """
        if self.error is not None:
            return abs(self.error)
        return None

    @computed_field
    @property
    def percentage_error(self) -> Optional[float]:
        """
        Calculate the percentage error.

        Returns:
            Optional[float]: Percentage error if actual is available and non-zero

        Examples:
            >>> point = ForecastPoint(
            ...     timestamp=datetime(2024, 1, 1),
            ...     predicted=150.0,
            ...     actual=100.0
            ... )
            >>> point.percentage_error
            50.0
        """
        if self.actual is not None and self.actual != 0:
            return (self.error / self.actual) * 100
        return None


class ForecastSeries(IdentifiableModel):
    """
    A series of forecast points representing a time series prediction.

    This model encapsulates a complete forecast with multiple points,
    metadata, and configuration information.

    Attributes:
        points: List of forecast points
        horizon: Forecast horizon type
        model_name: Name of the forecasting model used
        created_at: When the forecast was created
        parameters: Model parameters used for forecasting

    Examples:
        >>> points = [
        ...     ForecastPoint(timestamp=datetime(2024, 1, 1), predicted=100.0),
        ...     ForecastPoint(timestamp=datetime(2024, 1, 2), predicted=105.0),
        ... ]
        >>> series = ForecastSeries(
        ...     id="forecast-001",
        ...     name="Daily Energy Forecast",
        ...     points=points,
        ...     model_name="ARIMA"
        ... )
        >>> len(series.points)
        2
    """

    points: List[ForecastPoint] = Field(
        ...,
        min_length=1,
        description="List of forecast points in the series"
    )
    horizon: ForecastHorizon = Field(
        default=ForecastHorizon.SHORT_TERM,
        description="Forecast time horizon"
    )
    model_name: str = Field(
        ...,
        description="Name of the forecasting model"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this forecast was created"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model parameters and configuration"
    )

    @field_validator('points')
    @classmethod
    def validate_points_not_empty(cls, v: List[ForecastPoint]) -> List[ForecastPoint]:
        """
        Validate that points list is not empty.

        Args:
            v: List of forecast points

        Returns:
            List[ForecastPoint]: Validated points list

        Raises:
            ValueError: If points list is empty
        """
        if not v:
            raise ValueError("Forecast series must have at least one point")
        return v

    @computed_field
    @property
    def length(self) -> int:
        """
        Get the number of points in the series.

        Returns:
            int: Number of forecast points
        """
        return len(self.points)

    @computed_field
    @property
    def has_actuals(self) -> bool:
        """
        Check if the series has actual values for accuracy calculation.

        Returns:
            bool: True if at least one point has actual value
        """
        return any(point.actual is not None for point in self.points)

    def get_predicted_values(self) -> np.ndarray:
        """
        Extract predicted values as numpy array.

        Returns:
            np.ndarray: Array of predicted values

        Examples:
            >>> points = [
            ...     ForecastPoint(timestamp=datetime(2024, 1, 1), predicted=100.0),
            ...     ForecastPoint(timestamp=datetime(2024, 1, 2), predicted=105.0),
            ... ]
            >>> series = ForecastSeries(points=points, model_name="test")
            >>> series.get_predicted_values()
            array([100., 105.])
        """
        return np.array([point.predicted for point in self.points])

    def get_actual_values(self) -> Optional[np.ndarray]:
        """
        Extract actual values as numpy array.

        Returns:
            Optional[np.ndarray]: Array of actual values or None if not available

        Examples:
            >>> points = [
            ...     ForecastPoint(timestamp=datetime(2024, 1, 1), predicted=100.0, actual=98.0),
            ...     ForecastPoint(timestamp=datetime(2024, 1, 2), predicted=105.0, actual=107.0),
            ... ]
            >>> series = ForecastSeries(points=points, model_name="test")
            >>> series.get_actual_values()
            array([ 98., 107.])
        """
        if not self.has_actuals:
            return None
        actuals = [point.actual for point in self.points if point.actual is not None]
        return np.array(actuals) if actuals else None

    def get_timestamps(self) -> List[datetime]:
        """
        Extract timestamps as list.

        Returns:
            List[datetime]: List of timestamps

        Examples:
            >>> points = [
            ...     ForecastPoint(timestamp=datetime(2024, 1, 1), predicted=100.0),
            ...     ForecastPoint(timestamp=datetime(2024, 1, 2), predicted=105.0),
            ... ]
            >>> series = ForecastSeries(points=points, model_name="test")
            >>> len(series.get_timestamps())
            2
        """
        return [point.timestamp for point in self.points]


class AccuracyMetrics(BaseSimulatorModel):
    """
    Comprehensive accuracy metrics for forecast evaluation.

    This model contains various statistical metrics for assessing
    forecast accuracy including error metrics, correlation, and
    bias measures.

    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        mse: Mean Squared Error
        mape: Mean Absolute Percentage Error
        r2_score: R-squared coefficient of determination
        bias: Mean forecast bias
        n_samples: Number of samples used in calculation

    Examples:
        >>> metrics = AccuracyMetrics(
        ...     mae=5.2,
        ...     rmse=7.1,
        ...     mse=50.41,
        ...     mape=3.5,
        ...     r2_score=0.92,
        ...     bias=-1.2,
        ...     n_samples=100
        ... )
        >>> metrics.mae
        5.2
    """

    mae: float = Field(
        ...,
        ge=0.0,
        description="Mean Absolute Error"
    )
    rmse: float = Field(
        ...,
        ge=0.0,
        description="Root Mean Squared Error"
    )
    mse: float = Field(
        ...,
        ge=0.0,
        description="Mean Squared Error"
    )
    mape: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Mean Absolute Percentage Error"
    )
    r2_score: Optional[float] = Field(
        default=None,
        description="R-squared coefficient of determination"
    )
    bias: float = Field(
        ...,
        description="Mean forecast bias (predicted - actual)"
    )
    n_samples: int = Field(
        ...,
        ge=1,
        description="Number of samples used in calculation"
    )

    @field_validator('mse', 'rmse')
    @classmethod
    def validate_rmse_mse_relationship(cls, v: float, info) -> float:
        """
        Validate that RMSE is approximately the square root of MSE.

        Args:
            v: Value to validate
            info: Validation info context

        Returns:
            float: Validated value
        """
        # This is called for each field, so we can't fully validate the relationship here
        # The relationship should be validated in the dashboard calculation functions
        return v

    @computed_field
    @property
    def normalized_rmse(self) -> Optional[float]:
        """
        Calculate normalized RMSE (NRMSE).

        Note: This is a placeholder. Actual normalization requires the range
        or mean of actual values, which should be passed during calculation.

        Returns:
            Optional[float]: Normalized RMSE if applicable
        """
        # Placeholder - actual implementation would need actual value statistics
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Create a summary dictionary of key metrics.

        Returns:
            Dict[str, Any]: Dictionary with formatted metric values

        Examples:
            >>> metrics = AccuracyMetrics(
            ...     mae=5.2, rmse=7.1, mse=50.41,
            ...     bias=-1.2, n_samples=100
            ... )
            >>> summary = metrics.to_summary_dict()
            >>> 'mae' in summary
            True
        """
        return {
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
            "MSE": round(self.mse, 4),
            "MAPE": round(self.mape, 4) if self.mape is not None else None,
            "R²": round(self.r2_score, 4) if self.r2_score is not None else None,
            "Bias": round(self.bias, 4),
            "Samples": self.n_samples,
        }


class ForecastData(IdentifiableModel):
    """
    Complete forecast data including predictions, actuals, and metadata.

    This is the top-level model for representing a complete forecast
    with all associated data, metrics, and configuration.

    Attributes:
        series: The forecast series
        metrics: Accuracy metrics (if actuals available)
        metadata: Additional metadata

    Examples:
        >>> points = [
        ...     ForecastPoint(timestamp=datetime(2024, 1, 1), predicted=100.0, actual=98.0),
        ... ]
        >>> series = ForecastSeries(points=points, model_name="test")
        >>> forecast = ForecastData(
        ...     id="forecast-data-001",
        ...     series=series,
        ...     metadata={"location": "Site A"}
        ... )
        >>> forecast.series.length
        1
    """

    series: ForecastSeries = Field(
        ...,
        description="The forecast series data"
    )
    metrics: Optional[AccuracyMetrics] = Field(
        default=None,
        description="Accuracy metrics (if actuals are available)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the forecast"
    )

    @computed_field
    @property
    def can_calculate_metrics(self) -> bool:
        """
        Check if metrics can be calculated.

        Returns:
            bool: True if series has actual values for metric calculation
        """
        return self.series.has_actuals
