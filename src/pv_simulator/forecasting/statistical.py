"""
Statistical time-series forecasting models.

This module implements traditional statistical forecasting methods including
ARIMA, SARIMA, Exponential Smoothing, and State Space models using statsmodels.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import acf, adfuller, pacf

from pv_simulator.core.models import BaseForecaster
from pv_simulator.core.schemas import (
    ForecastResult,
    ModelMetrics,
    ModelType,
    SeasonalDecomposition,
    TimeSeriesData,
)


class ARIMAModel(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecasting model.

    ARIMA models are widely used for time series forecasting and can handle
    non-seasonal time series with trend components.

    Args:
        order: ARIMA order (p, d, q) where:
            - p: number of autoregressive terms
            - d: number of differences (integration order)
            - q: number of moving average terms
        **kwargs: Additional ARIMA parameters
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        **kwargs: Any,
    ) -> None:
        """Initialize ARIMA model with specified order."""
        super().__init__(ModelType.ARIMA, order=order, **kwargs)
        self.order = order

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "ARIMAModel":
        """
        Fit ARIMA model to time series data.

        Args:
            data: Time series data to fit
            exogenous: Optional exogenous variables
            **kwargs: Additional ARIMA fitting parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or fitting fails
        """
        try:
            # Convert to pandas Series
            ts_data = pd.Series(
                data.values, index=pd.DatetimeIndex(data.timestamps)
            )

            # Fit ARIMA model
            self.model = ARIMA(
                ts_data,
                order=self.order,
                exog=exogenous,
                **self.params,
            )
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True

            # Store training info
            self.training_history.append(
                {
                    "aic": self.fitted_model.aic,
                    "bic": self.fitted_model.bic,
                    "hqic": self.fitted_model.hqic,
                }
            )

            return self

        except Exception as e:
            raise ValueError(f"Failed to fit ARIMA model: {str(e)}")

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts using fitted ARIMA model.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables for forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional prediction parameters

        Returns:
            ForecastResult with predictions and confidence intervals

        Raises:
            ValueError: If model not fitted or prediction fails
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Generate forecast
            forecast = self.fitted_model.get_forecast(
                steps=horizon, exog=exogenous, **kwargs
            )

            # Get point predictions
            predictions = forecast.predicted_mean.tolist()

            # Get confidence intervals
            alpha = 1 - confidence_level
            conf_int = forecast.conf_int(alpha=alpha)
            lower_bound = conf_int.iloc[:, 0].tolist()
            upper_bound = conf_int.iloc[:, 1].tolist()

            # Generate forecast timestamps
            last_timestamp = self.fitted_model.data.dates[-1]
            freq = pd.infer_freq(self.fitted_model.data.dates)
            forecast_index = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]
            timestamps = forecast_index.to_pydatetime().tolist()

            return ForecastResult(
                model_type=self.model_type,
                timestamps=timestamps,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                metadata={
                    "order": self.order,
                    "aic": self.fitted_model.aic,
                    "bic": self.fitted_model.bic,
                },
            )

        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {str(e)}")

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """
        Evaluate ARIMA model performance.

        Args:
            actual: Actual observed values
            predicted: Predicted values

        Returns:
            ModelMetrics containing evaluation metrics
        """
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)


class SARIMAModel(BaseForecaster):
    """
    SARIMA (Seasonal ARIMA) forecasting model.

    SARIMA extends ARIMA to handle seasonal time series data by adding
    seasonal components to the model.

    Args:
        order: Non-seasonal ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s) where s is the seasonal period
        **kwargs: Additional SARIMAX parameters
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        **kwargs: Any,
    ) -> None:
        """Initialize SARIMA model with specified orders."""
        super().__init__(
            ModelType.SARIMA, order=order, seasonal_order=seasonal_order, **kwargs
        )
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "SARIMAModel":
        """
        Fit SARIMA model to time series data.

        Args:
            data: Time series data to fit
            exogenous: Optional exogenous variables
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            # Convert to pandas Series
            ts_data = pd.Series(
                data.values, index=pd.DatetimeIndex(data.timestamps)
            )

            # Fit SARIMAX model
            self.model = SARIMAX(
                ts_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=exogenous,
                **self.params,
            )
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True

            # Store training info
            self.training_history.append(
                {
                    "aic": self.fitted_model.aic,
                    "bic": self.fitted_model.bic,
                    "hqic": self.fitted_model.hqic,
                }
            )

            return self

        except Exception as e:
            raise ValueError(f"Failed to fit SARIMA model: {str(e)}")

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts using fitted SARIMA model.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables
            confidence_level: Confidence level for intervals
            **kwargs: Additional parameters

        Returns:
            ForecastResult with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Generate forecast
            forecast = self.fitted_model.get_forecast(
                steps=horizon, exog=exogenous, **kwargs
            )

            # Get predictions and intervals
            predictions = forecast.predicted_mean.tolist()
            alpha = 1 - confidence_level
            conf_int = forecast.conf_int(alpha=alpha)
            lower_bound = conf_int.iloc[:, 0].tolist()
            upper_bound = conf_int.iloc[:, 1].tolist()

            # Generate timestamps
            last_timestamp = self.fitted_model.data.dates[-1]
            freq = pd.infer_freq(self.fitted_model.data.dates)
            forecast_index = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]
            timestamps = forecast_index.to_pydatetime().tolist()

            return ForecastResult(
                model_type=self.model_type,
                timestamps=timestamps,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                metadata={
                    "order": self.order,
                    "seasonal_order": self.seasonal_order,
                    "aic": self.fitted_model.aic,
                },
            )

        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {str(e)}")

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate SARIMA model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)


def exponential_smoothing(
    data: TimeSeriesData,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
    seasonal_periods: Optional[int] = None,
    **kwargs: Any,
) -> BaseForecaster:
    """
    Create and fit an Exponential Smoothing model.

    Exponential Smoothing is effective for time series with trend and/or
    seasonal patterns.

    Args:
        data: Time series data to fit
        trend: Type of trend component ('add', 'mul', or None)
        seasonal: Type of seasonal component ('add', 'mul', or None)
        seasonal_periods: Number of periods in a season
        **kwargs: Additional parameters

    Returns:
        Fitted ExponentialSmoothingForecaster instance

    Example:
        >>> data = TimeSeriesData(...)
        >>> model = exponential_smoothing(data, seasonal_periods=12)
        >>> forecast = model.predict(horizon=12)
    """
    forecaster = ExponentialSmoothingForecaster(
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        **kwargs,
    )
    forecaster.fit(data)
    return forecaster


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Exponential Smoothing forecasting model.

    Supports simple, double, and triple exponential smoothing with
    additive or multiplicative trend and seasonal components.
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Exponential Smoothing model."""
        super().__init__(
            ModelType.EXPONENTIAL_SMOOTHING,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            **kwargs,
        )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "ExponentialSmoothingForecaster":
        """Fit Exponential Smoothing model."""
        try:
            ts_data = pd.Series(
                data.values, index=pd.DatetimeIndex(data.timestamps)
            )

            self.model = ExponentialSmoothing(
                ts_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                **self.params,
            )
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True

            return self

        except Exception as e:
            raise ValueError(f"Failed to fit Exponential Smoothing: {str(e)}")

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate forecasts using Exponential Smoothing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            predictions = forecast.tolist()

            # Generate timestamps
            last_timestamp = self.fitted_model.model.data.dates[-1]
            freq = pd.infer_freq(self.fitted_model.model.data.dates)
            forecast_index = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]
            timestamps = forecast_index.to_pydatetime().tolist()

            # Exponential Smoothing doesn't provide confidence intervals by default
            # Use simple approximation based on in-sample errors
            residuals = self.fitted_model.fittedvalues - self.fitted_model.model.endog
            std_error = np.std(residuals)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # Approximate

            lower_bound = (np.array(predictions) - z_score * std_error).tolist()
            upper_bound = (np.array(predictions) + z_score * std_error).tolist()

            return ForecastResult(
                model_type=self.model_type,
                timestamps=timestamps,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                metadata={
                    "trend": self.trend,
                    "seasonal": self.seasonal,
                    "seasonal_periods": self.seasonal_periods,
                },
            )

        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {str(e)}")

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)


def state_space_models(
    data: TimeSeriesData,
    level: bool = True,
    trend: bool = False,
    seasonal: Optional[int] = None,
    **kwargs: Any,
) -> BaseForecaster:
    """
    Create and fit a State Space model.

    State Space models provide a flexible framework for modeling
    time series with level, trend, and seasonal components.

    Args:
        data: Time series data
        level: Include level component
        trend: Include trend component
        seasonal: Seasonal period (None for no seasonality)
        **kwargs: Additional parameters

    Returns:
        Fitted StateSpaceForecaster instance
    """
    forecaster = StateSpaceForecaster(
        level=level, trend=trend, seasonal=seasonal, **kwargs
    )
    forecaster.fit(data)
    return forecaster


class StateSpaceForecaster(BaseForecaster):
    """State Space forecasting model using Unobserved Components."""

    def __init__(
        self,
        level: bool = True,
        trend: bool = False,
        seasonal: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize State Space model."""
        super().__init__(
            ModelType.STATE_SPACE,
            level=level,
            trend=trend,
            seasonal=seasonal,
            **kwargs,
        )
        self.level = level
        self.trend = trend
        self.seasonal = seasonal

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "StateSpaceForecaster":
        """Fit State Space model."""
        try:
            ts_data = pd.Series(
                data.values, index=pd.DatetimeIndex(data.timestamps)
            )

            self.model = UnobservedComponents(
                ts_data,
                level="local level" if self.level else None,
                trend=self.trend,
                seasonal=self.seasonal,
                exog=exogenous,
                **self.params,
            )
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True

            return self

        except Exception as e:
            raise ValueError(f"Failed to fit State Space model: {str(e)}")

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate forecasts using State Space model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            forecast = self.fitted_model.get_forecast(
                steps=horizon, exog=exogenous, **kwargs
            )

            predictions = forecast.predicted_mean.tolist()
            alpha = 1 - confidence_level
            conf_int = forecast.conf_int(alpha=alpha)
            lower_bound = conf_int.iloc[:, 0].tolist()
            upper_bound = conf_int.iloc[:, 1].tolist()

            last_timestamp = self.fitted_model.data.dates[-1]
            freq = pd.infer_freq(self.fitted_model.data.dates)
            forecast_index = pd.date_range(
                start=last_timestamp, periods=horizon + 1, freq=freq
            )[1:]
            timestamps = forecast_index.to_pydatetime().tolist()

            return ForecastResult(
                model_type=self.model_type,
                timestamps=timestamps,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                metadata={
                    "level": self.level,
                    "trend": self.trend,
                    "seasonal": self.seasonal,
                },
            )

        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {str(e)}")

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """Evaluate model performance."""
        from pv_simulator.forecasting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        return calculator.calculate_metrics(actual.values, predicted.predictions)


class StatisticalAnalyzer:
    """
    Statistical analysis tools for time series data.

    Provides methods for seasonality decomposition, trend analysis,
    autocorrelation analysis, and stationarity testing.
    """

    @staticmethod
    def seasonality_decomposition(
        data: TimeSeriesData,
        model: str = "additive",
        period: Optional[int] = None,
    ) -> SeasonalDecomposition:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            data: Time series data
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (auto-detected if None)

        Returns:
            SeasonalDecomposition with components
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        ts_data = pd.Series(data.values, index=pd.DatetimeIndex(data.timestamps))

        if period is None:
            # Try to infer period from frequency
            freq = pd.infer_freq(ts_data.index)
            if freq == "D":
                period = 7  # Weekly seasonality for daily data
            elif freq == "H":
                period = 24  # Daily seasonality for hourly data
            else:
                period = 12  # Default monthly seasonality

        decomposition = seasonal_decompose(
            ts_data, model=model, period=period, extrapolate_trend="freq"
        )

        # Calculate seasonality and trend strength
        var_residual = np.var(decomposition.resid.dropna())
        var_seasonal_resid = np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
        seasonality_strength = max(0, 1 - var_residual / var_seasonal_resid)

        var_trend_resid = np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
        trend_strength = max(0, 1 - var_residual / var_trend_resid)

        return SeasonalDecomposition(
            trend=decomposition.trend.fillna(0).tolist(),
            seasonal=decomposition.seasonal.fillna(0).tolist(),
            residual=decomposition.resid.fillna(0).tolist(),
            seasonality_strength=float(seasonality_strength),
            trend_strength=float(trend_strength),
        )

    @staticmethod
    def trend_analysis(
        data: TimeSeriesData,
    ) -> Dict[str, Any]:
        """
        Analyze trend in time series.

        Args:
            data: Time series data

        Returns:
            Dictionary with trend statistics
        """
        values = np.array(data.values)
        n = len(values)
        x = np.arange(n)

        # Linear regression for trend
        coeffs = np.polyfit(x, values, 1)
        slope, intercept = coeffs

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "trend_strength": abs(float(r_squared)),
        }

    @staticmethod
    def autocorrelation(
        data: TimeSeriesData, nlags: int = 40
    ) -> Dict[str, np.ndarray]:
        """
        Calculate autocorrelation (ACF) and partial autocorrelation (PACF).

        Args:
            data: Time series data
            nlags: Number of lags to calculate

        Returns:
            Dictionary with ACF and PACF values
        """
        values = np.array(data.values)

        acf_values = acf(values, nlags=nlags, fft=True)
        pacf_values = pacf(values, nlags=nlags)

        return {
            "acf": acf_values,
            "pacf": pacf_values,
            "lags": np.arange(nlags + 1),
        }

    @staticmethod
    def stationarity_test(data: TimeSeriesData) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            data: Time series data

        Returns:
            Dictionary with test results
        """
        values = np.array(data.values)
        result = adfuller(values)

        return {
            "adf_statistic": float(result[0]),
            "p_value": float(result[1]),
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }
