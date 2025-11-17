"""
Model evaluation metrics for forecasting.

This module provides comprehensive metrics for evaluating forecast accuracy
including MAE, RMSE, MAPE, R2, and specialized forecast evaluation tools.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pv_simulator.core.schemas import ForecastResult, ModelMetrics, TimeSeriesData


class MetricsCalculator:
    """
    Calculator for forecast evaluation metrics.

    Provides methods to calculate various accuracy metrics and compare
    forecasts against actual values.
    """

    @staticmethod
    def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MAE value
        """
        return float(mean_absolute_error(actual, predicted))

    @staticmethod
    def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            RMSE value
        """
        mse = mean_squared_error(actual, predicted)
        return float(np.sqrt(mse))

    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.inf

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return float(mape)

    @staticmethod
    def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

        SMAPE is less sensitive to outliers than MAPE.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            SMAPE value (as percentage)
        """
        numerator = np.abs(actual - predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0

        if not np.any(mask):
            return 0.0

        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        return float(smape)

    @staticmethod
    def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            R2 score
        """
        return float(r2_score(actual, predicted))

    @staticmethod
    def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MSE value
        """
        return float(mean_squared_error(actual, predicted))

    def calculate_metrics(
        self,
        actual: Union[List[float], np.ndarray],
        predicted: Union[List[float], np.ndarray],
    ) -> ModelMetrics:
        """
        Calculate all evaluation metrics.

        Args:
            actual: Actual observed values
            predicted: Predicted values

        Returns:
            ModelMetrics with all calculated metrics

        Raises:
            ValueError: If arrays have different lengths
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)

        if len(actual_arr) != len(predicted_arr):
            raise ValueError("Actual and predicted arrays must have same length")

        mae = self.calculate_mae(actual_arr, predicted_arr)
        rmse = self.calculate_rmse(actual_arr, predicted_arr)
        mape = self.calculate_mape(actual_arr, predicted_arr)
        r2 = self.calculate_r2(actual_arr, predicted_arr)
        mse = self.calculate_mse(actual_arr, predicted_arr)
        smape = self.calculate_smape(actual_arr, predicted_arr)

        return ModelMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            mse=mse,
            smape=smape,
        )

    @staticmethod
    def forecast_vs_actual(
        actual: TimeSeriesData,
        forecast: ForecastResult,
    ) -> pd.DataFrame:
        """
        Create a comparison DataFrame of forecast vs actual values.

        Args:
            actual: Actual time series data
            forecast: Forecast results

        Returns:
            DataFrame with actual, predicted, and error columns

        Example:
            >>> calculator = MetricsCalculator()
            >>> comparison = calculator.forecast_vs_actual(actual, forecast)
            >>> print(comparison.head())
        """
        # Align timestamps
        actual_df = pd.DataFrame(
            {"timestamp": actual.timestamps, "actual": actual.values}
        )
        forecast_df = pd.DataFrame(
            {
                "timestamp": forecast.timestamps,
                "predicted": forecast.predictions,
                "lower_bound": forecast.lower_bound or [None] * len(forecast.predictions),
                "upper_bound": forecast.upper_bound or [None] * len(forecast.predictions),
            }
        )

        # Merge on timestamp
        comparison = pd.merge(actual_df, forecast_df, on="timestamp", how="outer")
        comparison = comparison.sort_values("timestamp")

        # Calculate errors where both actual and predicted exist
        mask = comparison["actual"].notna() & comparison["predicted"].notna()
        comparison.loc[mask, "error"] = (
            comparison.loc[mask, "actual"] - comparison.loc[mask, "predicted"]
        )
        comparison.loc[mask, "abs_error"] = np.abs(comparison.loc[mask, "error"])
        comparison.loc[mask, "pct_error"] = (
            comparison.loc[mask, "error"] / comparison.loc[mask, "actual"] * 100
        )

        return comparison

    @staticmethod
    def calculate_coverage(
        actual: TimeSeriesData,
        forecast: ForecastResult,
    ) -> float:
        """
        Calculate prediction interval coverage.

        Measures the percentage of actual values that fall within the
        prediction intervals.

        Args:
            actual: Actual time series data
            forecast: Forecast with confidence intervals

        Returns:
            Coverage percentage (0-100)
        """
        if forecast.lower_bound is None or forecast.upper_bound is None:
            raise ValueError("Forecast must include confidence intervals")

        comparison = MetricsCalculator.forecast_vs_actual(actual, forecast)
        mask = comparison["actual"].notna()

        within_interval = (
            (comparison.loc[mask, "actual"] >= comparison.loc[mask, "lower_bound"])
            & (comparison.loc[mask, "actual"] <= comparison.loc[mask, "upper_bound"])
        )

        coverage = within_interval.sum() / len(within_interval) * 100
        return float(coverage)

    @staticmethod
    def rolling_forecast_origin(
        data: TimeSeriesData,
        forecaster,
        horizon: int,
        window_size: int,
        step_size: int = 1,
    ) -> List[Tuple[ForecastResult, ModelMetrics]]:
        """
        Perform rolling forecast origin cross-validation.

        This method evaluates forecast accuracy by iteratively:
        1. Training on a window of data
        2. Forecasting the next horizon periods
        3. Moving the window forward by step_size

        Args:
            data: Complete time series data
            forecaster: Forecaster instance (must have fit/predict methods)
            horizon: Forecast horizon
            window_size: Size of training window
            step_size: Number of periods to move window forward

        Returns:
            List of (ForecastResult, ModelMetrics) tuples

        Example:
            >>> results = MetricsCalculator.rolling_forecast_origin(
            ...     data, ARIMAModel(), horizon=12, window_size=100
            ... )
        """
        calculator = MetricsCalculator()
        results = []

        n = len(data.values)
        for start_idx in range(0, n - window_size - horizon + 1, step_size):
            # Training window
            train_end = start_idx + window_size
            train_data = TimeSeriesData(
                timestamps=data.timestamps[start_idx:train_end],
                values=data.values[start_idx:train_end],
                frequency=data.frequency,
            )

            # Test window
            test_start = train_end
            test_end = test_start + horizon
            test_data = TimeSeriesData(
                timestamps=data.timestamps[test_start:test_end],
                values=data.values[test_start:test_end],
                frequency=data.frequency,
            )

            # Fit and predict
            forecaster.fit(train_data)
            forecast = forecaster.predict(horizon=horizon)

            # Evaluate
            metrics = calculator.calculate_metrics(
                test_data.values, forecast.predictions
            )

            results.append((forecast, metrics))

        return results

    @staticmethod
    def calculate_forecast_bias(
        actual: Union[List[float], np.ndarray],
        predicted: Union[List[float], np.ndarray],
    ) -> float:
        """
        Calculate forecast bias (mean error).

        Positive bias indicates over-forecasting, negative indicates under-forecasting.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Mean bias
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        bias = np.mean(predicted_arr - actual_arr)
        return float(bias)

    @staticmethod
    def skill_score(
        model_metrics: ModelMetrics,
        baseline_metrics: ModelMetrics,
        metric: str = "rmse",
    ) -> float:
        """
        Calculate forecast skill score relative to a baseline.

        Skill score = 1 - (model_error / baseline_error)
        A score of 1 indicates perfect forecast, 0 indicates same as baseline,
        negative indicates worse than baseline.

        Args:
            model_metrics: Metrics from the forecasting model
            baseline_metrics: Metrics from baseline (e.g., naive forecast)
            metric: Metric to use for comparison ('rmse', 'mae', 'mape')

        Returns:
            Skill score
        """
        model_error = getattr(model_metrics, metric)
        baseline_error = getattr(baseline_metrics, metric)

        if baseline_error == 0:
            return 1.0 if model_error == 0 else -np.inf

        skill = 1 - (model_error / baseline_error)
        return float(skill)


class ForecastEvaluator:
    """
    Comprehensive forecast evaluation with multiple metrics and visualizations.
    """

    def __init__(self) -> None:
        """Initialize forecast evaluator."""
        self.calculator = MetricsCalculator()

    def evaluate_forecast(
        self,
        actual: TimeSeriesData,
        forecast: ForecastResult,
        baseline_forecast: Optional[ForecastResult] = None,
    ) -> Dict[str, Union[ModelMetrics, float, pd.DataFrame]]:
        """
        Perform comprehensive forecast evaluation.

        Args:
            actual: Actual time series data
            forecast: Forecast to evaluate
            baseline_forecast: Optional baseline forecast for comparison

        Returns:
            Dictionary with metrics, coverage, comparison, and skill scores
        """
        # Calculate primary metrics
        metrics = self.calculator.calculate_metrics(
            actual.values, forecast.predictions
        )

        # Create comparison DataFrame
        comparison = self.calculator.forecast_vs_actual(actual, forecast)

        # Calculate coverage if intervals available
        coverage = None
        if forecast.lower_bound is not None and forecast.upper_bound is not None:
            try:
                coverage = self.calculator.calculate_coverage(actual, forecast)
            except Exception:
                pass

        # Calculate forecast bias
        bias = self.calculator.calculate_forecast_bias(
            actual.values, forecast.predictions
        )

        result = {
            "metrics": metrics,
            "comparison": comparison,
            "bias": bias,
        }

        if coverage is not None:
            result["coverage"] = coverage

        # Calculate skill scores if baseline provided
        if baseline_forecast is not None:
            baseline_metrics = self.calculator.calculate_metrics(
                actual.values, baseline_forecast.predictions
            )

            skill_scores = {
                "rmse_skill": self.calculator.skill_score(
                    metrics, baseline_metrics, "rmse"
                ),
                "mae_skill": self.calculator.skill_score(
                    metrics, baseline_metrics, "mae"
                ),
                "mape_skill": self.calculator.skill_score(
                    metrics, baseline_metrics, "mape"
                ),
            }
            result["skill_scores"] = skill_scores

        return result
