"""
Seasonal pattern analysis and long-term forecasting.

This module provides tools for analyzing seasonal patterns, generating
long-term forecasts, and performing year-over-year comparisons.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from pv_simulator.core.models import BaseForecaster
from pv_simulator.core.schemas import (
    ForecastResult,
    ModelType,
    SeasonalDecomposition,
    TimeSeriesData,
)
from pv_simulator.forecasting.statistical import SARIMAModel, StatisticalAnalyzer


class SeasonalPattern(BaseModel):
    """Schema for seasonal pattern information."""

    period: int = Field(..., description="Seasonal period length")
    strength: float = Field(..., ge=0.0, le=1.0, description="Pattern strength")
    pattern_values: List[float] = Field(..., description="Seasonal pattern values")
    pattern_type: str = Field(..., description="Type of pattern (additive/multiplicative)")


class YearOverYearComparison(BaseModel):
    """Schema for year-over-year comparison results."""

    years: List[int] = Field(..., description="Years being compared")
    values_by_year: Dict[int, List[float]] = Field(
        ..., description="Values for each year"
    )
    growth_rates: Dict[str, float] = Field(..., description="Year-over-year growth rates")
    average_growth: float = Field(..., description="Average growth rate")
    trend: str = Field(..., description="Overall trend (increasing/decreasing/stable)")


class SeasonalAnalyzer:
    """
    Advanced seasonal pattern analyzer.

    Provides comprehensive tools for detecting, analyzing, and visualizing
    seasonal patterns in time series data.
    """

    def __init__(self) -> None:
        """Initialize seasonal analyzer."""
        self.statistical_analyzer = StatisticalAnalyzer()

    def detect_seasonality(
        self, data: TimeSeriesData, max_period: int = 365
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time series.

        Uses autocorrelation and spectral analysis to identify dominant
        seasonal periods.

        Args:
            data: Time series data
            max_period: Maximum period to check

        Returns:
            Dictionary with detected seasonal periods and strengths

        Example:
            >>> analyzer = SeasonalAnalyzer()
            >>> seasonality = analyzer.detect_seasonality(data)
            >>> print(f"Detected periods: {seasonality['periods']}")
        """
        # Calculate autocorrelation
        acf_result = self.statistical_analyzer.autocorrelation(
            data, nlags=min(max_period, len(data.values) - 1)
        )

        # Find peaks in ACF (potential seasonal periods)
        acf_values = acf_result["acf"][1:]  # Exclude lag 0
        lags = acf_result["lags"][1:]

        # Find local maxima
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                if acf_values[i] > 0.3:  # Significant correlation threshold
                    peaks.append({"lag": int(lags[i]), "strength": float(acf_values[i])})

        # Sort by strength
        peaks = sorted(peaks, key=lambda x: x["strength"], reverse=True)

        # Common seasonal periods to check
        common_periods = {
            7: "weekly",
            24: "daily (hourly data)",
            30: "monthly (daily data)",
            365: "yearly (daily data)",
        }

        detected_periods = []
        for peak in peaks[:5]:  # Top 5 peaks
            period_info = {
                "period": peak["lag"],
                "strength": peak["strength"],
                "type": common_periods.get(peak["lag"], "custom"),
            }
            detected_periods.append(period_info)

        return {
            "periods": detected_periods,
            "has_seasonality": len(detected_periods) > 0,
            "dominant_period": detected_periods[0]["period"] if detected_periods else None,
        }

    def extract_seasonal_pattern(
        self,
        data: TimeSeriesData,
        period: int,
        model: str = "additive",
    ) -> SeasonalPattern:
        """
        Extract seasonal pattern for a specific period.

        Args:
            data: Time series data
            period: Seasonal period
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            SeasonalPattern with extracted pattern information

        Example:
            >>> pattern = analyzer.extract_seasonal_pattern(data, period=7)
        """
        # Perform seasonal decomposition
        decomposition = self.statistical_analyzer.seasonality_decomposition(
            data, model=model, period=period
        )

        # Extract unique seasonal pattern (one cycle)
        seasonal_component = decomposition.seasonal
        pattern_values = seasonal_component[:period]

        return SeasonalPattern(
            period=period,
            strength=decomposition.seasonality_strength,
            pattern_values=pattern_values,
            pattern_type=model,
        )

    def compare_seasonal_patterns(
        self,
        data1: TimeSeriesData,
        data2: TimeSeriesData,
        period: int,
    ) -> Dict[str, Any]:
        """
        Compare seasonal patterns between two time series.

        Args:
            data1: First time series
            data2: Second time series
            period: Seasonal period to compare

        Returns:
            Dictionary with comparison results
        """
        # Extract patterns
        pattern1 = self.extract_seasonal_pattern(data1, period)
        pattern2 = self.extract_seasonal_pattern(data2, period)

        # Calculate correlation between patterns
        correlation = np.corrcoef(pattern1.pattern_values, pattern2.pattern_values)[0, 1]

        # Calculate pattern difference
        diff = np.array(pattern1.pattern_values) - np.array(pattern2.pattern_values)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))

        return {
            "correlation": float(correlation),
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "pattern1_strength": pattern1.strength,
            "pattern2_strength": pattern2.strength,
            "similarity": "high" if correlation > 0.7 else "medium" if correlation > 0.4 else "low",
        }

    def year_over_year_comparison(
        self, data: TimeSeriesData, aggregation: str = "mean"
    ) -> YearOverYearComparison:
        """
        Perform year-over-year comparison.

        Compares values across different years to identify growth trends
        and seasonal patterns.

        Args:
            data: Time series data
            aggregation: How to aggregate monthly data ('mean', 'sum', 'median')

        Returns:
            YearOverYearComparison with comparison results

        Example:
            >>> comparison = analyzer.year_over_year_comparison(data)
            >>> print(f"Average growth: {comparison.average_growth:.2%}")
        """
        # Convert to DataFrame
        df = pd.DataFrame({"timestamp": data.timestamps, "value": data.values})
        df["year"] = pd.to_datetime(df["timestamp"]).dt.year
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month

        # Group by year and month
        if aggregation == "mean":
            monthly_data = df.groupby(["year", "month"])["value"].mean()
        elif aggregation == "sum":
            monthly_data = df.groupby(["year", "month"])["value"].sum()
        elif aggregation == "median":
            monthly_data = df.groupby(["year", "month"])["value"].median()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Organize by year
        years = sorted(df["year"].unique())
        values_by_year = {}

        for year in years:
            if year in monthly_data.index.get_level_values(0):
                values_by_year[year] = monthly_data[year].tolist()

        # Calculate year-over-year growth rates
        growth_rates = {}
        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]

            if prev_year in values_by_year and curr_year in values_by_year:
                prev_total = sum(values_by_year[prev_year])
                curr_total = sum(values_by_year[curr_year])

                if prev_total > 0:
                    growth = (curr_total - prev_total) / prev_total
                    growth_rates[f"{prev_year}-{curr_year}"] = float(growth)

        # Calculate average growth
        avg_growth = float(np.mean(list(growth_rates.values()))) if growth_rates else 0.0

        # Determine trend
        if avg_growth > 0.05:
            trend = "increasing"
        elif avg_growth < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        return YearOverYearComparison(
            years=[int(y) for y in years],
            values_by_year={int(k): v for k, v in values_by_year.items()},
            growth_rates=growth_rates,
            average_growth=avg_growth,
            trend=trend,
        )


class LongTermForecaster:
    """
    Long-term forecasting with seasonal adjustments.

    Specialized forecaster for multi-year predictions with proper handling
    of seasonal patterns and trends.
    """

    def __init__(self, base_forecaster: Optional[BaseForecaster] = None) -> None:
        """
        Initialize long-term forecaster.

        Args:
            base_forecaster: Base forecasting model (uses SARIMA if None)
        """
        self.base_forecaster = base_forecaster
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.is_fitted = False

    def fit(self, data: TimeSeriesData, **kwargs: Any) -> "LongTermForecaster":
        """
        Fit long-term forecaster.

        Args:
            data: Time series data
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        # Detect seasonality
        seasonality_info = self.seasonal_analyzer.detect_seasonality(data)

        # Use SARIMA if seasonality detected and no base forecaster provided
        if self.base_forecaster is None:
            if seasonality_info["has_seasonality"]:
                seasonal_period = seasonality_info["dominant_period"]
                self.base_forecaster = SARIMAModel(
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, seasonal_period),
                )
            else:
                # Use non-seasonal model
                from pv_simulator.forecasting.statistical import ARIMAModel

                self.base_forecaster = ARIMAModel(order=(1, 1, 1))

        # Fit base forecaster
        self.base_forecaster.fit(data, **kwargs)
        self.is_fitted = True

        return self

    def predict(
        self,
        horizon: int,
        confidence_level: float = 0.95,
        scenario: str = "base",
    ) -> ForecastResult:
        """
        Generate long-term forecast.

        Args:
            horizon: Forecast horizon (can be multiple years)
            confidence_level: Confidence level for intervals
            scenario: Forecast scenario ('base', 'optimistic', 'pessimistic')

        Returns:
            ForecastResult with long-term predictions

        Example:
            >>> forecaster = LongTermForecaster()
            >>> forecaster.fit(historical_data)
            >>> # Forecast 3 years ahead (daily data)
            >>> forecast = forecaster.predict(horizon=365*3)
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")

        # Generate base forecast
        forecast = self.base_forecaster.predict(
            horizon=horizon, confidence_level=confidence_level
        )

        # Apply scenario adjustments
        if scenario == "optimistic":
            # Increase predictions by 10%
            forecast.predictions = [p * 1.1 for p in forecast.predictions]
            if forecast.lower_bound:
                forecast.lower_bound = [lb * 1.1 for lb in forecast.lower_bound]
            if forecast.upper_bound:
                forecast.upper_bound = [ub * 1.1 for ub in forecast.upper_bound]

        elif scenario == "pessimistic":
            # Decrease predictions by 10%
            forecast.predictions = [p * 0.9 for p in forecast.predictions]
            if forecast.lower_bound:
                forecast.lower_bound = [lb * 0.9 for lb in forecast.lower_bound]
            if forecast.upper_bound:
                forecast.upper_bound = [ub * 0.9 for ub in forecast.upper_bound]

        # Add metadata
        if forecast.metadata is None:
            forecast.metadata = {}
        forecast.metadata["scenario"] = scenario
        forecast.metadata["horizon_years"] = horizon / 365.0

        return forecast

    def multi_scenario_forecast(
        self, horizon: int, scenarios: Optional[List[str]] = None
    ) -> Dict[str, ForecastResult]:
        """
        Generate forecasts for multiple scenarios.

        Args:
            horizon: Forecast horizon
            scenarios: List of scenarios (uses defaults if None)

        Returns:
            Dictionary mapping scenario names to forecasts

        Example:
            >>> forecasts = forecaster.multi_scenario_forecast(
            ...     horizon=365*5, scenarios=['base', 'optimistic', 'pessimistic']
            ... )
        """
        scenarios = scenarios or ["base", "optimistic", "pessimistic"]
        results = {}

        for scenario in scenarios:
            results[scenario] = self.predict(horizon=horizon, scenario=scenario)

        return results

    def forecast_with_uncertainty(
        self, horizon: int, n_simulations: int = 1000
    ) -> ForecastResult:
        """
        Generate forecast with uncertainty quantification using Monte Carlo.

        Args:
            horizon: Forecast horizon
            n_simulations: Number of Monte Carlo simulations

        Returns:
            ForecastResult with prediction intervals from simulations
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")

        # Generate multiple forecasts with noise
        simulations = []

        base_forecast = self.base_forecaster.predict(horizon=horizon)
        base_predictions = np.array(base_forecast.predictions)

        # Estimate noise from prediction intervals if available
        if base_forecast.lower_bound and base_forecast.upper_bound:
            noise_std = (
                np.array(base_forecast.upper_bound) - np.array(base_forecast.lower_bound)
            ) / 4  # Approximate std from 95% CI

        else:
            # Use simple percentage-based noise
            noise_std = base_predictions * 0.1

        # Run simulations
        for _ in range(n_simulations):
            noise = np.random.normal(0, noise_std)
            sim_forecast = base_predictions + noise
            simulations.append(sim_forecast)

        simulations = np.array(simulations)

        # Calculate percentiles
        predictions = np.median(simulations, axis=0).tolist()
        lower_bound = np.percentile(simulations, 2.5, axis=0).tolist()
        upper_bound = np.percentile(simulations, 97.5, axis=0).tolist()

        return ForecastResult(
            model_type=ModelType.ENSEMBLE,
            timestamps=base_forecast.timestamps,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.95,
            metadata={"n_simulations": n_simulations, "method": "monte_carlo"},
        )
