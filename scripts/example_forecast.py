"""
Example script demonstrating time-series forecasting.

This script shows how to use the PV Circularity Simulator's forecasting
capabilities for energy prediction.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pv_simulator.core.schemas import TimeSeriesData, TimeSeriesFrequency
from pv_simulator.forecasting.statistical import ARIMAModel, SARIMAModel, StatisticalAnalyzer
from pv_simulator.forecasting.ml_forecaster import ProphetForecaster, XGBoostForecaster
from pv_simulator.forecasting.model_training import ModelTraining
from pv_simulator.forecasting.metrics import MetricsCalculator
from pv_simulator.forecasting.seasonal import SeasonalAnalyzer, LongTermForecaster


def create_sample_data() -> TimeSeriesData:
    """Create sample energy production data."""
    print("Creating sample PV energy production data...")

    # Generate 2 years of daily data
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(730)]

    # Simulate PV energy production with:
    # - Increasing trend (system degradation is minimal)
    # - Yearly seasonality (more sun in summer)
    # - Weekly pattern (slight variation)
    # - Random noise
    t = np.arange(len(timestamps))

    trend = 1000 + 0.5 * t  # Slight positive trend
    yearly_seasonal = 300 * np.sin(2 * np.pi * t / 365)  # Yearly pattern
    weekly_seasonal = 50 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
    noise = np.random.normal(0, 50, len(timestamps))

    values = (trend + yearly_seasonal + weekly_seasonal + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency=TimeSeriesFrequency.DAILY,
        name="pv_energy_production_kwh",
    )


def example_statistical_analysis():
    """Example 1: Statistical analysis of time series."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Statistical Analysis")
    print("=" * 80)

    data = create_sample_data()

    # Initialize analyzer
    analyzer = StatisticalAnalyzer()

    # Decompose seasonality
    print("\n1. Seasonal Decomposition:")
    decomposition = analyzer.seasonality_decomposition(data, model="additive", period=365)
    print(f"   Seasonality Strength: {decomposition.seasonality_strength:.3f}")
    print(f"   Trend Strength: {decomposition.trend_strength:.3f}")

    # Analyze trend
    print("\n2. Trend Analysis:")
    trend = analyzer.trend_analysis(data)
    print(f"   Trend Direction: {trend['trend_direction']}")
    print(f"   Slope: {trend['slope']:.4f} kWh/day")
    print(f"   R²: {trend['r_squared']:.3f}")

    # Test for stationarity
    print("\n3. Stationarity Test:")
    stationarity = analyzer.stationarity_test(data)
    print(f"   ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"   P-value: {stationarity['p_value']:.4f}")
    print(f"   Is Stationary: {stationarity['is_stationary']}")


def example_arima_forecast():
    """Example 2: ARIMA forecasting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ARIMA Forecasting")
    print("=" * 80)

    data = create_sample_data()

    # Split into train/test
    train_size = int(len(data.values) * 0.8)
    train_data = TimeSeriesData(
        timestamps=data.timestamps[:train_size],
        values=data.values[:train_size],
        frequency=data.frequency,
        name=data.name,
    )

    test_data = TimeSeriesData(
        timestamps=data.timestamps[train_size:],
        values=data.values[train_size:],
        frequency=data.frequency,
        name=data.name,
    )

    # Fit ARIMA model
    print("\n1. Fitting ARIMA model...")
    model = ARIMAModel(order=(2, 1, 2))
    model.fit(train_data)
    print(f"   Model fitted successfully!")
    print(f"   AIC: {model.fitted_model.aic:.2f}")
    print(f"   BIC: {model.fitted_model.bic:.2f}")

    # Generate forecast
    print("\n2. Generating forecast...")
    horizon = len(test_data.values)
    forecast = model.predict(horizon=horizon, confidence_level=0.95)

    # Evaluate
    print("\n3. Evaluating forecast:")
    metrics = model.evaluate(test_data, forecast)
    print(f"   MAE: {metrics.mae:.2f} kWh")
    print(f"   RMSE: {metrics.rmse:.2f} kWh")
    print(f"   MAPE: {metrics.mape:.2f}%")
    print(f"   R²: {metrics.r2:.3f}")


def example_prophet_forecast():
    """Example 3: Prophet forecasting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Prophet Forecasting")
    print("=" * 80)

    data = create_sample_data()

    # Fit Prophet model
    print("\n1. Fitting Prophet model...")
    model = ProphetForecaster(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=True,
    )
    model.fit(data)
    print("   Prophet model fitted successfully!")

    # Generate forecast for next 90 days
    print("\n2. Generating 90-day forecast...")
    forecast = model.predict(horizon=90, confidence_level=0.95)

    print(f"   Forecasted {len(forecast.predictions)} days")
    print(f"   Mean forecast: {np.mean(forecast.predictions):.2f} kWh/day")
    print(
        f"   Forecast range: {min(forecast.predictions):.2f} - {max(forecast.predictions):.2f} kWh/day"
    )


def example_model_selection():
    """Example 4: Automated model selection."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Automated Model Selection")
    print("=" * 80)

    data = create_sample_data()

    # Initialize trainer
    print("\n1. Comparing multiple models...")
    trainer = ModelTraining()

    # Select best model
    best_model_class, best_params, best_metrics = trainer.model_selection(
        data, horizon=30, metric="rmse"
    )

    print(f"\n2. Best Model: {best_model_class.__name__}")
    print(f"   Best Parameters: {best_params}")
    print(f"   Best RMSE: {best_metrics.rmse:.2f} kWh")
    print(f"   MAE: {best_metrics.mae:.2f} kWh")
    print(f"   MAPE: {best_metrics.mape:.2f}%")


def example_seasonal_analysis():
    """Example 5: Seasonal pattern analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Seasonal Pattern Analysis")
    print("=" * 80)

    data = create_sample_data()

    # Initialize analyzer
    analyzer = SeasonalAnalyzer()

    # Detect seasonality
    print("\n1. Detecting seasonal patterns...")
    seasonality = analyzer.detect_seasonality(data, max_period=365)
    print(f"   Has Seasonality: {seasonality['has_seasonality']}")
    print(f"   Dominant Period: {seasonality['dominant_period']} days")

    # Extract seasonal pattern
    if seasonality["has_seasonality"]:
        period = seasonality["dominant_period"]
        pattern = analyzer.extract_seasonal_pattern(data, period=period)
        print(f"\n2. Seasonal pattern (period={period}):")
        print(f"   Pattern Strength: {pattern.strength:.3f}")
        print(f"   Pattern Type: {pattern.pattern_type}")

    # Year-over-year comparison
    print("\n3. Year-over-year comparison:")
    yoy = analyzer.year_over_year_comparison(data)
    print(f"   Years analyzed: {yoy.years}")
    print(f"   Average growth: {yoy.average_growth:.2%}")
    print(f"   Trend: {yoy.trend}")


def example_long_term_forecast():
    """Example 6: Long-term forecasting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Long-term Multi-year Forecast")
    print("=" * 80)

    data = create_sample_data()

    # Initialize long-term forecaster
    print("\n1. Fitting long-term forecaster...")
    forecaster = LongTermForecaster()
    forecaster.fit(data)

    # Generate 3-year forecast
    print("\n2. Generating 3-year forecast...")
    forecast = forecaster.predict(horizon=365 * 3, scenario="base")

    print(f"   Forecast horizon: {len(forecast.predictions)} days")
    print(f"   Mean prediction: {np.mean(forecast.predictions):.2f} kWh/day")

    # Multi-scenario forecast
    print("\n3. Generating multi-scenario forecasts...")
    scenarios = forecaster.multi_scenario_forecast(
        horizon=365, scenarios=["base", "optimistic", "pessimistic"]
    )

    for scenario, forecast_result in scenarios.items():
        mean_pred = np.mean(forecast_result.predictions)
        print(f"   {scenario.capitalize()}: {mean_pred:.2f} kWh/day (avg)")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("PV CIRCULARITY SIMULATOR - FORECASTING EXAMPLES")
    print("=" * 80)

    try:
        example_statistical_analysis()
        example_arima_forecast()
        example_prophet_forecast()
        example_seasonal_analysis()
        example_long_term_forecast()

        # Note: Model selection is commented out as it takes longer
        # Uncomment to run: example_model_selection()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
