"""Example usage of TimeSeriesForecaster.

This example demonstrates how to use the TimeSeriesForecaster class
for photovoltaic energy production forecasting.
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_circularity.forecasting import TimeSeriesForecaster
from pv_circularity.utils.validators import (
    ARIMAConfig,
    EnsembleConfig,
    ForecastMethod,
    LSTMConfig,
    ProphetConfig,
    TimeSeriesData,
)


def generate_sample_pv_data(n_days: int = 365) -> TimeSeriesData:
    """Generate synthetic PV energy production data.

    Args:
        n_days: Number of days of data to generate.

    Returns:
        TimeSeriesData with synthetic PV production values.
    """
    np.random.seed(42)
    timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Create realistic PV production pattern
    t = np.arange(n_days)

    # Yearly seasonality (more production in summer)
    yearly = 20 * np.sin(2 * np.pi * t / 365 - np.pi / 2)

    # Weekly pattern (slightly lower on weekends)
    weekly = 2 * np.sin(2 * np.pi * t / 7)

    # Gradual degradation over time (0.5% per year)
    degradation = -0.005 * t / 365 * 100

    # Random weather effects
    noise = np.random.randn(n_days) * 5

    # Base production + patterns
    values = (100 + yearly + weekly + degradation + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency="D",
        name="pv_production_kwh",
    )


def example_arima_forecast():
    """Example: ARIMA forecasting for PV production."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: ARIMA Forecasting")
    print("=" * 80)

    # Generate data
    data = generate_sample_pv_data(n_days=365)

    # Initialize forecaster
    forecaster = TimeSeriesForecaster(data=data, verbose=True)

    # Configure ARIMA with seasonal component
    config = ARIMAConfig(
        p=2,  # AR order
        d=1,  # Differencing
        q=2,  # MA order
        seasonal_order=(1, 0, 1, 7),  # Weekly seasonality
        trend="c",
    )

    # Generate forecast for next 30 days
    result = forecaster.arima_forecast(
        steps=30,
        config=config,
        return_confidence_intervals=True,
    )

    print(f"\nForecast Method: {result.method.value}")
    print(f"Number of predictions: {len(result.predictions)}")
    print(f"AIC: {result.metrics['aic']:.2f}")
    print(f"BIC: {result.metrics['bic']:.2f}")
    print(f"\nFirst 5 predictions:")
    for i in range(5):
        print(
            f"  {result.timestamps[i].date()}: {result.predictions[i]:.2f} kWh "
            f"[{result.confidence_intervals['lower'][i]:.2f}, "
            f"{result.confidence_intervals['upper'][i]:.2f}]"
        )

    return forecaster, result


def example_prophet_forecast():
    """Example: Prophet forecasting for PV production."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Prophet Forecasting")
    print("=" * 80)

    # Generate data
    data = generate_sample_pv_data(n_days=365)

    # Initialize forecaster
    forecaster = TimeSeriesForecaster(data=data, verbose=True)

    # Configure Prophet
    config = ProphetConfig(
        growth="linear",
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # Generate forecast for next 30 days
    result = forecaster.prophet_forecast(
        steps=30,
        config=config,
        return_confidence_intervals=True,
    )

    print(f"\nForecast Method: {result.method.value}")
    print(f"Number of predictions: {len(result.predictions)}")
    print(f"\nFirst 5 predictions:")
    for i in range(5):
        print(
            f"  {result.timestamps[i].date()}: {result.predictions[i]:.2f} kWh "
            f"[{result.confidence_intervals['lower'][i]:.2f}, "
            f"{result.confidence_intervals['upper'][i]:.2f}]"
        )

    return forecaster, result


def example_lstm_forecast():
    """Example: LSTM forecasting for PV production."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: LSTM Neural Network Forecasting")
    print("=" * 80)

    try:
        # Generate data
        data = generate_sample_pv_data(n_days=365)

        # Initialize forecaster
        forecaster = TimeSeriesForecaster(data=data, verbose=True)

        # Configure LSTM
        config = LSTMConfig(
            n_layers=2,
            hidden_units=64,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,
            lookback_window=14,  # Use past 2 weeks
            validation_split=0.2,
        )

        # Generate forecast for next 30 days
        result = forecaster.lstm_forecast(
            steps=30,
            config=config,
            return_training_history=True,
        )

        print(f"\nForecast Method: {result.method.value}")
        print(f"Number of predictions: {len(result.predictions)}")
        print(f"Final Training Loss: {result.metrics['final_train_loss']:.4f}")
        print(f"Final Validation Loss: {result.metrics['final_val_loss']:.4f}")
        print(f"\nFirst 5 predictions:")
        for i in range(5):
            print(f"  {result.timestamps[i].date()}: {result.predictions[i]:.2f} kWh")

        return forecaster, result

    except RuntimeError as e:
        print(f"\nLSTM forecasting skipped: {e}")
        return None, None


def example_ensemble_forecast():
    """Example: Ensemble forecasting combining multiple methods."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Ensemble Forecasting")
    print("=" * 80)

    # Generate data
    data = generate_sample_pv_data(n_days=365)

    # Initialize forecaster
    forecaster = TimeSeriesForecaster(data=data, verbose=True)

    # Configure ensemble (ARIMA + Prophet, avoiding LSTM for speed)
    config = EnsembleConfig(
        methods=[ForecastMethod.ARIMA, ForecastMethod.PROPHET],
        weights=[0.6, 0.4],  # 60% ARIMA, 40% Prophet
        aggregation="weighted",
    )

    # Generate forecast
    result = forecaster.ensemble_predictions(
        steps=30,
        config=config,
        arima_config=ARIMAConfig(p=1, d=1, q=1),
        prophet_config=ProphetConfig(),
    )

    print(f"\nForecast Method: {result.method.value}")
    print(f"Methods used: {result.metrics['methods_used']}")
    print(f"Number of predictions: {len(result.predictions)}")
    print(f"\nFirst 5 predictions:")
    for i in range(5):
        pred = result.predictions[i]
        ci_min = result.confidence_intervals["lower"][i]
        ci_max = result.confidence_intervals["upper"][i]
        print(f"  {result.timestamps[i].date()}: {pred:.2f} kWh [{ci_min:.2f}, {ci_max:.2f}]")

    return forecaster, result


def visualize_forecast(data: TimeSeriesData, result, title: str):
    """Visualize forecast results.

    Args:
        data: Historical time series data.
        result: Forecast result.
        title: Plot title.
    """
    plt.figure(figsize=(14, 6))

    # Plot historical data
    plt.plot(data.timestamps, data.values, label="Historical", color="blue", linewidth=1.5)

    # Plot forecast
    plt.plot(result.timestamps, result.predictions, label="Forecast", color="red", linewidth=2)

    # Plot confidence intervals if available
    if result.confidence_intervals:
        plt.fill_between(
            result.timestamps,
            result.confidence_intervals["lower"],
            result.confidence_intervals["upper"],
            alpha=0.3,
            color="red",
            label="Confidence Interval",
        )

    plt.xlabel("Date")
    plt.ylabel("PV Production (kWh)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"forecast_{result.method.value}.png", dpi=150)
    print(f"\nSaved plot to: forecast_{result.method.value}.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("PV CIRCULARITY SIMULATOR - TIME SERIES FORECASTING EXAMPLES")
    print("=" * 80)

    # Run examples
    forecaster1, result1 = example_arima_forecast()
    if result1:
        visualize_forecast(forecaster1.data, result1, "ARIMA Forecast")

    forecaster2, result2 = example_prophet_forecast()
    if result2:
        visualize_forecast(forecaster2.data, result2, "Prophet Forecast")

    forecaster3, result3 = example_lstm_forecast()
    if result3:
        visualize_forecast(forecaster3.data, result3, "LSTM Forecast")

    forecaster4, result4 = example_ensemble_forecast()
    if result4:
        visualize_forecast(forecaster4.data, result4, "Ensemble Forecast")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
