"""
Example usage of the Forecast Dashboard with Accuracy Metrics.

This script demonstrates:
1. Creating forecast data with predictions and actuals
2. Calculating accuracy metrics (MAE, RMSE, MAPE, R²)
3. Generating confidence intervals
4. Creating interactive Plotly dashboards
5. Exporting visualizations to HTML

Run this example:
    python examples/forecast_dashboard_example.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add src to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pv_simulator.forecasting.models import ForecastPoint, ForecastSeries, ForecastData
from pv_simulator.dashboard.forecast_dashboard import (
    ForecastDashboard,
    accuracy_metrics,
    mae_rmse_calculation,
    confidence_intervals,
)


def create_sample_forecast_data(n_points: int = 48, seed: int = 42) -> ForecastData:
    """
    Create sample forecast data for demonstration.

    Args:
        n_points: Number of forecast points to generate
        seed: Random seed for reproducibility

    Returns:
        ForecastData: Sample forecast with predictions and actuals
    """
    np.random.seed(seed)

    # Generate timestamps (hourly forecast for next 48 hours)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]

    # Generate realistic PV energy production pattern
    # Base pattern: sinusoidal to simulate day/night cycle
    hours = np.arange(n_points)
    base_pattern = 150 * np.maximum(0, np.sin(hours * np.pi / 24))

    # Add trend and noise to predictions
    trend = np.linspace(0, 10, n_points)
    prediction_noise = np.random.normal(0, 5, n_points)
    predicted_values = base_pattern + trend + prediction_noise

    # Create actuals with different noise
    actual_noise = np.random.normal(0, 8, n_points)
    actual_values = base_pattern + trend + actual_noise

    # Create forecast points
    points = [
        ForecastPoint(
            timestamp=ts,
            predicted=max(0, pred),  # Energy can't be negative
            actual=max(0, act),
        )
        for ts, pred, act in zip(timestamps, predicted_values, actual_values)
    ]

    # Create forecast series
    series = ForecastSeries(
        id="pv-forecast-001",
        name="PV Energy Production Forecast",
        points=points,
        model_name="ARIMA(2,1,2)",
        parameters={
            "p": 2,
            "d": 1,
            "q": 2,
            "seasonal": False,
        },
    )

    # Create complete forecast data
    forecast_data = ForecastData(
        id="forecast-2024-01-01",
        name="48-Hour PV Production Forecast",
        description="Hourly PV energy production forecast with actual measurements",
        series=series,
        metadata={
            "location": "Solar Farm Site A",
            "capacity_kw": 500,
            "units": "kWh",
            "forecast_horizon": "48_hours",
        },
    )

    return forecast_data


def example_basic_metrics():
    """Example 1: Calculate basic accuracy metrics."""
    print("=" * 80)
    print("Example 1: Basic Accuracy Metrics Calculation")
    print("=" * 80)

    # Sample data
    actual = np.array([100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 125.0, 122.0])
    predicted = np.array([98.0, 112.0, 107.0, 113.0, 122.0, 116.0, 127.0, 120.0])

    # Calculate MAE and RMSE
    mae, rmse, mse = mae_rmse_calculation(actual, predicted)
    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE:  {mse:.4f}")

    # Calculate comprehensive metrics
    metrics = accuracy_metrics(actual, predicted)
    print(f"\nComprehensive Metrics:")
    print(f"  MAE:      {metrics.mae:.4f}")
    print(f"  RMSE:     {metrics.rmse:.4f}")
    print(f"  MSE:      {metrics.mse:.4f}")
    print(f"  MAPE:     {metrics.mape:.4f}%" if metrics.mape else "  MAPE:     N/A")
    print(f"  R²:       {metrics.r2_score:.4f}" if metrics.r2_score else "  R²:       N/A")
    print(f"  Bias:     {metrics.bias:.4f}")
    print(f"  Samples:  {metrics.n_samples}")


def example_confidence_intervals():
    """Example 2: Generate confidence intervals."""
    print("\n" + "=" * 80)
    print("Example 2: Confidence Interval Generation")
    print("=" * 80)

    predicted = np.array([100.0, 110.0, 105.0, 115.0, 120.0])

    # Method 1: Normal distribution (without residuals)
    lower_normal, upper_normal = confidence_intervals(
        predicted=predicted,
        confidence_level=0.95,
        method="normal",
    )

    print(f"\nNormal Method (95% CI):")
    for i, (pred, lower, upper) in enumerate(zip(predicted, lower_normal, upper_normal)):
        print(f"  Point {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")

    # Method 2: With historical residuals
    np.random.seed(42)
    residuals = np.random.normal(0, 5, 100)

    lower_percentile, upper_percentile = confidence_intervals(
        predicted=predicted,
        residuals=residuals,
        confidence_level=0.95,
        method="percentile",
    )

    print(f"\nPercentile Method (95% CI, with residuals):")
    for i, (pred, lower, upper) in enumerate(zip(predicted, lower_percentile, upper_percentile)):
        print(f"  Point {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")

    # Different confidence levels
    lower_90, upper_90 = confidence_intervals(predicted, confidence_level=0.90)
    lower_99, upper_99 = confidence_intervals(predicted, confidence_level=0.99)

    print(f"\nDifferent Confidence Levels for Point 1 (Predicted={predicted[0]:.2f}):")
    print(f"  90% CI: [{lower_90[0]:.2f}, {upper_90[0]:.2f}]")
    print(f"  95% CI: [{lower_normal[0]:.2f}, {upper_normal[0]:.2f}]")
    print(f"  99% CI: [{lower_99[0]:.2f}, {upper_99[0]:.2f}]")


def example_dashboard_creation():
    """Example 3: Create interactive dashboard."""
    print("\n" + "=" * 80)
    print("Example 3: Interactive Dashboard Creation")
    print("=" * 80)

    # Create sample forecast data
    forecast_data = create_sample_forecast_data(n_points=48)

    # Create dashboard
    dashboard = ForecastDashboard(
        forecast_data=forecast_data,
        title="PV Energy Production Forecast Dashboard",
        width=1400,
        height=1000,
    )

    print(f"\nDashboard created for: {forecast_data.name}")
    print(f"Forecast points: {forecast_data.series.length}")
    print(f"Model: {forecast_data.series.model_name}")

    # Display metrics
    if dashboard.forecast_data.metrics:
        print(f"\nAutomatically Calculated Metrics:")
        metrics_dict = dashboard.forecast_data.metrics.to_summary_dict()
        for key, value in metrics_dict.items():
            print(f"  {key}: {value}")

    # Create visualizations
    print(f"\nGenerating visualizations...")

    # 1. Basic forecast visualization
    fig_forecast = dashboard.forecast_visualization(
        show_confidence=True,
        show_actuals=True,
        confidence_level=0.95,
    )
    print(f"  ✓ Forecast time series plot created")

    # 2. Error analysis
    fig_error = dashboard.create_error_analysis()
    if fig_error:
        print(f"  ✓ Error analysis plot created")

    # 3. Metrics table
    fig_metrics = dashboard.create_metrics_table()
    if fig_metrics:
        print(f"  ✓ Metrics table created")

    # 4. Complete dashboard
    fig_dashboard = dashboard.create_dashboard(
        show_confidence=True,
        show_actuals=True,
        show_error_analysis=True,
        confidence_level=0.95,
    )
    print(f"  ✓ Complete dashboard created")

    return dashboard, fig_dashboard


def example_export_dashboard():
    """Example 4: Export dashboard to HTML."""
    print("\n" + "=" * 80)
    print("Example 4: Export Dashboard to HTML")
    print("=" * 80)

    # Create forecast data
    forecast_data = create_sample_forecast_data(n_points=72)  # 3-day forecast

    # Create dashboard
    dashboard = ForecastDashboard(
        forecast_data=forecast_data,
        title="72-Hour PV Production Forecast",
    )

    # Export to HTML
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "forecast_dashboard.html"

    dashboard.export_html(
        filename=str(output_file),
        show_confidence=True,
        show_actuals=True,
        show_error_analysis=True,
    )

    print(f"\nDashboard exported to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"\nOpen the file in a web browser to view the interactive dashboard.")


def example_forecast_comparison():
    """Example 5: Compare multiple forecast models."""
    print("\n" + "=" * 80)
    print("Example 5: Compare Multiple Forecast Models")
    print("=" * 80)

    # Create forecasts with different models
    np.random.seed(42)
    n_points = 24

    # Generate timestamps and actual values
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
    hours = np.arange(n_points)
    actual_base = 150 * np.maximum(0, np.sin(hours * np.pi / 24))
    actual = actual_base + np.random.normal(0, 5, n_points)

    models = {
        "ARIMA": {"noise_std": 6, "bias": 0},
        "LSTM": {"noise_std": 8, "bias": 2},
        "Prophet": {"noise_std": 7, "bias": -1},
    }

    results = {}

    for model_name, params in models.items():
        # Generate predictions
        predicted = (
            actual_base +
            np.random.normal(params["bias"], params["noise_std"], n_points)
        )

        # Create forecast points
        points = [
            ForecastPoint(timestamp=ts, predicted=max(0, pred), actual=max(0, act))
            for ts, pred, act in zip(timestamps, predicted, actual)
        ]

        # Create series and forecast data
        series = ForecastSeries(points=points, model_name=model_name)
        forecast_data = ForecastData(series=series)

        # Calculate metrics
        metrics = accuracy_metrics(
            actual=forecast_data.series.get_actual_values(),
            predicted=forecast_data.series.get_predicted_values(),
        )

        results[model_name] = metrics

    # Display comparison
    print(f"\nModel Comparison Results:")
    print(f"\n{'Model':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Bias':<10}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<10} "
            f"{metrics.mae:<10.4f} "
            f"{metrics.rmse:<10.4f} "
            f"{metrics.mape if metrics.mape else 'N/A':<10} "
            f"{metrics.r2_score if metrics.r2_score else 'N/A':<10} "
            f"{metrics.bias:<10.4f}"
        )

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1].rmse)
    print(f"\nBest Model (lowest RMSE): {best_model[0]} (RMSE={best_model[1].rmse:.4f})")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Forecast Dashboard & Accuracy Metrics - Example Usage".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run examples
    example_basic_metrics()
    example_confidence_intervals()
    dashboard, fig = example_dashboard_creation()
    example_export_dashboard()
    example_forecast_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The Forecast Dashboard provides:

1. ✓ Comprehensive accuracy metrics (MAE, RMSE, MAPE, R², Bias)
2. ✓ Flexible confidence interval calculations
3. ✓ Interactive Plotly visualizations
4. ✓ Error analysis and diagnostics
5. ✓ Production-ready with full validation

Key Features:
- Pydantic models for type safety and validation
- Statistical validation and error handling
- Export to HTML for sharing
- Support for multiple forecast comparison
- Full docstrings and examples

For more information, see the documentation or run:
    python -m pydoc pv_simulator.dashboard.forecast_dashboard
    """)


if __name__ == "__main__":
    main()
