# Forecast Dashboard & Accuracy Metrics Guide

## Overview

The Forecast Dashboard module provides comprehensive tools for visualizing energy forecasts and evaluating their accuracy. Built with Plotly for interactive visualizations and Pydantic for robust data validation, this module is production-ready with full type hints and extensive documentation.

## Features

### Core Functionality

1. **Accuracy Metrics Calculation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Squared Error (MSE)
   - Mean Absolute Percentage Error (MAPE)
   - R² (Coefficient of Determination)
   - Forecast Bias

2. **Confidence Intervals**
   - Normal distribution method
   - Percentile method
   - Bootstrap method
   - Configurable confidence levels

3. **Interactive Visualizations**
   - Time series forecast plots
   - Confidence interval bands
   - Error distribution histograms
   - Residual analysis
   - Metrics summary tables

4. **Statistical Validation**
   - Input validation
   - NaN and Inf checking
   - Array length matching
   - Confidence level validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from datetime import datetime, timedelta
import numpy as np

from pv_simulator.forecasting.models import ForecastPoint, ForecastSeries, ForecastData
from pv_simulator.dashboard import ForecastDashboard, accuracy_metrics

# Create forecast data
timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(24)]
predicted = np.array([100, 110, 105, 115, 120, ...])
actual = np.array([98, 112, 107, 113, 122, ...])

points = [
    ForecastPoint(timestamp=ts, predicted=pred, actual=act)
    for ts, pred, act in zip(timestamps, predicted, actual)
]

series = ForecastSeries(points=points, model_name="ARIMA")
forecast_data = ForecastData(series=series)

# Create dashboard
dashboard = ForecastDashboard(forecast_data, title="My Forecast")

# Generate visualizations
fig = dashboard.create_dashboard(show_confidence=True)
fig.show()

# Export to HTML
dashboard.export_html("forecast_dashboard.html")
```

## API Reference

### Functions

#### `mae_rmse_calculation(actual, predicted)`

Calculate MAE, RMSE, and MSE.

**Parameters:**
- `actual`: Array of actual values
- `predicted`: Array of predicted values

**Returns:**
- Tuple of (MAE, RMSE, MSE)

**Example:**
```python
mae, rmse, mse = mae_rmse_calculation(actual, predicted)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

#### `accuracy_metrics(actual, predicted, include_mape=True, include_r2=True)`

Calculate comprehensive accuracy metrics.

**Parameters:**
- `actual`: Array of actual values
- `predicted`: Array of predicted values
- `include_mape`: Whether to calculate MAPE
- `include_r2`: Whether to calculate R²

**Returns:**
- `AccuracyMetrics` Pydantic model

**Example:**
```python
metrics = accuracy_metrics(actual, predicted)
print(f"RMSE: {metrics.rmse:.4f}")
print(f"R²: {metrics.r2_score:.4f}")
```

#### `confidence_intervals(predicted, residuals=None, confidence_level=0.95, method='normal')`

Generate confidence intervals for predictions.

**Parameters:**
- `predicted`: Array of predicted values
- `residuals`: Optional array of residuals
- `confidence_level`: Confidence level (0.0 to 1.0)
- `method`: 'normal', 'percentile', or 'bootstrap'

**Returns:**
- Tuple of (lower_bounds, upper_bounds)

**Example:**
```python
lower, upper = confidence_intervals(
    predicted,
    residuals=residuals,
    confidence_level=0.95,
    method='percentile'
)
```

### Classes

#### `ForecastDashboard`

Interactive dashboard for forecast visualization.

**Initialization:**
```python
dashboard = ForecastDashboard(
    forecast_data,
    title="Forecast Dashboard",
    width=1400,
    height=1000
)
```

**Methods:**

- `forecast_visualization(show_confidence=True, show_actuals=True, confidence_level=0.95)`
  - Create forecast time series plot
  - Returns Plotly Figure

- `create_error_analysis()`
  - Create error distribution and residual plots
  - Returns Plotly Figure or None

- `create_metrics_table()`
  - Create formatted metrics table
  - Returns Plotly Figure or None

- `create_dashboard(show_confidence=True, show_actuals=True, show_error_analysis=True)`
  - Create comprehensive dashboard
  - Returns Plotly Figure

- `export_html(filename, **kwargs)`
  - Export dashboard to HTML file

## Pydantic Models

### `ForecastPoint`

Single forecast data point.

**Fields:**
- `timestamp`: datetime
- `predicted`: float
- `actual`: Optional[float]
- `lower_bound`: Optional[float]
- `upper_bound`: Optional[float]
- `confidence_level`: Optional[float]

**Computed Properties:**
- `error`: predicted - actual
- `absolute_error`: |error|
- `percentage_error`: (error / actual) * 100

### `ForecastSeries`

Collection of forecast points.

**Fields:**
- `id`: Optional[str]
- `name`: Optional[str]
- `points`: List[ForecastPoint]
- `horizon`: ForecastHorizon
- `model_name`: str
- `created_at`: datetime
- `parameters`: Optional[Dict]

**Computed Properties:**
- `length`: Number of points
- `has_actuals`: Whether actuals are available

**Methods:**
- `get_predicted_values()`: Extract predictions as numpy array
- `get_actual_values()`: Extract actuals as numpy array
- `get_timestamps()`: Extract timestamps as list

### `AccuracyMetrics`

Forecast accuracy metrics.

**Fields:**
- `mae`: float
- `rmse`: float
- `mse`: float
- `mape`: Optional[float]
- `r2_score`: Optional[float]
- `bias`: float
- `n_samples`: int

**Methods:**
- `to_summary_dict()`: Convert to formatted dictionary

### `ForecastData`

Complete forecast with metadata.

**Fields:**
- `id`: Optional[str]
- `name`: Optional[str]
- `series`: ForecastSeries
- `metrics`: Optional[AccuracyMetrics]
- `metadata`: Optional[Dict]

## Examples

### Example 1: Basic Metrics

```python
from pv_simulator.dashboard import mae_rmse_calculation, accuracy_metrics
import numpy as np

actual = np.array([100, 110, 105, 115, 120])
predicted = np.array([98, 112, 107, 113, 122])

# Quick MAE/RMSE
mae, rmse, mse = mae_rmse_calculation(actual, predicted)

# Comprehensive metrics
metrics = accuracy_metrics(actual, predicted)
print(metrics.to_summary_dict())
```

### Example 2: Confidence Intervals

```python
from pv_simulator.dashboard import confidence_intervals
import numpy as np

predicted = np.array([100, 110, 105, 115, 120])
residuals = np.random.normal(0, 5, 100)

# 95% confidence intervals
lower, upper = confidence_intervals(
    predicted,
    residuals=residuals,
    confidence_level=0.95,
    method='percentile'
)

# Print intervals
for pred, lo, hi in zip(predicted, lower, upper):
    print(f"Prediction: {pred:.1f}, CI: [{lo:.1f}, {hi:.1f}]")
```

### Example 3: Interactive Dashboard

```python
from datetime import datetime, timedelta
from pv_simulator.forecasting.models import ForecastPoint, ForecastSeries, ForecastData
from pv_simulator.dashboard import ForecastDashboard
import numpy as np

# Create data
timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(48)]
predicted = 100 + np.linspace(0, 20, 48) + np.random.normal(0, 2, 48)
actual = predicted + np.random.normal(0, 3, 48)

points = [
    ForecastPoint(timestamp=ts, predicted=pred, actual=act)
    for ts, pred, act in zip(timestamps, predicted, actual)
]

series = ForecastSeries(points=points, model_name="ARIMA")
forecast_data = ForecastData(series=series)

# Create and display dashboard
dashboard = ForecastDashboard(forecast_data, title="48-Hour Forecast")
fig = dashboard.create_dashboard()
fig.show()  # Opens in browser

# Or export to HTML
dashboard.export_html("my_forecast.html")
```

### Example 4: Model Comparison

```python
from pv_simulator.dashboard import accuracy_metrics

# Compare two models
models = {
    "ARIMA": (actual, predicted_arima),
    "LSTM": (actual, predicted_lstm),
}

for name, (act, pred) in models.items():
    metrics = accuracy_metrics(act, pred)
    print(f"{name}: RMSE={metrics.rmse:.4f}, R²={metrics.r2_score:.4f}")
```

## Testing

The module includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/unit/test_dashboard.py

# Run integration tests
pytest -m integration
```

## Best Practices

1. **Always validate input data**
   - Check for NaN and Inf values
   - Ensure arrays have matching lengths
   - Validate confidence levels are between 0 and 1

2. **Use appropriate methods**
   - Normal method: Fast, assumes normal distribution
   - Percentile method: Robust to non-normality
   - Bootstrap method: Most accurate but slower

3. **Handle missing actuals**
   - Dashboard gracefully handles missing actuals
   - Metrics are only calculated when actuals exist

4. **Export for sharing**
   - Use `export_html()` for standalone dashboards
   - HTML files include all interactivity

## Performance Considerations

- Dashboard creation is fast for up to 10,000 points
- Bootstrap method is slower but more accurate
- HTML exports are self-contained (can be large)
- Use `include_mape=False` when actuals have zeros

## Troubleshooting

### Common Issues

**Issue: ValueError with empty arrays**
- Solution: Ensure input arrays are not empty

**Issue: MAPE is None**
- Solution: Actual values contain zeros, use MAE instead

**Issue: R² is negative**
- Solution: Model performs worse than mean baseline

**Issue: Confidence intervals too wide**
- Solution: Consider using actual residuals from validation set

## References

1. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.)
2. Willmott, C.J., & Matsuura, K. (2005). Advantages of the mean absolute error
3. Chatfield, C. (1993). Calculating interval forecasts

## License

MIT License - see LICENSE file for details
