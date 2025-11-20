# PV Circularity Simulator

End-to-end PV lifecycle simulation platform with advanced time-series forecasting, machine learning models, and circular economy analysis.

## Features

### 🔮 Time-Series Forecasting System

#### Statistical Models (BATCH7-B09-S02)
- **ARIMA Model**: AutoRegressive Integrated Moving Average for trend-based forecasting
- **SARIMA Model**: Seasonal ARIMA for data with seasonal patterns
- **Exponential Smoothing**: Simple, double, and triple exponential smoothing
- **State Space Models**: Flexible framework for level, trend, and seasonal components
- **Statistical Analyzer**:
  - Seasonality decomposition
  - Trend analysis
  - Autocorrelation analysis
  - Stationarity testing

#### Machine Learning Models (BATCH7-B09-S01)
- **Prophet**: Facebook's forecasting model for multiple seasonality
- **XGBoost**: Gradient boosting with engineered features
- **LightGBM**: Fast gradient boosting alternative
- **LSTM Neural Networks**: Deep learning for long-term dependencies
- **Ensemble Methods**: Combine multiple models for robust predictions

#### Feature Engineering
- **Lag Features**: Past values for temporal dependencies
- **Rolling Features**: Moving averages, std, min, max
- **Temporal Features**: Hour, day, week, month, year with cyclical encoding
- **Weather Features**: Temperature, irradiance, wind speed, humidity integration

#### Model Training & Validation (BATCH7-B09-S05)
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Cross-Validation**: Time-series specific CV strategies
- **Model Selection**: Automated comparison and selection
- **Metrics Calculator**: MAE, RMSE, MAPE, R², SMAPE
- **Forecast vs Actual**: Comprehensive comparison tools

#### Seasonal Analysis (BATCH7-B09-S03)
- **Seasonal Analyzer**: Detect and extract seasonal patterns
- **Long-term Forecaster**: Multi-year predictions with uncertainty
- **Year-over-Year Comparison**: Growth trends and pattern analysis
- **Multi-scenario Forecasting**: Base, optimistic, pessimistic scenarios

#### Interactive Dashboards (BATCH7-B09-S04)
- **Forecast Dashboard**: Streamlit-based interactive UI
- **Interactive Charts**: Plotly visualizations with zoom, pan, hover
- **Confidence Intervals**: Multiple confidence levels visualization
- **Scenario Analysis**: Compare multiple forecast scenarios
- **Residual Analysis**: Diagnostic plots for model validation

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Using pip with optional dependencies

```bash
# Install with all features
pip install -e ".[all]"

# Or install specific components
pip install -e ".[forecasting]"  # Statistical and ML forecasting
pip install -e ".[deep-learning]"  # LSTM and neural networks
pip install -e ".[visualization]"  # Dashboards and plotting
pip install -e ".[dev]"  # Development tools
```

## Quick Start

### Basic Forecasting

```python
from datetime import datetime, timedelta
from pv_simulator.core.schemas import TimeSeriesData, TimeSeriesFrequency
from pv_simulator.forecasting.statistical import ARIMAModel

# Create time series data
timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
values = [100 + i * 0.5 for i in range(365)]  # Your actual data here

data = TimeSeriesData(
    timestamps=timestamps,
    values=values,
    frequency=TimeSeriesFrequency.DAILY,
    name="pv_energy_kwh"
)

# Fit ARIMA model
model = ARIMAModel(order=(2, 1, 2))
model.fit(data)

# Generate 30-day forecast
forecast = model.predict(horizon=30, confidence_level=0.95)

print(f"Forecasted values: {forecast.predictions}")
print(f"Lower bound: {forecast.lower_bound}")
print(f"Upper bound: {forecast.upper_bound}")
```

### Prophet Forecasting

```python
from pv_simulator.forecasting.ml_forecaster import ProphetForecaster

# Initialize Prophet model
model = ProphetForecaster(
    seasonality_mode="additive",
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Fit and predict
model.fit(data)
forecast = model.predict(horizon=90)
```

### Feature Engineering

```python
from pv_simulator.forecasting.feature_engineering import FeatureEngineering
from pv_simulator.core.schemas import FeatureConfig

# Configure feature engineering
config = FeatureConfig(
    lag_features=True,
    lag_periods=[1, 7, 30],
    rolling_features=True,
    rolling_windows=[7, 14, 30],
    temporal_features=True,
    cyclical_encoding=True
)

# Create features
fe = FeatureEngineering(config)
features_df = fe.create_all_features(data)
```

### Model Training with Hyperparameter Tuning

```python
from pv_simulator.forecasting.model_training import ModelTraining

# Initialize trainer
trainer = ModelTraining()

# Automated model selection
best_model, best_params, metrics = trainer.model_selection(
    data=data,
    horizon=30,
    metric="rmse"
)

print(f"Best model: {best_model.__name__}")
print(f"RMSE: {metrics.rmse:.2f}")
print(f"MAE: {metrics.mae:.2f}")
```

### Seasonal Analysis

```python
from pv_simulator.forecasting.seasonal import SeasonalAnalyzer

analyzer = SeasonalAnalyzer()

# Detect seasonality
seasonality = analyzer.detect_seasonality(data)
print(f"Dominant period: {seasonality['dominant_period']}")

# Year-over-year comparison
yoy = analyzer.year_over_year_comparison(data)
print(f"Average growth: {yoy.average_growth:.2%}")
print(f"Trend: {yoy.trend}")
```

### Long-term Forecasting

```python
from pv_simulator.forecasting.seasonal import LongTermForecaster

# Initialize forecaster
forecaster = LongTermForecaster()
forecaster.fit(data)

# Generate 3-year forecast
forecast = forecaster.predict(horizon=365*3, scenario="base")

# Multi-scenario forecast
scenarios = forecaster.multi_scenario_forecast(
    horizon=365*5,
    scenarios=["base", "optimistic", "pessimistic"]
)
```

### Interactive Dashboard

```python
from pv_simulator.dashboards.forecast_dashboard import ForecastDashboard

# Create dashboard
dashboard = ForecastDashboard("PV Energy Forecast")

# Build Streamlit dashboard (in a .py file)
dashboard.build_streamlit_dashboard(
    actual=historical_data,
    forecast=forecast_result,
    metrics=metrics
)

# Or create individual charts
fig = dashboard.interactive_charts(historical_data, forecast_result)
fig.show()
```

## Examples

Run the example script to see all features in action:

```bash
python scripts/example_forecast.py
```

This demonstrates:
1. Statistical analysis (decomposition, trend, autocorrelation)
2. ARIMA forecasting
3. Prophet forecasting
4. Automated model selection
5. Seasonal pattern analysis
6. Long-term multi-year forecasting

## Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/
│   ├── core/                      # Core models and schemas
│   │   ├── models.py              # Base forecaster classes
│   │   └── schemas.py             # Pydantic data models
│   ├── forecasting/               # Forecasting module
│   │   ├── statistical.py         # ARIMA, SARIMA, etc.
│   │   ├── ml_forecaster.py       # Prophet, XGBoost, LSTM
│   │   ├── feature_engineering.py # Feature creation
│   │   ├── model_training.py      # Training & tuning
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── seasonal.py            # Seasonal analysis
│   └── dashboards/                # Visualization
│       └── forecast_dashboard.py  # Streamlit dashboard
├── tests/                         # Test suite
├── scripts/                       # Example scripts
├── docs/                          # Documentation
└── notebooks/                     # Jupyter notebooks
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/test_forecasting/test_statistical.py
```

## Documentation

### Core Concepts

- **TimeSeriesData**: Pydantic model for time series with validation
- **ForecastResult**: Standardized forecast output with confidence intervals
- **ModelMetrics**: Comprehensive evaluation metrics
- **BaseForecaster**: Abstract base class for all forecasting models

### Model Types

All models inherit from `BaseForecaster` and implement:
- `fit(data)`: Train the model
- `predict(horizon)`: Generate forecasts
- `evaluate(actual, predicted)`: Calculate metrics

### Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **R²**: Coefficient of determination

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator},
  author = {PV Circularity Team},
  year = {2024},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Acknowledgments

Built with:
- [statsmodels](https://www.statsmodels.org/) - Statistical models
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [Streamlit](https://streamlit.io/) - Interactive dashboards
- [Plotly](https://plotly.com/) - Visualization
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

## Contact

For questions and support, please open an issue on GitHub.
