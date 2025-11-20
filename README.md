# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### 🔮 Time-Series Forecasting
Production-ready time-series forecasting with multiple methods:
- **ARIMA/SARIMA**: Classical statistical forecasting with seasonal components
- **Prophet**: Facebook's forecasting procedure for time series with strong seasonal effects
- **LSTM**: Deep learning with Long Short-Term Memory neural networks
- **Ensemble**: Combine multiple methods for robust predictions

### 🌡️ IR Image Processing
Comprehensive thermal image analysis for photovoltaic systems:
- Hot spot detection and analysis
- Temperature mapping and monitoring
- Seasonal decomposition of thermal time series
- Trend analysis (linear, polynomial, LOWESS)
- Residual analysis with statistical diagnostics

### 📊 Data Validation
- Pydantic models for robust data validation
- Type-safe configuration management
- Comprehensive error handling

## Installation

### Requirements
- Python 3.9+
- pip or conda

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

## Quick Start

### Time-Series Forecasting

```python
from datetime import datetime, timedelta
import numpy as np
from pv_circularity.forecasting import TimeSeriesForecaster
from pv_circularity.utils.validators import TimeSeriesData, ARIMAConfig

# Create time series data
timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(365)]
values = np.cumsum(np.random.randn(365)) + 100
ts_data = TimeSeriesData(timestamps=timestamps, values=values.tolist())

# Initialize forecaster
forecaster = TimeSeriesForecaster(data=ts_data)

# ARIMA forecast
arima_result = forecaster.arima_forecast(steps=30)
print(f"30-day forecast: {arima_result.predictions[:5]}")

# Prophet forecast
prophet_result = forecaster.prophet_forecast(steps=30)

# Ensemble forecast (combines multiple methods)
ensemble_result = forecaster.ensemble_predictions(steps=30)
```

### IR Image Processing

```python
from pv_circularity.processing import IRImageProcessing
import numpy as np

# Load IR image
processor = IRImageProcessing.from_file("thermal_image.png")

# Detect hot spots
hot_spots = processor.detect_hot_spots(threshold_percentile=95)
print(f"Found {len(hot_spots)} hot spots")

# Seasonal decomposition of temperature data
decomposition = processor.seasonal_decomposition(
    time_series_data=temp_series,
    period=7  # Weekly seasonality
)

# Trend analysis
trend = processor.trend_analysis(temp_series, method="linear")
print(f"Temperature trend: {trend['slope']:.4f}°C/day")

# Residual analysis
residuals = processor.residual_analysis(temp_series, decomposition_period=7)
print(f"Outliers detected: {residuals['outliers']['count']}")
```

## Examples

Comprehensive examples are available in the `examples/` directory:

```bash
# Time-series forecasting examples
python examples/forecasting_example.py

# IR image processing examples
python examples/ir_processing_example.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_circularity --cov-report=html

# Run specific test module
pytest tests/unit/test_time_series_forecaster.py

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -v
```

## Documentation

### Project Structure

```
pv-circularity-simulator/
├── src/pv_circularity/           # Main package
│   ├── forecasting/               # Time-series forecasting
│   │   ├── time_series_forecaster.py
│   │   └── models/
│   ├── processing/                # Image processing
│   │   └── ir_image_processing.py
│   ├── utils/                     # Utilities
│   │   └── validators.py          # Pydantic models
│   └── config/                    # Configuration
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── examples/                      # Usage examples
├── docs/                          # Documentation
└── notebooks/                     # Jupyter notebooks
```

### API Reference

#### TimeSeriesForecaster

Main class for time-series forecasting:

- `arima_forecast(steps, config)`: ARIMA/SARIMA forecasting
- `prophet_forecast(steps, config)`: Prophet forecasting
- `lstm_forecast(steps, config)`: LSTM neural network forecasting
- `ensemble_predictions(steps, config)`: Ensemble forecasting

#### IRImageProcessing

Main class for IR image processing:

- `detect_hot_spots(threshold_percentile, min_area)`: Detect thermal anomalies
- `seasonal_decomposition(data, period)`: Decompose into trend, seasonal, residual
- `trend_analysis(data, method)`: Analyze temperature trends
- `residual_analysis(data, period)`: Statistical analysis of residuals

## Configuration

All forecasting methods support comprehensive configuration via Pydantic models:

```python
from pv_circularity.utils.validators import (
    ARIMAConfig,
    ProphetConfig,
    LSTMConfig,
    EnsembleConfig
)

# ARIMA configuration
arima_config = ARIMAConfig(
    p=2, d=1, q=2,
    seasonal_order=(1, 0, 1, 7),
    trend='c'
)

# Prophet configuration
prophet_config = ProphetConfig(
    growth='linear',
    seasonality_mode='additive',
    yearly_seasonality=True
)

# LSTM configuration
lstm_config = LSTMConfig(
    n_layers=2,
    hidden_units=64,
    epochs=100,
    lookback_window=14
)
```

## Dependencies

Core dependencies:
- numpy, pandas, scipy, scikit-learn
- statsmodels (ARIMA)
- prophet (Facebook Prophet)
- tensorflow/keras (LSTM)
- opencv-python, Pillow, scikit-image (Image processing)
- pydantic (Data validation)
- matplotlib, seaborn, plotly (Visualization)

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review examples in `examples/`

## Acknowledgments

- statsmodels team for ARIMA implementation
- Facebook for Prophet forecasting library
- TensorFlow team for deep learning framework
- OpenCV community for image processing tools
