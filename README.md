# PV Circularity Simulator

End-to-end PV lifecycle simulation platform with comprehensive weather API integration for accurate energy forecasting and performance monitoring.

## Features

- **Cell Design** → **Module Engineering** → **System Planning** → **Performance Monitoring** → **Circularity (3R)**
- **Weather API Integration**: Real-time and historical weather data from multiple providers
- **CTM Loss Analysis**: Cell-to-module power loss analysis
- **SCAPS Integration**: Simulation tool for thin-film solar cells
- **Reliability Testing**: Long-term performance prediction
- **Energy Forecasting**: Accurate solar irradiance and weather-based predictions
- **Circular Economy Modeling**: Reduce, Reuse, Recycle analysis

## Weather API Integration (BATCH5-B06-S02)

This release includes comprehensive weather API integration with support for:

### Supported Weather Providers

1. **OpenWeatherMap** - Current weather, forecasts, and historical data
2. **Visual Crossing** - Comprehensive weather data with excellent historical coverage
3. **Meteomatics** - Professional-grade weather API with solar irradiance
4. **Tomorrow.io** - Advanced weather forecasting and nowcasting
5. **NREL PSM** - High-quality solar irradiance data for PV applications

### Core Components

#### WeatherAPIIntegrator
Unified interface for multiple weather providers with automatic fallback:
- `openweathermap_api()` - OpenWeatherMap integration
- `visualcrossing_api()` - Visual Crossing integration
- `meteomatics_api()` - Meteomatics integration
- `tomorrow_io_api()` - Tomorrow.io integration
- `nrel_psm_api()` - NREL Physical Solar Model integration

#### RealTimeWeatherFetcher
High-level interface for weather data operations:
- `current_conditions()` - Get current weather
- `forecast_data()` - Get weather forecasts
- `historical_backfill()` - Fetch historical data for gap filling
- `api_rate_limiting()` - Monitor API rate limits
- `cache_manager()` - Cache status and management

#### WeatherDataValidator
Data quality and preprocessing:
- `data_quality_checks()` - Comprehensive quality metrics
- `outlier_detection()` - Statistical outlier detection (IQR, Z-score)
- `gap_filling()` - Linear/forward/backward interpolation
- `unit_conversions()` - Temperature, wind, irradiance unit conversions
- `timestamp_synchronization()` - Timezone conversion and resampling

### Interactive Dashboard

Streamlit-based dashboard with:
- **Live Weather Display** - Real-time conditions with solar irradiance
- **Forecast Visualization** - Interactive charts for temperature, GHI, wind
- **API Configuration** - Provider selection and settings
- **Data Quality Metrics** - Cache status, provider availability, quality scores

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

Obtain API keys from:
- OpenWeatherMap: https://openweathermap.org/api
- Visual Crossing: https://www.visualcrossing.com/weather-api
- Meteomatics: https://www.meteomatics.com/
- Tomorrow.io: https://www.tomorrow.io/
- NREL: https://developer.nrel.gov/signup/

## Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Python API

#### Fetch Current Weather

```python
from pv_simulator.weather.fetcher import RealTimeWeatherFetcher

fetcher = RealTimeWeatherFetcher()

# Get current weather for a location
current = fetcher.current_conditions(
    latitude=40.7128,
    longitude=-74.0060
)

print(f"Temperature: {current.data.temperature}°C")
print(f"Solar GHI: {current.data.ghi} W/m²")
```

#### Fetch Weather Forecast

```python
from pv_simulator.weather.fetcher import RealTimeWeatherFetcher

fetcher = RealTimeWeatherFetcher()

# Get 7-day forecast
forecast = fetcher.forecast_data(
    latitude=40.7128,
    longitude=-74.0060,
    days=7
)

for point in forecast.forecast_data:
    print(f"{point.timestamp}: {point.temperature}°C, GHI: {point.ghi} W/m²")
```

#### Historical Data with Quality Checks

```python
from datetime import datetime, timedelta
from pv_simulator.weather.fetcher import RealTimeWeatherFetcher
from pv_simulator.weather.validator import WeatherDataValidator

fetcher = RealTimeWeatherFetcher()
validator = WeatherDataValidator()

# Fetch historical data
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)

historical = fetcher.historical_backfill(
    latitude=40.7128,
    longitude=-74.0060,
    start_date=start_date,
    end_date=end_date
)

# Validate data quality
metrics = validator.data_quality_checks(historical.historical_data)
print(f"Quality Score: {metrics.quality_score:.2f}")
print(f"Solar Data Completeness: {metrics.solar_completeness:.2%}")

# Fill gaps and detect outliers
cleaned_data, gaps_filled = validator.gap_filling(
    historical.historical_data,
    method="linear"
)
print(f"Filled {gaps_filled} data gaps")
```

#### Multi-Provider Comparison

```python
from pv_simulator.weather.fetcher import RealTimeWeatherFetcher

fetcher = RealTimeWeatherFetcher()

# Fetch from multiple providers
results = fetcher.fetch_multi_provider(
    latitude=40.7128,
    longitude=-74.0060,
    data_type="current"
)

for provider, result in results.items():
    if result["success"]:
        temp = result["data"]["data"]["temperature"]
        print(f"{provider}: {temp}°C")
```

## Configuration

Configuration is managed through environment variables (`.env` file):

```env
# Weather API Keys
OPENWEATHERMAP_API_KEY=your_key_here
VISUALCROSSING_API_KEY=your_key_here
METEOMATICS_USERNAME=your_username
METEOMATICS_PASSWORD=your_password
TOMORROW_IO_API_KEY=your_key_here
NREL_API_KEY=your_key_here

# Cache Configuration
CACHE_TYPE=sqlite  # Options: sqlite, redis, memory
CACHE_TTL=3600  # Cache TTL in seconds

# Rate Limiting (requests per minute)
OPENWEATHERMAP_RATE_LIMIT=60
VISUALCROSSING_RATE_LIMIT=1000
METEOMATICS_RATE_LIMIT=50
TOMORROW_IO_RATE_LIMIT=25
NREL_RATE_LIMIT=1000

# Data Quality
OUTLIER_DETECTION_ENABLED=true
GAP_FILLING_ENABLED=true
MAX_GAP_HOURS=3

# Units
TEMPERATURE_UNIT=celsius
IRRADIANCE_UNIT=w_per_m2
WIND_SPEED_UNIT=m_per_s
```

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src/pv_simulator --cov-report=html
```

## Architecture

```
pv-circularity-simulator/
├── src/pv_simulator/
│   ├── config.py              # Configuration management
│   ├── models/
│   │   └── weather.py         # Pydantic data models
│   ├── weather/
│   │   ├── cache.py           # Cache backends (SQLite, Redis, Memory)
│   │   ├── integrator.py      # WeatherAPIIntegrator
│   │   ├── fetcher.py         # RealTimeWeatherFetcher
│   │   ├── validator.py       # WeatherDataValidator
│   │   └── clients/
│   │       ├── base.py        # BaseWeatherClient with rate limiting
│   │       ├── openweathermap.py
│   │       ├── visualcrossing.py
│   │       ├── meteomatics.py
│   │       ├── tomorrow_io.py
│   │       └── nrel_psm.py
│   └── ui/
│       └── dashboard.py       # Streamlit dashboard
├── tests/                     # Comprehensive test suite
├── app.py                     # Main application entry point
└── pyproject.toml            # Project configuration
```

## Technology Stack

- **Python 3.9+** - Core language
- **Pydantic 2.0** - Data validation and settings
- **httpx** - Async HTTP client
- **pandas** - Time-series data manipulation
- **Streamlit** - Interactive dashboard
- **Plotly** - Interactive visualizations
- **SQLite/Redis** - Caching backends
- **pytest** - Testing framework

## Data Models

All weather data is validated using Pydantic models:

- **WeatherDataPoint**: Single weather observation with full meteorological parameters
- **CurrentWeather**: Current conditions with cache metadata
- **ForecastWeather**: Time-series forecast data
- **HistoricalWeather**: Historical weather records
- **DataQualityMetrics**: Comprehensive quality assessment
- **GeoLocation**: Geographic coordinates with validation

## Performance Features

- **Automatic Caching**: Configurable SQLite, Redis, or in-memory caching
- **Rate Limiting**: Token bucket algorithm prevents API quota exhaustion
- **Retry Logic**: Exponential backoff for failed requests
- **Provider Fallback**: Automatic failover to alternative providers
- **Async Support**: Non-blocking API requests for better performance

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] PV system simulation engine
- [ ] SCAPS integration
- [ ] CTM loss analysis tools
- [ ] Reliability prediction models
- [ ] Circularity assessment framework
- [ ] Machine learning for energy forecasting
- [ ] Integration with real-time monitoring systems

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Acknowledgments

Weather data provided by:
- OpenWeatherMap
- Visual Crossing
- Meteomatics
- Tomorrow.io
- NREL National Solar Radiation Database
