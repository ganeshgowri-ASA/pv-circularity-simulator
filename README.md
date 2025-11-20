# PV Circularity Simulator

End-to-end PV lifecycle simulation platform with comprehensive weather data management, TMY (Typical Meteorological Year) database, and circular economy modeling.

## 🌟 Features

### TMY Weather Database & Comprehensive Weather System (Batch 5-B06-S01)

A production-ready TMY data management and weather database system with global coverage, historical data analysis, and satellite integration for accurate energy yield predictions.

**Core Components:**

1. **TMYDataManager** - Load, parse, and validate TMY data
   - Support for TMY2, TMY3, EPW, CSV, and custom formats
   - Data interpolation for missing values
   - Comprehensive quality validation
   - Format conversion utilities

2. **WeatherDatabaseBuilder** - Integrate multiple weather data sources
   - NREL NSRDB integration (National Solar Radiation Database)
   - PVGIS integration (European Commission JRC)
   - Meteonorm parser support
   - Local weather station import
   - Satellite data integration (NetCDF, HDF5)

3. **HistoricalWeatherAnalyzer** - Multi-year statistical analysis
   - Multi-year statistics (mean, P90, P50, P10)
   - Extreme weather event detection
   - Climate change trend analysis
   - Seasonal variability assessment
   - Inter-annual variability metrics

4. **GlobalWeatherCoverage** - Worldwide location database
   - Global coordinate-to-weather mapping
   - Nearest station finder with distance calculations
   - Geographic interpolation (inverse distance weighting)
   - Elevation corrections for weather data
   - Location search by name or region

5. **TMYGenerator** - Synthetic TMY creation
   - Sandia TMY generation method
   - Representative month selection
   - Monthly data stitching with smoothing
   - Comprehensive sanity checks
   - Export to multiple formats (TMY3, EPW, CSV, JSON)

6. **WeatherDataUI** - Interactive Streamlit interface
   - Interactive location selector with map
   - Real-time TMY data visualization
   - Historical trends charts (irradiance, temperature, wind)
   - Data quality indicators and completeness metrics
   - Download TMY files in multiple formats

### Additional Planned Modules

- **Cell Design**: Advanced cell modeling and optimization
- **Module Engineering**: CTM loss analysis and module design
- **System Planning**: PV system configuration and sizing
- **Performance Monitoring**: Real-time monitoring and forecasting
- **Circular Economy**: 3R modeling (Reduce, Reuse, Recycle)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
NSRDB_API_KEY=your_nrel_api_key_here
PVGIS_API_KEY=optional_pvgis_key
```

**Note**: You can use `DEMO_KEY` for NSRDB for testing (rate limited).

### Running the Application

#### Streamlit Web Interface

```bash
streamlit run ui/app.py
```

Then open your browser to `http://localhost:8501`

#### Python API Usage

```python
from pv_simulator.services.tmy_manager import TMYDataManager
from pv_simulator.services.weather_database import WeatherDatabaseBuilder

# Initialize services
tmy_manager = TMYDataManager()
weather_db = WeatherDatabaseBuilder()

# Fetch TMY data from NREL NSRDB
tmy_data = weather_db.nrel_nsrdb_integration(
    latitude=39.7392,
    longitude=-104.9903
)

# Analyze data
print(f"Annual GHI: {tmy_data.get_annual_irradiation():.1f} kWh/m²")
print(f"Average Temperature: {tmy_data.get_average_temperature():.1f}°C")
print(f"Data Quality: {tmy_data.data_quality.value}")
```

## 📊 API Integrations

### NREL NSRDB (National Solar Radiation Database)

High-quality solar radiation data for the Americas:

```python
from pv_simulator.api.nsrdb_client import NSRDBClient

client = NSRDBClient(api_key="your_key")
tmy_data = client.get_tmy_data(latitude=39.7392, longitude=-104.9903)
```

### PVGIS (Photovoltaic Geographical Information System)

Solar radiation data for Europe, Africa, and Asia:

```python
from pv_simulator.api.pvgis_client import PVGISClient

client = PVGISClient()
tmy_data = client.get_tmy_data(latitude=52.52, longitude=13.40)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/unit/test_weather_models.py
```

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/          # Core library
│   ├── models/                # Pydantic data models
│   ├── api/                   # API clients (NSRDB, PVGIS)
│   ├── services/              # Business logic
│   ├── config/                # Configuration management
│   └── utils/                 # Utility functions
├── ui/                        # Streamlit web interface
│   ├── app.py                 # Main application
│   └── components/            # UI components
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test fixtures
├── data/                      # Data storage
│   ├── tmy_cache/            # Cached TMY files
│   ├── weather/              # Weather data
│   └── locations/            # Location database
└── docs/                      # Documentation
```

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement with tests
3. Run test suite: `pytest`
4. Format code: `black .`
5. Submit pull request

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- [API Documentation](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Weather Integration Guide](docs/weather_integration.md)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NREL for the NSRDB database
- European Commission JRC for PVGIS
- The open-source Python community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Version**: 0.1.0
**Status**: Active Development
**Last Updated**: 2025-11-17
