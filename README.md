# PV Circularity Simulator

End-to-end photovoltaic lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🌟 Features

### Real-Time Performance Monitoring System

The PV Circularity Simulator now includes a production-ready, real-time performance monitoring system with the following capabilities:

#### Core Components

1. **RealTimeMonitor** - Live data streaming and collection
   - `live_data_stream()` - Async generator for real-time data updates
   - `scada_integration()` - SCADA system integration via MQTT/Modbus
   - `inverter_data_parsing()` - Inverter telemetry parsing and validation
   - `string_level_monitoring()` - String-level performance tracking
   - `module_level_monitoring()` - Module-level monitoring with hot-spot detection

2. **PerformanceMetrics** - KPI calculation engine
   - `instantaneous_pr()` - Real-time Performance Ratio calculation
   - `capacity_factor()` - System capacity factor analysis
   - `specific_yield()` - Energy yield per installed capacity
   - `availability_tracking()` - System uptime and availability metrics
   - `grid_export_monitoring()` - Grid connection and export tracking

3. **AlertEngine** - Intelligent fault detection
   - `underperformance_detection()` - PR-based underperformance alerts
   - `equipment_fault_alerts()` - Equipment health monitoring
   - `grid_outage_detection()` - Grid anomaly and outage detection

#### Technical Stack

- **Protocols**: MQTT, Modbus TCP/RTU for device communication
- **Time-Series Database**: InfluxDB for high-frequency metrics, TimescaleDB for historical data
- **Real-Time Streaming**: WebSocket for live dashboard updates
- **Caching**: Redis for performance optimization
- **Visualization**: Streamlit dashboard with auto-refresh
- **Data Validation**: Pydantic models with full type safety
- **Testing**: Comprehensive pytest suite with async support

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for infrastructure)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings
```

### Starting Infrastructure Services

```bash
# Start all infrastructure services (InfluxDB, TimescaleDB, Redis, MQTT, Grafana, Prometheus)
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Running the Monitoring System

```bash
# Start the Streamlit dashboard
streamlit run streamlit_app/dashboard.py

# The dashboard will be available at http://localhost:8501
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_performance_metrics.py -v
```

## 📊 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PV Monitoring System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Inverters  │        │    SCADA     │                  │
│  │   (Modbus)   │        │    (MQTT)    │                  │
│  └──────┬───────┘        └──────┬───────┘                  │
│         │                       │                           │
│         └───────────┬───────────┘                           │
│                     │                                       │
│              ┌──────▼──────┐                                │
│              │RealTimeMonitor│                              │
│              └──────┬──────┘                                │
│                     │                                       │
│         ┌───────────┼───────────┐                           │
│         │           │           │                           │
│   ┌─────▼─────┐ ┌──▼──────┐ ┌─▼──────────┐                │
│   │Performance│ │  Alert  │ │  Database  │                │
│   │  Metrics  │ │ Engine  │ │  Storage   │                │
│   └─────┬─────┘ └──┬──────┘ └─┬──────────┘                │
│         │          │          │                             │
│         └──────────┼──────────┘                             │
│                    │                                        │
│              ┌─────▼──────┐                                 │
│              │ WebSocket  │                                 │
│              │  Handler   │                                 │
│              └─────┬──────┘                                 │
│                    │                                        │
│              ┌─────▼──────┐                                 │
│              │ Streamlit  │                                 │
│              │ Dashboard  │                                 │
│              └────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Collection**: Inverters/SCADA publish data via MQTT or Modbus
2. **Real-Time Processing**: RealTimeMonitor validates and streams data
3. **Metrics Calculation**: PerformanceMetrics computes KPIs
4. **Alert Detection**: AlertEngine monitors for anomalies
5. **Storage**: Data persisted to InfluxDB/TimescaleDB
6. **Streaming**: WebSocket broadcasts updates to clients
7. **Visualization**: Streamlit dashboard displays real-time metrics

## 🔧 Configuration

### Environment Variables

All configuration is managed via environment variables. See `.env.example` for available options:

- **Database**: PostgreSQL/TimescaleDB connection settings
- **InfluxDB**: Time-series database configuration
- **MQTT**: Broker connection and topic settings
- **Modbus**: TCP/RTU protocol configuration
- **Monitoring**: Sampling intervals and alert thresholds
- **WebSocket**: Server settings for live updates
- **Streamlit**: Dashboard configuration

### Monitoring Configuration

Key monitoring parameters in `config/settings.py`:

```python
# Sampling intervals (seconds)
MONITOR_INVERTER_SAMPLING_INTERVAL=5
MONITOR_STRING_SAMPLING_INTERVAL=60
MONITOR_MODULE_SAMPLING_INTERVAL=300

# Performance thresholds
MONITOR_UNDERPERFORMANCE_THRESHOLD_PCT=80.0
MONITOR_TEMPERATURE_ALERT_THRESHOLD_C=85.0

# Grid parameters
MONITOR_GRID_FREQUENCY_MIN_HZ=49.5
MONITOR_GRID_FREQUENCY_MAX_HZ=50.5
```

## 📈 Performance Metrics

### Supported Metrics

1. **Performance Ratio (PR)**
   - Instantaneous and time-averaged PR
   - Temperature-corrected calculations
   - Industry-standard formulas

2. **Capacity Factor**
   - Actual vs. potential energy production
   - Daily, monthly, yearly aggregations

3. **Specific Yield**
   - kWh per kWp installed capacity
   - Period-based calculations

4. **Availability**
   - System uptime tracking
   - Planned vs. unplanned downtime
   - Component-level availability

5. **Grid Export**
   - Real-time power export
   - Grid health monitoring (frequency, voltage)
   - Power quality metrics

## 🚨 Alert System

### Alert Types

- **Underperformance**: PR below configurable threshold
- **Equipment Faults**: Inverter errors, high temperature, low efficiency
- **Grid Outages**: Frequency/voltage out of range, connection loss
- **Communication Errors**: Protocol failures, data staleness

### Alert Severity Levels

- **CRITICAL**: Immediate attention required
- **HIGH**: Significant impact on production
- **MEDIUM**: Performance degradation
- **LOW**: Minor issues
- **INFO**: Informational notifications

### Alert Features

- Configurable thresholds
- Cooldown periods to prevent spam
- Rate limiting
- Acknowledgment and resolution tracking
- Callback system for custom handlers

## 📡 API Usage

### Example: Streaming Live Data

```python
from src.monitoring.realtime.monitor import RealTimeMonitor
from config.settings import get_settings

settings = get_settings()
monitor = RealTimeMonitor(settings)

# Start monitoring
await monitor.start()

# Stream inverter data
async for inv_data in monitor.inverter_data_parsing():
    print(f"Inverter {inv_data.inverter_id}: {inv_data.ac_power} kW")
```

### Example: Calculating Metrics

```python
from src.monitoring.metrics.performance import PerformanceMetrics

metrics = PerformanceMetrics(settings)

# Calculate instantaneous PR
pr_data = await metrics.instantaneous_pr(
    actual_power=850.0,
    irradiance=950.0,
    module_temperature=45.0
)

print(f"Performance Ratio: {pr_data.instantaneous_pr:.2f}%")
```

### Example: Alert Monitoring

```python
from src.monitoring.alerts.engine import AlertEngine

alert_engine = AlertEngine(settings)

# Register alert callback
async def handle_alert(alert):
    print(f"Alert: {alert.message}")

alert_engine.register_callback(handle_alert)

# Detect underperformance
alert = await alert_engine.underperformance_detection(pr_data)
if alert:
    print(f"Underperformance detected: {alert.deviation_percentage:.1f}%")
```

## 🐳 Docker Services

The `docker-compose.yml` provides the following services:

- **TimescaleDB** (PostgreSQL with time-series extension) - Port 5432
- **InfluxDB 2.x** - Port 8086
- **Redis** - Port 6379
- **Eclipse Mosquitto (MQTT)** - Ports 1883 (MQTT), 9001 (WebSocket)
- **Grafana** - Port 3000
- **Prometheus** - Port 9090
- **PgAdmin** (optional) - Port 5050

### Optional Services

Enable with profiles:

```bash
# Enable Kafka streaming
docker-compose --profile kafka up -d

# Enable admin tools
docker-compose --profile tools up -d

# Enable production setup with Nginx
docker-compose --profile production up -d
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_performance_metrics.py
│   ├── test_alert_engine.py
│   └── test_realtime_monitor.py
├── integration/               # Integration tests
│   ├── test_mqtt_integration.py
│   └── test_database_integration.py
└── conftest.py               # Pytest fixtures
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_performance_metrics.py

# With coverage report
pytest --cov=src --cov-report=term-missing

# Verbose output
pytest -v

# Only async tests
pytest -k "asyncio"
```

## 📚 Documentation

### Code Documentation

All modules include comprehensive docstrings following Google style:

- Module-level descriptions
- Class documentation with attributes
- Method documentation with args, returns, raises
- Usage examples

### API Reference

Generate API documentation:

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Generate documentation
sphinx-build -b html docs/ docs/_build/
```

## 🛠️ Development

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with MyPy
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📋 Roadmap

- [x] Real-time monitoring system
- [x] MQTT/Modbus protocol support
- [x] Performance metrics calculation
- [x] Alert engine
- [x] WebSocket streaming
- [x] Streamlit dashboard
- [ ] Machine learning for anomaly detection
- [ ] Predictive maintenance
- [ ] Advanced degradation modeling
- [ ] Mobile app integration
- [ ] Multi-site management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TimescaleDB for time-series data management
- InfluxDB for high-frequency metrics storage
- Pydantic for data validation
- Streamlit for rapid dashboard development
- FastAPI for modern API framework

## 📞 Support

For issues, questions, or contributions:

- GitHub Issues: [Create an issue](https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues)
- Documentation: See inline code documentation
- Examples: Check `examples/` directory

---

**Note**: This is a production-ready monitoring system with full docstrings, type hints, comprehensive tests, and enterprise-grade architecture. All components follow best practices for async Python development and real-time data processing.
