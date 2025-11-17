# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → **Performance monitoring** → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Real-time Performance Monitoring System (BATCH5-B07-S01)

#### RealTimeMonitor
- **Live Data Streaming**: Real-time data collection and streaming with WebSocket support
- **SCADA Integration**: Seamless integration with industrial SCADA systems
- **Inverter Data Parsing**: Comprehensive inverter data parsing and interpretation
- **Multi-level Monitoring**:
  - **String-level monitoring**: Track individual PV strings for anomaly detection
  - **Module-level monitoring**: Detailed module-level performance tracking

#### PerformanceMetrics
- **Instantaneous Performance Ratio (PR)**: Real-time PR calculation with temperature correction
- **Capacity Factor (CF)**: System utilization metrics
- **Specific Yield**: Energy production per installed capacity (kWh/kWp)
- **Availability Tracking**: System uptime and availability monitoring
- **Grid Export Monitoring**: Track energy export and self-consumption

#### AlertEngine
- **Underperformance Detection**: Automatic detection of below-threshold performance
- **Equipment Fault Alerts**: Real-time equipment failure detection
- **Grid Outage Detection**: Monitor grid parameters and detect outages
- **Communication Loss Detection**: Alert on device communication failures
- **Customizable Thresholds**: Configure alert thresholds per site/device

### Data Logger Integration & SCADA Systems (BATCH5-B07-S02)

#### DataLoggerIntegrator
Unified data collection from multiple industrial protocols:
- **modbus_tcp()**: Modbus TCP/RTU protocol support
- **sunspec_protocol()**: SunSpec-compliant inverter communication
- **proprietary_protocols()**: Support for major manufacturers:
  - SMA (Speedwire/Webconnect)
  - Fronius (Solar API)
  - Huawei (Modbus-based)
  - Sungrow (Modbus-based)
- **csv_file_import()**: Import historical data from CSV files

#### SCADAConnector
Industrial automation protocol support:
- **opc_ua()**: OPC Unified Architecture client
- **bacnet()**: BACnet/IP for building automation
- **iec61850()**: IEC 61850 for power utility automation
- **MQTT**: Pub/sub messaging for IoT devices

#### DataAggregator
Advanced data processing capabilities:
- **multi_site_aggregation()**: Aggregate data across multiple sites
- **data_normalization()**: Standardize units and formats
- **timestamp_alignment()**: Align data to regular time intervals
- **Statistical aggregation**: Support for sum, mean, max, min operations

## 🛠️ Technology Stack

- **Async Framework**: asyncio, aiohttp for high-performance async operations
- **Industrial Protocols**: pymodbus, asyncua, paho-mqtt for SCADA communication
- **Data Validation**: Pydantic v2 for robust data models and validation
- **Time-Series Database**: TimescaleDB/InfluxDB support for efficient data storage
- **Data Processing**: pandas, numpy, polars for high-speed data manipulation
- **Logging**: structlog for structured, context-aware logging
- **API**: FastAPI for RESTful API and WebSocket support
- **Monitoring**: Prometheus metrics integration

## 📦 Installation

### Requirements
- Python 3.10 or higher
- PostgreSQL with TimescaleDB extension (optional)
- InfluxDB 2.x (optional)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```bash
# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pv_circularity
DB_USER=pv_user
DB_PASSWORD=your_password

# MQTT broker
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883

# Monitoring settings
MONITORING_INTERVAL_SECONDS=5
PERFORMANCE_RATIO_THRESHOLD=0.75
```

3. Configure your devices in `config/scada_devices.yaml`

## 🎯 Quick Start

### Basic Usage

```python
import asyncio
from pv_circularity.monitoring import (
    RealTimeMonitor,
    PerformanceMetrics,
    AlertEngine,
    DataLoggerIntegrator,
)
from pv_circularity.models.scada import SCADADevice, ModbusConfig

# Define devices
devices = [
    SCADADevice(
        device_id="INV001",
        name="Main Inverter",
        device_type="inverter",
        protocol_type="modbus_tcp",
        modbus_config=ModbusConfig(
            host="192.168.1.100",
            port=502,
            slave_id=1,
        ),
    )
]

async def main():
    # Initialize real-time monitor
    monitor = RealTimeMonitor(devices, update_interval=5)
    await monitor.start_monitoring()

    # Start performance metrics calculation
    metrics = PerformanceMetrics()
    pr = await metrics.instantaneous_pr(
        actual_power=85.5,
        rated_power=100.0,
        irradiance=850,
    )

    # Enable alerts
    alerts = AlertEngine(underperformance_threshold=15.0)
    await alerts.start()

    # Stream live data
    async for data_batch in monitor.live_data_stream():
        print(f"Received {len(data_batch)} data points")

asyncio.run(main())
```

### Example: Multi-Protocol Data Collection

```python
from pv_circularity.monitoring import DataLoggerIntegrator

# Initialize with multiple protocol types
integrator = DataLoggerIntegrator(devices)
await integrator.initialize()

# Collect data from all devices
data = await integrator.collect_all_data()

for device_id, data_points in data.items():
    print(f"{device_id}: {len(data_points)} data points")
```

### Example: CSV Import

```python
from pv_circularity.monitoring.data_logger import CSVFileImporter

importer = CSVFileImporter(
    file_path="data/inverter_data.csv",
    device_id="INV001",
    timestamp_column="timestamp",
    value_columns={
        "ac_power": "AC Power (kW)",
        "dc_voltage": "DC Voltage (V)",
    },
)

data_points = await importer.import_data()
```

## 📚 Documentation

### Key Components

#### 1. RealTimeMonitor
```python
monitor = RealTimeMonitor(
    devices=device_list,
    update_interval=5,  # seconds
    buffer_size=1000
)
await monitor.start_monitoring()
```

#### 2. PerformanceMetrics
```python
metrics = PerformanceMetrics()

# Calculate various metrics
pr = await metrics.instantaneous_pr(actual_power, rated_power, irradiance)
cf = await metrics.capacity_factor(actual_energy, rated_power, period_hours)
sy = await metrics.specific_yield(actual_energy, rated_power)
avail = await metrics.availability_tracking(total_time, downtime)
```

#### 3. AlertEngine
```python
engine = AlertEngine(underperformance_threshold=15.0)

# Subscribe to alerts
def on_alert(alert):
    print(f"Alert: {alert.message}")

engine.subscribe_alerts(on_alert)
await engine.start()
```

#### 4. DataAggregator
```python
aggregator = DataAggregator()

# Multi-site aggregation
aggregated = await aggregator.multi_site_aggregation(site_data, method="sum")

# Timestamp alignment
aligned = await aggregator.timestamp_alignment(data_points, interval_seconds=60)

# Data normalization
normalized = await aggregator.data_normalization(data_points)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_circularity --cov-report=html

# Run specific test modules
pytest tests/unit/test_monitoring/
```

## 🏗️ Project Structure

```
pv-circularity-simulator/
├── src/pv_circularity/
│   ├── core/                    # Core utilities
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── logging_config.py   # Logging setup
│   │   └── utils.py            # Common utilities
│   ├── models/                  # Pydantic data models
│   │   ├── monitoring.py       # Monitoring data models
│   │   └── scada.py            # SCADA device models
│   ├── monitoring/              # Monitoring system
│   │   ├── data_logger/        # Data logging
│   │   │   ├── integrator.py  # Main integrator
│   │   │   └── csv_importer.py # CSV import
│   │   ├── scada/              # SCADA integration
│   │   │   ├── protocols/      # Protocol clients
│   │   │   ├── connector.py    # SCADA connector
│   │   │   └── aggregator.py   # Data aggregator
│   │   └── real_time/          # Real-time monitoring
│   │       ├── monitor.py      # Real-time monitor
│   │       ├── performance.py  # Performance metrics
│   │       └── alerts.py       # Alert engine
├── config/                      # Configuration files
│   ├── monitoring.yaml
│   └── scada_devices.yaml
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
└── docs/                        # Documentation

```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern async Python patterns for high performance
- Implements industry-standard protocols for maximum compatibility
- Designed for production-ready deployment in industrial environments

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.
