# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

- **Cell Design Simulation**: Model photovoltaic cell performance and characteristics
- **Module Engineering**: Cell-to-module (CTM) analysis and loss calculations
- **System Planning**: Design and optimize PV system configurations
- **Performance Monitoring**: Real-time monitoring with anomaly detection
- **Circular Economy**: 3R (Reduce, Reuse, Recycle) lifecycle modeling
- **SCAPS Integration**: Advanced semiconductor device simulation
- **Reliability Testing**: IEC standard compliance and accelerated testing
- **Energy Forecasting**: AI-enhanced weather-based predictions
- **Production-Ready Configuration Management**: Comprehensive configuration system

## Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Configuration Management

The simulator includes a comprehensive, production-ready configuration management system with:

- **Multiple Formats**: YAML and JSON support
- **Environment Variables**: Override any configuration via environment variables
- **Validation**: Schema-based validation with type checking and constraints
- **Multi-Environment**: Separate configs for development, staging, and production
- **Thread-Safe**: Safe for concurrent access

### Quick Start

```python
from pv_circularity_simulator.config import ConfigurationManager

# Load configuration
config = ConfigurationManager()
config.load_configuration('config/app.yaml')

# Get values
db_host = config.get('database.host')
api_port = config.get('api.port', default=8080)

# Set values
config.set('api.workers', 8)

# Save configuration
config.save_configuration()
```

### Environment Variables

Override configuration using environment variables with the `APP_` prefix:

```bash
export APP_DATABASE_HOST=prod-db.example.com
export APP_DATABASE_PORT=5432
export APP_API_DEBUG=false
```

See [config/README.md](config/README.md) for comprehensive documentation.

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity_simulator/
│       ├── config/              # Configuration management system
│       │   ├── configuration_manager.py
│       │   ├── validators.py
│       │   └── exceptions.py
│       └── __init__.py
├── config/
│   ├── examples/                # Example configuration files
│   │   ├── app_config.yaml
│   │   ├── app_config.json
│   │   └── config_schema.yaml
│   └── README.md                # Configuration documentation
├── tests/                       # Test suite
│   └── test_configuration_manager.py
├── examples/                    # Usage examples
│   └── configuration_usage.py
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup
```

## Usage Examples

### Basic Configuration

```python
from pv_circularity_simulator.config import settings_loader

# Load with environment-specific settings
config = settings_loader(
    'config/app.yaml',
    environment='production',
    apply_env_overrides=True
)
```

### Validation

```python
from pv_circularity_simulator.config import ConfigurationManager

schema = {
    "database": {
        "port": {"type": int, "min": 1, "max": 65535},
    }
}

config = ConfigurationManager(schema=schema)
config.load_configuration('config/app.yaml')
config.validate()  # Raises error if invalid
```

See [examples/configuration_usage.py](examples/configuration_usage.py) for more examples.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pv_circularity_simulator --cov-report=html
```

## Documentation

- [Configuration Management Guide](config/README.md)
- [API Reference](docs/api_reference.md) (coming soon)
- [User Guide](docs/user_guide.md) (coming soon)

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for photovoltaic lifecycle simulation and circular economy analysis
- Supports research in sustainable energy systems
- Integrates industry-standard simulation tools (SCAPS)
