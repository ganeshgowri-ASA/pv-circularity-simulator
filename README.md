# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Integration Layer & API Connectors ✅

Production-ready integration framework for connecting to external APIs with:

- **IntegrationManager**: Central orchestration of multiple API integrations
- **API Connectors**: REST API client with full HTTP method support (GET, POST, PUT, PATCH, DELETE)
- **Authentication**: Multiple schemes (API Key, Bearer Token, OAuth2, Basic Auth)
- **Rate Limiting**: Token bucket algorithm for request throttling
- **Retry Logic**: Exponential backoff with jitter for handling transient failures
- **Async Support**: Full async/await support for concurrent requests
- **Monitoring**: Built-in metrics and health checks
- **Type Safety**: Pydantic models for configuration and validation

See [INTEGRATION_LAYER.md](INTEGRATION_LAYER.md) for detailed documentation.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
    AuthConfig,
    AuthType,
)

# Create and configure integration
manager = IntegrationManager()
config = IntegrationConfig(
    name="weather-api",
    base_url="https://api.weather.example.com",
    auth=AuthConfig(
        auth_type=AuthType.API_KEY,
        api_key="your-api-key"
    )
)

# Register and use
manager.register_integration(config)
connector = manager.api_connectors("weather-api")
response = connector.get("/forecast")

if response.success:
    print(response.json_data)
```

## Documentation

- [Integration Layer Documentation](INTEGRATION_LAYER.md) - Comprehensive guide to the integration system
- [Examples](examples/) - Working examples and configuration samples

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity_simulator/
│       └── integration/          # Integration Layer & API Connectors
│           ├── manager.py        # IntegrationManager
│           ├── connectors.py     # API Connectors
│           ├── auth.py           # Authentication Handlers
│           ├── rate_limiter.py   # Rate Limiting
│           ├── retry.py          # Retry Logic
│           └── models.py         # Pydantic Models
├── tests/
│   └── integration/              # Comprehensive test suite
├── examples/                     # Usage examples
├── requirements.txt              # Dependencies
└── pyproject.toml               # Project configuration
```

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=pv_circularity_simulator --cov-report=html

# Specific module
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/

# Type checking
mypy src/
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and code is properly formatted before submitting.
