# Integration Layer & API Connectors

Production-ready integration framework for connecting to external APIs with authentication, rate limiting, and robust error handling.

## Features

### Core Components

1. **IntegrationManager** - Central orchestration of API integrations
   - Manage multiple named integrations
   - Centralized configuration management
   - Metrics collection and monitoring
   - Health check capabilities

2. **API Connectors** - REST API client with full HTTP method support
   - Synchronous and asynchronous request handling
   - Support for GET, POST, PUT, PATCH, DELETE methods
   - Automatic retry with exponential backoff
   - Rate limiting integration
   - Request/response validation with Pydantic

3. **Authentication Handlers** - Multiple authentication schemes
   - API Key authentication
   - Bearer Token authentication
   - OAuth2 with automatic token refresh
   - Basic authentication
   - No authentication
   - Extensible architecture for custom auth

4. **Rate Limiting** - Token bucket algorithm implementation
   - Thread-safe and async-safe
   - Configurable request rates
   - Burst handling
   - Real-time token availability tracking

5. **Error & Retry Logic** - Intelligent error handling
   - Exponential backoff with configurable base
   - Optional jitter to prevent thundering herd
   - Configurable retry conditions (status codes)
   - Maximum retry attempts and delays
   - Detailed error reporting

## Architecture

```
IntegrationManager
├── IntegrationConfig (Pydantic model)
│   ├── AuthConfig
│   ├── RateLimitConfig
│   └── RetryConfig
├── APIConnector
│   ├── AuthenticationHandler
│   ├── RateLimiter
│   └── RetryHandler
└── IntegrationMetrics
```

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

# Create manager
manager = IntegrationManager()

# Configure integration
config = IntegrationConfig(
    name="my-api",
    base_url="https://api.example.com",
    auth=AuthConfig(
        auth_type=AuthType.API_KEY,
        api_key="your-secret-key"
    )
)

# Register integration
manager.register_integration(config)

# Make requests
connector = manager.api_connectors("my-api")
response = connector.get("/endpoint")

if response.success:
    print(response.json_data)
```

## Configuration

### Integration Configuration

```python
IntegrationConfig(
    name="my-api",                           # Unique identifier
    base_url="https://api.example.com",      # API base URL
    auth=AuthConfig(...),                    # Authentication config
    rate_limit=RateLimitConfig(...),         # Rate limiting config
    retry=RetryConfig(...),                  # Retry config
    timeout=30.0,                            # Request timeout (seconds)
    verify_ssl=True,                         # SSL verification
    default_headers={"User-Agent": "..."}    # Default headers
)
```

### Authentication Configuration

#### API Key

```python
AuthConfig(
    auth_type=AuthType.API_KEY,
    api_key="your-api-key",
    api_key_header="X-API-Key"  # Custom header name
)
```

#### Bearer Token

```python
AuthConfig(
    auth_type=AuthType.BEARER_TOKEN,
    token="your-bearer-token"
)
```

#### OAuth2

```python
AuthConfig(
    auth_type=AuthType.OAUTH2,
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://auth.example.com/token"
)
```

#### Basic Authentication

```python
AuthConfig(
    auth_type=AuthType.BASIC,
    username="your-username",
    password="your-password"
)
```

### Rate Limiting Configuration

```python
RateLimitConfig(
    enabled=True,
    max_requests=100,        # Maximum requests
    time_window=60.0,        # Time window in seconds
    burst_size=120           # Maximum burst size (optional)
)
```

### Retry Configuration

```python
RetryConfig(
    enabled=True,
    max_retries=3,           # Maximum retry attempts
    initial_delay=1.0,       # Initial delay in seconds
    max_delay=60.0,          # Maximum delay in seconds
    exponential_base=2.0,    # Exponential backoff base
    jitter=True,             # Add random jitter
    retry_on_status_codes=[408, 429, 500, 502, 503, 504]
)
```

## API Reference

### IntegrationManager

#### Methods

- `register_integration(config)` - Register a new integration
- `unregister_integration(name)` - Remove an integration
- `update_integration(config)` - Update integration configuration
- `get_integration(name)` - Get integration configuration
- `list_integrations()` - List all registered integrations
- `api_connectors(name)` - Get API connector for integration
- `authentication_handlers(name)` - Get authentication handler
- `rate_limiting(name)` - Get rate limiter
- `error_retry_logic(name)` - Get retry handler
- `get_metrics(name)` - Get integration metrics
- `reset_metrics(name)` - Reset integration metrics
- `health_check(name, endpoint)` - Perform health check
- `get_status()` - Get status of all integrations

### APIConnector

#### Methods

- `request(request)` - Make synchronous request
- `request_async(request)` - Make asynchronous request
- `get(endpoint, params)` - GET request
- `get_async(endpoint, params)` - Async GET request
- `post(endpoint, json, data)` - POST request
- `post_async(endpoint, json, data)` - Async POST request
- `put(endpoint, json, data)` - PUT request
- `put_async(endpoint, json, data)` - Async PUT request
- `patch(endpoint, json, data)` - PATCH request
- `patch_async(endpoint, json, data)` - Async PATCH request
- `delete(endpoint)` - DELETE request
- `delete_async(endpoint)` - Async DELETE request

### RateLimiter

#### Methods

- `acquire(tokens, blocking)` - Acquire tokens (sync)
- `acquire_async(tokens, blocking)` - Acquire tokens (async)
- `get_available_tokens()` - Get current token count
- `reset()` - Reset to full capacity
- `update_config(config)` - Update configuration
- `get_stats()` - Get statistics

### RetryHandler

#### Methods

- `execute_with_retry(func, *args, **kwargs)` - Execute with retry (sync)
- `execute_with_retry_async(func, *args, **kwargs)` - Execute with retry (async)
- `should_retry(attempt, status_code, exception)` - Check if should retry
- `reset()` - Reset handler state
- `get_stats()` - Get statistics

## Monitoring & Metrics

### Integration Metrics

```python
metrics = manager.get_metrics("my-api")

print(f"Total Requests: {metrics.total_requests}")
print(f"Successful: {metrics.successful_requests}")
print(f"Failed: {metrics.failed_requests}")
print(f"Avg Response Time: {metrics.average_response_time}s")
print(f"Retry Attempts: {metrics.retry_attempts}")
```

### Overall Status

```python
status = manager.get_status()

for name, info in status['integrations'].items():
    print(f"{name}:")
    print(f"  Success Rate: {info['metrics']['success_rate']:.2%}")
    print(f"  Avg Response Time: {info['metrics']['average_response_time']:.3f}s")
```

## Advanced Usage

### Async Requests

```python
import asyncio

async def fetch_data():
    connector = manager.api_connectors("my-api")

    # Make multiple concurrent requests
    tasks = [
        connector.get_async(f"/items/{i}")
        for i in range(10)
    ]

    responses = await asyncio.gather(*tasks)
    return responses

responses = asyncio.run(fetch_data())
```

### Custom Request Configuration

```python
from pv_circularity_simulator.integration import APIRequest, HTTPMethod

request = APIRequest(
    method=HTTPMethod.POST,
    endpoint="/users",
    json={"name": "John", "email": "john@example.com"},
    headers={"X-Custom-Header": "value"},
    timeout=60.0  # Override default timeout
)

response = connector.request(request)
```

### Context Managers

```python
# Rate limiter as context manager
rate_limiter = manager.rate_limiting("my-api")

with rate_limiter:
    # Token automatically acquired
    response = connector.get("/endpoint")
```

## Testing

Run tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=pv_circularity_simulator --cov-report=html

# Specific test file
pytest tests/integration/test_models.py

# Async tests only
pytest tests/integration/test_rate_limiter.py -k async
```

## Production Considerations

### Security

1. **Never hardcode secrets** - Use environment variables
   ```python
   import os
   api_key = os.getenv("API_KEY")
   ```

2. **Always use HTTPS** in production
   ```python
   base_url="https://api.example.com"  # Not http://
   ```

3. **Enable SSL verification**
   ```python
   verify_ssl=True  # Default, never disable in production
   ```

### Performance

1. **Use async for concurrent requests**
   ```python
   responses = await asyncio.gather(*tasks)
   ```

2. **Configure appropriate timeouts**
   ```python
   timeout=30.0  # Based on your API's response time
   ```

3. **Set rate limits to match API quotas**
   ```python
   max_requests=100,
   time_window=60.0
   ```

### Reliability

1. **Enable retry logic**
   ```python
   retry=RetryConfig(
       enabled=True,
       max_retries=3
   )
   ```

2. **Monitor metrics**
   ```python
   metrics = manager.get_metrics("my-api")
   if metrics.failed_requests / metrics.total_requests > 0.1:
       # Alert: High failure rate
   ```

3. **Implement health checks**
   ```python
   if not manager.health_check("my-api"):
       # Handle unhealthy integration
   ```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pv_circularity_simulator.integration")
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Comprehensive usage examples
- `config_example.json` - Sample configuration file
- `README.md` - Detailed examples documentation

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code is formatted: `black src/ tests/`
3. Type checking passes: `mypy src/`
4. Linting passes: `ruff check src/`

## Support

For issues, questions, or contributions, please refer to the project repository.
