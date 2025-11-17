# Integration Layer Examples

This directory contains examples demonstrating how to use the Integration Layer & API Connectors.

## Files

- **basic_usage.py**: Comprehensive examples showing various features and use cases
- **config_example.json**: Sample configuration file for multiple integrations

## Running the Examples

### Prerequisites

Install the required dependencies:

```bash
pip install -r ../requirements.txt
```

### Basic Usage Examples

Run the basic usage examples:

```bash
cd examples
python basic_usage.py
```

This will demonstrate:
1. Basic API calls without authentication
2. API key authentication
3. Rate limiting configuration
4. Retry logic with exponential backoff
5. Custom POST requests
6. Asynchronous requests
7. Monitoring and metrics
8. Managing multiple integrations

## Example Use Cases

### 1. Simple GET Request

```python
from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
)

manager = IntegrationManager()
config = IntegrationConfig(
    name="my-api",
    base_url="https://api.example.com"
)
manager.register_integration(config)

connector = manager.api_connectors("my-api")
response = connector.get("/endpoint")

if response.success:
    print(response.json_data)
```

### 2. API Key Authentication

```python
from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
    AuthConfig,
    AuthType,
)

config = IntegrationConfig(
    name="secure-api",
    base_url="https://api.example.com",
    auth=AuthConfig(
        auth_type=AuthType.API_KEY,
        api_key="your-secret-key",
        api_key_header="X-API-Key"
    )
)
```

### 3. Rate Limiting

```python
from pv_circularity_simulator.integration import (
    IntegrationConfig,
    RateLimitConfig,
)

config = IntegrationConfig(
    name="limited-api",
    base_url="https://api.example.com",
    rate_limit=RateLimitConfig(
        enabled=True,
        max_requests=100,  # 100 requests
        time_window=60.0,  # per minute
    )
)
```

### 4. Retry Logic

```python
from pv_circularity_simulator.integration import (
    IntegrationConfig,
    RetryConfig,
)

config = IntegrationConfig(
    name="unreliable-api",
    base_url="https://api.example.com",
    retry=RetryConfig(
        enabled=True,
        max_retries=3,
        initial_delay=1.0,
        exponential_base=2.0,
        jitter=True
    )
)
```

### 5. Async Requests

```python
import asyncio

async def fetch_data():
    connector = manager.api_connectors("my-api")
    response = await connector.get_async("/endpoint")
    return response

# Run async
response = asyncio.run(fetch_data())
```

### 6. POST Request with JSON

```python
from pv_circularity_simulator.integration import APIRequest, HTTPMethod

request = APIRequest(
    method=HTTPMethod.POST,
    endpoint="/users",
    json={"name": "John", "email": "john@example.com"}
)

connector = manager.api_connectors("my-api")
response = connector.request(request)
```

### 7. Monitoring Metrics

```python
# Make some requests
connector.get("/endpoint1")
connector.get("/endpoint2")

# Get metrics
metrics = manager.get_metrics("my-api")
print(f"Total Requests: {metrics.total_requests}")
print(f"Success Rate: {metrics.successful_requests / metrics.total_requests}")
print(f"Avg Response Time: {metrics.average_response_time}s")
```

### 8. OAuth2 Authentication

```python
from pv_circularity_simulator.integration import (
    IntegrationConfig,
    AuthConfig,
    AuthType,
)

config = IntegrationConfig(
    name="oauth-api",
    base_url="https://api.example.com",
    auth=AuthConfig(
        auth_type=AuthType.OAUTH2,
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://auth.example.com/token"
    )
)
```

## Configuration File Format

See `config_example.json` for a complete configuration example. You can load this configuration programmatically:

```python
import json
from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
)

# Load configuration
with open('config_example.json', 'r') as f:
    config_data = json.load(f)

manager = IntegrationManager()

# Register each integration
for integration_data in config_data['integrations']:
    config = IntegrationConfig(**integration_data)
    manager.register_integration(config)
```

## Best Practices

1. **Always use HTTPS** in production environments
2. **Store secrets securely** - use environment variables or secret management systems
3. **Configure appropriate timeouts** based on your API's response time
4. **Set rate limits** to respect API quotas and prevent abuse
5. **Enable retry logic** for production systems to handle transient failures
6. **Monitor metrics** to track API performance and reliability
7. **Use async requests** when making multiple concurrent API calls
8. **Handle errors gracefully** - check `response.success` before accessing data

## Troubleshooting

### SSL Certificate Verification Errors

If you encounter SSL certificate errors in development:

```python
config = IntegrationConfig(
    name="my-api",
    base_url="https://api.example.com",
    verify_ssl=False  # Only for development!
)
```

**Warning**: Never disable SSL verification in production.

### Rate Limit Errors

If you're hitting rate limits:

1. Check your rate limit configuration
2. Monitor available tokens: `rate_limiter.get_available_tokens()`
3. Adjust `max_requests` and `time_window` settings
4. Use `burst_size` for temporary spikes

### Timeout Errors

If requests are timing out:

1. Increase the timeout: `config.timeout = 60.0`
2. Use per-request timeouts: `request.timeout = 120.0`
3. Enable retry logic for better resilience

## Additional Resources

- [Integration Layer Documentation](../docs/integration_layer.md) (if available)
- [API Reference](../docs/api_reference.md) (if available)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [httpx Documentation](https://www.python-httpx.org/)
