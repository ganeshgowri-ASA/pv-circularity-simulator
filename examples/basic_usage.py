"""
Basic usage examples for the Integration Layer & API Connectors.

This script demonstrates how to use the IntegrationManager to connect
to external APIs with authentication, rate limiting, and retry logic.
"""

import asyncio
from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
    AuthConfig,
    AuthType,
    RateLimitConfig,
    RetryConfig,
    APIRequest,
    HTTPMethod,
)


def example_basic_api_call():
    """
    Example: Making a basic API call without authentication.
    """
    print("\n=== Example 1: Basic API Call ===\n")

    # Create IntegrationManager
    manager = IntegrationManager()

    # Configure integration
    config = IntegrationConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com",
        auth=AuthConfig(auth_type=AuthType.NONE),
        timeout=10.0
    )

    # Register integration
    manager.register_integration(config)

    # Get connector
    connector = manager.api_connectors("jsonplaceholder")

    # Make request using convenience method
    response = connector.get("/posts/1")

    print(f"Status: {response.status_code}")
    print(f"Success: {response.success}")
    print(f"Response Time: {response.elapsed_time:.3f}s")
    if response.json_data:
        print(f"Title: {response.json_data.get('title', 'N/A')}")


def example_api_key_authentication():
    """
    Example: Using API key authentication.
    """
    print("\n=== Example 2: API Key Authentication ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="weather-api",
        base_url="https://api.weather.example.com",
        auth=AuthConfig(
            auth_type=AuthType.API_KEY,
            api_key="your-api-key-here",
            api_key_header="X-API-Key"
        )
    )

    manager.register_integration(config)

    # Note: This will fail without a real API key and endpoint
    print("Configuration created with API key authentication")
    print(f"Auth type: {config.auth.auth_type}")


def example_rate_limiting():
    """
    Example: Using rate limiting.
    """
    print("\n=== Example 3: Rate Limiting ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="rate-limited-api",
        base_url="https://api.example.com",
        rate_limit=RateLimitConfig(
            enabled=True,
            max_requests=10,  # 10 requests
            time_window=60.0,  # per 60 seconds
            burst_size=15  # allow bursts up to 15
        )
    )

    manager.register_integration(config)

    # Get rate limiter stats
    rate_limiter = manager.rate_limiting("rate-limited-api")
    stats = rate_limiter.get_stats()

    print(f"Rate Limiter Enabled: {stats['enabled']}")
    print(f"Max Tokens: {stats['max_tokens']}")
    print(f"Available Tokens: {stats['available_tokens']}")
    print(f"Refill Rate: {stats['refill_rate']:.2f} tokens/second")


def example_retry_logic():
    """
    Example: Configuring retry logic with exponential backoff.
    """
    print("\n=== Example 4: Retry Logic ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="unreliable-api",
        base_url="https://api.unreliable.example.com",
        retry=RetryConfig(
            enabled=True,
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            retry_on_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )

    manager.register_integration(config)

    # Get retry handler stats
    retry_handler = manager.error_retry_logic("unreliable-api")
    stats = retry_handler.get_stats()

    print(f"Retry Enabled: {stats['enabled']}")
    print(f"Max Retries: {stats['max_retries']}")
    print(f"Initial Delay: {stats['initial_delay']}s")
    print(f"Exponential Base: {stats['exponential_base']}")
    print(f"Retry Status Codes: {stats['retry_status_codes']}")


def example_custom_request():
    """
    Example: Making a custom POST request with data.
    """
    print("\n=== Example 5: Custom POST Request ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com"
    )

    manager.register_integration(config)

    # Create custom request
    request = APIRequest(
        method=HTTPMethod.POST,
        endpoint="/posts",
        json={
            "title": "Test Post",
            "body": "This is a test post",
            "userId": 1
        }
    )

    # Make request
    connector = manager.api_connectors("jsonplaceholder")
    response = connector.request(request)

    print(f"Status: {response.status_code}")
    print(f"Success: {response.success}")
    if response.json_data:
        print(f"Created Post ID: {response.json_data.get('id', 'N/A')}")


async def example_async_requests():
    """
    Example: Making asynchronous API requests.
    """
    print("\n=== Example 6: Async Requests ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com"
    )

    manager.register_integration(config)
    connector = manager.api_connectors("jsonplaceholder")

    # Make multiple async requests concurrently
    tasks = [
        connector.get_async(f"/posts/{i}")
        for i in range(1, 6)
    ]

    responses = await asyncio.gather(*tasks)

    print(f"Fetched {len(responses)} posts concurrently")
    for i, response in enumerate(responses, 1):
        if response.success and response.json_data:
            print(f"  Post {i}: {response.json_data.get('title', 'N/A')[:50]}...")


def example_monitoring_metrics():
    """
    Example: Monitoring integration metrics.
    """
    print("\n=== Example 7: Monitoring Metrics ===\n")

    manager = IntegrationManager()

    config = IntegrationConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com"
    )

    manager.register_integration(config)
    connector = manager.api_connectors("jsonplaceholder")

    # Make several requests
    for i in range(1, 6):
        connector.get(f"/posts/{i}")

    # Get metrics
    metrics = manager.get_metrics("jsonplaceholder")

    print(f"Integration: {metrics.integration_name}")
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Successful: {metrics.successful_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Avg Response Time: {metrics.average_response_time:.3f}s")
    print(f"Retry Attempts: {metrics.retry_attempts}")


def example_multiple_integrations():
    """
    Example: Managing multiple API integrations.
    """
    print("\n=== Example 8: Multiple Integrations ===\n")

    manager = IntegrationManager()

    # Register multiple integrations
    configs = [
        IntegrationConfig(
            name="api-1",
            base_url="https://api1.example.com",
            auth=AuthConfig(auth_type=AuthType.API_KEY, api_key="key1")
        ),
        IntegrationConfig(
            name="api-2",
            base_url="https://api2.example.com",
            auth=AuthConfig(auth_type=AuthType.BEARER_TOKEN, token="token2")
        ),
        IntegrationConfig(
            name="api-3",
            base_url="https://api3.example.com",
            auth=AuthConfig(auth_type=AuthType.NONE)
        ),
    ]

    for config in configs:
        manager.register_integration(config)

    # List all integrations
    print(f"Registered Integrations: {manager.list_integrations()}")

    # Get overall status
    status = manager.get_status()
    print(f"\nTotal Integrations: {status['total_integrations']}")
    for name, info in status['integrations'].items():
        print(f"\n{name}:")
        print(f"  Base URL: {info['base_url']}")
        print(f"  Auth Type: {info['auth_type']}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Integration Layer & API Connectors - Usage Examples")
    print("=" * 60)

    # Synchronous examples
    example_basic_api_call()
    example_api_key_authentication()
    example_rate_limiting()
    example_retry_logic()
    example_custom_request()
    example_monitoring_metrics()
    example_multiple_integrations()

    # Asynchronous example
    print("\n" + "=" * 60)
    print("Running async examples...")
    print("=" * 60)
    asyncio.run(example_async_requests())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
