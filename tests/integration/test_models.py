"""
Tests for Pydantic models in the integration layer.
"""

import pytest
from pydantic import ValidationError

from pv_circularity_simulator.integration.models import (
    AuthConfig,
    AuthType,
    HTTPMethod,
    IntegrationConfig,
    APIRequest,
    APIResponse,
    RateLimitConfig,
    RetryConfig,
    IntegrationMetrics,
)


class TestAuthConfig:
    """Tests for AuthConfig model."""

    def test_default_auth_config(self):
        """Test default AuthConfig creation."""
        config = AuthConfig()
        assert config.auth_type == AuthType.NONE
        assert config.api_key is None
        assert config.api_key_header == "X-API-Key"

    def test_api_key_auth_config(self):
        """Test API key authentication configuration."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_key="test-key-123"
        )
        assert config.auth_type == AuthType.API_KEY
        assert config.api_key.get_secret_value() == "test-key-123"

    def test_bearer_token_auth_config(self):
        """Test Bearer token authentication configuration."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            token="bearer-token-456"
        )
        assert config.auth_type == AuthType.BEARER_TOKEN
        assert config.token.get_secret_value() == "bearer-token-456"

    def test_oauth2_auth_config(self):
        """Test OAuth2 authentication configuration."""
        config = AuthConfig(
            auth_type=AuthType.OAUTH2,
            client_id="client-123",
            client_secret="secret-456",
            token_url="https://auth.example.com/token"
        )
        assert config.auth_type == AuthType.OAUTH2
        assert config.client_id == "client-123"
        assert config.client_secret.get_secret_value() == "secret-456"


class TestRateLimitConfig:
    """Tests for RateLimitConfig model."""

    def test_default_rate_limit_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.max_requests == 100
        assert config.time_window == 60.0
        assert config.burst_size == 100

    def test_custom_rate_limit_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            max_requests=50,
            time_window=30.0,
            burst_size=75
        )
        assert config.max_requests == 50
        assert config.time_window == 30.0
        assert config.burst_size == 75

    def test_invalid_rate_limit_config(self):
        """Test validation of invalid rate limit configuration."""
        with pytest.raises(ValidationError):
            RateLimitConfig(max_requests=0)

        with pytest.raises(ValidationError):
            RateLimitConfig(time_window=-1)


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_default_retry_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert 500 in config.retry_on_status_codes

    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            exponential_base=3.0,
            retry_on_status_codes=[503, 504]
        )
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.exponential_base == 3.0
        assert config.retry_on_status_codes == [503, 504]


class TestIntegrationConfig:
    """Tests for IntegrationConfig model."""

    def test_minimal_integration_config(self):
        """Test minimal integration configuration."""
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )
        assert config.name == "test-api"
        assert str(config.base_url) == "https://api.example.com/"
        assert config.timeout == 30.0
        assert config.verify_ssl is True

    def test_full_integration_config(self):
        """Test full integration configuration."""
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com",
            auth=AuthConfig(
                auth_type=AuthType.API_KEY,
                api_key="test-key"
            ),
            rate_limit=RateLimitConfig(max_requests=200),
            retry=RetryConfig(max_retries=5),
            timeout=60.0,
            verify_ssl=False,
            default_headers={"User-Agent": "TestClient"}
        )
        assert config.name == "test-api"
        assert config.timeout == 60.0
        assert config.verify_ssl is False
        assert config.default_headers["User-Agent"] == "TestClient"


class TestAPIRequest:
    """Tests for APIRequest model."""

    def test_minimal_request(self):
        """Test minimal API request."""
        request = APIRequest(endpoint="/users")
        assert request.method == HTTPMethod.GET
        assert request.endpoint == "/users"
        assert request.params is None

    def test_post_request_with_json(self):
        """Test POST request with JSON body."""
        request = APIRequest(
            method=HTTPMethod.POST,
            endpoint="/users",
            json={"name": "John", "email": "john@example.com"}
        )
        assert request.method == HTTPMethod.POST
        assert request.json["name"] == "John"

    def test_request_with_params(self):
        """Test request with query parameters."""
        request = APIRequest(
            endpoint="/users",
            params={"page": 1, "limit": 10}
        )
        assert request.params["page"] == 1
        assert request.params["limit"] == 10


class TestAPIResponse:
    """Tests for APIResponse model."""

    def test_successful_response(self):
        """Test successful API response."""
        response = APIResponse(
            status_code=200,
            success=True,
            elapsed_time=0.5,
            json_data={"result": "success"}
        )
        assert response.status_code == 200
        assert response.success is True
        assert response.elapsed_time == 0.5
        assert response.json_data["result"] == "success"

    def test_error_response(self):
        """Test error API response."""
        response = APIResponse(
            status_code=500,
            success=False,
            elapsed_time=1.2,
            error_message="Internal Server Error"
        )
        assert response.status_code == 500
        assert response.success is False
        assert response.error_message == "Internal Server Error"


class TestIntegrationMetrics:
    """Tests for IntegrationMetrics model."""

    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = IntegrationMetrics(integration_name="test-api")
        assert metrics.integration_name == "test-api"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.average_response_time == 0.0

    def test_metrics_with_data(self):
        """Test metrics with data."""
        metrics = IntegrationMetrics(
            integration_name="test-api",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_response_time=0.25
        )
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 5
        assert metrics.average_response_time == 0.25
