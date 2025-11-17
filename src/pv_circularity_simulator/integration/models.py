"""
Pydantic models for integration layer configuration and data validation.

This module defines all data models used throughout the integration layer,
ensuring type safety and validation for API requests, responses, and configurations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator, SecretStr


class AuthType(str, Enum):
    """Enumeration of supported authentication types."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    NONE = "none"


class HTTPMethod(str, Enum):
    """Enumeration of supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthConfig(BaseModel):
    """
    Configuration for API authentication.

    Attributes:
        auth_type: Type of authentication to use
        api_key: API key for API_KEY auth type
        api_key_header: Header name for API key (default: "X-API-Key")
        token: Bearer token or OAuth2 token
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        token_url: OAuth2 token endpoint URL
        username: Username for Basic auth
        password: Password for Basic auth
        additional_headers: Additional headers to include in requests
    """

    auth_type: AuthType = Field(
        default=AuthType.NONE,
        description="Type of authentication to use"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for authentication"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )
    token: Optional[SecretStr] = Field(
        default=None,
        description="Bearer token or OAuth2 token"
    )
    client_id: Optional[str] = Field(
        default=None,
        description="OAuth2 client ID"
    )
    client_secret: Optional[SecretStr] = Field(
        default=None,
        description="OAuth2 client secret"
    )
    token_url: Optional[HttpUrl] = Field(
        default=None,
        description="OAuth2 token endpoint URL"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for Basic authentication"
    )
    password: Optional[SecretStr] = Field(
        default=None,
        description="Password for Basic authentication"
    )
    additional_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to include in all requests"
    )

    @field_validator("auth_type")
    @classmethod
    def validate_auth_config(cls, v: AuthType, info) -> AuthType:
        """Validate that required fields are present for the selected auth type."""
        return v

    class Config:
        """Pydantic model configuration."""
        use_enum_values = False


class RateLimitConfig(BaseModel):
    """
    Configuration for API rate limiting using token bucket algorithm.

    Attributes:
        enabled: Whether rate limiting is enabled
        max_requests: Maximum number of requests allowed in the time window
        time_window: Time window in seconds for rate limiting
        burst_size: Maximum burst size (tokens that can accumulate)
    """

    enabled: bool = Field(
        default=True,
        description="Whether rate limiting is enabled"
    )
    max_requests: int = Field(
        default=100,
        gt=0,
        description="Maximum number of requests per time window"
    )
    time_window: float = Field(
        default=60.0,
        gt=0,
        description="Time window in seconds"
    )
    burst_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum burst size (defaults to max_requests if not set)"
    )

    @field_validator("burst_size")
    @classmethod
    def set_default_burst_size(cls, v: Optional[int], info) -> int:
        """Set burst_size to max_requests if not provided."""
        if v is None and "max_requests" in info.data:
            return info.data["max_requests"]
        return v or 100


class RetryConfig(BaseModel):
    """
    Configuration for retry logic with exponential backoff.

    Attributes:
        enabled: Whether retry logic is enabled
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to retry delays
        retry_on_status_codes: HTTP status codes that trigger retries
    """

    enabled: bool = Field(
        default=True,
        description="Whether retry logic is enabled"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts"
    )
    initial_delay: float = Field(
        default=1.0,
        gt=0,
        description="Initial delay in seconds before first retry"
    )
    max_delay: float = Field(
        default=60.0,
        gt=0,
        description="Maximum delay in seconds between retries"
    )
    exponential_base: float = Field(
        default=2.0,
        gt=1,
        description="Base for exponential backoff calculation"
    )
    jitter: bool = Field(
        default=True,
        description="Whether to add random jitter to retry delays"
    )
    retry_on_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retries"
    )


class IntegrationConfig(BaseModel):
    """
    Main configuration for API integration.

    Attributes:
        name: Name identifier for this integration
        base_url: Base URL for the API
        auth: Authentication configuration
        rate_limit: Rate limiting configuration
        retry: Retry configuration
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        default_headers: Default headers to include in all requests
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Name identifier for this integration"
    )
    base_url: HttpUrl = Field(
        ...,
        description="Base URL for the API"
    )
    auth: AuthConfig = Field(
        default_factory=AuthConfig,
        description="Authentication configuration"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates"
    )
    default_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Default headers to include in all requests"
    )


class APIRequest(BaseModel):
    """
    Model for API request parameters.

    Attributes:
        method: HTTP method to use
        endpoint: API endpoint path (appended to base_url)
        params: Query parameters
        data: Request body data (for POST, PUT, PATCH)
        json: JSON request body (alternative to data)
        headers: Additional headers for this request
        timeout: Override default timeout for this request
    """

    method: HTTPMethod = Field(
        default=HTTPMethod.GET,
        description="HTTP method to use"
    )
    endpoint: str = Field(
        ...,
        description="API endpoint path"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters"
    )
    data: Optional[Union[Dict[str, Any], str, bytes]] = Field(
        default=None,
        description="Request body data"
    )
    json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON request body"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional headers for this request"
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override default timeout"
    )


class APIResponse(BaseModel):
    """
    Model for API response data.

    Attributes:
        status_code: HTTP status code
        headers: Response headers
        data: Response body data
        json_data: Parsed JSON response (if applicable)
        success: Whether the request was successful
        error_message: Error message if request failed
        elapsed_time: Time taken for the request in seconds
        timestamp: Timestamp when response was received
        retry_count: Number of retries attempted
    """

    status_code: int = Field(
        ...,
        description="HTTP status code"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Response headers"
    )
    data: Optional[Union[str, bytes]] = Field(
        default=None,
        description="Response body data"
    )
    json_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed JSON response"
    )
    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if request failed"
    )
    elapsed_time: float = Field(
        ...,
        ge=0,
        description="Time taken for the request in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when response was received"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries attempted"
    )

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class IntegrationMetrics(BaseModel):
    """
    Metrics for monitoring integration performance.

    Attributes:
        integration_name: Name of the integration
        total_requests: Total number of requests made
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        average_response_time: Average response time in seconds
        rate_limit_hits: Number of times rate limit was hit
        retry_attempts: Total number of retry attempts
        last_request_time: Timestamp of last request
    """

    integration_name: str = Field(
        ...,
        description="Name of the integration"
    )
    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total number of requests made"
    )
    successful_requests: int = Field(
        default=0,
        ge=0,
        description="Number of successful requests"
    )
    failed_requests: int = Field(
        default=0,
        ge=0,
        description="Number of failed requests"
    )
    average_response_time: float = Field(
        default=0.0,
        ge=0,
        description="Average response time in seconds"
    )
    rate_limit_hits: int = Field(
        default=0,
        ge=0,
        description="Number of times rate limit was hit"
    )
    retry_attempts: int = Field(
        default=0,
        ge=0,
        description="Total number of retry attempts"
    )
    last_request_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last request"
    )

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
