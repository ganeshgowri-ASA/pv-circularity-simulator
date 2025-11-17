"""
Integration Layer & API Connectors for PV Circularity Simulator.

This module provides a production-ready integration framework for connecting
to external APIs with authentication, rate limiting, and robust error handling.
"""

from .manager import IntegrationManager
from .models import (
    IntegrationConfig,
    APIResponse,
    AuthConfig,
    RateLimitConfig,
    RetryConfig,
)
from .auth import (
    AuthenticationHandler,
    APIKeyAuth,
    BearerTokenAuth,
    OAuth2Auth,
)
from .connectors import APIConnector, RESTConnector
from .rate_limiter import RateLimiter
from .retry import RetryHandler

__all__ = [
    "IntegrationManager",
    "IntegrationConfig",
    "APIResponse",
    "AuthConfig",
    "RateLimitConfig",
    "RetryConfig",
    "AuthenticationHandler",
    "APIKeyAuth",
    "BearerTokenAuth",
    "OAuth2Auth",
    "APIConnector",
    "RESTConnector",
    "RateLimiter",
    "RetryHandler",
]
