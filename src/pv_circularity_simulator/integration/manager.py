"""
Integration Manager - Central orchestration of API integrations.

This module provides the IntegrationManager class, which serves as the main
entry point for managing multiple API integrations with unified configuration,
monitoring, and control.
"""

import logging
from typing import Any, Dict, List, Optional

from .auth import AuthenticationHandler, create_auth_handler
from .connectors import APIConnector, RESTConnector
from .models import (
    APIRequest,
    APIResponse,
    AuthConfig,
    IntegrationConfig,
    IntegrationMetrics,
    RateLimitConfig,
    RetryConfig,
)
from .rate_limiter import RateLimiter
from .retry import RetryHandler

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Central manager for API integrations.

    The IntegrationManager provides a unified interface for managing multiple
    API integrations. It handles configuration, connection management, and
    provides convenience methods for common operations.

    Features:
    - Manage multiple named integrations
    - Centralized configuration management
    - Metrics collection and monitoring
    - Automatic connection pooling
    - Health check capabilities

    Attributes:
        integrations: Dictionary mapping integration names to configurations
        connectors: Dictionary mapping integration names to API connectors
        metrics: Dictionary mapping integration names to metrics
    """

    def __init__(self):
        """Initialize the Integration Manager."""
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.connectors: Dict[str, APIConnector] = {}
        self.metrics: Dict[str, IntegrationMetrics] = {}

    def register_integration(
        self,
        config: IntegrationConfig,
        connector_class: type = RESTConnector
    ) -> None:
        """
        Register a new integration.

        Args:
            config: Integration configuration
            connector_class: API connector class to use (default: RESTConnector)

        Raises:
            ValueError: If integration with the same name already exists
        """
        if config.name in self.integrations:
            raise ValueError(
                f"Integration '{config.name}' already exists. "
                f"Use update_integration() to modify it."
            )

        logger.info(f"Registering integration: {config.name}")

        # Store configuration
        self.integrations[config.name] = config

        # Create connector
        self.connectors[config.name] = connector_class(config)

        # Initialize metrics
        self.metrics[config.name] = IntegrationMetrics(
            integration_name=config.name
        )

        logger.info(f"Integration '{config.name}' registered successfully")

    def unregister_integration(self, name: str) -> None:
        """
        Unregister an integration.

        Args:
            name: Name of the integration to remove

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.integrations:
            raise KeyError(f"Integration '{name}' not found")

        logger.info(f"Unregistering integration: {name}")

        del self.integrations[name]
        del self.connectors[name]
        del self.metrics[name]

        logger.info(f"Integration '{name}' unregistered successfully")

    def update_integration(self, config: IntegrationConfig) -> None:
        """
        Update an existing integration configuration.

        Args:
            config: New integration configuration

        Raises:
            KeyError: If integration does not exist
        """
        if config.name not in self.integrations:
            raise KeyError(
                f"Integration '{config.name}' not found. "
                f"Use register_integration() to create it."
            )

        logger.info(f"Updating integration: {config.name}")

        # Update configuration
        self.integrations[config.name] = config

        # Recreate connector with new config
        connector_class = type(self.connectors[config.name])
        self.connectors[config.name] = connector_class(config)

        logger.info(f"Integration '{config.name}' updated successfully")

    def get_integration(self, name: str) -> IntegrationConfig:
        """
        Get integration configuration.

        Args:
            name: Name of the integration

        Returns:
            Integration configuration

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.integrations:
            raise KeyError(f"Integration '{name}' not found")

        return self.integrations[name]

    def list_integrations(self) -> List[str]:
        """
        List all registered integration names.

        Returns:
            List of integration names
        """
        return list(self.integrations.keys())

    def api_connectors(self, name: str) -> APIConnector:
        """
        Get API connector for an integration.

        This method provides access to the API connector for making requests
        to a specific integration.

        Args:
            name: Name of the integration

        Returns:
            API connector instance

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.connectors:
            raise KeyError(f"Integration '{name}' not found")

        return self.connectors[name]

    def authentication_handlers(self, name: str) -> AuthenticationHandler:
        """
        Get authentication handler for an integration.

        Args:
            name: Name of the integration

        Returns:
            Authentication handler instance

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.connectors:
            raise KeyError(f"Integration '{name}' not found")

        return self.connectors[name].auth_handler

    def rate_limiting(self, name: str) -> RateLimiter:
        """
        Get rate limiter for an integration.

        Args:
            name: Name of the integration

        Returns:
            Rate limiter instance

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.connectors:
            raise KeyError(f"Integration '{name}' not found")

        return self.connectors[name].rate_limiter

    def error_retry_logic(self, name: str) -> RetryHandler:
        """
        Get retry handler for an integration.

        Args:
            name: Name of the integration

        Returns:
            Retry handler instance

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.connectors:
            raise KeyError(f"Integration '{name}' not found")

        return self.connectors[name].retry_handler

    def request(self, name: str, request: APIRequest) -> APIResponse:
        """
        Make a synchronous API request through an integration.

        Args:
            name: Name of the integration
            request: API request parameters

        Returns:
            API response

        Raises:
            KeyError: If integration does not exist
        """
        connector = self.api_connectors(name)
        response = connector.request(request)

        # Update metrics
        self._update_metrics(name, response)

        return response

    async def request_async(
        self,
        name: str,
        request: APIRequest
    ) -> APIResponse:
        """
        Make an asynchronous API request through an integration.

        Args:
            name: Name of the integration
            request: API request parameters

        Returns:
            API response

        Raises:
            KeyError: If integration does not exist
        """
        connector = self.api_connectors(name)
        response = await connector.request_async(request)

        # Update metrics
        self._update_metrics(name, response)

        return response

    def _update_metrics(self, name: str, response: APIResponse) -> None:
        """
        Update metrics for an integration based on response.

        Args:
            name: Integration name
            response: API response
        """
        metrics = self.metrics[name]

        # Update counters
        metrics.total_requests += 1
        if response.success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update retry attempts
        metrics.retry_attempts += response.retry_count

        # Update average response time (moving average)
        if metrics.total_requests == 1:
            metrics.average_response_time = response.elapsed_time
        else:
            # Calculate weighted average
            metrics.average_response_time = (
                (metrics.average_response_time * (metrics.total_requests - 1) +
                 response.elapsed_time) / metrics.total_requests
            )

        # Update last request time
        metrics.last_request_time = response.timestamp

    def get_metrics(self, name: str) -> IntegrationMetrics:
        """
        Get metrics for an integration.

        Args:
            name: Name of the integration

        Returns:
            Integration metrics

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.metrics:
            raise KeyError(f"Integration '{name}' not found")

        return self.metrics[name]

    def get_all_metrics(self) -> Dict[str, IntegrationMetrics]:
        """
        Get metrics for all integrations.

        Returns:
            Dictionary mapping integration names to metrics
        """
        return self.metrics.copy()

    def reset_metrics(self, name: str) -> None:
        """
        Reset metrics for an integration.

        Args:
            name: Name of the integration

        Raises:
            KeyError: If integration does not exist
        """
        if name not in self.metrics:
            raise KeyError(f"Integration '{name}' not found")

        self.metrics[name] = IntegrationMetrics(integration_name=name)
        logger.info(f"Metrics reset for integration: {name}")

    def health_check(self, name: str, endpoint: str = "/health") -> bool:
        """
        Perform a health check on an integration.

        Args:
            name: Name of the integration
            endpoint: Health check endpoint (default: "/health")

        Returns:
            True if integration is healthy, False otherwise
        """
        try:
            connector = self.api_connectors(name)
            response = connector.get(endpoint)
            return response.success
        except Exception as e:
            logger.error(f"Health check failed for '{name}': {e}")
            return False

    async def health_check_async(
        self,
        name: str,
        endpoint: str = "/health"
    ) -> bool:
        """
        Perform an asynchronous health check on an integration.

        Args:
            name: Name of the integration
            endpoint: Health check endpoint (default: "/health")

        Returns:
            True if integration is healthy, False otherwise
        """
        try:
            connector = self.api_connectors(name)
            response = await connector.get_async(endpoint)
            return response.success
        except Exception as e:
            logger.error(f"Health check failed for '{name}': {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all integrations.

        Returns:
            Dictionary containing status information for all integrations
        """
        status = {
            "total_integrations": len(self.integrations),
            "integrations": {}
        }

        for name in self.integrations:
            config = self.integrations[name]
            metrics = self.metrics[name]
            rate_limiter = self.rate_limiting(name)
            retry_handler = self.error_retry_logic(name)

            status["integrations"][name] = {
                "base_url": str(config.base_url),
                "auth_type": config.auth.auth_type.value,
                "rate_limit": rate_limiter.get_stats(),
                "retry_config": retry_handler.get_stats(),
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": (
                        metrics.successful_requests / metrics.total_requests
                        if metrics.total_requests > 0 else 0
                    ),
                    "average_response_time": metrics.average_response_time,
                    "retry_attempts": metrics.retry_attempts,
                    "last_request_time": (
                        metrics.last_request_time.isoformat()
                        if metrics.last_request_time else None
                    ),
                }
            }

        return status

    def __repr__(self) -> str:
        """String representation of IntegrationManager."""
        return (
            f"IntegrationManager("
            f"integrations={len(self.integrations)}, "
            f"names={list(self.integrations.keys())}"
            f")"
        )
