"""
Tests for IntegrationManager.
"""

import pytest

from pv_circularity_simulator.integration import (
    IntegrationManager,
    IntegrationConfig,
    AuthConfig,
    AuthType,
)


class TestIntegrationManager:
    """Tests for IntegrationManager class."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = IntegrationManager()
        assert len(manager.list_integrations()) == 0

    def test_register_integration(self):
        """Test registering a new integration."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        assert "test-api" in manager.list_integrations()

    def test_register_duplicate_integration(self):
        """Test registering duplicate integration raises error."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)

        with pytest.raises(ValueError, match="already exists"):
            manager.register_integration(config)

    def test_unregister_integration(self):
        """Test unregistering an integration."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        assert "test-api" in manager.list_integrations()

        manager.unregister_integration("test-api")
        assert "test-api" not in manager.list_integrations()

    def test_unregister_nonexistent_integration(self):
        """Test unregistering non-existent integration raises error."""
        manager = IntegrationManager()

        with pytest.raises(KeyError):
            manager.unregister_integration("nonexistent")

    def test_update_integration(self):
        """Test updating an integration."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com",
            timeout=30.0
        )

        manager.register_integration(config)

        # Update with new timeout
        updated_config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com",
            timeout=60.0
        )

        manager.update_integration(updated_config)

        # Verify update
        retrieved = manager.get_integration("test-api")
        assert retrieved.timeout == 60.0

    def test_get_integration(self):
        """Test getting integration configuration."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        retrieved = manager.get_integration("test-api")

        assert retrieved.name == "test-api"
        assert str(retrieved.base_url) == "https://api.example.com/"

    def test_api_connectors(self):
        """Test getting API connector."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        connector = manager.api_connectors("test-api")

        assert connector is not None
        assert connector.config.name == "test-api"

    def test_authentication_handlers(self):
        """Test getting authentication handler."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com",
            auth=AuthConfig(
                auth_type=AuthType.API_KEY,
                api_key="test-key"
            )
        )

        manager.register_integration(config)
        auth_handler = manager.authentication_handlers("test-api")

        assert auth_handler is not None
        assert auth_handler.config.auth_type == AuthType.API_KEY

    def test_rate_limiting(self):
        """Test getting rate limiter."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        rate_limiter = manager.rate_limiting("test-api")

        assert rate_limiter is not None
        assert rate_limiter.enabled is True

    def test_error_retry_logic(self):
        """Test getting retry handler."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        retry_handler = manager.error_retry_logic("test-api")

        assert retry_handler is not None
        assert retry_handler.config.enabled is True

    def test_get_metrics(self):
        """Test getting metrics for integration."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        metrics = manager.get_metrics("test-api")

        assert metrics.integration_name == "test-api"
        assert metrics.total_requests == 0

    def test_reset_metrics(self):
        """Test resetting metrics."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)

        # Manually update metrics
        metrics = manager.metrics["test-api"]
        metrics.total_requests = 100

        # Reset
        manager.reset_metrics("test-api")

        # Verify reset
        metrics = manager.get_metrics("test-api")
        assert metrics.total_requests == 0

    def test_get_status(self):
        """Test getting overall status."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        status = manager.get_status()

        assert status["total_integrations"] == 1
        assert "test-api" in status["integrations"]

    def test_manager_repr(self):
        """Test manager string representation."""
        manager = IntegrationManager()
        config = IntegrationConfig(
            name="test-api",
            base_url="https://api.example.com"
        )

        manager.register_integration(config)
        repr_str = repr(manager)

        assert "IntegrationManager" in repr_str
        assert "test-api" in repr_str
