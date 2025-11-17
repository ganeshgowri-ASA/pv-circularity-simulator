"""
Configuration management package for PV Circularity Simulator.

This package provides comprehensive configuration management capabilities including:
- YAML/JSON configuration file loading and parsing
- Environment variable integration
- Settings validation and type checking
- Configuration persistence
- Multi-environment support (development, staging, production)
"""

from .configuration_manager import (
    ConfigurationManager,
    settings_loader,
    preference_validator,
    environment_config_handler,
    settings_persistence,
)
from .validators import (
    ValidationError,
    ConfigValidator,
    SchemaValidator,
)
from .exceptions import (
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigPersistenceError,
)

__all__ = [
    "ConfigurationManager",
    "settings_loader",
    "preference_validator",
    "environment_config_handler",
    "settings_persistence",
    "ValidationError",
    "ConfigValidator",
    "SchemaValidator",
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "ConfigPersistenceError",
]
