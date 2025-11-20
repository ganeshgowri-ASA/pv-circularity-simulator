"""
Comprehensive Configuration Management System for PV Circularity Simulator.

This module provides a production-ready configuration management system with
support for YAML/JSON configuration files, environment variables, validation,
and persistence. It's designed to handle complex configuration scenarios across
different environments (development, staging, production) with proper error
handling and extensive validation.

Key Features:
    - Multi-format configuration loading (YAML, JSON)
    - Environment variable integration and override
    - Comprehensive validation with type checking and constraints
    - Configuration persistence and updates
    - Multi-environment support
    - Thread-safe operations
    - Extensive error handling and logging

Examples:
    Basic usage:
        >>> config = ConfigurationManager()
        >>> config.load_configuration('config/app.yaml')
        >>> db_host = config.get('database.host')
        >>> config.set('database.port', 5432)
        >>> config.save_configuration()

    With environment variables:
        >>> config = ConfigurationManager()
        >>> config.load_configuration('config/app.yaml')
        >>> config.apply_environment_overrides()
        >>> # Environment variables like APP_DATABASE_HOST override config values

    With validation:
        >>> schema = {
        ...     'database': {
        ...         'host': {'type': str, 'required': True},
        ...         'port': {'type': int, 'min': 1, 'max': 65535}
        ...     }
        ... }
        >>> config = ConfigurationManager(schema=schema)
        >>> config.load_configuration('config/app.yaml')
        >>> config.validate()  # Raises ConfigValidationError if invalid
"""

import os
import json
import yaml
import copy
import threading
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime

from .exceptions import (
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigPersistenceError,
    ConfigParseError,
    EnvironmentVariableError,
)
from .validators import ConfigValidator, SchemaValidator, ValidationError


class ConfigurationManager:
    """
    Production-ready configuration management system.

    This class provides comprehensive configuration management with support for
    multiple file formats, environment variables, validation, and persistence.
    It's thread-safe and designed for production use.

    Attributes:
        config: Current configuration dictionary
        schema: Optional validation schema
        config_file: Path to the loaded configuration file
        environment: Current environment name (e.g., 'development', 'production')
        env_prefix: Prefix for environment variable overrides

    Thread Safety:
        All public methods are thread-safe through the use of an internal lock.

    Examples:
        >>> config = ConfigurationManager()
        >>> config.load_configuration('config/app.yaml')
        >>> value = config.get('section.key', default='default_value')
        >>> config.set('section.new_key', 'new_value')
        >>> config.save_configuration()
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        schema: Optional[dict[str, Any]] = None,
        environment: str = "development",
        env_prefix: str = "APP",
        auto_load: bool = False,
        auto_validate: bool = True,
    ):
        """
        Initialize the ConfigurationManager.

        Args:
            config_file: Optional path to configuration file to load immediately
            schema: Optional validation schema for configuration
            environment: Environment name (e.g., 'development', 'staging', 'production')
            env_prefix: Prefix for environment variable overrides (e.g., 'APP')
            auto_load: Whether to automatically load config_file if provided
            auto_validate: Whether to automatically validate after loading

        Raises:
            ConfigFileNotFoundError: If auto_load is True and config_file doesn't exist
            ConfigValidationError: If auto_validate is True and validation fails
        """
        self._config: dict[str, Any] = {}
        self._schema = schema
        self._config_file: Optional[Path] = Path(config_file) if config_file else None
        self._environment = environment
        self._env_prefix = env_prefix
        self._auto_validate = auto_validate
        self._lock = threading.RLock()
        self._metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "last_modified": None,
            "last_loaded": None,
            "version": "1.0.0",
        }

        # Initialize validators
        self._validator = ConfigValidator()
        self._schema_validator = SchemaValidator(schema) if schema else None

        # Auto-load if requested
        if auto_load and config_file:
            self.load_configuration(config_file)

    @property
    def config(self) -> dict[str, Any]:
        """
        Get the current configuration dictionary (read-only copy).

        Returns:
            Deep copy of the current configuration
        """
        with self._lock:
            return copy.deepcopy(self._config)

    @property
    def schema(self) -> Optional[dict[str, Any]]:
        """
        Get the configuration schema.

        Returns:
            Configuration schema dictionary or None
        """
        return self._schema

    @property
    def config_file(self) -> Optional[Path]:
        """
        Get the path to the loaded configuration file.

        Returns:
            Path object or None if no file is loaded
        """
        return self._config_file

    @property
    def environment(self) -> str:
        """
        Get the current environment name.

        Returns:
            Environment name string
        """
        return self._environment

    @property
    def env_prefix(self) -> str:
        """
        Get the environment variable prefix.

        Returns:
            Environment variable prefix string
        """
        return self._env_prefix

    def load_configuration(
        self,
        config_file: Union[str, Path],
        merge: bool = False,
        validate: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Load configuration from a YAML or JSON file.

        This method loads configuration from a file and optionally validates it
        against the schema. It supports both YAML and JSON formats, automatically
        detecting the format based on file extension.

        Args:
            config_file: Path to configuration file (.yaml, .yml, or .json)
            merge: Whether to merge with existing configuration (default: False)
            validate: Whether to validate after loading (default: use auto_validate)

        Returns:
            Loaded configuration dictionary

        Raises:
            ConfigFileNotFoundError: If the configuration file doesn't exist
            ConfigParseError: If the file cannot be parsed
            ConfigValidationError: If validation is enabled and fails

        Examples:
            >>> config = ConfigurationManager()
            >>> config.load_configuration('config/app.yaml')
            {'database': {'host': 'localhost', 'port': 5432}}

            >>> # Merge with existing config
            >>> config.load_configuration('config/overrides.yaml', merge=True)
        """
        config_path = Path(config_file)

        with self._lock:
            # Check if file exists
            if not config_path.exists():
                raise ConfigFileNotFoundError(str(config_path))

            # Load based on file extension
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        loaded_config = yaml.safe_load(f) or {}
                    elif config_path.suffix.lower() == '.json':
                        loaded_config = json.load(f)
                    else:
                        raise ConfigParseError(
                            str(config_path),
                            "unknown",
                            message=f"Unsupported file format: {config_path.suffix}"
                        )
            except (yaml.YAMLError, json.JSONDecodeError) as e:
                raise ConfigParseError(
                    str(config_path),
                    config_path.suffix[1:],
                    original_error=e
                )
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to read configuration file: {config_path}",
                    {"error": str(e)}
                )

            # Merge or replace configuration
            if merge:
                self._deep_merge(self._config, loaded_config)
            else:
                self._config = loaded_config

            # Update metadata
            self._config_file = config_path
            self._metadata["last_loaded"] = datetime.now().isoformat()

            # Validate if requested
            should_validate = validate if validate is not None else self._auto_validate
            if should_validate:
                self.validate()

            return copy.deepcopy(self._config)

    def save_configuration(
        self,
        config_file: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
        create_backup: bool = True,
    ) -> Path:
        """
        Save current configuration to a file.

        This method persists the current configuration to disk in YAML or JSON
        format. It can create backups of existing files before overwriting.

        Args:
            config_file: Optional path to save to (default: use loaded file)
            format: Output format ('yaml' or 'json', default: auto-detect from extension)
            create_backup: Whether to create a backup of existing file

        Returns:
            Path where configuration was saved

        Raises:
            ConfigPersistenceError: If the save operation fails
            ConfigurationError: If no config_file is specified and none is loaded

        Examples:
            >>> config = ConfigurationManager()
            >>> config.load_configuration('config/app.yaml')
            >>> config.set('database.port', 5433)
            >>> config.save_configuration()  # Saves to config/app.yaml
            PosixPath('config/app.yaml')

            >>> # Save to different file
            >>> config.save_configuration('config/app_updated.json', format='json')
            PosixPath('config/app_updated.json')
        """
        with self._lock:
            # Determine target file
            if config_file:
                target_path = Path(config_file)
            elif self._config_file:
                target_path = self._config_file
            else:
                raise ConfigurationError(
                    "No configuration file specified and none is currently loaded"
                )

            # Determine format
            if format:
                save_format = format.lower()
            else:
                save_format = target_path.suffix[1:].lower()
                if save_format not in ['yaml', 'yml', 'json']:
                    save_format = 'yaml'

            # Create backup if requested
            if create_backup and target_path.exists():
                backup_path = target_path.with_suffix(
                    f"{target_path.suffix}.backup"
                )
                try:
                    import shutil
                    shutil.copy2(target_path, backup_path)
                except Exception as e:
                    raise ConfigPersistenceError(
                        "backup",
                        str(target_path),
                        original_error=e
                    )

            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    if save_format in ['yaml', 'yml']:
                        yaml.dump(
                            self._config,
                            f,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True
                        )
                    elif save_format == 'json':
                        json.dump(
                            self._config,
                            f,
                            indent=2,
                            ensure_ascii=False
                        )
            except Exception as e:
                raise ConfigPersistenceError(
                    "save",
                    str(target_path),
                    original_error=e
                )

            # Update metadata
            self._metadata["last_modified"] = datetime.now().isoformat()

            return target_path

    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        value_type: Optional[type] = None,
    ) -> Any:
        """
        Get a configuration value using dot notation.

        Retrieves configuration values using dot-separated keys for nested
        access (e.g., 'database.connection.host').

        Args:
            key: Configuration key in dot notation (e.g., 'section.subsection.key')
            default: Default value if key doesn't exist
            required: Whether to raise an error if key doesn't exist
            value_type: Optional type to validate the retrieved value

        Returns:
            Configuration value or default

        Raises:
            ConfigurationError: If required is True and key doesn't exist
            ConfigValidationError: If value_type is specified and doesn't match

        Examples:
            >>> config.get('database.host')
            'localhost'
            >>> config.get('database.port', default=5432)
            5432
            >>> config.get('api.key', required=True)
            ConfigurationError: Required configuration key 'api.key' not found
        """
        with self._lock:
            value = self._get_nested(self._config, key.split('.'))

            if value is None:
                if required:
                    raise ConfigurationError(
                        f"Required configuration key '{key}' not found"
                    )
                return default

            # Type validation if requested
            if value_type is not None and value is not None:
                try:
                    self._validator.validate_type(key, value, value_type)
                except ConfigValidationError:
                    raise

            return value

    def set(
        self,
        key: str,
        value: Any,
        validate: bool = True,
        create_path: bool = True,
    ) -> None:
        """
        Set a configuration value using dot notation.

        Sets configuration values using dot-separated keys, automatically
        creating nested dictionaries as needed.

        Args:
            key: Configuration key in dot notation (e.g., 'section.subsection.key')
            value: Value to set
            validate: Whether to validate the value if schema exists
            create_path: Whether to create intermediate keys if they don't exist

        Raises:
            ConfigurationError: If create_path is False and path doesn't exist
            ConfigValidationError: If validate is True and validation fails

        Examples:
            >>> config.set('database.host', 'localhost')
            >>> config.set('database.port', 5432)
            >>> config.set('new.nested.key', 'value')  # Creates nested structure
        """
        with self._lock:
            # Validate if schema exists and validation is requested
            if validate and self._schema_validator:
                # Create a temporary config for validation
                temp_config = copy.deepcopy(self._config)
                self._set_nested(temp_config, key.split('.'), value, create_path)
                self._schema_validator.validate(temp_config)

            # Set the value
            self._set_nested(self._config, key.split('.'), value, create_path)

            # Update metadata
            self._metadata["last_modified"] = datetime.now().isoformat()

    def delete(self, key: str, ignore_missing: bool = True) -> bool:
        """
        Delete a configuration key.

        Args:
            key: Configuration key in dot notation
            ignore_missing: Whether to ignore if key doesn't exist

        Returns:
            True if key was deleted, False if it didn't exist

        Raises:
            ConfigurationError: If ignore_missing is False and key doesn't exist

        Examples:
            >>> config.delete('database.old_setting')
            True
            >>> config.delete('nonexistent.key')
            False
        """
        with self._lock:
            keys = key.split('.')
            if len(keys) == 1:
                if keys[0] in self._config:
                    del self._config[keys[0]]
                    self._metadata["last_modified"] = datetime.now().isoformat()
                    return True
                elif not ignore_missing:
                    raise ConfigurationError(f"Configuration key '{key}' not found")
                return False

            # Navigate to parent
            parent = self._config
            for k in keys[:-1]:
                if k not in parent:
                    if not ignore_missing:
                        raise ConfigurationError(f"Configuration key '{key}' not found")
                    return False
                parent = parent[k]

            # Delete the key
            if keys[-1] in parent:
                del parent[keys[-1]]
                self._metadata["last_modified"] = datetime.now().isoformat()
                return True
            elif not ignore_missing:
                raise ConfigurationError(f"Configuration key '{key}' not found")
            return False

    def validate(self, config: Optional[dict[str, Any]] = None) -> bool:
        """
        Validate configuration against the schema.

        Args:
            config: Optional configuration to validate (default: current config)

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If validation fails
            ConfigurationError: If no schema is defined

        Examples:
            >>> config = ConfigurationManager(schema=my_schema)
            >>> config.load_configuration('config/app.yaml')
            >>> config.validate()
            True
        """
        if not self._schema_validator:
            raise ConfigurationError(
                "No validation schema defined for this configuration manager"
            )

        with self._lock:
            config_to_validate = config if config is not None else self._config
            return self._schema_validator.validate(config_to_validate)

    def apply_environment_overrides(
        self,
        prefix: Optional[str] = None,
        separator: str = "_",
        lowercase: bool = False,
    ) -> dict[str, str]:
        """
        Apply environment variable overrides to configuration.

        Environment variables are mapped to configuration keys using a prefix
        and separator. For example, with prefix "APP" and separator "_":
        - APP_DATABASE_HOST maps to config['database']['host']
        - APP_API_KEY maps to config['api']['key']

        Args:
            prefix: Environment variable prefix (default: use env_prefix)
            separator: Separator between key parts (default: "_")
            lowercase: Whether to convert keys to lowercase

        Returns:
            Dictionary of applied environment variable overrides

        Examples:
            >>> # With environment variable: APP_DATABASE_PORT=5433
            >>> config.apply_environment_overrides()
            {'DATABASE_PORT': '5433'}
            >>> config.get('database.port')
            '5433'
        """
        env_prefix = prefix or self._env_prefix
        applied = {}

        with self._lock:
            for env_key, env_value in os.environ.items():
                # Check if environment variable starts with prefix
                if not env_key.startswith(f"{env_prefix}{separator}"):
                    continue

                # Extract configuration key
                config_key = env_key[len(env_prefix) + len(separator):]
                key_parts = config_key.split(separator)

                if lowercase:
                    key_parts = [part.lower() for part in key_parts]

                # Convert to dot notation
                dot_key = '.'.join(key_parts)

                # Try to parse value as JSON first for complex types
                try:
                    parsed_value = json.loads(env_value)
                except (json.JSONDecodeError, ValueError):
                    # Use string value if not valid JSON
                    parsed_value = env_value

                # Set the value
                try:
                    self.set(dot_key, parsed_value, validate=False, create_path=True)
                    applied[config_key] = env_value
                except Exception:
                    # Skip if setting fails
                    continue

            # Update metadata
            if applied:
                self._metadata["last_modified"] = datetime.now().isoformat()

        return applied

    def get_environment_config(
        self,
        environment: Optional[str] = None,
        config_key: str = "environments",
    ) -> dict[str, Any]:
        """
        Get environment-specific configuration section.

        Retrieves configuration for a specific environment from a dedicated
        section in the configuration file.

        Args:
            environment: Environment name (default: use current environment)
            config_key: Configuration key containing environment configs

        Returns:
            Environment-specific configuration dictionary

        Raises:
            ConfigurationError: If environment configuration doesn't exist

        Examples:
            >>> # With config: {'environments': {'production': {'debug': False}}}
            >>> config.get_environment_config('production')
            {'debug': False}
        """
        env = environment or self._environment

        with self._lock:
            env_configs = self.get(config_key, default={})

            if not isinstance(env_configs, dict):
                raise ConfigurationError(
                    f"Environment configuration '{config_key}' must be a dictionary"
                )

            if env not in env_configs:
                raise ConfigurationError(
                    f"Configuration for environment '{env}' not found"
                )

            return copy.deepcopy(env_configs[env])

    def merge_environment_config(
        self,
        environment: Optional[str] = None,
        config_key: str = "environments",
    ) -> None:
        """
        Merge environment-specific configuration into main configuration.

        Args:
            environment: Environment name (default: use current environment)
            config_key: Configuration key containing environment configs

        Examples:
            >>> config.merge_environment_config('production')
            >>> # Production-specific settings now override base config
        """
        try:
            env_config = self.get_environment_config(environment, config_key)

            with self._lock:
                self._deep_merge(self._config, env_config)
                self._metadata["last_modified"] = datetime.now().isoformat()
        except ConfigurationError:
            # Silently ignore if environment config doesn't exist
            pass

    def reset(self) -> None:
        """
        Reset configuration to empty state.

        Examples:
            >>> config.reset()
            >>> config.get('database.host')
            None
        """
        with self._lock:
            self._config = {}
            self._config_file = None
            self._metadata["last_modified"] = datetime.now().isoformat()

    def reload(self, validate: Optional[bool] = None) -> dict[str, Any]:
        """
        Reload configuration from the original file.

        Args:
            validate: Whether to validate after reloading

        Returns:
            Reloaded configuration dictionary

        Raises:
            ConfigurationError: If no configuration file is loaded
            ConfigFileNotFoundError: If the configuration file no longer exists

        Examples:
            >>> config.reload()
            {'database': {'host': 'localhost', 'port': 5432}}
        """
        with self._lock:
            if not self._config_file:
                raise ConfigurationError(
                    "Cannot reload: no configuration file is currently loaded"
                )

            return self.load_configuration(
                self._config_file,
                merge=False,
                validate=validate
            )

    def get_metadata(self) -> dict[str, Any]:
        """
        Get configuration metadata.

        Returns:
            Dictionary containing metadata about the configuration

        Examples:
            >>> config.get_metadata()
            {
                'created_at': '2025-01-15T10:30:00',
                'last_modified': '2025-01-15T11:45:00',
                'last_loaded': '2025-01-15T10:30:00',
                'version': '1.0.0'
            }
        """
        with self._lock:
            return copy.deepcopy(self._metadata)

    def _get_nested(self, config: dict[str, Any], keys: list[str]) -> Any:
        """
        Get a nested value from configuration dictionary.

        Args:
            config: Configuration dictionary
            keys: List of keys for nested access

        Returns:
            Value at the nested location or None
        """
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_nested(
        self,
        config: dict[str, Any],
        keys: list[str],
        value: Any,
        create_path: bool = True,
    ) -> None:
        """
        Set a nested value in configuration dictionary.

        Args:
            config: Configuration dictionary
            keys: List of keys for nested access
            value: Value to set
            create_path: Whether to create intermediate dictionaries

        Raises:
            ConfigurationError: If create_path is False and path doesn't exist
        """
        current = config
        for key in keys[:-1]:
            if key not in current:
                if not create_path:
                    raise ConfigurationError(
                        f"Configuration path does not exist: {'.'.join(keys)}"
                    )
                current[key] = {}
            elif not isinstance(current[key], dict):
                if not create_path:
                    raise ConfigurationError(
                        f"Cannot create path through non-dict value: {key}"
                    )
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """
        Deep merge two dictionaries, modifying base in place.

        Args:
            base: Base dictionary to merge into
            override: Dictionary with override values
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# Standalone utility functions

def settings_loader(
    config_file: Union[str, Path],
    environment: Optional[str] = None,
    apply_env_overrides: bool = True,
    env_prefix: str = "APP",
) -> dict[str, Any]:
    """
    Load configuration settings from a file with optional environment overrides.

    This is a convenience function that creates a ConfigurationManager instance,
    loads the configuration, and optionally applies environment variable overrides.

    Args:
        config_file: Path to configuration file (.yaml, .yml, or .json)
        environment: Optional environment name for environment-specific config
        apply_env_overrides: Whether to apply environment variable overrides
        env_prefix: Prefix for environment variables

    Returns:
        Loaded and processed configuration dictionary

    Raises:
        ConfigFileNotFoundError: If the configuration file doesn't exist
        ConfigParseError: If the file cannot be parsed

    Examples:
        >>> config = settings_loader('config/app.yaml')
        >>> print(config['database']['host'])
        localhost

        >>> # With environment overrides
        >>> config = settings_loader('config/app.yaml', apply_env_overrides=True)
        >>> # APP_DATABASE_PORT environment variable overrides config file
    """
    manager = ConfigurationManager(
        environment=environment or "development",
        env_prefix=env_prefix,
        auto_validate=False,
    )

    manager.load_configuration(config_file, validate=False)

    # Merge environment-specific configuration if environment is specified
    if environment:
        try:
            manager.merge_environment_config(environment)
        except ConfigurationError:
            # Environment config doesn't exist, continue without it
            pass

    # Apply environment variable overrides
    if apply_env_overrides:
        manager.apply_environment_overrides()

    return manager.config


def preference_validator(
    config: dict[str, Any],
    schema: dict[str, Any],
    raise_on_error: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate configuration preferences against a schema.

    This function validates a configuration dictionary against a defined schema,
    collecting all validation errors.

    Args:
        config: Configuration dictionary to validate
        schema: Validation schema
        raise_on_error: Whether to raise exception on first error (default: True)

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Raises:
        ConfigValidationError: If raise_on_error is True and validation fails

    Examples:
        >>> schema = {
        ...     'database': {
        ...         'host': {'type': str, 'required': True},
        ...         'port': {'type': int, 'min': 1, 'max': 65535}
        ...     }
        ... }
        >>> config = {'database': {'host': 'localhost', 'port': 5432}}
        >>> is_valid, errors = preference_validator(config, schema)
        >>> print(is_valid)
        True

        >>> # Invalid config
        >>> invalid_config = {'database': {'host': 'localhost', 'port': 99999}}
        >>> is_valid, errors = preference_validator(
        ...     invalid_config, schema, raise_on_error=False
        ... )
        >>> print(is_valid, errors)
        False ['Validation failed for field database.port: must be <= 65535']
    """
    validator = SchemaValidator(schema)
    errors = []

    try:
        validator.validate(config)
        return True, []
    except ConfigValidationError as e:
        errors.append(str(e))
        if raise_on_error:
            raise
        return False, errors


def environment_config_handler(
    base_config: dict[str, Any],
    environment: str,
    env_prefix: str = "APP",
    env_separator: str = "_",
) -> dict[str, Any]:
    """
    Handle environment-specific configuration with environment variable overrides.

    This function combines base configuration with environment-specific settings
    and environment variable overrides to produce a final configuration for a
    specific environment.

    Args:
        base_config: Base configuration dictionary
        environment: Environment name (e.g., 'development', 'production')
        env_prefix: Prefix for environment variables
        env_separator: Separator for environment variable keys

    Returns:
        Final configuration dictionary with all overrides applied

    Examples:
        >>> base_config = {
        ...     'database': {'host': 'localhost', 'port': 5432},
        ...     'environments': {
        ...         'production': {'database': {'host': 'prod-db.example.com'}}
        ...     }
        ... }
        >>> config = environment_config_handler(base_config, 'production')
        >>> print(config['database']['host'])
        prod-db.example.com

        >>> # With environment variable APP_DATABASE_PORT=5433
        >>> config = environment_config_handler(base_config, 'production')
        >>> print(config['database']['port'])
        5433
    """
    manager = ConfigurationManager(
        environment=environment,
        env_prefix=env_prefix,
        auto_validate=False,
    )

    # Load base configuration
    manager._config = copy.deepcopy(base_config)

    # Merge environment-specific configuration
    manager.merge_environment_config(environment)

    # Apply environment variable overrides
    manager.apply_environment_overrides(separator=env_separator)

    return manager.config


def settings_persistence(
    config: dict[str, Any],
    config_file: Union[str, Path],
    format: str = "yaml",
    create_backup: bool = True,
    validate_schema: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Persist configuration settings to a file with optional validation.

    This function saves configuration to disk in YAML or JSON format, with
    optional schema validation and backup creation.

    Args:
        config: Configuration dictionary to save
        config_file: Path where configuration should be saved
        format: Output format ('yaml' or 'json')
        create_backup: Whether to create backup of existing file
        validate_schema: Optional schema to validate against before saving

    Returns:
        Path where configuration was saved

    Raises:
        ConfigValidationError: If validate_schema is provided and validation fails
        ConfigPersistenceError: If the save operation fails

    Examples:
        >>> config = {'database': {'host': 'localhost', 'port': 5432}}
        >>> path = settings_persistence(config, 'config/app.yaml')
        >>> print(path)
        config/app.yaml

        >>> # With validation
        >>> schema = {'database': {'host': {'type': str, 'required': True}}}
        >>> path = settings_persistence(
        ...     config, 'config/app.yaml', validate_schema=schema
        ... )
    """
    # Validate if schema is provided
    if validate_schema:
        validator = SchemaValidator(validate_schema)
        validator.validate(config)

    # Create manager and save
    manager = ConfigurationManager(auto_validate=False)
    manager._config = copy.deepcopy(config)

    return manager.save_configuration(
        config_file,
        format=format,
        create_backup=create_backup,
    )
