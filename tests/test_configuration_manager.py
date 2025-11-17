"""
Comprehensive test suite for the Configuration Management System.

This module contains extensive tests for all configuration management
functionality including loading, saving, validation, environment handling,
and error cases.
"""

import os
import json
import yaml
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pv_circularity_simulator.config import (
    ConfigurationManager,
    settings_loader,
    preference_validator,
    environment_config_handler,
    settings_persistence,
)
from pv_circularity_simulator.config.exceptions import (
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigPersistenceError,
    ConfigParseError,
    EnvironmentVariableError,
)
from pv_circularity_simulator.config.validators import (
    ConfigValidator,
    SchemaValidator,
    ValidationError,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "testuser",
            "password": "testpass",
            "database": "testdb",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8080,
            "debug": True,
            "workers": 4,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "file": "/var/log/app.log",
        },
        "environments": {
            "development": {
                "api": {"debug": True},
                "database": {"host": "localhost"},
            },
            "production": {
                "api": {"debug": False},
                "database": {"host": "prod-db.example.com"},
            },
        },
    }


@pytest.fixture
def sample_schema():
    """Sample validation schema for testing."""
    return {
        "database": {
            "host": {"type": str, "required": True, "non_empty": True},
            "port": {"type": int, "required": True, "min": 1, "max": 65535},
            "username": {"type": str, "required": True},
            "password": {"type": str, "required": True},
            "database": {"type": str, "required": True},
        },
        "api": {
            "host": {"type": str, "required": True},
            "port": {"type": int, "required": True, "min": 1, "max": 65535},
            "debug": {"type": bool, "required": False},
            "workers": {"type": int, "min": 1, "max": 100},
        },
        "logging": {
            "level": {
                "type": str,
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
            "format": {"type": str, "choices": ["json", "text"]},
        },
    }


@pytest.fixture
def yaml_config_file(temp_dir, sample_config):
    """Create a temporary YAML configuration file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def json_config_file(temp_dir, sample_config):
    """Create a temporary JSON configuration file."""
    config_file = temp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    return config_file


class TestConfigurationManager:
    """Test suite for ConfigurationManager class."""

    def test_initialization_default(self):
        """Test default initialization."""
        config = ConfigurationManager()
        assert config.config == {}
        assert config.schema is None
        assert config.config_file is None
        assert config.environment == "development"
        assert config.env_prefix == "APP"

    def test_initialization_with_schema(self, sample_schema):
        """Test initialization with schema."""
        config = ConfigurationManager(schema=sample_schema)
        assert config.schema == sample_schema

    def test_load_yaml_configuration(self, yaml_config_file, sample_config):
        """Test loading YAML configuration."""
        config = ConfigurationManager()
        loaded = config.load_configuration(yaml_config_file)
        assert loaded == sample_config
        assert config.config_file == yaml_config_file

    def test_load_json_configuration(self, json_config_file, sample_config):
        """Test loading JSON configuration."""
        config = ConfigurationManager()
        loaded = config.load_configuration(json_config_file)
        assert loaded == sample_config
        assert config.config_file == json_config_file

    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        config = ConfigurationManager()
        with pytest.raises(ConfigFileNotFoundError):
            config.load_configuration("nonexistent.yaml")

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML file."""
        invalid_file = temp_dir / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content:")

        config = ConfigurationManager()
        with pytest.raises(ConfigParseError):
            config.load_configuration(invalid_file)

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{invalid json}")

        config = ConfigurationManager()
        with pytest.raises(ConfigParseError):
            config.load_configuration(invalid_file)

    def test_save_yaml_configuration(self, temp_dir, sample_config):
        """Test saving configuration to YAML."""
        config = ConfigurationManager()
        config._config = sample_config

        output_file = temp_dir / "output.yaml"
        saved_path = config.save_configuration(output_file, format="yaml")

        assert saved_path == output_file
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded == sample_config

    def test_save_json_configuration(self, temp_dir, sample_config):
        """Test saving configuration to JSON."""
        config = ConfigurationManager()
        config._config = sample_config

        output_file = temp_dir / "output.json"
        saved_path = config.save_configuration(output_file, format="json")

        assert saved_path == output_file
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == sample_config

    def test_save_with_backup(self, temp_dir, sample_config):
        """Test saving with backup creation."""
        config = ConfigurationManager()
        config._config = sample_config

        output_file = temp_dir / "output.yaml"

        # Create initial file
        config.save_configuration(output_file, create_backup=False)

        # Save again with backup
        config._config["new_key"] = "new_value"
        config.save_configuration(output_file, create_backup=True)

        # Check backup exists
        backup_file = temp_dir / "output.yaml.backup"
        assert backup_file.exists()

    def test_get_simple_key(self, sample_config):
        """Test getting a simple configuration key."""
        config = ConfigurationManager()
        config._config = sample_config
        assert config.get("database.host") == "localhost"

    def test_get_nested_key(self, sample_config):
        """Test getting nested configuration key."""
        config = ConfigurationManager()
        config._config = sample_config
        assert config.get("database.port") == 5432

    def test_get_with_default(self, sample_config):
        """Test getting non-existent key with default."""
        config = ConfigurationManager()
        config._config = sample_config
        assert config.get("nonexistent.key", default="default") == "default"

    def test_get_required_missing(self, sample_config):
        """Test getting required but missing key."""
        config = ConfigurationManager()
        config._config = sample_config
        with pytest.raises(ConfigurationError):
            config.get("nonexistent.key", required=True)

    def test_get_with_type_validation(self, sample_config):
        """Test getting key with type validation."""
        config = ConfigurationManager()
        config._config = sample_config
        assert config.get("database.port", value_type=int) == 5432

        with pytest.raises(ConfigValidationError):
            config.get("database.port", value_type=str)

    def test_set_simple_key(self):
        """Test setting a simple configuration key."""
        config = ConfigurationManager()
        config.set("test_key", "test_value")
        assert config.get("test_key") == "test_value"

    def test_set_nested_key(self):
        """Test setting nested configuration key."""
        config = ConfigurationManager()
        config.set("section.subsection.key", "value")
        assert config.get("section.subsection.key") == "value"

    def test_set_without_path_creation(self):
        """Test setting key without automatic path creation."""
        config = ConfigurationManager()
        with pytest.raises(ConfigurationError):
            config.set("section.subsection.key", "value", create_path=False)

    def test_delete_key(self, sample_config):
        """Test deleting a configuration key."""
        config = ConfigurationManager()
        config._config = sample_config.copy()

        assert config.delete("database.password")
        assert config.get("database.password") is None

    def test_delete_nonexistent_key(self):
        """Test deleting non-existent key."""
        config = ConfigurationManager()
        assert config.delete("nonexistent.key") is False

        with pytest.raises(ConfigurationError):
            config.delete("nonexistent.key", ignore_missing=False)

    def test_validate_valid_config(self, sample_config, sample_schema):
        """Test validating valid configuration."""
        config = ConfigurationManager(schema=sample_schema)
        config._config = sample_config
        assert config.validate() is True

    def test_validate_invalid_config(self, sample_schema):
        """Test validating invalid configuration."""
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Invalid: exceeds max value
            }
        }
        config = ConfigurationManager(schema=sample_schema)
        config._config = invalid_config

        with pytest.raises(ConfigValidationError):
            config.validate()

    def test_validate_without_schema(self):
        """Test validating without schema."""
        config = ConfigurationManager()
        with pytest.raises(ConfigurationError):
            config.validate()

    def test_apply_environment_overrides(self, sample_config):
        """Test applying environment variable overrides."""
        config = ConfigurationManager(env_prefix="TEST")
        config._config = sample_config.copy()

        # Set environment variables
        os.environ["TEST_DATABASE_HOST"] = "env-host"
        os.environ["TEST_DATABASE_PORT"] = "6543"

        try:
            applied = config.apply_environment_overrides()
            assert "DATABASE_HOST" in applied
            assert config.get("database.host") == "env-host"
            assert config.get("database.port") == "6543"
        finally:
            # Clean up
            del os.environ["TEST_DATABASE_HOST"]
            del os.environ["TEST_DATABASE_PORT"]

    def test_apply_environment_overrides_json_values(self, sample_config):
        """Test applying environment overrides with JSON values."""
        config = ConfigurationManager(env_prefix="TEST")
        config._config = sample_config.copy()

        # Set JSON environment variable
        os.environ["TEST_API_WORKERS"] = "8"

        try:
            config.apply_environment_overrides()
            assert config.get("api.workers") == "8"
        finally:
            del os.environ["TEST_API_WORKERS"]

    def test_get_environment_config(self, sample_config):
        """Test getting environment-specific configuration."""
        config = ConfigurationManager(environment="production")
        config._config = sample_config

        env_config = config.get_environment_config()
        assert env_config["api"]["debug"] is False
        assert env_config["database"]["host"] == "prod-db.example.com"

    def test_get_environment_config_missing(self, sample_config):
        """Test getting non-existent environment configuration."""
        config = ConfigurationManager(environment="staging")
        config._config = sample_config

        with pytest.raises(ConfigurationError):
            config.get_environment_config()

    def test_merge_environment_config(self, sample_config):
        """Test merging environment-specific configuration."""
        config = ConfigurationManager(environment="production")
        config._config = sample_config.copy()

        config.merge_environment_config()

        # Check that production settings override base
        assert config.get("api.debug") is False
        assert config.get("database.host") == "prod-db.example.com"
        # Check that other settings remain
        assert config.get("database.port") == 5432

    def test_reset(self, sample_config):
        """Test resetting configuration."""
        config = ConfigurationManager()
        config._config = sample_config.copy()

        config.reset()
        assert config.config == {}
        assert config.config_file is None

    def test_reload(self, yaml_config_file, sample_config):
        """Test reloading configuration."""
        config = ConfigurationManager()
        config.load_configuration(yaml_config_file)

        # Modify config
        config.set("new.key", "value")
        assert config.get("new.key") == "value"

        # Reload
        config.reload()
        assert config.get("new.key") is None
        assert config.config == sample_config

    def test_reload_without_loaded_file(self):
        """Test reloading without loaded file."""
        config = ConfigurationManager()
        with pytest.raises(ConfigurationError):
            config.reload()

    def test_get_metadata(self):
        """Test getting configuration metadata."""
        config = ConfigurationManager()
        metadata = config.get_metadata()

        assert "created_at" in metadata
        assert "last_modified" in metadata
        assert "last_loaded" in metadata
        assert "version" in metadata

    def test_thread_safety(self, sample_config):
        """Test thread-safe operations."""
        import threading

        config = ConfigurationManager()
        config._config = sample_config.copy()

        results = []

        def worker():
            for i in range(100):
                config.set(f"thread.key{i}", i)
                value = config.get(f"thread.key{i}")
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no race conditions
        assert len(results) == 500


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    def test_validate_type_success(self):
        """Test successful type validation."""
        validator = ConfigValidator()
        assert validator.validate_type("test", 42, int) is True
        assert validator.validate_type("test", "string", str) is True
        assert validator.validate_type("test", True, bool) is True

    def test_validate_type_failure(self):
        """Test failed type validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_type("test", "42", int)

    def test_validate_type_allow_none(self):
        """Test type validation with None allowed."""
        validator = ConfigValidator()
        assert validator.validate_type("test", None, str, allow_none=True) is True

    def test_validate_range_success(self):
        """Test successful range validation."""
        validator = ConfigValidator()
        assert validator.validate_range("port", 8080, min_value=1, max_value=65535) is True

    def test_validate_range_failure_min(self):
        """Test range validation failure (below minimum)."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_range("port", 0, min_value=1)

    def test_validate_range_failure_max(self):
        """Test range validation failure (above maximum)."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_range("port", 70000, max_value=65535)

    def test_validate_choice_success(self):
        """Test successful choice validation."""
        validator = ConfigValidator()
        assert validator.validate_choice("env", "production", ["dev", "staging", "production"]) is True

    def test_validate_choice_failure(self):
        """Test failed choice validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_choice("env", "invalid", ["dev", "staging", "production"])

    def test_validate_choice_case_insensitive(self):
        """Test case-insensitive choice validation."""
        validator = ConfigValidator()
        assert validator.validate_choice(
            "env", "PRODUCTION", ["dev", "staging", "production"], case_sensitive=False
        ) is True

    def test_validate_regex_success(self):
        """Test successful regex validation."""
        validator = ConfigValidator()
        assert validator.validate_regex("email", "test@example.com", r".+@.+\..+") is True

    def test_validate_regex_failure(self):
        """Test failed regex validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_regex("email", "invalid-email", r".+@.+\..+")

    def test_validate_path_exists(self, temp_dir):
        """Test path validation for existing path."""
        test_file = temp_dir / "test.txt"
        test_file.touch()

        validator = ConfigValidator()
        assert validator.validate_path("path", test_file, must_exist=True) is True

    def test_validate_path_not_exists(self):
        """Test path validation for non-existent path."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_path("path", "/nonexistent/path", must_exist=True)

    def test_validate_url_success(self):
        """Test successful URL validation."""
        validator = ConfigValidator()
        assert validator.validate_url("url", "https://example.com/path") is True

    def test_validate_url_failure(self):
        """Test failed URL validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_url("url", "not-a-url")

    def test_validate_url_with_scheme(self):
        """Test URL validation with allowed schemes."""
        validator = ConfigValidator()
        assert validator.validate_url(
            "url", "https://example.com", allowed_schemes=["https"]
        ) is True

        with pytest.raises(ConfigValidationError):
            validator.validate_url(
                "url", "http://example.com", allowed_schemes=["https"]
            )

    def test_validate_email_success(self):
        """Test successful email validation."""
        validator = ConfigValidator()
        assert validator.validate_email("email", "user@example.com") is True

    def test_validate_email_failure(self):
        """Test failed email validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_email("email", "invalid-email")

    def test_validate_non_empty_success(self):
        """Test successful non-empty validation."""
        validator = ConfigValidator()
        assert validator.validate_non_empty("field", "value") is True
        assert validator.validate_non_empty("field", [1, 2, 3]) is True

    def test_validate_non_empty_failure(self):
        """Test failed non-empty validation."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError):
            validator.validate_non_empty("field", "")

        with pytest.raises(ConfigValidationError):
            validator.validate_non_empty("field", [])


class TestSchemaValidator:
    """Test suite for SchemaValidator class."""

    def test_validate_simple_schema(self):
        """Test validation with simple schema."""
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "min": 0, "max": 150},
        }
        validator = SchemaValidator(schema)

        config = {"name": "John", "age": 30}
        assert validator.validate(config) is True

    def test_validate_nested_schema(self):
        """Test validation with nested schema."""
        schema = {
            "database": {
                "host": {"type": str, "required": True},
                "port": {"type": int, "min": 1, "max": 65535},
            }
        }
        validator = SchemaValidator(schema)

        config = {"database": {"host": "localhost", "port": 5432}}
        assert validator.validate(config) is True

    def test_validate_missing_required_field(self):
        """Test validation with missing required field."""
        schema = {
            "name": {"type": str, "required": True},
        }
        validator = SchemaValidator(schema)

        config = {}
        with pytest.raises(ConfigValidationError):
            validator.validate(config)

    def test_validate_invalid_type(self):
        """Test validation with invalid type."""
        schema = {
            "port": {"type": int, "required": True},
        }
        validator = SchemaValidator(schema)

        config = {"port": "not-an-int"}
        with pytest.raises(ConfigValidationError):
            validator.validate(config)


class TestStandaloneFunctions:
    """Test suite for standalone utility functions."""

    def test_settings_loader(self, yaml_config_file, sample_config):
        """Test settings_loader function."""
        loaded = settings_loader(yaml_config_file, apply_env_overrides=False)
        assert loaded == sample_config

    def test_settings_loader_with_environment(self, yaml_config_file):
        """Test settings_loader with environment merge."""
        loaded = settings_loader(
            yaml_config_file,
            environment="production",
            apply_env_overrides=False
        )
        assert loaded["api"]["debug"] is False

    def test_settings_loader_with_env_overrides(self, yaml_config_file):
        """Test settings_loader with environment variable overrides."""
        os.environ["APP_DATABASE_HOST"] = "env-override"

        try:
            loaded = settings_loader(yaml_config_file, apply_env_overrides=True)
            assert loaded["database"]["host"] == "env-override"
        finally:
            del os.environ["APP_DATABASE_HOST"]

    def test_preference_validator_success(self, sample_config, sample_schema):
        """Test preference_validator with valid config."""
        is_valid, errors = preference_validator(sample_config, sample_schema)
        assert is_valid is True
        assert errors == []

    def test_preference_validator_failure(self, sample_schema):
        """Test preference_validator with invalid config."""
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Invalid
            }
        }
        is_valid, errors = preference_validator(
            invalid_config, sample_schema, raise_on_error=False
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_preference_validator_raise_on_error(self, sample_schema):
        """Test preference_validator with raise_on_error."""
        invalid_config = {"database": {"host": "localhost", "port": 99999}}

        with pytest.raises(ConfigValidationError):
            preference_validator(invalid_config, sample_schema, raise_on_error=True)

    def test_environment_config_handler(self, sample_config):
        """Test environment_config_handler function."""
        result = environment_config_handler(sample_config, "production")
        assert result["api"]["debug"] is False
        assert result["database"]["host"] == "prod-db.example.com"

    def test_environment_config_handler_with_env_vars(self, sample_config):
        """Test environment_config_handler with environment variables."""
        os.environ["APP_DATABASE_PORT"] = "6543"

        try:
            result = environment_config_handler(sample_config, "production")
            assert result["database"]["port"] == "6543"
        finally:
            del os.environ["APP_DATABASE_PORT"]

    def test_settings_persistence_yaml(self, temp_dir, sample_config):
        """Test settings_persistence with YAML format."""
        output_file = temp_dir / "output.yaml"
        saved_path = settings_persistence(sample_config, output_file, format="yaml")

        assert saved_path == output_file
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded == sample_config

    def test_settings_persistence_json(self, temp_dir, sample_config):
        """Test settings_persistence with JSON format."""
        output_file = temp_dir / "output.json"
        saved_path = settings_persistence(sample_config, output_file, format="json")

        assert saved_path == output_file
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == sample_config

    def test_settings_persistence_with_validation(self, temp_dir, sample_config, sample_schema):
        """Test settings_persistence with schema validation."""
        output_file = temp_dir / "output.yaml"
        saved_path = settings_persistence(
            sample_config,
            output_file,
            validate_schema=sample_schema
        )
        assert saved_path == output_file

    def test_settings_persistence_validation_failure(self, temp_dir, sample_schema):
        """Test settings_persistence with validation failure."""
        invalid_config = {"database": {"host": "localhost", "port": 99999}}
        output_file = temp_dir / "output.yaml"

        with pytest.raises(ConfigValidationError):
            settings_persistence(
                invalid_config,
                output_file,
                validate_schema=sample_schema
            )


class TestExceptions:
    """Test suite for custom exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Test error", {"key": "value"})
        assert str(error) == "Test error (key=value)"
        assert error.message == "Test error"
        assert error.context == {"key": "value"}

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError exception."""
        error = ConfigFileNotFoundError("/path/to/config.yaml")
        assert error.file_path == "/path/to/config.yaml"
        assert "not found" in str(error)

    def test_config_validation_error(self):
        """Test ConfigValidationError exception."""
        error = ConfigValidationError("port", 99999, "must be <= 65535")
        assert error.field == "port"
        assert error.value == 99999
        assert error.constraint == "must be <= 65535"
        assert "port" in str(error)

    def test_config_persistence_error(self):
        """Test ConfigPersistenceError exception."""
        error = ConfigPersistenceError("save", "/path/to/config.yaml")
        assert error.operation == "save"
        assert error.file_path == "/path/to/config.yaml"
        assert "save" in str(error)

    def test_config_parse_error(self):
        """Test ConfigParseError exception."""
        error = ConfigParseError("/path/to/config.yaml", "yaml", line_number=10)
        assert error.file_path == "/path/to/config.yaml"
        assert error.format == "yaml"
        assert error.line_number == 10
        assert "line 10" in str(error)

    def test_environment_variable_error(self):
        """Test EnvironmentVariableError exception."""
        error = EnvironmentVariableError("DATABASE_URL", "not set")
        assert error.variable_name == "DATABASE_URL"
        assert error.reason == "not set"
        assert "DATABASE_URL" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
