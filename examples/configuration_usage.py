"""
Example usage of the Configuration Management System.

This script demonstrates various ways to use the ConfigurationManager
for the PV Circularity Simulator project.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pv_circularity_simulator.config import (
    ConfigurationManager,
    settings_loader,
    preference_validator,
    environment_config_handler,
    settings_persistence,
)
from pv_circularity_simulator.config.exceptions import (
    ConfigValidationError,
    ConfigFileNotFoundError,
)


def example_basic_usage():
    """Example 1: Basic configuration loading and access."""
    print("=" * 60)
    print("Example 1: Basic Configuration Usage")
    print("=" * 60)

    # Create configuration manager
    config = ConfigurationManager()

    # Load configuration from YAML file
    config.load_configuration('config/examples/app_config.yaml')

    # Get configuration values using dot notation
    db_host = config.get('database.host')
    db_port = config.get('database.port')
    api_debug = config.get('api.debug')

    print(f"Database Host: {db_host}")
    print(f"Database Port: {db_port}")
    print(f"API Debug Mode: {api_debug}")

    # Get with default value
    timeout = config.get('api.timeout', default=30)
    print(f"API Timeout: {timeout}")

    # Set a new value
    config.set('api.workers', 8)
    print(f"Updated Workers: {config.get('api.workers')}")
    print()


def example_environment_specific():
    """Example 2: Environment-specific configuration."""
    print("=" * 60)
    print("Example 2: Environment-Specific Configuration")
    print("=" * 60)

    # Load for production environment
    config = ConfigurationManager(environment='production')
    config.load_configuration('config/examples/app_config.yaml')

    # Merge production-specific settings
    config.merge_environment_config()

    print(f"Environment: {config.environment}")
    print(f"API Debug (Production): {config.get('api.debug')}")
    print(f"Database Host (Production): {config.get('database.host')}")
    print(f"API Workers (Production): {config.get('api.workers')}")
    print()


def example_environment_variables():
    """Example 3: Environment variable overrides."""
    print("=" * 60)
    print("Example 3: Environment Variable Overrides")
    print("=" * 60)

    import os

    # Set some environment variables
    os.environ['APP_DATABASE_HOST'] = 'env-db.example.com'
    os.environ['APP_DATABASE_PORT'] = '6543'
    os.environ['APP_API_WORKERS'] = '16'

    # Load configuration
    config = ConfigurationManager(env_prefix='APP')
    config.load_configuration('config/examples/app_config.yaml')

    print("Before environment overrides:")
    print(f"  Database Host: {config.get('database.host')}")
    print(f"  Database Port: {config.get('database.port')}")

    # Apply environment variable overrides
    applied = config.apply_environment_overrides()

    print("\nApplied environment variables:")
    for key, value in applied.items():
        print(f"  {key} = {value}")

    print("\nAfter environment overrides:")
    print(f"  Database Host: {config.get('database.host')}")
    print(f"  Database Port: {config.get('database.port')}")
    print(f"  API Workers: {config.get('api.workers')}")

    # Clean up
    del os.environ['APP_DATABASE_HOST']
    del os.environ['APP_DATABASE_PORT']
    del os.environ['APP_API_WORKERS']
    print()


def example_validation():
    """Example 4: Configuration validation."""
    print("=" * 60)
    print("Example 4: Configuration Validation")
    print("=" * 60)

    # Define validation schema
    schema = {
        "database": {
            "host": {"type": str, "required": True, "non_empty": True},
            "port": {"type": int, "required": True, "min": 1, "max": 65535},
            "username": {"type": str, "required": True},
        },
        "api": {
            "port": {"type": int, "required": True, "min": 1, "max": 65535},
            "workers": {"type": int, "min": 1, "max": 128},
            "debug": {"type": bool},
        },
        "logging": {
            "level": {
                "type": str,
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
        },
    }

    # Create configuration manager with schema
    config = ConfigurationManager(schema=schema, auto_validate=True)

    try:
        # Load and validate configuration
        config.load_configuration('config/examples/app_config.yaml')
        print("✓ Configuration is valid!")

        # Try to set an invalid value
        try:
            config.set('database.port', 99999)  # Exceeds max value
        except ConfigValidationError as e:
            print(f"\n✗ Validation failed: {e.field}")
            print(f"  Constraint: {e.constraint}")
            print(f"  Value: {e.value}")

    except ConfigValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
    print()


def example_saving_configuration():
    """Example 5: Saving and persisting configuration."""
    print("=" * 60)
    print("Example 5: Saving Configuration")
    print("=" * 60)

    import tempfile

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load configuration
        config = ConfigurationManager()
        config.load_configuration('config/examples/app_config.yaml')

        # Modify some values
        config.set('database.pool_size', 30)
        config.set('api.workers', 12)
        config.set('custom.setting', 'new_value')

        # Save to YAML
        yaml_path = Path(tmpdir) / 'updated_config.yaml'
        config.save_configuration(yaml_path, format='yaml')
        print(f"✓ Saved to YAML: {yaml_path}")

        # Save to JSON
        json_path = Path(tmpdir) / 'updated_config.json'
        config.save_configuration(json_path, format='json')
        print(f"✓ Saved to JSON: {json_path}")

        # Verify by reloading
        config2 = ConfigurationManager()
        config2.load_configuration(yaml_path)
        print(f"\nVerification:")
        print(f"  Pool Size: {config2.get('database.pool_size')}")
        print(f"  API Workers: {config2.get('api.workers')}")
        print(f"  Custom Setting: {config2.get('custom.setting')}")
    print()


def example_standalone_functions():
    """Example 6: Using standalone utility functions."""
    print("=" * 60)
    print("Example 6: Standalone Utility Functions")
    print("=" * 60)

    # settings_loader - Load configuration with environment overrides
    config = settings_loader(
        'config/examples/app_config.yaml',
        environment='production',
        apply_env_overrides=False
    )
    print(f"Loaded config for production:")
    print(f"  API Debug: {config['api']['debug']}")

    # preference_validator - Validate configuration
    schema = {
        "database": {
            "host": {"type": str, "required": True},
            "port": {"type": int, "min": 1, "max": 65535},
        }
    }

    is_valid, errors = preference_validator(
        config,
        schema,
        raise_on_error=False
    )
    print(f"\nValidation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        print(f"Errors: {errors}")

    # environment_config_handler - Handle environment configuration
    base_config = {
        'api': {'debug': True, 'port': 8080},
        'environments': {
            'production': {'api': {'debug': False}}
        }
    }

    prod_config = environment_config_handler(base_config, 'production')
    print(f"\nProduction config debug mode: {prod_config['api']['debug']}")

    # settings_persistence - Save configuration
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'saved_config.yaml'
        saved = settings_persistence(
            config,
            output_path,
            format='yaml',
            create_backup=False
        )
        print(f"\n✓ Configuration persisted to: {saved}")
    print()


def example_advanced_features():
    """Example 7: Advanced features."""
    print("=" * 60)
    print("Example 7: Advanced Features")
    print("=" * 60)

    config = ConfigurationManager()
    config.load_configuration('config/examples/app_config.yaml')

    # Get metadata
    metadata = config.get_metadata()
    print("Configuration Metadata:")
    print(f"  Created: {metadata['created_at']}")
    print(f"  Version: {metadata['version']}")

    # Delete a key
    config.delete('simulation.cell_design.degradation_rate')
    print("\n✓ Deleted: simulation.cell_design.degradation_rate")

    # Reset configuration
    print(f"\nConfig before reset: {len(config.config)} top-level keys")
    config.reset()
    print(f"Config after reset: {len(config.config)} top-level keys")

    # Reload from file
    config.load_configuration('config/examples/app_config.yaml')
    print(f"Config after reload: {len(config.config)} top-level keys")
    print()


def example_error_handling():
    """Example 8: Error handling."""
    print("=" * 60)
    print("Example 8: Error Handling")
    print("=" * 60)

    # Try to load non-existent file
    try:
        config = ConfigurationManager()
        config.load_configuration('nonexistent.yaml')
    except ConfigFileNotFoundError as e:
        print(f"✓ Caught ConfigFileNotFoundError:")
        print(f"  File: {e.file_path}")

    # Try to get required but missing key
    try:
        config = ConfigurationManager()
        config.load_configuration('config/examples/app_config.yaml')
        value = config.get('nonexistent.key', required=True)
    except Exception as e:
        print(f"\n✓ Caught ConfigurationError:")
        print(f"  {str(e)}")

    # Try to validate invalid configuration
    try:
        schema = {
            "database": {
                "port": {"type": int, "min": 1, "max": 1000},
            }
        }
        config = ConfigurationManager(schema=schema)
        config.load_configuration('config/examples/app_config.yaml')
        config.validate()
    except ConfigValidationError as e:
        print(f"\n✓ Caught ConfigValidationError:")
        print(f"  Field: {e.field}")
        print(f"  Constraint: {e.constraint}")
        print(f"  Value: {e.value}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("PV Circularity Simulator - Configuration Management Examples")
    print("=" * 60)
    print()

    try:
        example_basic_usage()
        example_environment_specific()
        example_environment_variables()
        example_validation()
        example_saving_configuration()
        example_standalone_functions()
        example_advanced_features()
        example_error_handling()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print()

    except FileNotFoundError:
        print("\nNote: Some examples require config files in 'config/examples/'")
        print("Please ensure the example configuration files exist.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
