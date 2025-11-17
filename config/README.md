# Configuration Management System

## Overview

The PV Circularity Simulator uses a comprehensive configuration management system that supports:

- **Multiple file formats**: YAML and JSON
- **Environment variables**: Override configuration with environment variables
- **Validation**: Schema-based configuration validation
- **Multi-environment support**: Development, staging, production configurations
- **Persistence**: Save and reload configurations
- **Thread-safe operations**: Safe for concurrent access

## Quick Start

### Basic Usage

```python
from pv_circularity_simulator.config import ConfigurationManager

# Load configuration
config = ConfigurationManager()
config.load_configuration('config/app.yaml')

# Get configuration values
db_host = config.get('database.host')
api_port = config.get('api.port', default=8080)

# Set configuration values
config.set('database.pool_size', 20)

# Save configuration
config.save_configuration()
```

### Environment Variables

Override any configuration value using environment variables with the `APP_` prefix:

```bash
# Override database host
export APP_DATABASE_HOST=prod-db.example.com

# Override API port
export APP_DATABASE_PORT=5433

# Override nested values using underscores
export APP_API_RATE_LIMIT_REQUESTS_PER_MINUTE=200
```

```python
# Apply environment variable overrides
config = ConfigurationManager()
config.load_configuration('config/app.yaml')
config.apply_environment_overrides()

# Database host is now 'prod-db.example.com'
print(config.get('database.host'))
```

### Environment-Specific Configuration

```python
# Load base configuration
config = ConfigurationManager(environment='production')
config.load_configuration('config/app.yaml')

# Merge production-specific settings
config.merge_environment_config()

# Production settings override base settings
assert config.get('api.debug') == False
```

## Configuration File Structure

### YAML Example

```yaml
database:
  host: localhost
  port: 5432
  username: user
  password: pass

api:
  host: 0.0.0.0
  port: 8080
  debug: true

environments:
  development:
    api:
      debug: true
  production:
    api:
      debug: false
    database:
      host: prod-db.example.com
```

### JSON Example

```json
{
  "database": {
    "host": "localhost",
    "port": 5432
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8080
  }
}
```

## Validation

### Define a Schema

```python
schema = {
    "database": {
        "host": {"type": str, "required": True},
        "port": {"type": int, "min": 1, "max": 65535},
    },
    "api": {
        "port": {"type": int, "min": 1, "max": 65535},
        "debug": {"type": bool},
    }
}

config = ConfigurationManager(schema=schema)
config.load_configuration('config/app.yaml')
config.validate()  # Raises ConfigValidationError if invalid
```

### Validation Rules

Available validation rules:

- `type`: Python type (str, int, float, bool, dict, list)
- `required`: Whether the field is required
- `min`: Minimum value (for numbers)
- `max`: Maximum value (for numbers)
- `choices`: List of allowed values
- `pattern`: Regular expression pattern
- `non_empty`: Value must not be empty
- `is_path`: Validate as file system path
- `is_url`: Validate as URL
- `is_email`: Validate as email address

## Standalone Functions

### settings_loader()

Load configuration with environment overrides:

```python
from pv_circularity_simulator.config import settings_loader

config = settings_loader(
    'config/app.yaml',
    environment='production',
    apply_env_overrides=True
)
```

### preference_validator()

Validate configuration against a schema:

```python
from pv_circularity_simulator.config import preference_validator

schema = {...}
config = {...}

is_valid, errors = preference_validator(
    config,
    schema,
    raise_on_error=False
)

if not is_valid:
    print("Validation errors:", errors)
```

### environment_config_handler()

Handle environment-specific configuration:

```python
from pv_circularity_simulator.config import environment_config_handler

base_config = {...}
config = environment_config_handler(
    base_config,
    environment='production',
    env_prefix='APP'
)
```

### settings_persistence()

Save configuration to file:

```python
from pv_circularity_simulator.config import settings_persistence

config = {...}
saved_path = settings_persistence(
    config,
    'config/app.yaml',
    format='yaml',
    create_backup=True
)
```

## Advanced Usage

### Thread-Safe Operations

All ConfigurationManager methods are thread-safe:

```python
import threading

config = ConfigurationManager()
config.load_configuration('config/app.yaml')

def worker():
    value = config.get('database.host')
    config.set('worker.status', 'running')

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Configuration Merging

Merge multiple configuration sources:

```python
# Load base configuration
config = ConfigurationManager()
config.load_configuration('config/base.yaml')

# Merge additional configurations
config.load_configuration('config/overrides.yaml', merge=True)

# Apply environment-specific settings
config.merge_environment_config('production')

# Apply environment variable overrides
config.apply_environment_overrides()
```

### Configuration Metadata

Track configuration changes:

```python
config = ConfigurationManager()
config.load_configuration('config/app.yaml')

metadata = config.get_metadata()
print(f"Created: {metadata['created_at']}")
print(f"Last modified: {metadata['last_modified']}")
print(f"Last loaded: {metadata['last_loaded']}")
```

### Reloading Configuration

Reload configuration from disk:

```python
config = ConfigurationManager()
config.load_configuration('config/app.yaml')

# Configuration file is modified externally...

# Reload to get latest changes
config.reload()
```

## Error Handling

The configuration system uses specific exceptions:

```python
from pv_circularity_simulator.config.exceptions import (
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigPersistenceError,
    ConfigParseError,
)

try:
    config = ConfigurationManager()
    config.load_configuration('config/app.yaml')
    config.validate()
except ConfigFileNotFoundError as e:
    print(f"Configuration file not found: {e.file_path}")
except ConfigParseError as e:
    print(f"Failed to parse {e.format} file: {e.file_path}")
except ConfigValidationError as e:
    print(f"Validation failed for {e.field}: {e.constraint}")
```

## Best Practices

1. **Use environment variables for secrets**: Never commit passwords or API keys to configuration files
   ```yaml
   database:
     password: ${DATABASE_PASSWORD}  # Set via environment variable
   ```

2. **Validate on load**: Enable auto-validation in production
   ```python
   config = ConfigurationManager(schema=schema, auto_validate=True)
   ```

3. **Use environment-specific configs**: Separate development, staging, and production settings
   ```yaml
   environments:
     development: {...}
     production: {...}
   ```

4. **Create backups**: Enable backup creation when saving
   ```python
   config.save_configuration(create_backup=True)
   ```

5. **Use type hints**: Get values with type validation
   ```python
   port = config.get('api.port', value_type=int)
   ```

## Example Configuration Files

See the `config/examples/` directory for:

- `app_config.yaml` - Complete example YAML configuration
- `app_config.json` - Complete example JSON configuration
- `config_schema.yaml` - Example validation schema

## Testing

Run the test suite:

```bash
pytest tests/test_configuration_manager.py -v
```

## API Reference

### ConfigurationManager

- `__init__(config_file, schema, environment, env_prefix, auto_load, auto_validate)`
- `load_configuration(config_file, merge, validate)`
- `save_configuration(config_file, format, create_backup)`
- `get(key, default, required, value_type)`
- `set(key, value, validate, create_path)`
- `delete(key, ignore_missing)`
- `validate(config)`
- `apply_environment_overrides(prefix, separator, lowercase)`
- `get_environment_config(environment, config_key)`
- `merge_environment_config(environment, config_key)`
- `reset()`
- `reload(validate)`
- `get_metadata()`

### Properties

- `config` - Current configuration (read-only)
- `schema` - Validation schema
- `config_file` - Path to loaded configuration file
- `environment` - Current environment name
- `env_prefix` - Environment variable prefix

## Support

For issues and questions, please refer to the main project documentation or open an issue on GitHub.
