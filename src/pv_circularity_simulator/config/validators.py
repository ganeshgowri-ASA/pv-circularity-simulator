"""
Configuration validation utilities.

This module provides comprehensive validation capabilities for configuration
settings, including type checking, value constraints, schema validation,
and custom validation rules.
"""

import re
from typing import Any, Callable, Optional, Union, get_origin, get_args
from pathlib import Path
from .exceptions import ConfigValidationError


class ValidationError(Exception):
    """
    Exception raised during validation operations.

    This is a specific exception for validation failures that provides
    detailed information about what failed validation and why.
    """

    def __init__(self, message: str, field: Optional[str] = None):
        """
        Initialize a ValidationError.

        Args:
            message: Human-readable error message
            field: Optional field name that failed validation
        """
        self.field = field
        super().__init__(message)


class ConfigValidator:
    """
    Comprehensive configuration validator with built-in validation rules.

    This class provides a wide range of validation methods for common
    configuration value types and constraints. It can be extended with
    custom validation rules.

    Examples:
        >>> validator = ConfigValidator()
        >>> validator.validate_type("port", 8080, int)
        True
        >>> validator.validate_range("timeout", 30, min_value=1, max_value=3600)
        True
        >>> validator.validate_choice("environment", "prod", ["dev", "staging", "prod"])
        True
    """

    @staticmethod
    def validate_type(
        field: str,
        value: Any,
        expected_type: type,
        allow_none: bool = False
    ) -> bool:
        """
        Validate that a value matches the expected type.

        Args:
            field: Name of the configuration field
            value: Value to validate
            expected_type: Expected Python type
            allow_none: Whether None is an acceptable value

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If type validation fails
        """
        if value is None and allow_none:
            return True

        if not isinstance(value, expected_type):
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint=f"must be of type {expected_type.__name__}"
            )
        return True

    @staticmethod
    def validate_range(
        field: str,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        inclusive: bool = True
    ) -> bool:
        """
        Validate that a numeric value is within a specified range.

        Args:
            field: Name of the configuration field
            value: Numeric value to validate
            min_value: Minimum acceptable value (optional)
            max_value: Maximum acceptable value (optional)
            inclusive: Whether min/max values are inclusive (default: True)

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If range validation fails
        """
        if min_value is not None:
            if inclusive and value < min_value:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be >= {min_value}"
                )
            elif not inclusive and value <= min_value:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be > {min_value}"
                )

        if max_value is not None:
            if inclusive and value > max_value:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be <= {max_value}"
                )
            elif not inclusive and value >= max_value:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be < {max_value}"
                )

        return True

    @staticmethod
    def validate_choice(
        field: str,
        value: Any,
        choices: list[Any],
        case_sensitive: bool = True
    ) -> bool:
        """
        Validate that a value is one of the allowed choices.

        Args:
            field: Name of the configuration field
            value: Value to validate
            choices: List of allowed values
            case_sensitive: Whether string comparison is case-sensitive

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If choice validation fails
        """
        if not case_sensitive and isinstance(value, str):
            value_lower = value.lower()
            choices_lower = [c.lower() if isinstance(c, str) else c for c in choices]
            if value_lower not in choices_lower:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be one of {choices}"
                )
        else:
            if value not in choices:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"must be one of {choices}"
                )
        return True

    @staticmethod
    def validate_regex(
        field: str,
        value: str,
        pattern: str,
        flags: int = 0
    ) -> bool:
        """
        Validate that a string value matches a regular expression pattern.

        Args:
            field: Name of the configuration field
            value: String value to validate
            pattern: Regular expression pattern
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If regex validation fails
        """
        if not isinstance(value, str):
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="must be a string for regex validation"
            )

        if not re.match(pattern, value, flags):
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint=f"must match pattern: {pattern}"
            )
        return True

    @staticmethod
    def validate_path(
        field: str,
        value: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False
    ) -> bool:
        """
        Validate a file system path.

        Args:
            field: Name of the configuration field
            value: Path value to validate
            must_exist: Whether the path must exist
            must_be_file: Whether the path must be a file
            must_be_dir: Whether the path must be a directory

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If path validation fails
        """
        path = Path(value)

        if must_exist and not path.exists():
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="path must exist"
            )

        if must_be_file and not path.is_file():
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="path must be a file"
            )

        if must_be_dir and not path.is_dir():
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="path must be a directory"
            )

        return True

    @staticmethod
    def validate_url(field: str, value: str, allowed_schemes: Optional[list[str]] = None) -> bool:
        """
        Validate a URL format.

        Args:
            field: Name of the configuration field
            value: URL string to validate
            allowed_schemes: Optional list of allowed URL schemes (e.g., ['http', 'https'])

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If URL validation fails
        """
        url_pattern = re.compile(
            r'^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*):\/\/'
            r'(?P<authority>[^\/?#]+)'
            r'(?P<path>[^?#]*)'
            r'(?:\?(?P<query>[^#]*))?'
            r'(?:#(?P<fragment>.*))?$'
        )

        match = url_pattern.match(value)
        if not match:
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="must be a valid URL"
            )

        if allowed_schemes:
            scheme = match.group('scheme').lower()
            if scheme not in allowed_schemes:
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint=f"URL scheme must be one of {allowed_schemes}"
                )

        return True

    @staticmethod
    def validate_email(field: str, value: str) -> bool:
        """
        Validate an email address format.

        Args:
            field: Name of the configuration field
            value: Email string to validate

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If email validation fails
        """
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )

        if not email_pattern.match(value):
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="must be a valid email address"
            )
        return True

    @staticmethod
    def validate_non_empty(field: str, value: Any) -> bool:
        """
        Validate that a value is not empty.

        Args:
            field: Name of the configuration field
            value: Value to validate

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If the value is empty
        """
        if value is None or value == "" or (hasattr(value, '__len__') and len(value) == 0):
            raise ConfigValidationError(
                field=field,
                value=value,
                constraint="must not be empty"
            )
        return True


class SchemaValidator:
    """
    Schema-based configuration validator.

    This class validates configuration dictionaries against defined schemas,
    ensuring that all required fields are present, types are correct, and
    values meet specified constraints.

    Examples:
        >>> schema = {
        ...     "database": {
        ...         "host": {"type": str, "required": True},
        ...         "port": {"type": int, "required": True, "min": 1, "max": 65535},
        ...     }
        ... }
        >>> validator = SchemaValidator(schema)
        >>> config = {"database": {"host": "localhost", "port": 5432}}
        >>> validator.validate(config)
        True
    """

    def __init__(self, schema: dict[str, Any]):
        """
        Initialize a SchemaValidator with a configuration schema.

        Args:
            schema: Dictionary defining the configuration schema structure
        """
        self.schema = schema
        self.validator = ConfigValidator()

    def validate(self, config: dict[str, Any], path: str = "") -> bool:
        """
        Validate a configuration dictionary against the schema.

        Args:
            config: Configuration dictionary to validate
            path: Current path in the configuration (for nested validation)

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If schema validation fails
        """
        return self._validate_dict(config, self.schema, path)

    def _validate_dict(
        self,
        config: dict[str, Any],
        schema: dict[str, Any],
        path: str
    ) -> bool:
        """
        Recursively validate a dictionary against a schema.

        Args:
            config: Configuration dictionary to validate
            schema: Schema dictionary
            path: Current path in the configuration

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If validation fails
        """
        for key, rules in schema.items():
            field_path = f"{path}.{key}" if path else key

            # Check if field is required
            if isinstance(rules, dict) and rules.get("required", False):
                if key not in config:
                    raise ConfigValidationError(
                        field=field_path,
                        value=None,
                        constraint="field is required"
                    )

            # Skip validation if field is not present and not required
            if key not in config:
                continue

            value = config[key]

            # Handle nested dictionaries
            if isinstance(rules, dict) and "type" not in rules:
                if not isinstance(value, dict):
                    raise ConfigValidationError(
                        field=field_path,
                        value=value,
                        constraint="must be a dictionary"
                    )
                self._validate_dict(value, rules, field_path)
                continue

            # Validate based on rules
            if isinstance(rules, dict):
                self._validate_field(field_path, value, rules)

        return True

    def _validate_field(self, field: str, value: Any, rules: dict[str, Any]) -> bool:
        """
        Validate a single field against its rules.

        Args:
            field: Field name/path
            value: Field value
            rules: Validation rules dictionary

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If validation fails
        """
        # Type validation
        if "type" in rules:
            self.validator.validate_type(field, value, rules["type"])

        # Range validation
        if "min" in rules or "max" in rules:
            self.validator.validate_range(
                field, value,
                min_value=rules.get("min"),
                max_value=rules.get("max")
            )

        # Choice validation
        if "choices" in rules:
            self.validator.validate_choice(field, value, rules["choices"])

        # Regex validation
        if "pattern" in rules:
            self.validator.validate_regex(field, value, rules["pattern"])

        # Path validation
        if rules.get("is_path", False):
            self.validator.validate_path(
                field, value,
                must_exist=rules.get("must_exist", False),
                must_be_file=rules.get("must_be_file", False),
                must_be_dir=rules.get("must_be_dir", False)
            )

        # URL validation
        if rules.get("is_url", False):
            self.validator.validate_url(
                field, value,
                allowed_schemes=rules.get("allowed_schemes")
            )

        # Email validation
        if rules.get("is_email", False):
            self.validator.validate_email(field, value)

        # Non-empty validation
        if rules.get("non_empty", False):
            self.validator.validate_non_empty(field, value)

        # Custom validator function
        if "validator" in rules:
            custom_validator: Callable = rules["validator"]
            if not custom_validator(value):
                raise ConfigValidationError(
                    field=field,
                    value=value,
                    constraint="failed custom validation"
                )

        return True
