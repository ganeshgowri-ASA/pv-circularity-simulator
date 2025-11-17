"""
Custom exceptions for the configuration management system.

This module defines all custom exceptions used throughout the configuration
management package, providing clear error messages and exception hierarchies
for different types of configuration-related errors.
"""

from typing import Optional, Any


class ConfigurationError(Exception):
    """
    Base exception for all configuration-related errors.

    This is the parent class for all configuration management exceptions,
    allowing for broad exception catching when needed.

    Attributes:
        message: Human-readable error message
        context: Optional dictionary containing additional error context
    """

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        """
        Initialize a ConfigurationError.

        Args:
            message: Human-readable error message
            context: Optional dictionary containing additional error context
                    (e.g., file paths, configuration keys, etc.)
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the error with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ConfigFileNotFoundError(ConfigurationError):
    """
    Raised when a configuration file cannot be found.

    This exception is raised when attempting to load a configuration file
    that does not exist at the specified path.

    Attributes:
        file_path: Path to the missing configuration file
    """

    def __init__(self, file_path: str, message: Optional[str] = None):
        """
        Initialize a ConfigFileNotFoundError.

        Args:
            file_path: Path to the missing configuration file
            message: Optional custom error message
        """
        self.file_path = file_path
        default_message = f"Configuration file not found: {file_path}"
        super().__init__(message or default_message, {"file_path": file_path})


class ConfigValidationError(ConfigurationError):
    """
    Raised when configuration validation fails.

    This exception is raised when configuration values do not meet
    validation requirements, such as type mismatches, value constraints,
    or schema violations.

    Attributes:
        field: The configuration field that failed validation
        value: The invalid value
        constraint: Description of the validation constraint that was violated
    """

    def __init__(
        self,
        field: str,
        value: Any,
        constraint: str,
        message: Optional[str] = None
    ):
        """
        Initialize a ConfigValidationError.

        Args:
            field: The configuration field that failed validation
            value: The invalid value
            constraint: Description of the validation constraint
            message: Optional custom error message
        """
        self.field = field
        self.value = value
        self.constraint = constraint
        default_message = (
            f"Validation failed for field '{field}': {constraint}. "
            f"Received value: {value}"
        )
        super().__init__(
            message or default_message,
            {"field": field, "value": value, "constraint": constraint}
        )


class ConfigPersistenceError(ConfigurationError):
    """
    Raised when configuration persistence operations fail.

    This exception is raised when there are errors saving or persisting
    configuration data to disk, such as permission errors or I/O failures.

    Attributes:
        operation: The persistence operation that failed (e.g., 'save', 'write')
        file_path: Path where the configuration was being persisted
    """

    def __init__(
        self,
        operation: str,
        file_path: str,
        message: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a ConfigPersistenceError.

        Args:
            operation: The persistence operation that failed
            file_path: Path where the configuration was being persisted
            message: Optional custom error message
            original_error: Optional original exception that caused this error
        """
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error

        default_message = (
            f"Configuration persistence failed during {operation} "
            f"at {file_path}"
        )
        if original_error:
            default_message += f": {str(original_error)}"

        super().__init__(
            message or default_message,
            {"operation": operation, "file_path": file_path}
        )


class ConfigParseError(ConfigurationError):
    """
    Raised when configuration file parsing fails.

    This exception is raised when a configuration file cannot be parsed,
    typically due to invalid YAML/JSON syntax or unsupported file formats.

    Attributes:
        file_path: Path to the configuration file
        format: The expected file format (e.g., 'yaml', 'json')
        line_number: Optional line number where the parse error occurred
    """

    def __init__(
        self,
        file_path: str,
        format: str,
        message: Optional[str] = None,
        line_number: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a ConfigParseError.

        Args:
            file_path: Path to the configuration file
            format: The expected file format
            message: Optional custom error message
            line_number: Optional line number where the error occurred
            original_error: Optional original exception that caused this error
        """
        self.file_path = file_path
        self.format = format
        self.line_number = line_number
        self.original_error = original_error

        default_message = f"Failed to parse {format.upper()} file: {file_path}"
        if line_number:
            default_message += f" (line {line_number})"
        if original_error:
            default_message += f": {str(original_error)}"

        context = {"file_path": file_path, "format": format}
        if line_number:
            context["line_number"] = line_number

        super().__init__(message or default_message, context)


class EnvironmentVariableError(ConfigurationError):
    """
    Raised when required environment variables are missing or invalid.

    This exception is raised when required environment variables are not set
    or contain invalid values.

    Attributes:
        variable_name: Name of the environment variable
        reason: Reason why the environment variable is invalid
    """

    def __init__(
        self,
        variable_name: str,
        reason: str = "not set",
        message: Optional[str] = None
    ):
        """
        Initialize an EnvironmentVariableError.

        Args:
            variable_name: Name of the environment variable
            reason: Reason why the environment variable is invalid
            message: Optional custom error message
        """
        self.variable_name = variable_name
        self.reason = reason

        default_message = (
            f"Environment variable '{variable_name}' {reason}"
        )

        super().__init__(
            message or default_message,
            {"variable_name": variable_name, "reason": reason}
        )
