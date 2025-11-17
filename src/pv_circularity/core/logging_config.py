"""
Structured logging configuration using structlog.

This module sets up application-wide logging with support for both
JSON and console output formats.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import EventDict, Processor

from .config import settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add application context to log events.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Modified event dictionary with application context
    """
    event_dict["app"] = "pv-circularity-simulator"
    event_dict["environment"] = settings.environment
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add log level to event dictionary.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Modified event dictionary with log level
    """
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Censor sensitive data from logs.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Modified event dictionary with censored data
    """
    sensitive_keys = {
        "password",
        "token",
        "api_key",
        "secret",
        "credential",
        "auth",
        "authorization",
    }

    def _censor_dict(d: dict) -> dict:
        """Recursively censor dictionary values."""
        censored = {}
        for key, value in d.items():
            key_lower = key.lower()
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                censored[key] = "***REDACTED***"
            elif isinstance(value, dict):
                censored[key] = _censor_dict(value)
            else:
                censored[key] = value
        return censored

    return _censor_dict(event_dict)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Set up application logging with structlog.

    This function configures structured logging for the entire application,
    supporting both JSON and console output formats with automatic log rotation.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, uses value from settings.
        log_format: Log format ('json' or 'console').
                   If None, uses value from settings.
        log_file: Path to log file. If None, uses value from settings.

    Example:
        >>> setup_logging(log_level="INFO", log_format="console")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    # Use settings if not provided
    log_level = log_level or settings.logging.level
    log_format = log_format or settings.logging.format
    log_file = log_file or settings.logging.file_path

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Shared processors for all output formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_log_level,
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        censor_sensitive_data,
    ]

    # Configure output format
    if log_format == "json":
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:  # console format
        processors = shared_processors + [
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer(
                colors=True if sys.stdout.isatty() else False,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)

        # Always use JSON format for file logs
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("asyncua").setLevel(logging.WARNING)
    logging.getLogger("pymodbus").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        A configured structlog logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data", records_count=1000)
        >>> logger.error("Failed to connect", device_id="INV001", error="timeout")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    This is useful for adding request IDs, user IDs, or other contextual
    information that should appear in all logs within a certain scope.

    Args:
        **kwargs: Key-value pairs to bind to the logging context

    Example:
        >>> bind_context(request_id="123", user_id="user456")
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing request")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*args: str) -> None:
    """
    Remove previously bound context variables.

    Args:
        *args: Names of context variables to unbind

    Example:
        >>> unbind_context("request_id", "user_id")
    """
    structlog.contextvars.unbind_contextvars(*args)


def clear_context() -> None:
    """
    Clear all bound context variables.

    Example:
        >>> clear_context()
    """
    structlog.contextvars.clear_contextvars()


# Initialize logging on module import
setup_logging()
