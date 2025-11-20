"""
Custom exceptions for PV Circularity Simulator.

This module defines the exception hierarchy for the entire application,
providing specific error types for different subsystems.
"""

from typing import Optional, Any, Dict


class PVCircularityError(Exception):
    """
    Base exception for all PV Circularity Simulator errors.

    All custom exceptions in the application should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the base exception.

        Args:
            message: Human-readable error message
            details: Additional context about the error
            original_exception: The original exception if this is wrapping another error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a formatted error message."""
        base_msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} ({details_str})"
        if self.original_exception:
            base_msg = f"{base_msg} | Caused by: {str(self.original_exception)}"
        return base_msg


class ConfigurationError(PVCircularityError):
    """
    Raised when there are configuration-related errors.

    Examples:
        - Missing required environment variables
        - Invalid configuration values
        - Configuration file parsing errors
    """

    pass


class DatabaseError(PVCircularityError):
    """
    Raised when database operations fail.

    Examples:
        - Connection failures
        - Query execution errors
        - Transaction failures
        - Migration errors
    """

    pass


class SCADAConnectionError(PVCircularityError):
    """
    Raised when SCADA/protocol communication fails.

    Examples:
        - Modbus connection timeout
        - OPC UA authentication failure
        - MQTT broker unreachable
        - Protocol-specific errors
    """

    def __init__(
        self,
        message: str,
        protocol: Optional[str] = None,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SCADA connection error.

        Args:
            message: Error message
            protocol: Protocol name (e.g., 'modbus', 'opcua', 'mqtt')
            device_id: Device identifier
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if protocol:
            details["protocol"] = protocol
        if device_id:
            details["device_id"] = device_id
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class DataValidationError(PVCircularityError):
    """
    Raised when data validation fails.

    Examples:
        - Invalid data format
        - Missing required fields
        - Data outside expected range
        - Type mismatch
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize data validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            value: The invalid value
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class MonitoringError(PVCircularityError):
    """
    Raised when real-time monitoring operations fail.

    Examples:
        - Stream processing errors
        - Alert generation failures
        - Data aggregation errors
        - Performance calculation failures
    """

    pass


class DataLoggerError(PVCircularityError):
    """
    Raised when data logging operations fail.

    Examples:
        - Buffer overflow
        - File write errors
        - Data formatting errors
        - Storage system unavailable
    """

    pass


class ProtocolError(PVCircularityError):
    """
    Raised when protocol-specific operations fail.

    Examples:
        - Unsupported protocol version
        - Invalid protocol message
        - Protocol handshake failure
    """

    def __init__(
        self,
        message: str,
        protocol: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize protocol error.

        Args:
            message: Error message
            protocol: Protocol name
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if protocol:
            details["protocol"] = protocol
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class DeviceNotFoundError(PVCircularityError):
    """
    Raised when a device cannot be found or accessed.

    Examples:
        - Device ID not in configuration
        - Device offline
        - Device not responding
    """

    def __init__(
        self,
        message: str,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize device not found error.

        Args:
            message: Error message
            device_id: Device identifier
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if device_id:
            details["device_id"] = device_id
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AlertError(PVCircularityError):
    """
    Raised when alert processing fails.

    Examples:
        - Alert rule evaluation error
        - Notification delivery failure
        - Alert history storage error
    """

    pass


class AggregationError(PVCircularityError):
    """
    Raised when data aggregation operations fail.

    Examples:
        - Incompatible data sources
        - Timestamp alignment failure
        - Missing required data for aggregation
    """

    pass
