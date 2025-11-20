"""Custom exceptions for the PV Circularity Simulator."""

from typing import Any, Optional


class PVSimulatorError(Exception):
    """Base exception for all PV Simulator errors.

    This is the base exception class from which all other custom exceptions
    in the PV Simulator inherit.

    Attributes:
        message: Human-readable error description.
        details: Additional error details as a dictionary.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class FinancialAnalysisError(PVSimulatorError):
    """Exception raised for errors during financial analysis.

    This exception is raised when financial calculations encounter invalid
    conditions or fail to produce valid results.
    """
    pass


class InvalidCashFlowError(FinancialAnalysisError):
    """Exception raised for invalid cash flow data.

    This exception is raised when:
    - Cash flow arrays are empty or have invalid dimensions
    - Cash flow values are invalid (e.g., all zeros)
    - Periods and cash flows have mismatched lengths
    - Initial investment values are invalid
    """
    pass


class ConvergenceError(FinancialAnalysisError):
    """Exception raised when iterative calculations fail to converge.

    This exception is typically raised during IRR calculations when:
    - The Newton-Raphson method fails to converge
    - Maximum iterations are exceeded
    - No valid IRR exists for the given cash flows
    """

    def __init__(
        self,
        message: str,
        iterations: Optional[int] = None,
        last_value: Optional[float] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the convergence error.

        Args:
            message: Human-readable error description.
            iterations: Number of iterations attempted before failure.
            last_value: Last computed value before convergence failed.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        if iterations is not None:
            details["iterations"] = iterations
        if last_value is not None:
            details["last_value"] = last_value
        super().__init__(message, details)
