"""
Custom exceptions for the PV Circularity Simulator.

This module defines all custom exception types used throughout the application
for better error handling and debugging.
"""


class PVSimulatorError(Exception):
    """Base exception class for all PV Simulator errors."""

    pass


class CalculationError(PVSimulatorError):
    """Exception raised when a calculation fails or produces invalid results."""

    pass


class ValidationError(PVSimulatorError):
    """Exception raised when input validation fails."""

    pass


class ConvergenceError(CalculationError):
    """Exception raised when iterative calculation fails to converge."""

    def __init__(self, message: str, iterations: int = 0, tolerance: float = 0.0):
        """
        Initialize convergence error.

        Args:
            message: Error message
            iterations: Number of iterations attempted
            tolerance: Tolerance that could not be achieved
        """
        self.iterations = iterations
        self.tolerance = tolerance
        super().__init__(
            f"{message} (iterations: {iterations}, tolerance: {tolerance})"
        )


class InsufficientDataError(PVSimulatorError):
    """Exception raised when insufficient data is provided for calculation."""

    pass


class ConfigurationError(PVSimulatorError):
    """Exception raised when configuration is invalid or missing."""

    pass
