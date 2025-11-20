"""
Custom exceptions for PV circularity simulator.
"""


class DiagnosticError(Exception):
    """Base exception for all diagnostic-related errors."""

    pass


class InvalidThermalDataError(DiagnosticError):
    """Raised when thermal image data is invalid or corrupted."""

    pass


class InvalidIVCurveError(DiagnosticError):
    """Raised when IV curve data is invalid or does not meet requirements."""

    pass


class CalibrationError(DiagnosticError):
    """Raised when calibration fails or produces invalid results."""

    pass


class AnalysisError(DiagnosticError):
    """Raised when analysis cannot be completed."""

    pass


class InsufficientDataError(DiagnosticError):
    """Raised when insufficient data is provided for analysis."""

    pass


class ModelFittingError(DiagnosticError):
    """Raised when curve fitting or model optimization fails."""

    pass
