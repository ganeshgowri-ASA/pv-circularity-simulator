"""
Core module containing shared data models, constants, and utilities.
"""

from pv_circularity_simulator.core.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRON_CHARGE,
    STANDARD_IRRADIANCE,
    STANDARD_TEMPERATURE,
    STANDARD_AM,
)
from pv_circularity_simulator.core.exceptions import (
    DiagnosticError,
    InvalidThermalDataError,
    InvalidIVCurveError,
    CalibrationError,
)
from pv_circularity_simulator.core.models import (
    ThermalImageMetadata,
    ThermalImageData,
    IVCurveData,
    ElectricalParameters,
    AnalysisResult,
)

__all__ = [
    # Constants
    "BOLTZMANN_CONSTANT",
    "ELECTRON_CHARGE",
    "STANDARD_IRRADIANCE",
    "STANDARD_TEMPERATURE",
    "STANDARD_AM",
    # Exceptions
    "DiagnosticError",
    "InvalidThermalDataError",
    "InvalidIVCurveError",
    "CalibrationError",
    # Models
    "ThermalImageMetadata",
    "ThermalImageData",
    "IVCurveData",
    "ElectricalParameters",
    "AnalysisResult",
]
