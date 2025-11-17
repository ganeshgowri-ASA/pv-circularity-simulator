"""Core module containing base classes, exceptions, and common types."""

from pv_simulator.core.base import BaseAnalyzer, AnalysisResult
from pv_simulator.core.exceptions import (
    PVSimulatorError,
    FinancialAnalysisError,
    InvalidCashFlowError,
    ConvergenceError,
)

__all__ = [
    "BaseAnalyzer",
    "AnalysisResult",
    "PVSimulatorError",
    "FinancialAnalysisError",
    "InvalidCashFlowError",
    "ConvergenceError",
]
