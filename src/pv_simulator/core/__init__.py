"""
Core components for PV Circularity Simulator.

This module provides base classes and data models used throughout the platform.
"""

from pv_simulator.core.base_integrator import BaseIntegrator, IntegratorMetadata
from pv_simulator.core.models import (
    WindResourceData,
    WindResourceAssessment,
    TurbineSpecifications,
    TurbinePerformance,
    PVSystemConfig,
    HybridSystemConfig,
    HybridOptimizationResult,
    CoordinationStrategy,
    CoordinationResult,
    TurbineType,
    OptimizationObjective,
)

__all__ = [
    "BaseIntegrator",
    "IntegratorMetadata",
    "WindResourceData",
    "WindResourceAssessment",
    "TurbineSpecifications",
    "TurbinePerformance",
    "PVSystemConfig",
    "HybridSystemConfig",
    "HybridOptimizationResult",
    "CoordinationStrategy",
    "CoordinationResult",
    "TurbineType",
    "OptimizationObjective",
]
