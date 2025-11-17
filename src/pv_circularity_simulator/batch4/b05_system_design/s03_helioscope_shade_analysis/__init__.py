"""
S03: Helioscope Integration & Advanced Shade Analysis

This module provides comprehensive 3D shade analysis and system design tools,
including terrain modeling, horizon profiling, near/far shading analysis,
and advanced electrical modeling for accurate energy yield predictions.
"""

from .models import (
    SiteModel,
    ShadeAnalysisConfig,
    HorizonProfile,
    SunPosition,
    IrradianceComponents,
    ShadeAnalysisResult,
    ElectricalShadeResult,
    LayoutOptimizationResult,
)
from .sun_position import SunPositionCalculator
from .irradiance import IrradianceOnSurface
from .helioscope_model import HelioscapeModel
from .horizon_profiler import HorizonProfiler
from .shade_analysis import ShadeAnalysisEngine
from .electrical_shading import ElectricalShadingModel
from .layout_optimizer import SystemLayoutOptimizer

__all__ = [
    # Models
    "SiteModel",
    "ShadeAnalysisConfig",
    "HorizonProfile",
    "SunPosition",
    "IrradianceComponents",
    "ShadeAnalysisResult",
    "ElectricalShadeResult",
    "LayoutOptimizationResult",
    # Classes
    "SunPositionCalculator",
    "IrradianceOnSurface",
    "HelioscapeModel",
    "HorizonProfiler",
    "ShadeAnalysisEngine",
    "ElectricalShadingModel",
    "SystemLayoutOptimizer",
]
