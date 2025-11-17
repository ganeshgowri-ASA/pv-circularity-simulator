"""
PV Circularity Simulator - Circular Economy & 3R System for Photovoltaic Modules

This package provides comprehensive tools for analyzing the circular economy aspects
of PV modules, including material recovery, reuse assessment, repair optimization,
recycling economics, life cycle assessment, and visualization dashboards.
"""

from .material_recovery import MaterialRecoveryCalculator
from .reuse_analyzer import ReuseAnalyzer
from .repair_optimizer import RepairOptimizer
from .recycling_economics import RecyclingEconomics
from .lca_analyzer import LCAAnalyzer
from .circularity_ui import CircularityUI

__version__ = "0.1.0"

__all__ = [
    "MaterialRecoveryCalculator",
    "ReuseAnalyzer",
    "RepairOptimizer",
    "RecyclingEconomics",
    "LCAAnalyzer",
    "CircularityUI",
]
