"""
PV Circularity Simulator: End-to-end PV lifecycle simulation platform.

This package provides comprehensive modeling and simulation capabilities for
photovoltaic systems throughout their entire lifecycle, from design and manufacturing
to operation, performance monitoring, and end-of-life circularity analysis.

The simulator covers:
- Cell design and analysis
- Module engineering and optimization
- System planning and configuration
- Performance monitoring and forecasting
- Circular economy (3R: Recycling, Refurbishment, Reuse)
- Technical loss analysis (CTM, optical, thermal, etc.)
- Reliability testing frameworks
- Financial analysis and cost modeling

Author: ganeshgowri-ASA
License: MIT
Version: 0.1.0
"""

from pv_circularity_simulator.models import (
    CellModel,
    FinancialModel,
    MaterialModel,
    ModuleModel,
    PerformanceModel,
    SystemModel,
)

__version__ = "0.1.0"
__author__ = "ganeshgowri-ASA"
__license__ = "MIT"

__all__ = [
    "MaterialModel",
    "CellModel",
    "ModuleModel",
    "SystemModel",
    "PerformanceModel",
    "FinancialModel",
]
