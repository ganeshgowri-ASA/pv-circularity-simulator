"""
PV Circularity Simulator - End-to-end PV lifecycle simulation platform.

This package provides comprehensive tools for photovoltaic system simulation,
including weather API integration, performance monitoring, and circularity analysis.
"""

__version__ = "0.1.0"
__author__ = "PV Circularity Team"
__license__ = "MIT"

from pv_simulator.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
