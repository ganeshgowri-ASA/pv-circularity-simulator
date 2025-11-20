"""
PV Circularity Simulator

A comprehensive photovoltaic system simulator with AI-powered defect detection
and intelligent alerting capabilities.
"""

__version__ = "0.1.0"
__author__ = "PV Circularity Team"

from pv_simulator.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
