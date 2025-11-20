"""
PV Circularity Simulator - End-to-end PV lifecycle simulation platform.

This package provides comprehensive tools for simulating photovoltaic systems
throughout their lifecycle, including cell design, module engineering, system
planning, performance monitoring, and circular economy modeling.
"""

__version__ = "0.1.0"
__author__ = "PV Circularity Simulator Team"

from .config.configuration_manager import ConfigurationManager

__all__ = ["ConfigurationManager"]
