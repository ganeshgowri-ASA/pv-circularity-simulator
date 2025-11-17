"""
PV Circularity Simulator - An end-to-end PV lifecycle simulation platform.

This package provides tools for simulating photovoltaic systems with a focus on
circularity, including material flow analysis, reuse/repair/recycling strategies,
and sustainability impact assessment.
"""

__version__ = "0.1.0"
__author__ = "PV Circularity Team"

from .dashboards.circularity_dashboard import CircularityDashboardUI

__all__ = ["CircularityDashboardUI"]
