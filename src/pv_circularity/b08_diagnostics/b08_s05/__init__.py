"""
B08-S05: Diagnostics UI & Defect Management Dashboard.

This module provides comprehensive defect database management and
interactive Streamlit dashboards for visualization and analysis.
"""

from .defect_database import DefectDatabase, DatabaseConfig
from .diagnostics_ui import DiagnosticsUI

__all__ = [
    "DefectDatabase",
    "DatabaseConfig",
    "DiagnosticsUI",
]
