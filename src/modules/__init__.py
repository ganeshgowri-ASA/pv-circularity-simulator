"""
PV Circularity Simulator - Modules Package
===========================================

This package contains all the core modules for the PV Circularity Simulator platform.

Available Modules:
- dashboard: Main dashboard and project management interface

Author: PV Circularity Simulator Team
Version: 1.0.0
"""

from .dashboard import (
    render_dashboard,
    calculate_completion,
    run_full_simulation,
    generate_comprehensive_report,
    export_all_data,
    display_recent_activity,
    log_activity
)

__all__ = [
    'render_dashboard',
    'calculate_completion',
    'run_full_simulation',
    'generate_comprehensive_report',
    'export_all_data',
    'display_recent_activity',
    'log_activity'
]

__version__ = '1.0.0'
