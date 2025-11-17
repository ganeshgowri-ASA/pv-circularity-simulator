"""
Dashboard module for visualization and analytics.
"""

from pv_simulator.dashboard.forecast_dashboard import (
    ForecastDashboard,
    accuracy_metrics,
    mae_rmse_calculation,
    confidence_intervals,
)

__all__ = [
    "ForecastDashboard",
    "accuracy_metrics",
    "mae_rmse_calculation",
    "confidence_intervals",
]
