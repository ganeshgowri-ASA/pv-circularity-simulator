"""
B08-S04: Fault Reports & Maintenance Recommendations.

This module provides comprehensive fault reporting, maintenance scheduling,
and work order management for PV systems.
"""

from .fault_report_generator import FaultReportGenerator, CostEstimationConfig
from .maintenance_scheduler import MaintenanceScheduler, MaintenancePolicy
from .work_order_management import WorkOrderManagement, WorkOrderConfig

__all__ = [
    "FaultReportGenerator",
    "CostEstimationConfig",
    "MaintenanceScheduler",
    "MaintenancePolicy",
    "WorkOrderManagement",
    "WorkOrderConfig",
]
