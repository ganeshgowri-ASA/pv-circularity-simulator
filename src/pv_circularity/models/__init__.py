"""
Pydantic models for PV Circularity Simulator.

This module exports all data models used throughout the application.
"""

from .base import BaseSchema, Coordinates, GeoLocation
from .defects import (
    DefectType,
    DefectSeverity,
    ImpactCategory,
    Defect,
    DefectPattern,
    DefectHistory,
)
from .diagnostics import (
    DiagnosticStatus,
    RecommendedAction,
    DiagnosticResult,
    FaultReport,
    FleetAnalysis,
)
from .maintenance import (
    MaintenanceType,
    MaintenancePriority,
    WorkOrderStatus,
    SparePart,
    MaintenanceSchedule,
    Technician,
    WorkOrder,
    CorrectiveAction,
)

__all__ = [
    # Base models
    "BaseSchema",
    "Coordinates",
    "GeoLocation",
    # Defect models
    "DefectType",
    "DefectSeverity",
    "ImpactCategory",
    "Defect",
    "DefectPattern",
    "DefectHistory",
    # Diagnostic models
    "DiagnosticStatus",
    "RecommendedAction",
    "DiagnosticResult",
    "FaultReport",
    "FleetAnalysis",
    # Maintenance models
    "MaintenanceType",
    "MaintenancePriority",
    "WorkOrderStatus",
    "SparePart",
    "MaintenanceSchedule",
    "Technician",
    "WorkOrder",
    "CorrectiveAction",
]
