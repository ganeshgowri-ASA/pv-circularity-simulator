"""
Diagnostic-related Pydantic models for PV system analysis.

This module defines models for diagnostic results, analysis, and reporting.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import Field

from .base import BaseSchema
from .defects import Defect, DefectSeverity, DefectType


class DiagnosticStatus(str, Enum):
    """
    Status of diagnostic operations.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RecommendedAction(str, Enum):
    """
    Recommended actions for defect remediation.
    """

    MONITOR = "monitor"
    SCHEDULE_INSPECTION = "schedule_inspection"
    IMMEDIATE_REPAIR = "immediate_repair"
    REPLACE_PANEL = "replace_panel"
    REPLACE_MODULE = "replace_module"
    CLEAN = "clean"
    ADJUST = "adjust"
    NO_ACTION = "no_action"


class DiagnosticResult(BaseSchema):
    """
    Comprehensive diagnostic result model.

    Attributes:
        defect_id: Associated defect identifier
        defect: Complete defect object
        root_cause: Identified root cause
        root_cause_confidence: Confidence in root cause identification
        recommended_action: Recommended remediation action
        priority: Priority level (1-5, 1 being highest)
        estimated_impact: Estimated impact on system performance
        estimated_cost: Estimated repair/replacement cost
        time_to_failure: Estimated time until critical failure
        analysis_notes: Additional analysis notes
        related_defects: List of related defect IDs
    """

    defect_id: str = Field(description="Defect identifier")
    defect: Optional[Defect] = Field(None, description="Complete defect object")

    root_cause: str = Field(description="Root cause analysis")
    root_cause_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Root cause confidence"
    )

    recommended_action: RecommendedAction = Field(description="Recommended action")
    priority: int = Field(ge=1, le=5, description="Priority (1=highest)")

    estimated_impact: float = Field(
        ge=0.0,
        le=100.0,
        description="Estimated impact percentage"
    )
    estimated_cost: float = Field(ge=0.0, description="Estimated cost in USD")
    time_to_failure: Optional[int] = Field(
        None,
        description="Days until critical failure"
    )

    analysis_notes: Optional[str] = Field(None, description="Analysis notes")
    related_defects: List[str] = Field(
        default_factory=list,
        description="Related defect IDs"
    )


class FaultReport(BaseSchema):
    """
    Comprehensive fault report for PV systems.

    Attributes:
        report_title: Title of the report
        site_id: Identifier of the site
        report_type: Type of report (automated, manual, scheduled)
        diagnostics: List of diagnostic results
        summary: Executive summary
        total_defects: Total number of defects
        critical_defects: Number of critical defects
        estimated_total_cost: Total estimated repair cost
        estimated_power_loss: Total estimated power loss
        recommendations: High-level recommendations
        generated_by: User or system that generated the report
        report_format: Format of the report (PDF, HTML, JSON)
    """

    report_title: str = Field(description="Report title")
    site_id: str = Field(description="Site identifier")
    report_type: str = Field(default="automated", description="Report type")

    diagnostics: List[DiagnosticResult] = Field(
        default_factory=list,
        description="Diagnostic results"
    )

    summary: Optional[str] = Field(None, description="Executive summary")
    total_defects: int = Field(default=0, ge=0, description="Total defects")
    critical_defects: int = Field(default=0, ge=0, description="Critical defects")

    estimated_total_cost: float = Field(default=0.0, ge=0.0, description="Total cost estimate")
    estimated_power_loss: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Total power loss percentage"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="High-level recommendations"
    )
    generated_by: str = Field(default="system", description="Report generator")
    report_format: str = Field(default="JSON", description="Report format")


class FleetAnalysis(BaseSchema):
    """
    Fleet-wide analysis of PV installations.

    Attributes:
        fleet_id: Fleet identifier
        site_ids: List of site identifiers
        total_sites: Total number of sites
        total_panels: Total number of panels
        total_defects: Total number of defects
        defect_distribution: Distribution of defects by type
        severity_distribution: Distribution of defects by severity
        average_panel_age: Average panel age in years
        fleet_health_score: Overall fleet health score (0-100)
        trend_analysis: Trend analysis data
        benchmarks: Benchmark comparisons
    """

    fleet_id: str = Field(description="Fleet identifier")
    site_ids: List[str] = Field(default_factory=list, description="Site identifiers")

    total_sites: int = Field(default=0, ge=0, description="Total sites")
    total_panels: int = Field(default=0, ge=0, description="Total panels")
    total_defects: int = Field(default=0, ge=0, description="Total defects")

    defect_distribution: Dict[DefectType, int] = Field(
        default_factory=dict,
        description="Defect distribution by type"
    )
    severity_distribution: Dict[DefectSeverity, int] = Field(
        default_factory=dict,
        description="Severity distribution"
    )

    average_panel_age: float = Field(default=0.0, ge=0.0, description="Average panel age")
    fleet_health_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Fleet health score"
    )

    trend_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trend analysis"
    )
    benchmarks: Dict[str, Any] = Field(default_factory=dict, description="Benchmarks")
