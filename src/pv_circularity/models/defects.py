"""
Defect-related Pydantic models for PV panel diagnostics.

This module defines models for defect detection, categorization, and tracking.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import Field, field_validator

from .base import BaseSchema, Coordinates


class DefectType(str, Enum):
    """
    Types of defects detectable in PV panels.
    """

    CRACK = "crack"
    HOTSPOT = "hotspot"
    DELAMINATION = "delamination"
    DISCOLORATION = "discoloration"
    PID = "potential_induced_degradation"
    SNAIL_TRAIL = "snail_trail"
    CORROSION = "corrosion"
    BURN_MARK = "burn_mark"
    CELL_BREAKAGE = "cell_breakage"
    JUNCTION_BOX_FAILURE = "junction_box_failure"
    BACKSHEET_DAMAGE = "backsheet_damage"
    GLASS_BREAKAGE = "glass_breakage"
    SOILING = "soiling"
    SHADOWING = "shadowing"
    OTHER = "other"


class DefectSeverity(str, Enum):
    """
    Severity levels for detected defects.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactCategory(str, Enum):
    """
    Categories of impact from defects.
    """

    POWER_LOSS = "power_loss"
    SAFETY_HAZARD = "safety_hazard"
    DEGRADATION_ACCELERATION = "degradation_acceleration"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    FIRE_RISK = "fire_risk"


class Defect(BaseSchema):
    """
    Comprehensive defect model for PV panel diagnostics.

    Attributes:
        type: Type of defect detected
        severity: Severity level of the defect
        location: Spatial coordinates of the defect
        confidence: Detection confidence score (0.0-1.0)
        panel_id: Identifier of the affected panel
        module_id: Identifier of the affected module (optional)
        string_id: Identifier of the affected string (optional)
        description: Human-readable description
        image_path: Path to defect image (optional)
        detection_method: Method used to detect the defect
        estimated_power_loss: Estimated power loss percentage
        impacts: List of impact categories
        metadata: Additional metadata
    """

    type: DefectType = Field(description="Type of defect")
    severity: DefectSeverity = Field(description="Severity level")
    location: Coordinates = Field(description="Location coordinates")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")

    panel_id: str = Field(description="Affected panel identifier")
    module_id: Optional[str] = Field(None, description="Affected module identifier")
    string_id: Optional[str] = Field(None, description="Affected string identifier")

    description: Optional[str] = Field(None, description="Defect description")
    image_path: Optional[str] = Field(None, description="Path to defect image")
    detection_method: str = Field(default="automated", description="Detection method")

    estimated_power_loss: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated power loss percentage"
    )
    impacts: List[ImpactCategory] = Field(
        default_factory=list,
        description="Impact categories"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class DefectPattern(BaseSchema):
    """
    Identified pattern of recurring defects.

    Attributes:
        pattern_name: Name of the defect pattern
        defect_type: Type of defect in the pattern
        frequency: Occurrence frequency
        affected_panels: List of affected panel IDs
        common_characteristics: Common characteristics of the pattern
        root_cause: Identified root cause (if known)
        correlation_score: Statistical correlation score
    """

    pattern_name: str = Field(description="Pattern name")
    defect_type: DefectType = Field(description="Defect type")
    frequency: int = Field(ge=1, description="Occurrence frequency")
    affected_panels: List[str] = Field(description="Affected panel IDs")
    common_characteristics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Common characteristics"
    )
    root_cause: Optional[str] = Field(None, description="Root cause")
    correlation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Statistical correlation score"
    )


class DefectHistory(BaseSchema):
    """
    Historical record of defect progression.

    Attributes:
        defect_id: Original defect identifier
        snapshots: List of defect snapshots over time
        progression_rate: Rate of defect progression
        repair_attempts: Number of repair attempts
        current_status: Current status of the defect
    """

    defect_id: str = Field(description="Defect identifier")
    snapshots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Defect snapshots over time"
    )
    progression_rate: float = Field(description="Progression rate")
    repair_attempts: int = Field(default=0, ge=0, description="Number of repair attempts")
    current_status: str = Field(default="active", description="Current status")
