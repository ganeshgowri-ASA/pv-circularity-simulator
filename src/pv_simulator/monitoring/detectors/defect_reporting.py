"""
Defect Reporting Module for PV Panel Analysis.

This module provides comprehensive defect reporting capabilities including
severity classification, location mapping, and repair recommendations.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pv_simulator.monitoring.detectors.defect_classifier import (
    ClassifiedDefect,
    DefectSeverity,
    DefectType,
)

logger = logging.getLogger(__name__)


class RepairPriority(str, Enum):
    """Repair priority levels."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MONITOR = "monitor"


class RepairMethod(str, Enum):
    """Recommended repair methods."""

    REPLACE_PANEL = "replace_panel"
    REPLACE_CELL = "replace_cell"
    CLEAN_SURFACE = "clean_surface"
    RESEAL = "reseal"
    REWIRE = "rewire"
    MONITOR_ONLY = "monitor_only"
    FULL_INSPECTION = "full_inspection"


class LocationMap(BaseModel):
    """
    Physical location mapping for defects.

    Attributes:
        panel_id: Unique panel identifier
        array_position: Position in array (row, column)
        gps_coordinates: GPS location (latitude, longitude)
        zone: Physical zone/section identifier
        cell_grid_position: Position within cell grid
        normalized_coords: Normalized coordinates (0-1, 0-1)
    """

    panel_id: str
    array_position: Optional[Tuple[int, int]] = None
    gps_coordinates: Optional[Tuple[float, float]] = None
    zone: Optional[str] = None
    cell_grid_position: Optional[Tuple[int, int]] = None
    normalized_coords: Tuple[float, float] = Field(..., description="(x, y) normalized 0-1")


class RepairRecommendation(BaseModel):
    """
    Repair recommendation for a defect.

    Attributes:
        priority: Repair priority level
        methods: Recommended repair methods
        estimated_cost: Estimated repair cost (USD)
        estimated_time: Estimated time to repair (hours)
        parts_needed: List of required parts
        tools_needed: List of required tools
        safety_precautions: Safety warnings and precautions
        notes: Additional notes and instructions
    """

    priority: RepairPriority
    methods: List[RepairMethod]
    estimated_cost: Optional[float] = Field(None, ge=0.0)
    estimated_time: Optional[float] = Field(None, ge=0.0)
    parts_needed: List[str] = Field(default_factory=list)
    tools_needed: List[str] = Field(default_factory=list)
    safety_precautions: List[str] = Field(default_factory=list)
    notes: str = ""


class DefectReport(BaseModel):
    """
    Comprehensive defect report.

    Attributes:
        report_id: Unique report identifier
        timestamp: Report generation timestamp
        defect: Classified defect information
        location: Physical location mapping
        severity_assessment: Detailed severity assessment
        recommendation: Repair recommendation
        impact_analysis: Impact on system performance
        metadata: Additional metadata
    """

    report_id: str
    timestamp: str
    defect: ClassifiedDefect
    location: LocationMap
    severity_assessment: Dict[str, Any]
    recommendation: RepairRecommendation
    impact_analysis: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DefectReporting:
    """
    Comprehensive defect reporting and analysis system.

    This class provides methods for:
    - Severity classification and assessment
    - Physical location mapping
    - Repair recommendations
    - Impact analysis
    - Report generation
    """

    # Cost estimates for different repair types (USD)
    REPAIR_COSTS = {
        RepairMethod.REPLACE_PANEL: 250.0,
        RepairMethod.REPLACE_CELL: 75.0,
        RepairMethod.CLEAN_SURFACE: 25.0,
        RepairMethod.RESEAL: 50.0,
        RepairMethod.REWIRE: 100.0,
        RepairMethod.MONITOR_ONLY: 0.0,
        RepairMethod.FULL_INSPECTION: 150.0,
    }

    # Time estimates for different repair types (hours)
    REPAIR_TIMES = {
        RepairMethod.REPLACE_PANEL: 2.0,
        RepairMethod.REPLACE_CELL: 1.5,
        RepairMethod.CLEAN_SURFACE: 0.5,
        RepairMethod.RESEAL: 1.0,
        RepairMethod.REWIRE: 2.0,
        RepairMethod.MONITOR_ONLY: 0.0,
        RepairMethod.FULL_INSPECTION: 3.0,
    }

    def __init__(self):
        """Initialize defect reporting system."""
        logger.info("Initialized DefectReporting")

    def severity_classification(
        self,
        defect: ClassifiedDefect,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform detailed severity classification and assessment.

        Analyzes defect characteristics to provide comprehensive severity
        assessment including risk factors and performance impact.

        Args:
            defect: Classified defect to assess
            context: Optional contextual information (panel age, location, etc.)

        Returns:
            Dictionary containing detailed severity assessment

        Example:
            >>> reporter = DefectReporting()
            >>> assessment = reporter.severity_classification(defect)
            >>> print(f"Risk score: {assessment['risk_score']}")
            >>> print(f"Progression rate: {assessment['progression_rate']}")
        """
        context = context or {}

        # Base severity score (0-100)
        severity_scores = {
            DefectSeverity.LOW: 25,
            DefectSeverity.MEDIUM: 50,
            DefectSeverity.HIGH: 75,
            DefectSeverity.CRITICAL: 100,
        }
        base_score = severity_scores[defect.severity]

        # Adjust for defect type
        type_multipliers = {
            DefectType.CRACK: 1.2,
            DefectType.HOTSPOT: 1.5,
            DefectType.DELAMINATION: 1.3,
            DefectType.SOILING: 0.8,
            DefectType.PID: 1.4,
            DefectType.CORROSION: 1.3,
            DefectType.BURN_MARK: 1.6,
            DefectType.CELL_BREAKAGE: 1.5,
            DefectType.UNKNOWN: 1.0,
        }
        type_multiplier = type_multipliers.get(defect.defect_type, 1.0)

        # Adjust for area affected
        area_factor = min(2.0, 1.0 + defect.area_percentage / 50.0)

        # Calculate final risk score
        risk_score = min(100.0, base_score * type_multiplier * area_factor)

        # Estimate progression rate (% per month)
        progression_rates = {
            DefectType.CRACK: 5.0,
            DefectType.HOTSPOT: 8.0,
            DefectType.DELAMINATION: 3.0,
            DefectType.SOILING: 2.0,
            DefectType.PID: 6.0,
            DefectType.CORROSION: 4.0,
            DefectType.BURN_MARK: 10.0,
            DefectType.CELL_BREAKAGE: 7.0,
            DefectType.UNKNOWN: 5.0,
        }
        progression_rate = progression_rates.get(defect.defect_type, 5.0)

        # Environmental factors
        if context.get("high_humidity"):
            progression_rate *= 1.3
        if context.get("coastal_location"):
            progression_rate *= 1.2
        if context.get("extreme_temperatures"):
            progression_rate *= 1.2

        # Age factor
        panel_age = context.get("panel_age_years", 0)
        age_factor = 1.0 + (panel_age / 25.0)  # Increase with age

        assessment = {
            "severity_level": defect.severity.value,
            "risk_score": float(risk_score),
            "progression_rate": float(progression_rate * age_factor),
            "area_affected_pct": float(defect.area_percentage),
            "confidence": float(defect.confidence),
            "type_multiplier": float(type_multiplier),
            "environmental_factors": {
                "high_humidity": context.get("high_humidity", False),
                "coastal_location": context.get("coastal_location", False),
                "extreme_temperatures": context.get("extreme_temperatures", False),
            },
            "requires_immediate_action": risk_score >= 75,
            "estimated_lifetime_impact_years": float(
                self._estimate_lifetime_impact(defect, progression_rate)
            ),
        }

        logger.info(
            f"Severity assessment: {defect.defect_type.value} - "
            f"risk_score={risk_score:.1f}, progression={progression_rate:.1f}%/mo"
        )

        return assessment

    def location_mapping(
        self,
        defect: ClassifiedDefect,
        panel_id: str,
        array_position: Optional[Tuple[int, int]] = None,
        gps_coords: Optional[Tuple[float, float]] = None,
        zone: Optional[str] = None,
    ) -> LocationMap:
        """
        Map defect to physical location in the PV array.

        Creates comprehensive location mapping for defect tracking
        and repair planning.

        Args:
            defect: Classified defect
            panel_id: Unique panel identifier
            array_position: Position in array (row, col)
            gps_coords: GPS coordinates (latitude, longitude)
            zone: Physical zone identifier

        Returns:
            LocationMap with comprehensive location information

        Example:
            >>> reporter = DefectReporting()
            >>> location = reporter.location_mapping(
            ...     defect,
            ...     panel_id="PNL-A-123",
            ...     array_position=(5, 8),
            ...     gps_coords=(37.7749, -122.4194)
            ... )
            >>> print(f"Panel: {location.panel_id} at {location.array_position}")
        """
        # Extract cell grid position if available
        cell_position = None
        if hasattr(defect.location, "cell_row") and defect.location.cell_row is not None:
            cell_position = (defect.location.cell_row, defect.location.cell_col)

        location_map = LocationMap(
            panel_id=panel_id,
            array_position=array_position,
            gps_coordinates=gps_coords,
            zone=zone,
            cell_grid_position=cell_position,
            normalized_coords=(defect.location.x, defect.location.y),
        )

        logger.info(f"Location mapping: panel={panel_id}, position={array_position}, zone={zone}")

        return location_map

    def repair_recommendations(
        self,
        defect: ClassifiedDefect,
        severity_assessment: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RepairRecommendation:
        """
        Generate repair recommendations based on defect analysis.

        Provides actionable repair recommendations including methods,
        costs, timeline, and required resources.

        Args:
            defect: Classified defect
            severity_assessment: Severity assessment results
            context: Optional context (warranty status, budget, etc.)

        Returns:
            RepairRecommendation with detailed repair plan

        Example:
            >>> reporter = DefectReporting()
            >>> assessment = reporter.severity_classification(defect)
            >>> recommendation = reporter.repair_recommendations(defect, assessment)
            >>> print(f"Priority: {recommendation.priority}")
            >>> print(f"Methods: {recommendation.methods}")
            >>> print(f"Cost: ${recommendation.estimated_cost:.2f}")
        """
        context = context or {}

        # Determine repair priority
        risk_score = severity_assessment["risk_score"]
        if risk_score >= 90 or defect.severity == DefectSeverity.CRITICAL:
            priority = RepairPriority.IMMEDIATE
        elif risk_score >= 70 or defect.severity == DefectSeverity.HIGH:
            priority = RepairPriority.HIGH
        elif risk_score >= 40 or defect.severity == DefectSeverity.MEDIUM:
            priority = RepairPriority.MEDIUM
        elif risk_score >= 20:
            priority = RepairPriority.LOW
        else:
            priority = RepairPriority.MONITOR

        # Determine repair methods based on defect type
        methods = self._get_repair_methods(defect)

        # Calculate costs and time
        estimated_cost = sum(self.REPAIR_COSTS.get(method, 0) for method in methods)
        estimated_time = max(self.REPAIR_TIMES.get(method, 0) for method in methods)

        # Adjust for context
        if context.get("under_warranty"):
            estimated_cost *= 0.5  # Warranty coverage

        # Get required parts and tools
        parts_needed = self._get_required_parts(defect, methods)
        tools_needed = self._get_required_tools(methods)

        # Safety precautions
        safety_precautions = self._get_safety_precautions(defect, methods)

        # Generate notes
        notes = self._generate_repair_notes(defect, severity_assessment, methods)

        recommendation = RepairRecommendation(
            priority=priority,
            methods=methods,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            parts_needed=parts_needed,
            tools_needed=tools_needed,
            safety_precautions=safety_precautions,
            notes=notes,
        )

        logger.info(
            f"Repair recommendation: priority={priority.value}, "
            f"cost=${estimated_cost:.2f}, time={estimated_time:.1f}h"
        )

        return recommendation

    def generate_report(
        self,
        defect: ClassifiedDefect,
        panel_id: str,
        array_position: Optional[Tuple[int, int]] = None,
        gps_coords: Optional[Tuple[float, float]] = None,
        zone: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefectReport:
        """
        Generate comprehensive defect report.

        Creates a complete report combining all analysis components.

        Args:
            defect: Classified defect
            panel_id: Unique panel identifier
            array_position: Position in array
            gps_coords: GPS coordinates
            zone: Physical zone
            context: Additional context

        Returns:
            Complete DefectReport

        Example:
            >>> reporter = DefectReporting()
            >>> report = reporter.generate_report(
            ...     defect,
            ...     panel_id="PNL-001",
            ...     array_position=(1, 5),
            ...     context={"panel_age_years": 10}
            ... )
            >>> print(f"Report ID: {report.report_id}")
        """
        # Generate report ID
        timestamp = datetime.utcnow()
        report_id = f"RPT-{panel_id}-{timestamp.strftime('%Y%m%d%H%M%S')}"

        # Perform analysis
        severity_assessment = self.severity_classification(defect, context)
        location = self.location_mapping(defect, panel_id, array_position, gps_coords, zone)
        recommendation = self.repair_recommendations(defect, severity_assessment, context)
        impact = self._analyze_performance_impact(defect, severity_assessment)

        # Compile metadata
        metadata = {
            "detection_method": defect.characteristics.get("detection_method", "unknown"),
            "image_quality": context.get("image_quality", "unknown") if context else "unknown",
            "weather_conditions": context.get("weather_conditions") if context else None,
            "panel_age_years": context.get("panel_age_years") if context else None,
        }

        report = DefectReport(
            report_id=report_id,
            timestamp=timestamp.isoformat(),
            defect=defect,
            location=location,
            severity_assessment=severity_assessment,
            recommendation=recommendation,
            impact_analysis=impact,
            metadata=metadata,
        )

        logger.info(f"Generated report: {report_id} for {defect.defect_type.value}")

        return report

    # Helper methods

    def _estimate_lifetime_impact(self, defect: ClassifiedDefect, progression_rate: float) -> float:
        """Estimate impact on panel lifetime (years)."""
        # Base lifetime: 25 years
        base_lifetime = 25.0

        # Calculate reduction based on severity and progression
        severity_impact = {
            DefectSeverity.LOW: 0.5,
            DefectSeverity.MEDIUM: 2.0,
            DefectSeverity.HIGH: 5.0,
            DefectSeverity.CRITICAL: 10.0,
        }

        impact_years = severity_impact.get(defect.severity, 2.0)
        impact_years *= (1.0 + progression_rate / 10.0)

        return min(base_lifetime, impact_years)

    def _get_repair_methods(self, defect: ClassifiedDefect) -> List[RepairMethod]:
        """Determine appropriate repair methods for defect type."""
        method_map = {
            DefectType.CRACK: [RepairMethod.REPLACE_PANEL, RepairMethod.REPLACE_CELL],
            DefectType.HOTSPOT: [RepairMethod.FULL_INSPECTION, RepairMethod.REWIRE],
            DefectType.DELAMINATION: [RepairMethod.REPLACE_PANEL, RepairMethod.RESEAL],
            DefectType.SOILING: [RepairMethod.CLEAN_SURFACE],
            DefectType.PID: [RepairMethod.REPLACE_PANEL, RepairMethod.REWIRE],
            DefectType.CORROSION: [RepairMethod.REPLACE_PANEL, RepairMethod.RESEAL],
            DefectType.BURN_MARK: [RepairMethod.REPLACE_PANEL, RepairMethod.FULL_INSPECTION],
            DefectType.CELL_BREAKAGE: [RepairMethod.REPLACE_CELL, RepairMethod.REPLACE_PANEL],
            DefectType.UNKNOWN: [RepairMethod.FULL_INSPECTION],
        }

        methods = method_map.get(defect.defect_type, [RepairMethod.FULL_INSPECTION])

        # For low severity, add monitoring option
        if defect.severity == DefectSeverity.LOW:
            methods.append(RepairMethod.MONITOR_ONLY)

        return methods

    def _get_required_parts(
        self, defect: ClassifiedDefect, methods: List[RepairMethod]
    ) -> List[str]:
        """Get list of required parts for repair."""
        parts = []

        if RepairMethod.REPLACE_PANEL in methods:
            parts.extend(["Replacement PV panel", "Mounting hardware", "DC connectors"])
        if RepairMethod.REPLACE_CELL in methods:
            parts.extend(["Replacement solar cell", "Encapsulant", "Junction box"])
        if RepairMethod.CLEAN_SURFACE in methods:
            parts.extend(["Cleaning solution", "Microfiber cloths"])
        if RepairMethod.RESEAL in methods:
            parts.extend(["Sealant compound", "Frame gaskets"])
        if RepairMethod.REWIRE in methods:
            parts.extend(["Electrical wire", "Connectors", "Junction box"])

        return list(set(parts))  # Remove duplicates

    def _get_required_tools(self, methods: List[RepairMethod]) -> List[str]:
        """Get list of required tools for repair."""
        tools = ["Multimeter", "Safety gloves", "Safety glasses"]

        if RepairMethod.REPLACE_PANEL in methods or RepairMethod.REPLACE_CELL in methods:
            tools.extend(["Socket wrench set", "Wire strippers", "Crimping tool", "Torque wrench"])
        if RepairMethod.CLEAN_SURFACE in methods:
            tools.extend(["Soft-bristle brush", "Squeegee", "Water hose"])
        if RepairMethod.RESEAL in methods:
            tools.extend(["Caulking gun", "Scraper", "Cleaning solvent"])
        if RepairMethod.REWIRE in methods:
            tools.extend(["Wire cutters", "Screwdrivers", "Heat shrink tubing", "Heat gun"])

        return list(set(tools))

    def _get_safety_precautions(
        self, defect: ClassifiedDefect, methods: List[RepairMethod]
    ) -> List[str]:
        """Get safety precautions for repair work."""
        precautions = [
            "Disconnect panel from electrical system before work",
            "Wear appropriate PPE (gloves, safety glasses)",
            "Work during daylight hours only",
            "Ensure stable footing and fall protection if on roof",
        ]

        if defect.defect_type == DefectType.HOTSPOT:
            precautions.append("CAUTION: Panel may be hot - allow cooling time")
        if defect.defect_type == DefectType.BURN_MARK:
            precautions.append("WARNING: Fire risk - inspect thoroughly for electrical faults")
        if RepairMethod.REPLACE_PANEL in methods:
            precautions.append("Use proper lifting technique - panels are heavy")

        return precautions

    def _generate_repair_notes(
        self,
        defect: ClassifiedDefect,
        severity_assessment: Dict[str, Any],
        methods: List[RepairMethod],
    ) -> str:
        """Generate detailed repair notes."""
        notes = []

        notes.append(
            f"Defect Type: {defect.defect_type.value.replace('_', ' ').title()}"
        )
        notes.append(f"Severity: {defect.severity.value.upper()}")
        notes.append(
            f"Risk Score: {severity_assessment['risk_score']:.1f}/100"
        )
        notes.append(
            f"Area Affected: {defect.area_percentage:.2f}%"
        )

        if severity_assessment.get("requires_immediate_action"):
            notes.append("\n**IMMEDIATE ACTION REQUIRED**")

        notes.append(f"\nProgression Rate: {severity_assessment['progression_rate']:.1f}% per month")
        notes.append(
            f"Estimated Lifetime Impact: "
            f"{severity_assessment['estimated_lifetime_impact_years']:.1f} years"
        )

        if defect.defect_type == DefectType.SOILING:
            notes.append("\nRegular cleaning schedule recommended to prevent recurrence.")
        elif defect.defect_type == DefectType.HOTSPOT:
            notes.append("\nInvestigate electrical connections and bypass diodes.")
        elif defect.defect_type == DefectType.PID:
            notes.append("\nConsider system-level voltage optimization to prevent PID.")

        return "\n".join(notes)

    def _analyze_performance_impact(
        self, defect: ClassifiedDefect, severity_assessment: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze impact on panel performance."""
        # Base power loss by defect type (%)
        power_loss_map = {
            DefectType.CRACK: 5.0,
            DefectType.HOTSPOT: 10.0,
            DefectType.DELAMINATION: 8.0,
            DefectType.SOILING: 3.0,
            DefectType.PID: 15.0,
            DefectType.CORROSION: 7.0,
            DefectType.BURN_MARK: 20.0,
            DefectType.CELL_BREAKAGE: 12.0,
            DefectType.UNKNOWN: 5.0,
        }

        base_power_loss = power_loss_map.get(defect.defect_type, 5.0)

        # Scale by severity
        severity_multiplier = {
            DefectSeverity.LOW: 0.5,
            DefectSeverity.MEDIUM: 1.0,
            DefectSeverity.HIGH: 1.5,
            DefectSeverity.CRITICAL: 2.0,
        }
        multiplier = severity_multiplier.get(defect.severity, 1.0)

        estimated_power_loss = min(100.0, base_power_loss * multiplier * (1 + defect.area_percentage / 100))

        # Calculate efficiency degradation
        efficiency_degradation = estimated_power_loss * 0.8  # Typically 80% of power loss

        # Estimate annual energy loss (kWh) - assumes 250W panel, 5 peak hours/day
        daily_energy = 250 * 5 / 1000  # kWh
        annual_energy_loss = daily_energy * 365 * (estimated_power_loss / 100)

        return {
            "estimated_power_loss_pct": float(estimated_power_loss),
            "efficiency_degradation_pct": float(efficiency_degradation),
            "annual_energy_loss_kwh": float(annual_energy_loss),
            "performance_ratio_impact": float(estimated_power_loss / 100),
        }
