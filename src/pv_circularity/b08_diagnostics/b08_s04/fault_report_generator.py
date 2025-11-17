"""
Fault Report Generator for PV System Diagnostics (B08-S04).

This module provides automated fault report generation with defect categorization,
severity assessment, and repair cost estimation capabilities.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json

from pydantic import BaseModel, Field

from ...models import (
    Defect,
    DefectType,
    DefectSeverity,
    DiagnosticResult,
    FaultReport,
    RecommendedAction,
)


class CostEstimationConfig(BaseModel):
    """
    Configuration for repair cost estimation.

    Attributes:
        base_labor_rate: Base hourly labor rate
        panel_replacement_cost: Average panel replacement cost
        module_replacement_cost: Average module replacement cost
        cleaning_cost_per_panel: Cost per panel for cleaning
        inspection_cost_per_hour: Inspection cost per hour
        emergency_multiplier: Cost multiplier for emergency repairs
    """

    base_labor_rate: float = Field(default=75.0, description="Base labor rate per hour")
    panel_replacement_cost: float = Field(default=300.0, description="Panel replacement cost")
    module_replacement_cost: float = Field(
        default=150.0,
        description="Module replacement cost"
    )
    cleaning_cost_per_panel: float = Field(default=25.0, description="Cleaning cost per panel")
    inspection_cost_per_hour: float = Field(
        default=50.0,
        description="Inspection cost per hour"
    )
    emergency_multiplier: float = Field(default=1.5, description="Emergency repair multiplier")


class FaultReportGenerator:
    """
    Generates comprehensive fault reports for PV systems.

    This class provides automated report generation with intelligent defect
    categorization, severity assessment, and cost estimation.

    Attributes:
        cost_config: Cost estimation configuration
        severity_thresholds: Thresholds for severity assessment
        defect_categories: Categorization rules for defects
    """

    def __init__(self, cost_config: Optional[CostEstimationConfig] = None):
        """
        Initialize the FaultReportGenerator.

        Args:
            cost_config: Optional cost estimation configuration
        """
        self.cost_config = cost_config or CostEstimationConfig()
        self.severity_thresholds = self._initialize_severity_thresholds()
        self.defect_categories = self._initialize_defect_categories()

    def _initialize_severity_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize severity assessment thresholds.

        Returns:
            Dictionary of severity thresholds by defect type
        """
        return {
            DefectType.CRACK: {
                "low": 0.0,
                "medium": 2.0,
                "high": 5.0,
                "critical": 10.0,
            },
            DefectType.HOTSPOT: {
                "low": 5.0,
                "medium": 10.0,
                "high": 15.0,
                "critical": 20.0,
            },
            DefectType.DELAMINATION: {
                "low": 1.0,
                "medium": 5.0,
                "high": 10.0,
                "critical": 20.0,
            },
            DefectType.PID: {
                "low": 5.0,
                "medium": 15.0,
                "high": 30.0,
                "critical": 50.0,
            },
        }

    def _initialize_defect_categories(self) -> Dict[str, List[DefectType]]:
        """
        Initialize defect categorization rules.

        Returns:
            Dictionary of defect categories with associated types
        """
        return {
            "structural": [
                DefectType.CRACK,
                DefectType.CELL_BREAKAGE,
                DefectType.GLASS_BREAKAGE,
                DefectType.BACKSHEET_DAMAGE,
            ],
            "electrical": [
                DefectType.HOTSPOT,
                DefectType.PID,
                DefectType.JUNCTION_BOX_FAILURE,
            ],
            "environmental": [
                DefectType.DELAMINATION,
                DefectType.CORROSION,
                DefectType.SOILING,
                DefectType.DISCOLORATION,
            ],
            "performance": [
                DefectType.SNAIL_TRAIL,
                DefectType.SHADOWING,
            ],
        }

    def automated_report_generation(
        self,
        site_id: str,
        defects: List[Defect],
        report_title: Optional[str] = None,
        include_recommendations: bool = True,
    ) -> FaultReport:
        """
        Generate an automated comprehensive fault report.

        This method analyzes all defects, categorizes them, assesses severity,
        estimates costs, and generates actionable recommendations.

        Args:
            site_id: Identifier of the site being reported on
            defects: List of detected defects
            report_title: Optional custom report title
            include_recommendations: Whether to include recommendations

        Returns:
            Comprehensive FaultReport object

        Example:
            >>> generator = FaultReportGenerator()
            >>> defects = [defect1, defect2, defect3]
            >>> report = generator.automated_report_generation(
            ...     site_id="SITE-001",
            ...     defects=defects
            ... )
            >>> print(f"Total defects: {report.total_defects}")
        """
        if not report_title:
            report_title = f"Automated Fault Report - {site_id} - {datetime.utcnow().date()}"

        # Perform defect categorization
        categorized_defects = self.defect_categorization(defects)

        # Generate diagnostic results for each defect
        diagnostics: List[DiagnosticResult] = []
        total_cost = 0.0
        total_power_loss = 0.0
        critical_count = 0

        for defect in defects:
            # Assess severity if not already set
            if not self._is_severity_appropriate(defect):
                defect.severity = self.severity_assessment(defect)

            # Estimate repair cost
            cost = self.repair_cost_estimation(defect)

            # Create diagnostic result
            diagnostic = DiagnosticResult(
                defect_id=defect.id,
                defect=defect,
                root_cause=self._determine_root_cause(defect),
                root_cause_confidence=0.75,  # Default confidence
                recommended_action=self._determine_recommended_action(defect),
                priority=self._calculate_priority(defect),
                estimated_impact=defect.estimated_power_loss,
                estimated_cost=cost,
                time_to_failure=self._estimate_time_to_failure(defect),
                analysis_notes=self._generate_analysis_notes(defect),
            )

            diagnostics.append(diagnostic)
            total_cost += cost
            total_power_loss += defect.estimated_power_loss

            if defect.severity == DefectSeverity.CRITICAL:
                critical_count += 1

        # Generate summary
        summary = self._generate_executive_summary(
            categorized_defects,
            len(defects),
            critical_count,
            total_cost,
            total_power_loss,
        )

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(diagnostics, categorized_defects)

        # Create and return fault report
        report = FaultReport(
            report_title=report_title,
            site_id=site_id,
            report_type="automated",
            diagnostics=diagnostics,
            summary=summary,
            total_defects=len(defects),
            critical_defects=critical_count,
            estimated_total_cost=total_cost,
            estimated_power_loss=min(total_power_loss, 100.0),  # Cap at 100%
            recommendations=recommendations,
            generated_by="FaultReportGenerator",
        )

        return report

    def defect_categorization(self, defects: List[Defect]) -> Dict[str, List[Defect]]:
        """
        Categorize defects by type and characteristics.

        Groups defects into logical categories (structural, electrical,
        environmental, performance) for easier analysis and reporting.

        Args:
            defects: List of defects to categorize

        Returns:
            Dictionary mapping categories to lists of defects

        Example:
            >>> generator = FaultReportGenerator()
            >>> categorized = generator.defect_categorization(defects)
            >>> print(f"Structural defects: {len(categorized['structural'])}")
        """
        categorized: Dict[str, List[Defect]] = defaultdict(list)

        for defect in defects:
            # Determine category based on defect type
            for category, types in self.defect_categories.items():
                if defect.type in types:
                    categorized[category].append(defect)
                    break
            else:
                # If no category matches, add to "other"
                categorized["other"].append(defect)

        return dict(categorized)

    def severity_assessment(self, defect: Defect) -> DefectSeverity:
        """
        Assess the severity of a defect based on multiple factors.

        Evaluates defect characteristics including type, location, estimated
        power loss, and confidence to determine appropriate severity level.

        Args:
            defect: Defect to assess

        Returns:
            Assessed DefectSeverity level

        Example:
            >>> generator = FaultReportGenerator()
            >>> defect = Defect(
            ...     type=DefectType.CRACK,
            ...     estimated_power_loss=8.0,
            ...     ...
            ... )
            >>> severity = generator.severity_assessment(defect)
            >>> print(severity)  # DefectSeverity.HIGH
        """
        # Get thresholds for this defect type
        thresholds = self.severity_thresholds.get(
            defect.type,
            {
                "low": 0.0,
                "medium": 5.0,
                "high": 15.0,
                "critical": 30.0,
            },
        )

        power_loss = defect.estimated_power_loss

        # Determine severity based on power loss
        if power_loss >= thresholds["critical"]:
            severity = DefectSeverity.CRITICAL
        elif power_loss >= thresholds["high"]:
            severity = DefectSeverity.HIGH
        elif power_loss >= thresholds["medium"]:
            severity = DefectSeverity.MEDIUM
        else:
            severity = DefectSeverity.LOW

        # Adjust based on confidence (low confidence may reduce severity)
        if defect.confidence < 0.7 and severity != DefectSeverity.LOW:
            # Downgrade severity if confidence is low
            severity_order = [
                DefectSeverity.LOW,
                DefectSeverity.MEDIUM,
                DefectSeverity.HIGH,
                DefectSeverity.CRITICAL,
            ]
            current_idx = severity_order.index(severity)
            if current_idx > 0:
                severity = severity_order[current_idx - 1]

        return severity

    def repair_cost_estimation(
        self,
        defect: Defect,
        is_emergency: bool = False,
    ) -> float:
        """
        Estimate the cost to repair or remediate a defect.

        Calculates repair costs based on defect type, severity, required
        materials, and labor. Includes adjustments for emergency repairs.

        Args:
            defect: Defect for cost estimation
            is_emergency: Whether this is an emergency repair

        Returns:
            Estimated repair cost in USD

        Example:
            >>> generator = FaultReportGenerator()
            >>> defect = Defect(
            ...     type=DefectType.CRACK,
            ...     severity=DefectSeverity.HIGH,
            ...     ...
            ... )
            >>> cost = generator.repair_cost_estimation(defect)
            >>> print(f"Estimated cost: ${cost:.2f}")
        """
        base_cost = 0.0
        labor_hours = 0.0

        # Estimate cost based on defect type and severity
        if defect.type in [DefectType.CRACK, DefectType.CELL_BREAKAGE, DefectType.GLASS_BREAKAGE]:
            if defect.severity in [DefectSeverity.HIGH, DefectSeverity.CRITICAL]:
                # Requires panel replacement
                base_cost = self.cost_config.panel_replacement_cost
                labor_hours = 2.0
            else:
                # May require module replacement
                base_cost = self.cost_config.module_replacement_cost
                labor_hours = 1.0

        elif defect.type == DefectType.HOTSPOT:
            if defect.severity == DefectSeverity.CRITICAL:
                # Immediate replacement required
                base_cost = self.cost_config.panel_replacement_cost
                labor_hours = 2.0
            else:
                # Monitoring and inspection
                labor_hours = 0.5
                base_cost = 0.0

        elif defect.type == DefectType.SOILING:
            # Cleaning required
            base_cost = self.cost_config.cleaning_cost_per_panel
            labor_hours = 0.25

        elif defect.type == DefectType.DELAMINATION:
            if defect.severity in [DefectSeverity.HIGH, DefectSeverity.CRITICAL]:
                base_cost = self.cost_config.panel_replacement_cost
                labor_hours = 2.0
            else:
                # Monitoring
                labor_hours = 0.5

        elif defect.type in [DefectType.JUNCTION_BOX_FAILURE, DefectType.BACKSHEET_DAMAGE]:
            # Requires repair or replacement
            base_cost = self.cost_config.module_replacement_cost * 0.5
            labor_hours = 1.5

        else:
            # Default estimation
            severity_multiplier = {
                DefectSeverity.LOW: 0.25,
                DefectSeverity.MEDIUM: 0.5,
                DefectSeverity.HIGH: 0.75,
                DefectSeverity.CRITICAL: 1.0,
            }
            base_cost = (
                self.cost_config.panel_replacement_cost
                * severity_multiplier.get(defect.severity, 0.5)
            )
            labor_hours = 1.0

        # Calculate total cost
        labor_cost = labor_hours * self.cost_config.base_labor_rate
        total_cost = base_cost + labor_cost

        # Apply emergency multiplier if needed
        if is_emergency:
            total_cost *= self.cost_config.emergency_multiplier

        return round(total_cost, 2)

    def _is_severity_appropriate(self, defect: Defect) -> bool:
        """Check if the defect's current severity seems appropriate."""
        # This is a heuristic check
        return True  # For now, trust the input

    def _determine_root_cause(self, defect: Defect) -> str:
        """Determine the likely root cause of a defect."""
        root_causes = {
            DefectType.CRACK: "Mechanical stress, thermal cycling, or installation damage",
            DefectType.HOTSPOT: "Shading, cell mismatch, or reverse bias condition",
            DefectType.DELAMINATION: "Moisture ingress, UV degradation, or manufacturing defect",
            DefectType.PID: "High voltage stress and humidity",
            DefectType.SOILING: "Environmental dust, pollen, or bird droppings",
            DefectType.CORROSION: "Moisture ingress and material degradation",
            DefectType.SNAIL_TRAIL: "Silver paste degradation",
        }
        return root_causes.get(defect.type, "Requires further investigation")

    def _determine_recommended_action(self, defect: Defect) -> RecommendedAction:
        """Determine the recommended action for a defect."""
        if defect.severity == DefectSeverity.CRITICAL:
            return RecommendedAction.IMMEDIATE_REPAIR
        elif defect.severity == DefectSeverity.HIGH:
            if defect.type in [DefectType.CRACK, DefectType.HOTSPOT]:
                return RecommendedAction.REPLACE_PANEL
            return RecommendedAction.SCHEDULE_INSPECTION
        elif defect.severity == DefectSeverity.MEDIUM:
            return RecommendedAction.SCHEDULE_INSPECTION
        else:
            if defect.type == DefectType.SOILING:
                return RecommendedAction.CLEAN
            return RecommendedAction.MONITOR

    def _calculate_priority(self, defect: Defect) -> int:
        """Calculate priority (1-5, 1 being highest)."""
        severity_priority = {
            DefectSeverity.CRITICAL: 1,
            DefectSeverity.HIGH: 2,
            DefectSeverity.MEDIUM: 3,
            DefectSeverity.LOW: 4,
        }
        return severity_priority.get(defect.severity, 5)

    def _estimate_time_to_failure(self, defect: Defect) -> Optional[int]:
        """Estimate days until critical failure."""
        if defect.severity == DefectSeverity.CRITICAL:
            return 7  # Immediate attention within a week
        elif defect.severity == DefectSeverity.HIGH:
            return 30  # Within a month
        elif defect.severity == DefectSeverity.MEDIUM:
            return 90  # Within a quarter
        return None  # No immediate failure expected

    def _generate_analysis_notes(self, defect: Defect) -> str:
        """Generate analysis notes for a defect."""
        notes = [
            f"Defect type: {defect.type.value}",
            f"Severity: {defect.severity.value}",
            f"Confidence: {defect.confidence:.2f}",
            f"Estimated power loss: {defect.estimated_power_loss:.2f}%",
        ]
        if defect.description:
            notes.append(f"Description: {defect.description}")
        return " | ".join(notes)

    def _generate_executive_summary(
        self,
        categorized_defects: Dict[str, List[Defect]],
        total_defects: int,
        critical_count: int,
        total_cost: float,
        total_power_loss: float,
    ) -> str:
        """Generate an executive summary for the report."""
        summary_parts = [
            f"Total defects identified: {total_defects}",
            f"Critical defects: {critical_count}",
            f"Estimated total repair cost: ${total_cost:,.2f}",
            f"Estimated total power loss: {total_power_loss:.2f}%",
            "",
            "Defects by category:",
        ]

        for category, defects in categorized_defects.items():
            summary_parts.append(f"  - {category.capitalize()}: {len(defects)}")

        return "\n".join(summary_parts)

    def _generate_recommendations(
        self,
        diagnostics: List[DiagnosticResult],
        categorized_defects: Dict[str, List[Defect]],
    ) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []

        # Count critical and high severity defects
        critical_count = sum(
            1 for d in diagnostics if d.defect and d.defect.severity == DefectSeverity.CRITICAL
        )
        high_count = sum(
            1 for d in diagnostics if d.defect and d.defect.severity == DefectSeverity.HIGH
        )

        if critical_count > 0:
            recommendations.append(
                f"URGENT: {critical_count} critical defect(s) require immediate attention "
                "to prevent system failure or safety hazards."
            )

        if high_count > 5:
            recommendations.append(
                f"{high_count} high-severity defects identified. "
                "Recommend scheduling comprehensive maintenance within 30 days."
            )

        # Category-specific recommendations
        if "electrical" in categorized_defects and len(categorized_defects["electrical"]) > 3:
            recommendations.append(
                "Multiple electrical defects detected. "
                "Consider comprehensive electrical system inspection."
            )

        if "structural" in categorized_defects and len(categorized_defects["structural"]) > 5:
            recommendations.append(
                "High number of structural defects may indicate installation "
                "or environmental issues. Recommend root cause analysis."
            )

        if not recommendations:
            recommendations.append(
                "Continue routine monitoring and maintenance schedule. "
                "No immediate critical actions required."
            )

        return recommendations
