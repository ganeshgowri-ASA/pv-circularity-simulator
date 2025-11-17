"""
Tests for FaultReportGenerator (B08-S04).
"""

from datetime import datetime
from typing import List
import pytest

from src.pv_circularity.models import (
    Defect,
    DefectType,
    DefectSeverity,
    FaultReport,
)
from src.pv_circularity.b08_diagnostics.b08_s04 import (
    FaultReportGenerator,
    CostEstimationConfig,
)


class TestFaultReportGenerator:
    """Test suite for FaultReportGenerator."""

    def test_initialization(self):
        """Test that FaultReportGenerator initializes correctly."""
        generator = FaultReportGenerator()
        assert generator is not None
        assert generator.cost_config is not None
        assert isinstance(generator.cost_config, CostEstimationConfig)

    def test_initialization_with_custom_config(self):
        """Test initialization with custom cost configuration."""
        config = CostEstimationConfig(
            base_labor_rate=100.0,
            panel_replacement_cost=400.0,
        )
        generator = FaultReportGenerator(cost_config=config)
        assert generator.cost_config.base_labor_rate == 100.0
        assert generator.cost_config.panel_replacement_cost == 400.0

    def test_automated_report_generation(self, sample_defects: List[Defect]):
        """Test automated report generation."""
        generator = FaultReportGenerator()
        report = generator.automated_report_generation(
            site_id="SITE-001",
            defects=sample_defects,
        )

        assert isinstance(report, FaultReport)
        assert report.site_id == "SITE-001"
        assert report.total_defects == len(sample_defects)
        assert len(report.diagnostics) == len(sample_defects)
        assert report.estimated_total_cost > 0
        assert len(report.recommendations) > 0

    def test_automated_report_generation_with_custom_title(self, sample_defects: List[Defect]):
        """Test report generation with custom title."""
        generator = FaultReportGenerator()
        custom_title = "Custom Test Report"
        report = generator.automated_report_generation(
            site_id="SITE-001",
            defects=sample_defects,
            report_title=custom_title,
        )

        assert report.report_title == custom_title

    def test_defect_categorization(self, sample_defects: List[Defect]):
        """Test defect categorization."""
        generator = FaultReportGenerator()
        categorized = generator.defect_categorization(sample_defects)

        assert isinstance(categorized, dict)
        assert len(categorized) > 0

        # Check that all defects are categorized
        total_categorized = sum(len(defects) for defects in categorized.values())
        assert total_categorized == len(sample_defects)

    def test_severity_assessment_low(self, sample_defect: Defect):
        """Test severity assessment for low severity defect."""
        generator = FaultReportGenerator()
        sample_defect.estimated_power_loss = 1.0

        severity = generator.severity_assessment(sample_defect)
        assert severity == DefectSeverity.LOW

    def test_severity_assessment_medium(self, sample_defect: Defect):
        """Test severity assessment for medium severity defect."""
        generator = FaultReportGenerator()
        sample_defect.estimated_power_loss = 4.0

        severity = generator.severity_assessment(sample_defect)
        assert severity == DefectSeverity.MEDIUM

    def test_severity_assessment_high(self, sample_defect: Defect):
        """Test severity assessment for high severity defect."""
        generator = FaultReportGenerator()
        sample_defect.estimated_power_loss = 8.0

        severity = generator.severity_assessment(sample_defect)
        assert severity == DefectSeverity.HIGH

    def test_severity_assessment_critical(self, sample_defect: Defect):
        """Test severity assessment for critical severity defect."""
        generator = FaultReportGenerator()
        sample_defect.estimated_power_loss = 15.0

        severity = generator.severity_assessment(sample_defect)
        assert severity == DefectSeverity.CRITICAL

    def test_repair_cost_estimation_crack(self, sample_defect: Defect):
        """Test repair cost estimation for crack."""
        generator = FaultReportGenerator()
        sample_defect.type = DefectType.CRACK
        sample_defect.severity = DefectSeverity.HIGH

        cost = generator.repair_cost_estimation(sample_defect)
        assert cost > 0
        assert isinstance(cost, float)

    def test_repair_cost_estimation_hotspot(self, sample_defect: Defect):
        """Test repair cost estimation for hotspot."""
        generator = FaultReportGenerator()
        sample_defect.type = DefectType.HOTSPOT
        sample_defect.severity = DefectSeverity.CRITICAL

        cost = generator.repair_cost_estimation(sample_defect)
        assert cost > 0

    def test_repair_cost_estimation_soiling(self, sample_defect: Defect):
        """Test repair cost estimation for soiling."""
        generator = FaultReportGenerator()
        sample_defect.type = DefectType.SOILING
        sample_defect.severity = DefectSeverity.LOW

        cost = generator.repair_cost_estimation(sample_defect)
        assert cost > 0
        # Soiling should be cheaper than panel replacement
        assert cost < 100

    def test_repair_cost_estimation_emergency(self, sample_defect: Defect):
        """Test emergency repair cost estimation."""
        generator = FaultReportGenerator()
        sample_defect.type = DefectType.CRACK
        sample_defect.severity = DefectSeverity.HIGH

        normal_cost = generator.repair_cost_estimation(sample_defect, is_emergency=False)
        emergency_cost = generator.repair_cost_estimation(sample_defect, is_emergency=True)

        assert emergency_cost > normal_cost
        assert emergency_cost == normal_cost * generator.cost_config.emergency_multiplier

    def test_empty_defects_list(self):
        """Test report generation with empty defects list."""
        generator = FaultReportGenerator()
        report = generator.automated_report_generation(
            site_id="SITE-001",
            defects=[],
        )

        assert report.total_defects == 0
        assert report.critical_defects == 0
        assert report.estimated_total_cost == 0.0

    def test_critical_defects_recommendations(self, sample_defects: List[Defect]):
        """Test that critical defects generate appropriate recommendations."""
        generator = FaultReportGenerator()

        # Make some defects critical
        for defect in sample_defects[:2]:
            defect.severity = DefectSeverity.CRITICAL

        report = generator.automated_report_generation(
            site_id="SITE-001",
            defects=sample_defects,
        )

        assert report.critical_defects == 2
        # Should have recommendations due to critical defects
        assert len(report.recommendations) > 0
        # Check that recommendations mention critical defects
        recommendations_text = " ".join(report.recommendations).lower()
        assert "critical" in recommendations_text or "urgent" in recommendations_text
