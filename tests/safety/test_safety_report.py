"""Unit tests for safety qualification report generation.

Tests the SafetyQualificationReport class for report generation and export.
"""

from datetime import datetime
from pathlib import Path
import json
import pytest

from src.safety.iec61730_tester import IEC61730SafetyTester
from src.safety.safety_report import SafetyQualificationReport
from src.models.safety_models import (
    SafetyTestConfig,
    SafetyClass,
    ApplicationClass,
    FireClass,
)


class TestSafetyQualificationReport:
    """Test suite for SafetyQualificationReport class."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration fixture."""
        return SafetyTestConfig(
            module_id="TEST-001",
            manufacturer="TestCorp",
            model_number="TC-400",
            serial_number="SN-001",
            max_system_voltage_v=1000.0,
            module_area_m2=2.0,
            application_class=ApplicationClass.CLASS_B,
            target_safety_class=SafetyClass.CLASS_II,
            target_fire_class=FireClass.CLASS_A,
            test_laboratory="Test Lab",
            test_date=datetime.now(),
            perform_electrical_tests=True,
            perform_mechanical_tests=True,
            perform_fire_tests=True,
            perform_environmental_tests=True,
        )

    @pytest.fixture
    def test_results(self, test_config):
        """Create test results fixture."""
        tester = IEC61730SafetyTester(test_config)
        return tester.run_all_tests()

    @pytest.fixture
    def test_certificate(self, test_config, test_results):
        """Create test certificate fixture (only if tests passed)."""
        if test_results.overall_pass:
            tester = IEC61730SafetyTester(test_config)
            tester.results = test_results
            return tester.export_safety_certificate()
        return None

    @pytest.fixture
    def report_generator(self, test_results, test_certificate):
        """Create report generator fixture."""
        return SafetyQualificationReport(
            test_results=test_results,
            certificate=test_certificate,
        )

    def test_initialization(self, report_generator, test_results):
        """Test report generator initialization."""
        assert report_generator.test_results == test_results
        assert report_generator.report_date is not None

    def test_generate_summary(self, report_generator):
        """Test summary generation."""
        summary = report_generator.generate_summary()

        assert summary is not None
        assert "module_id" in summary
        assert "manufacturer" in summary
        assert "model_number" in summary
        assert "test_date" in summary
        assert "overall_status" in summary
        assert "test_results_summary" in summary
        assert "failures" in summary
        assert "warnings" in summary

        assert summary["overall_status"] in ["PASS", "FAIL"]

    def test_generate_summary_with_classification(self, report_generator):
        """Test summary generation with safety classification."""
        summary = report_generator.generate_summary()

        if report_generator.test_results.classification:
            assert "safety_classification" in summary
            assert summary["safety_classification"] is not None
            assert "safety_class" in summary["safety_classification"]
            assert "application_class" in summary["safety_classification"]
            assert "fire_class" in summary["safety_classification"]

    def test_generate_detailed_report(self, report_generator):
        """Test detailed report generation."""
        report = report_generator.generate_detailed_report()

        assert report is not None
        assert "report_metadata" in report
        assert "module_information" in report
        assert "electrical_tests" in report
        assert "mechanical_tests" in report
        assert "fire_tests" in report
        assert "environmental_tests" in report
        assert "construction_requirements" in report
        assert "safety_classification" in report
        assert "overall_assessment" in report

    def test_generate_detailed_report_metadata(self, report_generator):
        """Test detailed report metadata section."""
        report = report_generator.generate_detailed_report()

        metadata = report["report_metadata"]
        assert "report_date" in metadata
        assert "standard" in metadata
        assert "IEC 61730" in metadata["standard"]
        assert "test_laboratory" in metadata

    def test_generate_detailed_report_module_info(self, report_generator):
        """Test detailed report module information section."""
        report = report_generator.generate_detailed_report()

        module_info = report["module_information"]
        assert "module_id" in module_info
        assert "manufacturer" in module_info
        assert "model_number" in module_info
        assert "module_area_m2" in module_info
        assert "max_system_voltage_v" in module_info

    def test_export_to_json(self, report_generator, tmp_path):
        """Test JSON export functionality."""
        json_path = tmp_path / "test_report.json"

        report_generator.export_to_json(json_path)

        assert json_path.exists()

        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)

        assert "report_metadata" in data
        assert "module_information" in data
        assert "overall_assessment" in data

    def test_export_to_pdf(self, report_generator, tmp_path):
        """Test PDF export functionality."""
        pdf_path = tmp_path / "test_report.pdf"

        report_generator.export_to_pdf(pdf_path)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0  # File is not empty

    def test_format_electrical_tests(self, report_generator):
        """Test electrical tests formatting."""
        formatted = report_generator._format_electrical_tests()

        if formatted:
            assert "overall_pass" in formatted
            assert "tests" in formatted
            assert isinstance(formatted["tests"], dict)

    def test_format_mechanical_tests(self, report_generator):
        """Test mechanical tests formatting."""
        formatted = report_generator._format_mechanical_tests()

        if formatted:
            assert "overall_pass" in formatted
            assert "tests" in formatted
            assert isinstance(formatted["tests"], dict)

    def test_format_fire_tests(self, report_generator):
        """Test fire tests formatting."""
        formatted = report_generator._format_fire_tests()

        if formatted:
            assert "fire_classification" in formatted
            assert "tests" in formatted
            assert isinstance(formatted["tests"], dict)

    def test_format_environmental_tests(self, report_generator):
        """Test environmental tests formatting."""
        formatted = report_generator._format_environmental_tests()

        if formatted:
            assert "overall_pass" in formatted
            assert "tests" in formatted
            assert isinstance(formatted["tests"], dict)

    def test_format_construction_requirements(self, report_generator):
        """Test construction requirements formatting."""
        formatted = report_generator._format_construction_requirements()

        assert isinstance(formatted, list)

        if len(formatted) > 0:
            req = formatted[0]
            assert "requirement_id" in req
            assert "description" in req
            assert "compliant" in req

    def test_format_classification(self, report_generator):
        """Test classification formatting."""
        formatted = report_generator._format_classification()

        if formatted:
            assert "safety_class" in formatted
            assert "application_class" in formatted
            assert "fire_class" in formatted
            assert "max_system_voltage_v" in formatted
            assert "rationale" in formatted

    def test_format_certificate(self, report_generator):
        """Test certificate formatting."""
        if report_generator.certificate:
            formatted = report_generator._format_certificate()

            assert formatted is not None
            assert "certificate_number" in formatted
            assert "issue_date" in formatted
            assert "certification_body" in formatted
            assert "certified_safety_class" in formatted
            assert "certified_application_class" in formatted

    def test_report_with_failed_tests(self, test_config):
        """Test report generation with failed tests."""
        # Create a configuration that might fail some tests
        tester = IEC61730SafetyTester(test_config)
        results = tester.run_all_tests()

        report_gen = SafetyQualificationReport(
            test_results=results,
            certificate=None,
        )

        summary = report_gen.generate_summary()

        assert summary is not None
        assert "overall_status" in summary

        if not results.overall_pass:
            assert summary["overall_status"] == "FAIL"
            assert len(summary["failures"]) > 0 or len(summary["warnings"]) > 0

    def test_pdf_generation_all_sections(self, report_generator, tmp_path):
        """Test that PDF includes all sections."""
        pdf_path = tmp_path / "full_report.pdf"

        # Generate PDF
        report_generator.export_to_pdf(pdf_path)

        # Verify file exists and has content
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000  # Should be reasonably sized

    def test_json_export_complete_data(self, report_generator, tmp_path):
        """Test that JSON export includes complete data."""
        json_path = tmp_path / "complete_report.json"

        report_generator.export_to_json(json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Verify all major sections are present
        assert "report_metadata" in data
        assert "module_information" in data
        assert "electrical_tests" in data
        assert "mechanical_tests" in data
        assert "fire_tests" in data
        assert "environmental_tests" in data
        assert "construction_requirements" in data
        assert "overall_assessment" in data

    def test_summary_failures_list(self, report_generator):
        """Test that failures are properly listed in summary."""
        summary = report_generator.generate_summary()

        assert "failures" in summary
        assert isinstance(summary["failures"], list)

        # If tests failed, failures list should not be empty
        if not report_generator.test_results.overall_pass:
            assert len(summary["failures"]) > 0

    def test_summary_warnings_list(self, report_generator):
        """Test that warnings are properly listed in summary."""
        summary = report_generator.generate_summary()

        assert "warnings" in summary
        assert isinstance(summary["warnings"], list)
