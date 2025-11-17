"""Safety qualification report generation per IEC 61730.

This module provides comprehensive reporting capabilities for safety test results,
including test documentation, pass/fail criteria, safety certificates, and
integration with certification bodies.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from loguru import logger
from fpdf import FPDF

from ..models.safety_models import (
    SafetyTestResult,
    SafetyCertificate,
    TestStatus,
    SafetyClass,
    FireClass,
    ApplicationClass,
)


class SafetyQualificationReport:
    """Generates comprehensive safety qualification reports and certificates.

    This class creates detailed documentation of safety test results, including:
    - Comprehensive test result summaries
    - Pass/fail criteria for each test
    - Safety classification documentation
    - Compliance certificates
    - Integration with certification bodies (TUV, UL, IEC, etc.)
    - Export to PDF and JSON formats

    Attributes:
        test_results: Complete safety test results.
        certificate: Safety certificate (if module passed).
        report_date: Date of report generation.
    """

    def __init__(
        self,
        test_results: SafetyTestResult,
        certificate: Optional[SafetyCertificate] = None,
    ) -> None:
        """Initialize safety qualification report generator.

        Args:
            test_results: Complete safety test results to document.
            certificate: Optional safety certificate if module passed all tests.
        """
        self.test_results = test_results
        self.certificate = certificate
        self.report_date = datetime.now()

        logger.info(
            f"Initialized SafetyQualificationReport for module "
            f"{test_results.config.module_id}"
        )

    def generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of test results.

        Creates a high-level summary suitable for quick review, including:
        - Overall pass/fail status
        - Safety classification
        - Critical test results
        - Any failures or warnings

        Returns:
            Dictionary containing summary information.
        """
        logger.info("Generating test results summary")

        summary = {
            "module_id": self.test_results.config.module_id,
            "manufacturer": self.test_results.config.manufacturer,
            "model_number": self.test_results.config.model_number,
            "test_date": self.test_results.config.test_date.isoformat(),
            "overall_status": "PASS" if self.test_results.overall_pass else "FAIL",
            "safety_classification": None,
            "test_results_summary": {},
            "failures": [],
            "warnings": [],
        }

        # Add classification if available
        if self.test_results.classification:
            summary["safety_classification"] = {
                "safety_class": self.test_results.classification.safety_class.value,
                "application_class": self.test_results.classification.application_class.value,
                "fire_class": self.test_results.classification.fire_class.value,
                "max_system_voltage_v": self.test_results.classification.max_system_voltage_v,
            }

        # Summarize test results
        if self.test_results.electrical_tests:
            summary["test_results_summary"]["electrical"] = (
                "PASS" if self.test_results.electrical_tests.all_tests_passed
                else "FAIL"
            )
            if not self.test_results.electrical_tests.all_tests_passed:
                summary["failures"].append("Electrical safety tests failed")

        if self.test_results.mechanical_tests:
            summary["test_results_summary"]["mechanical"] = (
                "PASS" if self.test_results.mechanical_tests.all_tests_passed
                else "FAIL"
            )
            if not self.test_results.mechanical_tests.all_tests_passed:
                summary["failures"].append("Mechanical safety tests failed")

        if self.test_results.fire_tests:
            summary["test_results_summary"]["fire"] = (
                self.test_results.fire_tests.fire_classification.value
            )
            if self.test_results.fire_tests.fire_classification == FireClass.NOT_RATED:
                summary["warnings"].append("Fire classification: Not Rated")

        if self.test_results.environmental_tests:
            summary["test_results_summary"]["environmental"] = (
                "PASS" if self.test_results.environmental_tests.all_tests_passed
                else "FAIL"
            )
            if not self.test_results.environmental_tests.all_tests_passed:
                summary["failures"].append("Environmental safety tests failed")

        # Check construction requirements
        if self.test_results.construction_requirements:
            non_compliant = [
                req for req in self.test_results.construction_requirements
                if not req.compliant
            ]
            if non_compliant:
                summary["failures"].append(
                    f"{len(non_compliant)} construction requirement(s) not met"
                )
                summary["test_results_summary"]["construction"] = "FAIL"
            else:
                summary["test_results_summary"]["construction"] = "PASS"

        logger.info(f"Summary generated: {summary['overall_status']}")
        return summary

    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive detailed test report.

        Creates a complete report with all test data, measurements, and results.

        Returns:
            Dictionary containing detailed test information.
        """
        logger.info("Generating detailed test report")

        report = {
            "report_metadata": {
                "report_date": self.report_date.isoformat(),
                "standard": "IEC 61730-1:2016 and IEC 61730-2:2016",
                "test_laboratory": self.test_results.config.test_laboratory,
            },
            "module_information": {
                "module_id": self.test_results.config.module_id,
                "manufacturer": self.test_results.config.manufacturer,
                "model_number": self.test_results.config.model_number,
                "serial_number": self.test_results.config.serial_number,
                "module_area_m2": self.test_results.config.module_area_m2,
                "max_system_voltage_v": self.test_results.config.max_system_voltage_v,
            },
            "electrical_tests": self._format_electrical_tests(),
            "mechanical_tests": self._format_mechanical_tests(),
            "fire_tests": self._format_fire_tests(),
            "environmental_tests": self._format_environmental_tests(),
            "construction_requirements": self._format_construction_requirements(),
            "safety_classification": self._format_classification(),
            "overall_assessment": {
                "pass": self.test_results.overall_pass,
                "completion_date": (
                    self.test_results.test_completion_date.isoformat()
                    if self.test_results.test_completion_date else None
                ),
                "notes": self.test_results.notes,
            },
        }

        if self.certificate:
            report["certificate"] = self._format_certificate()

        logger.info("Detailed report generated")
        return report

    def export_to_json(self, filepath: Path) -> None:
        """Export detailed report to JSON file.

        Args:
            filepath: Path where JSON file should be saved.
        """
        logger.info(f"Exporting report to JSON: {filepath}")

        report = self.generate_detailed_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved: {filepath}")

    def export_to_pdf(self, filepath: Path) -> None:
        """Export comprehensive report to PDF file.

        Creates a formatted PDF document with:
        - Title page with module information
        - Test results summary
        - Detailed test results for each category
        - Pass/fail criteria
        - Safety classification
        - Certificate information (if applicable)

        Args:
            filepath: Path where PDF file should be saved.
        """
        logger.info(f"Generating PDF report: {filepath}")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        self._add_title_page(pdf)

        # Summary
        self._add_summary_section(pdf)

        # Electrical tests
        if self.test_results.electrical_tests:
            self._add_electrical_tests_section(pdf)

        # Mechanical tests
        if self.test_results.mechanical_tests:
            self._add_mechanical_tests_section(pdf)

        # Fire tests
        if self.test_results.fire_tests:
            self._add_fire_tests_section(pdf)

        # Environmental tests
        if self.test_results.environmental_tests:
            self._add_environmental_tests_section(pdf)

        # Construction requirements
        if self.test_results.construction_requirements:
            self._add_construction_requirements_section(pdf)

        # Classification
        if self.test_results.classification:
            self._add_classification_section(pdf)

        # Certificate
        if self.certificate:
            self._add_certificate_section(pdf)

        # Save PDF
        pdf.output(str(filepath))
        logger.info(f"PDF report saved: {filepath}")

    # Private helper methods for formatting

    def _format_electrical_tests(self) -> Optional[Dict[str, Any]]:
        """Format electrical test results for report."""
        if not self.test_results.electrical_tests:
            return None

        tests = self.test_results.electrical_tests
        formatted = {"overall_pass": tests.all_tests_passed, "tests": {}}

        if tests.insulation_resistance:
            formatted["tests"]["insulation_resistance"] = {
                "status": tests.insulation_resistance.status.value,
                "test_voltage_v": tests.insulation_resistance.test_voltage_v,
                "measured_resistance_mohm": tests.insulation_resistance.measured_resistance_mohm,
                "minimum_required_mohm": tests.insulation_resistance.minimum_required_mohm,
                "passed": tests.insulation_resistance.passed,
            }

        if tests.wet_leakage_current:
            formatted["tests"]["wet_leakage_current"] = {
                "status": tests.wet_leakage_current.status.value,
                "leakage_current_ua": tests.wet_leakage_current.leakage_current_ua,
                "maximum_allowed_ua": tests.wet_leakage_current.maximum_allowed_ua,
                "passed": tests.wet_leakage_current.passed,
            }

        if tests.dielectric_strength:
            formatted["tests"]["dielectric_strength"] = {
                "status": tests.dielectric_strength.status.value,
                "test_voltage_v": tests.dielectric_strength.test_voltage_v,
                "breakdown_occurred": tests.dielectric_strength.breakdown_occurred,
                "passed": tests.dielectric_strength.passed,
            }

        if tests.ground_continuity:
            formatted["tests"]["ground_continuity"] = {
                "status": tests.ground_continuity.status.value,
                "measured_resistance_ohm": tests.ground_continuity.measured_resistance_ohm,
                "maximum_allowed_ohm": tests.ground_continuity.maximum_allowed_ohm,
                "passed": tests.ground_continuity.passed,
            }

        if tests.bypass_diode_thermal:
            formatted["tests"]["bypass_diode_thermal"] = {
                "status": tests.bypass_diode_thermal.status.value,
                "peak_temperature_c": tests.bypass_diode_thermal.peak_temperature_c,
                "maximum_allowed_c": tests.bypass_diode_thermal.maximum_allowed_c,
                "thermal_runaway": tests.bypass_diode_thermal.thermal_runaway_detected,
                "passed": tests.bypass_diode_thermal.passed,
            }

        return formatted

    def _format_mechanical_tests(self) -> Optional[Dict[str, Any]]:
        """Format mechanical test results for report."""
        if not self.test_results.mechanical_tests:
            return None

        tests = self.test_results.mechanical_tests
        formatted = {"overall_pass": tests.all_tests_passed, "tests": {}}

        if tests.mechanical_load:
            formatted["tests"]["mechanical_load"] = {
                "status": tests.mechanical_load.status.value,
                "applied_load_pa": tests.mechanical_load.applied_load_pa,
                "max_deflection_mm": tests.mechanical_load.maximum_deflection_mm,
                "visual_defects": tests.mechanical_load.visual_defects_found,
                "passed": tests.mechanical_load.passed,
            }

        if tests.impact:
            formatted["tests"]["impact"] = {
                "status": tests.impact.status.value,
                "ice_ball_diameter_mm": tests.impact.ice_ball_diameter_mm,
                "impact_velocity_ms": tests.impact.impact_velocity_ms,
                "cracks_detected": tests.impact.cracks_detected,
                "electrical_safety_maintained": tests.impact.electrical_safety_maintained,
                "passed": tests.impact.passed,
            }

        if tests.robustness_of_terminations:
            formatted["tests"]["robustness_of_terminations"] = {
                "status": tests.robustness_of_terminations.status.value,
                "pull_force_n": tests.robustness_of_terminations.pull_force_n,
                "cable_displaced": tests.robustness_of_terminations.cable_displaced,
                "terminal_damaged": tests.robustness_of_terminations.terminal_damaged,
                "passed": tests.robustness_of_terminations.passed,
            }

        return formatted

    def _format_fire_tests(self) -> Optional[Dict[str, Any]]:
        """Format fire test results for report."""
        if not self.test_results.fire_tests:
            return None

        tests = self.test_results.fire_tests
        formatted = {
            "fire_classification": tests.fire_classification.value,
            "tests": {}
        }

        if tests.spread_of_flame:
            formatted["tests"]["spread_of_flame"] = {
                "status": tests.spread_of_flame.status.value,
                "flame_spread_distance_cm": tests.spread_of_flame.flame_spread_distance_cm,
                "sustained_flaming": tests.spread_of_flame.sustained_flaming_observed,
                "roof_deck_penetration": tests.spread_of_flame.roof_deck_penetration,
            }

        if tests.fire_penetration:
            formatted["tests"]["fire_penetration"] = {
                "status": tests.fire_penetration.status.value,
                "burn_through": tests.fire_penetration.burn_through_occurred,
                "roof_deck_damage": tests.fire_penetration.roof_deck_damage,
            }

        if tests.fire_brand:
            formatted["tests"]["fire_brand"] = {
                "status": tests.fire_brand.status.value,
                "brand_size_class": tests.fire_brand.brand_size_class,
                "ignition_occurred": tests.fire_brand.ignition_occurred,
                "sustained_burning": tests.fire_brand.sustained_burning,
            }

        return formatted

    def _format_environmental_tests(self) -> Optional[Dict[str, Any]]:
        """Format environmental test results for report."""
        if not self.test_results.environmental_tests:
            return None

        tests = self.test_results.environmental_tests
        formatted = {"overall_pass": tests.all_tests_passed, "tests": {}}

        if tests.uv_preconditioning:
            formatted["tests"]["uv_preconditioning"] = {
                "status": tests.uv_preconditioning.status.value,
                "uv_dose_kwh_m2": tests.uv_preconditioning.uv_dose_kwh_m2,
                "required_dose_kwh_m2": tests.uv_preconditioning.required_dose_kwh_m2,
                "passed": tests.uv_preconditioning.passed,
            }

        if tests.thermal_cycling:
            formatted["tests"]["thermal_cycling"] = {
                "status": tests.thermal_cycling.status.value,
                "cycles_completed": tests.thermal_cycling.cycles_completed,
                "required_cycles": tests.thermal_cycling.required_cycles,
                "electrical_failure": tests.thermal_cycling.electrical_failure,
                "passed": tests.thermal_cycling.passed,
            }

        if tests.humidity_freeze:
            formatted["tests"]["humidity_freeze"] = {
                "status": tests.humidity_freeze.status.value,
                "cycles_completed": tests.humidity_freeze.cycles_completed,
                "required_cycles": tests.humidity_freeze.required_cycles,
                "electrical_failure": tests.humidity_freeze.electrical_failure,
                "passed": tests.humidity_freeze.passed,
            }

        return formatted

    def _format_construction_requirements(self) -> List[Dict[str, Any]]:
        """Format construction requirements for report."""
        return [
            {
                "requirement_id": req.requirement_id,
                "description": req.requirement_description,
                "compliant": req.compliant,
                "notes": req.notes,
            }
            for req in self.test_results.construction_requirements
        ]

    def _format_classification(self) -> Optional[Dict[str, Any]]:
        """Format safety classification for report."""
        if not self.test_results.classification:
            return None

        return {
            "safety_class": self.test_results.classification.safety_class.value,
            "application_class": self.test_results.classification.application_class.value,
            "fire_class": self.test_results.classification.fire_class.value,
            "max_system_voltage_v": self.test_results.classification.max_system_voltage_v,
            "rationale": self.test_results.classification.classification_rationale,
        }

    def _format_certificate(self) -> Dict[str, Any]:
        """Format certificate information for report."""
        return {
            "certificate_number": self.certificate.certificate_number,
            "issue_date": self.certificate.issue_date.isoformat(),
            "expiry_date": (
                self.certificate.expiry_date.isoformat()
                if self.certificate.expiry_date else None
            ),
            "certification_body": self.certificate.certification_body,
            "certified_safety_class": self.certificate.certified_safety_class.value,
            "certified_application_class": self.certificate.certified_application_class.value,
            "certified_fire_class": (
                self.certificate.certified_fire_class.value
                if self.certificate.certified_fire_class else None
            ),
        }

    # PDF generation helper methods

    def _add_title_page(self, pdf: FPDF) -> None:
        """Add title page to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, 'IEC 61730 Safety Test Report', 0, 1, 'C')

        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f'Module: {self.test_results.config.model_number}', 0, 1, 'C')
        pdf.cell(0, 10, f'Manufacturer: {self.test_results.config.manufacturer}', 0, 1, 'C')
        pdf.cell(0, 10, f'Test Date: {self.test_results.config.test_date.strftime("%Y-%m-%d")}', 0, 1, 'C')

        pdf.ln(20)
        pdf.set_font('Arial', 'B', 16)
        status_text = 'PASS' if self.test_results.overall_pass else 'FAIL'
        pdf.cell(0, 10, f'Overall Status: {status_text}', 0, 1, 'C')

    def _add_summary_section(self, pdf: FPDF) -> None:
        """Add summary section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 10)
        summary = self.generate_summary()

        for key, value in summary['test_results_summary'].items():
            pdf.cell(0, 8, f'{key.capitalize()}: {value}', 0, 1)

    def _add_electrical_tests_section(self, pdf: FPDF) -> None:
        """Add electrical tests section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Electrical Safety Tests', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 10)
        formatted = self._format_electrical_tests()
        if formatted:
            for test_name, test_data in formatted['tests'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, test_name.replace('_', ' ').title(), 0, 1)
                pdf.set_font('Arial', '', 10)
                for key, value in test_data.items():
                    pdf.cell(0, 6, f'  {key}: {value}', 0, 1)
                pdf.ln(3)

    def _add_mechanical_tests_section(self, pdf: FPDF) -> None:
        """Add mechanical tests section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Mechanical Safety Tests', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 10)
        formatted = self._format_mechanical_tests()
        if formatted:
            for test_name, test_data in formatted['tests'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, test_name.replace('_', ' ').title(), 0, 1)
                pdf.set_font('Arial', '', 10)
                for key, value in test_data.items():
                    pdf.cell(0, 6, f'  {key}: {value}', 0, 1)
                pdf.ln(3)

    def _add_fire_tests_section(self, pdf: FPDF) -> None:
        """Add fire tests section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Fire Safety Tests', 0, 1)
        pdf.ln(5)

        formatted = self._format_fire_tests()
        if formatted:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f"Fire Classification: {formatted['fire_classification']}", 0, 1)
            pdf.ln(3)

            pdf.set_font('Arial', '', 10)
            for test_name, test_data in formatted['tests'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, test_name.replace('_', ' ').title(), 0, 1)
                pdf.set_font('Arial', '', 10)
                for key, value in test_data.items():
                    pdf.cell(0, 6, f'  {key}: {value}', 0, 1)
                pdf.ln(3)

    def _add_environmental_tests_section(self, pdf: FPDF) -> None:
        """Add environmental tests section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Environmental Safety Tests', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 10)
        formatted = self._format_environmental_tests()
        if formatted:
            for test_name, test_data in formatted['tests'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, test_name.replace('_', ' ').title(), 0, 1)
                pdf.set_font('Arial', '', 10)
                for key, value in test_data.items():
                    pdf.cell(0, 6, f'  {key}: {value}', 0, 1)
                pdf.ln(3)

    def _add_construction_requirements_section(self, pdf: FPDF) -> None:
        """Add construction requirements section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Construction Requirements (IEC 61730-1)', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 9)
        for req in self.test_results.construction_requirements:
            status = 'PASS' if req.compliant else 'FAIL'
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 7, f'{req.requirement_id}: {status}', 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 5, f'{req.requirement_description}')
            if req.notes:
                pdf.multi_cell(0, 5, f'Notes: {req.notes}')
            pdf.ln(2)

    def _add_classification_section(self, pdf: FPDF) -> None:
        """Add classification section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Safety Classification', 0, 1)
        pdf.ln(5)

        if self.test_results.classification:
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 8, f"Safety Class: {self.test_results.classification.safety_class.value}", 0, 1)
            pdf.cell(0, 8, f"Application Class: {self.test_results.classification.application_class.value}", 0, 1)
            pdf.cell(0, 8, f"Fire Class: {self.test_results.classification.fire_class.value}", 0, 1)
            pdf.cell(0, 8, f"Max System Voltage: {self.test_results.classification.max_system_voltage_v}V", 0, 1)
            pdf.ln(5)
            pdf.multi_cell(0, 6, f"Rationale: {self.test_results.classification.classification_rationale}")

    def _add_certificate_section(self, pdf: FPDF) -> None:
        """Add certificate section to PDF."""
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 15, 'IEC 61730 Safety Certificate', 0, 1, 'C')
        pdf.ln(10)

        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f"Certificate Number: {self.certificate.certificate_number}", 0, 1)
        pdf.cell(0, 8, f"Certification Body: {self.certificate.certification_body}", 0, 1)
        pdf.cell(0, 8, f"Issue Date: {self.certificate.issue_date.strftime('%Y-%m-%d')}", 0, 1)
        if self.certificate.expiry_date:
            pdf.cell(0, 8, f"Expiry Date: {self.certificate.expiry_date.strftime('%Y-%m-%d')}", 0, 1)

        pdf.ln(5)
        pdf.cell(0, 8, f"Certified Safety Class: {self.certificate.certified_safety_class.value}", 0, 1)
        pdf.cell(0, 8, f"Certified Application Class: {self.certificate.certified_application_class.value}", 0, 1)
        if self.certificate.certified_fire_class:
            pdf.cell(0, 8, f"Certified Fire Class: {self.certificate.certified_fire_class.value}", 0, 1)
