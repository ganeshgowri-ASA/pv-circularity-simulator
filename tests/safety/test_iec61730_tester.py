"""Unit tests for IEC 61730 safety tester orchestrator.

Tests the main IEC61730SafetyTester class that coordinates all safety testing.
"""

from datetime import datetime
import pytest

from src.safety.iec61730_tester import IEC61730SafetyTester
from src.models.safety_models import (
    SafetyTestConfig,
    SafetyClass,
    ApplicationClass,
    FireClass,
)


class TestIEC61730SafetyTester:
    """Test suite for IEC61730SafetyTester class."""

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
    def safety_tester(self, test_config):
        """Create safety tester fixture."""
        return IEC61730SafetyTester(test_config)

    def test_initialization(self, safety_tester, test_config):
        """Test safety tester initialization."""
        assert safety_tester.config == test_config
        assert safety_tester.results is None
        assert safety_tester.electrical_tester is not None
        assert safety_tester.fire_tester is not None

    def test_electrical_safety_tests(self, safety_tester):
        """Test electrical safety tests execution."""
        result = safety_tester.electrical_safety_tests()

        assert result is not None
        assert result.insulation_resistance is not None
        assert result.wet_leakage_current is not None
        assert result.dielectric_strength is not None
        assert result.bypass_diode_thermal is not None
        assert isinstance(result.all_tests_passed, bool)

    def test_mechanical_safety_tests(self, safety_tester):
        """Test mechanical safety tests execution."""
        result = safety_tester.mechanical_safety_tests()

        assert result is not None
        assert result.mechanical_load is not None
        assert result.impact is not None
        assert result.robustness_of_terminations is not None
        assert isinstance(result.all_tests_passed, bool)

    def test_fire_safety_tests(self, safety_tester):
        """Test fire safety tests execution."""
        result = safety_tester.fire_safety_tests()

        assert result is not None
        assert result.spread_of_flame is not None
        assert result.fire_penetration is not None
        assert result.fire_brand is not None
        assert result.fire_classification in [
            FireClass.CLASS_A,
            FireClass.CLASS_B,
            FireClass.CLASS_C,
            FireClass.NOT_RATED
        ]

    def test_environmental_safety_tests(self, safety_tester):
        """Test environmental safety tests execution."""
        result = safety_tester.environmental_safety_tests()

        assert result is not None
        assert result.uv_preconditioning is not None
        assert result.thermal_cycling is not None
        assert result.humidity_freeze is not None
        assert isinstance(result.all_tests_passed, bool)

    def test_construction_requirements_check(self, safety_tester):
        """Test construction requirements validation."""
        requirements = safety_tester.construction_requirements_check()

        assert requirements is not None
        assert len(requirements) > 0

        for req in requirements:
            assert req.requirement_id is not None
            assert req.requirement_description is not None
            assert isinstance(req.compliant, bool)

    def test_generate_safety_classification(self, safety_tester):
        """Test safety classification generation."""
        # Run tests first
        electrical_results = safety_tester.electrical_safety_tests()
        mechanical_results = safety_tester.mechanical_safety_tests()
        fire_results = safety_tester.fire_safety_tests()
        construction_results = safety_tester.construction_requirements_check()

        # Generate classification
        classification = safety_tester.generate_safety_classification(
            electrical_results,
            mechanical_results,
            fire_results,
            construction_results,
        )

        assert classification is not None
        assert classification.safety_class in [
            SafetyClass.CLASS_I,
            SafetyClass.CLASS_II,
            SafetyClass.CLASS_III
        ]
        assert classification.application_class in [
            ApplicationClass.CLASS_A,
            ApplicationClass.CLASS_B,
            ApplicationClass.CLASS_C
        ]
        assert classification.fire_class in [
            FireClass.CLASS_A,
            FireClass.CLASS_B,
            FireClass.CLASS_C,
            FireClass.NOT_RATED
        ]
        assert classification.max_system_voltage_v == safety_tester.config.max_system_voltage_v
        assert len(classification.classification_rationale) > 0

    def test_run_all_tests(self, safety_tester):
        """Test running complete test suite."""
        results = safety_tester.run_all_tests()

        assert results is not None
        assert results.config == safety_tester.config
        assert results.electrical_tests is not None
        assert results.mechanical_tests is not None
        assert results.fire_tests is not None
        assert results.environmental_tests is not None
        assert len(results.construction_requirements) > 0
        assert results.classification is not None
        assert isinstance(results.overall_pass, bool)
        assert results.test_completion_date is not None

        # Check that results are stored in tester
        assert safety_tester.results == results

    def test_run_all_tests_with_selective_tests(self):
        """Test running tests with selective test categories."""
        config = SafetyTestConfig(
            module_id="TEST-002",
            manufacturer="TestCorp",
            model_number="TC-400",
            max_system_voltage_v=1000.0,
            module_area_m2=2.0,
            application_class=ApplicationClass.CLASS_B,
            target_safety_class=SafetyClass.CLASS_II,
            test_laboratory="Test Lab",
            test_date=datetime.now(),
            perform_electrical_tests=True,
            perform_mechanical_tests=False,
            perform_fire_tests=False,
            perform_environmental_tests=True,
        )

        tester = IEC61730SafetyTester(config)
        results = tester.run_all_tests()

        assert results.electrical_tests is not None
        assert results.mechanical_tests is None
        assert results.fire_tests is None
        assert results.environmental_tests is not None

    def test_export_safety_certificate_without_tests(self, safety_tester):
        """Test that certificate export fails without running tests."""
        with pytest.raises(ValueError, match="tests have not been completed"):
            safety_tester.export_safety_certificate()

    def test_export_safety_certificate_with_passed_tests(self, safety_tester):
        """Test certificate export after passing tests."""
        # Run tests
        results = safety_tester.run_all_tests()

        # Only try to generate certificate if tests passed
        if results.overall_pass:
            certificate = safety_tester.export_safety_certificate(
                certification_body="Test Certification Body"
            )

            assert certificate is not None
            assert certificate.certificate_number is not None
            assert certificate.certification_body == "Test Certification Body"
            assert certificate.issue_date is not None
            assert certificate.expiry_date is not None
            assert certificate.module_info == safety_tester.config
            assert certificate.test_results == results
            assert certificate.certified_safety_class is not None
            assert certificate.certified_application_class is not None

    def test_mechanical_load_test(self, safety_tester):
        """Test mechanical load test execution."""
        result = safety_tester._mechanical_load_test()

        assert result is not None
        assert result.applied_load_pa == 2400.0
        assert result.maximum_deflection_mm > 0
        assert result.permanent_deformation_mm >= 0
        assert result.cycles_completed == 3
        assert isinstance(result.visual_defects_found, bool)

    def test_impact_resistance_test(self, safety_tester):
        """Test impact resistance test execution."""
        result = safety_tester._impact_resistance_test()

        assert result is not None
        assert result.ice_ball_diameter_mm == 25.0
        assert result.impact_velocity_ms > 0
        assert result.impact_locations == 11
        assert isinstance(result.cracks_detected, bool)
        assert isinstance(result.electrical_safety_maintained, bool)

    def test_robustness_of_terminations_test(self, safety_tester):
        """Test robustness of terminations test execution."""
        result = safety_tester._robustness_of_terminations_test()

        assert result is not None
        assert result.pull_force_n == 100.0
        assert result.torque_nm == 1.0
        assert isinstance(result.cable_displaced, bool)
        assert isinstance(result.terminal_damaged, bool)

    def test_uv_preconditioning_test(self, safety_tester):
        """Test UV preconditioning test execution."""
        result = safety_tester._uv_preconditioning_test()

        assert result is not None
        assert result.uv_dose_kwh_m2 > 0
        assert result.required_dose_kwh_m2 == 15.0
        assert result.test_duration_h > 0
        assert isinstance(result.visual_degradation, bool)

    def test_thermal_cycling_test(self, safety_tester):
        """Test thermal cycling test execution."""
        result = safety_tester._thermal_cycling_test()

        assert result is not None
        assert result.cycles_completed >= 0
        assert result.required_cycles == 200
        assert result.min_temperature_c == -40.0
        assert result.max_temperature_c == 85.0
        assert isinstance(result.electrical_failure, bool)

    def test_humidity_freeze_test(self, safety_tester):
        """Test humidity-freeze test execution."""
        result = safety_tester._humidity_freeze_test()

        assert result is not None
        assert result.cycles_completed >= 0
        assert result.required_cycles == 10
        assert result.humidity_phase_c == 85.0
        assert result.humidity_phase_rh == 85.0
        assert result.freeze_phase_c == -40.0
        assert isinstance(result.electrical_failure, bool)

    def test_class_i_module_configuration(self):
        """Test configuration and testing of Class I module."""
        config = SafetyTestConfig(
            module_id="TEST-CLASS-I",
            manufacturer="TestCorp",
            model_number="TC-400-I",
            max_system_voltage_v=1000.0,
            module_area_m2=2.0,
            application_class=ApplicationClass.CLASS_B,
            target_safety_class=SafetyClass.CLASS_I,
            test_laboratory="Test Lab",
            test_date=datetime.now(),
            perform_electrical_tests=True,
            perform_mechanical_tests=False,
            perform_fire_tests=False,
            perform_environmental_tests=False,
        )

        tester = IEC61730SafetyTester(config)
        results = tester.run_all_tests()

        # Class I should have ground continuity test
        assert results.electrical_tests.ground_continuity is not None

    def test_all_required_tests_passed_logic(self, safety_tester):
        """Test the all_required_tests_passed logic."""
        results = safety_tester.run_all_tests()

        # Verify the logic matches individual test results
        tests_passed = []

        if results.config.perform_electrical_tests and results.electrical_tests:
            tests_passed.append(results.electrical_tests.all_tests_passed)

        if results.config.perform_mechanical_tests and results.mechanical_tests:
            tests_passed.append(results.mechanical_tests.all_tests_passed)

        if results.config.perform_fire_tests and results.fire_tests:
            tests_passed.append(results.fire_tests.fire_tests_passed)

        if results.config.perform_environmental_tests and results.environmental_tests:
            tests_passed.append(results.environmental_tests.all_tests_passed)

        if results.construction_requirements:
            tests_passed.append(all(req.compliant for req in results.construction_requirements))

        expected_all_passed = all(tests_passed) if tests_passed else False

        assert results.all_required_tests_passed == expected_all_passed
