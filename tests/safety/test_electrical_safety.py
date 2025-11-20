"""Unit tests for electrical safety testing.

Tests all electrical safety test methods per IEC 61730-2 MST 01-05.
"""

import pytest
from src.safety.electrical_safety import ElectricalSafetyTest
from src.models.safety_models import SafetyClass, TestStatus


class TestElectricalSafetyTest:
    """Test suite for ElectricalSafetyTest class."""

    @pytest.fixture
    def electrical_tester(self):
        """Create electrical safety tester fixture."""
        return ElectricalSafetyTest(
            module_id="TEST-001",
            max_system_voltage_v=1000.0,
            safety_class=SafetyClass.CLASS_II,
            test_temperature_c=25.0,
            test_humidity_percent=50.0,
        )

    @pytest.fixture
    def electrical_tester_class_i(self):
        """Create Class I electrical safety tester fixture."""
        return ElectricalSafetyTest(
            module_id="TEST-001",
            max_system_voltage_v=1000.0,
            safety_class=SafetyClass.CLASS_I,
            test_temperature_c=25.0,
            test_humidity_percent=50.0,
        )

    def test_initialization(self, electrical_tester):
        """Test electrical safety tester initialization."""
        assert electrical_tester.module_id == "TEST-001"
        assert electrical_tester.max_system_voltage_v == 1000.0
        assert electrical_tester.safety_class == SafetyClass.CLASS_II
        assert electrical_tester.test_temperature_c == 25.0
        assert electrical_tester.test_humidity_percent == 50.0

    def test_insulation_resistance_test(self, electrical_tester):
        """Test insulation resistance test execution."""
        result = electrical_tester.insulation_resistance_test()

        assert result is not None
        assert result.test_voltage_v in [500.0, 1000.0]
        assert result.measured_resistance_mohm >= 0
        assert result.minimum_required_mohm == 40.0
        assert result.test_duration_s == 60.0
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert isinstance(result.passed, bool)

    def test_insulation_resistance_test_custom_voltage(self, electrical_tester):
        """Test insulation resistance with custom voltage."""
        result = electrical_tester.insulation_resistance_test(test_voltage_v=500.0)

        assert result.test_voltage_v == 500.0

    def test_insulation_resistance_test_custom_duration(self, electrical_tester):
        """Test insulation resistance with custom duration."""
        result = electrical_tester.insulation_resistance_test(test_duration_s=120.0)

        assert result.test_duration_s == 120.0

    def test_wet_leakage_current_test(self, electrical_tester):
        """Test wet leakage current test execution."""
        result = electrical_tester.wet_leakage_current_test()

        assert result is not None
        assert result.leakage_current_ua >= 0
        assert result.maximum_allowed_ua == 275.0
        assert result.test_voltage_v > 0
        assert result.water_spray_duration_min == 10.0
        assert result.water_resistivity_ohm_cm > 0
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert isinstance(result.passed, bool)

    def test_wet_leakage_current_test_custom_params(self, electrical_tester):
        """Test wet leakage current with custom parameters."""
        result = electrical_tester.wet_leakage_current_test(
            water_resistivity_ohm_cm=10000.0,
            spray_duration_min=15.0
        )

        assert result.water_resistivity_ohm_cm == 10000.0
        assert result.water_spray_duration_min == 15.0

    def test_dielectric_strength_test(self, electrical_tester):
        """Test dielectric strength test execution."""
        result = electrical_tester.dielectric_strength_test()

        assert result is not None
        assert result.test_voltage_v > 0
        assert result.vmax_dc_v == electrical_tester.max_system_voltage_v
        assert result.test_duration_s == 60.0
        assert isinstance(result.breakdown_occurred, bool)
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert isinstance(result.passed, bool)

        if result.breakdown_occurred:
            assert result.breakdown_voltage_v is not None
            assert result.breakdown_voltage_v > 0
        else:
            assert result.breakdown_voltage_v is None

    def test_ground_continuity_test_class_i(self, electrical_tester_class_i):
        """Test ground continuity test for Class I module."""
        result = electrical_tester_class_i.ground_continuity_test()

        assert result is not None
        assert result.measured_resistance_ohm >= 0
        assert result.maximum_allowed_ohm == 0.1
        assert result.test_current_a == 10.0
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert isinstance(result.passed, bool)

    def test_ground_continuity_test_class_ii_raises_error(self, electrical_tester):
        """Test that ground continuity test raises error for Class II."""
        with pytest.raises(ValueError, match="only applies to Class I"):
            electrical_tester.ground_continuity_test()

    def test_ground_continuity_test_custom_current(self, electrical_tester_class_i):
        """Test ground continuity with custom test current."""
        result = electrical_tester_class_i.ground_continuity_test(test_current_a=20.0)

        assert result.test_current_a == 20.0

    def test_bypass_diode_thermal_test(self, electrical_tester):
        """Test bypass diode thermal test execution."""
        result = electrical_tester.bypass_diode_thermal_test(
            fault_current_a=12.5,
            test_duration_h=2.0,
            max_diode_temp_c=150.0
        )

        assert result is not None
        assert result.peak_temperature_c > 0
        assert result.maximum_allowed_c == 150.0
        assert result.fault_current_a == 12.5
        assert result.test_duration_h == 2.0
        assert isinstance(result.thermal_runaway_detected, bool)
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert isinstance(result.passed, bool)

    def test_run_all_electrical_tests_class_ii(self, electrical_tester):
        """Test running all electrical tests for Class II module."""
        result = electrical_tester.run_all_electrical_tests()

        assert result is not None
        assert result.insulation_resistance is not None
        assert result.wet_leakage_current is not None
        assert result.dielectric_strength is not None
        assert result.ground_continuity is None  # Not applicable for Class II
        assert result.bypass_diode_thermal is not None
        assert isinstance(result.all_tests_passed, bool)

    def test_run_all_electrical_tests_class_i(self, electrical_tester_class_i):
        """Test running all electrical tests for Class I module."""
        result = electrical_tester_class_i.run_all_electrical_tests()

        assert result is not None
        assert result.insulation_resistance is not None
        assert result.wet_leakage_current is not None
        assert result.dielectric_strength is not None
        assert result.ground_continuity is not None  # Required for Class I
        assert result.bypass_diode_thermal is not None
        assert isinstance(result.all_tests_passed, bool)

    def test_run_all_electrical_tests_force_include_ground(self, electrical_tester):
        """Test running all tests with forced ground continuity."""
        # This should fail because it's Class II
        with pytest.raises(ValueError):
            electrical_tester.run_all_electrical_tests(include_ground_continuity=True)

    def test_insulation_resistance_pass_fail_logic(self, electrical_tester):
        """Test insulation resistance pass/fail determination."""
        result = electrical_tester.insulation_resistance_test()

        if result.measured_resistance_mohm >= result.minimum_required_mohm:
            assert result.passed is True
            assert result.status == TestStatus.PASSED
        else:
            assert result.passed is False
            assert result.status == TestStatus.FAILED

    def test_wet_leakage_current_pass_fail_logic(self, electrical_tester):
        """Test wet leakage current pass/fail determination."""
        result = electrical_tester.wet_leakage_current_test()

        if result.leakage_current_ua <= result.maximum_allowed_ua:
            assert result.passed is True
            assert result.status == TestStatus.PASSED
        else:
            assert result.passed is False
            assert result.status == TestStatus.FAILED

    def test_dielectric_strength_pass_fail_logic(self, electrical_tester):
        """Test dielectric strength pass/fail determination."""
        result = electrical_tester.dielectric_strength_test()

        if not result.breakdown_occurred:
            assert result.passed is True
            assert result.status == TestStatus.PASSED
        else:
            assert result.passed is False
            assert result.status == TestStatus.FAILED
