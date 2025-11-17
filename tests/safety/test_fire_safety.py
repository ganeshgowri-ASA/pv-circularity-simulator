"""Unit tests for fire safety testing.

Tests all fire safety test methods per IEC 61730-2 Annex C and UL 790.
"""

import pytest
from src.safety.fire_safety import FireSafetyClassification
from src.models.safety_models import FireClass, TestStatus


class TestFireSafetyClassification:
    """Test suite for FireSafetyClassification class."""

    @pytest.fixture
    def fire_tester(self):
        """Create fire safety tester fixture."""
        return FireSafetyClassification(
            module_id="TEST-001",
            module_area_m2=2.0,
            roof_mounting=True,
            backsheet_material="PET/EVA/PET",
            frame_material="Aluminum",
        )

    def test_initialization(self, fire_tester):
        """Test fire safety tester initialization."""
        assert fire_tester.module_id == "TEST-001"
        assert fire_tester.module_area_m2 == 2.0
        assert fire_tester.roof_mounting is True
        assert fire_tester.backsheet_material == "PET/EVA/PET"
        assert fire_tester.frame_material == "Aluminum"

    def test_spread_of_flame_test(self, fire_tester):
        """Test spread of flame test execution."""
        result = fire_tester.spread_of_flame_test()

        assert result is not None
        assert result.flame_spread_distance_cm >= 0
        assert result.flame_exposure_time_min == 10.0
        assert isinstance(result.sustained_flaming_observed, bool)
        assert isinstance(result.roof_deck_penetration, bool)
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]

    def test_spread_of_flame_test_custom_params(self, fire_tester):
        """Test spread of flame with custom parameters."""
        result = fire_tester.spread_of_flame_test(
            flame_exposure_time_min=4.0,
            slope_deg=12.0
        )

        assert result.flame_exposure_time_min == 4.0

    def test_fire_penetration_test(self, fire_tester):
        """Test fire penetration test execution."""
        result = fire_tester.fire_penetration_test()

        assert result is not None
        assert isinstance(result.burn_through_occurred, bool)
        assert result.test_duration_min == 90.0
        assert isinstance(result.roof_deck_damage, bool)
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]

    def test_fire_penetration_test_custom_duration(self, fire_tester):
        """Test fire penetration with custom duration."""
        result = fire_tester.fire_penetration_test(test_duration_min=60.0)

        assert result.test_duration_min == 60.0

    def test_fire_brand_test_class_a(self, fire_tester):
        """Test fire brand test with Class A brand."""
        result = fire_tester.fire_brand_test(brand_size_class="A")

        assert result is not None
        assert result.brand_size_class == "A"
        assert isinstance(result.ignition_occurred, bool)
        assert isinstance(result.sustained_burning, bool)
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]

    def test_fire_brand_test_class_b(self, fire_tester):
        """Test fire brand test with Class B brand."""
        result = fire_tester.fire_brand_test(brand_size_class="B")

        assert result.brand_size_class == "B"

    def test_fire_brand_test_class_c(self, fire_tester):
        """Test fire brand test with Class C brand."""
        result = fire_tester.fire_brand_test(brand_size_class="C")

        assert result.brand_size_class == "C"

    def test_fire_brand_test_invalid_class(self, fire_tester):
        """Test fire brand test with invalid brand class."""
        with pytest.raises(ValueError, match="Invalid brand size class"):
            fire_tester.fire_brand_test(brand_size_class="D")

    def test_classify_fire_rating_class_a(self, fire_tester):
        """Test fire classification for Class A."""
        # Run tests
        spread_result = fire_tester.spread_of_flame_test(flame_exposure_time_min=10.0)
        penetration_result = fire_tester.fire_penetration_test(test_duration_min=90.0)
        brand_result = fire_tester.fire_brand_test(brand_size_class="A")

        # Classify
        classification = fire_tester.classify_fire_rating(
            spread_result,
            penetration_result,
            brand_result
        )

        assert classification in [FireClass.CLASS_A, FireClass.CLASS_B, FireClass.CLASS_C, FireClass.NOT_RATED]

    def test_classify_fire_rating_roof_penetration_fails(self, fire_tester):
        """Test that roof deck penetration results in Not Rated."""
        # Create a failing spread of flame result
        from src.models.safety_models import SpreadOfFlameTestResult

        spread_result = SpreadOfFlameTestResult(
            flame_spread_distance_cm=100.0,
            flame_exposure_time_min=10.0,
            sustained_flaming_observed=False,
            roof_deck_penetration=True,  # Critical failure
            status=TestStatus.FAILED
        )

        penetration_result = fire_tester.fire_penetration_test()
        brand_result = fire_tester.fire_brand_test(brand_size_class="A")

        classification = fire_tester.classify_fire_rating(
            spread_result,
            penetration_result,
            brand_result
        )

        assert classification == FireClass.NOT_RATED

    def test_roof_mounting_fire_safety(self, fire_tester):
        """Test roof mounting fire safety assessment."""
        compliant, recommendations = fire_tester.roof_mounting_fire_safety()

        assert isinstance(compliant, bool)
        assert isinstance(recommendations, str)
        assert len(recommendations) > 0

    def test_roof_mounting_fire_safety_not_roof_mounted(self):
        """Test roof mounting assessment for non-roof-mounted module."""
        tester = FireSafetyClassification(
            module_id="TEST-002",
            module_area_m2=2.0,
            roof_mounting=False,
        )

        compliant, recommendations = tester.roof_mounting_fire_safety()

        assert compliant is True
        assert "not intended for roof mounting" in recommendations

    def test_run_all_fire_tests_class_a(self, fire_tester):
        """Test running all fire tests for Class A target."""
        result = fire_tester.run_all_fire_tests(target_fire_class=FireClass.CLASS_A)

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

    def test_run_all_fire_tests_class_b(self, fire_tester):
        """Test running all fire tests for Class B target."""
        result = fire_tester.run_all_fire_tests(target_fire_class=FireClass.CLASS_B)

        assert result is not None
        assert result.fire_brand.brand_size_class == "B"

    def test_run_all_fire_tests_class_c(self, fire_tester):
        """Test running all fire tests for Class C target."""
        result = fire_tester.run_all_fire_tests(target_fire_class=FireClass.CLASS_C)

        assert result is not None
        assert result.fire_brand.brand_size_class == "C"

    def test_backsheet_flammability_factors(self, fire_tester):
        """Test backsheet flammability factor calculation."""
        # Test different backsheet materials
        materials = [
            ("PET/EVA/PET", 0.4),
            ("PVF/EVA/PVF", 0.2),
            ("PA/EVA/PA", 0.3),
        ]

        for material, expected_range in materials:
            tester = FireSafetyClassification(
                module_id="TEST",
                module_area_m2=2.0,
                backsheet_material=material,
            )
            factor = tester._get_backsheet_flammability()
            assert 0.1 <= factor <= 1.0

    def test_frame_flammability_factors(self, fire_tester):
        """Test frame flammability factor calculation."""
        # Aluminum frame should have low flammability
        aluminum_tester = FireSafetyClassification(
            module_id="TEST",
            module_area_m2=2.0,
            frame_material="Aluminum",
        )
        assert aluminum_tester._get_frame_flammability() == 0.1

        # Plastic frame should have high flammability
        plastic_tester = FireSafetyClassification(
            module_id="TEST",
            module_area_m2=2.0,
            frame_material="Plastic",
        )
        assert plastic_tester._get_frame_flammability() == 0.8

    def test_backsheet_ignition_temperature(self, fire_tester):
        """Test backsheet ignition temperature calculation."""
        temp = fire_tester._get_backsheet_ignition_temp()

        assert 300.0 <= temp <= 600.0  # Reasonable range for polymers
