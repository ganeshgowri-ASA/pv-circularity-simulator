"""Comprehensive unit tests for ReuseAssessor class."""

import pytest
from datetime import datetime

from pv_circularity_simulator.circularity.reuse_assessor import ReuseAssessor
from pv_circularity_simulator.core.enums import (
    ModuleCondition,
    PerformanceLevel,
    ReusePotential,
    DegradationType,
    MarketSegment,
)
from pv_circularity_simulator.core.models import ModuleData, PerformanceMetrics


class TestReuseAssessorInit:
    """Tests for ReuseAssessor initialization."""

    def test_default_initialization(self):
        """Test ReuseAssessor with default parameters."""
        assessor = ReuseAssessor()
        assert assessor.degradation_rate_per_year == 0.005
        assert assessor.expected_lifetime_years == 25.0
        assert assessor.minimum_performance_threshold == 0.70
        assert assessor.base_module_price_per_watt == 0.50

    def test_custom_initialization(self):
        """Test ReuseAssessor with custom parameters."""
        assessor = ReuseAssessor(
            degradation_rate_per_year=0.008,
            expected_lifetime_years=30.0,
            minimum_performance_threshold=0.65,
            base_module_price_per_watt=0.75,
        )
        assert assessor.degradation_rate_per_year == 0.008
        assert assessor.expected_lifetime_years == 30.0
        assert assessor.minimum_performance_threshold == 0.65
        assert assessor.base_module_price_per_watt == 0.75

    def test_invalid_degradation_rate(self):
        """Test that invalid degradation rate raises ValueError."""
        with pytest.raises(ValueError, match="Degradation rate must be between"):
            ReuseAssessor(degradation_rate_per_year=0.10)

        with pytest.raises(ValueError, match="Degradation rate must be between"):
            ReuseAssessor(degradation_rate_per_year=-0.01)

    def test_invalid_lifetime(self):
        """Test that invalid lifetime raises ValueError."""
        with pytest.raises(ValueError, match="Expected lifetime must be between"):
            ReuseAssessor(expected_lifetime_years=5.0)

        with pytest.raises(ValueError, match="Expected lifetime must be between"):
            ReuseAssessor(expected_lifetime_years=100.0)

    def test_invalid_performance_threshold(self):
        """Test that invalid performance threshold raises ValueError."""
        with pytest.raises(ValueError, match="Performance threshold must be between"):
            ReuseAssessor(minimum_performance_threshold=1.5)

        with pytest.raises(ValueError, match="Performance threshold must be between"):
            ReuseAssessor(minimum_performance_threshold=0.0)

    def test_invalid_base_price(self):
        """Test that invalid base price raises ValueError."""
        with pytest.raises(ValueError, match="Base price must be between"):
            ReuseAssessor(base_module_price_per_watt=10.0)

        with pytest.raises(ValueError, match="Base price must be between"):
            ReuseAssessor(base_module_price_per_watt=-1.0)


class TestConditionGrading:
    """Tests for condition_grading method."""

    def test_new_module_excellent_condition(self, assessor, new_module):
        """Test that new modules receive excellent condition grades."""
        assessment = assessor.condition_grading(new_module)

        assert assessment.overall_condition == ModuleCondition.EXCELLENT
        assert assessment.condition_score >= 85
        assert assessment.visual_inspection_pass is True
        assert assessment.electrical_safety_pass is True
        assert len(assessment.defect_severity_scores) == 0

    def test_mid_life_module_good_condition(self, assessor, mid_life_module):
        """Test that mid-life modules with minor defects receive good grades."""
        assessment = assessor.condition_grading(mid_life_module)

        assert assessment.overall_condition in [ModuleCondition.GOOD, ModuleCondition.FAIR]
        assert 50 <= assessment.condition_score < 85
        assert DegradationType.DISCOLORATION.value in assessment.defect_severity_scores

    def test_old_module_fair_to_poor_condition(self, assessor, old_module):
        """Test that old modules with multiple defects receive lower grades."""
        assessment = assessor.condition_grading(old_module)

        assert assessment.overall_condition in [
            ModuleCondition.FAIR,
            ModuleCondition.POOR,
        ]
        assert assessment.condition_score < 70
        assert len(assessment.defect_severity_scores) == 3

    def test_failed_module_poor_condition(self, assessor, failed_module):
        """Test that failed modules receive poor/failed grades."""
        assessment = assessor.condition_grading(
            failed_module,
            visual_inspection_pass=False,
            electrical_safety_pass=False,
        )

        assert assessment.overall_condition in [
            ModuleCondition.POOR,
            ModuleCondition.FAILED,
        ]
        assert assessment.condition_score < 50
        assert assessment.visual_inspection_pass is False
        assert assessment.electrical_safety_pass is False

    def test_failed_visual_inspection_penalty(self, assessor, new_module):
        """Test that failed visual inspection applies penalty."""
        pass_assessment = assessor.condition_grading(
            new_module, visual_inspection_pass=True
        )
        fail_assessment = assessor.condition_grading(
            new_module, visual_inspection_pass=False
        )

        assert fail_assessment.condition_score < pass_assessment.condition_score
        assert fail_assessment.condition_score <= pass_assessment.condition_score * 0.7

    def test_failed_electrical_safety_penalty(self, assessor, new_module):
        """Test that failed electrical safety applies penalty."""
        pass_assessment = assessor.condition_grading(
            new_module, electrical_safety_pass=True
        )
        fail_assessment = assessor.condition_grading(
            new_module, electrical_safety_pass=False
        )

        assert fail_assessment.condition_score < pass_assessment.condition_score

    def test_custom_structural_integrity_score(self, assessor, mid_life_module):
        """Test that custom structural integrity score is used."""
        assessment = assessor.condition_grading(
            mid_life_module, structural_integrity_score=95.0
        )

        assert assessment.structural_integrity_score == 95.0

    def test_invalid_structural_integrity_score(self, assessor, new_module):
        """Test that invalid structural score raises ValueError."""
        with pytest.raises(ValueError, match="Structural integrity score must be"):
            assessor.condition_grading(new_module, structural_integrity_score=150.0)

        with pytest.raises(ValueError, match="Structural integrity score must be"):
            assessor.condition_grading(new_module, structural_integrity_score=-10.0)

    def test_defect_severity_weights(self, assessor):
        """Test that different defects have appropriate severity weights."""
        # Hot spot should be more severe than discoloration
        module_hotspot = ModuleData(
            module_id="TEST-1",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=5.0,
            degradation_types=[DegradationType.HOT_SPOT],
        )

        module_discolor = ModuleData(
            module_id="TEST-2",
            manufacturer="Test",
            model="T2",
            nameplate_power_w=300.0,
            age_years=5.0,
            degradation_types=[DegradationType.DISCOLORATION],
        )

        assessment_hotspot = assessor.condition_grading(module_hotspot)
        assessment_discolor = assessor.condition_grading(module_discolor)

        assert assessment_hotspot.condition_score < assessment_discolor.condition_score


class TestPerformanceTesting:
    """Tests for performance_testing method."""

    def test_high_performance_classification(self, assessor, new_module, high_performance):
        """Test that high-performing modules are classified as HIGH."""
        level, score = assessor.performance_testing(new_module, high_performance)

        assert level == PerformanceLevel.HIGH
        assert score >= 85

    def test_medium_performance_classification(
        self, assessor, mid_life_module, medium_performance
    ):
        """Test that medium-performing modules are classified as MEDIUM."""
        level, score = assessor.performance_testing(mid_life_module, medium_performance)

        assert level == PerformanceLevel.MEDIUM
        assert 60 <= score < 90

    def test_low_performance_classification(self, assessor, old_module, low_performance):
        """Test that low-performing modules are classified as LOW."""
        level, score = assessor.performance_testing(old_module, low_performance)

        assert level == PerformanceLevel.LOW
        assert 40 <= score <= 75  # Allow for age-adjusted performance bonus

    def test_critical_performance_classification(
        self, assessor, failed_module, critical_performance
    ):
        """Test that critically low modules are classified as CRITICAL."""
        level, score = assessor.performance_testing(failed_module, critical_performance)

        assert level == PerformanceLevel.CRITICAL
        assert score < 50

    def test_fill_factor_impact(self, assessor, new_module):
        """Test that fill factor impacts performance score."""
        high_ff = PerformanceMetrics(
            measured_power_w=340.0,
            open_circuit_voltage_v=46.8,
            short_circuit_current_a=9.2,
            max_power_voltage_v=38.5,
            max_power_current_a=8.83,
            fill_factor=0.82,  # High fill factor
            efficiency_percent=18.5,
        )

        low_ff = PerformanceMetrics(
            measured_power_w=340.0,
            open_circuit_voltage_v=46.8,
            short_circuit_current_a=9.2,
            max_power_voltage_v=38.5,
            max_power_current_a=8.83,
            fill_factor=0.65,  # Low fill factor
            efficiency_percent=18.5,
        )

        _, high_score = assessor.performance_testing(new_module, high_ff)
        _, low_score = assessor.performance_testing(new_module, low_ff)

        assert high_score > low_score

    def test_age_adjusted_performance(self, assessor):
        """Test that performance is age-adjusted."""
        young_module = ModuleData(
            module_id="YOUNG",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=2.0,
        )

        old_module = ModuleData(
            module_id="OLD",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=15.0,
        )

        performance = PerformanceMetrics(
            measured_power_w=270.0,  # 90% of nameplate
            open_circuit_voltage_v=44.0,
            short_circuit_current_a=8.5,
            max_power_voltage_v=36.0,
            max_power_current_a=7.5,
            fill_factor=0.76,
            efficiency_percent=16.0,
        )

        _, young_score = assessor.performance_testing(young_module, performance)
        _, old_score = assessor.performance_testing(old_module, performance)

        # Old module should get better age-adjusted score since 270W is
        # better than expected for 15-year-old module
        assert old_score > young_score


class TestReusePotentialScoring:
    """Tests for reuse_potential_scoring method."""

    def test_direct_reuse_potential(self, assessor, new_module):
        """Test that excellent modules get DIRECT_REUSE classification."""
        condition = assessor.condition_grading(new_module)
        perf_level, perf_score = assessor.performance_testing(
            new_module,
            PerformanceMetrics(
                measured_power_w=340.0,
                open_circuit_voltage_v=46.8,
                short_circuit_current_a=9.2,
                max_power_voltage_v=38.5,
                max_power_current_a=8.83,
                fill_factor=0.79,
                efficiency_percent=18.5,
            ),
        )

        potential, score, lifetime = assessor.reuse_potential_scoring(
            new_module, condition, perf_level, perf_score
        )

        assert potential == ReusePotential.DIRECT_REUSE
        assert score >= 80
        assert lifetime > 20

    def test_secondary_market_potential(self, assessor, mid_life_module, medium_performance):
        """Test that good modules get SECONDARY_MARKET classification."""
        condition = assessor.condition_grading(mid_life_module)
        perf_level, perf_score = assessor.performance_testing(
            mid_life_module, medium_performance
        )

        potential, score, lifetime = assessor.reuse_potential_scoring(
            mid_life_module, condition, perf_level, perf_score
        )

        assert potential in [ReusePotential.DIRECT_REUSE, ReusePotential.SECONDARY_MARKET]
        assert 60 <= score < 90
        assert lifetime > 5

    def test_component_recovery_potential(self, assessor, old_module, low_performance):
        """Test that degraded modules get COMPONENT_RECOVERY classification."""
        condition = assessor.condition_grading(old_module)
        perf_level, perf_score = assessor.performance_testing(old_module, low_performance)

        potential, score, lifetime = assessor.reuse_potential_scoring(
            old_module, condition, perf_level, perf_score
        )

        # Old modules with low performance may be recycle_only or component_recovery
        assert potential in [
            ReusePotential.SECONDARY_MARKET,
            ReusePotential.COMPONENT_RECOVERY,
            ReusePotential.RECYCLE_ONLY,
        ]
        assert 20 <= score < 70

    def test_recycle_only_potential(self, assessor, failed_module, critical_performance):
        """Test that failed modules get RECYCLE_ONLY classification."""
        condition = assessor.condition_grading(
            failed_module, visual_inspection_pass=False, electrical_safety_pass=False
        )
        perf_level, perf_score = assessor.performance_testing(
            failed_module, critical_performance
        )

        potential, score, lifetime = assessor.reuse_potential_scoring(
            failed_module, condition, perf_level, perf_score
        )

        assert potential in [
            ReusePotential.COMPONENT_RECOVERY,
            ReusePotential.RECYCLE_ONLY,
        ]
        assert score < 50
        assert lifetime < 5

    def test_remaining_lifetime_calculation(self, assessor, mid_life_module):
        """Test that remaining lifetime is calculated correctly."""
        condition = assessor.condition_grading(mid_life_module)
        performance = PerformanceMetrics(
            measured_power_w=270.0,  # 90% of 300W
            open_circuit_voltage_v=44.2,
            short_circuit_current_a=8.1,
            max_power_voltage_v=36.0,
            max_power_current_a=7.5,
            fill_factor=0.76,
            efficiency_percent=16.2,
        )
        perf_level, perf_score = assessor.performance_testing(mid_life_module, performance)

        _, _, lifetime = assessor.reuse_potential_scoring(
            mid_life_module, condition, perf_level, perf_score
        )

        # Module is 10 years old with 90% performance
        # Should have significant remaining lifetime
        assert lifetime > 10
        assert lifetime <= 25  # May reach expected lifetime limit

    def test_safety_failure_impact(self, assessor, mid_life_module, medium_performance):
        """Test that safety failures reduce reusability score."""
        safe_condition = assessor.condition_grading(
            mid_life_module, electrical_safety_pass=True
        )
        unsafe_condition = assessor.condition_grading(
            mid_life_module, electrical_safety_pass=False
        )

        perf_level, perf_score = assessor.performance_testing(
            mid_life_module, medium_performance
        )

        _, safe_score, _ = assessor.reuse_potential_scoring(
            mid_life_module, safe_condition, perf_level, perf_score
        )
        _, unsafe_score, _ = assessor.reuse_potential_scoring(
            mid_life_module, unsafe_condition, perf_level, perf_score
        )

        assert unsafe_score < safe_score


class TestSecondaryMarketValuation:
    """Tests for secondary_market_valuation method."""

    def test_high_value_for_excellent_modules(self, assessor, new_module):
        """Test that excellent modules receive high valuations."""
        condition = assessor.condition_grading(new_module)
        perf_level, perf_score = assessor.performance_testing(
            new_module,
            PerformanceMetrics(
                measured_power_w=340.0,
                open_circuit_voltage_v=46.8,
                short_circuit_current_a=9.2,
                max_power_voltage_v=38.5,
                max_power_current_a=8.83,
                fill_factor=0.79,
                efficiency_percent=18.5,
            ),
        )
        _, score, lifetime = assessor.reuse_potential_scoring(
            new_module, condition, perf_level, perf_score
        )

        valuation = assessor.secondary_market_valuation(
            new_module, score, perf_level, condition, lifetime
        )

        # New 350W module should be valued high
        assert valuation.final_value_usd > 100
        assert valuation.condition_multiplier >= 0.75
        assert valuation.performance_multiplier >= 0.75
        assert valuation.age_multiplier >= 0.9
        assert valuation.value_confidence >= 0.75

    def test_low_value_for_poor_modules(self, assessor, failed_module, critical_performance):
        """Test that poor modules receive low valuations."""
        condition = assessor.condition_grading(
            failed_module, visual_inspection_pass=False, electrical_safety_pass=False
        )
        perf_level, perf_score = assessor.performance_testing(
            failed_module, critical_performance
        )
        _, score, lifetime = assessor.reuse_potential_scoring(
            failed_module, condition, perf_level, perf_score
        )

        valuation = assessor.secondary_market_valuation(
            failed_module, score, perf_level, condition, lifetime
        )

        # Failed module should have minimal value
        assert valuation.final_value_usd < 50
        assert valuation.condition_multiplier < 0.4
        assert valuation.performance_multiplier < 0.4

    def test_market_segment_classification(self, assessor, new_module, mid_life_module):
        """Test that market segments are classified correctly."""
        # Premium module
        condition_premium = assessor.condition_grading(new_module)
        perf_high = PerformanceMetrics(
            measured_power_w=340.0,
            open_circuit_voltage_v=46.8,
            short_circuit_current_a=9.2,
            max_power_voltage_v=38.5,
            max_power_current_a=8.83,
            fill_factor=0.79,
            efficiency_percent=18.5,
        )
        perf_level_high, perf_score_high = assessor.performance_testing(
            new_module, perf_high
        )
        _, score_high, lifetime_high = assessor.reuse_potential_scoring(
            new_module, condition_premium, perf_level_high, perf_score_high
        )

        valuation_premium = assessor.secondary_market_valuation(
            new_module, score_high, perf_level_high, condition_premium, lifetime_high
        )

        assert valuation_premium.market_segment in [
            MarketSegment.PREMIUM,
            MarketSegment.STANDARD,
        ]

        # Off-grid module
        condition_mid = assessor.condition_grading(mid_life_module)
        perf_medium = PerformanceMetrics(
            measured_power_w=240.0,
            open_circuit_voltage_v=44.2,
            short_circuit_current_a=8.1,
            max_power_voltage_v=36.0,
            max_power_current_a=6.67,
            fill_factor=0.76,
            efficiency_percent=16.2,
        )
        perf_level_med, perf_score_med = assessor.performance_testing(
            mid_life_module, perf_medium
        )
        _, score_med, lifetime_med = assessor.reuse_potential_scoring(
            mid_life_module, condition_mid, perf_level_med, perf_score_med
        )

        valuation_offgrid = assessor.secondary_market_valuation(
            mid_life_module, score_med, perf_level_med, condition_mid, lifetime_med
        )

        assert valuation_offgrid.market_segment in [
            MarketSegment.STANDARD,
            MarketSegment.OFF_GRID,
            MarketSegment.DEVELOPING,
        ]

    def test_comparable_sales_generation(self, assessor, mid_life_module, medium_performance):
        """Test that comparable sales are generated."""
        condition = assessor.condition_grading(mid_life_module)
        perf_level, perf_score = assessor.performance_testing(
            mid_life_module, medium_performance
        )
        _, score, lifetime = assessor.reuse_potential_scoring(
            mid_life_module, condition, perf_level, perf_score
        )

        valuation = assessor.secondary_market_valuation(
            mid_life_module, score, perf_level, condition, lifetime
        )

        assert len(valuation.comparable_sales) >= 3
        assert len(valuation.comparable_sales) <= 5
        # Comparables should be close to estimated value
        for sale in valuation.comparable_sales:
            assert 0.5 * valuation.final_value_usd <= sale <= 1.5 * valuation.final_value_usd

    def test_valuation_multipliers_range(self, assessor, mid_life_module, medium_performance):
        """Test that all multipliers are within expected ranges."""
        condition = assessor.condition_grading(mid_life_module)
        perf_level, perf_score = assessor.performance_testing(
            mid_life_module, medium_performance
        )
        _, score, lifetime = assessor.reuse_potential_scoring(
            mid_life_module, condition, perf_level, perf_score
        )

        valuation = assessor.secondary_market_valuation(
            mid_life_module, score, perf_level, condition, lifetime
        )

        assert 0 <= valuation.condition_multiplier <= 1
        assert 0 <= valuation.performance_multiplier <= 1
        assert 0 <= valuation.age_multiplier <= 1
        assert 0 <= valuation.market_demand_multiplier <= 2
        assert 0 <= valuation.value_confidence <= 1


class TestAssessModule:
    """Tests for the main assess_module method."""

    def test_complete_assessment_new_module(self, assessor, new_module):
        """Test complete assessment of a new module."""
        performance = PerformanceMetrics(
            measured_power_w=340.0,
            open_circuit_voltage_v=46.8,
            short_circuit_current_a=9.2,
            max_power_voltage_v=38.5,
            max_power_current_a=8.83,
            fill_factor=0.79,
            efficiency_percent=18.5,
        )

        result = assessor.assess_module(new_module, performance)

        assert result.module_id == "PV-NEW-001"
        assert result.reuse_potential == ReusePotential.DIRECT_REUSE
        assert result.reusability_score >= 80
        assert result.performance_level == PerformanceLevel.HIGH
        assert result.remaining_lifetime_years > 20
        assert result.market_value_usd > 100
        assert len(result.recommended_applications) > 0
        assert len(result.limiting_factors) >= 1
        assert result.confidence_level > 0.7

    def test_complete_assessment_mid_life_module(
        self, assessor, mid_life_module, medium_performance
    ):
        """Test complete assessment of a mid-life module."""
        result = assessor.assess_module(mid_life_module, medium_performance)

        assert result.module_id == "PV-MID-001"
        assert result.reuse_potential in [
            ReusePotential.DIRECT_REUSE,
            ReusePotential.SECONDARY_MARKET,
        ]
        assert 60 <= result.reusability_score < 90
        assert result.performance_level in [PerformanceLevel.HIGH, PerformanceLevel.MEDIUM]
        # Remaining lifetime depends on performance - can be higher for good modules
        assert result.remaining_lifetime_years > 5
        assert result.market_value_usd > 50
        assert len(result.recommended_applications) > 0

    def test_complete_assessment_old_module(self, assessor, old_module, low_performance):
        """Test complete assessment of an old module."""
        result = assessor.assess_module(old_module, low_performance)

        assert result.module_id == "PV-OLD-001"
        # Old modules with low performance and multiple defects may be recycle_only
        assert result.reuse_potential in [
            ReusePotential.SECONDARY_MARKET,
            ReusePotential.COMPONENT_RECOVERY,
            ReusePotential.RECYCLE_ONLY,
        ]
        assert 20 <= result.reusability_score < 70
        assert result.performance_level == PerformanceLevel.LOW
        assert result.remaining_lifetime_years < 10

    def test_complete_assessment_failed_module(
        self, assessor, failed_module, critical_performance
    ):
        """Test complete assessment of a failed module."""
        result = assessor.assess_module(
            failed_module,
            critical_performance,
            visual_inspection_pass=False,
            electrical_safety_pass=False,
        )

        assert result.module_id == "PV-FAIL-001"
        assert result.reuse_potential in [
            ReusePotential.COMPONENT_RECOVERY,
            ReusePotential.RECYCLE_ONLY,
        ]
        assert result.reusability_score < 50
        assert result.performance_level == PerformanceLevel.CRITICAL
        assert result.market_value_usd < 50

    def test_assessment_recommended_applications(self, assessor, new_module):
        """Test that recommended applications are populated."""
        performance = PerformanceMetrics(
            measured_power_w=340.0,
            open_circuit_voltage_v=46.8,
            short_circuit_current_a=9.2,
            max_power_voltage_v=38.5,
            max_power_current_a=8.83,
            fill_factor=0.79,
            efficiency_percent=18.5,
        )

        result = assessor.assess_module(new_module, performance)

        assert len(result.recommended_applications) > 0
        assert isinstance(result.recommended_applications[0], str)

    def test_assessment_limiting_factors(self, assessor, old_module, low_performance):
        """Test that limiting factors are identified."""
        result = assessor.assess_module(old_module, low_performance)

        assert len(result.limiting_factors) > 0
        # Should identify multiple limiting factors for old, poorly performing modules
        # Age may be mentioned directly or indirectly through condition/performance factors
        assert any(
            any(keyword in factor.lower() for keyword in ["age", "condition", "performance", "defect"])
            for factor in result.limiting_factors
        )

    def test_assessment_with_structural_score(self, assessor, mid_life_module, medium_performance):
        """Test assessment with custom structural integrity score."""
        result = assessor.assess_module(
            mid_life_module, medium_performance, structural_integrity_score=85.0
        )

        assert result.condition_assessment.structural_integrity_score == 85.0

    def test_assessment_data_types(self, assessor, new_module, high_performance):
        """Test that assessment returns correct data types."""
        result = assessor.assess_module(new_module, high_performance)

        assert isinstance(result.reusability_score, float)
        assert isinstance(result.remaining_lifetime_years, float)
        assert isinstance(result.market_value_usd, float)
        assert isinstance(result.confidence_level, float)
        assert isinstance(result.recommended_applications, list)
        assert isinstance(result.limiting_factors, list)
        assert isinstance(result.assessment_date, datetime)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_age_module(self, assessor):
        """Test module with zero age."""
        module = ModuleData(
            module_id="ZERO",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=0.0,
        )

        performance = PerformanceMetrics(
            measured_power_w=300.0,
            open_circuit_voltage_v=45.0,
            short_circuit_current_a=8.5,
            max_power_voltage_v=37.0,
            max_power_current_a=8.1,
            fill_factor=0.78,
            efficiency_percent=17.0,
        )

        result = assessor.assess_module(module, performance)

        assert result.reusability_score >= 0
        assert result.remaining_lifetime_years >= 0

    def test_very_old_module(self, assessor):
        """Test module approaching maximum age."""
        module = ModuleData(
            module_id="ANCIENT",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=40.0,
        )

        performance = PerformanceMetrics(
            measured_power_w=150.0,
            open_circuit_voltage_v=40.0,
            short_circuit_current_a=6.0,
            max_power_voltage_v=33.0,
            max_power_current_a=4.5,
            fill_factor=0.65,
            efficiency_percent=12.0,
        )

        result = assessor.assess_module(module, performance)

        assert result.reusability_score >= 0
        assert result.remaining_lifetime_years >= 0
        assert result.market_value_usd >= 0

    def test_perfect_performance(self, assessor):
        """Test module with perfect (100%) performance."""
        module = ModuleData(
            module_id="PERFECT",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=5.0,
        )

        performance = PerformanceMetrics(
            measured_power_w=300.0,  # Exactly nameplate
            open_circuit_voltage_v=45.0,
            short_circuit_current_a=8.5,
            max_power_voltage_v=37.0,
            max_power_current_a=8.1,
            fill_factor=0.85,
            efficiency_percent=20.0,
        )

        result = assessor.assess_module(module, performance)

        assert result.performance_level == PerformanceLevel.HIGH
        assert result.reusability_score >= 80

    def test_multiple_critical_defects(self, assessor):
        """Test module with multiple critical defects."""
        module = ModuleData(
            module_id="MULTI-DEFECT",
            manufacturer="Test",
            model="T1",
            nameplate_power_w=300.0,
            age_years=10.0,
            degradation_types=[
                DegradationType.HOT_SPOT,
                DegradationType.DELAMINATION,
                DegradationType.CELL_CRACK,
                DegradationType.BYPASS_DIODE,
            ],
        )

        performance = PerformanceMetrics(
            measured_power_w=120.0,
            open_circuit_voltage_v=38.0,
            short_circuit_current_a=5.5,
            max_power_voltage_v=30.0,
            max_power_current_a=4.0,
            fill_factor=0.60,
            efficiency_percent=10.0,
        )

        result = assessor.assess_module(module, performance)

        # Should have low reusability due to multiple defects
        assert result.reusability_score < 60
        assert len(result.limiting_factors) >= 2
