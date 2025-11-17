"""Reuse Assessment & Module Grading for PV Modules.

This module provides comprehensive reuse assessment capabilities for photovoltaic modules,
including condition grading, performance testing, reuse potential scoring, and secondary
market valuation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math

from pv_circularity_simulator.core.enums import (
    ModuleCondition,
    PerformanceLevel,
    ReusePotential,
    DegradationType,
    MarketSegment,
)
from pv_circularity_simulator.core.models import (
    ModuleData,
    PerformanceMetrics,
    ConditionAssessment,
    ReuseAssessmentResult,
    MarketValuation,
)


class ReuseAssessor:
    """Comprehensive reuse assessment system for PV modules.

    The ReuseAssessor class provides a complete framework for evaluating the reuse potential
    of photovoltaic modules through multi-dimensional assessment including physical condition,
    electrical performance, remaining lifetime, and market value.

    The assessment process follows industry best practices and standards including:
    - IEC 61215: Crystalline silicon terrestrial photovoltaic modules
    - IEC 61730: Photovoltaic module safety qualification
    - IEC TS 63126: Guidelines for qualifying PV modules, components and materials for operation
      at high temperatures

    Attributes:
        degradation_rate_per_year: Annual degradation rate (default 0.5% per year)
        expected_lifetime_years: Expected module lifetime (default 25 years)
        minimum_performance_threshold: Minimum performance for reuse (default 70%)
        base_module_price_per_watt: Base price per watt for valuation (default $0.50/W)

    Example:
        >>> from pv_circularity_simulator import ReuseAssessor
        >>> from pv_circularity_simulator.core.models import ModuleData, PerformanceMetrics
        >>>
        >>> assessor = ReuseAssessor()
        >>> module = ModuleData(
        ...     module_id="PV-12345",
        ...     manufacturer="SolarTech",
        ...     model="ST-300",
        ...     nameplate_power_w=300.0,
        ...     age_years=10.0
        ... )
        >>> performance = PerformanceMetrics(
        ...     measured_power_w=270.0,
        ...     open_circuit_voltage_v=45.5,
        ...     short_circuit_current_a=8.5,
        ...     max_power_voltage_v=37.2,
        ...     max_power_current_a=7.26,
        ...     fill_factor=0.78,
        ...     efficiency_percent=16.5
        ... )
        >>> result = assessor.assess_module(module, performance)
        >>> print(f"Reusability Score: {result.reusability_score:.1f}/100")
    """

    def __init__(
        self,
        degradation_rate_per_year: float = 0.005,
        expected_lifetime_years: float = 25.0,
        minimum_performance_threshold: float = 0.70,
        base_module_price_per_watt: float = 0.50,
    ) -> None:
        """Initialize the ReuseAssessor with configurable parameters.

        Args:
            degradation_rate_per_year: Annual performance degradation rate (e.g., 0.005 = 0.5%)
            expected_lifetime_years: Expected operational lifetime in years
            minimum_performance_threshold: Minimum performance ratio for reuse consideration
            base_module_price_per_watt: Base market price per watt for new modules (USD/W)

        Raises:
            ValueError: If any parameter is outside valid ranges
        """
        if not 0 < degradation_rate_per_year < 0.05:
            raise ValueError("Degradation rate must be between 0 and 5% per year")
        if not 10 <= expected_lifetime_years <= 50:
            raise ValueError("Expected lifetime must be between 10 and 50 years")
        if not 0 < minimum_performance_threshold <= 1.0:
            raise ValueError("Performance threshold must be between 0 and 1")
        if not 0 < base_module_price_per_watt <= 5.0:
            raise ValueError("Base price must be between 0 and $5/W")

        self.degradation_rate_per_year = degradation_rate_per_year
        self.expected_lifetime_years = expected_lifetime_years
        self.minimum_performance_threshold = minimum_performance_threshold
        self.base_module_price_per_watt = base_module_price_per_watt

        # Defect severity weights (0-1 scale)
        self._defect_severity_weights: Dict[DegradationType, float] = {
            DegradationType.POWER_LOSS: 0.9,
            DegradationType.HOT_SPOT: 0.85,
            DegradationType.DELAMINATION: 0.75,
            DegradationType.CELL_CRACK: 0.70,
            DegradationType.DISCOLORATION: 0.40,
            DegradationType.CORROSION: 0.60,
            DegradationType.JUNCTION_BOX: 0.55,
            DegradationType.BYPASS_DIODE: 0.65,
            DegradationType.ENCAPSULANT: 0.50,
        }

    def condition_grading(
        self,
        module: ModuleData,
        visual_inspection_pass: bool = True,
        electrical_safety_pass: bool = True,
        structural_integrity_score: Optional[float] = None,
    ) -> ConditionAssessment:
        """Grade the physical condition of a PV module.

        Performs comprehensive physical condition assessment based on visual inspection,
        structural integrity, electrical safety, and identified degradation patterns.
        The grading algorithm considers multiple factors:

        - Visual defects and their severity
        - Degradation types and patterns
        - Module age and expected wear
        - Structural integrity
        - Electrical safety compliance

        Args:
            module: ModuleData object containing module information and defects
            visual_inspection_pass: Whether visual inspection passed standards
            electrical_safety_pass: Whether electrical safety tests passed
            structural_integrity_score: Optional structural integrity score (0-100)
                If not provided, will be calculated based on age and defects

        Returns:
            ConditionAssessment object containing:
                - Overall condition grade (EXCELLENT to FAILED)
                - Numerical condition score (0-100)
                - Detailed defect severity scores
                - Inspection results and metadata

        Raises:
            ValueError: If structural_integrity_score is outside 0-100 range

        Note:
            The condition score is calculated using a weighted combination of:
            - Base score adjusted for age (40%)
            - Defect severity impact (30%)
            - Structural integrity (20%)
            - Safety compliance (10%)

        Example:
            >>> module = ModuleData(
            ...     module_id="PV-001",
            ...     manufacturer="SolarCo",
            ...     model="SC-250",
            ...     nameplate_power_w=250.0,
            ...     age_years=8.0,
            ...     degradation_types=[DegradationType.DISCOLORATION]
            ... )
            >>> assessment = assessor.condition_grading(module)
            >>> print(f"Condition: {assessment.overall_condition}")
            >>> print(f"Score: {assessment.condition_score:.1f}/100")
        """
        if structural_integrity_score is not None:
            if not 0 <= structural_integrity_score <= 100:
                raise ValueError("Structural integrity score must be between 0 and 100")

        # Calculate base score adjusted for age (newer = higher score)
        age_factor = max(0, 1 - (module.age_years / self.expected_lifetime_years))
        base_score = 100 * age_factor

        # Calculate defect severity scores
        defect_severity_scores: Dict[str, float] = {}
        total_defect_penalty = 0.0

        for degradation_type in module.degradation_types:
            severity_weight = self._defect_severity_weights.get(degradation_type, 0.5)
            defect_severity_scores[degradation_type.value] = severity_weight * 100
            total_defect_penalty += severity_weight * 20  # Max 20 points per defect

        # Additional penalty for visual defects
        visual_defect_penalty = min(len(module.visual_defects) * 5, 30)

        # Calculate structural integrity score if not provided
        if structural_integrity_score is None:
            # Base structural score on age and number of defects
            structural_base = 100 - (module.age_years * 2)
            structural_defect_penalty = len(module.degradation_types) * 8
            structural_integrity_score = max(
                0, min(100, structural_base - structural_defect_penalty)
            )

        # Calculate defect score (inverse of penalty, 0-100)
        # Cap total penalty at 100 points
        defect_score = max(0, 100 - min(total_defect_penalty, 100))

        # Calculate safety score (0 or 100)
        safety_score = 100 if electrical_safety_pass else 0

        # Calculate final condition score with weighted components (all 0-100)
        condition_score = (
            base_score * 0.4  # Age factor (40%)
            + defect_score * 0.3  # Defect score (30%)
            + structural_integrity_score * 0.2  # Structural integrity (20%)
            + safety_score * 0.1  # Safety (10%)
        )

        # Apply visual inspection penalty
        if not visual_inspection_pass:
            condition_score *= 0.7  # 30% penalty for failed visual inspection

        condition_score = max(0, min(100, condition_score))

        # Determine overall condition grade
        overall_condition = self._score_to_condition(condition_score)

        return ConditionAssessment(
            overall_condition=overall_condition,
            condition_score=condition_score,
            visual_inspection_pass=visual_inspection_pass,
            structural_integrity_score=structural_integrity_score,
            electrical_safety_pass=electrical_safety_pass,
            defect_severity_scores=defect_severity_scores,
            inspection_date=datetime.now(),
        )

    def performance_testing(
        self,
        module: ModuleData,
        performance: PerformanceMetrics,
    ) -> Tuple[PerformanceLevel, float]:
        """Test and classify module electrical performance.

        Analyzes electrical performance test results to determine the module's current
        performance level and calculate a comprehensive performance score. The analysis
        includes:

        - Power output ratio vs. nameplate rating
        - Fill factor analysis
        - Efficiency assessment
        - I-V curve characteristic evaluation
        - Temperature coefficient consideration
        - Age-adjusted performance expectations

        Args:
            module: ModuleData object with module specifications
            performance: PerformanceMetrics object with test results

        Returns:
            Tuple containing:
                - PerformanceLevel: Classification (HIGH, MEDIUM, LOW, CRITICAL)
                - float: Performance score (0-100)

        Note:
            Performance levels are classified as:
            - HIGH: ≥90% of expected performance
            - MEDIUM: 70-90% of expected performance
            - LOW: 50-70% of expected performance
            - CRITICAL: <50% of expected performance

            The performance score considers multiple factors:
            - Power retention (50%)
            - Fill factor quality (25%)
            - Efficiency vs. age (15%)
            - I-V characteristics (10%)

        Example:
            >>> module = ModuleData(
            ...     module_id="PV-002",
            ...     manufacturer="SunPower",
            ...     model="SP-350",
            ...     nameplate_power_w=350.0,
            ...     age_years=5.0
            ... )
            >>> performance = PerformanceMetrics(
            ...     measured_power_w=320.0,
            ...     open_circuit_voltage_v=48.2,
            ...     short_circuit_current_a=9.1,
            ...     max_power_voltage_v=39.8,
            ...     max_power_current_a=8.04,
            ...     fill_factor=0.79,
            ...     efficiency_percent=18.2
            ... )
            >>> level, score = assessor.performance_testing(module, performance)
            >>> print(f"Performance: {level}, Score: {score:.1f}")
        """
        # Calculate expected power considering age-related degradation
        expected_degradation = 1 - (module.age_years * self.degradation_rate_per_year)
        expected_power = module.nameplate_power_w * expected_degradation

        # Calculate power retention ratio
        power_ratio = performance.measured_power_w / module.nameplate_power_w
        power_score = min(100, power_ratio * 100)

        # Calculate performance relative to age-adjusted expectations
        age_adjusted_ratio = performance.measured_power_w / expected_power
        age_adjusted_score = min(100, age_adjusted_ratio * 100)

        # Fill factor assessment (typical range 0.70-0.85 for crystalline silicon)
        fill_factor_score = 0.0
        if performance.fill_factor >= 0.75:
            fill_factor_score = 100
        elif performance.fill_factor >= 0.70:
            fill_factor_score = 80
        elif performance.fill_factor >= 0.65:
            fill_factor_score = 60
        elif performance.fill_factor >= 0.60:
            fill_factor_score = 40
        else:
            fill_factor_score = 20

        # Efficiency assessment relative to age
        # Typical modules: 15-22% efficiency when new
        efficiency_score = min(100, (performance.efficiency_percent / 20.0) * 100)

        # I-V curve characteristics score
        # Check if voltage and current values are within reasonable ranges
        voc_ratio = performance.open_circuit_voltage_v / performance.max_power_voltage_v
        isc_ratio = performance.short_circuit_current_a / performance.max_power_current_a

        iv_score = 100.0
        # Typical Voc/Vmp ratio should be around 1.15-1.30
        if not 1.10 <= voc_ratio <= 1.40:
            iv_score -= 20
        # Typical Isc/Imp ratio should be around 1.10-1.20
        if not 1.05 <= isc_ratio <= 1.30:
            iv_score -= 20

        iv_score = max(0, iv_score)

        # Weighted performance score
        performance_score = (
            age_adjusted_score * 0.50  # Power retention (50%)
            + fill_factor_score * 0.25  # Fill factor (25%)
            + efficiency_score * 0.15  # Efficiency (15%)
            + iv_score * 0.10  # I-V characteristics (10%)
        )

        performance_score = max(0, min(100, performance_score))

        # Classify performance level
        performance_level = self._score_to_performance_level(power_ratio)

        return performance_level, performance_score

    def reuse_potential_scoring(
        self,
        module: ModuleData,
        condition_assessment: ConditionAssessment,
        performance_level: PerformanceLevel,
        performance_score: float,
    ) -> Tuple[ReusePotential, float, float]:
        """Calculate comprehensive reuse potential score and classification.

        Evaluates the overall reuse potential by integrating condition assessment,
        performance testing results, module age, and market factors. The scoring
        algorithm uses a multi-criteria decision analysis approach.

        Args:
            module: ModuleData object with module information
            condition_assessment: ConditionAssessment from condition_grading()
            performance_level: PerformanceLevel from performance_testing()
            performance_score: Performance score from performance_testing()

        Returns:
            Tuple containing:
                - ReusePotential: Classification for reuse pathway
                - float: Overall reusability score (0-100)
                - float: Estimated remaining lifetime in years

        Note:
            Reuse potential classifications:
            - DIRECT_REUSE: Score ≥80, suitable for premium applications
            - SECONDARY_MARKET: Score 60-80, suitable for standard applications
            - COMPONENT_RECOVERY: Score 40-60, extract valuable components
            - RECYCLE_ONLY: Score 20-40, recycle materials only
            - DISPOSE: Score <20, requires proper disposal

            Reusability score factors:
            - Condition assessment (35%)
            - Performance score (35%)
            - Remaining lifetime (20%)
            - Safety and compliance (10%)

        Example:
            >>> condition = assessor.condition_grading(module)
            >>> perf_level, perf_score = assessor.performance_testing(module, performance)
            >>> potential, score, lifetime = assessor.reuse_potential_scoring(
            ...     module, condition, perf_level, perf_score
            ... )
            >>> print(f"Reuse: {potential}, Score: {score:.1f}, Life: {lifetime:.1f}y")
        """
        # Calculate remaining lifetime based on current performance
        # Assuming linear degradation model
        if performance_score >= 90:
            # High performance - long remaining life
            performance_retention = performance_score / 100
            years_of_degradation = module.age_years
            remaining_capacity = performance_retention - self.minimum_performance_threshold

            if self.degradation_rate_per_year > 0:
                remaining_years = remaining_capacity / self.degradation_rate_per_year
            else:
                remaining_years = self.expected_lifetime_years - module.age_years

            remaining_lifetime = max(0, min(remaining_years, self.expected_lifetime_years))
        else:
            # Lower performance - calculate based on threshold
            current_performance_ratio = performance_score / 100
            remaining_performance = current_performance_ratio - self.minimum_performance_threshold

            if remaining_performance <= 0:
                remaining_lifetime = 0
            elif self.degradation_rate_per_year > 0:
                remaining_lifetime = max(
                    0, remaining_performance / self.degradation_rate_per_year
                )
            else:
                remaining_lifetime = max(0, self.expected_lifetime_years - module.age_years)

        # Calculate lifetime score (0-100)
        lifetime_score = min(
            100, (remaining_lifetime / self.expected_lifetime_years) * 100
        )

        # Safety and compliance score
        safety_score = 0.0
        if condition_assessment.electrical_safety_pass:
            safety_score += 50
        if condition_assessment.visual_inspection_pass:
            safety_score += 50

        # Weighted reusability score
        reusability_score = (
            condition_assessment.condition_score * 0.35  # Condition (35%)
            + performance_score * 0.35  # Performance (35%)
            + lifetime_score * 0.20  # Remaining lifetime (20%)
            + safety_score * 0.10  # Safety & compliance (10%)
        )

        # Apply penalties for critical issues
        if condition_assessment.overall_condition == ModuleCondition.FAILED:
            reusability_score *= 0.3
        elif condition_assessment.overall_condition == ModuleCondition.POOR:
            reusability_score *= 0.6

        if performance_level == PerformanceLevel.CRITICAL:
            reusability_score *= 0.4

        # Apply bonus for excellent condition and performance
        if (condition_assessment.overall_condition == ModuleCondition.EXCELLENT and
            performance_level == PerformanceLevel.HIGH):
            reusability_score = min(100, reusability_score * 1.1)

        reusability_score = max(0, min(100, reusability_score))

        # Determine reuse potential classification
        reuse_potential = self._score_to_reuse_potential(
            reusability_score, condition_assessment, performance_level
        )

        return reuse_potential, reusability_score, remaining_lifetime

    def secondary_market_valuation(
        self,
        module: ModuleData,
        reusability_score: float,
        performance_level: PerformanceLevel,
        condition_assessment: ConditionAssessment,
        remaining_lifetime: float,
    ) -> MarketValuation:
        """Calculate secondary market value for reused modules.

        Determines the fair market value of a used PV module based on condition,
        performance, remaining lifetime, and market factors. The valuation model
        considers both intrinsic module value and market dynamics.

        Args:
            module: ModuleData object with module specifications
            reusability_score: Overall reusability score from reuse_potential_scoring()
            performance_level: Performance classification
            condition_assessment: Condition assessment results
            remaining_lifetime: Estimated remaining lifetime in years

        Returns:
            MarketValuation object containing:
                - Base value and multipliers
                - Final estimated market value in USD
                - Valuation confidence level
                - Target market segment
                - Comparable sales data (when available)

        Note:
            Valuation multipliers:
            - Condition: 0.3-1.0 based on condition grade
            - Performance: 0.4-1.0 based on performance level
            - Age: 0.3-1.0 based on remaining lifetime ratio
            - Market demand: 0.8-1.5 based on segment and availability

            Market segments:
            - PREMIUM: High-performance modules for demanding applications
            - STANDARD: General residential/commercial use
            - OFF_GRID: Remote and off-grid applications
            - DEVELOPING: Developing market applications
            - INDUSTRIAL: Large-scale industrial use

        Example:
            >>> valuation = assessor.secondary_market_valuation(
            ...     module, reusability_score, perf_level, condition, lifetime
            ... )
            >>> print(f"Value: ${valuation.final_value_usd:.2f}")
            >>> print(f"Segment: {valuation.market_segment}")
            >>> print(f"Confidence: {valuation.value_confidence:.2f}")
        """
        # Calculate base value from nameplate power
        base_value_usd = module.nameplate_power_w * self.base_module_price_per_watt

        # Condition multiplier (0.3 - 1.0)
        condition_multipliers = {
            ModuleCondition.EXCELLENT: 0.90,
            ModuleCondition.GOOD: 0.75,
            ModuleCondition.FAIR: 0.55,
            ModuleCondition.POOR: 0.35,
            ModuleCondition.FAILED: 0.10,
        }
        condition_multiplier = condition_multipliers.get(
            condition_assessment.overall_condition, 0.5
        )

        # Performance multiplier (0.4 - 1.0)
        performance_multipliers = {
            PerformanceLevel.HIGH: 0.95,
            PerformanceLevel.MEDIUM: 0.75,
            PerformanceLevel.LOW: 0.50,
            PerformanceLevel.CRITICAL: 0.20,
        }
        performance_multiplier = performance_multipliers.get(performance_level, 0.5)

        # Age multiplier based on remaining lifetime (0.3 - 1.0)
        lifetime_ratio = remaining_lifetime / self.expected_lifetime_years
        age_multiplier = 0.3 + (0.7 * lifetime_ratio)
        age_multiplier = max(0.3, min(1.0, age_multiplier))

        # Determine market segment based on performance and condition
        market_segment = self._determine_market_segment(
            performance_level, condition_assessment.overall_condition, reusability_score
        )

        # Market demand multiplier (0.8 - 1.5)
        # Premium segments have lower demand but higher prices
        # Developing markets have higher demand for affordable modules
        market_demand_multipliers = {
            MarketSegment.PREMIUM: 0.90,
            MarketSegment.STANDARD: 1.00,
            MarketSegment.OFF_GRID: 1.20,
            MarketSegment.DEVELOPING: 1.30,
            MarketSegment.INDUSTRIAL: 1.10,
        }
        market_demand_multiplier = market_demand_multipliers.get(market_segment, 1.0)

        # Calculate final value
        final_value_usd = (
            base_value_usd
            * condition_multiplier
            * performance_multiplier
            * age_multiplier
            * market_demand_multiplier
        )

        # Ensure minimum value for salvageable modules
        if reusability_score > 20:
            final_value_usd = max(final_value_usd, module.nameplate_power_w * 0.05)

        # Calculate valuation confidence (0-1)
        # Higher confidence for modules with complete data and mid-range values
        confidence_factors = []

        # Data completeness
        if module.manufacture_date is not None:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.8)

        # Condition assessment certainty
        if condition_assessment.visual_inspection_pass:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)

        # Performance testing confidence
        if performance_level in [PerformanceLevel.HIGH, PerformanceLevel.MEDIUM]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)

        # Market segment certainty
        if market_segment in [MarketSegment.STANDARD, MarketSegment.OFF_GRID]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.75)

        value_confidence = sum(confidence_factors) / len(confidence_factors)

        # Generate comparable sales (simulated for demonstration)
        # In production, this would query actual market data
        comparable_sales = self._generate_comparable_sales(
            final_value_usd, value_confidence
        )

        return MarketValuation(
            base_value_usd=base_value_usd,
            condition_multiplier=condition_multiplier,
            performance_multiplier=performance_multiplier,
            age_multiplier=age_multiplier,
            market_demand_multiplier=market_demand_multiplier,
            final_value_usd=final_value_usd,
            value_confidence=value_confidence,
            comparable_sales=comparable_sales,
            market_segment=market_segment,
            valuation_date=datetime.now(),
        )

    def assess_module(
        self,
        module: ModuleData,
        performance: PerformanceMetrics,
        visual_inspection_pass: bool = True,
        electrical_safety_pass: bool = True,
        structural_integrity_score: Optional[float] = None,
    ) -> ReuseAssessmentResult:
        """Perform complete reuse assessment for a PV module.

        This is the main entry point for comprehensive module assessment, integrating
        all assessment methods into a single workflow. It performs:

        1. Condition grading
        2. Performance testing
        3. Reuse potential scoring
        4. Secondary market valuation

        Args:
            module: ModuleData object with module information
            performance: PerformanceMetrics object with test results
            visual_inspection_pass: Whether visual inspection passed
            electrical_safety_pass: Whether electrical safety tests passed
            structural_integrity_score: Optional structural score (0-100)

        Returns:
            ReuseAssessmentResult containing complete assessment with:
                - Condition and performance analysis
                - Reuse potential classification and score
                - Market valuation
                - Recommended applications
                - Limiting factors
                - Confidence metrics

        Example:
            >>> from pv_circularity_simulator import ReuseAssessor
            >>> from pv_circularity_simulator.core.models import ModuleData, PerformanceMetrics
            >>>
            >>> assessor = ReuseAssessor()
            >>> module = ModuleData(
            ...     module_id="PV-12345",
            ...     manufacturer="SolarTech",
            ...     model="ST-300",
            ...     nameplate_power_w=300.0,
            ...     age_years=10.0,
            ...     degradation_types=[DegradationType.DISCOLORATION]
            ... )
            >>> performance = PerformanceMetrics(
            ...     measured_power_w=270.0,
            ...     open_circuit_voltage_v=45.5,
            ...     short_circuit_current_a=8.5,
            ...     max_power_voltage_v=37.2,
            ...     max_power_current_a=7.26,
            ...     fill_factor=0.78,
            ...     efficiency_percent=16.5
            ... )
            >>> result = assessor.assess_module(module, performance)
            >>> print(f"Module: {result.module_id}")
            >>> print(f"Reuse Potential: {result.reuse_potential}")
            >>> print(f"Score: {result.reusability_score:.1f}/100")
            >>> print(f"Market Value: ${result.market_value_usd:.2f}")
            >>> print(f"Applications: {', '.join(result.recommended_applications)}")
        """
        # Step 1: Condition grading
        condition_assessment = self.condition_grading(
            module=module,
            visual_inspection_pass=visual_inspection_pass,
            electrical_safety_pass=electrical_safety_pass,
            structural_integrity_score=structural_integrity_score,
        )

        # Step 2: Performance testing
        performance_level, performance_score = self.performance_testing(
            module=module, performance=performance
        )

        # Step 3: Reuse potential scoring
        reuse_potential, reusability_score, remaining_lifetime = self.reuse_potential_scoring(
            module=module,
            condition_assessment=condition_assessment,
            performance_level=performance_level,
            performance_score=performance_score,
        )

        # Step 4: Secondary market valuation
        market_valuation = self.secondary_market_valuation(
            module=module,
            reusability_score=reusability_score,
            performance_level=performance_level,
            condition_assessment=condition_assessment,
            remaining_lifetime=remaining_lifetime,
        )

        # Determine recommended applications
        recommended_applications = self._determine_applications(
            reuse_potential, performance_level, market_valuation.market_segment
        )

        # Identify limiting factors
        limiting_factors = self._identify_limiting_factors(
            module, condition_assessment, performance_level, reusability_score
        )

        # Calculate overall confidence
        confidence_level = (
            market_valuation.value_confidence * 0.6  # Market valuation confidence
            + (1.0 if condition_assessment.electrical_safety_pass else 0.5) * 0.2
            + (1.0 if condition_assessment.visual_inspection_pass else 0.5) * 0.2
        )

        return ReuseAssessmentResult(
            module_id=module.module_id,
            reuse_potential=reuse_potential,
            reusability_score=reusability_score,
            condition_assessment=condition_assessment,
            performance_metrics=performance,
            performance_level=performance_level,
            remaining_lifetime_years=remaining_lifetime,
            recommended_applications=recommended_applications,
            market_value_usd=market_valuation.final_value_usd,
            market_segment=market_valuation.market_segment,
            confidence_level=confidence_level,
            limiting_factors=limiting_factors,
            assessment_date=datetime.now(),
        )

    # Helper methods

    def _score_to_condition(self, score: float) -> ModuleCondition:
        """Convert numerical score to condition grade."""
        if score >= 85:
            return ModuleCondition.EXCELLENT
        elif score >= 70:
            return ModuleCondition.GOOD
        elif score >= 50:
            return ModuleCondition.FAIR
        elif score >= 30:
            return ModuleCondition.POOR
        else:
            return ModuleCondition.FAILED

    def _score_to_performance_level(self, power_ratio: float) -> PerformanceLevel:
        """Convert power ratio to performance level."""
        if power_ratio >= 0.90:
            return PerformanceLevel.HIGH
        elif power_ratio >= 0.70:
            return PerformanceLevel.MEDIUM
        elif power_ratio >= 0.50:
            return PerformanceLevel.LOW
        else:
            return PerformanceLevel.CRITICAL

    def _score_to_reuse_potential(
        self,
        score: float,
        condition: ConditionAssessment,
        performance: PerformanceLevel,
    ) -> ReusePotential:
        """Convert score and assessments to reuse potential classification."""
        # Critical failures lead to recycling or disposal
        if condition.overall_condition == ModuleCondition.FAILED:
            return ReusePotential.RECYCLE_ONLY
        if performance == PerformanceLevel.CRITICAL:
            return ReusePotential.COMPONENT_RECOVERY

        # Score-based classification with adjustments
        if score >= 80:
            return ReusePotential.DIRECT_REUSE
        elif score >= 60:
            return ReusePotential.SECONDARY_MARKET
        elif score >= 40:
            return ReusePotential.COMPONENT_RECOVERY
        elif score >= 20:
            return ReusePotential.RECYCLE_ONLY
        else:
            return ReusePotential.DISPOSE

    def _determine_market_segment(
        self,
        performance: PerformanceLevel,
        condition: ModuleCondition,
        score: float,
    ) -> MarketSegment:
        """Determine target market segment for the module."""
        if performance == PerformanceLevel.HIGH and condition == ModuleCondition.EXCELLENT:
            return MarketSegment.PREMIUM
        elif performance == PerformanceLevel.HIGH and condition in [
            ModuleCondition.EXCELLENT,
            ModuleCondition.GOOD,
        ]:
            return MarketSegment.STANDARD
        elif performance in [PerformanceLevel.MEDIUM, PerformanceLevel.HIGH]:
            return MarketSegment.OFF_GRID
        elif score >= 50:
            return MarketSegment.DEVELOPING
        else:
            return MarketSegment.INDUSTRIAL

    def _determine_applications(
        self,
        reuse_potential: ReusePotential,
        performance: PerformanceLevel,
        segment: MarketSegment,
    ) -> List[str]:
        """Determine recommended applications based on assessment."""
        applications = []

        if reuse_potential == ReusePotential.DIRECT_REUSE:
            if segment == MarketSegment.PREMIUM:
                applications.extend([
                    "Grid-tied residential systems",
                    "Commercial rooftop installations",
                    "Premium off-grid systems",
                ])
            elif segment == MarketSegment.STANDARD:
                applications.extend([
                    "Residential solar systems",
                    "Small commercial installations",
                    "Community solar projects",
                ])
            elif segment == MarketSegment.OFF_GRID:
                applications.extend([
                    "Off-grid residential systems",
                    "Remote power applications",
                    "Backup power systems",
                    "Agricultural installations",
                ])
            elif segment == MarketSegment.INDUSTRIAL:
                applications.extend([
                    "Industrial power systems",
                    "Utility-scale installations",
                    "Large commercial projects",
                ])

        elif reuse_potential == ReusePotential.SECONDARY_MARKET:
            if segment == MarketSegment.OFF_GRID:
                applications.extend([
                    "Off-grid cabin systems",
                    "RV and marine applications",
                    "Remote telecommunications",
                    "Agricultural applications",
                ])
            elif segment == MarketSegment.DEVELOPING:
                applications.extend([
                    "Rural electrification projects",
                    "Developing market installations",
                    "Emergency power systems",
                    "Educational installations",
                ])

        elif reuse_potential == ReusePotential.COMPONENT_RECOVERY:
            applications.extend([
                "Cell recovery for manufacturing",
                "Glass recycling",
                "Frame and junction box reuse",
                "Semiconductor material recovery",
            ])

        elif reuse_potential == ReusePotential.RECYCLE_ONLY:
            applications.extend([
                "Silicon recycling",
                "Metal recovery (aluminum, copper, silver)",
                "Glass recycling",
            ])

        return applications if applications else ["Not suitable for reuse"]

    def _identify_limiting_factors(
        self,
        module: ModuleData,
        condition: ConditionAssessment,
        performance: PerformanceLevel,
        score: float,
    ) -> List[str]:
        """Identify factors limiting reuse potential."""
        factors = []

        # Age-related factors
        if module.age_years > 20:
            factors.append(f"Advanced age ({module.age_years:.1f} years)")

        # Condition factors
        if condition.overall_condition in [ModuleCondition.POOR, ModuleCondition.FAILED]:
            factors.append(f"Poor physical condition ({condition.overall_condition.value})")

        if not condition.electrical_safety_pass:
            factors.append("Electrical safety concerns")

        if not condition.visual_inspection_pass:
            factors.append("Visual inspection failures")

        if condition.structural_integrity_score < 60:
            factors.append(
                f"Low structural integrity ({condition.structural_integrity_score:.0f}/100)"
            )

        # Performance factors
        if performance in [PerformanceLevel.LOW, PerformanceLevel.CRITICAL]:
            factors.append(f"Low performance level ({performance.value})")

        # Defect factors
        critical_defects = [
            DegradationType.HOT_SPOT,
            DegradationType.DELAMINATION,
            DegradationType.CELL_CRACK,
        ]
        for defect in module.degradation_types:
            if defect in critical_defects:
                factors.append(f"Critical defect: {defect.value}")

        # Score factors
        if score < 50:
            factors.append(f"Low overall reusability score ({score:.0f}/100)")

        return factors if factors else ["No significant limiting factors"]

    def _generate_comparable_sales(
        self, estimated_value: float, confidence: float
    ) -> List[float]:
        """Generate comparable sales data (simulated).

        In production, this would query actual market data from sales databases.
        """
        # Generate 3-5 comparable values with some variance
        import random

        random.seed(int(estimated_value * 100))  # Reproducible for same value

        num_comparables = random.randint(3, 5)
        variance = 0.15 * (1 - confidence)  # Lower confidence = higher variance

        comparables = []
        for _ in range(num_comparables):
            multiplier = 1 + random.uniform(-variance, variance)
            comparable = estimated_value * multiplier
            comparables.append(round(comparable, 2))

        return sorted(comparables)
