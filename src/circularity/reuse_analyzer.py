"""
Reuse Assessment & Second-Life Applications (B11-S02)

This module provides tools for evaluating PV modules for reuse potential,
assessing residual capacity, and analyzing second-life market opportunities.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum


class ModuleCondition(str, Enum):
    """Module physical condition categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


class DefectType(str, Enum):
    """Types of module defects."""
    CRACK = "crack"
    DELAMINATION = "delamination"
    DISCOLORATION = "discoloration"
    HOTSPOT = "hotspot"
    CORROSION = "corrosion"
    SNAIL_TRAIL = "snail_trail"
    PID = "pid"
    BURN_MARK = "burn_mark"


class ModuleTestResults(BaseModel):
    """Results from module testing procedures."""

    visual_inspection_passed: bool = Field(description="Visual inspection result")
    electrical_test_passed: bool = Field(description="Electrical test result")
    insulation_test_passed: bool = Field(description="Insulation resistance test result")
    current_power_w: float = Field(ge=0, description="Current power output in watts")
    rated_power_w: float = Field(ge=0, description="Original rated power in watts")
    voltage_v: float = Field(ge=0, description="Open circuit voltage")
    current_a: float = Field(ge=0, description="Short circuit current")
    fill_factor: float = Field(ge=0, le=1, description="Fill factor")
    insulation_resistance_mohm: float = Field(ge=0, description="Insulation resistance in MΩ")
    defects: List[DefectType] = Field(default_factory=list, description="Detected defects")
    condition: ModuleCondition = Field(description="Overall module condition")

    @property
    def power_degradation(self) -> float:
        """Power degradation from rated capacity (0-1)."""
        if self.rated_power_w == 0:
            return 0.0
        return (self.rated_power_w - self.current_power_w) / self.rated_power_w

    @property
    def capacity_retention(self) -> float:
        """Remaining capacity as fraction of original (0-1)."""
        return 1.0 - self.power_degradation

    @property
    def passed_all_tests(self) -> bool:
        """Whether module passed all critical tests."""
        return (
            self.visual_inspection_passed and
            self.electrical_test_passed and
            self.insulation_test_passed
        )


class ReuseEligibility(BaseModel):
    """Reuse eligibility assessment results."""

    is_eligible: bool = Field(description="Whether module is eligible for reuse")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in assessment (0-1)")
    reasons: List[str] = Field(default_factory=list, description="Reasons for eligibility decision")
    estimated_remaining_life_years: float = Field(ge=0, description="Estimated remaining useful life")
    warranty_eligible: bool = Field(description="Whether eligible for reuse warranty")
    certification_required: List[str] = Field(default_factory=list, description="Required certifications")


class SecondLifeMarket(BaseModel):
    """Second-life market opportunity analysis."""

    market_name: str = Field(description="Market segment name")
    application: str = Field(description="Application type")
    typical_price_per_watt: float = Field(ge=0, description="Market price per watt in USD")
    demand_level: Literal["low", "medium", "high"] = Field(description="Market demand level")
    technical_requirements: Dict[str, float] = Field(description="Technical requirements")
    market_size_mw: Optional[float] = Field(default=None, ge=0, description="Market size in MW")
    growth_rate: Optional[float] = Field(default=None, description="Annual growth rate")


class ResidualCapacityAnalysis(BaseModel):
    """Analysis of module residual capacity."""

    current_capacity_w: float = Field(ge=0, description="Current capacity in watts")
    original_capacity_w: float = Field(ge=0, description="Original capacity in watts")
    capacity_retention_pct: float = Field(ge=0, le=100, description="Capacity retention percentage")
    degradation_rate_per_year: float = Field(description="Historical degradation rate %/year")
    estimated_remaining_life_years: float = Field(ge=0, description="Estimated remaining useful life")
    performance_ratio: float = Field(ge=0, le=1, description="Performance ratio")
    temperature_coefficient: float = Field(description="Temperature coefficient %/°C")


class ReuseAnalyzer:
    """
    Analyzer for PV module reuse assessment and second-life applications.

    Provides methods for module testing evaluation, residual capacity analysis,
    and second-life market matching.
    """

    def __init__(
        self,
        min_capacity_retention: float = 0.80,
        min_insulation_resistance: float = 40.0,
        safety_factor: float = 0.9
    ):
        """
        Initialize the reuse analyzer.

        Args:
            min_capacity_retention: Minimum capacity retention for reuse (0-1)
            min_insulation_resistance: Minimum insulation resistance in MΩ
            safety_factor: Safety factor for lifetime estimation
        """
        self.min_capacity_retention = min_capacity_retention
        self.min_insulation_resistance = min_insulation_resistance
        self.safety_factor = safety_factor

    def module_testing(
        self,
        test_results: ModuleTestResults,
        age_years: float,
        operating_conditions: Optional[Dict[str, float]] = None
    ) -> ReuseEligibility:
        """
        Evaluate module testing results for reuse eligibility.

        Args:
            test_results: Module test results
            age_years: Module age in years
            operating_conditions: Historical operating conditions

        Returns:
            ReuseEligibility assessment
        """
        reasons = []
        is_eligible = True
        confidence = 1.0

        # Check critical safety tests
        if not test_results.passed_all_tests:
            is_eligible = False
            reasons.append("Failed critical safety tests")
            confidence *= 0.5

        # Check insulation resistance
        if test_results.insulation_resistance_mohm < self.min_insulation_resistance:
            is_eligible = False
            reasons.append(f"Insulation resistance too low ({test_results.insulation_resistance_mohm:.1f} MΩ)")
            confidence *= 0.3

        # Check capacity retention
        if test_results.capacity_retention < self.min_capacity_retention:
            is_eligible = False
            reasons.append(f"Capacity retention too low ({test_results.capacity_retention*100:.1f}%)")
            confidence *= 0.6

        # Check for critical defects
        critical_defects = [DefectType.BURN_MARK, DefectType.DELAMINATION]
        if any(defect in test_results.defects for defect in critical_defects):
            is_eligible = False
            reasons.append(f"Critical defects detected: {[d.value for d in test_results.defects if d in critical_defects]}")
            confidence *= 0.4

        # Assess condition
        condition_scores = {
            ModuleCondition.EXCELLENT: 1.0,
            ModuleCondition.GOOD: 0.9,
            ModuleCondition.FAIR: 0.7,
            ModuleCondition.POOR: 0.4,
            ModuleCondition.FAILED: 0.0
        }
        condition_score = condition_scores.get(test_results.condition, 0.5)
        confidence *= condition_score

        if test_results.condition in [ModuleCondition.POOR, ModuleCondition.FAILED]:
            is_eligible = False
            reasons.append(f"Module condition: {test_results.condition.value}")

        # Estimate remaining life
        degradation_rate = test_results.power_degradation / age_years if age_years > 0 else 0.02
        remaining_capacity = test_results.capacity_retention - self.min_capacity_retention
        estimated_life = (remaining_capacity / degradation_rate) * self.safety_factor if degradation_rate > 0 else 15.0
        estimated_life = max(0, min(estimated_life, 25 - age_years))

        # Determine warranty eligibility (typically requires >85% capacity)
        warranty_eligible = test_results.capacity_retention >= 0.85 and is_eligible

        # Required certifications
        certifications = []
        if is_eligible:
            certifications.append("IEC 61215 re-certification")
            if warranty_eligible:
                certifications.append("Reuse warranty certification")
            if test_results.insulation_resistance_mohm >= 50:
                certifications.append("Enhanced safety certification")

        if is_eligible:
            reasons.append(f"Passed all criteria with {test_results.capacity_retention*100:.1f}% capacity retention")

        return ReuseEligibility(
            is_eligible=is_eligible,
            confidence_score=max(0, min(1, confidence)),
            reasons=reasons,
            estimated_remaining_life_years=estimated_life,
            warranty_eligible=warranty_eligible,
            certification_required=certifications
        )

    def residual_capacity(
        self,
        current_power_w: float,
        rated_power_w: float,
        age_years: float,
        operating_hours: Optional[float] = None,
        performance_data: Optional[List[float]] = None
    ) -> ResidualCapacityAnalysis:
        """
        Analyze residual capacity of PV module.

        Args:
            current_power_w: Current power output in watts
            rated_power_w: Original rated power in watts
            age_years: Module age in years
            operating_hours: Total operating hours (optional)
            performance_data: Historical performance data (optional)

        Returns:
            ResidualCapacityAnalysis with detailed capacity assessment
        """
        # Calculate capacity retention
        capacity_retention = (current_power_w / rated_power_w) if rated_power_w > 0 else 0
        capacity_retention_pct = capacity_retention * 100

        # Calculate degradation rate
        if performance_data and len(performance_data) > 1:
            # Use actual historical data if available
            degradation_rate = self._calculate_degradation_rate(performance_data, age_years)
        else:
            # Estimate from current state
            degradation_pct = (1 - capacity_retention) * 100
            degradation_rate = degradation_pct / age_years if age_years > 0 else 0.5

        # Estimate remaining life
        # Assuming end-of-life at 80% capacity
        remaining_capacity_pct = capacity_retention_pct - 80.0
        if degradation_rate > 0:
            estimated_life = (remaining_capacity_pct / degradation_rate) * self.safety_factor
            estimated_life = max(0, min(estimated_life, 30 - age_years))
        else:
            estimated_life = 20.0

        # Calculate performance ratio (accounting for environmental factors)
        # Typical performance ratio is 0.75-0.85 for well-maintained systems
        performance_ratio = capacity_retention * 0.85

        # Typical temperature coefficient for silicon modules
        temperature_coefficient = -0.4  # %/°C

        return ResidualCapacityAnalysis(
            current_capacity_w=current_power_w,
            original_capacity_w=rated_power_w,
            capacity_retention_pct=capacity_retention_pct,
            degradation_rate_per_year=degradation_rate,
            estimated_remaining_life_years=estimated_life,
            performance_ratio=performance_ratio,
            temperature_coefficient=temperature_coefficient
        )

    def second_life_markets(
        self,
        capacity_retention: float,
        available_quantity_kw: float,
        module_specs: Dict[str, float],
        location: str = "global"
    ) -> List[SecondLifeMarket]:
        """
        Identify and analyze second-life market opportunities.

        Args:
            capacity_retention: Current capacity retention (0-1)
            available_quantity_kw: Available quantity in kW
            module_specs: Module specifications (voltage, current, etc.)
            location: Geographic location for market analysis

        Returns:
            List of SecondLifeMarket opportunities ranked by suitability
        """
        markets = []

        # Market 1: Off-grid residential (developing countries)
        if capacity_retention >= 0.70:
            markets.append(SecondLifeMarket(
                market_name="Off-Grid Residential",
                application="Remote home electrification",
                typical_price_per_watt=0.15 if capacity_retention >= 0.80 else 0.10,
                demand_level="high",
                technical_requirements={
                    "min_capacity_retention": 0.70,
                    "min_voltage": 12.0,
                    "max_age_years": 15.0
                },
                market_size_mw=5000.0,
                growth_rate=0.08
            ))

        # Market 2: Agricultural applications
        if capacity_retention >= 0.75:
            markets.append(SecondLifeMarket(
                market_name="Agricultural Applications",
                application="Irrigation pumps, farm equipment",
                typical_price_per_watt=0.20,
                demand_level="medium",
                technical_requirements={
                    "min_capacity_retention": 0.75,
                    "min_power_w": 100.0,
                    "reliability": 0.85
                },
                market_size_mw=2000.0,
                growth_rate=0.12
            ))

        # Market 3: Energy storage systems
        if capacity_retention >= 0.80:
            markets.append(SecondLifeMarket(
                market_name="Energy Storage Integration",
                application="Battery charging, grid stabilization",
                typical_price_per_watt=0.25,
                demand_level="high",
                technical_requirements={
                    "min_capacity_retention": 0.80,
                    "power_tolerance": 0.05,
                    "min_efficiency": 0.75
                },
                market_size_mw=8000.0,
                growth_rate=0.15
            ))

        # Market 4: Street lighting and public infrastructure
        if capacity_retention >= 0.75:
            markets.append(SecondLifeMarket(
                market_name="Street Lighting",
                application="LED street lights, parking lots",
                typical_price_per_watt=0.18,
                demand_level="medium",
                technical_requirements={
                    "min_capacity_retention": 0.75,
                    "certification": "UL/CE",
                    "warranty_years": 5.0
                },
                market_size_mw=3000.0,
                growth_rate=0.10
            ))

        # Market 5: Telecom backup power
        if capacity_retention >= 0.85:
            markets.append(SecondLifeMarket(
                market_name="Telecom Backup Power",
                application="Cell tower backup, remote communication",
                typical_price_per_watt=0.30,
                demand_level="medium",
                technical_requirements={
                    "min_capacity_retention": 0.85,
                    "reliability": 0.95,
                    "certification": "Telecom standards"
                },
                market_size_mw=1500.0,
                growth_rate=0.08
            ))

        # Market 6: Educational and demonstration
        if capacity_retention >= 0.65:
            markets.append(SecondLifeMarket(
                market_name="Educational/Demonstration",
                application="Schools, training facilities",
                typical_price_per_watt=0.12,
                demand_level="low",
                technical_requirements={
                    "min_capacity_retention": 0.65,
                    "visual_condition": "acceptable"
                },
                market_size_mw=500.0,
                growth_rate=0.05
            ))

        # Market 7: Emergency/disaster relief
        if capacity_retention >= 0.70:
            markets.append(SecondLifeMarket(
                market_name="Emergency Relief",
                application="Disaster relief, emergency power",
                typical_price_per_watt=0.16,
                demand_level="medium",
                technical_requirements={
                    "min_capacity_retention": 0.70,
                    "portability": "high",
                    "durability": "high"
                },
                market_size_mw=1000.0,
                growth_rate=0.06
            ))

        # Sort markets by price per watt (revenue potential)
        markets.sort(key=lambda m: m.typical_price_per_watt, reverse=True)

        return markets

    @staticmethod
    def _calculate_degradation_rate(
        performance_data: List[float],
        age_years: float
    ) -> float:
        """
        Calculate degradation rate from historical performance data.

        Args:
            performance_data: List of performance values over time
            age_years: Total age in years

        Returns:
            Annual degradation rate as percentage
        """
        if len(performance_data) < 2:
            return 0.5  # Default degradation rate

        # Fit linear regression to performance data
        x = np.linspace(0, age_years, len(performance_data))
        y = np.array(performance_data)

        # Calculate slope (degradation rate)
        if len(x) > 1:
            coefficients = np.polyfit(x, y, 1)
            slope = coefficients[0]
            # Convert to annual percentage degradation
            degradation_rate = -slope * 100 / y[0] if y[0] != 0 else 0.5
            return max(0, degradation_rate)
        return 0.5

    def batch_assessment(
        self,
        modules: List[ModuleTestResults],
        ages_years: List[float]
    ) -> Dict:
        """
        Perform batch assessment of multiple modules.

        Args:
            modules: List of module test results
            ages_years: List of module ages

        Returns:
            Dictionary with aggregated assessment results
        """
        if len(modules) != len(ages_years):
            raise ValueError("modules and ages_years must have same length")

        assessments = []
        for module, age in zip(modules, ages_years):
            assessment = self.module_testing(module, age)
            assessments.append(assessment)

        # Aggregate statistics
        eligible_count = sum(1 for a in assessments if a.is_eligible)
        total_count = len(assessments)
        eligible_rate = eligible_count / total_count if total_count > 0 else 0

        avg_confidence = np.mean([a.confidence_score for a in assessments])
        avg_remaining_life = np.mean([a.estimated_remaining_life_years for a in assessments])

        return {
            "total_modules": total_count,
            "eligible_modules": eligible_count,
            "eligibility_rate": eligible_rate,
            "average_confidence": avg_confidence,
            "average_remaining_life_years": avg_remaining_life,
            "individual_assessments": [a.model_dump() for a in assessments]
        }
