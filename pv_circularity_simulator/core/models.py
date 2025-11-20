"""Core Pydantic models for PV module assessment and grading."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

from pv_circularity_simulator.core.enums import (
    ModuleCondition,
    PerformanceLevel,
    ReusePotential,
    DegradationType,
    MarketSegment,
)


class ModuleData(BaseModel):
    """Input data for a PV module to be assessed.

    Attributes:
        module_id: Unique identifier for the module
        manufacturer: Module manufacturer name
        model: Module model number
        nameplate_power_w: Original nameplate power rating in watts
        manufacture_date: Date of manufacture
        installation_date: Date of installation (if known)
        age_years: Age of module in years
        visual_defects: List of observed visual defects
        degradation_types: List of identified degradation types
        location: Geographic location or installation site
        environmental_conditions: Description of environmental exposure
    """
    module_id: str = Field(..., description="Unique module identifier")
    manufacturer: str = Field(..., description="Module manufacturer")
    model: str = Field(..., description="Module model number")
    nameplate_power_w: float = Field(..., gt=0, description="Nameplate power in watts")
    manufacture_date: Optional[datetime] = Field(None, description="Manufacture date")
    installation_date: Optional[datetime] = Field(None, description="Installation date")
    age_years: float = Field(..., ge=0, description="Module age in years")
    visual_defects: List[str] = Field(default_factory=list, description="Visual defects")
    degradation_types: List[DegradationType] = Field(
        default_factory=list, description="Degradation types"
    )
    location: Optional[str] = Field(None, description="Module location")
    environmental_conditions: Optional[str] = Field(None, description="Environmental exposure")

    @field_validator("age_years")
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Validate module age is reasonable (0-50 years)."""
        if v > 50:
            raise ValueError("Module age cannot exceed 50 years")
        return v


class PerformanceMetrics(BaseModel):
    """Performance test results for a PV module.

    Attributes:
        measured_power_w: Measured power output in watts
        open_circuit_voltage_v: Open circuit voltage
        short_circuit_current_a: Short circuit current
        max_power_voltage_v: Voltage at maximum power point
        max_power_current_a: Current at maximum power point
        fill_factor: Fill factor (0-1)
        efficiency_percent: Module efficiency percentage
        temperature_coefficient: Temperature coefficient (%/°C)
        series_resistance_ohm: Series resistance
        shunt_resistance_ohm: Shunt resistance
        test_conditions: Description of test conditions (e.g., STC, NOCT)
        test_date: Date of performance testing
    """
    measured_power_w: float = Field(..., gt=0, description="Measured power output")
    open_circuit_voltage_v: float = Field(..., gt=0, description="Open circuit voltage")
    short_circuit_current_a: float = Field(..., gt=0, description="Short circuit current")
    max_power_voltage_v: float = Field(..., gt=0, description="MPP voltage")
    max_power_current_a: float = Field(..., gt=0, description="MPP current")
    fill_factor: float = Field(..., ge=0, le=1, description="Fill factor")
    efficiency_percent: float = Field(..., gt=0, le=100, description="Efficiency")
    temperature_coefficient: Optional[float] = Field(None, description="Temperature coefficient")
    series_resistance_ohm: Optional[float] = Field(None, ge=0, description="Series resistance")
    shunt_resistance_ohm: Optional[float] = Field(None, ge=0, description="Shunt resistance")
    test_conditions: str = Field(default="STC", description="Test conditions")
    test_date: datetime = Field(default_factory=datetime.now, description="Test date")

    @field_validator("fill_factor")
    @classmethod
    def validate_fill_factor(cls, v: float) -> float:
        """Validate fill factor is within reasonable range."""
        if v < 0.5:
            raise ValueError("Fill factor below 0.5 indicates measurement issues")
        return v


class ConditionAssessment(BaseModel):
    """Physical condition assessment results.

    Attributes:
        overall_condition: Overall condition grade
        condition_score: Numerical condition score (0-100)
        visual_inspection_pass: Whether visual inspection passed
        structural_integrity_score: Structural integrity score (0-100)
        electrical_safety_pass: Whether electrical safety checks passed
        defect_severity_scores: Severity scores for each defect type
        inspection_notes: Additional notes from inspection
        inspector_id: ID of the inspector
        inspection_date: Date of inspection
    """
    overall_condition: ModuleCondition = Field(..., description="Overall condition grade")
    condition_score: float = Field(..., ge=0, le=100, description="Condition score")
    visual_inspection_pass: bool = Field(..., description="Visual inspection result")
    structural_integrity_score: float = Field(..., ge=0, le=100, description="Structural score")
    electrical_safety_pass: bool = Field(..., description="Electrical safety result")
    defect_severity_scores: Dict[str, float] = Field(
        default_factory=dict, description="Defect severity scores"
    )
    inspection_notes: Optional[str] = Field(None, description="Inspector notes")
    inspector_id: Optional[str] = Field(None, description="Inspector ID")
    inspection_date: datetime = Field(default_factory=datetime.now, description="Inspection date")


class ReuseAssessmentResult(BaseModel):
    """Complete reuse assessment results for a module.

    Attributes:
        module_id: Module identifier
        reuse_potential: Overall reuse potential classification
        reusability_score: Numerical reusability score (0-100)
        condition_assessment: Physical condition assessment
        performance_metrics: Performance test results
        performance_level: Performance classification
        remaining_lifetime_years: Estimated remaining useful lifetime
        recommended_applications: List of recommended use cases
        market_value_usd: Estimated secondary market value in USD
        market_segment: Target market segment
        confidence_level: Confidence in assessment (0-1)
        limiting_factors: Factors limiting reuse potential
        assessment_date: Date of assessment
        assessor_notes: Additional notes from assessor
    """
    module_id: str = Field(..., description="Module identifier")
    reuse_potential: ReusePotential = Field(..., description="Reuse potential classification")
    reusability_score: float = Field(..., ge=0, le=100, description="Reusability score")
    condition_assessment: ConditionAssessment = Field(..., description="Condition assessment")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    performance_level: PerformanceLevel = Field(..., description="Performance classification")
    remaining_lifetime_years: float = Field(..., ge=0, description="Remaining lifetime")
    recommended_applications: List[str] = Field(
        default_factory=list, description="Recommended applications"
    )
    market_value_usd: float = Field(..., ge=0, description="Market value in USD")
    market_segment: MarketSegment = Field(..., description="Target market segment")
    confidence_level: float = Field(..., ge=0, le=1, description="Assessment confidence")
    limiting_factors: List[str] = Field(default_factory=list, description="Limiting factors")
    assessment_date: datetime = Field(default_factory=datetime.now, description="Assessment date")
    assessor_notes: Optional[str] = Field(None, description="Assessor notes")


class MarketValuation(BaseModel):
    """Secondary market valuation details.

    Attributes:
        base_value_usd: Base value in USD
        condition_multiplier: Multiplier based on condition (0-1)
        performance_multiplier: Multiplier based on performance (0-1)
        age_multiplier: Multiplier based on age (0-1)
        market_demand_multiplier: Multiplier based on market demand (0-2)
        final_value_usd: Final estimated value in USD
        value_confidence: Confidence in valuation (0-1)
        comparable_sales: List of comparable sale prices
        market_segment: Target market segment
        valuation_date: Date of valuation
    """
    base_value_usd: float = Field(..., ge=0, description="Base value")
    condition_multiplier: float = Field(..., ge=0, le=1, description="Condition multiplier")
    performance_multiplier: float = Field(..., ge=0, le=1, description="Performance multiplier")
    age_multiplier: float = Field(..., ge=0, le=1, description="Age multiplier")
    market_demand_multiplier: float = Field(..., ge=0, le=2, description="Market multiplier")
    final_value_usd: float = Field(..., ge=0, description="Final value")
    value_confidence: float = Field(..., ge=0, le=1, description="Valuation confidence")
    comparable_sales: List[float] = Field(default_factory=list, description="Comparable sales")
    market_segment: MarketSegment = Field(..., description="Market segment")
    valuation_date: datetime = Field(default_factory=datetime.now, description="Valuation date")
