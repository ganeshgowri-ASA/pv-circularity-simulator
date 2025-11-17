"""Pydantic models for PV system components, analysis, and economics."""

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from pv_simulator.core.enums import (
    ClimateZone,
    ComponentType,
    HealthStatus,
    ModuleTechnology,
    RepowerStrategy,
)


class Location(BaseModel):
    """Geographic location and environmental data for a PV system.

    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        climate_zone: Climate classification affecting performance
        avg_annual_irradiance: Average annual solar irradiance (kWh/m²/year)
        avg_temperature: Average annual temperature (°C)
        elevation: Elevation above sea level (meters)
    """

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    climate_zone: ClimateZone = Field(..., description="Climate zone classification")
    avg_annual_irradiance: float = Field(
        ..., gt=0, description="Average annual solar irradiance (kWh/m²/year)"
    )
    avg_temperature: float = Field(..., description="Average annual temperature (°C)")
    elevation: Optional[float] = Field(default=0, description="Elevation above sea level (m)")


class PVModule(BaseModel):
    """PV module specifications and performance characteristics.

    Attributes:
        technology: Module technology type
        rated_power: Nameplate power rating (W)
        efficiency: Module efficiency (0-1 fraction)
        area: Module area (m²)
        degradation_rate: Annual degradation rate (fraction/year)
        temperature_coefficient: Power temperature coefficient (%/°C)
        warranty_years: Performance warranty period (years)
        cost_per_watt: Module cost ($/W)
    """

    technology: ModuleTechnology = Field(..., description="Module technology type")
    rated_power: float = Field(..., gt=0, description="Nameplate power rating (W)")
    efficiency: float = Field(..., gt=0, le=1, description="Module efficiency (fraction)")
    area: float = Field(..., gt=0, description="Module area (m²)")
    degradation_rate: float = Field(
        default=0.005, ge=0, le=0.05, description="Annual degradation rate (fraction/year)"
    )
    temperature_coefficient: float = Field(
        default=-0.4, description="Power temperature coefficient (%/°C)"
    )
    warranty_years: int = Field(default=25, ge=0, description="Performance warranty (years)")
    cost_per_watt: float = Field(..., gt=0, description="Module cost ($/W)")


class ComponentHealth(BaseModel):
    """Health and performance status of a system component.

    Attributes:
        component_type: Type of component
        status: Current health status
        performance_ratio: Current performance vs. rated (0-1 fraction)
        age_years: Age of component (years)
        expected_lifetime: Expected lifetime (years)
        failure_probability: Probability of failure in next year (0-1)
        maintenance_cost_annual: Annual maintenance cost ($)
        replacement_cost: Cost to replace component ($)
    """

    component_type: ComponentType = Field(..., description="Component type")
    status: HealthStatus = Field(..., description="Current health status")
    performance_ratio: float = Field(
        ..., ge=0, le=1, description="Current performance vs. rated (fraction)"
    )
    age_years: float = Field(..., ge=0, description="Component age (years)")
    expected_lifetime: float = Field(..., gt=0, description="Expected lifetime (years)")
    failure_probability: float = Field(
        default=0.0, ge=0, le=1, description="Failure probability next year"
    )
    maintenance_cost_annual: float = Field(
        default=0.0, ge=0, description="Annual maintenance cost ($)"
    )
    replacement_cost: float = Field(..., gt=0, description="Replacement cost ($)")


class PVSystem(BaseModel):
    """Complete PV system specification and current state.

    Attributes:
        system_id: Unique system identifier
        installation_date: Date of original installation
        location: Geographic and environmental location data
        module: Module specifications
        num_modules: Number of modules in the system
        dc_capacity: DC nameplate capacity (kW)
        ac_capacity: AC capacity after inverter (kW)
        inverter_efficiency: Inverter efficiency (0-1 fraction)
        system_losses: Total system losses (0-1 fraction)
        component_health: Health status of all components
        current_performance_ratio: Overall system performance ratio
        avg_annual_production: Average annual energy production (kWh/year)
    """

    system_id: str = Field(..., description="Unique system identifier")
    installation_date: date = Field(..., description="Installation date")
    location: Location = Field(..., description="System location")
    module: PVModule = Field(..., description="Module specifications")
    num_modules: int = Field(..., gt=0, description="Number of modules")
    dc_capacity: float = Field(..., gt=0, description="DC nameplate capacity (kW)")
    ac_capacity: float = Field(..., gt=0, description="AC capacity (kW)")
    inverter_efficiency: float = Field(
        default=0.96, gt=0, le=1, description="Inverter efficiency (fraction)"
    )
    system_losses: float = Field(
        default=0.14, ge=0, lt=1, description="Total system losses (fraction)"
    )
    component_health: List[ComponentHealth] = Field(
        default_factory=list, description="Component health status"
    )
    current_performance_ratio: float = Field(
        default=0.75, ge=0, le=1, description="Overall performance ratio"
    )
    avg_annual_production: float = Field(
        ..., ge=0, description="Average annual production (kWh/year)"
    )

    @field_validator("ac_capacity")
    @classmethod
    def validate_ac_capacity(cls, v: float, info) -> float:
        """Validate AC capacity is less than or equal to DC capacity."""
        if "dc_capacity" in info.data and v > info.data["dc_capacity"]:
            raise ValueError("AC capacity cannot exceed DC capacity")
        return v


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for repower project.

    Attributes:
        module_costs: Total module costs ($)
        inverter_costs: Inverter replacement/upgrade costs ($)
        bos_costs: Balance of system costs ($)
        labor_costs: Installation labor costs ($)
        permitting_costs: Permitting and inspection costs ($)
        engineering_costs: Engineering and design costs ($)
        decommissioning_costs: Old system removal costs ($)
        contingency: Contingency reserve (%)
        total_capex: Total capital expenditure ($)
    """

    module_costs: float = Field(default=0.0, ge=0, description="Module costs ($)")
    inverter_costs: float = Field(default=0.0, ge=0, description="Inverter costs ($)")
    bos_costs: float = Field(default=0.0, ge=0, description="Balance of system costs ($)")
    labor_costs: float = Field(default=0.0, ge=0, description="Labor costs ($)")
    permitting_costs: float = Field(default=0.0, ge=0, description="Permitting costs ($)")
    engineering_costs: float = Field(default=0.0, ge=0, description="Engineering costs ($)")
    decommissioning_costs: float = Field(
        default=0.0, ge=0, description="Decommissioning costs ($)"
    )
    contingency: float = Field(
        default=0.10, ge=0, le=0.5, description="Contingency reserve (fraction)"
    )
    total_capex: float = Field(default=0.0, ge=0, description="Total CAPEX ($)")

    def calculate_total(self) -> float:
        """Calculate total CAPEX including contingency.

        Returns:
            Total capital expenditure with contingency
        """
        base_cost = (
            self.module_costs
            + self.inverter_costs
            + self.bos_costs
            + self.labor_costs
            + self.permitting_costs
            + self.engineering_costs
            + self.decommissioning_costs
        )
        return base_cost * (1 + self.contingency)


class EconomicMetrics(BaseModel):
    """Economic performance metrics for repower analysis.

    Attributes:
        lcoe: Levelized cost of energy ($/kWh)
        npv: Net present value ($)
        irr: Internal rate of return (fraction)
        payback_period: Simple payback period (years)
        roi: Return on investment (fraction)
        benefit_cost_ratio: Benefit-cost ratio
        annual_energy_value: Annual energy production value ($/year)
        annual_opex: Annual operating expenses ($/year)
        discount_rate: Discount rate for NPV calculation (fraction)
        analysis_period: Analysis period (years)
    """

    lcoe: float = Field(..., ge=0, description="Levelized cost of energy ($/kWh)")
    npv: float = Field(..., description="Net present value ($)")
    irr: float = Field(..., description="Internal rate of return (fraction)")
    payback_period: float = Field(..., ge=0, description="Payback period (years)")
    roi: float = Field(..., description="Return on investment (fraction)")
    benefit_cost_ratio: float = Field(..., ge=0, description="Benefit-cost ratio")
    annual_energy_value: float = Field(
        ..., ge=0, description="Annual energy value ($/year)"
    )
    annual_opex: float = Field(..., ge=0, description="Annual OPEX ($/year)")
    discount_rate: float = Field(
        default=0.06, gt=0, lt=1, description="Discount rate (fraction)"
    )
    analysis_period: int = Field(default=25, gt=0, description="Analysis period (years)")


class RepowerScenario(BaseModel):
    """Complete repower scenario with technical and economic details.

    Attributes:
        scenario_id: Unique scenario identifier
        strategy: Repower strategy
        new_dc_capacity: New DC capacity after repower (kW)
        capacity_increase: Capacity increase vs. original (fraction)
        new_module: New module specifications
        num_new_modules: Number of new modules
        components_to_replace: List of components to replace
        cost_breakdown: Detailed cost breakdown
        economic_metrics: Economic performance metrics
        estimated_annual_production: Estimated production (kWh/year)
        performance_improvement: Performance improvement (fraction)
        technical_feasibility_score: Feasibility score (0-100)
        recommended: Whether this scenario is recommended
        notes: Additional notes and considerations
    """

    scenario_id: str = Field(..., description="Scenario identifier")
    strategy: RepowerStrategy = Field(..., description="Repower strategy")
    new_dc_capacity: float = Field(..., gt=0, description="New DC capacity (kW)")
    capacity_increase: float = Field(..., description="Capacity increase (fraction)")
    new_module: Optional[PVModule] = Field(default=None, description="New module specs")
    num_new_modules: int = Field(..., ge=0, description="Number of new modules")
    components_to_replace: List[ComponentType] = Field(
        default_factory=list, description="Components to replace"
    )
    cost_breakdown: CostBreakdown = Field(..., description="Cost breakdown")
    economic_metrics: Optional[EconomicMetrics] = Field(
        default=None, description="Economic metrics"
    )
    estimated_annual_production: float = Field(
        ..., ge=0, description="Estimated production (kWh/year)"
    )
    performance_improvement: float = Field(
        ..., description="Performance improvement (fraction)"
    )
    technical_feasibility_score: float = Field(
        ..., ge=0, le=100, description="Feasibility score (0-100)"
    )
    recommended: bool = Field(default=False, description="Recommended scenario flag")
    notes: List[str] = Field(default_factory=list, description="Additional notes")


class CapacityUpgradeAnalysis(BaseModel):
    """Results of capacity upgrade analysis.

    Attributes:
        current_capacity: Current DC capacity (kW)
        max_additional_capacity: Maximum additional capacity possible (kW)
        space_available: Available space for new modules (m²)
        structural_capacity_available: Available structural capacity (kg)
        electrical_capacity_available: Available electrical capacity (A)
        upgrade_scenarios: List of possible upgrade scenarios
        recommended_upgrade: Recommended upgrade capacity (kW)
        limiting_factor: Primary limiting factor for upgrades
    """

    current_capacity: float = Field(..., ge=0, description="Current DC capacity (kW)")
    max_additional_capacity: float = Field(
        ..., ge=0, description="Max additional capacity (kW)"
    )
    space_available: float = Field(..., ge=0, description="Available space (m²)")
    structural_capacity_available: float = Field(
        ..., ge=0, description="Available structural capacity (kg)"
    )
    electrical_capacity_available: float = Field(
        ..., ge=0, description="Available electrical capacity (A)"
    )
    upgrade_scenarios: List[Dict[str, float]] = Field(
        default_factory=list, description="Upgrade scenarios"
    )
    recommended_upgrade: float = Field(..., ge=0, description="Recommended upgrade (kW)")
    limiting_factor: str = Field(..., description="Primary limiting factor")


class ComponentReplacementPlan(BaseModel):
    """Detailed component replacement planning.

    Attributes:
        immediate_replacements: Components requiring immediate replacement
        short_term_replacements: Components to replace within 1 year
        medium_term_replacements: Components to replace within 1-3 years
        long_term_replacements: Components to replace within 3-5 years
        total_replacement_cost: Total replacement cost ($)
        priority_order: Prioritized replacement order
        risk_mitigation_plan: Risk mitigation strategies
    """

    immediate_replacements: List[ComponentHealth] = Field(
        default_factory=list, description="Immediate replacements"
    )
    short_term_replacements: List[ComponentHealth] = Field(
        default_factory=list, description="Short-term replacements (1 year)"
    )
    medium_term_replacements: List[ComponentHealth] = Field(
        default_factory=list, description="Medium-term replacements (1-3 years)"
    )
    long_term_replacements: List[ComponentHealth] = Field(
        default_factory=list, description="Long-term replacements (3-5 years)"
    )
    total_replacement_cost: float = Field(..., ge=0, description="Total cost ($)")
    priority_order: List[ComponentType] = Field(
        default_factory=list, description="Priority order"
    )
    risk_mitigation_plan: Dict[str, str] = Field(
        default_factory=dict, description="Risk mitigation strategies"
    )


class TechnicalFeasibilityResult(BaseModel):
    """Technical feasibility assessment results.

    Attributes:
        is_feasible: Overall feasibility determination
        feasibility_score: Overall score (0-100)
        structural_feasibility: Structural assessment score (0-100)
        electrical_feasibility: Electrical assessment score (0-100)
        spatial_feasibility: Space availability score (0-100)
        regulatory_feasibility: Regulatory compliance score (0-100)
        integration_feasibility: System integration score (0-100)
        constraints: List of identified constraints
        risks: List of identified risks
        recommendations: List of recommendations
    """

    is_feasible: bool = Field(..., description="Overall feasibility")
    feasibility_score: float = Field(..., ge=0, le=100, description="Overall score")
    structural_feasibility: float = Field(
        ..., ge=0, le=100, description="Structural score"
    )
    electrical_feasibility: float = Field(
        ..., ge=0, le=100, description="Electrical score"
    )
    spatial_feasibility: float = Field(..., ge=0, le=100, description="Spatial score")
    regulatory_feasibility: float = Field(
        ..., ge=0, le=100, description="Regulatory score"
    )
    integration_feasibility: float = Field(
        ..., ge=0, le=100, description="Integration score"
    )
    constraints: List[str] = Field(default_factory=list, description="Constraints")
    risks: List[str] = Field(default_factory=list, description="Risks")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class EconomicViabilityResult(BaseModel):
    """Economic viability assessment results.

    Attributes:
        is_viable: Overall economic viability determination
        viability_score: Overall score (0-100)
        scenarios_analyzed: List of scenarios analyzed
        best_scenario: Best performing scenario
        sensitivity_analysis: Sensitivity analysis results
        break_even_scenarios: Break-even conditions
        financing_options: Potential financing structures
        incentives_available: Available incentives and rebates
    """

    is_viable: bool = Field(..., description="Overall economic viability")
    viability_score: float = Field(..., ge=0, le=100, description="Viability score")
    scenarios_analyzed: List[RepowerScenario] = Field(
        default_factory=list, description="Scenarios analyzed"
    )
    best_scenario: Optional[RepowerScenario] = Field(
        default=None, description="Best scenario"
    )
    sensitivity_analysis: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Sensitivity analysis"
    )
    break_even_scenarios: Dict[str, float] = Field(
        default_factory=dict, description="Break-even conditions"
    )
    financing_options: List[Dict[str, str]] = Field(
        default_factory=list, description="Financing options"
    )
    incentives_available: Dict[str, float] = Field(
        default_factory=dict, description="Available incentives ($)"
    )
