"""
Financial models for PV circularity simulator.

This module defines comprehensive Pydantic models for financial analysis
and circular economy modeling, including:
- Capital costs (equipment, installation)
- Operating costs (O&M, insurance)
- Financial analysis (NPV, IRR, LCOE, payback)
- End-of-life costs and revenues
- Circular economy metrics (3R: Recycle, Refurbish, Reuse)
- Material recovery data and environmental impact

All models include full validation for financial constraints and
production-ready error handling.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.core import NamedModel, UUIDModel


class Currency(str, Enum):
    """Enumeration of supported currencies."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"


class CapitalCost(NamedModel):
    """
    Capital expenditure (CAPEX) breakdown for PV systems.

    Represents all upfront costs required to install a PV system.

    Attributes:
        cell_cost_usd_per_wp: Cell cost in USD per watt-peak
        module_cost_usd_per_wp: Module cost in USD per watt-peak
        inverter_cost_usd: Inverter equipment cost in USD
        mounting_structure_cost_usd: Mounting/racking structure cost in USD
        electrical_bos_cost_usd: Electrical balance-of-system cost in USD
        installation_labor_cost_usd: Installation labor cost in USD
        permit_fees_usd: Permits and regulatory fees in USD
        grid_connection_cost_usd: Grid connection cost in USD
        monitoring_equipment_cost_usd: Monitoring system cost in USD
        design_engineering_cost_usd: Design and engineering cost in USD
        contingency_cost_usd: Contingency/buffer cost in USD
        total_capex_usd: Total capital expenditure in USD
        currency: Currency for all cost values
    """

    cell_cost_usd_per_wp: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Cell manufacturing cost in USD per watt-peak",
    )
    module_cost_usd_per_wp: float = Field(
        default=0.0,
        ge=0,
        le=5,
        description="Module cost in USD per watt-peak (typical: 0.15-0.40)",
    )
    inverter_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Inverter equipment cost in USD",
    )
    mounting_structure_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Mounting/racking structure cost in USD",
    )
    electrical_bos_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Electrical balance-of-system (wiring, breakers, etc.) cost in USD",
    )
    installation_labor_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Installation labor cost in USD",
    )
    permit_fees_usd: float = Field(
        default=0.0,
        ge=0,
        description="Permits, inspections, and regulatory fees in USD",
    )
    grid_connection_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Grid interconnection cost in USD",
    )
    monitoring_equipment_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Monitoring system and equipment cost in USD",
    )
    design_engineering_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Design, engineering, and project management cost in USD",
    )
    contingency_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Contingency/buffer for unexpected costs in USD",
    )
    total_capex_usd: float = Field(
        ...,
        gt=0,
        description="Total capital expenditure in USD",
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency for all cost values",
    )

    @model_validator(mode="after")
    def validate_total_cost(self) -> "CapitalCost":
        """Validate that total CAPEX matches sum of components."""
        component_sum = (
            self.inverter_cost_usd
            + self.mounting_structure_cost_usd
            + self.electrical_bos_cost_usd
            + self.installation_labor_cost_usd
            + self.permit_fees_usd
            + self.grid_connection_cost_usd
            + self.monitoring_equipment_cost_usd
            + self.design_engineering_cost_usd
            + self.contingency_cost_usd
        )

        # Module cost needs system capacity to calculate
        # So we allow some tolerance if not all components are specified
        if component_sum > 0:
            if abs(self.total_capex_usd - component_sum) > max(100, self.total_capex_usd * 0.1):
                import warnings
                warnings.warn(
                    f"Total CAPEX ({self.total_capex_usd:,.0f} USD) differs from "
                    f"sum of non-module components ({component_sum:,.0f} USD). "
                    f"This may be correct if module costs are included separately."
                )

        return self

    def calculate_cost_per_watt(self, system_capacity_w: float) -> float:
        """
        Calculate total installed cost per watt.

        Args:
            system_capacity_w: System capacity in watts (DC)

        Returns:
            float: Cost per watt in USD/W

        Raises:
            ValueError: If capacity is not positive
        """
        if system_capacity_w <= 0:
            raise ValueError("System capacity must be positive")
        return self.total_capex_usd / system_capacity_w


class OperatingCost(NamedModel):
    """
    Operating expenditure (OPEX) for PV systems.

    Represents annual recurring costs for system operation and maintenance.

    Attributes:
        maintenance_annual_usd: Annual maintenance cost in USD
        cleaning_annual_usd: Annual cleaning cost in USD
        insurance_annual_usd: Annual insurance premium in USD
        monitoring_annual_usd: Annual monitoring service cost in USD
        inverter_replacement_cost_usd: Inverter replacement cost (amortized)
        land_lease_annual_usd: Annual land lease cost in USD (if applicable)
        property_tax_annual_usd: Annual property tax in USD (if applicable)
        grid_connection_annual_usd: Annual grid connection fees in USD
        performance_penalty_annual_usd: Performance guarantee penalty in USD
        total_opex_annual_usd: Total annual operating expenditure in USD
        escalation_rate_percentage: Annual cost escalation rate in %
        currency: Currency for all cost values
    """

    maintenance_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual maintenance cost in USD (typical: $10-20/kW/year)",
    )
    cleaning_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual cleaning cost in USD",
    )
    insurance_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual insurance premium in USD",
    )
    monitoring_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual monitoring and data service cost in USD",
    )
    inverter_replacement_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Inverter replacement cost (amortized annually) in USD",
    )
    land_lease_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual land lease cost in USD (if applicable)",
    )
    property_tax_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual property tax in USD (if applicable)",
    )
    grid_connection_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual grid connection and service fees in USD",
    )
    performance_penalty_annual_usd: float = Field(
        default=0.0,
        ge=0,
        description="Average annual performance guarantee penalty in USD",
    )
    total_opex_annual_usd: float = Field(
        ...,
        ge=0,
        description="Total annual operating expenditure in USD",
    )
    escalation_rate_percentage: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Annual OPEX escalation rate in % (typical: 2-3%)",
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency for all cost values",
    )

    @model_validator(mode="after")
    def validate_total_opex(self) -> "OperatingCost":
        """Validate that total OPEX matches sum of components."""
        component_sum = (
            self.maintenance_annual_usd
            + self.cleaning_annual_usd
            + self.insurance_annual_usd
            + self.monitoring_annual_usd
            + self.inverter_replacement_cost_usd
            + self.land_lease_annual_usd
            + self.property_tax_annual_usd
            + self.grid_connection_annual_usd
            + self.performance_penalty_annual_usd
        )

        if abs(self.total_opex_annual_usd - component_sum) > max(10, self.total_opex_annual_usd * 0.01):
            import warnings
            warnings.warn(
                f"Total annual OPEX ({self.total_opex_annual_usd:,.0f} USD) differs from "
                f"sum of components ({component_sum:,.0f} USD)"
            )

        return self

    def calculate_opex_at_year(self, year: int) -> float:
        """
        Calculate operating cost at a specific year with escalation.

        Args:
            year: Year number (0 = first year)

        Returns:
            float: Operating cost in that year in USD

        Raises:
            ValueError: If year is negative
        """
        if year < 0:
            raise ValueError("Year cannot be negative")

        escalation_factor = (1.0 + self.escalation_rate_percentage / 100.0) ** year
        return self.total_opex_annual_usd * escalation_factor


class FinancialAnalysis(NamedModel):
    """
    Comprehensive financial analysis results for a PV system.

    Includes all major financial metrics and investment returns.

    Attributes:
        investment_cost_usd: Total initial investment in USD
        annual_revenue_usd: Average annual revenue in USD
        annual_savings_usd: Average annual energy cost savings in USD
        electricity_price_usd_kwh: Electricity price in USD per kWh
        electricity_price_escalation_percentage: Annual electricity price increase
        discount_rate_percentage: Discount rate for NPV calculation
        analysis_period_years: Analysis period in years
        npv_usd: Net Present Value in USD
        irr_percentage: Internal Rate of Return in %
        payback_period_years: Simple payback period in years
        discounted_payback_period_years: Discounted payback period in years
        lcoe_usd_kwh: Levelized Cost of Energy in USD per kWh
        benefit_cost_ratio: Benefit-cost ratio (BCR)
        profitability_index: Profitability index (PI)
        return_on_investment_percentage: Return on investment in %
    """

    investment_cost_usd: float = Field(
        ...,
        gt=0,
        description="Total initial investment (CAPEX) in USD",
    )
    annual_revenue_usd: float = Field(
        default=0.0,
        ge=0,
        description="Average annual revenue from energy sales in USD",
    )
    annual_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Average annual energy cost savings in USD",
    )
    electricity_price_usd_kwh: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Electricity price in USD per kWh (typical: 0.08-0.25)",
    )
    electricity_price_escalation_percentage: float = Field(
        default=3.0,
        ge=0,
        le=20,
        description="Annual electricity price escalation rate in %",
    )
    discount_rate_percentage: float = Field(
        default=5.0,
        gt=0,
        le=20,
        description="Discount rate for NPV calculation in %",
    )
    analysis_period_years: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Financial analysis period in years",
    )
    npv_usd: float = Field(
        ...,
        description="Net Present Value in USD (can be negative)",
    )
    irr_percentage: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Internal Rate of Return in % (None if undefined)",
    )
    payback_period_years: Optional[float] = Field(
        None,
        ge=0,
        description="Simple payback period in years (None if > analysis period)",
    )
    discounted_payback_period_years: Optional[float] = Field(
        None,
        ge=0,
        description="Discounted payback period in years (None if > analysis period)",
    )
    lcoe_usd_kwh: float = Field(
        ...,
        gt=0,
        le=10,
        description="Levelized Cost of Energy in USD per kWh",
    )
    benefit_cost_ratio: float = Field(
        ...,
        ge=0,
        description="Benefit-cost ratio (present value benefits / costs)",
    )
    profitability_index: float = Field(
        ...,
        ge=0,
        description="Profitability index (NPV / initial investment + 1)",
    )
    return_on_investment_percentage: float = Field(
        ...,
        description="Total return on investment in % (can be negative)",
    )

    @field_validator("lcoe_usd_kwh")
    @classmethod
    def validate_lcoe(cls, v: float) -> float:
        """Validate LCOE is within reasonable range."""
        if v > 0.5:
            import warnings
            warnings.warn(
                f"LCOE {v:.3f} USD/kWh is very high. "
                f"Typical utility-scale PV LCOE is 0.03-0.10 USD/kWh."
            )
        return v

    @model_validator(mode="after")
    def validate_financial_consistency(self) -> "FinancialAnalysis":
        """Validate consistency of financial metrics."""
        # NPV should be consistent with profitability index
        # PI = (NPV / Investment) + 1
        calculated_pi = (self.npv_usd / self.investment_cost_usd) + 1.0
        if abs(calculated_pi - self.profitability_index) > 0.01:
            import warnings
            warnings.warn(
                f"Profitability index ({self.profitability_index:.3f}) should equal "
                f"(NPV / Investment) + 1 ({calculated_pi:.3f})"
            )

        # BCR > 1 should mean NPV > 0 (for typical projects)
        if self.benefit_cost_ratio > 1.0 and self.npv_usd < 0:
            import warnings
            warnings.warn(
                "Benefit-cost ratio > 1 but NPV < 0. This may indicate inconsistent calculations."
            )

        return self


class EndOfLifeScenario(NamedModel):
    """
    End-of-life treatment scenario for PV modules/systems.

    Defines the percentage allocation to different circular economy pathways.

    Attributes:
        recycling_percentage: Percentage sent to recycling (0-100)
        refurbishment_percentage: Percentage sent for refurbishment (0-100)
        reuse_percentage: Percentage reused directly (0-100)
        landfill_percentage: Percentage sent to landfill (0-100)
        scenario_name: Name of this scenario (e.g., "best_case", "baseline")
    """

    recycling_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of materials sent to recycling (0-100%)",
    )
    refurbishment_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of modules sent for refurbishment (0-100%)",
    )
    reuse_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of modules/components reused directly (0-100%)",
    )
    landfill_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage sent to landfill (0-100%)",
    )
    scenario_name: str = Field(
        default="baseline",
        max_length=50,
        description="Name of this end-of-life scenario",
    )

    @model_validator(mode="after")
    def validate_percentages_sum(self) -> "EndOfLifeScenario":
        """Validate that all percentages sum to 100%."""
        total = (
            self.recycling_percentage
            + self.refurbishment_percentage
            + self.reuse_percentage
            + self.landfill_percentage
        )

        if abs(total - 100.0) > 0.1:
            raise ValueError(
                f"End-of-life percentages must sum to 100% (got {total:.1f}%)"
            )

        return self


class MaterialRecoveryData(NamedModel):
    """
    Material recovery data for recycling processes.

    Tracks the recovery of specific materials and their economic/environmental value.

    Attributes:
        material_type: Type of material (silicon, silver, aluminum, glass, etc.)
        total_mass_kg: Total mass of this material in the system in kg
        recovery_rate_percentage: Recovery efficiency in recycling (0-100%)
        recovered_mass_kg: Mass recovered through recycling in kg
        purity_percentage: Purity of recovered material (0-100%)
        recovery_cost_usd_per_kg: Cost to recover per kg in USD
        market_value_usd_per_kg: Market value of recovered material per kg in USD
        environmental_impact_avoided_kg_co2_eq: CO₂ emissions avoided per kg recovered
    """

    material_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Type of material (silicon, silver, aluminum, copper, glass, etc.)",
    )
    total_mass_kg: float = Field(
        ...,
        gt=0,
        description="Total mass of this material in the system in kg",
    )
    recovery_rate_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recovery efficiency in recycling process (0-100%)",
    )
    recovered_mass_kg: float = Field(
        ...,
        ge=0,
        description="Mass of material recovered through recycling in kg",
    )
    purity_percentage: float = Field(
        default=95.0,
        ge=0,
        le=100,
        description="Purity of recovered material (0-100%)",
    )
    recovery_cost_usd_per_kg: float = Field(
        default=0.0,
        ge=0,
        description="Cost to recover this material per kg in USD",
    )
    market_value_usd_per_kg: float = Field(
        default=0.0,
        ge=0,
        description="Market value of recovered material per kg in USD",
    )
    environmental_impact_avoided_kg_co2_eq: float = Field(
        default=0.0,
        ge=0,
        description="CO₂ equivalent emissions avoided per kg of recovered material",
    )

    @model_validator(mode="after")
    def validate_recovery_consistency(self) -> "MaterialRecoveryData":
        """Validate recovered mass is consistent with total and recovery rate."""
        expected_recovered = self.total_mass_kg * (self.recovery_rate_percentage / 100.0)
        if abs(self.recovered_mass_kg - expected_recovered) > 0.01:
            raise ValueError(
                f"Recovered mass ({self.recovered_mass_kg:.2f} kg) should equal "
                f"total mass × recovery rate ({expected_recovered:.2f} kg)"
            )
        return self

    def calculate_net_value(self) -> float:
        """
        Calculate net value (revenue - cost) from material recovery.

        Returns:
            float: Net value in USD (can be negative if costs exceed value)
        """
        revenue = self.recovered_mass_kg * self.market_value_usd_per_kg
        cost = self.recovered_mass_kg * self.recovery_cost_usd_per_kg
        return revenue - cost

    def calculate_total_environmental_benefit(self) -> float:
        """
        Calculate total environmental benefit from material recovery.

        Returns:
            float: Total CO₂ equivalent emissions avoided in kg
        """
        return self.recovered_mass_kg * self.environmental_impact_avoided_kg_co2_eq


class CircularityMetrics(NamedModel):
    """
    Circular economy metrics for PV systems.

    Comprehensive metrics tracking circularity performance across the lifecycle.

    Attributes:
        material_circularity_indicator: MCI score (0-1, higher is better)
        recycled_content_percentage: Percentage of recycled materials used (0-100%)
        recyclability_rate_percentage: Percentage of materials recyclable at EOL (0-100%)
        reuse_rate_percentage: Percentage of components reused (0-100%)
        refurbishment_rate_percentage: Percentage of modules refurbished (0-100%)
        total_waste_kg: Total waste generated in kg
        waste_diverted_from_landfill_kg: Waste diverted from landfill in kg
        circular_economy_value_usd: Total economic value from circularity in USD
        environmental_footprint_saved_kg_co2_eq: CO₂ emissions saved through circularity
        resource_efficiency_percentage: Resource efficiency score (0-100%)
    """

    material_circularity_indicator: float = Field(
        default=0.0,
        ge=0,
        le=1.0,
        description="Material Circularity Indicator (MCI) score (0-1, higher is better)",
    )
    recycled_content_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of recycled materials used in manufacturing (0-100%)",
    )
    recyclability_rate_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of materials that are recyclable at end-of-life (0-100%)",
    )
    reuse_rate_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of components that can be reused (0-100%)",
    )
    refurbishment_rate_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percentage of modules that can be refurbished (0-100%)",
    )
    total_waste_kg: float = Field(
        default=0.0,
        ge=0,
        description="Total waste generated over lifecycle in kg",
    )
    waste_diverted_from_landfill_kg: float = Field(
        default=0.0,
        ge=0,
        description="Waste diverted from landfill through recycling/reuse in kg",
    )
    circular_economy_value_usd: float = Field(
        default=0.0,
        description="Total economic value recovered through circular economy in USD",
    )
    environmental_footprint_saved_kg_co2_eq: float = Field(
        default=0.0,
        ge=0,
        description="CO₂ equivalent emissions saved through circular practices in kg",
    )
    resource_efficiency_percentage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Overall resource efficiency score (0-100%)",
    )

    @field_validator("waste_diverted_from_landfill_kg")
    @classmethod
    def validate_waste_diversion(cls, v: float, info) -> float:
        """Validate waste diversion doesn't exceed total waste."""
        # Note: In Pydantic v2, we need to access other fields via info.data
        if "total_waste_kg" in info.data and v > info.data["total_waste_kg"]:
            raise ValueError(
                f"Waste diverted ({v:.2f} kg) cannot exceed total waste "
                f"({info.data['total_waste_kg']:.2f} kg)"
            )
        return v

    def calculate_waste_diversion_rate(self) -> float:
        """
        Calculate waste diversion rate.

        Returns:
            float: Waste diversion rate as percentage (0-100%)
        """
        if self.total_waste_kg == 0:
            return 0.0
        return (self.waste_diverted_from_landfill_kg / self.total_waste_kg) * 100.0


class EndOfLifeCost(NamedModel):
    """
    End-of-life costs and revenues for PV systems.

    Attributes:
        decommissioning_cost_usd: Cost to decommission system in USD
        transportation_cost_usd: Transportation cost to recycling facility in USD
        recycling_cost_usd: Recycling processing cost in USD
        disposal_cost_usd: Landfill disposal cost in USD
        total_eol_cost_usd: Total end-of-life cost in USD
        material_recovery_revenue_usd: Revenue from recovered materials in USD
        refurbished_module_revenue_usd: Revenue from selling refurbished modules in USD
        reused_component_revenue_usd: Revenue from selling reused components in USD
        total_eol_revenue_usd: Total end-of-life revenue in USD
        net_eol_cost_usd: Net end-of-life cost (cost - revenue) in USD
        residual_value_usd: Residual/salvage value in USD
    """

    decommissioning_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Cost to decommission and remove system in USD",
    )
    transportation_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Transportation cost to recycling facility in USD",
    )
    recycling_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Recycling processing cost in USD",
    )
    disposal_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Landfill disposal cost (for non-recycled waste) in USD",
    )
    total_eol_cost_usd: float = Field(
        ...,
        ge=0,
        description="Total end-of-life cost in USD",
    )
    material_recovery_revenue_usd: float = Field(
        default=0.0,
        ge=0,
        description="Revenue from selling recovered materials in USD",
    )
    refurbished_module_revenue_usd: float = Field(
        default=0.0,
        ge=0,
        description="Revenue from selling refurbished modules in USD",
    )
    reused_component_revenue_usd: float = Field(
        default=0.0,
        ge=0,
        description="Revenue from selling reused components in USD",
    )
    total_eol_revenue_usd: float = Field(
        ...,
        ge=0,
        description="Total end-of-life revenue in USD",
    )
    net_eol_cost_usd: float = Field(
        ...,
        description="Net end-of-life cost (cost - revenue) in USD (can be negative)",
    )
    residual_value_usd: float = Field(
        default=0.0,
        ge=0,
        description="Residual/salvage value of system in USD",
    )

    @model_validator(mode="after")
    def validate_eol_consistency(self) -> "EndOfLifeCost":
        """Validate EOL cost calculations are consistent."""
        # Validate total cost
        calculated_total_cost = (
            self.decommissioning_cost_usd
            + self.transportation_cost_usd
            + self.recycling_cost_usd
            + self.disposal_cost_usd
        )
        if abs(self.total_eol_cost_usd - calculated_total_cost) > 1.0:
            import warnings
            warnings.warn(
                f"Total EOL cost ({self.total_eol_cost_usd:,.0f} USD) differs from "
                f"sum of components ({calculated_total_cost:,.0f} USD)"
            )

        # Validate total revenue
        calculated_total_revenue = (
            self.material_recovery_revenue_usd
            + self.refurbished_module_revenue_usd
            + self.reused_component_revenue_usd
        )
        if abs(self.total_eol_revenue_usd - calculated_total_revenue) > 1.0:
            import warnings
            warnings.warn(
                f"Total EOL revenue ({self.total_eol_revenue_usd:,.0f} USD) differs from "
                f"sum of components ({calculated_total_revenue:,.0f} USD)"
            )

        # Validate net cost
        calculated_net_cost = self.total_eol_cost_usd - self.total_eol_revenue_usd
        if abs(self.net_eol_cost_usd - calculated_net_cost) > 1.0:
            import warnings
            warnings.warn(
                f"Net EOL cost ({self.net_eol_cost_usd:,.0f} USD) should equal "
                f"total cost - total revenue ({calculated_net_cost:,.0f} USD)"
            )

        return self


class FinancialModel(UUIDModel):
    """
    Comprehensive financial model for PV systems including circular economy.

    Combines all financial aspects: capital costs, operating costs, revenues,
    financial analysis, and circular economy metrics.

    Attributes:
        name: Human-readable name for this financial model
        system_id: Reference to the PV system
        capex: Capital expenditure breakdown
        opex: Operating expenditure breakdown
        financial_analysis: Comprehensive financial analysis results
        eol_scenario: End-of-life treatment scenario
        eol_costs: End-of-life costs and revenues
        material_recovery: List of material recovery data
        circularity_metrics: Circular economy performance metrics
        currency: Primary currency for this model
        analysis_date: Date of this financial analysis
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for this financial model",
    )
    system_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Reference ID to the PV system",
    )
    capex: CapitalCost = Field(
        ...,
        description="Capital expenditure breakdown",
    )
    opex: OperatingCost = Field(
        ...,
        description="Operating expenditure breakdown",
    )
    financial_analysis: FinancialAnalysis = Field(
        ...,
        description="Comprehensive financial analysis results",
    )
    eol_scenario: EndOfLifeScenario = Field(
        ...,
        description="End-of-life treatment scenario (3R: Recycle, Refurbish, Reuse)",
    )
    eol_costs: EndOfLifeCost = Field(
        ...,
        description="End-of-life costs and revenues",
    )
    material_recovery: List[MaterialRecoveryData] = Field(
        default_factory=list,
        description="Material recovery data for each material type",
    )
    circularity_metrics: CircularityMetrics = Field(
        ...,
        description="Circular economy performance metrics",
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Primary currency for this financial model",
    )
    analysis_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of this financial analysis in YYYY-MM-DD format",
    )

    @model_validator(mode="after")
    def validate_financial_model_consistency(self) -> "FinancialModel":
        """Validate consistency across all financial components."""
        # CAPEX should match investment cost in financial analysis
        if abs(self.capex.total_capex_usd - self.financial_analysis.investment_cost_usd) > 100:
            import warnings
            warnings.warn(
                f"CAPEX ({self.capex.total_capex_usd:,.0f} USD) differs from "
                f"investment cost in analysis ({self.financial_analysis.investment_cost_usd:,.0f} USD)"
            )

        # Material recovery revenue should align with EOL revenue
        total_material_revenue = sum(
            mr.calculate_net_value() for mr in self.material_recovery
        )
        if abs(total_material_revenue - self.eol_costs.material_recovery_revenue_usd) > 100:
            import warnings
            warnings.warn(
                f"Sum of material recovery values ({total_material_revenue:,.0f} USD) differs from "
                f"EOL material recovery revenue ({self.eol_costs.material_recovery_revenue_usd:,.0f} USD)"
            )

        return self

    def calculate_total_lifecycle_cost(self, lifetime_years: int = 25) -> float:
        """
        Calculate total lifecycle cost (LCC) over system lifetime.

        Args:
            lifetime_years: System lifetime in years

        Returns:
            float: Total lifecycle cost in USD (present value)

        Raises:
            ValueError: If lifetime is not positive
        """
        if lifetime_years <= 0:
            raise ValueError("Lifetime must be positive")

        discount_rate = self.financial_analysis.discount_rate_percentage / 100.0

        # Initial investment
        lcc = self.capex.total_capex_usd

        # Operating costs (discounted)
        for year in range(1, lifetime_years + 1):
            opex_year = self.opex.calculate_opex_at_year(year - 1)
            discount_factor = 1.0 / ((1.0 + discount_rate) ** year)
            lcc += opex_year * discount_factor

        # End-of-life cost (discounted)
        eol_discount_factor = 1.0 / ((1.0 + discount_rate) ** lifetime_years)
        lcc += self.eol_costs.net_eol_cost_usd * eol_discount_factor

        return lcc

    def calculate_total_circular_economy_benefit(self) -> Dict[str, float]:
        """
        Calculate total circular economy benefits (economic and environmental).

        Returns:
            Dict with 'economic_usd' and 'environmental_kg_co2_eq' keys
        """
        economic_benefit = self.circularity_metrics.circular_economy_value_usd
        environmental_benefit = self.circularity_metrics.environmental_footprint_saved_kg_co2_eq

        return {
            "economic_usd": economic_benefit,
            "environmental_kg_co2_eq": environmental_benefit,
        }
