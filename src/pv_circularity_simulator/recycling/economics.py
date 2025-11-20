"""
Recycling Economics & Material Recovery Module

This module provides comprehensive economic modeling for photovoltaic panel recycling,
including material extraction costs, recovery rates, revenue calculations, and
environmental credits with Life Cycle Assessment (LCA) integration.

Author: PV Circularity Team
License: MIT
"""

from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict


class PVMaterialType(str, Enum):
    """
    Enumeration of recoverable materials from PV panels.

    Materials are categorized by their economic value and recovery complexity.
    """

    SILICON = "silicon"
    SILVER = "silver"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    GLASS = "glass"
    POLYMER = "polymer"
    LEAD = "lead"
    TIN = "tin"
    CADMIUM = "cadmium"
    TELLURIUM = "tellurium"
    INDIUM = "indium"
    GALLIUM = "gallium"
    SELENIUM = "selenium"


class RecyclingTechnology(str, Enum):
    """
    Types of recycling technologies for PV panels.

    Each technology has different costs, recovery rates, and environmental impacts.
    """

    MECHANICAL = "mechanical"  # Physical separation (crushing, grinding)
    THERMAL = "thermal"  # Pyrolysis, incineration
    CHEMICAL = "chemical"  # Acid/base leaching, solvent extraction
    HYBRID = "hybrid"  # Combination of methods
    ADVANCED = "advanced"  # Electrochemical, supercritical fluid


class MaterialComposition(BaseModel):
    """
    Material composition of a PV panel.

    Attributes:
        material_type: Type of material
        mass_kg: Mass of material in kilograms
        purity_percent: Initial purity percentage (0-100)
        market_value_per_kg: Current market value in USD/kg
    """

    material_type: PVMaterialType
    mass_kg: float = Field(gt=0, description="Mass in kilograms")
    purity_percent: float = Field(ge=0, le=100, description="Material purity percentage")
    market_value_per_kg: float = Field(ge=0, description="Market value in USD/kg")

    @field_validator('mass_kg', 'market_value_per_kg')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure values are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    model_config = ConfigDict(frozen=True)


class MaterialExtractionCosts(BaseModel):
    """
    Cost breakdown for extracting materials from PV panels.

    This model captures all cost components involved in the recycling process,
    from collection to final material purification.

    Attributes:
        technology: Recycling technology used
        collection_cost_per_panel: Cost to collect and transport panels (USD)
        preprocessing_cost_per_kg: Cost for dismantling and preprocessing (USD/kg)
        processing_cost_per_kg: Cost for material extraction processing (USD/kg)
        purification_cost_per_kg: Cost for material purification (USD/kg)
        labor_cost_per_hour: Labor cost in USD per hour
        processing_time_hours: Time required for processing in hours
        energy_cost_per_kwh: Energy cost in USD per kWh
        energy_consumption_kwh_per_kg: Energy consumption in kWh per kg
        disposal_cost_per_kg: Cost for disposing non-recoverable waste (USD/kg)
        equipment_depreciation_per_panel: Equipment depreciation per panel (USD)
        overhead_multiplier: Overhead cost multiplier (e.g., 1.2 = 20% overhead)
    """

    technology: RecyclingTechnology
    collection_cost_per_panel: float = Field(ge=0, description="Collection cost USD/panel")
    preprocessing_cost_per_kg: float = Field(ge=0, description="Preprocessing USD/kg")
    processing_cost_per_kg: float = Field(ge=0, description="Processing USD/kg")
    purification_cost_per_kg: float = Field(ge=0, description="Purification USD/kg")
    labor_cost_per_hour: float = Field(ge=0, description="Labor cost USD/hour")
    processing_time_hours: float = Field(gt=0, description="Processing time in hours")
    energy_cost_per_kwh: float = Field(ge=0, description="Energy cost USD/kWh")
    energy_consumption_kwh_per_kg: float = Field(ge=0, description="Energy kWh/kg")
    disposal_cost_per_kg: float = Field(ge=0, description="Disposal USD/kg")
    equipment_depreciation_per_panel: float = Field(ge=0, description="Depreciation USD/panel")
    overhead_multiplier: float = Field(ge=1.0, le=3.0, default=1.2,
                                      description="Overhead multiplier")

    @computed_field
    @property
    def total_fixed_cost(self) -> float:
        """Calculate total fixed costs per panel."""
        return (
            self.collection_cost_per_panel +
            self.equipment_depreciation_per_panel +
            (self.labor_cost_per_hour * self.processing_time_hours)
        )

    def calculate_total_cost(self, total_mass_kg: float,
                           non_recoverable_mass_kg: float) -> float:
        """
        Calculate total extraction cost for a given mass.

        Args:
            total_mass_kg: Total mass of panel in kg
            non_recoverable_mass_kg: Mass of non-recoverable waste in kg

        Returns:
            Total cost in USD
        """
        variable_costs = (
            (self.preprocessing_cost_per_kg * total_mass_kg) +
            (self.processing_cost_per_kg * total_mass_kg) +
            (self.purification_cost_per_kg * total_mass_kg) +
            (self.energy_cost_per_kwh * self.energy_consumption_kwh_per_kg * total_mass_kg) +
            (self.disposal_cost_per_kg * non_recoverable_mass_kg)
        )

        total_cost = (self.total_fixed_cost + variable_costs) * self.overhead_multiplier
        return total_cost

    model_config = ConfigDict(frozen=False)


class RecoveryRates(BaseModel):
    """
    Material recovery rates for different materials and technologies.

    Recovery rates represent the percentage of material that can be successfully
    recovered and meet quality standards for reuse.

    Attributes:
        technology: Recycling technology used
        material_recovery_rates: Dictionary mapping materials to recovery percentages
        technology_efficiency: Overall technology efficiency factor (0-1)
        quality_grade: Quality grade of recovered materials ('A', 'B', 'C')
    """

    technology: RecyclingTechnology
    material_recovery_rates: Dict[PVMaterialType, float] = Field(
        description="Recovery rates by material (0-100%)"
    )
    technology_efficiency: float = Field(ge=0, le=1, default=0.85,
                                        description="Overall efficiency factor")
    quality_grade: str = Field(default="B", pattern="^[ABC]$",
                              description="Quality grade A/B/C")

    @field_validator('material_recovery_rates')
    @classmethod
    def validate_recovery_rates(cls, v: Dict[PVMaterialType, float]) -> Dict[PVMaterialType, float]:
        """Validate that all recovery rates are between 0 and 100."""
        for material, rate in v.items():
            if not 0 <= rate <= 100:
                raise ValueError(f"Recovery rate for {material} must be between 0 and 100")
        return v

    def get_effective_recovery_rate(self, material: PVMaterialType) -> float:
        """
        Calculate effective recovery rate accounting for technology efficiency.

        Args:
            material: Material type

        Returns:
            Effective recovery rate as percentage (0-100)
        """
        base_rate = self.material_recovery_rates.get(material, 0.0)
        return base_rate * self.technology_efficiency

    def get_quality_multiplier(self) -> float:
        """
        Get quality multiplier based on grade.

        Returns:
            Quality multiplier for pricing (A=1.0, B=0.8, C=0.6)
        """
        multipliers = {"A": 1.0, "B": 0.8, "C": 0.6}
        return multipliers.get(self.quality_grade, 0.8)

    model_config = ConfigDict(frozen=False)


class RecyclingRevenue(BaseModel):
    """
    Revenue calculation from recovered materials.

    Attributes:
        recovered_materials: List of recovered material compositions
        market_price_adjustments: Price adjustments by material (multiplier)
        quality_discount: Discount factor for lower quality materials (0-1)
        transportation_cost_per_kg: Cost to transport materials to buyers (USD/kg)
        sales_commission_percent: Commission on sales (0-100%)
    """

    recovered_materials: List[MaterialComposition]
    market_price_adjustments: Dict[PVMaterialType, float] = Field(
        default_factory=dict,
        description="Price adjustment multipliers by material"
    )
    quality_discount: float = Field(ge=0, le=1, default=0.9,
                                   description="Quality discount factor")
    transportation_cost_per_kg: float = Field(ge=0, default=0.5,
                                             description="Transport cost USD/kg")
    sales_commission_percent: float = Field(ge=0, le=100, default=5.0,
                                           description="Sales commission %")

    @computed_field
    @property
    def total_recovered_mass_kg(self) -> float:
        """Total mass of recovered materials."""
        return sum(mat.mass_kg for mat in self.recovered_materials)

    @computed_field
    @property
    def gross_revenue(self) -> float:
        """Calculate gross revenue before costs and commissions."""
        revenue = 0.0
        for material in self.recovered_materials:
            price_adjustment = self.market_price_adjustments.get(
                material.material_type, 1.0
            )
            material_revenue = (
                material.mass_kg *
                material.market_value_per_kg *
                price_adjustment *
                self.quality_discount
            )
            revenue += material_revenue
        return revenue

    @computed_field
    @property
    def net_revenue(self) -> float:
        """Calculate net revenue after transportation and commissions."""
        transport_cost = self.total_recovered_mass_kg * self.transportation_cost_per_kg
        commission = self.gross_revenue * (self.sales_commission_percent / 100)
        return self.gross_revenue - transport_cost - commission

    def get_revenue_by_material(self) -> Dict[PVMaterialType, float]:
        """
        Break down revenue by material type.

        Returns:
            Dictionary mapping material types to revenue in USD
        """
        revenue_breakdown = {}
        for material in self.recovered_materials:
            price_adjustment = self.market_price_adjustments.get(
                material.material_type, 1.0
            )
            revenue = (
                material.mass_kg *
                material.market_value_per_kg *
                price_adjustment *
                self.quality_discount
            )
            revenue_breakdown[material.material_type] = revenue
        return revenue_breakdown

    model_config = ConfigDict(frozen=False)


class EnvironmentalCredits(BaseModel):
    """
    Environmental credits and benefits from recycling.

    This model integrates Life Cycle Assessment (LCA) metrics to quantify
    environmental benefits of recycling versus virgin material production.

    Attributes:
        carbon_credits_per_kg: Carbon credits earned per kg recycled (USD/kg)
        avoided_emissions_kg_co2: CO2 emissions avoided vs. virgin production (kg CO2)
        energy_savings_kwh: Energy saved vs. virgin production (kWh)
        water_savings_liters: Water saved vs. virgin production (liters)
        landfill_diversion_credits: Credits for diverting waste from landfill (USD)
        regulatory_compliance_value: Value of meeting regulatory requirements (USD)
        epr_credit_value: Extended Producer Responsibility credit value (USD)
        carbon_price_per_ton_co2: Carbon price in USD per ton CO2
        renewable_energy_certificates: Value of RECs if applicable (USD)
    """

    carbon_credits_per_kg: float = Field(ge=0, default=0.0,
                                         description="Carbon credits USD/kg")
    avoided_emissions_kg_co2: float = Field(ge=0, description="Avoided CO2 emissions kg")
    energy_savings_kwh: float = Field(ge=0, description="Energy saved kWh")
    water_savings_liters: float = Field(ge=0, default=0.0,
                                       description="Water saved liters")
    landfill_diversion_credits: float = Field(ge=0, default=0.0,
                                             description="Landfill credits USD")
    regulatory_compliance_value: float = Field(ge=0, default=0.0,
                                              description="Compliance value USD")
    epr_credit_value: float = Field(ge=0, default=0.0,
                                   description="EPR credit USD")
    carbon_price_per_ton_co2: float = Field(ge=0, default=50.0,
                                           description="Carbon price USD/ton")
    renewable_energy_certificates: float = Field(ge=0, default=0.0,
                                                description="REC value USD")

    @computed_field
    @property
    def total_carbon_value(self) -> float:
        """Calculate total value from carbon emissions reduction."""
        return (self.avoided_emissions_kg_co2 / 1000) * self.carbon_price_per_ton_co2

    @computed_field
    @property
    def total_environmental_value(self) -> float:
        """Calculate total environmental value in USD."""
        return (
            self.total_carbon_value +
            self.landfill_diversion_credits +
            self.regulatory_compliance_value +
            self.epr_credit_value +
            self.renewable_energy_certificates
        )

    def get_lca_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive LCA metrics.

        Returns:
            Dictionary of LCA impact indicators
        """
        return {
            "global_warming_potential_kg_co2_eq": self.avoided_emissions_kg_co2,
            "primary_energy_demand_kwh": self.energy_savings_kwh,
            "water_consumption_liters": self.water_savings_liters,
            "carbon_value_usd": self.total_carbon_value,
            "total_environmental_value_usd": self.total_environmental_value,
        }

    model_config = ConfigDict(frozen=False)


class RecyclingEconomics:
    """
    Comprehensive recycling economics analysis for PV panels.

    This class integrates all aspects of recycling economics including costs,
    revenues, recovery rates, and environmental benefits to provide a complete
    economic assessment of PV panel recycling operations.

    Attributes:
        panel_composition: List of materials in the panel
        extraction_costs: Material extraction cost model
        recovery_rates_model: Material recovery rates model
        panel_mass_kg: Total mass of panel in kg

    Example:
        >>> # Define panel composition
        >>> composition = [
        ...     MaterialComposition(
        ...         material_type=PVMaterialType.SILICON,
        ...         mass_kg=5.0,
        ...         purity_percent=99.0,
        ...         market_value_per_kg=15.0
        ...     ),
        ...     MaterialComposition(
        ...         material_type=PVMaterialType.SILVER,
        ...         mass_kg=0.015,
        ...         purity_percent=95.0,
        ...         market_value_per_kg=600.0
        ...     ),
        ... ]
        >>>
        >>> # Define extraction costs
        >>> costs = MaterialExtractionCosts(
        ...     technology=RecyclingTechnology.HYBRID,
        ...     collection_cost_per_panel=5.0,
        ...     preprocessing_cost_per_kg=2.0,
        ...     processing_cost_per_kg=3.0,
        ...     purification_cost_per_kg=1.5,
        ...     labor_cost_per_hour=25.0,
        ...     processing_time_hours=0.5,
        ...     energy_cost_per_kwh=0.12,
        ...     energy_consumption_kwh_per_kg=2.0,
        ...     disposal_cost_per_kg=0.5,
        ...     equipment_depreciation_per_panel=2.0,
        ... )
        >>>
        >>> # Define recovery rates
        >>> recovery = RecoveryRates(
        ...     technology=RecyclingTechnology.HYBRID,
        ...     material_recovery_rates={
        ...         PVMaterialType.SILICON: 95.0,
        ...         PVMaterialType.SILVER: 90.0,
        ...         PVMaterialType.GLASS: 98.0,
        ...         PVMaterialType.ALUMINUM: 97.0,
        ...     },
        ...     technology_efficiency=0.85,
        ...     quality_grade="B"
        ... )
        >>>
        >>> # Create economics model
        >>> economics = RecyclingEconomics(
        ...     panel_composition=composition,
        ...     extraction_costs=costs,
        ...     recovery_rates_model=recovery,
        ...     panel_mass_kg=20.0
        ... )
        >>>
        >>> # Analyze economics
        >>> total_costs = economics.material_extraction_costs()
        >>> rates = economics.recovery_rates()
        >>> revenue = economics.recycling_revenue_calculation()
        >>> credits = economics.environmental_credits()
    """

    def __init__(
        self,
        panel_composition: List[MaterialComposition],
        extraction_costs: MaterialExtractionCosts,
        recovery_rates_model: RecoveryRates,
        panel_mass_kg: float,
    ):
        """
        Initialize RecyclingEconomics model.

        Args:
            panel_composition: List of materials in the panel
            extraction_costs: Material extraction cost model
            recovery_rates_model: Material recovery rates model
            panel_mass_kg: Total mass of panel in kg

        Raises:
            ValueError: If inputs are invalid
        """
        if panel_mass_kg <= 0:
            raise ValueError("Panel mass must be positive")

        if not panel_composition:
            raise ValueError("Panel composition cannot be empty")

        self.panel_composition = panel_composition
        self.extraction_costs = extraction_costs
        self.recovery_rates_model = recovery_rates_model
        self.panel_mass_kg = panel_mass_kg

        # Validate technology consistency
        if extraction_costs.technology != recovery_rates_model.technology:
            raise ValueError(
                "Extraction costs and recovery rates must use the same technology"
            )

    def material_extraction_costs(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate comprehensive material extraction costs.

        This method computes all cost components involved in extracting and
        processing materials from PV panels, including fixed costs, variable
        costs, and overhead.

        Returns:
            Dictionary containing:
                - total_cost: Total extraction cost in USD
                - cost_per_kg: Cost per kilogram of panel
                - fixed_costs: Fixed costs (collection, labor, depreciation)
                - variable_costs: Variable costs (processing, energy, disposal)
                - overhead_costs: Overhead costs
                - cost_breakdown: Detailed breakdown by cost component

        Example:
            >>> costs = economics.material_extraction_costs()
            >>> print(f"Total cost: ${costs['total_cost']:.2f}")
            >>> print(f"Cost per kg: ${costs['cost_per_kg']:.2f}")
        """
        # Calculate total recoverable and non-recoverable mass
        total_composition_mass = sum(mat.mass_kg for mat in self.panel_composition)
        non_recoverable_mass = self.panel_mass_kg - total_composition_mass

        # Ensure non-negative non-recoverable mass
        non_recoverable_mass = max(0, non_recoverable_mass)

        # Calculate costs using the model
        total_cost = self.extraction_costs.calculate_total_cost(
            total_mass_kg=self.panel_mass_kg,
            non_recoverable_mass_kg=non_recoverable_mass
        )

        # Calculate detailed breakdown
        preprocessing = self.extraction_costs.preprocessing_cost_per_kg * self.panel_mass_kg
        processing = self.extraction_costs.processing_cost_per_kg * self.panel_mass_kg
        purification = self.extraction_costs.purification_cost_per_kg * self.panel_mass_kg
        energy = (self.extraction_costs.energy_cost_per_kwh *
                 self.extraction_costs.energy_consumption_kwh_per_kg *
                 self.panel_mass_kg)
        disposal = self.extraction_costs.disposal_cost_per_kg * non_recoverable_mass
        labor = (self.extraction_costs.labor_cost_per_hour *
                self.extraction_costs.processing_time_hours)

        variable_costs = preprocessing + processing + purification + energy + disposal
        fixed_costs = self.extraction_costs.total_fixed_cost
        overhead_costs = (fixed_costs + variable_costs) * (
            self.extraction_costs.overhead_multiplier - 1.0
        )

        return {
            "total_cost": total_cost,
            "cost_per_kg": total_cost / self.panel_mass_kg,
            "fixed_costs": fixed_costs,
            "variable_costs": variable_costs,
            "overhead_costs": overhead_costs,
            "cost_breakdown": {
                "collection": self.extraction_costs.collection_cost_per_panel,
                "preprocessing": preprocessing,
                "processing": processing,
                "purification": purification,
                "labor": labor,
                "energy": energy,
                "disposal": disposal,
                "equipment_depreciation": self.extraction_costs.equipment_depreciation_per_panel,
                "overhead": overhead_costs,
            },
        }

    def recovery_rates(self) -> Dict[str, Union[Dict[PVMaterialType, float], float]]:
        """
        Calculate effective material recovery rates.

        This method computes the actual recovery rates for each material,
        accounting for technology efficiency and quality grades.

        Returns:
            Dictionary containing:
                - material_recovery_rates: Effective recovery rates by material (%)
                - recovered_masses: Recovered mass by material (kg)
                - total_recovery_rate: Overall recovery rate (%)
                - total_recovered_mass: Total recovered mass (kg)
                - quality_grade: Quality grade of recovered materials
                - technology_efficiency: Technology efficiency factor

        Example:
            >>> rates = economics.recovery_rates()
            >>> for material, rate in rates['material_recovery_rates'].items():
            ...     print(f"{material}: {rate:.1f}%")
        """
        effective_rates = {}
        recovered_masses = {}
        total_input_mass = 0.0
        total_recovered = 0.0

        for material_comp in self.panel_composition:
            material = material_comp.material_type
            effective_rate = self.recovery_rates_model.get_effective_recovery_rate(material)
            effective_rates[material] = effective_rate

            recovered_mass = material_comp.mass_kg * (effective_rate / 100.0)
            recovered_masses[material] = recovered_mass

            total_input_mass += material_comp.mass_kg
            total_recovered += recovered_mass

        overall_recovery_rate = (
            (total_recovered / total_input_mass * 100.0) if total_input_mass > 0 else 0.0
        )

        return {
            "material_recovery_rates": effective_rates,
            "recovered_masses": recovered_masses,
            "total_recovery_rate": overall_recovery_rate,
            "total_recovered_mass": total_recovered,
            "quality_grade": self.recovery_rates_model.quality_grade,
            "technology_efficiency": self.recovery_rates_model.technology_efficiency,
        }

    def recycling_revenue_calculation(
        self,
        market_price_adjustments: Optional[Dict[PVMaterialType, float]] = None,
        quality_discount: Optional[float] = None,
    ) -> Dict[str, Union[float, Dict[PVMaterialType, float]]]:
        """
        Calculate revenue from recycled materials.

        This method computes gross and net revenue from selling recovered materials,
        accounting for market prices, quality discounts, transportation, and commissions.

        Args:
            market_price_adjustments: Optional price adjustment multipliers by material
            quality_discount: Optional quality discount factor (0-1)

        Returns:
            Dictionary containing:
                - gross_revenue: Total revenue before costs
                - net_revenue: Revenue after transportation and commissions
                - revenue_by_material: Revenue breakdown by material
                - total_recovered_mass: Total mass recovered (kg)
                - transportation_cost: Total transportation cost
                - sales_commission: Total sales commission
                - revenue_per_kg: Revenue per kg of panel

        Example:
            >>> revenue = economics.recycling_revenue_calculation()
            >>> print(f"Net revenue: ${revenue['net_revenue']:.2f}")
            >>> print(f"Gross revenue: ${revenue['gross_revenue']:.2f}")
        """
        # Get recovered materials
        recovery_data = self.recovery_rates()
        recovered_masses = recovery_data["recovered_masses"]

        # Create recovered material compositions
        recovered_materials = []
        for material_comp in self.panel_composition:
            material = material_comp.material_type
            recovered_mass = recovered_masses.get(material, 0.0)

            if recovered_mass > 0:
                # Apply quality grade discount to purity
                quality_mult = self.recovery_rates_model.get_quality_multiplier()
                adjusted_purity = material_comp.purity_percent * quality_mult

                recovered_materials.append(
                    MaterialComposition(
                        material_type=material,
                        mass_kg=recovered_mass,
                        purity_percent=min(adjusted_purity, 100.0),
                        market_value_per_kg=material_comp.market_value_per_kg,
                    )
                )

        # Create revenue model
        revenue_model = RecyclingRevenue(
            recovered_materials=recovered_materials,
            market_price_adjustments=market_price_adjustments or {},
            quality_discount=quality_discount or self.recovery_rates_model.get_quality_multiplier(),
        )

        # Calculate revenue breakdown
        revenue_by_material = revenue_model.get_revenue_by_material()

        transport_cost = (
            revenue_model.total_recovered_mass_kg *
            revenue_model.transportation_cost_per_kg
        )
        sales_commission = (
            revenue_model.gross_revenue *
            (revenue_model.sales_commission_percent / 100)
        )

        return {
            "gross_revenue": revenue_model.gross_revenue,
            "net_revenue": revenue_model.net_revenue,
            "revenue_by_material": revenue_by_material,
            "total_recovered_mass": revenue_model.total_recovered_mass_kg,
            "transportation_cost": transport_cost,
            "sales_commission": sales_commission,
            "revenue_per_kg": revenue_model.net_revenue / self.panel_mass_kg,
        }

    def environmental_credits(
        self,
        carbon_price_per_ton_co2: float = 50.0,
        include_epr: bool = True,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate environmental credits and benefits.

        This method quantifies environmental benefits using Life Cycle Assessment
        (LCA) metrics, including avoided emissions, energy savings, and regulatory credits.

        Args:
            carbon_price_per_ton_co2: Price of carbon in USD per ton CO2
            include_epr: Whether to include Extended Producer Responsibility credits

        Returns:
            Dictionary containing:
                - total_environmental_value: Total environmental value in USD
                - carbon_value: Value from avoided CO2 emissions
                - avoided_emissions_kg_co2: Total CO2 emissions avoided
                - energy_savings_kwh: Energy saved vs. virgin production
                - water_savings_liters: Water saved vs. virgin production
                - lca_metrics: Comprehensive LCA impact indicators
                - credits_breakdown: Breakdown of credit sources

        Example:
            >>> credits = economics.environmental_credits(carbon_price_per_ton_co2=75.0)
            >>> print(f"Environmental value: ${credits['total_environmental_value']:.2f}")
            >>> print(f"CO2 avoided: {credits['avoided_emissions_kg_co2']:.1f} kg")
        """
        # Calculate avoided emissions based on recovered materials
        # These are typical LCA values for virgin vs. recycled production
        emission_factors = {
            PVMaterialType.SILICON: 50.0,  # kg CO2 per kg virgin silicon
            PVMaterialType.SILVER: 150.0,  # kg CO2 per kg virgin silver
            PVMaterialType.ALUMINUM: 12.0,  # kg CO2 per kg virgin aluminum
            PVMaterialType.COPPER: 4.5,    # kg CO2 per kg virgin copper
            PVMaterialType.GLASS: 0.85,    # kg CO2 per kg virgin glass
            PVMaterialType.POLYMER: 3.5,   # kg CO2 per kg virgin polymer
        }

        energy_factors = {
            PVMaterialType.SILICON: 180.0,  # kWh per kg virgin silicon
            PVMaterialType.SILVER: 500.0,   # kWh per kg virgin silver
            PVMaterialType.ALUMINUM: 45.0,  # kWh per kg virgin aluminum
            PVMaterialType.COPPER: 25.0,    # kWh per kg virgin copper
            PVMaterialType.GLASS: 6.0,      # kWh per kg virgin glass
            PVMaterialType.POLYMER: 80.0,   # kWh per kg virgin polymer
        }

        water_factors = {
            PVMaterialType.SILICON: 5000.0,  # liters per kg virgin silicon
            PVMaterialType.SILVER: 12000.0,  # liters per kg virgin silver
            PVMaterialType.ALUMINUM: 800.0,  # liters per kg virgin aluminum
            PVMaterialType.COPPER: 600.0,    # liters per kg virgin copper
            PVMaterialType.GLASS: 50.0,      # liters per kg virgin glass
        }

        # Get recovered masses
        recovery_data = self.recovery_rates()
        recovered_masses = recovery_data["recovered_masses"]

        total_avoided_co2 = 0.0
        total_energy_saved = 0.0
        total_water_saved = 0.0

        for material, mass in recovered_masses.items():
            total_avoided_co2 += mass * emission_factors.get(material, 0.0)
            total_energy_saved += mass * energy_factors.get(material, 0.0)
            total_water_saved += mass * water_factors.get(material, 0.0)

        # Calculate EPR credit (if applicable)
        epr_credit = 0.0
        if include_epr:
            # Typical EPR credit is $10-30 per panel for compliance
            epr_credit = 15.0

        # Landfill diversion credit
        landfill_credit = recovery_data["total_recovered_mass"] * 0.1  # $0.1/kg diverted

        # Create environmental credits model
        env_credits = EnvironmentalCredits(
            avoided_emissions_kg_co2=total_avoided_co2,
            energy_savings_kwh=total_energy_saved,
            water_savings_liters=total_water_saved,
            landfill_diversion_credits=landfill_credit,
            epr_credit_value=epr_credit,
            carbon_price_per_ton_co2=carbon_price_per_ton_co2,
        )

        lca_metrics = env_credits.get_lca_metrics()

        return {
            "total_environmental_value": env_credits.total_environmental_value,
            "carbon_value": env_credits.total_carbon_value,
            "avoided_emissions_kg_co2": total_avoided_co2,
            "energy_savings_kwh": total_energy_saved,
            "water_savings_liters": total_water_saved,
            "lca_metrics": lca_metrics,
            "credits_breakdown": {
                "carbon_credits": env_credits.total_carbon_value,
                "landfill_diversion": landfill_credit,
                "epr_credits": epr_credit,
                "regulatory_compliance": env_credits.regulatory_compliance_value,
                "renewable_energy_certificates": env_credits.renewable_energy_certificates,
            },
        }

    def net_economic_value(
        self,
        carbon_price_per_ton_co2: float = 50.0,
        include_environmental_credits: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate net economic value of recycling.

        This comprehensive method combines all economic factors to determine
        the overall profitability of recycling a PV panel.

        Args:
            carbon_price_per_ton_co2: Price of carbon in USD per ton CO2
            include_environmental_credits: Whether to include environmental value

        Returns:
            Dictionary containing:
                - net_value: Net economic value (revenue + credits - costs)
                - total_revenue: Total revenue from materials
                - total_costs: Total extraction and processing costs
                - environmental_value: Environmental credits value
                - roi_percent: Return on investment percentage
                - breakeven_carbon_price: Carbon price needed for breakeven

        Example:
            >>> net_value = economics.net_economic_value()
            >>> if net_value['net_value'] > 0:
            ...     print(f"Recycling is profitable: ${net_value['net_value']:.2f}")
            ...     print(f"ROI: {net_value['roi_percent']:.1f}%")
        """
        costs = self.material_extraction_costs()
        revenue = self.recycling_revenue_calculation()

        total_costs = costs["total_cost"]
        total_revenue = revenue["net_revenue"]

        environmental_value = 0.0
        if include_environmental_credits:
            credits = self.environmental_credits(
                carbon_price_per_ton_co2=carbon_price_per_ton_co2
            )
            environmental_value = credits["total_environmental_value"]

        net_value = total_revenue + environmental_value - total_costs

        # Calculate ROI
        roi_percent = (net_value / total_costs * 100.0) if total_costs > 0 else 0.0

        # Calculate breakeven carbon price
        deficit = total_costs - total_revenue
        credits_without_carbon = self.environmental_credits(carbon_price_per_ton_co2=0.0)
        avoided_co2_tons = credits_without_carbon["avoided_emissions_kg_co2"] / 1000.0

        breakeven_carbon_price = 0.0
        if avoided_co2_tons > 0 and deficit > 0:
            other_credits = sum(
                v for k, v in credits_without_carbon["credits_breakdown"].items()
                if k != "carbon_credits"
            )
            remaining_deficit = deficit - other_credits
            if remaining_deficit > 0:
                breakeven_carbon_price = remaining_deficit / avoided_co2_tons

        return {
            "net_value": net_value,
            "total_revenue": total_revenue,
            "total_costs": total_costs,
            "environmental_value": environmental_value,
            "roi_percent": roi_percent,
            "breakeven_carbon_price": breakeven_carbon_price,
        }
