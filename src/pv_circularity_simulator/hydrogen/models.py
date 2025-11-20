"""
Pydantic models for hydrogen system components.

This module defines the data models used for hydrogen system integration,
including electrolyzers, storage systems, fuel cells, and power-to-X pathways.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, computed_field


class ElectrolyzerType(str, Enum):
    """Types of electrolyzer technologies."""

    PEM = "PEM"  # Proton Exchange Membrane
    ALKALINE = "Alkaline"  # Alkaline Electrolyzer
    SOEC = "SOEC"  # Solid Oxide Electrolyzer Cell
    AEM = "AEM"  # Anion Exchange Membrane


class ElectrolyzerConfig(BaseModel):
    """
    Configuration for electrolyzer modeling.

    Attributes:
        electrolyzer_type: Type of electrolyzer technology
        rated_power_kw: Rated power capacity in kW
        efficiency: System efficiency (0-1, LHV basis)
        min_load_fraction: Minimum load as fraction of rated power (0-1)
        max_load_fraction: Maximum load as fraction of rated power (0-1)
        cold_start_time_min: Time required for cold start in minutes
        response_time_s: Response time for load changes in seconds
        operating_temperature_c: Operating temperature in Celsius
        operating_pressure_bar: Operating pressure in bar
        stack_lifetime_hours: Expected stack lifetime in hours
        capex_per_kw: Capital expenditure per kW
        opex_fraction: Operating expenditure as fraction of CAPEX per year
        degradation_rate_per_year: Annual degradation rate (fraction)
    """

    electrolyzer_type: ElectrolyzerType
    rated_power_kw: float = Field(..., gt=0, description="Rated power in kW")
    efficiency: float = Field(default=0.65, ge=0.3, le=0.9, description="System efficiency (LHV)")
    min_load_fraction: float = Field(default=0.1, ge=0, le=1)
    max_load_fraction: float = Field(default=1.0, ge=0, le=1)
    cold_start_time_min: float = Field(default=10.0, ge=0)
    response_time_s: float = Field(default=1.0, ge=0)
    operating_temperature_c: float = Field(default=80.0)
    operating_pressure_bar: float = Field(default=30.0, gt=0)
    stack_lifetime_hours: float = Field(default=80000.0, gt=0)
    capex_per_kw: float = Field(default=1000.0, gt=0, description="CAPEX in $/kW")
    opex_fraction: float = Field(default=0.03, ge=0, description="Annual OPEX as fraction of CAPEX")
    degradation_rate_per_year: float = Field(default=0.01, ge=0, le=0.1)

    @field_validator('max_load_fraction')
    @classmethod
    def validate_max_load(cls, v: float, info) -> float:
        """Ensure max_load_fraction >= min_load_fraction."""
        if 'min_load_fraction' in info.data and v < info.data['min_load_fraction']:
            raise ValueError("max_load_fraction must be >= min_load_fraction")
        return v

    @computed_field
    @property
    def h2_production_rate_kg_h(self) -> float:
        """Calculate nominal hydrogen production rate in kg/h."""
        # H2 LHV = 33.33 kWh/kg (lower heating value)
        h2_lhv_kwh_per_kg = 33.33
        return (self.rated_power_kw * self.efficiency) / h2_lhv_kwh_per_kg

    model_config = {"use_enum_values": False}


class ElectrolyzerResults(BaseModel):
    """
    Results from electrolyzer modeling.

    Attributes:
        h2_production_kg: Total hydrogen production in kg
        energy_consumption_kwh: Total energy consumption in kWh
        average_efficiency: Average system efficiency achieved
        capacity_factor: Fraction of time at rated capacity
        degradation_factor: Current degradation factor (1.0 = new)
        operating_hours: Total operating hours
        equivalent_full_load_hours: Equivalent hours at full load
        levelized_cost_h2: Levelized cost of hydrogen in $/kg
        specific_energy_consumption: kWh per kg H2 produced
        annual_h2_production_kg: Annual hydrogen production
        performance_metrics: Additional performance metrics
    """

    h2_production_kg: float = Field(..., ge=0)
    energy_consumption_kwh: float = Field(..., ge=0)
    average_efficiency: float = Field(..., ge=0, le=1)
    capacity_factor: float = Field(..., ge=0, le=1)
    degradation_factor: float = Field(default=1.0, ge=0, le=1)
    operating_hours: float = Field(..., ge=0)
    equivalent_full_load_hours: float = Field(..., ge=0)
    levelized_cost_h2: float = Field(..., ge=0, description="$/kg H2")
    specific_energy_consumption: float = Field(..., ge=0, description="kWh/kg H2")
    annual_h2_production_kg: float = Field(..., ge=0)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class StorageType(str, Enum):
    """Types of hydrogen storage technologies."""

    COMPRESSED_GAS = "compressed_gas"  # High-pressure tanks
    LIQUID_H2 = "liquid_h2"  # Cryogenic storage
    METAL_HYDRIDE = "metal_hydride"  # Metal hydride storage
    LOHC = "lohc"  # Liquid Organic Hydrogen Carriers
    UNDERGROUND = "underground"  # Salt caverns, aquifers


class StorageConfig(BaseModel):
    """
    Configuration for hydrogen storage design.

    Attributes:
        storage_type: Type of storage technology
        capacity_kg: Storage capacity in kg H2
        pressure_bar: Storage pressure in bar (for compressed gas)
        temperature_k: Storage temperature in Kelvin
        charging_rate_kg_h: Maximum charging rate in kg/h
        discharging_rate_kg_h: Maximum discharging rate in kg/h
        round_trip_efficiency: Round-trip efficiency (0-1)
        self_discharge_rate_per_day: Daily self-discharge rate (fraction)
        capex_per_kg: Capital cost per kg storage capacity
        opex_fraction: Annual OPEX as fraction of CAPEX
        lifetime_years: Expected lifetime in years
        min_soc_fraction: Minimum state of charge (fraction)
        max_soc_fraction: Maximum state of charge (fraction)
    """

    storage_type: StorageType
    capacity_kg: float = Field(..., gt=0, description="Storage capacity in kg H2")
    pressure_bar: Optional[float] = Field(default=350.0, gt=0)
    temperature_k: float = Field(default=293.15, gt=0)
    charging_rate_kg_h: float = Field(..., gt=0)
    discharging_rate_kg_h: float = Field(..., gt=0)
    round_trip_efficiency: float = Field(default=0.95, ge=0.5, le=1.0)
    self_discharge_rate_per_day: float = Field(default=0.001, ge=0, le=0.1)
    capex_per_kg: float = Field(default=500.0, gt=0, description="$/kg capacity")
    opex_fraction: float = Field(default=0.02, ge=0)
    lifetime_years: float = Field(default=20.0, gt=0)
    min_soc_fraction: float = Field(default=0.1, ge=0, le=1)
    max_soc_fraction: float = Field(default=0.95, ge=0, le=1)

    @field_validator('max_soc_fraction')
    @classmethod
    def validate_max_soc(cls, v: float, info) -> float:
        """Ensure max_soc_fraction > min_soc_fraction."""
        if 'min_soc_fraction' in info.data and v <= info.data['min_soc_fraction']:
            raise ValueError("max_soc_fraction must be > min_soc_fraction")
        return v

    @computed_field
    @property
    def usable_capacity_kg(self) -> float:
        """Calculate usable storage capacity based on SOC limits."""
        return self.capacity_kg * (self.max_soc_fraction - self.min_soc_fraction)


class StorageResults(BaseModel):
    """
    Results from hydrogen storage design analysis.

    Attributes:
        total_capacity_kg: Total storage capacity
        usable_capacity_kg: Usable capacity (accounting for SOC limits)
        average_soc: Average state of charge over period
        total_charged_kg: Total hydrogen charged
        total_discharged_kg: Total hydrogen discharged
        total_losses_kg: Total losses (self-discharge, inefficiency)
        average_efficiency: Average round-trip efficiency
        cycling_count: Number of charge-discharge cycles
        capacity_utilization: Fraction of capacity utilized
        levelized_cost_storage: Levelized cost of storage in $/kg
        storage_metrics: Additional storage metrics
    """

    total_capacity_kg: float = Field(..., ge=0)
    usable_capacity_kg: float = Field(..., ge=0)
    average_soc: float = Field(..., ge=0, le=1)
    total_charged_kg: float = Field(..., ge=0)
    total_discharged_kg: float = Field(..., ge=0)
    total_losses_kg: float = Field(..., ge=0)
    average_efficiency: float = Field(..., ge=0, le=1)
    cycling_count: float = Field(..., ge=0)
    capacity_utilization: float = Field(..., ge=0, le=1)
    levelized_cost_storage: float = Field(..., ge=0, description="$/kg H2 stored")
    storage_metrics: Dict[str, float] = Field(default_factory=dict)


class FuelCellType(str, Enum):
    """Types of fuel cell technologies."""

    PEMFC = "PEMFC"  # Proton Exchange Membrane Fuel Cell
    SOFC = "SOFC"  # Solid Oxide Fuel Cell
    MCFC = "MCFC"  # Molten Carbonate Fuel Cell
    AFC = "AFC"  # Alkaline Fuel Cell
    PAFC = "PAFC"  # Phosphoric Acid Fuel Cell


class FuelCellConfig(BaseModel):
    """
    Configuration for fuel cell integration.

    Attributes:
        fuel_cell_type: Type of fuel cell technology
        rated_power_kw: Rated electrical power output in kW
        efficiency: Electrical efficiency at rated power (0-1, LHV basis)
        min_load_fraction: Minimum load as fraction of rated power
        max_load_fraction: Maximum load as fraction of rated power
        cold_start_time_min: Cold start time in minutes
        response_time_s: Response time for load changes in seconds
        operating_temperature_c: Operating temperature in Celsius
        stack_lifetime_hours: Expected stack lifetime in hours
        degradation_rate_per_hour: Degradation per operating hour
        capex_per_kw: Capital cost per kW output
        opex_fraction: Annual OPEX as fraction of CAPEX
        heat_recovery_fraction: Fraction of waste heat recoverable
        cogeneration_enabled: Whether CHP (combined heat-power) is enabled
    """

    fuel_cell_type: FuelCellType
    rated_power_kw: float = Field(..., gt=0)
    efficiency: float = Field(default=0.55, ge=0.3, le=0.7, description="Electrical efficiency (LHV)")
    min_load_fraction: float = Field(default=0.05, ge=0, le=1)
    max_load_fraction: float = Field(default=1.0, ge=0, le=1)
    cold_start_time_min: float = Field(default=5.0, ge=0)
    response_time_s: float = Field(default=0.5, ge=0)
    operating_temperature_c: float = Field(default=80.0)
    stack_lifetime_hours: float = Field(default=40000.0, gt=0)
    degradation_rate_per_hour: float = Field(default=1e-5, ge=0)
    capex_per_kw: float = Field(default=1500.0, gt=0)
    opex_fraction: float = Field(default=0.04, ge=0)
    heat_recovery_fraction: float = Field(default=0.3, ge=0, le=1)
    cogeneration_enabled: bool = Field(default=False)

    @computed_field
    @property
    def h2_consumption_rate_kg_h(self) -> float:
        """Calculate nominal hydrogen consumption rate in kg/h."""
        h2_lhv_kwh_per_kg = 33.33
        return self.rated_power_kw / (self.efficiency * h2_lhv_kwh_per_kg)

    @computed_field
    @property
    def thermal_power_kw(self) -> float:
        """Calculate recoverable thermal power output."""
        total_input = self.rated_power_kw / self.efficiency
        waste_heat = total_input - self.rated_power_kw
        return waste_heat * self.heat_recovery_fraction


class FuelCellResults(BaseModel):
    """
    Results from fuel cell integration analysis.

    Attributes:
        electrical_output_kwh: Total electrical energy produced
        thermal_output_kwh: Total thermal energy recovered
        h2_consumed_kg: Total hydrogen consumed
        average_efficiency: Average electrical efficiency
        capacity_factor: Capacity factor achieved
        operating_hours: Total operating hours
        degradation_factor: Current degradation factor
        levelized_cost_electricity: LCOE in $/kWh
        specific_h2_consumption: kg H2 per kWh electrical
        cogeneration_efficiency: Combined efficiency (if CHP enabled)
        performance_metrics: Additional performance metrics
    """

    electrical_output_kwh: float = Field(..., ge=0)
    thermal_output_kwh: float = Field(default=0.0, ge=0)
    h2_consumed_kg: float = Field(..., ge=0)
    average_efficiency: float = Field(..., ge=0, le=1)
    capacity_factor: float = Field(..., ge=0, le=1)
    operating_hours: float = Field(..., ge=0)
    degradation_factor: float = Field(default=1.0, ge=0, le=1)
    levelized_cost_electricity: float = Field(..., ge=0, description="$/kWh")
    specific_h2_consumption: float = Field(..., ge=0, description="kg H2/kWh")
    cogeneration_efficiency: Optional[float] = Field(default=None, ge=0, le=1)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class PowerToXPathway(str, Enum):
    """Power-to-X conversion pathways."""

    POWER_TO_H2 = "power_to_h2"  # Direct hydrogen production
    POWER_TO_METHANE = "power_to_methane"  # Methanation (Sabatier)
    POWER_TO_METHANOL = "power_to_methanol"  # Methanol synthesis
    POWER_TO_AMMONIA = "power_to_ammonia"  # Haber-Bosch process
    POWER_TO_LIQUID = "power_to_liquid"  # Fischer-Tropsch fuels
    POWER_TO_SNG = "power_to_sng"  # Synthetic natural gas


class PowerToXConfig(BaseModel):
    """
    Configuration for Power-to-X pathway analysis.

    Attributes:
        pathway: Power-to-X conversion pathway
        electrolyzer_config: Electrolyzer configuration for H2 production
        conversion_efficiency: Efficiency of H2 to final product (0-1)
        co2_source: Source of CO2 (if required)
        co2_capture_cost_per_ton: Cost of CO2 capture in $/ton
        n2_source: Source of nitrogen (for ammonia)
        process_temperature_c: Process temperature in Celsius
        process_pressure_bar: Process pressure in bar
        catalyst_type: Type of catalyst used
        catalyst_lifetime_hours: Catalyst lifetime in hours
        capex_conversion_per_kw: CAPEX for conversion unit per kW H2 input
        opex_fraction: Annual OPEX as fraction of total CAPEX
        product_storage_capacity: Storage capacity for final product
        product_lhv_kwh_per_kg: Lower heating value of product in kWh/kg
    """

    pathway: PowerToXPathway
    electrolyzer_config: ElectrolyzerConfig
    conversion_efficiency: float = Field(..., ge=0.5, le=1.0)
    co2_source: Optional[str] = Field(default=None)
    co2_capture_cost_per_ton: float = Field(default=100.0, ge=0)
    n2_source: Optional[str] = Field(default=None)
    process_temperature_c: float = Field(default=300.0)
    process_pressure_bar: float = Field(default=50.0, gt=0)
    catalyst_type: Optional[str] = Field(default=None)
    catalyst_lifetime_hours: float = Field(default=10000.0, gt=0)
    capex_conversion_per_kw: float = Field(default=800.0, gt=0)
    opex_fraction: float = Field(default=0.035, ge=0)
    product_storage_capacity: float = Field(default=1000.0, gt=0)
    product_lhv_kwh_per_kg: float = Field(default=13.9, gt=0)  # e.g., methanol

    @computed_field
    @property
    def overall_efficiency(self) -> float:
        """Calculate overall power-to-product efficiency."""
        return self.electrolyzer_config.efficiency * self.conversion_efficiency

    @computed_field
    @property
    def requires_co2(self) -> bool:
        """Check if pathway requires CO2 input."""
        return self.pathway in [
            PowerToXPathway.POWER_TO_METHANE,
            PowerToXPathway.POWER_TO_METHANOL,
            PowerToXPathway.POWER_TO_LIQUID,
            PowerToXPathway.POWER_TO_SNG,
        ]

    @computed_field
    @property
    def requires_n2(self) -> bool:
        """Check if pathway requires nitrogen input."""
        return self.pathway == PowerToXPathway.POWER_TO_AMMONIA


class PowerToXResults(BaseModel):
    """
    Results from Power-to-X pathway analysis.

    Attributes:
        product_output_kg: Total product output in kg
        h2_intermediate_kg: Hydrogen produced as intermediate
        energy_input_kwh: Total electrical energy input
        co2_consumed_kg: CO2 consumed (if applicable)
        n2_consumed_kg: Nitrogen consumed (if applicable)
        overall_efficiency: Overall power-to-product efficiency
        specific_energy_consumption: kWh per kg product
        levelized_cost_product: Levelized cost in $/kg product
        carbon_intensity: kg CO2 eq per kg product
        capacity_factor: Overall capacity factor
        economic_metrics: Economic analysis results
        environmental_metrics: Environmental impact metrics
    """

    product_output_kg: float = Field(..., ge=0)
    h2_intermediate_kg: float = Field(..., ge=0)
    energy_input_kwh: float = Field(..., ge=0)
    co2_consumed_kg: Optional[float] = Field(default=0.0, ge=0)
    n2_consumed_kg: Optional[float] = Field(default=0.0, ge=0)
    overall_efficiency: float = Field(..., ge=0, le=1)
    specific_energy_consumption: float = Field(..., ge=0)
    levelized_cost_product: float = Field(..., ge=0)
    carbon_intensity: float = Field(..., description="kg CO2 eq/kg product")
    capacity_factor: float = Field(..., ge=0, le=1)
    economic_metrics: Dict[str, float] = Field(default_factory=dict)
    environmental_metrics: Dict[str, float] = Field(default_factory=dict)
