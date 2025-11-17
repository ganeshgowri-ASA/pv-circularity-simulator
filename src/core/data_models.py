"""
Comprehensive Pydantic data models for PV Circularity Simulator.

This module defines all core data structures used throughout the simulator,
including materials, cells, modules, systems, and circularity metrics.
All models include validation, constraints, and comprehensive documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import PositiveFloat, NonNegativeFloat, confloat, conint


# ============================================================================
# Enumerations
# ============================================================================

class CellTechnology(str, Enum):
    """Supported photovoltaic cell technologies."""
    MONOCRYSTALLINE = "monocrystalline"
    POLYCRYSTALLINE = "polycrystalline"
    PERC = "perc"
    TOPCON = "topcon"
    HJT = "hjt"  # Heterojunction
    IBC = "ibc"  # Interdigitated Back Contact
    THIN_FILM_CIGS = "thin_film_cigs"
    THIN_FILM_CDTE = "thin_film_cdte"
    PEROVSKITE = "perovskite"
    TANDEM = "tandem"


class MountingType(str, Enum):
    """PV system mounting configurations."""
    FIXED_TILT = "fixed_tilt"
    SINGLE_AXIS_TRACKING = "single_axis_tracking"
    DUAL_AXIS_TRACKING = "dual_axis_tracking"
    ROOF_MOUNTED = "roof_mounted"
    GROUND_MOUNTED = "ground_mounted"
    BUILDING_INTEGRATED = "building_integrated"
    FLOATING = "floating"
    CARPORT = "carport"


class MaterialType(str, Enum):
    """Types of materials used in PV modules."""
    SILICON = "silicon"
    GLASS = "glass"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    SILVER = "silver"
    EVA = "eva"  # Ethylene Vinyl Acetate
    BACKSHEET = "backsheet"
    JUNCTION_BOX = "junction_box"
    ENCAPSULANT = "encapsulant"
    FRAME = "frame"
    SOLDER = "solder"
    OTHER = "other"


# ============================================================================
# Material Models
# ============================================================================

class MaterialProperties(BaseModel):
    """Physical and chemical properties of a material."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    density: PositiveFloat = Field(
        ...,
        description="Material density in kg/m³"
    )
    thermal_conductivity: NonNegativeFloat = Field(
        ...,
        description="Thermal conductivity in W/(m·K)"
    )
    specific_heat: PositiveFloat = Field(
        ...,
        description="Specific heat capacity in J/(kg·K)"
    )
    melting_point: Optional[float] = Field(
        None,
        description="Melting point in Celsius"
    )
    recyclability_rate: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Recyclability rate (0-1, where 1 is 100% recyclable)"
    )
    embodied_energy: NonNegativeFloat = Field(
        ...,
        description="Embodied energy in MJ/kg"
    )
    carbon_footprint: NonNegativeFloat = Field(
        ...,
        description="Carbon footprint in kg CO2-eq/kg"
    )
    toxicity_score: confloat(ge=0.0, le=10.0) = Field(
        default=0.0,
        description="Toxicity score (0-10, where 0 is non-toxic)"
    )

    @field_validator('melting_point')
    @classmethod
    def validate_melting_point(cls, v: Optional[float]) -> Optional[float]:
        """Validate melting point is within reasonable range."""
        if v is not None and (v < -273.15 or v > 5000):
            raise ValueError("Melting point must be between -273.15°C and 5000°C")
        return v


class Material(BaseModel):
    """Material used in PV module construction."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material name"
    )
    material_type: MaterialType = Field(
        ...,
        description="Type of material"
    )
    properties: MaterialProperties = Field(
        ...,
        description="Physical and chemical properties"
    )
    mass_per_module: PositiveFloat = Field(
        ...,
        description="Mass of this material per module in kg"
    )
    cost_per_kg: NonNegativeFloat = Field(
        ...,
        description="Cost per kilogram in USD"
    )
    supplier: Optional[str] = Field(
        None,
        max_length=200,
        description="Material supplier name"
    )
    supply_chain_risk: confloat(ge=0.0, le=1.0) = Field(
        default=0.5,
        description="Supply chain risk score (0-1)"
    )

    @property
    def total_cost(self) -> float:
        """Calculate total cost of material per module."""
        return self.mass_per_module * self.cost_per_kg

    @property
    def total_embodied_energy(self) -> float:
        """Calculate total embodied energy per module in MJ."""
        return self.mass_per_module * self.properties.embodied_energy

    @property
    def total_carbon_footprint(self) -> float:
        """Calculate total carbon footprint per module in kg CO2-eq."""
        return self.mass_per_module * self.properties.carbon_footprint


class CircularityMetrics(BaseModel):
    """Circularity metrics for materials and modules."""

    model_config = ConfigDict(validate_assignment=True)

    recyclability_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Overall recyclability score (0-1)"
    )
    recycled_content_ratio: confloat(ge=0.0, le=1.0) = Field(
        default=0.0,
        description="Ratio of recycled content used (0-1)"
    )
    reusability_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Reusability potential score (0-1)"
    )
    repairability_score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Repairability score (0-1)"
    )
    material_recovery_potential: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Material recovery potential at end-of-life (0-1)"
    )
    lifetime_extension_potential: confloat(ge=0.0, le=1.0) = Field(
        default=0.5,
        description="Potential for lifetime extension (0-1)"
    )
    circular_economy_index: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Overall circular economy index (0-100)"
    )

    @model_validator(mode='after')
    def calculate_circular_economy_index(self) -> 'CircularityMetrics':
        """Calculate overall circular economy index if not provided."""
        if self.circular_economy_index is None:
            # Weighted average of all metrics
            self.circular_economy_index = (
                self.recyclability_score * 30 +
                self.recycled_content_ratio * 15 +
                self.reusability_score * 20 +
                self.repairability_score * 15 +
                self.material_recovery_potential * 15 +
                self.lifetime_extension_potential * 5
            )
        return self


# ============================================================================
# Cell Models
# ============================================================================

class TemperatureCoefficients(BaseModel):
    """Temperature coefficients for cell performance."""

    model_config = ConfigDict(validate_assignment=True)

    power: float = Field(
        ...,
        ge=-1.0,
        le=0.0,
        description="Power temperature coefficient in %/°C (typically negative)"
    )
    voltage: float = Field(
        ...,
        ge=-1.0,
        le=0.0,
        description="Voltage temperature coefficient in %/°C (typically negative)"
    )
    current: float = Field(
        ...,
        ge=-0.1,
        le=0.2,
        description="Current temperature coefficient in %/°C (typically positive)"
    )

    @field_validator('power', 'voltage', 'current')
    @classmethod
    def validate_reasonable_range(cls, v: float, info) -> float:
        """Validate temperature coefficients are in reasonable ranges."""
        field_name = info.field_name
        if field_name in ['power', 'voltage'] and v > 0:
            raise ValueError(f"{field_name} temperature coefficient should be negative")
        if field_name == 'current' and v < 0:
            raise ValueError("Current temperature coefficient should be positive")
        return v


class Cell(BaseModel):
    """Photovoltaic cell specifications."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    technology: CellTechnology = Field(
        ...,
        description="Cell technology type"
    )
    efficiency: confloat(gt=0.0, le=1.0) = Field(
        ...,
        description="Cell efficiency (0-1, where 1 is 100%)"
    )
    area: PositiveFloat = Field(
        ...,
        description="Cell area in m²"
    )
    thickness: PositiveFloat = Field(
        ...,
        description="Cell thickness in micrometers (µm)"
    )
    power_output: PositiveFloat = Field(
        ...,
        description="Cell power output in watts at STC"
    )
    voltage_at_max_power: PositiveFloat = Field(
        ...,
        description="Voltage at maximum power point (Vmp) in volts"
    )
    current_at_max_power: PositiveFloat = Field(
        ...,
        description="Current at maximum power point (Imp) in amperes"
    )
    open_circuit_voltage: PositiveFloat = Field(
        ...,
        description="Open circuit voltage (Voc) in volts"
    )
    short_circuit_current: PositiveFloat = Field(
        ...,
        description="Short circuit current (Isc) in amperes"
    )
    fill_factor: confloat(gt=0.0, le=1.0) = Field(
        ...,
        description="Fill factor (0-1)"
    )
    temperature_coefficients: TemperatureCoefficients = Field(
        ...,
        description="Temperature coefficients for performance"
    )
    degradation_rate: confloat(ge=0.0, le=0.05) = Field(
        default=0.005,
        description="Annual degradation rate (0-0.05, typically 0.5%/year)"
    )
    manufacturing_cost: NonNegativeFloat = Field(
        ...,
        description="Manufacturing cost per cell in USD"
    )

    @model_validator(mode='after')
    def validate_electrical_parameters(self) -> 'Cell':
        """Validate electrical parameters are consistent."""
        # P = V * I
        calculated_power = self.voltage_at_max_power * self.current_at_max_power
        if abs(calculated_power - self.power_output) > 0.1:
            raise ValueError(
                f"Power mismatch: Vmp * Imp = {calculated_power:.2f}W "
                f"but power_output = {self.power_output:.2f}W"
            )

        # Vmp should be less than Voc
        if self.voltage_at_max_power >= self.open_circuit_voltage:
            raise ValueError("Vmp must be less than Voc")

        # Imp should be less than Isc
        if self.current_at_max_power >= self.short_circuit_current:
            raise ValueError("Imp must be less than Isc")

        # Fill factor validation
        max_power_theoretical = self.open_circuit_voltage * self.short_circuit_current
        calculated_ff = self.power_output / max_power_theoretical
        if abs(calculated_ff - self.fill_factor) > 0.01:
            raise ValueError(
                f"Fill factor mismatch: calculated {calculated_ff:.3f} "
                f"but specified {self.fill_factor:.3f}"
            )

        return self


# ============================================================================
# Module Models
# ============================================================================

class CuttingPattern(BaseModel):
    """Cell cutting pattern for module design."""

    model_config = ConfigDict(validate_assignment=True)

    pattern_type: str = Field(
        ...,
        description="Pattern type (e.g., 'full', 'half-cut', 'third-cut', 'quarter-cut')"
    )
    segments_per_cell: conint(ge=1, le=12) = Field(
        ...,
        description="Number of segments per cell after cutting"
    )
    cutting_loss: confloat(ge=0.0, le=0.1) = Field(
        default=0.01,
        description="Material loss due to cutting (0-0.1, typically 1%)"
    )
    efficiency_gain: confloat(ge=0.0, le=0.2) = Field(
        default=0.0,
        description="Efficiency gain from reduced resistive losses (0-0.2)"
    )
    cost_increase: confloat(ge=0.0, le=1.0) = Field(
        default=0.05,
        description="Relative cost increase due to cutting process (0-1)"
    )

    @field_validator('pattern_type')
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        """Validate pattern type is recognized."""
        valid_patterns = ['full', 'half-cut', 'third-cut', 'quarter-cut', 'shingled']
        if v.lower() not in valid_patterns:
            raise ValueError(f"Pattern type must be one of: {', '.join(valid_patterns)}")
        return v.lower()


class ModuleLayout(BaseModel):
    """Physical layout configuration of cells in a module."""

    model_config = ConfigDict(validate_assignment=True)

    cells_in_series: conint(ge=1, le=200) = Field(
        ...,
        description="Number of cells connected in series"
    )
    cells_in_parallel: conint(ge=1, le=10) = Field(
        default=1,
        description="Number of parallel strings"
    )
    bypass_diodes: conint(ge=0, le=20) = Field(
        ...,
        description="Number of bypass diodes"
    )
    rows: conint(ge=1, le=20) = Field(
        ...,
        description="Number of cell rows"
    )
    columns: conint(ge=1, le=20) = Field(
        ...,
        description="Number of cell columns"
    )

    @model_validator(mode='after')
    def validate_layout_consistency(self) -> 'ModuleLayout':
        """Validate layout configuration is consistent."""
        total_cells = self.cells_in_series * self.cells_in_parallel
        layout_cells = self.rows * self.columns

        if total_cells != layout_cells:
            raise ValueError(
                f"Cell count mismatch: {self.cells_in_series} × {self.cells_in_parallel} = {total_cells} "
                f"but {self.rows} × {self.columns} = {layout_cells}"
            )

        # Typical rule: one bypass diode per 20-24 cells in series
        if self.cells_in_series > self.bypass_diodes * 30:
            raise ValueError(
                f"Insufficient bypass diodes: {self.bypass_diodes} for {self.cells_in_series} series cells"
            )

        return self


class Module(BaseModel):
    """Complete PV module specification."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Module model name/identifier"
    )
    manufacturer: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Module manufacturer"
    )
    cell: Cell = Field(
        ...,
        description="Cell specifications"
    )
    layout: ModuleLayout = Field(
        ...,
        description="Module layout configuration"
    )
    cutting_pattern: Optional[CuttingPattern] = Field(
        None,
        description="Cell cutting pattern (if applicable)"
    )
    materials: List[Material] = Field(
        default_factory=list,
        description="List of materials used in module"
    )
    rated_power: PositiveFloat = Field(
        ...,
        description="Rated power output in watts (STC)"
    )
    length: PositiveFloat = Field(
        ...,
        description="Module length in meters"
    )
    width: PositiveFloat = Field(
        ...,
        description="Module width in meters"
    )
    thickness: PositiveFloat = Field(
        ...,
        description="Module thickness in meters"
    )
    weight: PositiveFloat = Field(
        ...,
        description="Module weight in kilograms"
    )
    efficiency: confloat(gt=0.0, le=1.0) = Field(
        ...,
        description="Module efficiency (0-1)"
    )
    warranty_years: conint(ge=1, le=50) = Field(
        default=25,
        description="Product warranty in years"
    )
    performance_warranty_years: conint(ge=1, le=50) = Field(
        default=25,
        description="Performance warranty in years"
    )
    circularity_metrics: Optional[CircularityMetrics] = Field(
        None,
        description="Circularity metrics for the module"
    )
    manufacturing_date: Optional[datetime] = Field(
        None,
        description="Manufacturing date"
    )

    @property
    def area(self) -> float:
        """Calculate module area in m²."""
        return self.length * self.width

    @property
    def total_cells(self) -> int:
        """Calculate total number of cells in module."""
        return self.layout.cells_in_series * self.layout.cells_in_parallel

    @property
    def power_density(self) -> float:
        """Calculate power density in W/m²."""
        return self.rated_power / self.area

    @property
    def total_material_cost(self) -> float:
        """Calculate total material cost."""
        return sum(material.total_cost for material in self.materials)

    @model_validator(mode='after')
    def validate_module_specifications(self) -> 'Module':
        """Validate module specifications are consistent."""
        # Efficiency check
        total_cell_power = self.cell.power_output * self.total_cells

        # Account for cutting pattern if present
        if self.cutting_pattern:
            efficiency_factor = 1 - self.cutting_pattern.cutting_loss + self.cutting_pattern.efficiency_gain
            total_cell_power *= efficiency_factor

        # Allow some tolerance for CTM losses (typically 2-5%)
        if self.rated_power > total_cell_power:
            raise ValueError(
                f"Module power ({self.rated_power}W) exceeds total cell power ({total_cell_power:.1f}W)"
            )

        ctm_ratio = self.rated_power / total_cell_power
        if ctm_ratio < 0.90:  # More than 10% CTM loss is suspicious
            raise ValueError(
                f"CTM ratio too low: {ctm_ratio:.1%} (module: {self.rated_power}W, cells: {total_cell_power:.1f}W)"
            )

        # Performance warranty should not exceed product warranty
        if self.performance_warranty_years > self.warranty_years:
            raise ValueError(
                "Performance warranty cannot exceed product warranty"
            )

        return self


# ============================================================================
# System Models
# ============================================================================

class Location(BaseModel):
    """Geographic location for PV system."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    latitude: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Latitude in decimal degrees"
    )
    longitude: confloat(ge=-180.0, le=180.0) = Field(
        ...,
        description="Longitude in decimal degrees"
    )
    altitude: float = Field(
        default=0.0,
        ge=-500.0,
        le=9000.0,
        description="Altitude above sea level in meters"
    )
    timezone: str = Field(
        ...,
        description="Timezone (e.g., 'America/New_York', 'UTC')"
    )
    city: Optional[str] = Field(
        None,
        max_length=100,
        description="City name"
    )
    country: Optional[str] = Field(
        None,
        max_length=100,
        description="Country name"
    )
    climate_zone: Optional[str] = Field(
        None,
        description="Climate zone classification"
    )


class PVSystem(BaseModel):
    """Complete PV system configuration."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    system_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="System name/identifier"
    )
    location: Location = Field(
        ...,
        description="System geographic location"
    )
    modules: List[Module] = Field(
        ...,
        min_length=1,
        description="List of modules in the system"
    )
    module_quantity: conint(ge=1) = Field(
        ...,
        description="Total number of modules"
    )
    mounting_type: MountingType = Field(
        ...,
        description="Mounting configuration"
    )
    tilt_angle: confloat(ge=0.0, le=90.0) = Field(
        ...,
        description="Tilt angle in degrees from horizontal"
    )
    azimuth_angle: confloat(ge=0.0, lt=360.0) = Field(
        ...,
        description="Azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)"
    )
    dc_capacity: PositiveFloat = Field(
        ...,
        description="Total DC capacity in kilowatts"
    )
    ac_capacity: PositiveFloat = Field(
        ...,
        description="Total AC capacity in kilowatts"
    )
    inverter_efficiency: confloat(gt=0.0, le=1.0) = Field(
        default=0.96,
        description="Inverter efficiency (0-1)"
    )
    system_losses: confloat(ge=0.0, le=0.5) = Field(
        default=0.14,
        description="Total system losses excluding inverter (0-0.5, typically 14%)"
    )
    installation_date: Optional[datetime] = Field(
        None,
        description="System installation date"
    )
    expected_lifetime: conint(ge=1, le=50) = Field(
        default=25,
        description="Expected system lifetime in years"
    )

    @property
    def dc_ac_ratio(self) -> float:
        """Calculate DC to AC ratio (inverter loading ratio)."""
        return self.dc_capacity / self.ac_capacity

    @property
    def total_module_area(self) -> float:
        """Calculate total module area in m²."""
        return sum(module.area * self.module_quantity for module in self.modules)

    @model_validator(mode='after')
    def validate_system_configuration(self) -> 'PVSystem':
        """Validate system configuration is consistent."""
        # Calculate expected DC capacity from modules
        total_module_power = sum(
            module.rated_power * self.module_quantity for module in self.modules
        )
        expected_dc_kw = total_module_power / 1000.0

        if abs(expected_dc_kw - self.dc_capacity) > 0.5:
            raise ValueError(
                f"DC capacity mismatch: specified {self.dc_capacity}kW "
                f"but modules total {expected_dc_kw:.2f}kW"
            )

        # AC capacity should be less than DC capacity
        if self.ac_capacity > self.dc_capacity:
            raise ValueError("AC capacity cannot exceed DC capacity")

        # Typical DC/AC ratio is between 1.0 and 1.5
        if self.dc_ac_ratio < 0.9 or self.dc_ac_ratio > 2.0:
            raise ValueError(
                f"DC/AC ratio {self.dc_ac_ratio:.2f} is outside typical range (0.9-2.0)"
            )

        # Validate tilt angle based on mounting type
        if self.mounting_type == MountingType.DUAL_AXIS_TRACKING and self.tilt_angle != 0:
            raise ValueError("Dual-axis tracking systems should have 0° tilt (dynamic)")

        return self


# ============================================================================
# Performance & Financial Models
# ============================================================================

class PerformanceData(BaseModel):
    """Performance monitoring data for PV system."""

    model_config = ConfigDict(validate_assignment=True)

    timestamp: datetime = Field(
        ...,
        description="Timestamp of measurement"
    )
    dc_power: NonNegativeFloat = Field(
        ...,
        description="DC power output in watts"
    )
    ac_power: NonNegativeFloat = Field(
        ...,
        description="AC power output in watts"
    )
    dc_voltage: NonNegativeFloat = Field(
        ...,
        description="DC voltage in volts"
    )
    dc_current: NonNegativeFloat = Field(
        ...,
        description="DC current in amperes"
    )
    ac_voltage: NonNegativeFloat = Field(
        ...,
        description="AC voltage in volts"
    )
    ac_current: NonNegativeFloat = Field(
        ...,
        description="AC current in amperes"
    )
    irradiance: NonNegativeFloat = Field(
        ...,
        description="Plane of array irradiance in W/m²"
    )
    module_temperature: float = Field(
        ...,
        ge=-40.0,
        le=100.0,
        description="Module temperature in Celsius"
    )
    ambient_temperature: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature in Celsius"
    )
    wind_speed: NonNegativeFloat = Field(
        default=0.0,
        description="Wind speed in m/s"
    )
    energy_today: NonNegativeFloat = Field(
        default=0.0,
        description="Energy produced today in kWh"
    )
    energy_total: NonNegativeFloat = Field(
        default=0.0,
        description="Cumulative energy produced in kWh"
    )
    performance_ratio: Optional[confloat(ge=0.0, le=2.0)] = Field(
        None,
        description="Performance ratio (actual/theoretical yield)"
    )

    @model_validator(mode='after')
    def validate_performance_data(self) -> 'PerformanceData':
        """Validate performance measurements are consistent."""
        # DC power check
        if self.dc_voltage > 0 and self.dc_current > 0:
            calculated_dc = self.dc_voltage * self.dc_current
            if abs(calculated_dc - self.dc_power) > max(0.1 * self.dc_power, 10):
                raise ValueError(
                    f"DC power mismatch: V×I = {calculated_dc:.1f}W "
                    f"but dc_power = {self.dc_power:.1f}W"
                )

        # AC power should not exceed DC power
        if self.ac_power > self.dc_power * 1.01:  # Small tolerance for measurement error
            raise ValueError("AC power cannot exceed DC power")

        # Module temperature should be higher than ambient under irradiance
        if self.irradiance > 100 and self.module_temperature < self.ambient_temperature:
            raise ValueError(
                "Module temperature should be higher than ambient under irradiance"
            )

        return self


class FinancialModel(BaseModel):
    """Financial model for PV system investment."""

    model_config = ConfigDict(validate_assignment=True)

    system_cost: PositiveFloat = Field(
        ...,
        description="Total system cost in USD"
    )
    module_cost: PositiveFloat = Field(
        ...,
        description="Total module cost in USD"
    )
    inverter_cost: PositiveFloat = Field(
        ...,
        description="Total inverter cost in USD"
    )
    balance_of_system_cost: NonNegativeFloat = Field(
        ...,
        description="Balance of system cost in USD"
    )
    installation_cost: NonNegativeFloat = Field(
        ...,
        description="Installation labor cost in USD"
    )
    annual_om_cost: NonNegativeFloat = Field(
        ...,
        description="Annual operation & maintenance cost in USD"
    )
    electricity_rate: PositiveFloat = Field(
        ...,
        description="Electricity rate in USD/kWh"
    )
    electricity_rate_escalation: confloat(ge=-0.1, le=0.2) = Field(
        default=0.02,
        description="Annual electricity rate escalation (0.02 = 2%)"
    )
    discount_rate: confloat(ge=0.0, le=0.3) = Field(
        default=0.06,
        description="Discount rate for NPV calculation (0.06 = 6%)"
    )
    incentives: NonNegativeFloat = Field(
        default=0.0,
        description="Total incentives and tax credits in USD"
    )
    degradation_rate: confloat(ge=0.0, le=0.05) = Field(
        default=0.005,
        description="Annual system degradation rate (0.005 = 0.5%/year)"
    )
    system_lifetime: conint(ge=1, le=50) = Field(
        default=25,
        description="System lifetime for financial analysis in years"
    )
    salvage_value: NonNegativeFloat = Field(
        default=0.0,
        description="System salvage value at end of life in USD"
    )

    # Calculated metrics (can be set or auto-calculated)
    lcoe: Optional[PositiveFloat] = Field(
        None,
        description="Levelized cost of energy in USD/kWh"
    )
    npv: Optional[float] = Field(
        None,
        description="Net present value in USD"
    )
    irr: Optional[confloat(ge=-1.0, le=2.0)] = Field(
        None,
        description="Internal rate of return (0.1 = 10%)"
    )
    payback_period: Optional[PositiveFloat] = Field(
        None,
        description="Simple payback period in years"
    )

    @model_validator(mode='after')
    def validate_cost_breakdown(self) -> 'FinancialModel':
        """Validate cost breakdown sums correctly."""
        component_costs = (
            self.module_cost +
            self.inverter_cost +
            self.balance_of_system_cost +
            self.installation_cost
        )

        # Allow small tolerance for rounding
        if abs(component_costs - self.system_cost) > 0.01 * self.system_cost:
            raise ValueError(
                f"Cost breakdown mismatch: components sum to ${component_costs:.2f} "
                f"but system_cost is ${self.system_cost:.2f}"
            )

        return self

    def calculate_lcoe(self, annual_energy_kwh: float) -> float:
        """
        Calculate Levelized Cost of Energy (LCOE).

        Args:
            annual_energy_kwh: Annual energy production in kWh

        Returns:
            LCOE in USD/kWh
        """
        if annual_energy_kwh <= 0:
            raise ValueError("Annual energy must be positive")

        # Calculate total lifetime costs (discounted)
        total_costs = self.system_cost - self.incentives

        for year in range(1, self.system_lifetime + 1):
            om_cost_present_value = self.annual_om_cost / ((1 + self.discount_rate) ** year)
            total_costs += om_cost_present_value

        # Subtract salvage value
        salvage_pv = self.salvage_value / ((1 + self.discount_rate) ** self.system_lifetime)
        total_costs -= salvage_pv

        # Calculate total lifetime energy (discounted, with degradation)
        total_energy = 0.0
        for year in range(1, self.system_lifetime + 1):
            degradation_factor = (1 - self.degradation_rate) ** (year - 1)
            energy_this_year = annual_energy_kwh * degradation_factor
            energy_pv = energy_this_year / ((1 + self.discount_rate) ** year)
            total_energy += energy_pv

        lcoe = total_costs / total_energy
        self.lcoe = lcoe
        return lcoe

    def calculate_npv(self, annual_energy_kwh: float) -> float:
        """
        Calculate Net Present Value (NPV).

        Args:
            annual_energy_kwh: Annual energy production in kWh

        Returns:
            NPV in USD
        """
        if annual_energy_kwh <= 0:
            raise ValueError("Annual energy must be positive")

        # Initial investment (negative cash flow)
        npv = -(self.system_cost - self.incentives)

        # Annual cash flows
        for year in range(1, self.system_lifetime + 1):
            # Energy production with degradation
            degradation_factor = (1 - self.degradation_rate) ** (year - 1)
            energy_this_year = annual_energy_kwh * degradation_factor

            # Electricity rate with escalation
            rate_this_year = self.electricity_rate * ((1 + self.electricity_rate_escalation) ** (year - 1))

            # Revenue from energy
            revenue = energy_this_year * rate_this_year

            # Net cash flow
            cash_flow = revenue - self.annual_om_cost

            # Discount to present value
            pv_factor = 1 / ((1 + self.discount_rate) ** year)
            npv += cash_flow * pv_factor

        # Add salvage value
        salvage_pv = self.salvage_value / ((1 + self.discount_rate) ** self.system_lifetime)
        npv += salvage_pv

        self.npv = npv
        return npv

    def calculate_simple_payback(self, annual_energy_kwh: float) -> float:
        """
        Calculate simple payback period (without discounting).

        Args:
            annual_energy_kwh: Annual energy production in kWh

        Returns:
            Payback period in years
        """
        if annual_energy_kwh <= 0:
            raise ValueError("Annual energy must be positive")

        net_investment = self.system_cost - self.incentives
        annual_savings = annual_energy_kwh * self.electricity_rate - self.annual_om_cost

        if annual_savings <= 0:
            raise ValueError("Annual savings must be positive for payback calculation")

        payback = net_investment / annual_savings
        self.payback_period = payback
        return payback


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_monocrystalline_cell() -> Cell:
    """Create a default monocrystalline cell for testing/examples."""
    return Cell(
        technology=CellTechnology.MONOCRYSTALLINE,
        efficiency=0.225,
        area=0.0244,  # 156.75mm x 156.75mm
        thickness=180.0,  # µm
        power_output=5.5,
        voltage_at_max_power=0.55,
        current_at_max_power=10.0,
        open_circuit_voltage=0.66,
        short_circuit_current=10.5,
        fill_factor=0.796,
        temperature_coefficients=TemperatureCoefficients(
            power=-0.35,
            voltage=-0.27,
            current=0.05
        ),
        degradation_rate=0.005,
        manufacturing_cost=0.50
    )


def create_example_silicon_material() -> Material:
    """Create an example silicon material."""
    return Material(
        name="High-purity Polysilicon",
        material_type=MaterialType.SILICON,
        properties=MaterialProperties(
            density=2330.0,
            thermal_conductivity=150.0,
            specific_heat=700.0,
            melting_point=1414.0,
            recyclability_rate=0.95,
            embodied_energy=200.0,
            carbon_footprint=15.0,
            toxicity_score=1.0
        ),
        mass_per_module=2.5,
        cost_per_kg=15.0,
        supplier="Example Silicon Co.",
        supply_chain_risk=0.3
    )
