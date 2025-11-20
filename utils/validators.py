"""
Data validation models using Pydantic.

This module provides comprehensive data validation for all application inputs
including cell parameters, module specifications, system configurations, and more.
"""

from typing import Optional, List, Dict, Tuple, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import numpy as np


# ============================================================================
# MATERIAL AND CELL DESIGN VALIDATORS
# ============================================================================

class MaterialProperties(BaseModel):
    """Material properties for PV cells."""

    name: str = Field(..., description="Material name")
    bandgap: float = Field(..., gt=0, le=4.0, description="Bandgap energy in eV")
    efficiency: float = Field(..., gt=0, le=50, description="Cell efficiency in %")
    cost_per_wp: float = Field(..., gt=0, description="Cost per Watt-peak in $/Wp")
    degradation_rate: float = Field(..., ge=0, le=10, description="Annual degradation rate in %/year")
    recyclability: float = Field(..., ge=0, le=100, description="Recyclability percentage")
    density: float = Field(..., gt=0, description="Material density in kg/m³")
    thermal_conductivity: float = Field(..., gt=0, description="Thermal conductivity in W/(m·K)")
    temp_coefficient: float = Field(..., ge=-1.0, le=0, description="Temperature coefficient in %/°C")

    @field_validator('bandgap')
    @classmethod
    def validate_bandgap(cls, v: float) -> float:
        """Validate bandgap is within reasonable range for PV materials."""
        if not (0.5 <= v <= 3.5):
            raise ValueError("Bandgap must be between 0.5 and 3.5 eV for practical PV materials")
        return v


class CellDesignParameters(BaseModel):
    """Parameters for solar cell design."""

    material: str = Field(..., description="Primary material (c-Si, perovskite, CIGS, CdTe, tandem)")
    architecture: str = Field(..., description="Cell architecture (n-type, p-type, heterojunction, tandem)")
    thickness: float = Field(..., gt=0, le=500, description="Device thickness in micrometers")
    area: float = Field(..., gt=0, le=300, description="Cell area in cm²")
    substrate: str = Field(default="glass", description="Substrate material")
    front_contact: str = Field(default="ITO", description="Front contact material")
    rear_contact: str = Field(default="aluminum", description="Rear contact material")
    anti_reflection_coating: bool = Field(default=True, description="ARC present")
    passivation_layer: bool = Field(default=True, description="Passivation layer present")

    @field_validator('material')
    @classmethod
    def validate_material(cls, v: str) -> str:
        """Validate material selection."""
        valid_materials = {"c-Si", "perovskite", "CIGS", "CdTe", "tandem", "bifacial"}
        if v not in valid_materials:
            raise ValueError(f"Material must be one of {valid_materials}")
        return v


class SCAPSSimulationInput(BaseModel):
    """Input parameters for SCAPS-1D simulation."""

    cell_design: CellDesignParameters
    temperature: float = Field(default=298.15, gt=0, le=400, description="Temperature in Kelvin")
    irradiance: float = Field(default=1000, gt=0, le=2000, description="Irradiance in W/m²")
    spectrum: str = Field(default="AM1.5G", description="Solar spectrum")
    voltage_range: Tuple[float, float] = Field(default=(-0.5, 1.5), description="Voltage sweep range in V")
    voltage_steps: int = Field(default=100, gt=10, le=1000, description="Number of voltage steps")

    @field_validator('spectrum')
    @classmethod
    def validate_spectrum(cls, v: str) -> str:
        """Validate spectrum selection."""
        valid_spectra = {"AM0", "AM1.5G", "AM1.5D"}
        if v not in valid_spectra:
            raise ValueError(f"Spectrum must be one of {valid_spectra}")
        return v


# ============================================================================
# MODULE DESIGN VALIDATORS
# ============================================================================

class CTMLossFactors(BaseModel):
    """Cell-to-Module loss factors (k1-k24)."""

    k1_reflection: float = Field(default=2.5, ge=0, le=10, description="Reflection losses (%)")
    k2_soiling: float = Field(default=1.8, ge=0, le=10, description="Soiling losses (%)")
    k3_temperature: float = Field(default=3.2, ge=0, le=15, description="Temperature losses (%)")
    k4_series_resistance: float = Field(default=2.1, ge=0, le=5, description="Series resistance losses (%)")
    k5_mismatch: float = Field(default=1.5, ge=0, le=5, description="Mismatch losses (%)")
    k6_wiring: float = Field(default=0.8, ge=0, le=5, description="Wiring losses (%)")
    k7_shading: float = Field(default=0.0, ge=0, le=20, description="Shading losses (%)")
    k8_spectral: float = Field(default=1.5, ge=0, le=5, description="Spectral losses (%)")
    k9_iam: float = Field(default=2.0, ge=0, le=10, description="IAM losses (%)")
    k10_encapsulation: float = Field(default=1.0, ge=0, le=5, description="Cell encapsulation (%)")
    k11_glass_absorption: float = Field(default=1.5, ge=0, le=5, description="Front glass absorption (%)")
    k12_backsheet: float = Field(default=0.8, ge=0, le=3, description="Back sheet reflection (%)")
    k13_cell_spacing: float = Field(default=3.0, ge=0, le=10, description="Cell spacing losses (%)")
    k14_ribbon_shading: float = Field(default=2.0, ge=0, le=5, description="Ribbon shading (%)")
    k15_busbar_shading: float = Field(default=2.5, ge=0, le=5, description="Busbar shading (%)")
    k16_temp_non_uniformity: float = Field(default=1.0, ge=0, le=5, description="Temperature non-uniformity (%)")
    k17_degradation: float = Field(default=1.5, ge=0, le=5, description="Degradation losses (%)")
    k18_lid: float = Field(default=2.0, ge=0, le=10, description="Light-induced degradation (%)")
    k19_pid: float = Field(default=0.0, ge=0, le=20, description="Potential-induced degradation (%)")
    k20_quality: float = Field(default=1.0, ge=0, le=5, description="Quality losses (%)")
    k21_manufacturing: float = Field(default=1.0, ge=0, le=3, description="Manufacturing tolerances (%)")
    k22_optical: float = Field(default=1.5, ge=0, le=5, description="Optical losses (%)")
    k23_interconnection: float = Field(default=1.0, ge=0, le=3, description="Interconnection losses (%)")
    k24_other: float = Field(default=1.0, ge=0, le=5, description="Other module losses (%)")

    def total_loss(self) -> float:
        """Calculate total CTM loss percentage."""
        return sum([getattr(self, f) for f in self.model_fields if f.startswith('k')])

    def efficiency_factor(self) -> float:
        """Calculate efficiency multiplication factor (1 - total_loss/100)."""
        return 1.0 - (self.total_loss() / 100.0)


class ModuleSpecification(BaseModel):
    """PV module specifications."""

    model_name: str = Field(..., description="Module model name")
    manufacturer: str = Field(..., description="Manufacturer name")
    cell_type: str = Field(..., description="Cell technology")
    num_cells: int = Field(..., gt=0, le=200, description="Number of cells in module")
    cell_efficiency: float = Field(..., gt=0, le=30, description="Cell efficiency (%)")
    module_efficiency: float = Field(..., gt=0, le=25, description="Module efficiency (%)")
    rated_power: float = Field(..., gt=0, le=1000, description="Rated power in Watts")
    voc: float = Field(..., gt=0, le=100, description="Open-circuit voltage in V")
    isc: float = Field(..., gt=0, le=20, description="Short-circuit current in A")
    vmp: float = Field(..., gt=0, le=100, description="Maximum power voltage in V")
    imp: float = Field(..., gt=0, le=20, description="Maximum power current in A")
    temp_coeff_pmax: float = Field(..., ge=-1.0, le=0, description="Temperature coefficient of Pmax (%/°C)")
    temp_coeff_voc: float = Field(..., ge=-0.5, le=0, description="Temperature coefficient of Voc (%/°C)")
    temp_coeff_isc: float = Field(..., ge=0, le=0.2, description="Temperature coefficient of Isc (%/°C)")
    length: float = Field(..., gt=0, le=3000, description="Module length in mm")
    width: float = Field(..., gt=0, le=2000, description="Module width in mm")
    weight: float = Field(..., gt=0, le=50, description="Module weight in kg")
    ctm_losses: Optional[CTMLossFactors] = Field(default=None, description="CTM loss factors")

    @model_validator(mode='after')
    def validate_power_parameters(self) -> 'ModuleSpecification':
        """Validate that power parameters are consistent."""
        calculated_power = self.vmp * self.imp
        if abs(calculated_power - self.rated_power) > self.rated_power * 0.1:
            raise ValueError(f"Inconsistent power parameters: Vmp×Imp={calculated_power:.1f}W, Rated={self.rated_power}W")
        return self


# ============================================================================
# SYSTEM DESIGN VALIDATORS
# ============================================================================

class SystemConfiguration(BaseModel):
    """PV system configuration parameters."""

    system_name: str = Field(..., description="System identifier")
    capacity_dc: float = Field(..., gt=0, description="DC system capacity in kW")
    capacity_ac: float = Field(..., gt=0, description="AC system capacity in kW")
    num_modules: int = Field(..., gt=0, le=100000, description="Total number of modules")
    modules_per_string: int = Field(..., gt=0, le=50, description="Modules per string")
    num_strings: int = Field(..., gt=0, le=1000, description="Number of strings")
    inverter_type: Literal["string", "central", "micro", "hybrid"] = Field(..., description="Inverter type")
    inverter_efficiency: float = Field(..., gt=0.8, le=1.0, description="Inverter efficiency")
    mounting_type: Literal["fixed_tilt", "single_axis", "dual_axis", "rooftop"] = Field(..., description="Mounting configuration")
    tilt_angle: float = Field(..., ge=0, le=90, description="Module tilt angle in degrees")
    azimuth_angle: float = Field(..., ge=0, le=360, description="Azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)")
    dc_ac_ratio: float = Field(..., gt=0.8, le=2.0, description="DC to AC ratio")

    @model_validator(mode='after')
    def validate_system_sizing(self) -> 'SystemConfiguration':
        """Validate system sizing is consistent."""
        calculated_dc_ac = self.capacity_dc / self.capacity_ac
        if abs(calculated_dc_ac - self.dc_ac_ratio) > 0.1:
            raise ValueError(f"DC/AC ratio mismatch: calculated={calculated_dc_ac:.2f}, specified={self.dc_ac_ratio:.2f}")
        return self


class LocationData(BaseModel):
    """Geographic location and weather data."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    elevation: float = Field(default=0, ge=-500, le=9000, description="Elevation in meters")
    timezone: str = Field(..., description="Timezone (e.g., 'America/New_York')")
    climate_zone: Optional[Literal["tropical", "arid", "temperate", "continental", "polar"]] = Field(default=None)
    avg_ghi: Optional[float] = Field(default=None, ge=0, le=3000, description="Average GHI in kWh/m²/year")
    avg_dni: Optional[float] = Field(default=None, ge=0, le=3500, description="Average DNI in kWh/m²/year")
    avg_temperature: Optional[float] = Field(default=None, ge=-50, le=60, description="Average temperature in °C")


# ============================================================================
# MONITORING AND PERFORMANCE VALIDATORS
# ============================================================================

class PerformanceMetrics(BaseModel):
    """Real-time and historical performance metrics."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    power_dc: float = Field(..., ge=0, description="DC power in kW")
    power_ac: float = Field(..., ge=0, description="AC power in kW")
    voltage_dc: float = Field(..., ge=0, description="DC voltage in V")
    current_dc: float = Field(..., ge=0, description="DC current in A")
    energy_today: float = Field(..., ge=0, description="Energy produced today in kWh")
    energy_total: float = Field(..., ge=0, description="Total lifetime energy in kWh")
    performance_ratio: float = Field(..., ge=0, le=1.5, description="Performance ratio")
    system_efficiency: float = Field(..., ge=0, le=0.3, description="System efficiency")
    inverter_efficiency: float = Field(..., ge=0, le=1.0, description="Inverter efficiency")
    irradiance: Optional[float] = Field(default=None, ge=0, le=2000, description="Irradiance in W/m²")
    module_temperature: Optional[float] = Field(default=None, ge=-50, le=100, description="Module temperature in °C")
    ambient_temperature: Optional[float] = Field(default=None, ge=-50, le=60, description="Ambient temperature in °C")
    wind_speed: Optional[float] = Field(default=None, ge=0, le=100, description="Wind speed in m/s")


class FaultDetection(BaseModel):
    """Fault detection and diagnostics."""

    fault_id: str = Field(..., description="Unique fault identifier")
    timestamp: datetime = Field(..., description="Detection timestamp")
    fault_type: Literal["hotspot", "cell_crack", "bypass_diode", "soiling", "delamination", "pid", "other"] = Field(..., description="Fault type")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Fault severity")
    location: str = Field(..., description="Fault location (string, module, cell)")
    detection_method: Literal["thermal_imaging", "el_imaging", "iv_curve", "visual_inspection", "scada"] = Field(..., description="Detection method")
    power_loss_estimate: Optional[float] = Field(default=None, ge=0, le=100, description="Estimated power loss in %")
    temperature_delta: Optional[float] = Field(default=None, description="Temperature difference in °C (for hotspots)")
    recommended_action: Optional[str] = Field(default=None, description="Recommended corrective action")
    resolved: bool = Field(default=False, description="Resolution status")


class EnergyForecast(BaseModel):
    """Energy production forecast."""

    forecast_date: datetime = Field(..., description="Forecast date")
    forecast_horizon: int = Field(..., gt=0, le=30, description="Forecast horizon in days")
    method: Literal["statistical", "ml_ensemble", "prophet", "lstm", "hybrid"] = Field(..., description="Forecasting method")
    predicted_energy: List[float] = Field(..., description="Predicted daily energy in kWh")
    confidence_interval_lower: Optional[List[float]] = Field(default=None, description="Lower confidence bound")
    confidence_interval_upper: Optional[List[float]] = Field(default=None, description="Upper confidence bound")
    weather_features: Optional[Dict[str, List[float]]] = Field(default=None, description="Weather input features")

    @field_validator('predicted_energy')
    @classmethod
    def validate_forecast_length(cls, v: List[float], info) -> List[float]:
        """Validate forecast length matches horizon."""
        horizon = info.data.get('forecast_horizon')
        if horizon and len(v) != horizon:
            raise ValueError(f"Predicted energy length ({len(v)}) must match forecast horizon ({horizon})")
        return v


# ============================================================================
# CIRCULARITY AND LIFECYCLE VALIDATORS
# ============================================================================

class CircularityAssessment(BaseModel):
    """Circular economy assessment (3R: Reuse, Repair, Recycle)."""

    module_id: str = Field(..., description="Module identifier")
    age: float = Field(..., ge=0, le=50, description="Module age in years")
    remaining_capacity: float = Field(..., ge=0, le=100, description="Remaining capacity in % of nameplate")
    physical_condition: Literal["excellent", "good", "fair", "poor"] = Field(..., description="Physical condition")

    # Reuse assessment
    reuse_potential: float = Field(..., ge=0, le=100, description="Reuse potential score (%)")
    reuse_value: Optional[float] = Field(default=None, ge=0, description="Estimated reuse value in $")
    reuse_applications: Optional[List[str]] = Field(default=None, description="Suitable reuse applications")

    # Repair assessment
    repair_feasibility: bool = Field(..., description="Whether repair is feasible")
    repair_cost: Optional[float] = Field(default=None, ge=0, description="Estimated repair cost in $")
    repair_actions: Optional[List[str]] = Field(default=None, description="Required repair actions")
    life_extension: Optional[float] = Field(default=None, ge=0, le=20, description="Expected life extension in years")

    # Recycle assessment
    recycling_revenue: float = Field(..., ge=0, description="Expected recycling revenue in $")
    materials_recovered: Optional[Dict[str, float]] = Field(default=None, description="Recovered materials in kg")
    recycling_cost: Optional[float] = Field(default=None, ge=0, description="Recycling cost in $")

    # Overall circularity score
    circularity_score: float = Field(..., ge=0, le=100, description="Overall circularity score")

    @model_validator(mode='after')
    def validate_reuse_criteria(self) -> 'CircularityAssessment':
        """Validate reuse potential against criteria."""
        if self.remaining_capacity >= 80 and self.physical_condition in ["excellent", "good"] and self.age < 15:
            if self.reuse_potential < 60:
                raise ValueError("Reuse potential should be high given module condition")
        return self


class RevampRepowerPlan(BaseModel):
    """Revamp and repower planning."""

    plan_id: str = Field(..., description="Plan identifier")
    current_system: SystemConfiguration
    strategy: Literal["full_repower", "partial_repower", "revamp", "augmentation"] = Field(..., description="Strategy type")
    target_capacity: float = Field(..., gt=0, description="Target capacity in kW")
    modules_to_replace: int = Field(..., ge=0, description="Number of modules to replace")
    modules_to_retain: int = Field(..., ge=0, description="Number of modules to retain")
    estimated_cost: float = Field(..., gt=0, description="Estimated cost in $")
    expected_performance_gain: float = Field(..., ge=0, le=100, description="Expected performance gain in %")
    payback_period: float = Field(..., gt=0, le=50, description="Payback period in years")


# ============================================================================
# FINANCIAL VALIDATORS
# ============================================================================

class FinancialAnalysis(BaseModel):
    """Financial analysis and bankability assessment."""

    project_name: str = Field(..., description="Project name")
    total_capex: float = Field(..., gt=0, description="Total capital expenditure in $")
    annual_opex: float = Field(..., ge=0, description="Annual operating expenditure in $")
    electricity_price: float = Field(..., gt=0, le=1.0, description="Electricity price in $/kWh")
    discount_rate: float = Field(..., gt=0, le=0.3, description="Discount rate (WACC)")
    project_lifetime: int = Field(..., gt=0, le=50, description="Project lifetime in years")

    # Calculated metrics
    lcoe: Optional[float] = Field(default=None, ge=0, description="Levelized cost of energy in $/kWh")
    npv: Optional[float] = Field(default=None, description="Net present value in $")
    irr: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Internal rate of return")
    payback_period: Optional[float] = Field(default=None, ge=0, description="Simple payback period in years")
    equity_irr: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Equity IRR")
    debt_service_coverage: Optional[float] = Field(default=None, ge=0, description="Debt service coverage ratio")

    # Incentives and tax benefits
    federal_itc: float = Field(default=0.30, ge=0, le=1.0, description="Federal Investment Tax Credit")
    state_incentives: Optional[float] = Field(default=None, ge=0, description="State incentives in $")
    depreciation_benefit: Optional[float] = Field(default=None, ge=0, description="Depreciation benefit in $")


class HybridSystemDesign(BaseModel):
    """Hybrid energy system design (PV + Storage/Wind/H2)."""

    system_name: str = Field(..., description="System name")
    pv_capacity: float = Field(..., gt=0, description="PV capacity in kW")

    # Battery storage
    battery_enabled: bool = Field(default=False, description="Battery storage included")
    battery_capacity: Optional[float] = Field(default=None, ge=0, description="Battery capacity in kWh")
    battery_power: Optional[float] = Field(default=None, ge=0, description="Battery power in kW")
    battery_type: Optional[Literal["lithium_ion", "lead_acid", "flow_battery"]] = Field(default=None)

    # Wind
    wind_enabled: bool = Field(default=False, description="Wind turbine included")
    wind_capacity: Optional[float] = Field(default=None, ge=0, description="Wind capacity in kW")

    # Hydrogen
    hydrogen_enabled: bool = Field(default=False, description="Hydrogen system included")
    electrolyzer_capacity: Optional[float] = Field(default=None, ge=0, description="Electrolyzer capacity in kW")
    h2_storage_capacity: Optional[float] = Field(default=None, ge=0, description="H2 storage in kg")
    fuel_cell_capacity: Optional[float] = Field(default=None, ge=0, description="Fuel cell capacity in kW")

    # System metrics
    self_sufficiency: Optional[float] = Field(default=None, ge=0, le=1.0, description="Energy self-sufficiency ratio")
    self_consumption: Optional[float] = Field(default=None, ge=0, le=1.0, description="Self-consumption ratio")

    @model_validator(mode='after')
    def validate_hybrid_components(self) -> 'HybridSystemDesign':
        """Validate hybrid component configurations."""
        if self.battery_enabled and (self.battery_capacity is None or self.battery_power is None):
            raise ValueError("Battery capacity and power must be specified when battery is enabled")
        if self.wind_enabled and self.wind_capacity is None:
            raise ValueError("Wind capacity must be specified when wind is enabled")
        if self.hydrogen_enabled and (self.electrolyzer_capacity is None or self.h2_storage_capacity is None):
            raise ValueError("Electrolyzer and H2 storage must be specified when hydrogen is enabled")
        return self


# ============================================================================
# IEC TESTING VALIDATORS
# ============================================================================

class IECTestResult(BaseModel):
    """IEC standard test result."""

    test_id: str = Field(..., description="Test identifier")
    standard: Literal["IEC_61215", "IEC_61730", "IEC_63202", "IEC_63209", "IEC_TS_63279"] = Field(..., description="IEC standard")
    test_name: str = Field(..., description="Test name")
    test_date: datetime = Field(..., description="Test date")
    module_id: str = Field(..., description="Module under test")

    # Test parameters
    test_conditions: Dict[str, float] = Field(..., description="Test conditions")

    # Results
    passed: bool = Field(..., description="Pass/fail status")
    measured_values: Dict[str, float] = Field(..., description="Measured values")
    acceptance_criteria: Dict[str, Tuple[float, float]] = Field(..., description="Acceptance ranges")
    deviations: Optional[List[str]] = Field(default=None, description="Deviations from standard")

    # Certification
    certified: bool = Field(default=False, description="Certification status")
    certificate_number: Optional[str] = Field(default=None, description="Certificate number")
    expiry_date: Optional[datetime] = Field(default=None, description="Certificate expiry")
