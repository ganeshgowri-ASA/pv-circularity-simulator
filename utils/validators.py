"""
Data Validation Models for PV Circularity Simulator

This module contains Pydantic models for validating data inputs across
all functional branches of the application.

Author: PV Circularity Simulator Team
Version: 1.0 (71 Sessions Integrated)
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# BRANCH 01: MATERIALS DATABASE
# ============================================================================

class Material(BaseModel):
    """Material specification and properties."""

    name: str = Field(..., description="Material name")
    efficiency: float = Field(..., ge=5.0, le=50.0, description="Efficiency (%)")
    cost_per_wp: float = Field(..., ge=0.1, le=5.0, description="Cost ($/Wp)")
    degradation_rate: float = Field(..., ge=0.1, le=5.0, description="Annual degradation (%/yr)")
    recyclability: int = Field(..., ge=0, le=100, description="Recyclability score")
    bandgap_ev: Optional[float] = Field(None, ge=0.5, le=3.5, description="Bandgap (eV)")
    temp_coefficient: Optional[float] = Field(None, ge=-1.0, le=0.0, description="Temperature coefficient (%/°C)")
    density: Optional[float] = Field(None, description="Density (g/cm³)")
    thermal_conductivity: Optional[float] = Field(None, description="Thermal conductivity (W/m·K)")
    lifespan_years: int = Field(25, ge=10, le=50, description="Expected lifespan (years)")
    carbon_footprint: Optional[float] = Field(None, description="Carbon footprint (kg CO2/kWp)")

    class Config:
        validate_assignment = True


# ============================================================================
# BRANCH 02: CELL DESIGN (SCAPS-1D)
# ============================================================================

class CellDesign(BaseModel):
    """Solar cell design parameters."""

    substrate: str = Field(..., description="Substrate material")
    thickness_um: float = Field(..., ge=0.1, le=10.0, description="Device thickness (μm)")
    architecture: str = Field("n-type", description="Cell architecture")
    voc_mv: float = Field(..., ge=400, le=1000, description="Open-circuit voltage (mV)")
    jsc_ma_cm2: float = Field(..., ge=20, le=50, description="Short-circuit current density (mA/cm²)")
    fill_factor: float = Field(..., ge=0.6, le=0.9, description="Fill factor")
    area_cm2: float = Field(156.75, ge=100, le=300, description="Cell area (cm²)")
    simulation_temp_k: float = Field(300, ge=250, le=400, description="Simulation temperature (K)")

    @field_validator('fill_factor')
    @classmethod
    def validate_fill_factor(cls, v):
        """Validate fill factor is realistic."""
        if v < 0.6 or v > 0.9:
            raise ValueError("Fill factor must be between 0.6 and 0.9")
        return v

    @property
    def efficiency(self) -> float:
        """Calculate cell efficiency."""
        pmax = self.voc_mv * self.jsc_ma_cm2 * self.fill_factor / 1000
        irradiance = 1000  # W/m²
        return (pmax / (irradiance * self.area_cm2 / 10000)) * 100


# ============================================================================
# BRANCH 03: MODULE DESIGN (CTM)
# ============================================================================

class CTMLoss(BaseModel):
    """Cell-to-Module loss factor."""

    factor_id: str = Field(..., pattern=r"k\d+", description="Factor ID (k1-k24)")
    description: str = Field(..., description="Loss description")
    loss_pct: float = Field(..., ge=0.0, le=10.0, description="Loss percentage")
    category: str = Field("Optical", description="Loss category")


class ModuleDesign(BaseModel):
    """PV module design specification."""

    num_cells: int = Field(60, ge=36, le=144, description="Number of cells")
    cell_efficiency: float = Field(..., ge=15.0, le=26.0, description="Cell efficiency (%)")
    module_area_m2: float = Field(1.64, ge=1.0, le=3.0, description="Module area (m²)")
    ctm_losses: List[CTMLoss] = Field(default_factory=list, description="CTM loss factors")
    bypass_diodes: int = Field(3, ge=2, le=6, description="Number of bypass diodes")
    bus_bars: int = Field(5, ge=3, le=12, description="Number of bus bars")
    encapsulation: str = Field("EVA", description="Encapsulation material")
    backsheet: str = Field("White", description="Backsheet type")
    frame_material: str = Field("Aluminum", description="Frame material")

    @property
    def total_ctm_loss(self) -> float:
        """Calculate total CTM loss."""
        return sum(loss.loss_pct for loss in self.ctm_losses)

    @property
    def module_efficiency(self) -> float:
        """Calculate module efficiency after CTM losses."""
        return self.cell_efficiency * (1 - self.total_ctm_loss / 100)

    @property
    def rated_power_w(self) -> float:
        """Calculate module rated power."""
        return self.module_efficiency * self.module_area_m2 * 1000 / 100


# ============================================================================
# BRANCH 04: IEC TESTING
# ============================================================================

class IECTest(BaseModel):
    """IEC test specification and results."""

    standard: str = Field(..., pattern=r"IEC_\d+", description="IEC standard")
    test_name: str = Field(..., description="Test name")
    status: str = Field("Pending", description="Test status")
    pass_criteria: str = Field(..., description="Pass criteria")
    measured_value: Optional[float] = Field(None, description="Measured value")
    test_date: Optional[datetime] = Field(None, description="Test date")
    duration_hours: float = Field(..., ge=0, description="Test duration (hours)")
    result: Optional[str] = Field(None, description="Pass/Fail/In Progress")

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate test status."""
        allowed = ["Pending", "In Progress", "Completed", "Failed"]
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return v


# ============================================================================
# BRANCH 05: SYSTEM DESIGN
# ============================================================================

class SystemDesign(BaseModel):
    """PV system design parameters."""

    capacity_kw: float = Field(..., ge=1.0, le=10000.0, description="System capacity (kW)")
    num_modules: int = Field(..., ge=4, description="Number of modules")
    module_power_w: float = Field(..., ge=200, le=700, description="Module power (W)")
    num_strings: int = Field(..., ge=1, le=100, description="Number of strings")
    modules_per_string: int = Field(..., ge=4, le=30, description="Modules per string")
    inverter_type: str = Field(..., description="Inverter type")
    inverter_capacity_kw: float = Field(..., ge=1.0, description="Inverter capacity (kW)")
    mounting_type: str = Field("Fixed Tilt", description="Mounting type")
    tilt_angle: float = Field(..., ge=0, le=90, description="Tilt angle (degrees)")
    azimuth: float = Field(180, ge=0, le=360, description="Azimuth (degrees)")
    location: str = Field(..., description="Installation location")

    @model_validator(mode='after')
    def validate_system_sizing(self):
        """Validate system sizing is consistent."""
        if self.num_modules != self.num_strings * self.modules_per_string:
            raise ValueError("num_modules must equal num_strings × modules_per_string")

        return self

    @property
    def dc_ac_ratio(self) -> float:
        """Calculate DC/AC ratio."""
        return self.capacity_kw / self.inverter_capacity_kw


# ============================================================================
# BRANCH 06: WEATHER EYA
# ============================================================================

class WeatherData(BaseModel):
    """Weather data for energy yield assessment."""

    location: str = Field(..., description="Location name")
    annual_ghi_kwh_m2: float = Field(..., ge=800, le=3000, description="Annual GHI (kWh/m²)")
    avg_temp_c: float = Field(..., ge=-20, le=50, description="Average temperature (°C)")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    wind_speed_ms: float = Field(..., ge=0, le=20, description="Average wind speed (m/s)")
    soiling_rate_pct: float = Field(0.15, ge=0, le=1, description="Soiling rate (%/day)")
    rainfall_mm: float = Field(..., ge=0, description="Annual rainfall (mm)")
    snow_days: int = Field(0, ge=0, le=365, description="Snow days per year")

    @field_validator('humidity_pct')
    @classmethod
    def validate_humidity(cls, v):
        """Validate humidity is in valid range."""
        if v < 0 or v > 100:
            raise ValueError("Humidity must be between 0 and 100%")
        return v


# ============================================================================
# BRANCH 07: PERFORMANCE MONITORING
# ============================================================================

class PerformanceData(BaseModel):
    """Real-time performance monitoring data."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    dc_power_kw: float = Field(..., ge=0, description="DC power (kW)")
    ac_power_kw: float = Field(..., ge=0, description="AC power (kW)")
    irradiance_w_m2: float = Field(..., ge=0, le=1500, description="Irradiance (W/m²)")
    module_temp_c: float = Field(..., ge=-40, le=100, description="Module temperature (°C)")
    ambient_temp_c: float = Field(..., ge=-40, le=60, description="Ambient temperature (°C)")
    wind_speed_ms: float = Field(..., ge=0, le=30, description="Wind speed (m/s)")
    energy_today_kwh: float = Field(..., ge=0, description="Energy today (kWh)")
    performance_ratio: Optional[float] = Field(None, ge=0, le=1.5, description="Performance ratio")

    @property
    def inverter_efficiency(self) -> float:
        """Calculate inverter efficiency."""
        if self.dc_power_kw == 0:
            return 0
        return min(self.ac_power_kw / self.dc_power_kw, 1.0)


# ============================================================================
# BRANCH 08: FAULT DIAGNOSTICS
# ============================================================================

class FaultDetection(BaseModel):
    """Fault detection and diagnostics."""

    fault_type: str = Field(..., description="Type of fault detected")
    severity: str = Field(..., description="Severity level")
    detection_method: str = Field(..., description="Detection method")
    location: str = Field(..., description="Fault location")
    timestamp: datetime = Field(..., description="Detection timestamp")
    power_loss_pct: float = Field(..., ge=0, le=100, description="Estimated power loss (%)")
    recommended_action: str = Field(..., description="Recommended corrective action")
    status: str = Field("Open", description="Fault status")

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        """Validate severity level."""
        allowed = ["Low", "Medium", "High", "Critical"]
        if v not in allowed:
            raise ValueError(f"Severity must be one of {allowed}")
        return v


# ============================================================================
# BRANCH 09: ENERGY FORECASTING
# ============================================================================

class EnergyForecast(BaseModel):
    """Energy production forecast."""

    forecast_date: datetime = Field(..., description="Forecast date")
    forecast_kwh: float = Field(..., ge=0, description="Forecasted energy (kWh)")
    confidence_interval_lower: float = Field(..., ge=0, description="Lower CI (kWh)")
    confidence_interval_upper: float = Field(..., ge=0, description="Upper CI (kWh)")
    model_type: str = Field("Ensemble", description="Forecasting model")
    irradiance_forecast_w_m2: float = Field(..., ge=0, description="Irradiance forecast (W/m²)")
    temp_forecast_c: float = Field(..., description="Temperature forecast (°C)")

    @model_validator(mode='after')
    def validate_confidence_interval(self):
        """Validate confidence interval bounds."""
        if self.confidence_interval_lower > self.forecast_kwh or self.confidence_interval_upper < self.forecast_kwh:
            raise ValueError("Forecast must be within confidence interval")

        return self


# ============================================================================
# BRANCH 10: REVAMP PLANNING
# ============================================================================

class RevampOption(BaseModel):
    """System revamp and retrofit option."""

    option_name: str = Field(..., description="Revamp option name")
    cost_total: float = Field(..., ge=0, description="Total cost ($)")
    efficiency_gain_pct: float = Field(..., ge=0, le=100, description="Efficiency gain (%)")
    lifespan_extension_years: int = Field(..., ge=0, le=25, description="Lifespan extension (years)")
    payback_period_years: float = Field(..., ge=0, description="Payback period (years)")
    npv: float = Field(..., description="Net present value ($)")
    irr_pct: float = Field(..., description="Internal rate of return (%)")
    scope_of_work: str = Field(..., description="Scope of work")

    @field_validator('irr_pct')
    @classmethod
    def validate_irr(cls, v):
        """Validate IRR is reasonable."""
        if v < -50 or v > 100:
            raise ValueError("IRR must be between -50% and 100%")
        return v


# ============================================================================
# BRANCH 11: CIRCULARITY 3R (Reduce, Reuse, Recycle)
# ============================================================================

class CircularityAssessment(BaseModel):
    """Circularity assessment for modules."""

    module_id: str = Field(..., description="Module identifier")
    age_years: float = Field(..., ge=0, le=50, description="Module age (years)")
    reuse_potential_pct: float = Field(..., ge=0, le=100, description="Reuse potential (%)")
    repair_value_usd: float = Field(..., ge=0, description="Repair value ($)")
    recycling_revenue_usd: float = Field(..., ge=0, description="Recycling revenue ($)")
    material_recovery_rate: Dict[str, float] = Field(default_factory=dict, description="Material recovery rates")
    recycling_process: str = Field("Mechanical", description="Recycling process")
    circularity_score: float = Field(..., ge=0, le=100, description="Overall circularity score")
    environmental_impact_kg_co2: float = Field(..., description="Environmental impact (kg CO2)")

    @field_validator('circularity_score')
    @classmethod
    def validate_score(cls, v):
        """Validate circularity score."""
        if v < 0 or v > 100:
            raise ValueError("Circularity score must be between 0 and 100")
        return v


# ============================================================================
# BRANCH 12: HYBRID ENERGY SYSTEMS
# ============================================================================

class HybridSystem(BaseModel):
    """Hybrid PV + Battery Energy Storage System."""

    pv_capacity_kw: float = Field(..., ge=1, description="PV capacity (kW)")
    battery_capacity_kwh: float = Field(..., ge=1, description="Battery capacity (kWh)")
    battery_type: str = Field(..., description="Battery technology")
    battery_power_kw: float = Field(..., ge=1, description="Battery power rating (kW)")
    roundtrip_efficiency_pct: float = Field(..., ge=70, le=99, description="Roundtrip efficiency (%)")
    cycle_life: int = Field(..., ge=1000, description="Expected cycle life")
    depth_of_discharge_pct: float = Field(..., ge=50, le=100, description="Depth of discharge (%)")
    control_strategy: str = Field("Peak Shaving", description="Control strategy")

    @property
    def usable_capacity_kwh(self) -> float:
        """Calculate usable battery capacity."""
        return self.battery_capacity_kwh * self.depth_of_discharge_pct / 100


# ============================================================================
# BRANCH 13: FINANCIAL ANALYSIS
# ============================================================================

class FinancialModel(BaseModel):
    """Financial analysis model."""

    capex_usd: float = Field(..., ge=0, description="Capital expenditure ($)")
    opex_annual_usd: float = Field(..., ge=0, description="Annual operating expense ($)")
    energy_price_kwh: float = Field(..., ge=0, le=1, description="Energy price ($/kWh)")
    discount_rate_pct: float = Field(..., ge=0, le=30, description="Discount rate (%)")
    project_lifetime_years: int = Field(25, ge=10, le=50, description="Project lifetime (years)")
    incentives_usd: float = Field(0, ge=0, description="Incentives and subsidies ($)")
    lcoe_usd_kwh: Optional[float] = Field(None, description="Levelized cost of energy ($/kWh)")
    npv_usd: Optional[float] = Field(None, description="Net present value ($)")
    irr_pct: Optional[float] = Field(None, description="Internal rate of return (%)")
    payback_years: Optional[float] = Field(None, description="Simple payback period (years)")

    @field_validator('discount_rate_pct')
    @classmethod
    def validate_discount_rate(cls, v):
        """Validate discount rate is reasonable."""
        if v < 0 or v > 30:
            raise ValueError("Discount rate must be between 0% and 30%")
        return v


# ============================================================================
# BRANCH 14: INFRASTRUCTURE
# ============================================================================

class Infrastructure(BaseModel):
    """System infrastructure specification."""

    site_name: str = Field(..., description="Site name")
    grid_connection_kva: float = Field(..., ge=0, description="Grid connection capacity (kVA)")
    transformer_capacity_kva: float = Field(..., ge=0, description="Transformer capacity (kVA)")
    cable_type: str = Field(..., description="Cable type")
    cable_length_m: float = Field(..., ge=0, description="Total cable length (m)")
    monitoring_system: str = Field(..., description="Monitoring system type")
    scada_installed: bool = Field(False, description="SCADA system installed")
    communication_protocol: str = Field("Modbus", description="Communication protocol")
    metering_type: str = Field("Bidirectional", description="Metering type")

    @field_validator('grid_connection_kva')
    @classmethod
    def validate_grid_connection(cls, v):
        """Validate grid connection capacity."""
        if v <= 0:
            raise ValueError("Grid connection capacity must be positive")
        return v


# ============================================================================
# BRANCH 15: APPLICATION CONFIGURATION
# ============================================================================

class AppConfig(BaseModel):
    """Application configuration settings."""

    user_name: str = Field(..., description="User name")
    organization: str = Field(..., description="Organization name")
    timezone: str = Field("UTC", description="Timezone")
    units: str = Field("Metric", description="Unit system")
    data_update_interval_sec: int = Field(60, ge=1, le=3600, description="Data update interval (seconds)")
    export_format: str = Field("CSV", description="Default export format")
    theme: str = Field("Light", description="UI theme")
    notifications_enabled: bool = Field(True, description="Enable notifications")

    @field_validator('units')
    @classmethod
    def validate_units(cls, v):
        """Validate unit system."""
        allowed = ["Metric", "Imperial"]
        if v not in allowed:
            raise ValueError(f"Units must be one of {allowed}")
        return v


# ============================================================================
# SHARED VALIDATORS
# ============================================================================

class DateRange(BaseModel):
    """Date range for analysis."""

    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")

    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate end date is after start date."""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")

        return self


class GeoLocation(BaseModel):
    """Geographic location."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    altitude_m: float = Field(0, description="Altitude (m)")

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v):
        """Validate latitude range."""
        if v < -90 or v > 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_data(model: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """
    Validate data against a Pydantic model.

    Args:
        model: Pydantic model class
        data: Data dictionary to validate

    Returns:
        Validated model instance

    Raises:
        ValidationError: If validation fails
    """
    return model(**data)


def validate_batch(model: type[BaseModel], data_list: List[Dict[str, Any]]) -> List[BaseModel]:
    """
    Validate a batch of data against a Pydantic model.

    Args:
        model: Pydantic model class
        data_list: List of data dictionaries to validate

    Returns:
        List of validated model instances

    Raises:
        ValidationError: If any validation fails
    """
    return [model(**data) for data in data_list]
