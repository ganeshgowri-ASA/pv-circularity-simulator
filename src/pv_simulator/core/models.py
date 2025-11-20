"""
Pydantic data models for PV Circularity Simulator.

This module defines all data structures used throughout the simulator,
with validation and type safety provided by Pydantic.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TurbineType(str, Enum):
    """Wind turbine type classifications."""

    HORIZONTAL_AXIS = "horizontal_axis"
    VERTICAL_AXIS = "vertical_axis"
    OFFSHORE = "offshore"
    ONSHORE = "onshore"


class OptimizationObjective(str, Enum):
    """Optimization objectives for hybrid systems."""

    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_ENERGY = "maximize_energy"
    MINIMIZE_CURTAILMENT = "minimize_curtailment"
    MAXIMIZE_CAPACITY_FACTOR = "maximize_capacity_factor"


class WindResourceData(BaseModel):
    """Wind resource assessment data.

    Attributes:
        site_id: Unique identifier for the assessment site
        latitude: Site latitude in decimal degrees
        longitude: Site longitude in decimal degrees
        elevation_m: Site elevation above sea level in meters
        wind_speeds_ms: Time series of wind speeds in m/s
        wind_directions_deg: Time series of wind directions in degrees
        air_density_kgm3: Air density in kg/m³
        temperature_c: Temperature in Celsius
        pressure_pa: Atmospheric pressure in Pascals
        measurement_height_m: Height of wind speed measurements in meters
        assessment_period_days: Duration of assessment period in days
        data_quality_score: Quality score (0-1) of measured data
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    site_id: str = Field(..., description="Unique site identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    elevation_m: float = Field(..., ge=0, description="Elevation in meters")
    wind_speeds_ms: List[float] = Field(..., min_length=1, description="Wind speeds in m/s")
    wind_directions_deg: List[float] = Field(..., min_length=1, description="Wind directions in degrees")
    air_density_kgm3: float = Field(default=1.225, gt=0, description="Air density in kg/m³")
    temperature_c: Optional[float] = Field(default=None, description="Temperature in Celsius")
    pressure_pa: Optional[float] = Field(default=None, gt=0, description="Pressure in Pascals")
    measurement_height_m: float = Field(..., gt=0, description="Measurement height in meters")
    assessment_period_days: int = Field(..., gt=0, description="Assessment period in days")
    data_quality_score: float = Field(default=1.0, ge=0, le=1, description="Data quality score")

    @field_validator('wind_directions_deg')
    @classmethod
    def validate_wind_directions(cls, v: List[float]) -> List[float]:
        """Ensure wind directions are in valid range [0, 360)."""
        for direction in v:
            if not 0 <= direction < 360:
                raise ValueError(f"Wind direction {direction} must be in range [0, 360)")
        return v


class WindResourceAssessment(BaseModel):
    """Results from wind resource assessment analysis.

    Attributes:
        mean_wind_speed_ms: Mean wind speed at hub height in m/s
        weibull_k: Weibull shape parameter
        weibull_c: Weibull scale parameter in m/s
        wind_power_density_wm2: Wind power density in W/m²
        turbulence_intensity: Turbulence intensity ratio (0-1)
        wind_shear_exponent: Wind shear exponent (alpha)
        prevailing_direction_deg: Prevailing wind direction in degrees
        capacity_factor_estimate: Estimated capacity factor (0-1)
        annual_energy_potential_mwh: Estimated annual energy potential in MWh
    """

    model_config = ConfigDict(frozen=False)

    mean_wind_speed_ms: float = Field(..., gt=0, description="Mean wind speed in m/s")
    weibull_k: float = Field(..., gt=0, description="Weibull shape parameter")
    weibull_c: float = Field(..., gt=0, description="Weibull scale parameter in m/s")
    wind_power_density_wm2: float = Field(..., ge=0, description="Wind power density in W/m²")
    turbulence_intensity: float = Field(..., ge=0, le=1, description="Turbulence intensity")
    wind_shear_exponent: float = Field(default=0.14, ge=0, description="Wind shear exponent")
    prevailing_direction_deg: float = Field(..., ge=0, lt=360, description="Prevailing direction")
    capacity_factor_estimate: float = Field(..., ge=0, le=1, description="Capacity factor")
    annual_energy_potential_mwh: float = Field(..., ge=0, description="Annual energy in MWh")


class TurbineSpecifications(BaseModel):
    """Wind turbine technical specifications.

    Attributes:
        turbine_id: Unique turbine identifier
        manufacturer: Turbine manufacturer name
        model: Turbine model designation
        rated_power_kw: Rated power output in kW
        rotor_diameter_m: Rotor diameter in meters
        hub_height_m: Hub height above ground in meters
        cut_in_speed_ms: Cut-in wind speed in m/s
        rated_speed_ms: Rated wind speed in m/s
        cut_out_speed_ms: Cut-out wind speed in m/s
        power_curve_speeds_ms: Wind speeds for power curve in m/s
        power_curve_kw: Power outputs for power curve in kW
        turbine_type: Type of wind turbine
        efficiency: Overall turbine efficiency (0-1)
    """

    model_config = ConfigDict(frozen=False)

    turbine_id: str = Field(..., description="Unique turbine identifier")
    manufacturer: str = Field(..., min_length=1, description="Manufacturer name")
    model: str = Field(..., min_length=1, description="Model designation")
    rated_power_kw: float = Field(..., gt=0, description="Rated power in kW")
    rotor_diameter_m: float = Field(..., gt=0, description="Rotor diameter in meters")
    hub_height_m: float = Field(..., gt=0, description="Hub height in meters")
    cut_in_speed_ms: float = Field(..., gt=0, description="Cut-in speed in m/s")
    rated_speed_ms: float = Field(..., gt=0, description="Rated speed in m/s")
    cut_out_speed_ms: float = Field(..., gt=0, description="Cut-out speed in m/s")
    power_curve_speeds_ms: List[float] = Field(..., min_length=2, description="Power curve speeds")
    power_curve_kw: List[float] = Field(..., min_length=2, description="Power curve outputs")
    turbine_type: TurbineType = Field(default=TurbineType.ONSHORE, description="Turbine type")
    efficiency: float = Field(default=0.95, ge=0, le=1, description="Overall efficiency")

    @field_validator('power_curve_kw', 'power_curve_speeds_ms')
    @classmethod
    def validate_power_curve_lengths(cls, v: List[float], info) -> List[float]:
        """Ensure power curve arrays have matching lengths."""
        if info.data.get('power_curve_speeds_ms') and info.data.get('power_curve_kw'):
            if len(info.data['power_curve_speeds_ms']) != len(info.data['power_curve_kw']):
                raise ValueError("Power curve speeds and outputs must have same length")
        return v


class TurbinePerformance(BaseModel):
    """Wind turbine performance modeling results.

    Attributes:
        turbine_id: Turbine identifier
        capacity_factor: Calculated capacity factor (0-1)
        annual_energy_production_mwh: Annual energy production in MWh
        power_output_timeseries_kw: Time series of power output in kW
        availability_factor: Turbine availability factor (0-1)
        wake_losses_percent: Wake effect losses in percent
        electrical_losses_percent: Electrical system losses in percent
        environmental_losses_percent: Environmental losses in percent
        net_capacity_factor: Net capacity factor after all losses (0-1)
    """

    model_config = ConfigDict(frozen=False)

    turbine_id: str = Field(..., description="Turbine identifier")
    capacity_factor: float = Field(..., ge=0, le=1, description="Capacity factor")
    annual_energy_production_mwh: float = Field(..., ge=0, description="Annual energy in MWh")
    power_output_timeseries_kw: List[float] = Field(default_factory=list, description="Power timeseries")
    availability_factor: float = Field(default=0.97, ge=0, le=1, description="Availability factor")
    wake_losses_percent: float = Field(default=0.0, ge=0, le=100, description="Wake losses")
    electrical_losses_percent: float = Field(default=2.0, ge=0, le=100, description="Electrical losses")
    environmental_losses_percent: float = Field(default=1.0, ge=0, le=100, description="Environmental losses")
    net_capacity_factor: float = Field(..., ge=0, le=1, description="Net capacity factor")


class PVSystemConfig(BaseModel):
    """Photovoltaic system configuration.

    Attributes:
        system_id: Unique system identifier
        capacity_mw: System capacity in MW
        module_efficiency: Module efficiency (0-1)
        inverter_efficiency: Inverter efficiency (0-1)
        tilt_angle_deg: Panel tilt angle in degrees
        azimuth_deg: Panel azimuth in degrees
        temperature_coefficient: Temperature coefficient per °C
    """

    model_config = ConfigDict(frozen=False)

    system_id: str = Field(..., description="System identifier")
    capacity_mw: float = Field(..., gt=0, description="Capacity in MW")
    module_efficiency: float = Field(default=0.20, ge=0, le=1, description="Module efficiency")
    inverter_efficiency: float = Field(default=0.98, ge=0, le=1, description="Inverter efficiency")
    tilt_angle_deg: float = Field(default=30.0, ge=0, le=90, description="Tilt angle")
    azimuth_deg: float = Field(default=180.0, ge=0, lt=360, description="Azimuth")
    temperature_coefficient: float = Field(default=-0.004, description="Temp coefficient")


class HybridSystemConfig(BaseModel):
    """Hybrid wind-PV system configuration.

    Attributes:
        system_id: Unique hybrid system identifier
        site_name: Name of installation site
        pv_capacity_mw: PV capacity in MW
        wind_capacity_mw: Wind capacity in MW
        num_turbines: Number of wind turbines
        pv_system: PV system configuration
        turbine_specs: Wind turbine specifications
        shared_infrastructure: Whether systems share infrastructure
        storage_capacity_mwh: Optional battery storage capacity in MWh
        grid_connection_capacity_mw: Grid connection capacity in MW
    """

    model_config = ConfigDict(frozen=False)

    system_id: str = Field(..., description="Hybrid system identifier")
    site_name: str = Field(..., min_length=1, description="Site name")
    pv_capacity_mw: float = Field(..., ge=0, description="PV capacity in MW")
    wind_capacity_mw: float = Field(..., ge=0, description="Wind capacity in MW")
    num_turbines: int = Field(..., gt=0, description="Number of turbines")
    pv_system: PVSystemConfig = Field(..., description="PV system configuration")
    turbine_specs: TurbineSpecifications = Field(..., description="Turbine specifications")
    shared_infrastructure: bool = Field(default=True, description="Shared infrastructure")
    storage_capacity_mwh: Optional[float] = Field(default=None, ge=0, description="Storage capacity")
    grid_connection_capacity_mw: float = Field(..., gt=0, description="Grid connection capacity")

    @field_validator('wind_capacity_mw')
    @classmethod
    def validate_wind_capacity(cls, v: float, info) -> float:
        """Ensure wind capacity matches turbine configuration."""
        if 'num_turbines' in info.data and 'turbine_specs' in info.data:
            expected = info.data['num_turbines'] * info.data['turbine_specs'].rated_power_kw / 1000
            # Allow 1% tolerance
            if abs(v - expected) > expected * 0.01:
                raise ValueError(
                    f"Wind capacity {v} MW doesn't match turbine config {expected} MW"
                )
        return v


class HybridOptimizationResult(BaseModel):
    """Results from hybrid system optimization.

    Attributes:
        optimal_pv_capacity_mw: Optimized PV capacity in MW
        optimal_wind_capacity_mw: Optimized wind capacity in MW
        optimal_storage_capacity_mwh: Optimized storage capacity in MWh
        objective_value: Value of optimization objective
        total_annual_energy_mwh: Total annual energy production in MWh
        capacity_factor_combined: Combined capacity factor (0-1)
        curtailment_percent: Energy curtailment percentage
        levelized_cost_of_energy: LCOE in $/MWh
        optimization_objective: Objective function used
        convergence_status: Whether optimization converged
        iterations: Number of optimization iterations
    """

    model_config = ConfigDict(frozen=False)

    optimal_pv_capacity_mw: float = Field(..., ge=0, description="Optimal PV capacity")
    optimal_wind_capacity_mw: float = Field(..., ge=0, description="Optimal wind capacity")
    optimal_storage_capacity_mwh: Optional[float] = Field(default=None, ge=0, description="Optimal storage")
    objective_value: float = Field(..., description="Objective function value")
    total_annual_energy_mwh: float = Field(..., ge=0, description="Total annual energy")
    capacity_factor_combined: float = Field(..., ge=0, le=1, description="Combined capacity factor")
    curtailment_percent: float = Field(default=0.0, ge=0, le=100, description="Curtailment percentage")
    levelized_cost_of_energy: float = Field(..., gt=0, description="LCOE in $/MWh")
    optimization_objective: OptimizationObjective = Field(..., description="Optimization objective")
    convergence_status: bool = Field(..., description="Convergence status")
    iterations: int = Field(..., ge=0, description="Number of iterations")


class CoordinationStrategy(BaseModel):
    """Wind-PV coordination strategy parameters.

    Attributes:
        strategy_name: Name of coordination strategy
        dispatch_priority: Priority order for dispatching resources
        ramp_rate_limit_mw_per_min: Maximum ramp rate in MW/min
        forecast_horizon_hours: Forecast horizon in hours
        enable_storage_arbitrage: Enable battery storage arbitrage
        curtailment_strategy: Strategy for handling curtailment
        grid_support_enabled: Enable grid support services
    """

    model_config = ConfigDict(frozen=False)

    strategy_name: str = Field(..., min_length=1, description="Strategy name")
    dispatch_priority: List[str] = Field(..., min_length=1, description="Dispatch priority")
    ramp_rate_limit_mw_per_min: float = Field(..., gt=0, description="Ramp rate limit")
    forecast_horizon_hours: int = Field(default=24, gt=0, description="Forecast horizon")
    enable_storage_arbitrage: bool = Field(default=True, description="Enable arbitrage")
    curtailment_strategy: str = Field(default="proportional", description="Curtailment strategy")
    grid_support_enabled: bool = Field(default=True, description="Grid support enabled")


class CoordinationResult(BaseModel):
    """Results from wind-PV coordination analysis.

    Attributes:
        timestamp: Timestamp of coordination result
        pv_dispatch_mw: PV dispatch in MW
        wind_dispatch_mw: Wind dispatch in MW
        storage_dispatch_mw: Storage dispatch in MW (positive = discharge)
        total_dispatch_mw: Total system dispatch in MW
        curtailed_energy_mw: Curtailed energy in MW
        grid_export_mw: Energy exported to grid in MW
        frequency_regulation_mw: Frequency regulation contribution in MW
        voltage_support_mvar: Voltage support in MVAr
        coordination_efficiency: Coordination efficiency (0-1)
    """

    model_config = ConfigDict(frozen=False)

    timestamp: datetime = Field(..., description="Result timestamp")
    pv_dispatch_mw: float = Field(..., ge=0, description="PV dispatch")
    wind_dispatch_mw: float = Field(..., ge=0, description="Wind dispatch")
    storage_dispatch_mw: float = Field(default=0.0, description="Storage dispatch")
    total_dispatch_mw: float = Field(..., ge=0, description="Total dispatch")
    curtailed_energy_mw: float = Field(default=0.0, ge=0, description="Curtailed energy")
    grid_export_mw: float = Field(..., ge=0, description="Grid export")
    frequency_regulation_mw: float = Field(default=0.0, description="Frequency regulation")
    voltage_support_mvar: float = Field(default=0.0, description="Voltage support")
    coordination_efficiency: float = Field(..., ge=0, le=1, description="Coordination efficiency")
