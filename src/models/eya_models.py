"""Pydantic models for Energy Yield Analysis.

This module defines comprehensive data models for energy yield analysis,
including project information, weather data, system configuration, and
performance metrics.
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ModuleType(str, Enum):
    """PV module technology types."""

    MONO_SI = "mono-Si"
    POLY_SI = "poly-Si"
    CDTE = "CdTe"
    CIGS = "CIGS"
    PEROVSKITE = "perovskite"


class MountingType(str, Enum):
    """System mounting configuration."""

    FIXED_TILT = "fixed_tilt"
    SINGLE_AXIS = "single_axis"
    DUAL_AXIS = "dual_axis"
    ROOF_MOUNTED = "roof_mounted"


class ProjectInfo(BaseModel):
    """Project information and metadata.

    Attributes:
        project_name: Name of the PV project
        location: Geographic location (city, country)
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        altitude: Altitude in meters above sea level
        timezone: IANA timezone identifier
        commissioning_date: Expected commissioning date
        project_lifetime: Project lifetime in years
    """

    project_name: str = Field(..., description="Name of the PV project")
    location: str = Field(..., description="Geographic location")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    altitude: float = Field(default=0, ge=0, description="Altitude in meters")
    timezone: str = Field(default="UTC", description="IANA timezone identifier")
    commissioning_date: datetime = Field(..., description="Expected commissioning date")
    project_lifetime: int = Field(default=25, ge=1, le=50, description="Project lifetime in years")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone identifier."""
        # Basic validation - in production, use pytz or zoneinfo
        if not v:
            return "UTC"
        return v


class SystemConfiguration(BaseModel):
    """PV system configuration parameters.

    Attributes:
        capacity_dc: DC capacity in kWp
        capacity_ac: AC capacity in kWac
        module_type: PV module technology type
        module_efficiency: Module efficiency (0-1)
        module_count: Total number of modules
        inverter_efficiency: Inverter efficiency (0-1)
        mounting_type: System mounting configuration
        tilt_angle: Tilt angle in degrees
        azimuth_angle: Azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)
        gcr: Ground coverage ratio (0-1)
        dc_ac_ratio: DC to AC ratio
    """

    capacity_dc: float = Field(..., gt=0, description="DC capacity in kWp")
    capacity_ac: float = Field(..., gt=0, description="AC capacity in kWac")
    module_type: ModuleType = Field(..., description="PV module technology type")
    module_efficiency: float = Field(..., gt=0, le=1, description="Module efficiency")
    module_count: int = Field(..., gt=0, description="Total number of modules")
    inverter_efficiency: float = Field(default=0.98, gt=0, le=1, description="Inverter efficiency")
    mounting_type: MountingType = Field(..., description="Mounting configuration")
    tilt_angle: float = Field(..., ge=0, le=90, description="Tilt angle in degrees")
    azimuth_angle: float = Field(..., ge=0, le=360, description="Azimuth angle in degrees")
    gcr: float = Field(default=0.4, gt=0, le=1, description="Ground coverage ratio")
    dc_ac_ratio: float = Field(default=1.2, gt=0, description="DC to AC ratio")

    @field_validator("capacity_ac")
    @classmethod
    def validate_ac_capacity(cls, v: float, info) -> float:
        """Validate AC capacity is less than DC capacity."""
        if "capacity_dc" in info.data and v > info.data["capacity_dc"]:
            raise ValueError("AC capacity cannot exceed DC capacity")
        return v


class WeatherData(BaseModel):
    """Weather data for energy yield calculations.

    Attributes:
        timestamp: Timestamp of the data point
        ghi: Global Horizontal Irradiance in W/m²
        dni: Direct Normal Irradiance in W/m²
        dhi: Diffuse Horizontal Irradiance in W/m²
        temperature: Ambient temperature in °C
        wind_speed: Wind speed in m/s
        humidity: Relative humidity (0-1)
        pressure: Atmospheric pressure in Pa
    """

    timestamp: datetime = Field(..., description="Timestamp of data point")
    ghi: float = Field(..., ge=0, description="Global Horizontal Irradiance in W/m²")
    dni: float = Field(..., ge=0, description="Direct Normal Irradiance in W/m²")
    dhi: float = Field(..., ge=0, description="Diffuse Horizontal Irradiance in W/m²")
    temperature: float = Field(..., description="Ambient temperature in °C")
    wind_speed: float = Field(default=0, ge=0, description="Wind speed in m/s")
    humidity: float = Field(default=0.5, ge=0, le=1, description="Relative humidity")
    pressure: float = Field(default=101325, ge=0, description="Atmospheric pressure in Pa")


class EnergyOutput(BaseModel):
    """Energy output and production metrics.

    Attributes:
        timestamp: Timestamp of the period
        dc_energy: DC energy production in kWh
        ac_energy: AC energy production in kWh
        exported_energy: Energy exported to grid in kWh
        specific_yield: Specific yield in kWh/kWp
        capacity_factor: Capacity factor (0-1)
    """

    timestamp: datetime = Field(..., description="Timestamp of period")
    dc_energy: float = Field(..., ge=0, description="DC energy in kWh")
    ac_energy: float = Field(..., ge=0, description="AC energy in kWh")
    exported_energy: float = Field(..., ge=0, description="Exported energy in kWh")
    specific_yield: float = Field(..., ge=0, description="Specific yield in kWh/kWp")
    capacity_factor: float = Field(..., ge=0, le=1, description="Capacity factor")


class LossBreakdown(BaseModel):
    """Detailed breakdown of energy losses.

    Attributes:
        soiling_loss: Soiling losses in %
        shading_loss: Shading losses in %
        snow_loss: Snow cover losses in %
        mismatch_loss: Module mismatch losses in %
        wiring_loss: DC wiring losses in %
        connection_loss: Connection losses in %
        lid_loss: Light-induced degradation losses in %
        nameplate_loss: Nameplate rating losses in %
        age_loss: Age-related degradation losses in %
        temperature_loss: Temperature-related losses in %
        inverter_loss: Inverter losses in %
        transformer_loss: Transformer losses in %
        availability_loss: System availability losses in %
        total_loss: Total system losses in %
    """

    soiling_loss: float = Field(default=2.0, ge=0, description="Soiling losses in %")
    shading_loss: float = Field(default=3.0, ge=0, description="Shading losses in %")
    snow_loss: float = Field(default=0.5, ge=0, description="Snow losses in %")
    mismatch_loss: float = Field(default=2.0, ge=0, description="Mismatch losses in %")
    wiring_loss: float = Field(default=1.5, ge=0, description="Wiring losses in %")
    connection_loss: float = Field(default=0.5, ge=0, description="Connection losses in %")
    lid_loss: float = Field(default=1.5, ge=0, description="LID losses in %")
    nameplate_loss: float = Field(default=1.0, ge=0, description="Nameplate losses in %")
    age_loss: float = Field(default=0.5, ge=0, description="Age losses in %")
    temperature_loss: float = Field(default=5.0, ge=0, description="Temperature losses in %")
    inverter_loss: float = Field(default=2.0, ge=0, description="Inverter losses in %")
    transformer_loss: float = Field(default=1.0, ge=0, description="Transformer losses in %")
    availability_loss: float = Field(default=2.0, ge=0, description="Availability losses in %")
    total_loss: float = Field(default=0.0, ge=0, description="Total losses in %")

    def calculate_total_loss(self) -> float:
        """Calculate total system losses."""
        # Multiplicative loss calculation
        total = 1.0
        for field_name, field_value in self.model_dump().items():
            if field_name != "total_loss":
                total *= (1 - field_value / 100)
        return (1 - total) * 100


class PerformanceMetrics(BaseModel):
    """System performance metrics.

    Attributes:
        performance_ratio: Performance ratio (0-1)
        reference_yield: Reference yield in kWh/kWp
        final_yield: Final yield in kWh/kWp
        array_yield: Array yield in kWh/kWp
        system_losses: System losses in kWh/kWp
        capture_losses: Capture losses in kWh/kWp
    """

    performance_ratio: float = Field(..., ge=0, le=1, description="Performance ratio")
    reference_yield: float = Field(..., ge=0, description="Reference yield in kWh/kWp")
    final_yield: float = Field(..., ge=0, description="Final yield in kWh/kWp")
    array_yield: float = Field(..., ge=0, description="Array yield in kWh/kWp")
    system_losses: float = Field(..., ge=0, description="System losses in kWh/kWp")
    capture_losses: float = Field(..., ge=0, description="Capture losses in kWh/kWp")


class FinancialMetrics(BaseModel):
    """Financial analysis metrics.

    Attributes:
        capex: Capital expenditure in USD
        opex_annual: Annual operational expenditure in USD/year
        energy_price: Energy price in USD/kWh
        degradation_rate: Annual degradation rate (0-1)
        discount_rate: Discount rate for NPV (0-1)
        lcoe: Levelized cost of energy in USD/kWh
        npv: Net present value in USD
        irr: Internal rate of return (0-1)
        payback_period: Simple payback period in years
    """

    capex: float = Field(..., gt=0, description="Capital expenditure in USD")
    opex_annual: float = Field(..., ge=0, description="Annual OPEX in USD/year")
    energy_price: float = Field(..., gt=0, description="Energy price in USD/kWh")
    degradation_rate: float = Field(default=0.005, ge=0, le=0.02, description="Annual degradation")
    discount_rate: float = Field(default=0.05, ge=0, le=0.2, description="Discount rate")
    lcoe: Optional[float] = Field(default=None, ge=0, description="LCOE in USD/kWh")
    npv: Optional[float] = Field(default=None, description="NPV in USD")
    irr: Optional[float] = Field(default=None, ge=0, description="IRR")
    payback_period: Optional[float] = Field(default=None, ge=0, description="Payback in years")


class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis results.

    Attributes:
        parameter_name: Name of the parameter being varied
        base_value: Base case value
        variation_range: Range of variation (min, max)
        results: Dictionary mapping parameter values to output metrics
    """

    parameter_name: str = Field(..., description="Parameter name")
    base_value: float = Field(..., description="Base case value")
    variation_range: tuple[float, float] = Field(..., description="Variation range (min, max)")
    results: Dict[float, float] = Field(default_factory=dict, description="Results mapping")


class ProbabilisticAnalysis(BaseModel):
    """Probabilistic energy yield analysis (P50, P90, P99).

    Attributes:
        p99: P99 exceedance probability energy yield in kWh/year
        p90: P90 exceedance probability energy yield in kWh/year
        p75: P75 exceedance probability energy yield in kWh/year
        p50: P50 (median) energy yield in kWh/year
        mean: Mean energy yield in kWh/year
        std_dev: Standard deviation of energy yield
        confidence_intervals: Confidence intervals for different probabilities
    """

    p99: float = Field(..., ge=0, description="P99 energy yield in kWh/year")
    p90: float = Field(..., ge=0, description="P90 energy yield in kWh/year")
    p75: float = Field(..., ge=0, description="P75 energy yield in kWh/year")
    p50: float = Field(..., ge=0, description="P50 energy yield in kWh/year")
    mean: float = Field(..., ge=0, description="Mean energy yield in kWh/year")
    std_dev: float = Field(..., ge=0, description="Standard deviation")
    confidence_intervals: Dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Confidence intervals"
    )
