"""
Pydantic models for thermal and temperature data structures.

This module defines data models for thermal modeling, including environmental conditions,
thermal parameters, temperature coefficients, and model outputs.
"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import numpy as np

from pv_simulator.utils.constants import (
    STC_TEMPERATURE,
    DEFAULT_MODULE_HEAT_CAPACITY,
    DEFAULT_ABSORPTIVITY,
    DEFAULT_EMISSIVITY,
)


class TemperatureConditions(BaseModel):
    """
    Environmental conditions affecting module temperature.

    Attributes:
        ambient_temp: Ambient air temperature in °C
        irradiance: Solar irradiance in W/m²
        wind_speed: Wind speed in m/s
        relative_humidity: Relative humidity as percentage (0-100)
        atmospheric_pressure: Atmospheric pressure in Pa
        sky_temperature: Sky temperature in °C (optional, calculated if not provided)
        ground_temperature: Ground temperature in °C (optional)
    """

    ambient_temp: float = Field(
        ...,
        description="Ambient air temperature in °C",
        ge=-50.0,
        le=60.0,
    )
    irradiance: float = Field(
        ...,
        description="Solar irradiance in W/m²",
        ge=0.0,
        le=1500.0,
    )
    wind_speed: float = Field(
        ...,
        description="Wind speed in m/s",
        ge=0.0,
        le=50.0,
    )
    relative_humidity: Optional[float] = Field(
        default=50.0,
        description="Relative humidity as percentage",
        ge=0.0,
        le=100.0,
    )
    atmospheric_pressure: Optional[float] = Field(
        default=101325.0,
        description="Atmospheric pressure in Pa",
        ge=50000.0,
        le=110000.0,
    )
    sky_temperature: Optional[float] = Field(
        default=None,
        description="Sky temperature in °C (calculated if not provided)",
    )
    ground_temperature: Optional[float] = Field(
        default=None,
        description="Ground temperature in °C",
    )

    model_config = {"frozen": False, "validate_assignment": True}

    @field_validator("sky_temperature", mode="before")
    @classmethod
    def calculate_sky_temp(cls, v: Optional[float], info) -> Optional[float]:
        """Calculate sky temperature if not provided using ambient temperature."""
        if v is None and "ambient_temp" in info.data:
            # Simplified sky temperature model: T_sky ≈ T_ambient - 10°C
            return info.data["ambient_temp"] - 10.0
        return v


class MountingConfiguration(BaseModel):
    """
    Module mounting configuration affecting thermal behavior.

    Attributes:
        mounting_type: Type of mounting configuration
        tilt_angle: Module tilt angle from horizontal in degrees
        azimuth: Module azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)
        height_above_ground: Height of module above ground in meters
        standoff_distance: Distance between module and mounting surface in meters
        ventilation_gap: Ventilation gap size in meters (for roof-mounted)
    """

    mounting_type: Literal["open_rack", "roof_mounted", "ground_mounted", "building_integrated"] = Field(
        default="open_rack",
        description="Type of mounting configuration",
    )
    tilt_angle: float = Field(
        default=30.0,
        description="Module tilt angle from horizontal in degrees",
        ge=0.0,
        le=90.0,
    )
    azimuth: float = Field(
        default=180.0,
        description="Module azimuth angle in degrees",
        ge=0.0,
        le=360.0,
    )
    height_above_ground: float = Field(
        default=1.0,
        description="Height of module above ground in meters",
        ge=0.0,
        le=100.0,
    )
    standoff_distance: Optional[float] = Field(
        default=0.1,
        description="Distance between module and mounting surface in meters",
        ge=0.0,
        le=1.0,
    )
    ventilation_gap: Optional[float] = Field(
        default=0.05,
        description="Ventilation gap size in meters",
        ge=0.0,
        le=0.5,
    )

    model_config = {"frozen": False}


class ThermalParameters(BaseModel):
    """
    Thermal properties of the PV module.

    Attributes:
        heat_capacity: Module heat capacity in J/(m²·K)
        absorptivity: Solar absorptivity (dimensionless, 0-1)
        emissivity: Thermal emissivity (dimensionless, 0-1)
        module_length: Module length in meters
        module_width: Module width in meters
        module_area: Module area in m² (calculated if not provided)
        glass_thickness: Front glass thickness in meters
        cell_area_fraction: Fraction of module area covered by cells (dimensionless, 0-1)
    """

    heat_capacity: float = Field(
        default=DEFAULT_MODULE_HEAT_CAPACITY,
        description="Module heat capacity in J/(m²·K)",
        ge=1000.0,
        le=50000.0,
    )
    absorptivity: float = Field(
        default=DEFAULT_ABSORPTIVITY,
        description="Solar absorptivity (dimensionless)",
        ge=0.0,
        le=1.0,
    )
    emissivity: float = Field(
        default=DEFAULT_EMISSIVITY,
        description="Thermal emissivity (dimensionless)",
        ge=0.0,
        le=1.0,
    )
    module_length: float = Field(
        default=1.65,
        description="Module length in meters",
        ge=0.1,
        le=5.0,
    )
    module_width: float = Field(
        default=1.0,
        description="Module width in meters",
        ge=0.1,
        le=3.0,
    )
    module_area: Optional[float] = Field(
        default=None,
        description="Module area in m² (calculated if not provided)",
        ge=0.01,
        le=15.0,
    )
    glass_thickness: float = Field(
        default=0.003,
        description="Front glass thickness in meters",
        ge=0.001,
        le=0.01,
    )
    cell_area_fraction: float = Field(
        default=0.85,
        description="Fraction of module area covered by cells",
        ge=0.5,
        le=1.0,
    )

    model_config = {"frozen": False}

    @field_validator("module_area", mode="before")
    @classmethod
    def calculate_area(cls, v: Optional[float], info) -> float:
        """Calculate module area if not provided."""
        if v is None and "module_length" in info.data and "module_width" in info.data:
            return info.data["module_length"] * info.data["module_width"]
        return v if v is not None else 1.65  # Default area


class TemperatureCoefficients(BaseModel):
    """
    Temperature coefficients for PV module performance.

    Attributes:
        power: Power temperature coefficient in 1/°C (typically negative)
        voc: Open-circuit voltage temperature coefficient in 1/°C (typically negative)
        isc: Short-circuit current temperature coefficient in 1/°C (typically positive)
        vmpp: Maximum power point voltage temperature coefficient in 1/°C
        impp: Maximum power point current temperature coefficient in 1/°C
        efficiency: Efficiency temperature coefficient in 1/°C (typically negative)
        reference_temp: Reference temperature for coefficients in °C
    """

    power: float = Field(
        ...,
        description="Power temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.01,
    )
    voc: float = Field(
        ...,
        description="Open-circuit voltage temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.01,
    )
    isc: float = Field(
        default=0.0005,
        description="Short-circuit current temperature coefficient in 1/°C",
        ge=-0.001,
        le=0.001,
    )
    vmpp: Optional[float] = Field(
        default=None,
        description="Maximum power point voltage temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.01,
    )
    impp: Optional[float] = Field(
        default=None,
        description="Maximum power point current temperature coefficient in 1/°C",
        ge=-0.001,
        le=0.001,
    )
    efficiency: Optional[float] = Field(
        default=None,
        description="Efficiency temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.01,
    )
    reference_temp: float = Field(
        default=STC_TEMPERATURE,
        description="Reference temperature for coefficients in °C",
        ge=-50.0,
        le=100.0,
    )

    model_config = {"frozen": False}


class HeatTransferCoefficients(BaseModel):
    """
    Heat transfer coefficients for thermal modeling.

    Attributes:
        convective_front: Front surface convective heat transfer coefficient in W/(m²·K)
        convective_back: Back surface convective heat transfer coefficient in W/(m²·K)
        radiative_front: Front surface radiative heat transfer coefficient in W/(m²·K)
        radiative_back: Back surface radiative heat transfer coefficient in W/(m²·K)
        conductive: Conductive heat transfer coefficient through module in W/(m²·K)
        total_front: Total front surface heat loss coefficient in W/(m²·K)
        total_back: Total back surface heat loss coefficient in W/(m²·K)
    """

    convective_front: float = Field(
        ...,
        description="Front surface convective heat transfer coefficient in W/(m²·K)",
        ge=0.0,
        le=100.0,
    )
    convective_back: float = Field(
        ...,
        description="Back surface convective heat transfer coefficient in W/(m²·K)",
        ge=0.0,
        le=100.0,
    )
    radiative_front: float = Field(
        ...,
        description="Front surface radiative heat transfer coefficient in W/(m²·K)",
        ge=0.0,
        le=50.0,
    )
    radiative_back: float = Field(
        ...,
        description="Back surface radiative heat transfer coefficient in W/(m²·K)",
        ge=0.0,
        le=50.0,
    )
    conductive: Optional[float] = Field(
        default=None,
        description="Conductive heat transfer coefficient through module in W/(m²·K)",
        ge=0.0,
        le=1000.0,
    )
    total_front: Optional[float] = Field(
        default=None,
        description="Total front surface heat loss coefficient in W/(m²·K)",
    )
    total_back: Optional[float] = Field(
        default=None,
        description="Total back surface heat loss coefficient in W/(m²·K)",
    )

    model_config = {"frozen": False}

    @field_validator("total_front", mode="before")
    @classmethod
    def calculate_total_front(cls, v: Optional[float], info) -> float:
        """Calculate total front heat loss coefficient if not provided."""
        if v is None and "convective_front" in info.data and "radiative_front" in info.data:
            return info.data["convective_front"] + info.data["radiative_front"]
        return v if v is not None else 0.0

    @field_validator("total_back", mode="before")
    @classmethod
    def calculate_total_back(cls, v: Optional[float], info) -> float:
        """Calculate total back heat loss coefficient if not provided."""
        if v is None and "convective_back" in info.data and "radiative_back" in info.data:
            return info.data["convective_back"] + info.data["radiative_back"]
        return v if v is not None else 0.0


class ThermalModelOutput(BaseModel):
    """
    Output from thermal modeling calculations.

    Attributes:
        cell_temperature: Cell temperature in °C
        module_temperature: Average module temperature in °C
        back_surface_temperature: Back surface temperature in °C
        thermal_loss: Thermal power loss in W
        thermal_efficiency_loss: Efficiency loss due to temperature in percentage
        heat_transfer_coeffs: Heat transfer coefficients used
        timestamp: Calculation timestamp
        model_name: Name of the thermal model used
        conditions: Environmental conditions used
        mounting: Mounting configuration used
    """

    cell_temperature: float = Field(
        ...,
        description="Cell temperature in °C",
        ge=-50.0,
        le=150.0,
    )
    module_temperature: float = Field(
        ...,
        description="Average module temperature in °C",
        ge=-50.0,
        le=150.0,
    )
    back_surface_temperature: Optional[float] = Field(
        default=None,
        description="Back surface temperature in °C",
        ge=-50.0,
        le=150.0,
    )
    thermal_loss: Optional[float] = Field(
        default=None,
        description="Thermal power loss in W",
        ge=0.0,
    )
    thermal_efficiency_loss: Optional[float] = Field(
        default=None,
        description="Efficiency loss due to temperature in percentage",
        ge=0.0,
        le=100.0,
    )
    heat_transfer_coeffs: Optional[HeatTransferCoefficients] = Field(
        default=None,
        description="Heat transfer coefficients used",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Calculation timestamp",
    )
    model_name: str = Field(
        ...,
        description="Name of the thermal model used",
    )
    conditions: Optional[TemperatureConditions] = Field(
        default=None,
        description="Environmental conditions used",
    )
    mounting: Optional[MountingConfiguration] = Field(
        default=None,
        description="Mounting configuration used",
    )

    model_config = {"frozen": False, "arbitrary_types_allowed": True}
