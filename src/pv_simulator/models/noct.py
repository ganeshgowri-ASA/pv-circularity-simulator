"""
Pydantic models for NOCT (Nominal Operating Cell Temperature) data.

This module defines data models for NOCT specifications, test conditions,
and module-specific NOCT data, including B03 module integration.
"""

from typing import Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator

from pv_simulator.utils.constants import (
    NOCT_IRRADIANCE,
    NOCT_AMBIENT_TEMP,
    NOCT_WIND_SPEED,
    NOCT_TILT,
)


class NOCTTestConditions(BaseModel):
    """
    Standard NOCT test conditions per IEC 61215.

    NOCT (Nominal Operating Cell Temperature) is defined as the cell temperature
    reached under specific test conditions designed to represent typical operating conditions.

    Standard NOCT test conditions:
    - Irradiance: 800 W/m² (solar spectrum)
    - Ambient temperature: 20°C
    - Wind speed: 1 m/s
    - Tilt angle: 45° (or latitude-dependent)
    - Mounting: Open rack
    - Electrical load: Open circuit

    Attributes:
        irradiance: Solar irradiance in W/m²
        ambient_temp: Ambient air temperature in °C
        wind_speed: Wind speed in m/s
        tilt_angle: Module tilt angle in degrees
        mounting_type: Type of mounting configuration
        electrical_load: Electrical loading condition
    """

    irradiance: float = Field(
        default=NOCT_IRRADIANCE,
        description="Solar irradiance in W/m²",
        ge=0.0,
        le=1500.0,
    )
    ambient_temp: float = Field(
        default=NOCT_AMBIENT_TEMP,
        description="Ambient air temperature in °C",
        ge=-50.0,
        le=60.0,
    )
    wind_speed: float = Field(
        default=NOCT_WIND_SPEED,
        description="Wind speed in m/s",
        ge=0.0,
        le=50.0,
    )
    tilt_angle: float = Field(
        default=NOCT_TILT,
        description="Module tilt angle in degrees",
        ge=0.0,
        le=90.0,
    )
    mounting_type: Literal["open_rack", "roof_mounted", "ground_mounted", "building_integrated"] = Field(
        default="open_rack",
        description="Type of mounting configuration",
    )
    electrical_load: Literal["open_circuit", "maximum_power", "short_circuit"] = Field(
        default="open_circuit",
        description="Electrical loading condition",
    )

    model_config = {"frozen": False}

    @field_validator("irradiance")
    @classmethod
    def validate_noct_irradiance(cls, v: float) -> float:
        """Validate that irradiance is close to NOCT standard (800 W/m²)."""
        if abs(v - NOCT_IRRADIANCE) > 50:
            # Warning: not standard NOCT conditions, but allow it
            pass
        return v


class NOCTSpecification(BaseModel):
    """
    NOCT specification for a PV module.

    Attributes:
        noct_celsius: Nominal Operating Cell Temperature in °C
        test_conditions: Test conditions under which NOCT was measured
        measurement_uncertainty: Measurement uncertainty in °C
        test_date: Date of NOCT measurement
        test_standard: Standard used for testing (e.g., IEC 61215)
        notes: Additional notes about the measurement
    """

    noct_celsius: float = Field(
        ...,
        description="Nominal Operating Cell Temperature in °C",
        ge=20.0,
        le=70.0,
    )
    test_conditions: NOCTTestConditions = Field(
        default_factory=NOCTTestConditions,
        description="Test conditions under which NOCT was measured",
    )
    measurement_uncertainty: Optional[float] = Field(
        default=2.0,
        description="Measurement uncertainty in °C",
        ge=0.0,
        le=10.0,
    )
    test_date: Optional[date] = Field(
        default=None,
        description="Date of NOCT measurement",
    )
    test_standard: str = Field(
        default="IEC 61215",
        description="Standard used for testing",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the measurement",
    )

    model_config = {"frozen": False}


class ModuleNOCTData(BaseModel):
    """
    Complete NOCT data for a specific PV module, including B03 integration.

    This model contains all NOCT-related information for a module, including
    temperature coefficients, thermal parameters, and physical specifications.
    Designed to integrate with B03 NOCT database.

    Attributes:
        module_id: Unique module identifier (e.g., B03-001)
        manufacturer: Module manufacturer
        model_name: Module model name
        technology: Cell technology type
        noct_spec: NOCT specification
        temp_coeff_power: Power temperature coefficient in 1/°C
        temp_coeff_voc: Voc temperature coefficient in 1/°C
        temp_coeff_isc: Isc temperature coefficient in 1/°C
        rated_power_stc: Rated power at STC in W
        efficiency_stc: Efficiency at STC in percentage
        module_area: Module area in m²
        cell_count: Number of cells in module
        heat_capacity: Module heat capacity in J/(m²·K)
        absorptivity: Solar absorptivity (dimensionless)
        emissivity: Thermal emissivity (dimensionless)
        b03_verified: Whether this data is verified for B03 integration
        data_source: Source of the data
    """

    module_id: str = Field(
        ...,
        description="Unique module identifier",
        min_length=1,
        max_length=50,
    )
    manufacturer: str = Field(
        ...,
        description="Module manufacturer",
        min_length=1,
        max_length=100,
    )
    model_name: str = Field(
        ...,
        description="Module model name",
        min_length=1,
        max_length=100,
    )
    technology: Literal[
        "mono_si",
        "poly_si",
        "cdte",
        "cigs",
        "perovskite",
        "hjt",
        "bifacial",
        "tandem",
        "other",
    ] = Field(
        ...,
        description="Cell technology type",
    )
    noct_spec: NOCTSpecification = Field(
        ...,
        description="NOCT specification",
    )
    temp_coeff_power: float = Field(
        ...,
        description="Power temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.0,
    )
    temp_coeff_voc: float = Field(
        ...,
        description="Voc temperature coefficient in 1/°C",
        ge=-0.01,
        le=0.0,
    )
    temp_coeff_isc: float = Field(
        default=0.0005,
        description="Isc temperature coefficient in 1/°C",
        ge=0.0,
        le=0.001,
    )
    rated_power_stc: float = Field(
        ...,
        description="Rated power at STC in W",
        ge=10.0,
        le=1000.0,
    )
    efficiency_stc: float = Field(
        ...,
        description="Efficiency at STC in percentage",
        ge=5.0,
        le=30.0,
    )
    module_area: float = Field(
        ...,
        description="Module area in m²",
        ge=0.1,
        le=10.0,
    )
    cell_count: int = Field(
        default=60,
        description="Number of cells in module",
        ge=36,
        le=144,
    )
    heat_capacity: Optional[float] = Field(
        default=11000.0,
        description="Module heat capacity in J/(m²·K)",
        ge=5000.0,
        le=30000.0,
    )
    absorptivity: float = Field(
        default=0.9,
        description="Solar absorptivity (dimensionless)",
        ge=0.5,
        le=1.0,
    )
    emissivity: float = Field(
        default=0.85,
        description="Thermal emissivity (dimensionless)",
        ge=0.5,
        le=1.0,
    )
    b03_verified: bool = Field(
        default=False,
        description="Whether this data is verified for B03 integration",
    )
    data_source: str = Field(
        default="manufacturer_datasheet",
        description="Source of the data",
    )

    model_config = {"frozen": False}

    @field_validator("module_id")
    @classmethod
    def validate_b03_module_id(cls, v: str) -> str:
        """Validate module ID format for B03 integration."""
        if v.startswith("B03-"):
            # B03 module - ensure proper format
            if not v[4:].isdigit() or len(v) < 8:
                raise ValueError(
                    f"Invalid B03 module ID format: {v}. Expected format: B03-XXXXX"
                )
        return v

    def get_thermal_parameters(self) -> dict:
        """
        Extract thermal parameters as a dictionary for use in calculations.

        Returns:
            Dictionary containing thermal parameters
        """
        return {
            "heat_capacity": self.heat_capacity,
            "absorptivity": self.absorptivity,
            "emissivity": self.emissivity,
            "module_area": self.module_area,
            "noct": self.noct_spec.noct_celsius,
        }

    def get_temperature_coefficients(self) -> dict:
        """
        Extract temperature coefficients as a dictionary.

        Returns:
            Dictionary containing temperature coefficients
        """
        return {
            "power": self.temp_coeff_power,
            "voc": self.temp_coeff_voc,
            "isc": self.temp_coeff_isc,
        }

    def estimate_cell_temperature(self, ambient_temp: float, irradiance: float) -> float:
        """
        Estimate cell temperature using simple NOCT-based model.

        Args:
            ambient_temp: Ambient temperature in °C
            irradiance: Solar irradiance in W/m²

        Returns:
            Estimated cell temperature in °C

        Note:
            This is a simplified estimation. For more accurate results,
            use the full thermal models in CellTemperatureModel.
        """
        noct = self.noct_spec.noct_celsius
        noct_irrad = self.noct_spec.test_conditions.irradiance
        noct_ambient = self.noct_spec.test_conditions.ambient_temp

        # Simple linear NOCT-based model
        temp_rise_at_noct = noct - noct_ambient
        temp_rise = temp_rise_at_noct * (irradiance / noct_irrad)
        cell_temp = ambient_temp + temp_rise

        return cell_temp
