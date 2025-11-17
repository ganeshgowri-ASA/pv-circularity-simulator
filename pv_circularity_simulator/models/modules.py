"""
Module models for PV circularity simulator.

This module defines comprehensive Pydantic models for photovoltaic modules,
including:
- Module configuration (cells in series/parallel)
- Electrical parameters at module level
- Mechanical properties (size, weight, mounting)
- Thermal characteristics
- Degradation models
- Warranty and certification information

All models include full validation for physical constraints and
production-ready error handling.
"""

from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.cells import CellModel
from pv_circularity_simulator.models.core import NamedModel, UUIDModel


class FrameMaterial(str, Enum):
    """Enumeration of module frame materials."""

    ALUMINUM = "aluminum"
    STEEL = "steel"
    COMPOSITE = "composite"
    FRAMELESS = "frameless"


class GlassType(str, Enum):
    """Enumeration of module glass types."""

    TEMPERED = "tempered"
    ANTI_REFLECTIVE = "anti_reflective"
    BIFACIAL = "bifacial"
    LOW_IRON = "low_iron"


class ModuleConfiguration(NamedModel):
    """
    Configuration of cells within a module.

    Defines how cells are arranged and connected electrically
    within the module structure.

    Attributes:
        cells_in_series: Number of cells connected in series
        cells_in_parallel: Number of cells connected in parallel (usually 1)
        total_cells: Total number of cells in module
        bypass_diodes: Number of bypass diodes
        cells_per_bypass_diode: Number of cells per bypass diode string
        half_cut_cells: Whether cells are half-cut (improves performance)
    """

    cells_in_series: int = Field(
        ...,
        ge=1,
        le=200,
        description="Number of cells connected in series (typical: 60, 72, 144)",
    )
    cells_in_parallel: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of cells connected in parallel (usually 1)",
    )
    total_cells: int = Field(
        ...,
        ge=1,
        description="Total number of cells in the module",
    )
    bypass_diodes: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of bypass diodes (typical: 3 for 60/72-cell modules)",
    )
    cells_per_bypass_diode: int = Field(
        ...,
        ge=1,
        description="Number of cells per bypass diode string",
    )
    half_cut_cells: bool = Field(
        default=False,
        description="Whether cells are half-cut (improves performance and reduces losses)",
    )

    @model_validator(mode="after")
    def validate_cell_configuration(self) -> "ModuleConfiguration":
        """Validate cell configuration consistency."""
        # Total cells should equal series × parallel
        expected_total = self.cells_in_series * self.cells_in_parallel
        if self.total_cells != expected_total:
            raise ValueError(
                f"Total cells ({self.total_cells}) should equal "
                f"cells_in_series × cells_in_parallel ({expected_total})"
            )

        # Cells per bypass diode should divide evenly
        if self.bypass_diodes > 0:
            if self.cells_in_series % self.bypass_diodes != 0:
                raise ValueError(
                    f"Cells in series ({self.cells_in_series}) should be divisible by "
                    f"number of bypass diodes ({self.bypass_diodes})"
                )
            expected_cells_per_diode = self.cells_in_series // self.bypass_diodes
            if self.cells_per_bypass_diode != expected_cells_per_diode:
                raise ValueError(
                    f"Cells per bypass diode ({self.cells_per_bypass_diode}) should equal "
                    f"cells_in_series / bypass_diodes ({expected_cells_per_diode})"
                )

        return self


class ElectricalParameters(NamedModel):
    """
    Electrical parameters of a PV module at STC.

    Standard Test Conditions (STC):
    - 1000 W/m² irradiance
    - 25°C cell temperature
    - AM 1.5 spectrum

    Attributes:
        pmax_w: Maximum power at STC in watts
        voc_v: Open circuit voltage in volts
        isc_a: Short circuit current in amperes
        vmpp_v: Voltage at maximum power point in volts
        impp_a: Current at maximum power point in amperes
        efficiency_percentage: Module efficiency percentage
        temperature_coefficient_pmax: Temperature coefficient of Pmax (%/°C)
        temperature_coefficient_voc: Temperature coefficient of Voc (%/°C)
        temperature_coefficient_isc: Temperature coefficient of Isc (%/°C)
        max_system_voltage_v: Maximum system voltage rating in volts
        max_series_fuse_a: Maximum series fuse rating in amperes
    """

    pmax_w: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Maximum power at STC in watts (typical: 250-600 W)",
    )
    voc_v: float = Field(
        ...,
        gt=0,
        le=100,
        description="Open circuit voltage in volts (typical: 30-50 V)",
    )
    isc_a: float = Field(
        ...,
        gt=0,
        le=20,
        description="Short circuit current in amperes (typical: 8-12 A)",
    )
    vmpp_v: float = Field(
        ...,
        gt=0,
        description="Voltage at maximum power point in volts",
    )
    impp_a: float = Field(
        ...,
        gt=0,
        description="Current at maximum power point in amperes",
    )
    efficiency_percentage: float = Field(
        ...,
        gt=0,
        le=30,
        description="Module efficiency percentage (typical: 15-22%)",
    )
    temperature_coefficient_pmax: float = Field(
        default=-0.4,
        ge=-1.0,
        le=0.0,
        description="Temperature coefficient of Pmax in %/°C (typical: -0.3 to -0.5)",
    )
    temperature_coefficient_voc: float = Field(
        default=-0.3,
        ge=-1.0,
        le=0.0,
        description="Temperature coefficient of Voc in %/°C (typical: -0.28 to -0.35)",
    )
    temperature_coefficient_isc: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Temperature coefficient of Isc in %/°C (typical: 0.03 to 0.06)",
    )
    max_system_voltage_v: int = Field(
        default=1000,
        gt=0,
        le=1500,
        description="Maximum system voltage rating in volts (typical: 1000 or 1500)",
    )
    max_series_fuse_a: int = Field(
        default=15,
        gt=0,
        le=30,
        description="Maximum series fuse rating in amperes",
    )

    @model_validator(mode="after")
    def validate_electrical_consistency(self) -> "ElectricalParameters":
        """Validate electrical parameters are physically consistent."""
        # Vmpp should be less than Voc
        if self.vmpp_v >= self.voc_v:
            raise ValueError(f"Vmpp ({self.vmpp_v}V) must be less than Voc ({self.voc_v}V)")

        # Impp should be less than Isc
        if self.impp_a >= self.isc_a:
            raise ValueError(f"Impp ({self.impp_a}A) must be less than Isc ({self.isc_a}A)")

        # Pmax should approximately equal Vmpp × Impp
        calculated_pmax = self.vmpp_v * self.impp_a
        if abs(calculated_pmax - self.pmax_w) > 1.0:
            raise ValueError(
                f"Pmax ({self.pmax_w}W) should equal Vmpp × Impp ({calculated_pmax:.2f}W)"
            )

        return self


class MechanicalProperties(NamedModel):
    """
    Mechanical and physical properties of a PV module.

    Attributes:
        length_mm: Module length in millimeters
        width_mm: Module width in millimeters
        thickness_mm: Module thickness in millimeters
        weight_kg: Module weight in kilograms
        frame_material: Frame material type
        glass_type: Front glass type
        glass_thickness_mm: Front glass thickness in mm
        back_glass_thickness_mm: Back glass thickness (for bifacial, optional)
        junction_box_type: Type of junction box
        cable_length_mm: Length of attached cables in mm
        mounting_holes: Number of mounting holes
    """

    length_mm: float = Field(
        ...,
        gt=0,
        le=3000,
        description="Module length in millimeters (typical: 1600-2100 mm)",
    )
    width_mm: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Module width in millimeters (typical: 1000-1300 mm)",
    )
    thickness_mm: float = Field(
        default=35.0,
        gt=0,
        le=100,
        description="Module thickness in millimeters (typical: 30-50 mm)",
    )
    weight_kg: float = Field(
        ...,
        gt=0,
        le=50,
        description="Module weight in kilograms (typical: 15-30 kg)",
    )
    frame_material: FrameMaterial = Field(
        default=FrameMaterial.ALUMINUM,
        description="Frame material type",
    )
    glass_type: GlassType = Field(
        default=GlassType.TEMPERED,
        description="Front glass type",
    )
    glass_thickness_mm: float = Field(
        default=3.2,
        gt=0,
        le=10,
        description="Front glass thickness in mm (typical: 3.2-4.0 mm)",
    )
    back_glass_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        le=10,
        description="Back glass thickness in mm (for bifacial modules)",
    )
    junction_box_type: str = Field(
        default="standard",
        max_length=50,
        description="Type of junction box (standard, smart, etc.)",
    )
    cable_length_mm: int = Field(
        default=1000,
        ge=0,
        le=3000,
        description="Length of attached cables in millimeters (typical: 1000-1200 mm)",
    )
    mounting_holes: int = Field(
        default=4,
        ge=2,
        le=12,
        description="Number of mounting holes",
    )

    def calculate_area_m2(self) -> float:
        """
        Calculate module area in square meters.

        Returns:
            float: Module area in m²
        """
        length_m = self.length_mm / 1000.0
        width_m = self.width_mm / 1000.0
        return length_m * width_m

    def calculate_power_density_w_m2(self, pmax_w: float) -> float:
        """
        Calculate power density in watts per square meter.

        Args:
            pmax_w: Maximum power in watts

        Returns:
            float: Power density in W/m²
        """
        area_m2 = self.calculate_area_m2()
        return pmax_w / area_m2


class ThermalProperties(NamedModel):
    """
    Thermal properties and operating conditions of a PV module.

    Attributes:
        noct_c: Nominal Operating Cell Temperature in Celsius
        operating_temp_min_c: Minimum operating temperature in Celsius
        operating_temp_max_c: Maximum operating temperature in Celsius
        thermal_resistance_k_w: Thermal resistance in K/W
        heat_capacity_j_k: Heat capacity in J/K
    """

    noct_c: float = Field(
        default=45.0,
        ge=-40,
        le=100,
        description="Nominal Operating Cell Temperature in °C (typical: 42-47°C)",
    )
    operating_temp_min_c: float = Field(
        default=-40.0,
        ge=-60,
        le=0,
        description="Minimum operating temperature in °C (typical: -40°C)",
    )
    operating_temp_max_c: float = Field(
        default=85.0,
        ge=50,
        le=125,
        description="Maximum operating temperature in °C (typical: 85°C)",
    )
    thermal_resistance_k_w: float = Field(
        default=0.03,
        gt=0,
        le=1.0,
        description="Thermal resistance in K/W",
    )
    heat_capacity_j_k: float = Field(
        default=5000.0,
        gt=0,
        description="Heat capacity in J/K",
    )

    @field_validator("operating_temp_min_c", "operating_temp_max_c")
    @classmethod
    def validate_temperature_range(cls, v: float) -> float:
        """Validate temperature is within realistic operating range."""
        if v < -60 or v > 125:
            raise ValueError("Temperature must be between -60°C and 125°C")
        return v

    @model_validator(mode="after")
    def validate_temp_range_consistency(self) -> "ThermalProperties":
        """Validate that min temp is less than max temp."""
        if self.operating_temp_min_c >= self.operating_temp_max_c:
            raise ValueError(
                f"Minimum operating temperature ({self.operating_temp_min_c}°C) "
                f"must be less than maximum ({self.operating_temp_max_c}°C)"
            )
        return self


class ModuleModel(UUIDModel):
    """
    Comprehensive photovoltaic module model.

    A module consists of multiple PV cells connected together,
    encapsulated, and framed for installation. This model captures
    all relevant characteristics of a PV module.

    Attributes:
        name: Human-readable name/identifier for the module
        model_number: Manufacturer's model number
        cell_reference: Reference cell used in this module
        configuration: Cell configuration (series/parallel arrangement)
        electrical: Electrical parameters at STC
        mechanical: Physical and mechanical properties
        thermal: Thermal characteristics
        manufacturer: Manufacturer name
        manufacturing_date: Date of manufacture
        warranty_years_product: Product warranty in years
        warranty_years_performance: Performance warranty in years
        performance_guarantee_25y_percent: Guaranteed power output after 25 years (%)
        certifications: Certification information (type: certificate_number)
        bifacial_factor: Bifacial factor for bifacial modules (0-1)
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name/identifier for the module",
    )
    model_number: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Manufacturer's model number",
    )
    cell_reference: CellModel = Field(
        ...,
        description="Reference cell model used in this module",
    )
    configuration: ModuleConfiguration = Field(
        ...,
        description="Cell configuration within the module",
    )
    electrical: ElectricalParameters = Field(
        ...,
        description="Electrical parameters at standard test conditions",
    )
    mechanical: MechanicalProperties = Field(
        ...,
        description="Physical and mechanical properties",
    )
    thermal: ThermalProperties = Field(
        ...,
        description="Thermal characteristics and operating conditions",
    )
    manufacturer: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Manufacturer name",
    )
    manufacturing_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Manufacturing date in YYYY-MM-DD format",
    )
    warranty_years_product: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Product warranty in years (typical: 10-25 years)",
    )
    warranty_years_performance: int = Field(
        default=25,
        ge=0,
        le=50,
        description="Performance warranty in years (typical: 25-30 years)",
    )
    performance_guarantee_25y_percent: float = Field(
        default=80.0,
        ge=60.0,
        le=100.0,
        description="Guaranteed minimum power output after 25 years as % of initial",
    )
    certifications: Optional[Dict[str, str]] = Field(
        None,
        description="Certifications (type: certificate_number), e.g., IEC 61215, UL 1703",
    )
    bifacial_factor: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Bifacial factor for bifacial modules (0-1, typical: 0.7-0.9)",
    )

    @model_validator(mode="after")
    def validate_module_consistency(self) -> "ModuleModel":
        """Validate consistency across module components."""
        # Module voltage should be approximately cell voltage × cells in series
        expected_voc = (
            self.cell_reference.electrical.voc_v * self.configuration.cells_in_series
        )
        if abs(self.electrical.voc_v - expected_voc) > expected_voc * 0.1:
            raise ValueError(
                f"Module Voc ({self.electrical.voc_v}V) should be approximately "
                f"cell Voc × cells in series ({expected_voc:.1f}V)"
            )

        # Module current should be approximately cell current × cells in parallel
        expected_isc = (
            self.cell_reference.electrical.isc_a * self.configuration.cells_in_parallel
        )
        if abs(self.electrical.isc_a - expected_isc) > expected_isc * 0.1:
            raise ValueError(
                f"Module Isc ({self.electrical.isc_a}A) should be approximately "
                f"cell Isc × cells in parallel ({expected_isc:.1f}A)"
            )

        return self

    def calculate_stc_power(self) -> float:
        """
        Calculate module power at STC.

        Returns:
            float: Power at STC in watts
        """
        return self.electrical.vmpp_v * self.electrical.impp_a

    def calculate_power_at_temperature(self, cell_temp_c: float) -> float:
        """
        Calculate module power at a specific cell temperature.

        Args:
            cell_temp_c: Cell temperature in Celsius

        Returns:
            float: Estimated power output in watts

        Raises:
            ValueError: If temperature is outside operating range
        """
        if not (
            self.thermal.operating_temp_min_c
            <= cell_temp_c
            <= self.thermal.operating_temp_max_c
        ):
            raise ValueError(
                f"Cell temperature {cell_temp_c}°C is outside operating range "
                f"({self.thermal.operating_temp_min_c}°C to "
                f"{self.thermal.operating_temp_max_c}°C)"
            )

        # Temperature difference from STC (25°C)
        delta_t = cell_temp_c - 25.0

        # Power derating based on temperature coefficient
        power_factor = 1.0 + (self.electrical.temperature_coefficient_pmax / 100.0) * delta_t

        return self.electrical.pmax_w * power_factor

    def estimate_annual_energy_kwh(
        self,
        annual_irradiation_kwh_m2: float,
        performance_ratio: float = 0.80,
    ) -> float:
        """
        Estimate annual energy production.

        Args:
            annual_irradiation_kwh_m2: Annual irradiation in kWh/m²
            performance_ratio: System performance ratio (0-1)

        Returns:
            float: Estimated annual energy in kWh

        Raises:
            ValueError: If inputs are invalid
        """
        if annual_irradiation_kwh_m2 <= 0:
            raise ValueError("Annual irradiation must be positive")
        if not (0 < performance_ratio <= 1):
            raise ValueError("Performance ratio must be between 0 and 1")

        area_m2 = self.mechanical.calculate_area_m2()
        module_efficiency = self.electrical.efficiency_percentage / 100.0

        annual_energy_kwh = (
            annual_irradiation_kwh_m2 * area_m2 * module_efficiency * performance_ratio
        )

        return annual_energy_kwh
