"""
Cell models for PV circularity simulator.

This module defines comprehensive Pydantic models for photovoltaic cells,
including:
- Cell types and architectures
- Geometry and dimensions
- Electrical characteristics
- Performance metrics
- CTM (Contact Transport Mechanism) losses
- Temperature coefficients

All models include full validation for physical constraints and
production-ready error handling.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.core import NamedModel, UUIDModel
from pv_circularity_simulator.models.materials import (
    ContactMaterial,
    PassivationMaterial,
    SiliconMaterial,
)


class CellType(str, Enum):
    """
    Enumeration of photovoltaic cell types.

    Categorizes cells by their fundamental technology
    and semiconductor structure.
    """

    MONOCRYSTALLINE_SILICON = "monocrystalline_silicon"
    POLYCRYSTALLINE_SILICON = "polycrystalline_silicon"
    THIN_FILM_CDTE = "thin_film_cdte"
    THIN_FILM_CIGS = "thin_film_cigs"
    THIN_FILM_AMORPHOUS_SI = "thin_film_amorphous_si"
    PERC = "perc"  # Passivated Emitter and Rear Cell
    TOPCON = "topcon"  # Tunnel Oxide Passivated Contact
    HJT = "hjt"  # Heterojunction Technology
    IBC = "ibc"  # Interdigitated Back Contact
    BIFACIAL = "bifacial"
    MULTI_JUNCTION = "multi_junction"
    PEROVSKITE = "perovskite"
    ORGANIC = "organic"
    OTHER = "other"


class CellArchitecture(str, Enum):
    """
    Enumeration of cell architectures.

    Defines the structural design and contact arrangement
    of the photovoltaic cell.
    """

    STANDARD = "standard"  # Standard full-area Al-BSF
    PERC = "perc"  # Passivated Emitter and Rear Cell
    PERT = "pert"  # Passivated Emitter, Rear Totally diffused
    PERL = "perl"  # Passivated Emitter, Rear Locally diffused
    TOPCON = "topcon"  # Tunnel Oxide Passivated Contact
    HJT = "hjt"  # Heterojunction with Intrinsic Thin layer
    IBC = "ibc"  # Interdigitated Back Contact
    REAR_CONTACT = "rear_contact"  # All contacts on rear
    BIFACIAL = "bifacial"  # Can generate power from both sides


class CellGeometry(NamedModel):
    """
    Geometric specifications of a PV cell.

    Defines the physical dimensions and shape of the cell,
    which affects both performance and manufacturing.

    Attributes:
        width_mm: Cell width in millimeters
        height_mm: Cell height in millimeters
        thickness_um: Cell thickness in micrometers
        area_cm2: Active area in cm² (calculated or specified)
        corner_radius_mm: Radius of rounded corners (0 for square)
        busbar_count: Number of busbars
        busbar_width_mm: Width of each busbar in mm
    """

    width_mm: float = Field(
        ...,
        gt=0,
        le=250,
        description="Cell width in millimeters (typical: 156-210 mm)",
    )
    height_mm: float = Field(
        ...,
        gt=0,
        le=250,
        description="Cell height in millimeters (typical: 156-210 mm)",
    )
    thickness_um: float = Field(
        default=180.0,
        gt=0,
        le=500,
        description="Cell thickness in micrometers (typical: 120-200 µm)",
    )
    area_cm2: Optional[float] = Field(
        None,
        gt=0,
        description="Active cell area in cm² (auto-calculated if not provided)",
    )
    corner_radius_mm: float = Field(
        default=0.0,
        ge=0,
        le=20,
        description="Radius of rounded corners in mm (0 for square corners)",
    )
    busbar_count: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of busbars (0 for busbar-free designs)",
    )
    busbar_width_mm: float = Field(
        default=1.5,
        gt=0,
        le=5,
        description="Width of each busbar in millimeters",
    )

    @model_validator(mode="after")
    def calculate_area(self) -> "CellGeometry":
        """Calculate active area if not provided."""
        if self.area_cm2 is None:
            # Convert mm to cm and calculate area
            width_cm = self.width_mm / 10.0
            height_cm = self.height_mm / 10.0
            self.area_cm2 = width_cm * height_cm

            # Subtract area lost to rounded corners (approximation)
            if self.corner_radius_mm > 0:
                radius_cm = self.corner_radius_mm / 10.0
                # Lost area = 4 * (square - quarter circle)
                lost_area = 4 * (radius_cm**2 - 0.785398 * radius_cm**2)
                self.area_cm2 -= lost_area

        return self

    def calculate_volume_cm3(self) -> float:
        """
        Calculate cell volume in cubic centimeters.

        Returns:
            float: Volume in cm³
        """
        thickness_cm = self.thickness_um / 10000.0  # Convert µm to cm
        return self.area_cm2 * thickness_cm


class CellElectricalCharacteristics(NamedModel):
    """
    Electrical characteristics of a PV cell at standard test conditions (STC).

    STC: 1000 W/m² irradiance, 25°C cell temperature, AM 1.5 spectrum

    Attributes:
        voc_v: Open circuit voltage in volts
        isc_a: Short circuit current in amperes
        vmpp_v: Voltage at maximum power point in volts
        impp_a: Current at maximum power point in amperes
        pmpp_w: Power at maximum power point in watts
        fill_factor: Fill factor (0-1, typically 0.7-0.85)
        efficiency_percentage: Cell efficiency percentage
        series_resistance_ohm: Series resistance in ohms
        shunt_resistance_ohm: Shunt resistance in ohms
        ideality_factor: Diode ideality factor (typically 1-2)
    """

    voc_v: float = Field(
        ...,
        gt=0,
        le=1.5,
        description="Open circuit voltage in volts (typical: 0.5-0.75 V)",
    )
    isc_a: float = Field(
        ...,
        gt=0,
        le=15,
        description="Short circuit current in amperes (depends on cell area)",
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
    pmpp_w: float = Field(
        ...,
        gt=0,
        description="Power at maximum power point in watts",
    )
    fill_factor: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Fill factor (dimensionless, 0-1, typically 0.7-0.85)",
    )
    efficiency_percentage: float = Field(
        ...,
        gt=0,
        le=50,
        description="Cell efficiency percentage (typical: 15-26%)",
    )
    series_resistance_ohm: float = Field(
        default=0.005,
        ge=0,
        le=1.0,
        description="Series resistance in ohms (lower is better)",
    )
    shunt_resistance_ohm: float = Field(
        default=1000.0,
        gt=0,
        description="Shunt resistance in ohms (higher is better)",
    )
    ideality_factor: float = Field(
        default=1.0,
        gt=0,
        le=2.5,
        description="Diode ideality factor (typically 1.0-2.0)",
    )

    @model_validator(mode="after")
    def validate_electrical_consistency(self) -> "CellElectricalCharacteristics":
        """Validate electrical parameters are physically consistent."""
        # Vmpp should be less than Voc (typically 80-90% of Voc)
        if self.vmpp_v >= self.voc_v:
            raise ValueError(f"Vmpp ({self.vmpp_v}V) must be less than Voc ({self.voc_v}V)")

        # Impp should be less than Isc (typically 90-95% of Isc)
        if self.impp_a >= self.isc_a:
            raise ValueError(f"Impp ({self.impp_a}A) must be less than Isc ({self.isc_a}A)")

        # Pmpp should equal Vmpp × Impp (with small tolerance)
        calculated_pmpp = self.vmpp_v * self.impp_a
        if abs(calculated_pmpp - self.pmpp_w) > 0.1:
            raise ValueError(
                f"Pmpp ({self.pmpp_w}W) should equal Vmpp × Impp ({calculated_pmpp:.2f}W)"
            )

        # Fill factor should equal Pmpp / (Voc × Isc)
        calculated_ff = self.pmpp_w / (self.voc_v * self.isc_a)
        if abs(calculated_ff - self.fill_factor) > 0.01:
            raise ValueError(
                f"Fill factor ({self.fill_factor:.3f}) should equal "
                f"Pmpp/(Voc×Isc) ({calculated_ff:.3f})"
            )

        return self

    def calculate_efficiency(self, area_cm2: float, irradiance_w_m2: float = 1000.0) -> float:
        """
        Calculate cell efficiency.

        Args:
            area_cm2: Cell area in cm²
            irradiance_w_m2: Irradiance in W/m² (default: 1000 W/m² for STC)

        Returns:
            float: Efficiency as percentage

        Raises:
            ValueError: If inputs are invalid
        """
        if area_cm2 <= 0:
            raise ValueError("Cell area must be positive")
        if irradiance_w_m2 <= 0:
            raise ValueError("Irradiance must be positive")

        area_m2 = area_cm2 / 10000.0  # Convert cm² to m²
        incident_power_w = irradiance_w_m2 * area_m2
        efficiency = (self.pmpp_w / incident_power_w) * 100.0

        return efficiency


class CellDesign(NamedModel):
    """
    Complete cell design specification.

    This model captures the full design of a PV cell including
    materials, geometry, and expected performance.

    Attributes:
        architecture: Cell architecture type
        front_contact_fraction: Fraction of front surface covered by contacts (0-1)
        rear_contact_fraction: Fraction of rear surface covered by contacts (0-1)
        texture_type: Type of surface texturing (pyramid, random, etc.)
        anti_reflective_coating: Whether AR coating is present
    """

    architecture: CellArchitecture = Field(
        ...,
        description="Cell architecture type",
    )
    front_contact_fraction: float = Field(
        default=0.05,
        ge=0,
        le=0.3,
        description="Fraction of front surface covered by contacts (0-1)",
    )
    rear_contact_fraction: float = Field(
        default=0.8,
        ge=0,
        le=1.0,
        description="Fraction of rear surface covered by contacts (0-1)",
    )
    texture_type: str = Field(
        default="pyramid",
        max_length=50,
        description="Type of surface texturing (pyramid, random, inverted, etc.)",
    )
    anti_reflective_coating: bool = Field(
        default=True,
        description="Whether anti-reflective coating is present",
    )

    @field_validator("front_contact_fraction")
    @classmethod
    def validate_front_contact_fraction(cls, v: float) -> float:
        """Validate front contact fraction is realistic."""
        if v > 0.15:
            # Front contacts should typically be < 10% to minimize shading
            import warnings
            warnings.warn(
                f"Front contact fraction {v:.1%} is high and may cause excessive shading losses"
            )
        return v


class CellModel(UUIDModel):
    """
    Comprehensive photovoltaic cell model.

    This model represents a complete PV cell with all its characteristics,
    including materials, geometry, electrical properties, and performance.

    Attributes:
        name: Human-readable name/identifier for the cell
        cell_type: Type of PV cell technology
        substrate_material: Base silicon or semiconductor material
        passivation_front: Front surface passivation material (optional)
        passivation_rear: Rear surface passivation material (optional)
        contact_front: Front contact material
        contact_rear: Rear contact material
        geometry: Physical dimensions and geometry
        electrical: Electrical characteristics at STC
        design: Cell design specifications
        manufacturer: Manufacturer name (optional)
        manufacturing_date: Date of manufacture (optional)
        certification: Certification information (optional)
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name/identifier for the cell",
    )
    cell_type: CellType = Field(
        ...,
        description="Type of PV cell technology",
    )
    substrate_material: SiliconMaterial = Field(
        ...,
        description="Base silicon or semiconductor material",
    )
    passivation_front: Optional[PassivationMaterial] = Field(
        None,
        description="Front surface passivation material",
    )
    passivation_rear: Optional[PassivationMaterial] = Field(
        None,
        description="Rear surface passivation material",
    )
    contact_front: ContactMaterial = Field(
        ...,
        description="Front contact material (fingers and busbars)",
    )
    contact_rear: ContactMaterial = Field(
        ...,
        description="Rear contact material",
    )
    geometry: CellGeometry = Field(
        ...,
        description="Physical dimensions and geometry of the cell",
    )
    electrical: CellElectricalCharacteristics = Field(
        ...,
        description="Electrical characteristics at standard test conditions",
    )
    design: CellDesign = Field(
        ...,
        description="Cell design specifications and architecture",
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=255,
        description="Manufacturer name",
    )
    manufacturing_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Manufacturing date in YYYY-MM-DD format",
    )
    certification: Optional[Dict[str, str]] = Field(
        None,
        description="Certification information (type: certificate_number)",
    )

    @model_validator(mode="after")
    def validate_cell_consistency(self) -> "CellModel":
        """Validate consistency across cell components."""
        # Verify efficiency calculation matches electrical characteristics
        calculated_eff = self.electrical.calculate_efficiency(self.geometry.area_cm2)
        if abs(calculated_eff - self.electrical.efficiency_percentage) > 0.5:
            raise ValueError(
                f"Efficiency mismatch: specified {self.electrical.efficiency_percentage:.2f}%, "
                f"calculated {calculated_eff:.2f}%"
            )

        # Verify cell type matches substrate material
        if self.cell_type == CellType.MONOCRYSTALLINE_SILICON:
            from pv_circularity_simulator.models.materials import CrystalType
            if self.substrate_material.crystal_type != CrystalType.MONOCRYSTALLINE:
                raise ValueError(
                    "Monocrystalline cell must use monocrystalline silicon substrate"
                )

        return self

    def calculate_power_density(self) -> float:
        """
        Calculate power density in W/m².

        Returns:
            float: Power density in watts per square meter
        """
        area_m2 = self.geometry.area_cm2 / 10000.0
        return self.electrical.pmpp_w / area_m2

    def estimate_annual_energy_kwh(
        self,
        peak_sun_hours_per_day: float = 5.0,
        performance_ratio: float = 0.75,
    ) -> float:
        """
        Estimate annual energy production for a single cell.

        Args:
            peak_sun_hours_per_day: Average peak sun hours per day
            performance_ratio: System performance ratio (accounts for all losses)

        Returns:
            float: Estimated annual energy production in kWh

        Raises:
            ValueError: If inputs are out of realistic range
        """
        if not (0 < peak_sun_hours_per_day <= 12):
            raise ValueError("Peak sun hours must be between 0 and 12")
        if not (0 < performance_ratio <= 1):
            raise ValueError("Performance ratio must be between 0 and 1")

        daily_energy_kwh = (
            self.electrical.pmpp_w / 1000.0
        ) * peak_sun_hours_per_day * performance_ratio
        annual_energy_kwh = daily_energy_kwh * 365.0

        return annual_energy_kwh
