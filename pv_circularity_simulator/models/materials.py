"""
Material models for PV circularity simulator.

This module defines comprehensive Pydantic models for materials used in
photovoltaic cells and modules, including:
- Base material properties
- Silicon materials (mono/polycrystalline)
- Passivation materials
- Contact materials (metallic contacts)
- Material composition and properties

All models include full validation for physical constraints and
production-ready error handling.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from pv_circularity_simulator.models.core import NamedModel, QuantityModel, UUIDModel


class MaterialType(str, Enum):
    """
    Enumeration of material types used in PV cells.

    This enum categorizes materials by their primary function
    in the photovoltaic cell structure.
    """

    SILICON = "silicon"
    PASSIVATION = "passivation"
    CONTACT = "contact"
    ANTI_REFLECTIVE = "anti_reflective"
    ENCAPSULATION = "encapsulation"
    BACKSHEET = "backsheet"
    GLASS = "glass"
    FRAME = "frame"
    JUNCTION_BOX = "junction_box"
    ADHESIVE = "adhesive"
    OTHER = "other"


class CrystalType(str, Enum):
    """
    Enumeration of silicon crystal types.

    Defines the crystal structure of silicon materials,
    which significantly affects cell performance.
    """

    MONOCRYSTALLINE = "monocrystalline"
    POLYCRYSTALLINE = "polycrystalline"
    AMORPHOUS = "amorphous"
    MICROCRYSTALLINE = "microcrystalline"
    RIBBON = "ribbon"


class MaterialProperties(NamedModel):
    """
    Physical and chemical properties of a material.

    This model captures the essential properties that affect
    PV cell performance and manufacturability.

    Attributes:
        density_kg_m3: Material density in kg/m³
        thermal_conductivity_w_mk: Thermal conductivity in W/(m·K)
        specific_heat_j_kgk: Specific heat capacity in J/(kg·K)
        resistivity_ohm_m: Electrical resistivity in Ω·m (None for insulators)
        band_gap_ev: Band gap energy in eV (None for metals)
        refractive_index: Refractive index (dimensionless)
        absorption_coefficient_m1: Absorption coefficient in m⁻¹
        recyclability_percentage: Material recyclability (0-100%)
        environmental_impact_kg_co2_eq: Environmental impact in kg CO₂ equivalent
    """

    density_kg_m3: float = Field(
        ...,
        gt=0,
        description="Material density in kg/m³",
    )
    thermal_conductivity_w_mk: float = Field(
        ...,
        gt=0,
        description="Thermal conductivity in W/(m·K)",
    )
    specific_heat_j_kgk: float = Field(
        ...,
        gt=0,
        description="Specific heat capacity in J/(kg·K)",
    )
    resistivity_ohm_m: Optional[float] = Field(
        None,
        ge=0,
        description="Electrical resistivity in Ω·m (None for insulators)",
    )
    band_gap_ev: Optional[float] = Field(
        None,
        ge=0,
        lt=10,
        description="Band gap energy in eV (None for metals)",
    )
    refractive_index: Optional[float] = Field(
        None,
        gt=1.0,
        lt=5.0,
        description="Refractive index (dimensionless)",
    )
    absorption_coefficient_m1: Optional[float] = Field(
        None,
        ge=0,
        description="Absorption coefficient in m⁻¹",
    )
    recyclability_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Material recyclability percentage (0-100%)",
    )
    environmental_impact_kg_co2_eq: float = Field(
        default=0.0,
        ge=0.0,
        description="Environmental impact in kg CO₂ equivalent per kg of material",
    )

    @field_validator("density_kg_m3")
    @classmethod
    def validate_density(cls, v: float) -> float:
        """Validate density is within realistic range for PV materials."""
        if v > 30000:  # Heavier than osmium
            raise ValueError("Density exceeds maximum realistic value (30000 kg/m³)")
        return v

    @field_validator("thermal_conductivity_w_mk")
    @classmethod
    def validate_thermal_conductivity(cls, v: float) -> float:
        """Validate thermal conductivity is within realistic range."""
        if v > 500:  # Higher than diamond
            raise ValueError("Thermal conductivity exceeds maximum realistic value (500 W/(m·K))")
        return v


class MaterialModel(UUIDModel):
    """
    Comprehensive model for materials used in PV cells and modules.

    This model represents a material with its type, composition,
    properties, and lifecycle information. It serves as the base
    for all material-specific models.

    Attributes:
        name: Human-readable name of the material
        material_type: Type/category of the material
        composition: Chemical composition (element: percentage)
        properties: Physical and chemical properties
        supplier: Optional supplier information
        cost_per_kg_usd: Cost per kilogram in USD
        lead_time_days: Lead time for procurement in days
        minimum_order_kg: Minimum order quantity in kg
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name of the material",
    )
    material_type: MaterialType = Field(
        ...,
        description="Type/category of the material",
    )
    composition: Dict[str, float] = Field(
        default_factory=dict,
        description="Chemical composition as element: weight percentage (must sum to ~100%)",
    )
    properties: MaterialProperties = Field(
        ...,
        description="Physical and chemical properties of the material",
    )
    supplier: Optional[str] = Field(
        None,
        max_length=255,
        description="Supplier or manufacturer name",
    )
    cost_per_kg_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per kilogram in USD",
    )
    lead_time_days: int = Field(
        default=0,
        ge=0,
        description="Lead time for procurement in days",
    )
    minimum_order_kg: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum order quantity in kg",
    )

    @field_validator("composition")
    @classmethod
    def validate_composition(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that composition percentages are valid and sum to ~100%."""
        if not v:
            return v

        # Check each element percentage is valid
        for element, percentage in v.items():
            if percentage < 0 or percentage > 100:
                raise ValueError(
                    f"Composition percentage for {element} must be between 0 and 100%"
                )

        # Check total is approximately 100% (allow 1% tolerance for impurities)
        total = sum(v.values())
        if abs(total - 100.0) > 1.0:
            raise ValueError(
                f"Composition percentages must sum to approximately 100% (got {total:.2f}%)"
            )

        return v

    def calculate_mass(self, volume_m3: float) -> float:
        """
        Calculate mass based on volume and density.

        Args:
            volume_m3: Volume in cubic meters

        Returns:
            float: Mass in kilograms

        Raises:
            ValueError: If volume is negative
        """
        if volume_m3 < 0:
            raise ValueError("Volume cannot be negative")
        return volume_m3 * self.properties.density_kg_m3

    def calculate_cost(self, mass_kg: float) -> float:
        """
        Calculate total cost for a given mass.

        Args:
            mass_kg: Mass in kilograms

        Returns:
            float: Total cost in USD

        Raises:
            ValueError: If mass is negative
        """
        if mass_kg < 0:
            raise ValueError("Mass cannot be negative")
        return mass_kg * self.cost_per_kg_usd


class SiliconMaterial(MaterialModel):
    """
    Silicon material model for PV cells.

    Silicon is the primary semiconductor material in most PV cells.
    This model captures silicon-specific properties including
    crystal type, purity, and doping characteristics.

    Attributes:
        crystal_type: Type of crystal structure
        purity_percentage: Silicon purity (99.0-99.9999%)
        doping_type: N-type or P-type doping
        doping_concentration_cm3: Dopant concentration in cm⁻³
        minority_carrier_lifetime_us: Minority carrier lifetime in microseconds
        wafer_thickness_um: Wafer thickness in micrometers
    """

    crystal_type: CrystalType = Field(
        ...,
        description="Type of crystal structure (mono/poly/amorphous)",
    )
    purity_percentage: float = Field(
        ...,
        ge=99.0,
        le=99.9999,
        description="Silicon purity percentage (99.0-99.9999%)",
    )
    doping_type: str = Field(
        ...,
        pattern="^[NP]-type$",
        description="Doping type (N-type or P-type)",
    )
    doping_concentration_cm3: float = Field(
        ...,
        gt=0,
        description="Dopant concentration in cm⁻³",
    )
    minority_carrier_lifetime_us: float = Field(
        ...,
        gt=0,
        description="Minority carrier lifetime in microseconds",
    )
    wafer_thickness_um: float = Field(
        default=180.0,
        gt=0,
        le=500,
        description="Wafer thickness in micrometers (typical: 120-200 µm)",
    )

    @model_validator(mode="after")
    def validate_silicon_material(self) -> "SiliconMaterial":
        """Validate that material type is silicon and band gap is appropriate."""
        if self.material_type != MaterialType.SILICON:
            raise ValueError("SiliconMaterial must have material_type = SILICON")

        # Silicon band gap should be around 1.12 eV at room temperature
        if self.properties.band_gap_ev is not None:
            if not (1.0 <= self.properties.band_gap_ev <= 1.2):
                raise ValueError(
                    f"Silicon band gap should be ~1.12 eV (got {self.properties.band_gap_ev} eV)"
                )

        return self

    @field_validator("doping_concentration_cm3")
    @classmethod
    def validate_doping_concentration(cls, v: float) -> float:
        """Validate doping concentration is within realistic range."""
        if v < 1e10 or v > 1e20:
            raise ValueError(
                "Doping concentration must be between 1e10 and 1e20 cm⁻³"
            )
        return v


class PassivationMaterial(MaterialModel):
    """
    Passivation material model for PV cells.

    Passivation layers reduce surface recombination and improve
    cell efficiency. Common materials include SiO₂, SiNₓ, and Al₂O₃.

    Attributes:
        layer_thickness_nm: Thickness of passivation layer in nanometers
        surface_recombination_velocity_cm_s: SRV in cm/s (lower is better)
        deposition_method: Method used for deposition (PECVD, ALD, thermal, etc.)
        annealing_temperature_c: Annealing temperature in Celsius
    """

    layer_thickness_nm: float = Field(
        ...,
        gt=0,
        le=500,
        description="Thickness of passivation layer in nanometers",
    )
    surface_recombination_velocity_cm_s: float = Field(
        ...,
        ge=0,
        description="Surface recombination velocity in cm/s (lower is better)",
    )
    deposition_method: str = Field(
        ...,
        max_length=50,
        description="Deposition method (PECVD, ALD, thermal oxidation, etc.)",
    )
    annealing_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=1500,
        description="Annealing temperature in Celsius",
    )

    @model_validator(mode="after")
    def validate_passivation_material(self) -> "PassivationMaterial":
        """Validate that material type is passivation."""
        if self.material_type != MaterialType.PASSIVATION:
            raise ValueError("PassivationMaterial must have material_type = PASSIVATION")
        return self

    @field_validator("surface_recombination_velocity_cm_s")
    @classmethod
    def validate_srv(cls, v: float) -> float:
        """Validate SRV is within realistic range."""
        if v > 1e6:
            raise ValueError(
                "Surface recombination velocity exceeds realistic maximum (1e6 cm/s)"
            )
        return v


class ContactMaterial(MaterialModel):
    """
    Contact material model for PV cells.

    Contact materials provide electrical connections to the cell.
    Common materials include silver, aluminum, copper, and their alloys.

    Attributes:
        conductivity_s_m: Electrical conductivity in S/m (Siemens per meter)
        contact_resistance_ohm_cm2: Contact resistance in Ω·cm²
        metal_type: Primary metal used (Ag, Al, Cu, etc.)
        layer_thickness_um: Thickness of contact layer in micrometers
        finger_width_um: Width of contact fingers in micrometers
        finger_spacing_mm: Spacing between fingers in millimeters
    """

    conductivity_s_m: float = Field(
        ...,
        gt=0,
        description="Electrical conductivity in S/m (Siemens per meter)",
    )
    contact_resistance_ohm_cm2: float = Field(
        ...,
        gt=0,
        description="Contact resistance in Ω·cm²",
    )
    metal_type: str = Field(
        ...,
        max_length=20,
        description="Primary metal type (Ag, Al, Cu, etc.)",
    )
    layer_thickness_um: float = Field(
        ...,
        gt=0,
        le=100,
        description="Thickness of contact layer in micrometers",
    )
    finger_width_um: float = Field(
        default=50.0,
        gt=0,
        le=500,
        description="Width of contact fingers in micrometers",
    )
    finger_spacing_mm: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Spacing between contact fingers in millimeters",
    )

    @model_validator(mode="after")
    def validate_contact_material(self) -> "ContactMaterial":
        """Validate that material type is contact."""
        if self.material_type != MaterialType.CONTACT:
            raise ValueError("ContactMaterial must have material_type = CONTACT")
        return self

    @field_validator("conductivity_s_m")
    @classmethod
    def validate_conductivity(cls, v: float) -> float:
        """Validate conductivity is within realistic range for metals."""
        # Silver has highest conductivity ~6.3e7 S/m
        if v > 1e8:
            raise ValueError(
                "Conductivity exceeds realistic maximum for metals (1e8 S/m)"
            )
        return v

    def calculate_resistance(self, length_m: float, cross_section_m2: float) -> float:
        """
        Calculate electrical resistance based on geometry.

        Args:
            length_m: Length of conductor in meters
            cross_section_m2: Cross-sectional area in m²

        Returns:
            float: Resistance in Ohms

        Raises:
            ValueError: If inputs are invalid
        """
        if length_m <= 0:
            raise ValueError("Length must be positive")
        if cross_section_m2 <= 0:
            raise ValueError("Cross-sectional area must be positive")

        resistivity = 1.0 / self.conductivity_s_m
        return resistivity * length_m / cross_section_m2
