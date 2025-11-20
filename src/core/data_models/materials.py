"""
Material data models for photovoltaic cell simulation.

This module defines the Material class and related optical/electrical properties
for semiconductor materials, TCOs, passivation layers, and metals.
"""

from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
from enum import Enum


class MaterialType(str, Enum):
    """Types of materials used in PV cells."""
    SEMICONDUCTOR = "semiconductor"
    TCO = "tco"
    PASSIVATION = "passivation"
    CONTACT = "contact"
    METAL = "metal"
    DIELECTRIC = "dielectric"


class DopingType(str, Enum):
    """Doping type for semiconductor materials."""
    N_TYPE = "n"
    P_TYPE = "p"
    INTRINSIC = "i"


class OpticalProperties(BaseModel):
    """Optical properties of a material."""

    wavelengths: List[float] = Field(
        default_factory=list,
        description="Wavelength points (nm)"
    )
    n: List[float] = Field(
        default_factory=list,
        description="Refractive index (real part)"
    )
    k: List[float] = Field(
        default_factory=list,
        description="Extinction coefficient (imaginary part)"
    )
    alpha: Optional[List[float]] = Field(
        None,
        description="Absorption coefficient (cm⁻¹)"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_n(self, wavelength: float) -> float:
        """Get refractive index at specific wavelength via interpolation."""
        if not self.wavelengths or not self.n:
            return 1.0
        return float(np.interp(wavelength, self.wavelengths, self.n))

    def get_k(self, wavelength: float) -> float:
        """Get extinction coefficient at specific wavelength."""
        if not self.wavelengths or not self.k:
            return 0.0
        return float(np.interp(wavelength, self.wavelengths, self.k))

    def get_alpha(self, wavelength: float) -> float:
        """Get absorption coefficient at specific wavelength."""
        if self.alpha and self.wavelengths:
            return float(np.interp(wavelength, self.wavelengths, self.alpha))
        # Calculate from extinction coefficient: α = 4πk/λ
        k_val = self.get_k(wavelength)
        return 4 * np.pi * k_val / (wavelength * 1e-7)  # Convert nm to cm


class ElectricalProperties(BaseModel):
    """Electrical properties of semiconductor materials."""

    electron_mobility: float = Field(
        1400.0,
        description="Electron mobility (cm²/V·s)",
        gt=0
    )
    hole_mobility: float = Field(
        450.0,
        description="Hole mobility (cm²/V·s)",
        gt=0
    )
    electron_lifetime: float = Field(
        1e-6,
        description="Electron lifetime (s)",
        gt=0
    )
    hole_lifetime: float = Field(
        1e-6,
        description="Hole lifetime (s)",
        gt=0
    )
    nc: float = Field(
        2.8e19,
        description="Effective density of states in conduction band (cm⁻³)",
        gt=0
    )
    nv: float = Field(
        1.04e19,
        description="Effective density of states in valence band (cm⁻³)",
        gt=0
    )
    electron_thermal_velocity: float = Field(
        1e7,
        description="Electron thermal velocity (cm/s)",
        gt=0
    )
    hole_thermal_velocity: float = Field(
        1e7,
        description="Hole thermal velocity (cm/s)",
        gt=0
    )

    class Config:
        arbitrary_types_allowed = True


class Material(BaseModel):
    """
    Complete material definition for PV simulation.

    This class encapsulates all physical, optical, and electrical properties
    needed for device simulation.
    """

    name: str = Field(..., description="Material name (e.g., 'Si', 'ITO')")

    material_type: MaterialType = Field(
        MaterialType.SEMICONDUCTOR,
        description="Type of material"
    )

    # Basic properties
    bandgap: float = Field(
        1.12,
        description="Bandgap energy (eV)",
        ge=0
    )

    electron_affinity: float = Field(
        4.05,
        description="Electron affinity (eV)",
        ge=0
    )

    dielectric_constant: float = Field(
        11.9,
        description="Relative dielectric constant",
        gt=0
    )

    density: Optional[float] = Field(
        2.33,
        description="Material density (g/cm³)",
        gt=0
    )

    # Optical properties
    optical: Optional[OpticalProperties] = Field(
        None,
        description="Wavelength-dependent optical properties"
    )

    # Electrical properties
    electrical: Optional[ElectricalProperties] = Field(
        None,
        description="Electrical transport properties"
    )

    # Work function for metals
    work_function: Optional[float] = Field(
        None,
        description="Work function (eV) - for metals",
        ge=0
    )

    # Conductivity for TCOs and metals
    conductivity: Optional[float] = Field(
        None,
        description="Electrical conductivity (S/cm)",
        ge=0
    )

    # Surface recombination velocity
    surface_recombination_velocity: Optional[float] = Field(
        None,
        description="Surface recombination velocity (cm/s)",
        ge=0
    )

    # Additional metadata
    description: Optional[str] = Field(
        None,
        description="Material description"
    )

    reference: Optional[str] = Field(
        None,
        description="Data source/reference"
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

    @validator('work_function')
    def validate_work_function(cls, v, values):
        """Ensure work function is set for metals."""
        if values.get('material_type') == MaterialType.METAL and v is None:
            return 4.5  # Default work function
        return v

    def get_conduction_band_edge(self) -> float:
        """Get conduction band edge energy (eV) relative to vacuum."""
        return -self.electron_affinity

    def get_valence_band_edge(self) -> float:
        """Get valence band edge energy (eV) relative to vacuum."""
        return -self.electron_affinity - self.bandgap

    def to_dict(self) -> Dict[str, Any]:
        """Convert material to dictionary representation."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Create Material instance from dictionary."""
        return cls(**data)


class MaterialDatabase:
    """
    Database of materials for PV cell simulation.

    Provides CRUD operations for material library management.
    """

    def __init__(self):
        """Initialize empty material database."""
        self._materials: Dict[str, Material] = {}

    def add_material(self, material: Material) -> None:
        """
        Add a material to the database.

        Args:
            material: Material instance to add
        """
        self._materials[material.name] = material

    def get_material(self, name: str) -> Optional[Material]:
        """
        Retrieve a material by name.

        Args:
            name: Material name

        Returns:
            Material instance or None if not found
        """
        return self._materials.get(name)

    def remove_material(self, name: str) -> bool:
        """
        Remove a material from the database.

        Args:
            name: Material name

        Returns:
            True if removed, False if not found
        """
        if name in self._materials:
            del self._materials[name]
            return True
        return False

    def list_materials(
        self,
        material_type: Optional[MaterialType] = None
    ) -> List[str]:
        """
        List all materials, optionally filtered by type.

        Args:
            material_type: Filter by material type

        Returns:
            List of material names
        """
        if material_type is None:
            return list(self._materials.keys())
        return [
            name for name, mat in self._materials.items()
            if mat.material_type == material_type
        ]

    def get_all_materials(self) -> Dict[str, Material]:
        """Get all materials in the database."""
        return self._materials.copy()

    def load_from_dict(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Load materials from dictionary representation.

        Args:
            data: Dictionary mapping material names to properties
        """
        for name, props in data.items():
            material = Material.from_dict(props)
            self.add_material(material)

    def save_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Save materials to dictionary representation.

        Returns:
            Dictionary mapping material names to properties
        """
        return {
            name: mat.to_dict()
            for name, mat in self._materials.items()
        }
