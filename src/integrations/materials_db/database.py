"""
Materials database loader and manager.

This module provides functionality to load and manage the materials database
from YAML/JSON files and provides a centralized access point.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_models.materials import (
    Material,
    MaterialDatabase,
    MaterialType,
    OpticalProperties,
    ElectricalProperties
)


def load_default_materials() -> MaterialDatabase:
    """
    Load default materials database with common PV materials.

    Returns:
        MaterialDatabase instance with preloaded materials
    """
    db = MaterialDatabase()

    # Silicon materials
    si_p_plus = Material(
        name="Si(p+)",
        material_type=MaterialType.SEMICONDUCTOR,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.9,
        density=2.33,
        electrical=ElectricalProperties(
            electron_mobility=450,
            hole_mobility=150,
            electron_lifetime=1e-7,
            hole_lifetime=1e-6,
            nc=2.8e19,
            nv=1.04e19,
        ),
        description="Heavily p-doped crystalline silicon (BSF, substrate)"
    )

    si_p = Material(
        name="Si(p)",
        material_type=MaterialType.SEMICONDUCTOR,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.9,
        density=2.33,
        electrical=ElectricalProperties(
            electron_mobility=1000,
            hole_mobility=400,
            electron_lifetime=1e-5,
            hole_lifetime=1e-4,
            nc=2.8e19,
            nv=1.04e19,
        ),
        description="Lightly p-doped crystalline silicon (base)"
    )

    si_n = Material(
        name="Si(n)",
        material_type=MaterialType.SEMICONDUCTOR,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.9,
        density=2.33,
        electrical=ElectricalProperties(
            electron_mobility=1400,
            hole_mobility=450,
            electron_lifetime=1e-6,
            hole_lifetime=1e-6,
            nc=2.8e19,
            nv=1.04e19,
        ),
        description="Lightly n-doped crystalline silicon"
    )

    si_n_plus = Material(
        name="Si(n+)",
        material_type=MaterialType.SEMICONDUCTOR,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.9,
        density=2.33,
        electrical=ElectricalProperties(
            electron_mobility=1100,
            hole_mobility=400,
            electron_lifetime=1e-7,
            hole_lifetime=1e-7,
            nc=2.8e19,
            nv=1.04e19,
        ),
        description="Heavily n-doped crystalline silicon (emitter)"
    )

    # Amorphous silicon
    a_si_i = Material(
        name="a-Si:H(i)",
        material_type=MaterialType.PASSIVATION,
        bandgap=1.7,
        electron_affinity=3.9,
        dielectric_constant=11.8,
        density=2.0,
        electrical=ElectricalProperties(
            electron_mobility=20,
            hole_mobility=5,
            electron_lifetime=1e-8,
            hole_lifetime=1e-8,
            nc=2.5e20,
            nv=2.5e20,
        ),
        surface_recombination_velocity=10.0,
        description="Intrinsic amorphous silicon (passivation)"
    )

    a_si_n = Material(
        name="a-Si:H(n)",
        material_type=MaterialType.CONTACT,
        bandgap=1.7,
        electron_affinity=3.9,
        dielectric_constant=11.8,
        density=2.0,
        electrical=ElectricalProperties(
            electron_mobility=20,
            hole_mobility=5,
            electron_lifetime=1e-9,
            hole_lifetime=1e-9,
            nc=2.5e20,
            nv=2.5e20,
        ),
        description="n-type amorphous silicon contact"
    )

    a_si_p = Material(
        name="a-Si:H(p)",
        material_type=MaterialType.CONTACT,
        bandgap=1.7,
        electron_affinity=3.9,
        dielectric_constant=11.8,
        density=2.0,
        electrical=ElectricalProperties(
            electron_mobility=5,
            hole_mobility=2,
            electron_lifetime=1e-9,
            hole_lifetime=1e-9,
            nc=2.5e20,
            nv=2.5e20,
        ),
        description="p-type amorphous silicon contact"
    )

    # Poly-Si contacts (for TOPCon)
    poly_si_n_plus = Material(
        name="Poly-Si(n+)",
        material_type=MaterialType.CONTACT,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.9,
        density=2.33,
        electrical=ElectricalProperties(
            electron_mobility=50,
            hole_mobility=20,
            electron_lifetime=1e-8,
            hole_lifetime=1e-8,
            nc=2.8e19,
            nv=1.04e19,
        ),
        conductivity=100.0,
        description="Heavily n-doped polysilicon (TOPCon)"
    )

    # Transparent Conductive Oxides
    ito = Material(
        name="ITO",
        material_type=MaterialType.TCO,
        bandgap=3.5,
        electron_affinity=4.7,
        dielectric_constant=9.0,
        density=7.12,
        conductivity=1000.0,
        work_function=4.7,
        description="Indium Tin Oxide - transparent conductor"
    )

    azo = Material(
        name="AZO",
        material_type=MaterialType.TCO,
        bandgap=3.3,
        electron_affinity=4.5,
        dielectric_constant=9.0,
        density=5.6,
        conductivity=800.0,
        work_function=4.5,
        description="Aluminum-doped Zinc Oxide"
    )

    # Passivation layers
    sio2 = Material(
        name="SiO2",
        material_type=MaterialType.PASSIVATION,
        bandgap=9.0,
        electron_affinity=0.9,
        dielectric_constant=3.9,
        density=2.65,
        surface_recombination_velocity=10.0,
        description="Silicon dioxide - thermal oxide passivation"
    )

    al2o3 = Material(
        name="Al2O3",
        material_type=MaterialType.PASSIVATION,
        bandgap=7.0,
        electron_affinity=1.5,
        dielectric_constant=9.0,
        density=3.95,
        surface_recombination_velocity=5.0,
        description="Aluminum oxide - excellent passivation for p-type Si"
    )

    sinx = Material(
        name="SiNx",
        material_type=MaterialType.PASSIVATION,
        bandgap=5.0,
        electron_affinity=2.0,
        dielectric_constant=7.5,
        density=3.1,
        surface_recombination_velocity=20.0,
        description="Silicon nitride - ARC and passivation"
    )

    # Metals
    aluminum = Material(
        name="Al",
        material_type=MaterialType.METAL,
        bandgap=0.0,
        electron_affinity=0.0,
        dielectric_constant=1.0,
        density=2.70,
        conductivity=3.77e5,
        work_function=4.28,
        description="Aluminum - rear contact metal"
    )

    silver = Material(
        name="Ag",
        material_type=MaterialType.METAL,
        bandgap=0.0,
        electron_affinity=0.0,
        dielectric_constant=1.0,
        density=10.49,
        conductivity=6.30e5,
        work_function=4.26,
        description="Silver - front grid metal"
    )

    # Add all materials to database
    for material in [
        si_p_plus, si_p, si_n, si_n_plus,
        a_si_i, a_si_n, a_si_p, poly_si_n_plus,
        ito, azo,
        sio2, al2o3, sinx,
        aluminum, silver
    ]:
        db.add_material(material)

    return db


def save_materials_to_json(
    db: MaterialDatabase,
    filepath: Path
) -> None:
    """
    Save materials database to JSON file.

    Args:
        db: MaterialDatabase instance
        filepath: Path to save JSON file
    """
    data = db.save_to_dict()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_materials_from_json(filepath: Path) -> MaterialDatabase:
    """
    Load materials database from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        MaterialDatabase instance
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    db = MaterialDatabase()
    db.load_from_dict(data)
    return db


# Global materials database instance
_global_db: Optional[MaterialDatabase] = None


def get_materials_database() -> MaterialDatabase:
    """
    Get the global materials database instance.

    Returns:
        MaterialDatabase instance with default materials
    """
    global _global_db
    if _global_db is None:
        _global_db = load_default_materials()
    return _global_db


def reload_materials_database() -> MaterialDatabase:
    """
    Force reload of materials database.

    Returns:
        Freshly loaded MaterialDatabase instance
    """
    global _global_db
    _global_db = load_default_materials()
    return _global_db
