"""Material data models and database."""

from typing import Dict, Literal
from pydantic import BaseModel, Field


class Material(BaseModel):
    """Material specification."""

    name: str = Field(..., description="Material name")
    category: Literal["cell", "glass", "encapsulant", "backsheet", "frame", "junction_box", "cable", "connector"] = Field(
        ..., description="Material category"
    )
    density_kg_m3: float = Field(..., gt=0, description="Density in kg/m³")
    cost_per_kg_usd: float = Field(..., gt=0, description="Cost per kg in USD")
    recyclability_pct: float = Field(default=0, ge=0, le=100, description="Recyclability percentage")
    embodied_energy_mj_kg: float = Field(..., gt=0, description="Embodied energy in MJ/kg")
    carbon_footprint_kg_co2_kg: float = Field(..., gt=0, description="Carbon footprint in kg CO2/kg")


class MaterialDatabase:
    """Database of common PV materials."""

    @staticmethod
    def get_default_materials() -> Dict[str, Material]:
        """Get default material database."""
        return {
            "silicon_cell": Material(
                name="Silicon Cell",
                category="cell",
                density_kg_m3=2330,
                cost_per_kg_usd=50.0,
                recyclability_pct=95,
                embodied_energy_mj_kg=800,
                carbon_footprint_kg_co2_kg=45,
            ),
            "glass_front": Material(
                name="Front Glass (3.2mm)",
                category="glass",
                density_kg_m3=2500,
                cost_per_kg_usd=1.5,
                recyclability_pct=100,
                embodied_energy_mj_kg=15,
                carbon_footprint_kg_co2_kg=0.85,
            ),
            "glass_back": Material(
                name="Back Glass (2.0mm)",
                category="glass",
                density_kg_m3=2500,
                cost_per_kg_usd=1.5,
                recyclability_pct=100,
                embodied_energy_mj_kg=15,
                carbon_footprint_kg_co2_kg=0.85,
            ),
            "eva": Material(
                name="EVA Encapsulant",
                category="encapsulant",
                density_kg_m3=960,
                cost_per_kg_usd=8.0,
                recyclability_pct=20,
                embodied_energy_mj_kg=100,
                carbon_footprint_kg_co2_kg=3.5,
            ),
            "poe": Material(
                name="POE Encapsulant",
                category="encapsulant",
                density_kg_m3=870,
                cost_per_kg_usd=12.0,
                recyclability_pct=30,
                embodied_energy_mj_kg=110,
                carbon_footprint_kg_co2_kg=3.8,
            ),
            "backsheet": Material(
                name="PET Backsheet",
                category="backsheet",
                density_kg_m3=1380,
                cost_per_kg_usd=15.0,
                recyclability_pct=40,
                embodied_energy_mj_kg=80,
                carbon_footprint_kg_co2_kg=3.2,
            ),
            "aluminum_frame": Material(
                name="Aluminum Frame",
                category="frame",
                density_kg_m3=2700,
                cost_per_kg_usd=3.5,
                recyclability_pct=95,
                embodied_energy_mj_kg=200,
                carbon_footprint_kg_co2_kg=12,
            ),
            "junction_box": Material(
                name="Junction Box",
                category="junction_box",
                density_kg_m3=1200,
                cost_per_kg_usd=25.0,
                recyclability_pct=60,
                embodied_energy_mj_kg=120,
                carbon_footprint_kg_co2_kg=5,
            ),
            "copper_cable": Material(
                name="Copper Cable",
                category="cable",
                density_kg_m3=8960,
                cost_per_kg_usd=8.0,
                recyclability_pct=95,
                embodied_energy_mj_kg=70,
                carbon_footprint_kg_co2_kg=4,
            ),
        }
