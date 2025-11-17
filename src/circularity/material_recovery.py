"""
Material Recovery & Recycling Economics (B11-S01)

This module provides tools for analyzing material recovery from end-of-life PV modules,
including metal, glass, and silicon recovery rates and associated recycling costs.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ModuleComposition(BaseModel):
    """Composition of a PV module by material type (in kg)."""

    glass: float = Field(ge=0, description="Glass mass in kg")
    aluminum: float = Field(ge=0, description="Aluminum frame mass in kg")
    silicon: float = Field(ge=0, description="Silicon wafer mass in kg")
    silver: float = Field(ge=0, description="Silver contact mass in kg")
    copper: float = Field(ge=0, description="Copper wiring mass in kg")
    eva_polymer: float = Field(ge=0, description="EVA encapsulant mass in kg")
    backsheet: float = Field(ge=0, description="Backsheet mass in kg")
    junction_box: float = Field(ge=0, description="Junction box mass in kg")
    other: float = Field(default=0, ge=0, description="Other materials mass in kg")

    @property
    def total_mass(self) -> float:
        """Total module mass in kg."""
        return (
            self.glass + self.aluminum + self.silicon + self.silver +
            self.copper + self.eva_polymer + self.backsheet +
            self.junction_box + self.other
        )


class RecoveryRates(BaseModel):
    """Material recovery efficiency rates (0-1 scale)."""

    glass: float = Field(ge=0, le=1, default=0.95, description="Glass recovery rate")
    aluminum: float = Field(ge=0, le=1, default=0.98, description="Aluminum recovery rate")
    silicon: float = Field(ge=0, le=1, default=0.85, description="Silicon recovery rate")
    silver: float = Field(ge=0, le=1, default=0.90, description="Silver recovery rate")
    copper: float = Field(ge=0, le=1, default=0.95, description="Copper recovery rate")


class RecyclingCostStructure(BaseModel):
    """Cost structure for recycling operations (in USD)."""

    collection_cost_per_module: float = Field(ge=0, description="Collection cost per module")
    transportation_cost_per_km: float = Field(ge=0, description="Transportation cost per km")
    dismantling_cost_per_module: float = Field(ge=0, description="Dismantling cost per module")
    thermal_treatment_cost_per_kg: float = Field(ge=0, description="Thermal treatment cost per kg")
    chemical_treatment_cost_per_kg: float = Field(ge=0, description="Chemical treatment cost per kg")
    mechanical_separation_cost_per_kg: float = Field(ge=0, description="Mechanical separation cost per kg")
    waste_disposal_cost_per_kg: float = Field(ge=0, description="Waste disposal cost per kg")
    facility_overhead_rate: float = Field(ge=0, le=1, default=0.2, description="Facility overhead as fraction of direct costs")


class RecoveredMaterials(BaseModel):
    """Recovered materials from recycling process."""

    glass_kg: float = Field(ge=0, description="Recovered glass in kg")
    aluminum_kg: float = Field(ge=0, description="Recovered aluminum in kg")
    silicon_kg: float = Field(ge=0, description="Recovered silicon in kg")
    silver_kg: float = Field(ge=0, description="Recovered silver in kg")
    copper_kg: float = Field(ge=0, description="Recovered copper in kg")
    waste_kg: float = Field(ge=0, description="Non-recoverable waste in kg")

    @property
    def total_recovered(self) -> float:
        """Total recovered material mass in kg."""
        return self.glass_kg + self.aluminum_kg + self.silicon_kg + self.silver_kg + self.copper_kg

    @property
    def recovery_rate(self) -> float:
        """Overall recovery rate."""
        total = self.total_recovered + self.waste_kg
        return self.total_recovered / total if total > 0 else 0.0


class RecyclingCostBreakdown(BaseModel):
    """Detailed breakdown of recycling costs."""

    collection_cost: float = Field(ge=0, description="Collection cost")
    transportation_cost: float = Field(ge=0, description="Transportation cost")
    dismantling_cost: float = Field(ge=0, description="Dismantling cost")
    thermal_treatment_cost: float = Field(ge=0, description="Thermal treatment cost")
    chemical_treatment_cost: float = Field(ge=0, description="Chemical treatment cost")
    mechanical_separation_cost: float = Field(ge=0, description="Mechanical separation cost")
    waste_disposal_cost: float = Field(ge=0, description="Waste disposal cost")
    overhead_cost: float = Field(ge=0, description="Overhead cost")

    @property
    def total_cost(self) -> float:
        """Total recycling cost."""
        return (
            self.collection_cost + self.transportation_cost +
            self.dismantling_cost + self.thermal_treatment_cost +
            self.chemical_treatment_cost + self.mechanical_separation_cost +
            self.waste_disposal_cost + self.overhead_cost
        )

    @property
    def cost_per_kg(self) -> float:
        """Cost per kg of material processed."""
        return self.total_cost


class MaterialRecoveryCalculator:
    """
    Calculator for material recovery from end-of-life PV modules.

    Provides methods for calculating recovery rates and costs for different
    materials including metals, glass, and silicon.
    """

    def __init__(
        self,
        recovery_rates: Optional[RecoveryRates] = None,
        cost_structure: Optional[RecyclingCostStructure] = None
    ):
        """
        Initialize the material recovery calculator.

        Args:
            recovery_rates: Material recovery efficiency rates
            cost_structure: Cost structure for recycling operations
        """
        self.recovery_rates = recovery_rates or RecoveryRates()
        self.cost_structure = cost_structure or self._default_cost_structure()

    @staticmethod
    def _default_cost_structure() -> RecyclingCostStructure:
        """Default cost structure based on industry data."""
        return RecyclingCostStructure(
            collection_cost_per_module=5.0,
            transportation_cost_per_km=0.5,
            dismantling_cost_per_module=8.0,
            thermal_treatment_cost_per_kg=0.8,
            chemical_treatment_cost_per_kg=1.5,
            mechanical_separation_cost_per_kg=0.3,
            waste_disposal_cost_per_kg=0.2,
            facility_overhead_rate=0.2
        )

    def metal_recovery(
        self,
        composition: ModuleComposition,
        recovery_method: str = "combined"
    ) -> Dict[str, float]:
        """
        Calculate metal recovery from PV module.

        Args:
            composition: Module material composition
            recovery_method: Recovery method ('thermal', 'chemical', 'mechanical', 'combined')

        Returns:
            Dictionary with recovered metal masses and recovery rates
        """
        # Adjust recovery rates based on method
        method_efficiency = {
            "thermal": 0.85,
            "chemical": 0.95,
            "mechanical": 0.75,
            "combined": 1.0
        }

        efficiency = method_efficiency.get(recovery_method, 1.0)

        # Calculate recovered metals
        recovered = {
            "aluminum_kg": composition.aluminum * self.recovery_rates.aluminum * efficiency,
            "silver_kg": composition.silver * self.recovery_rates.silver * efficiency,
            "copper_kg": composition.copper * self.recovery_rates.copper * efficiency,
            "aluminum_recovery_rate": self.recovery_rates.aluminum * efficiency,
            "silver_recovery_rate": self.recovery_rates.silver * efficiency,
            "copper_recovery_rate": self.recovery_rates.copper * efficiency,
            "total_metal_kg": 0.0
        }

        recovered["total_metal_kg"] = (
            recovered["aluminum_kg"] +
            recovered["silver_kg"] +
            recovered["copper_kg"]
        )

        # Calculate metal losses
        recovered["metal_loss_kg"] = (
            composition.aluminum * (1 - self.recovery_rates.aluminum * efficiency) +
            composition.silver * (1 - self.recovery_rates.silver * efficiency) +
            composition.copper * (1 - self.recovery_rates.copper * efficiency)
        )

        return recovered

    def glass_recovery(
        self,
        composition: ModuleComposition,
        processing_quality: str = "standard"
    ) -> Dict[str, float]:
        """
        Calculate glass recovery from PV module.

        Args:
            composition: Module material composition
            processing_quality: Quality level ('low', 'standard', 'high')

        Returns:
            Dictionary with recovered glass mass and quality metrics
        """
        # Quality factors affect recovery rate and purity
        quality_factors = {
            "low": {"rate": 0.85, "purity": 0.80},
            "standard": {"rate": 0.95, "purity": 0.90},
            "high": {"rate": 0.98, "purity": 0.95}
        }

        factors = quality_factors.get(processing_quality, quality_factors["standard"])

        recovery_rate = self.recovery_rates.glass * factors["rate"]
        recovered_glass = composition.glass * recovery_rate

        return {
            "recovered_glass_kg": recovered_glass,
            "glass_recovery_rate": recovery_rate,
            "glass_purity": factors["purity"],
            "cullet_grade": processing_quality,
            "glass_loss_kg": composition.glass * (1 - recovery_rate),
            "contamination_kg": recovered_glass * (1 - factors["purity"])
        }

    def silicon_recovery(
        self,
        composition: ModuleComposition,
        recovery_technique: str = "thermal_chemical"
    ) -> Dict[str, float]:
        """
        Calculate silicon recovery from PV module.

        Args:
            composition: Module material composition
            recovery_technique: Recovery technique ('thermal', 'chemical', 'thermal_chemical', 'mechanical')

        Returns:
            Dictionary with recovered silicon mass and purity
        """
        # Different techniques have different recovery rates and purities
        technique_params = {
            "thermal": {"rate": 0.75, "purity": 0.85},
            "chemical": {"rate": 0.88, "purity": 0.92},
            "thermal_chemical": {"rate": 0.85, "purity": 0.90},
            "mechanical": {"rate": 0.65, "purity": 0.75}
        }

        params = technique_params.get(recovery_technique, technique_params["thermal_chemical"])

        recovery_rate = self.recovery_rates.silicon * params["rate"]
        recovered_silicon = composition.silicon * recovery_rate

        # Silicon can be recovered as different grades
        solar_grade_fraction = params["purity"]
        metallurgical_grade_fraction = 1 - solar_grade_fraction

        return {
            "recovered_silicon_kg": recovered_silicon,
            "silicon_recovery_rate": recovery_rate,
            "solar_grade_silicon_kg": recovered_silicon * solar_grade_fraction,
            "metallurgical_grade_silicon_kg": recovered_silicon * metallurgical_grade_fraction,
            "silicon_purity": params["purity"],
            "silicon_loss_kg": composition.silicon * (1 - recovery_rate),
            "recovery_technique": recovery_technique
        }

    def recycling_costs(
        self,
        composition: ModuleComposition,
        num_modules: int = 1,
        transport_distance_km: float = 100.0,
        process_method: str = "combined"
    ) -> RecyclingCostBreakdown:
        """
        Calculate detailed recycling costs for PV modules.

        Args:
            composition: Module material composition
            num_modules: Number of modules to recycle
            transport_distance_km: Transportation distance in km
            process_method: Processing method ('thermal', 'chemical', 'mechanical', 'combined')

        Returns:
            RecyclingCostBreakdown with detailed cost analysis
        """
        total_mass = composition.total_mass * num_modules

        # Base costs
        collection_cost = self.cost_structure.collection_cost_per_module * num_modules
        transportation_cost = (
            self.cost_structure.transportation_cost_per_km *
            transport_distance_km *
            num_modules
        )
        dismantling_cost = self.cost_structure.dismantling_cost_per_module * num_modules

        # Processing costs depend on method
        if process_method == "thermal":
            thermal_cost = self.cost_structure.thermal_treatment_cost_per_kg * total_mass
            chemical_cost = 0.0
            mechanical_cost = self.cost_structure.mechanical_separation_cost_per_kg * total_mass * 0.5
        elif process_method == "chemical":
            thermal_cost = 0.0
            chemical_cost = self.cost_structure.chemical_treatment_cost_per_kg * total_mass
            mechanical_cost = self.cost_structure.mechanical_separation_cost_per_kg * total_mass * 0.5
        elif process_method == "mechanical":
            thermal_cost = 0.0
            chemical_cost = 0.0
            mechanical_cost = self.cost_structure.mechanical_separation_cost_per_kg * total_mass
        else:  # combined
            thermal_cost = self.cost_structure.thermal_treatment_cost_per_kg * total_mass * 0.4
            chemical_cost = self.cost_structure.chemical_treatment_cost_per_kg * total_mass * 0.3
            mechanical_cost = self.cost_structure.mechanical_separation_cost_per_kg * total_mass

        # Estimate waste based on recovery rates
        waste_mass = total_mass * (1 - 0.85)  # Assuming 85% average recovery
        waste_disposal_cost = self.cost_structure.waste_disposal_cost_per_kg * waste_mass

        # Calculate direct costs
        direct_costs = (
            collection_cost + transportation_cost + dismantling_cost +
            thermal_cost + chemical_cost + mechanical_cost + waste_disposal_cost
        )

        # Add overhead
        overhead_cost = direct_costs * self.cost_structure.facility_overhead_rate

        return RecyclingCostBreakdown(
            collection_cost=collection_cost,
            transportation_cost=transportation_cost,
            dismantling_cost=dismantling_cost,
            thermal_treatment_cost=thermal_cost,
            chemical_treatment_cost=chemical_cost,
            mechanical_separation_cost=mechanical_cost,
            waste_disposal_cost=waste_disposal_cost,
            overhead_cost=overhead_cost
        )

    def full_recovery_analysis(
        self,
        composition: ModuleComposition,
        num_modules: int = 1,
        transport_distance_km: float = 100.0
    ) -> Dict:
        """
        Perform comprehensive recovery analysis including all materials and costs.

        Args:
            composition: Module material composition
            num_modules: Number of modules to recycle
            transport_distance_km: Transportation distance in km

        Returns:
            Dictionary with complete recovery and cost analysis
        """
        # Get material recovery
        metal_recovery = self.metal_recovery(composition)
        glass_recovery = self.glass_recovery(composition)
        silicon_recovery = self.silicon_recovery(composition)

        # Calculate total recovered materials
        total_recovered = RecoveredMaterials(
            glass_kg=glass_recovery["recovered_glass_kg"] * num_modules,
            aluminum_kg=metal_recovery["aluminum_kg"] * num_modules,
            silicon_kg=silicon_recovery["recovered_silicon_kg"] * num_modules,
            silver_kg=metal_recovery["silver_kg"] * num_modules,
            copper_kg=metal_recovery["copper_kg"] * num_modules,
            waste_kg=(
                composition.total_mass -
                glass_recovery["recovered_glass_kg"] -
                metal_recovery["total_metal_kg"] -
                silicon_recovery["recovered_silicon_kg"]
            ) * num_modules
        )

        # Get costs
        costs = self.recycling_costs(composition, num_modules, transport_distance_km)

        return {
            "metal_recovery": metal_recovery,
            "glass_recovery": glass_recovery,
            "silicon_recovery": silicon_recovery,
            "total_recovered_materials": total_recovered.model_dump(),
            "overall_recovery_rate": total_recovered.recovery_rate,
            "cost_breakdown": costs.model_dump(),
            "total_cost_usd": costs.total_cost,
            "cost_per_module_usd": costs.total_cost / num_modules if num_modules > 0 else 0,
            "cost_per_kg_usd": costs.total_cost / (composition.total_mass * num_modules) if num_modules > 0 else 0
        }
