"""Weight calculation and breakdown."""

from typing import Dict, List
from pydantic import BaseModel, Field
from .calculator import BOMResult


class WeightBreakdown(BaseModel):
    """Detailed weight breakdown."""

    total_weight_kg: float = Field(..., description="Total module weight")
    weight_per_m2_kg: float = Field(..., description="Weight per square meter")
    weight_per_watt_kg: float = Field(..., description="Weight per watt")
    component_weights: Dict[str, float] = Field(..., description="Weight by component")
    material_weights: Dict[str, float] = Field(..., description="Weight by material type")


class WeightCalculator:
    """Calculator for module weights."""

    def calculate(self, bom_result: BOMResult, module_area_m2: float, module_power_w: float) -> WeightBreakdown:
        """Calculate detailed weight breakdown.

        Args:
            bom_result: BOM calculation result
            module_area_m2: Module area in m²
            module_power_w: Module power rating in watts

        Returns:
            WeightBreakdown with detailed analysis
        """
        total_weight = bom_result.total_weight_kg

        # Component weights
        component_weights = {item.component: item.weight_kg for item in bom_result.items}

        # Material weights (aggregate by material type)
        material_weights: Dict[str, float] = {}
        for item in bom_result.items:
            material = item.material
            if material in material_weights:
                material_weights[material] += item.weight_kg
            else:
                material_weights[material] = item.weight_kg

        return WeightBreakdown(
            total_weight_kg=total_weight,
            weight_per_m2_kg=total_weight / module_area_m2,
            weight_per_watt_kg=total_weight / module_power_w,
            component_weights=component_weights,
            material_weights=material_weights,
        )

    def estimate_shipping_cost(
        self,
        module_weight_kg: float,
        num_modules: int,
        distance_km: float,
        cost_per_kg_km: float = 0.0001,
    ) -> Dict[str, float]:
        """Estimate shipping cost.

        Args:
            module_weight_kg: Single module weight
            num_modules: Number of modules
            distance_km: Shipping distance
            cost_per_kg_km: Cost per kg per km

        Returns:
            Dictionary with shipping cost breakdown
        """
        total_weight_kg = module_weight_kg * num_modules
        shipping_cost_usd = total_weight_kg * distance_km * cost_per_kg_km

        return {
            "total_weight_kg": total_weight_kg,
            "distance_km": distance_km,
            "shipping_cost_usd": shipping_cost_usd,
            "cost_per_module_usd": shipping_cost_usd / num_modules,
        }
