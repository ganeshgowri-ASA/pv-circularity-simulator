"""BOM (Bill of Materials) calculator."""

from typing import List, Optional
from pydantic import BaseModel, Field
from ..models.module import ModuleConfiguration
from ..models.material import MaterialDatabase


class BOMItem(BaseModel):
    """Single item in Bill of Materials."""

    component: str = Field(..., description="Component name")
    material: str = Field(..., description="Material type")
    quantity: float = Field(..., description="Quantity (area, length, or count)")
    unit: str = Field(..., description="Unit (m², m, kg, pieces)")
    weight_kg: float = Field(..., description="Weight in kg")
    cost_usd: float = Field(..., description="Cost in USD")
    recyclability_pct: float = Field(..., description="Recyclability percentage")


class BOMResult(BaseModel):
    """Complete Bill of Materials result."""

    items: List[BOMItem] = Field(..., description="List of BOM items")
    total_weight_kg: float = Field(..., description="Total module weight in kg")
    total_cost_usd: float = Field(..., description="Total module cost in USD")
    average_recyclability_pct: float = Field(..., description="Weighted average recyclability")


class BOMCalculator:
    """Calculator for module Bill of Materials."""

    def __init__(self):
        """Initialize BOM calculator with material database."""
        self.materials = MaterialDatabase.get_default_materials()

    def calculate(self, config: ModuleConfiguration) -> BOMResult:
        """Calculate complete BOM for module configuration.

        Args:
            config: Module configuration

        Returns:
            BOMResult with detailed breakdown
        """
        items = []
        module_area_m2 = config.area_m2

        # 1. Solar Cells
        cell_area_m2 = (config.cell_design.template.length_mm * config.cell_design.template.width_mm) / 1_000_000
        cell_volume_m3 = cell_area_m2 * (config.cell_design.template.thickness_um / 1_000_000)
        cell_weight_kg = cell_volume_m3 * self.materials["silicon_cell"].density_kg_m3
        total_cell_weight = cell_weight_kg * config.layout.num_cells
        total_cell_cost = total_cell_weight * self.materials["silicon_cell"].cost_per_kg_usd

        items.append(
            BOMItem(
                component="Solar Cells",
                material="Silicon",
                quantity=config.layout.num_cells,
                unit="pieces",
                weight_kg=total_cell_weight,
                cost_usd=total_cell_cost,
                recyclability_pct=self.materials["silicon_cell"].recyclability_pct,
            )
        )

        # 2. Front Glass
        glass_volume_m3 = module_area_m2 * (config.glass_front_mm / 1000)
        glass_weight_kg = glass_volume_m3 * self.materials["glass_front"].density_kg_m3
        glass_cost = glass_weight_kg * self.materials["glass_front"].cost_per_kg_usd

        items.append(
            BOMItem(
                component="Front Glass",
                material=f"Tempered Glass ({config.glass_front_mm}mm)",
                quantity=module_area_m2,
                unit="m²",
                weight_kg=glass_weight_kg,
                cost_usd=glass_cost,
                recyclability_pct=self.materials["glass_front"].recyclability_pct,
            )
        )

        # 3. Encapsulant (EVA/POE)
        encapsulant_thickness_mm = 0.45 * 2  # Top and bottom layers
        encapsulant_volume_m3 = module_area_m2 * (encapsulant_thickness_mm / 1000)
        encapsulant_material = self.materials[config.encapsulant_type.lower()]
        encapsulant_weight_kg = encapsulant_volume_m3 * encapsulant_material.density_kg_m3
        encapsulant_cost = encapsulant_weight_kg * encapsulant_material.cost_per_kg_usd

        items.append(
            BOMItem(
                component="Encapsulant",
                material=config.encapsulant_type,
                quantity=module_area_m2 * 2,
                unit="m²",
                weight_kg=encapsulant_weight_kg,
                cost_usd=encapsulant_cost,
                recyclability_pct=encapsulant_material.recyclability_pct,
            )
        )

        # 4. Backsheet or Back Glass
        if config.glass_back_mm:
            # Bifacial: Back glass
            back_glass_volume_m3 = module_area_m2 * (config.glass_back_mm / 1000)
            back_glass_weight_kg = back_glass_volume_m3 * self.materials["glass_back"].density_kg_m3
            back_glass_cost = back_glass_weight_kg * self.materials["glass_back"].cost_per_kg_usd

            items.append(
                BOMItem(
                    component="Back Glass",
                    material=f"Tempered Glass ({config.glass_back_mm}mm)",
                    quantity=module_area_m2,
                    unit="m²",
                    weight_kg=back_glass_weight_kg,
                    cost_usd=back_glass_cost,
                    recyclability_pct=self.materials["glass_back"].recyclability_pct,
                )
            )
        else:
            # Monofacial: Backsheet
            backsheet_thickness_mm = 0.35
            backsheet_volume_m3 = module_area_m2 * (backsheet_thickness_mm / 1000)
            backsheet_weight_kg = backsheet_volume_m3 * self.materials["backsheet"].density_kg_m3
            backsheet_cost = backsheet_weight_kg * self.materials["backsheet"].cost_per_kg_usd

            items.append(
                BOMItem(
                    component="Backsheet",
                    material=config.backsheet_type or "PET",
                    quantity=module_area_m2,
                    unit="m²",
                    weight_kg=backsheet_weight_kg,
                    cost_usd=backsheet_cost,
                    recyclability_pct=self.materials["backsheet"].recyclability_pct,
                )
            )

        # 5. Frame (if not frameless)
        if config.frame_type != "frameless":
            # Estimate frame weight based on perimeter
            perimeter_m = 2 * (config.length_mm + config.width_mm) / 1000
            frame_cross_section_m2 = 0.0003  # Typical 30mm x 10mm profile
            frame_volume_m3 = perimeter_m * frame_cross_section_m2
            frame_weight_kg = frame_volume_m3 * self.materials["aluminum_frame"].density_kg_m3
            frame_cost = frame_weight_kg * self.materials["aluminum_frame"].cost_per_kg_usd

            items.append(
                BOMItem(
                    component="Frame",
                    material=config.frame_type.capitalize(),
                    quantity=perimeter_m,
                    unit="m",
                    weight_kg=frame_weight_kg,
                    cost_usd=frame_cost,
                    recyclability_pct=self.materials["aluminum_frame"].recyclability_pct,
                )
            )

        # 6. Junction Box
        jbox_weight_kg = 0.15  # Typical weight
        jbox_cost = jbox_weight_kg * self.materials["junction_box"].cost_per_kg_usd

        items.append(
            BOMItem(
                component="Junction Box",
                material=config.junction_box_type,
                quantity=1,
                unit="pieces",
                weight_kg=jbox_weight_kg,
                cost_usd=jbox_cost,
                recyclability_pct=self.materials["junction_box"].recyclability_pct,
            )
        )

        # 7. Cables
        cable_length_m = config.cable_length_mm / 1000
        cable_weight_per_m = 0.05  # kg/m for typical 4mm² cable
        cable_weight_kg = cable_length_m * cable_weight_per_m
        cable_cost = cable_weight_kg * self.materials["copper_cable"].cost_per_kg_usd

        items.append(
            BOMItem(
                component="Cables",
                material="Copper Cable",
                quantity=cable_length_m,
                unit="m",
                weight_kg=cable_weight_kg,
                cost_usd=cable_cost,
                recyclability_pct=self.materials["copper_cable"].recyclability_pct,
            )
        )

        # 8. Connectors (MC4 or similar)
        connector_weight_kg = 0.05  # Typical weight for pair
        connector_cost = 2.0  # USD per pair

        items.append(
            BOMItem(
                component="Connectors",
                material=config.connector_type,
                quantity=2,
                unit="pieces",
                weight_kg=connector_weight_kg,
                cost_usd=connector_cost,
                recyclability_pct=60,
            )
        )

        # Calculate totals
        total_weight = sum(item.weight_kg for item in items)
        total_cost = sum(item.cost_usd for item in items)
        weighted_recyclability = sum(item.weight_kg * item.recyclability_pct for item in items) / total_weight

        return BOMResult(
            items=items,
            total_weight_kg=total_weight,
            total_cost_usd=total_cost,
            average_recyclability_pct=weighted_recyclability,
        )
