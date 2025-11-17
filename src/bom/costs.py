"""Cost calculation and breakdown."""

from typing import Dict
from pydantic import BaseModel, Field
from .calculator import BOMResult


class CostBreakdown(BaseModel):
    """Detailed cost breakdown."""

    material_cost_usd: float = Field(..., description="Total material cost")
    labor_cost_usd: float = Field(..., description="Labor cost")
    overhead_cost_usd: float = Field(..., description="Overhead cost")
    total_manufacturing_cost_usd: float = Field(..., description="Total manufacturing cost")
    margin_pct: float = Field(..., description="Profit margin percentage")
    selling_price_usd: float = Field(..., description="Selling price")
    cost_per_watt_usd: float = Field(..., description="Cost per watt ($/W)")


class CostCalculator:
    """Calculator for module costs."""

    def __init__(
        self,
        labor_cost_multiplier: float = 0.3,
        overhead_multiplier: float = 0.2,
        margin_pct: float = 20.0,
    ):
        """Initialize cost calculator.

        Args:
            labor_cost_multiplier: Labor cost as fraction of material cost
            overhead_multiplier: Overhead as fraction of material cost
            margin_pct: Profit margin percentage
        """
        self.labor_cost_multiplier = labor_cost_multiplier
        self.overhead_multiplier = overhead_multiplier
        self.margin_pct = margin_pct

    def calculate(self, bom_result: BOMResult, module_power_w: float) -> CostBreakdown:
        """Calculate detailed cost breakdown.

        Args:
            bom_result: BOM calculation result
            module_power_w: Module power rating in watts

        Returns:
            CostBreakdown with detailed pricing
        """
        material_cost = bom_result.total_cost_usd
        labor_cost = material_cost * self.labor_cost_multiplier
        overhead_cost = material_cost * self.overhead_multiplier

        total_mfg_cost = material_cost + labor_cost + overhead_cost
        selling_price = total_mfg_cost * (1 + self.margin_pct / 100)

        cost_per_watt = selling_price / module_power_w

        return CostBreakdown(
            material_cost_usd=material_cost,
            labor_cost_usd=labor_cost,
            overhead_cost_usd=overhead_cost,
            total_manufacturing_cost_usd=total_mfg_cost,
            margin_pct=self.margin_pct,
            selling_price_usd=selling_price,
            cost_per_watt_usd=cost_per_watt,
        )

    def calculate_lcoe_contribution(
        self,
        selling_price_usd: float,
        module_power_w: float,
        lifetime_years: int = 25,
        discount_rate: float = 0.05,
    ) -> Dict[str, float]:
        """Calculate contribution to Levelized Cost of Energy.

        Args:
            selling_price_usd: Module selling price
            module_power_w: Module power rating
            lifetime_years: Expected lifetime
            discount_rate: Discount rate for NPV

        Returns:
            Dictionary with LCOE metrics
        """
        # Simplified LCOE calculation (module cost contribution only)
        # Real LCOE includes BOS, O&M, land, etc.

        # Annual degradation (simplified)
        year_1_degradation = 0.02  # 2% first year
        annual_degradation = 0.005  # 0.5% per year after

        # Calculate total energy production over lifetime (kWh)
        total_energy_kwh = 0
        for year in range(lifetime_years):
            if year == 0:
                power_factor = 1 - year_1_degradation
            else:
                power_factor = (1 - year_1_degradation) * ((1 - annual_degradation) ** year)

            # Assume 1500 hours of peak equivalent per year
            annual_energy_kwh = module_power_w * 1500 * power_factor / 1000

            # Discount to present value
            discount_factor = 1 / ((1 + discount_rate) ** year)
            total_energy_kwh += annual_energy_kwh * discount_factor

        # Module cost contribution to LCOE ($/kWh)
        lcoe_module_usd_kwh = selling_price_usd / total_energy_kwh

        return {
            "total_energy_kwh": total_energy_kwh,
            "lcoe_module_usd_kwh": lcoe_module_usd_kwh,
            "cost_per_watt_usd": selling_price_usd / module_power_w,
        }
