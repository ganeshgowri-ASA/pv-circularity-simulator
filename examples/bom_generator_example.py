"""Example usage of BOM Generator and Cost Calculator.

This example demonstrates how to use the BOMGenerator class to:
1. Generate a Bill of Materials from a module design
2. Calculate material costs with waste factors and pricing tiers
3. Analyze supplier pricing
4. Export BOM to various formats
5. Optimize costs
"""

from decimal import Decimal

from src.modules.bom_generator import BOMGenerator
from src.modules.models import (
    ComponentCategory,
    Currency,
    CurrencyExchangeRate,
    Material,
    MaterialWithTiers,
    ModuleDesign,
    PricingTier,
)


def main():
    """Run BOM generator example."""
    print("=" * 80)
    print("PV Module BOM Generator - Example Usage")
    print("=" * 80)

    # 1. Define materials database
    print("\n1. Creating materials database...")
    materials = [
        Material(
            id="MAT-WAFER-001",
            name="Silicon Wafer",
            category=ComponentCategory.CELL,
            supplier="SolarTech Inc.",
            unit="pieces",
            base_price=Decimal("5.50"),
            currency=Currency.USD,
            waste_factor=0.05,
            transportation_cost_per_unit=Decimal("0.10"),
        ),
        MaterialWithTiers(
            id="MAT-GLASS-001",
            name="Front Glass (tempered)",
            category=ComponentCategory.MODULE,
            supplier="GlassCo",
            unit="m2",
            base_price=Decimal("18.00"),
            currency=Currency.USD,
            waste_factor=0.08,
            pricing_tiers=[
                PricingTier(
                    min_quantity=Decimal("0"),
                    max_quantity=Decimal("100"),
                    discount_percentage=0,
                    unit_price=Decimal("18.00"),
                ),
                PricingTier(
                    min_quantity=Decimal("100"),
                    max_quantity=Decimal("500"),
                    discount_percentage=10,
                    unit_price=Decimal("16.20"),
                ),
                PricingTier(
                    min_quantity=Decimal("500"),
                    max_quantity=None,
                    discount_percentage=20,
                    unit_price=Decimal("14.40"),
                ),
            ],
        ),
        # Add more materials as needed...
    ]

    # 2. Initialize BOM Generator
    print("2. Initializing BOM Generator...")
    generator = BOMGenerator(
        materials=materials, manufacturing_overhead_rate=0.15, default_currency=Currency.USD
    )

    # Add exchange rates for multi-currency support
    generator.add_exchange_rate(
        CurrencyExchangeRate(
            from_currency=Currency.USD, to_currency=Currency.EUR, rate=Decimal("0.92")
        )
    )

    # 3. Define module design
    print("3. Defining module design...")
    module_design = ModuleDesign(
        module_id="MOD-2025-400W",
        module_type="mono-Si",
        power_rating=400.0,
        efficiency=20.5,
        dimensions={"length": 1640, "width": 990, "thickness": 35},
        num_cells=60,
        cell_size=156.75,
        frame_type="aluminum",
        glass_type="tempered",
        backsheet_type="standard",
        encapsulant_type="EVA",
        junction_box_type="standard",
    )

    # 4. Generate BOM
    print("4. Generating BOM from module design...")
    bom = generator.generate_bom(module_design)
    print(f"   Generated BOM with {len(bom)} line items")
    print("\n   BOM Preview:")
    print(bom.head(10).to_string())

    # 5. Calculate costs
    print("\n5. Calculating material costs...")
    result = generator.calculate_material_costs(bom)
    bom_with_costs = result["bom_with_costs"]
    cost_breakdown = result["cost_breakdown"]
    missing_materials = result["missing_materials"]

    if missing_materials:
        print(f"   Warning: {len(missing_materials)} materials not found in database")
        print(f"   Missing: {', '.join(missing_materials)}")

    print("\n   Cost Breakdown:")
    print(f"   - Cell Costs:          ${float(cost_breakdown.cell_costs):,.2f}")
    print(f"   - Module Costs:        ${float(cost_breakdown.module_costs):,.2f}")
    print(f"   - Interconnect Costs:  ${float(cost_breakdown.interconnect_costs):,.2f}")
    print(f"   - Adhesive Costs:      ${float(cost_breakdown.adhesive_costs):,.2f}")
    print(f"   - Material Subtotal:   ${float(cost_breakdown.material_subtotal):,.2f}")
    print(f"   - Waste Costs:         ${float(cost_breakdown.waste_costs):,.2f}")
    print(f"   - Transportation:      ${float(cost_breakdown.transportation_costs):,.2f}")
    print(f"   - Mfg. Overhead (15%): ${float(cost_breakdown.manufacturing_overhead):,.2f}")
    print(f"   {'=' * 40}")
    print(f"   TOTAL COST:            ${float(cost_breakdown.total_cost):,.2f}")

    # 6. Calculate total module cost
    total_cost = generator.calculate_module_cost(result)
    print(f"\n6. Total Module Cost: ${total_cost:,.2f}")

    # 7. Budget analysis
    print("\n7. Budget Analysis...")
    budgeted_cost = 150.00
    budget_analysis = generator.analyze_budget(budgeted_cost=budgeted_cost, actual_cost=total_cost)
    print(f"   Budgeted: ${float(budget_analysis.budgeted_cost):,.2f}")
    print(f"   Actual:   ${float(budget_analysis.actual_cost):,.2f}")
    print(f"   Variance: ${float(budget_analysis.variance):,.2f} ({budget_analysis.variance_percentage:.1f}%)")
    if budget_analysis.over_budget:
        print("   ⚠️  OVER BUDGET")
    else:
        print("   ✓ Under Budget")

    # 8. Export BOM
    print("\n8. Exporting BOM to files...")
    csv_path = generator.export_bom(bom_with_costs, format="csv", output_path="bom_example.csv")
    print(f"   CSV exported to: {csv_path}")

    excel_path = generator.export_bom(
        bom_with_costs, format="excel", output_path="bom_example.xlsx"
    )
    print(f"   Excel exported to: {excel_path}")

    json_path = generator.export_bom(
        bom_with_costs, format="json", output_path="bom_example.json"
    )
    print(f"   JSON exported to: {json_path}")

    # 9. Cost breakdown by category
    print("\n9. Cost Breakdown by Category:")
    category_breakdown = generator.get_cost_breakdown_by_category(bom_with_costs)
    for category, cost in sorted(category_breakdown.items()):
        print(f"   {category:15s}: ${cost:,.2f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
