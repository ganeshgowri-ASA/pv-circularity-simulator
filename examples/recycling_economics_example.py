"""
Comprehensive Example: Recycling Economics & Material Recovery

This example demonstrates how to use the RecyclingEconomics module to analyze
the economic viability of PV panel recycling, including costs, revenues, and
environmental benefits.
"""

from pv_circularity_simulator.recycling import (
    RecyclingEconomics,
    MaterialComposition,
    MaterialExtractionCosts,
    RecoveryRates,
    PVMaterialType,
    RecyclingTechnology,
)


def main():
    """Run comprehensive recycling economics analysis."""

    print("=" * 80)
    print("PV PANEL RECYCLING ECONOMICS ANALYSIS")
    print("=" * 80)

    # Step 1: Define panel material composition
    print("\n1. PANEL COMPOSITION")
    print("-" * 80)

    panel_composition = [
        MaterialComposition(
            material_type=PVMaterialType.GLASS,
            mass_kg=12.0,
            purity_percent=99.0,
            market_value_per_kg=0.08,
        ),
        MaterialComposition(
            material_type=PVMaterialType.ALUMINUM,
            mass_kg=3.5,
            purity_percent=98.0,
            market_value_per_kg=2.2,
        ),
        MaterialComposition(
            material_type=PVMaterialType.SILICON,
            mass_kg=3.0,
            purity_percent=99.0,
            market_value_per_kg=18.0,
        ),
        MaterialComposition(
            material_type=PVMaterialType.COPPER,
            mass_kg=0.8,
            purity_percent=99.5,
            market_value_per_kg=8.5,
        ),
        MaterialComposition(
            material_type=PVMaterialType.SILVER,
            mass_kg=0.012,
            purity_percent=99.0,
            market_value_per_kg=650.0,
        ),
        MaterialComposition(
            material_type=PVMaterialType.POLYMER,
            mass_kg=1.2,
            purity_percent=95.0,
            market_value_per_kg=0.5,
        ),
    ]

    total_valuable_mass = sum(m.mass_kg for m in panel_composition)
    panel_mass_kg = 22.0

    print(f"Total panel mass: {panel_mass_kg:.2f} kg")
    print(f"Valuable materials: {total_valuable_mass:.2f} kg")
    print(f"Other materials: {panel_mass_kg - total_valuable_mass:.2f} kg")
    print("\nMaterial breakdown:")
    for material in panel_composition:
        value = material.mass_kg * material.market_value_per_kg
        print(f"  {material.material_type.value:12s}: "
              f"{material.mass_kg:6.3f} kg × ${material.market_value_per_kg:7.2f}/kg = ${value:7.2f}")

    # Step 2: Define extraction costs
    print("\n2. EXTRACTION COSTS")
    print("-" * 80)

    extraction_costs = MaterialExtractionCosts(
        technology=RecyclingTechnology.HYBRID,
        collection_cost_per_panel=8.0,
        preprocessing_cost_per_kg=1.5,
        processing_cost_per_kg=2.5,
        purification_cost_per_kg=2.0,
        labor_cost_per_hour=30.0,
        processing_time_hours=1.0,
        energy_cost_per_kwh=0.15,
        energy_consumption_kwh_per_kg=3.0,
        disposal_cost_per_kg=0.3,
        equipment_depreciation_per_panel=3.0,
        overhead_multiplier=1.25,
    )

    print(f"Technology: {extraction_costs.technology.value}")
    print(f"Collection cost: ${extraction_costs.collection_cost_per_panel:.2f} per panel")
    print(f"Labor cost: ${extraction_costs.labor_cost_per_hour:.2f}/hr × "
          f"{extraction_costs.processing_time_hours:.1f} hr")
    print(f"Energy: {extraction_costs.energy_consumption_kwh_per_kg:.1f} kWh/kg × "
          f"${extraction_costs.energy_cost_per_kwh:.2f}/kWh")
    print(f"Overhead multiplier: {extraction_costs.overhead_multiplier:.2f}x")

    # Step 3: Define recovery rates
    print("\n3. RECOVERY RATES")
    print("-" * 80)

    recovery_rates = RecoveryRates(
        technology=RecyclingTechnology.HYBRID,
        material_recovery_rates={
            PVMaterialType.GLASS: 98.0,
            PVMaterialType.ALUMINUM: 95.0,
            PVMaterialType.SILICON: 92.0,
            PVMaterialType.COPPER: 96.0,
            PVMaterialType.SILVER: 85.0,
            PVMaterialType.POLYMER: 60.0,
        },
        technology_efficiency=0.88,
        quality_grade="B",
    )

    print(f"Technology: {recovery_rates.technology.value}")
    print(f"Technology efficiency: {recovery_rates.technology_efficiency:.1%}")
    print(f"Quality grade: {recovery_rates.quality_grade}")
    print("\nBase recovery rates:")
    for material, rate in recovery_rates.material_recovery_rates.items():
        effective = recovery_rates.get_effective_recovery_rate(material)
        print(f"  {material.value:12s}: {rate:5.1f}% → {effective:5.1f}% (effective)")

    # Step 4: Create economics model
    print("\n4. ECONOMIC ANALYSIS")
    print("-" * 80)

    economics = RecyclingEconomics(
        panel_composition=panel_composition,
        extraction_costs=extraction_costs,
        recovery_rates_model=recovery_rates,
        panel_mass_kg=panel_mass_kg,
    )

    # Calculate costs
    costs = economics.material_extraction_costs()
    print("\nCost Analysis:")
    print(f"  Fixed costs:     ${costs['fixed_costs']:8.2f}")
    print(f"  Variable costs:  ${costs['variable_costs']:8.2f}")
    print(f"  Overhead costs:  ${costs['overhead_costs']:8.2f}")
    print(f"  TOTAL COST:      ${costs['total_cost']:8.2f}")
    print(f"  Cost per kg:     ${costs['cost_per_kg']:8.2f}/kg")

    print("\nDetailed cost breakdown:")
    for item, value in costs['cost_breakdown'].items():
        print(f"  {item:24s}: ${value:7.2f}")

    # Calculate recovery
    recovery = economics.recovery_rates()
    print("\nMaterial Recovery:")
    print(f"  Total recovery rate: {recovery['total_recovery_rate']:.1f}%")
    print(f"  Total recovered:     {recovery['total_recovered_mass']:.2f} kg")

    print("\nRecovered by material:")
    for material, mass in recovery['recovered_masses'].items():
        rate = recovery['material_recovery_rates'][material]
        print(f"  {material.value:12s}: {mass:6.3f} kg ({rate:5.1f}%)")

    # Calculate revenue
    revenue = economics.recycling_revenue_calculation()
    print("\nRevenue Analysis:")
    print(f"  Gross revenue:        ${revenue['gross_revenue']:8.2f}")
    print(f"  Transportation cost:  ${revenue['transportation_cost']:8.2f}")
    print(f"  Sales commission:     ${revenue['sales_commission']:8.2f}")
    print(f"  NET REVENUE:          ${revenue['net_revenue']:8.2f}")
    print(f"  Revenue per kg:       ${revenue['revenue_per_kg']:8.2f}/kg")

    print("\nRevenue by material:")
    for material, value in revenue['revenue_by_material'].items():
        percentage = (value / revenue['gross_revenue'] * 100) if revenue['gross_revenue'] > 0 else 0
        print(f"  {material.value:12s}: ${value:7.2f} ({percentage:5.1f}%)")

    # Calculate environmental credits
    env_credits = economics.environmental_credits(carbon_price_per_ton_co2=60.0)
    print("\n5. ENVIRONMENTAL BENEFITS")
    print("-" * 80)
    print("\nLCA Impact Avoided (vs. virgin production):")
    print(f"  CO2 emissions:  {env_credits['avoided_emissions_kg_co2']:10.1f} kg CO2")
    print(f"  Energy saved:   {env_credits['energy_savings_kwh']:10.1f} kWh")
    print(f"  Water saved:    {env_credits['water_savings_liters']:10.1f} liters")

    print("\nEnvironmental Credits:")
    for source, value in env_credits['credits_breakdown'].items():
        print(f"  {source:28s}: ${value:7.2f}")
    print(f"  TOTAL ENVIRONMENTAL VALUE:     ${env_credits['total_environmental_value']:7.2f}")

    # Net economic value
    net_value = economics.net_economic_value(
        carbon_price_per_ton_co2=60.0,
        include_environmental_credits=True,
    )

    print("\n6. OVERALL ECONOMICS")
    print("-" * 80)
    print(f"  Total costs:            ${net_value['total_costs']:8.2f}")
    print(f"  Total revenue:          ${net_value['total_revenue']:8.2f}")
    print(f"  Environmental credits:  ${net_value['environmental_value']:8.2f}")
    print(f"  " + "=" * 42)
    print(f"  NET ECONOMIC VALUE:     ${net_value['net_value']:8.2f}")
    print(f"  ROI:                    {net_value['roi_percent']:8.1f}%")

    if net_value['net_value'] > 0:
        print("\n  ✓ RECYCLING IS ECONOMICALLY VIABLE")
    else:
        print("\n  ✗ RECYCLING IS NOT ECONOMICALLY VIABLE")
        print(f"  Breakeven carbon price: ${net_value['breakeven_carbon_price']:.2f}/ton CO2")

    # Scenario analysis
    print("\n7. SCENARIO ANALYSIS")
    print("-" * 80)
    print("\nCarbon price sensitivity:")

    carbon_prices = [0, 25, 50, 75, 100, 150]
    for carbon_price in carbon_prices:
        scenario = economics.net_economic_value(
            carbon_price_per_ton_co2=carbon_price,
            include_environmental_credits=True,
        )
        status = "✓ Profitable" if scenario['net_value'] > 0 else "✗ Unprofitable"
        print(f"  ${carbon_price:3d}/ton CO2: "
              f"Net value = ${scenario['net_value']:7.2f} | ROI = {scenario['roi_percent']:6.1f}% | {status}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
