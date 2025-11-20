"""
Basic usage examples for PV Circularity Simulator.

This script demonstrates the core functionality of each component
in the circularity system.
"""

from src.circularity import (
    MaterialRecoveryCalculator,
    ReuseAnalyzer,
    RepairOptimizer,
    RecyclingEconomics,
    LCAAnalyzer,
    CircularityUI,
)
from src.circularity.material_recovery import ModuleComposition
from src.circularity.reuse_analyzer import ModuleTestResults, ModuleCondition, DefectType
from src.circularity.repair_optimizer import Defect, DefectSeverity, RepairCost


def example_material_recovery():
    """Example: Material Recovery & Recycling Economics."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Material Recovery & Recycling Economics")
    print("="*80)

    # Define a standard PV module composition
    module = ModuleComposition(
        glass=15.0,          # kg
        aluminum=2.5,        # kg
        silicon=0.5,         # kg
        silver=0.005,        # kg
        copper=0.2,          # kg
        eva_polymer=1.0,     # kg
        backsheet=0.5,       # kg
        junction_box=0.3     # kg
    )

    print(f"\nModule Composition:")
    print(f"  Total mass: {module.total_mass:.2f} kg")
    print(f"  Glass: {module.glass:.2f} kg ({module.glass/module.total_mass*100:.1f}%)")
    print(f"  Aluminum: {module.aluminum:.2f} kg ({module.aluminum/module.total_mass*100:.1f}%)")
    print(f"  Silicon: {module.silicon:.2f} kg ({module.silicon/module.total_mass*100:.1f}%)")

    # Initialize calculator
    calculator = MaterialRecoveryCalculator()

    # Perform full recovery analysis
    results = calculator.full_recovery_analysis(
        composition=module,
        num_modules=1000,
        transport_distance_km=200.0
    )

    print(f"\nRecovery Analysis (1000 modules):")
    print(f"  Overall recovery rate: {results['overall_recovery_rate']*100:.1f}%")
    print(f"  Total cost: ${results['total_cost_usd']:,.2f}")
    print(f"  Cost per module: ${results['cost_per_module_usd']:.2f}")
    print(f"  Cost per kg: ${results['cost_per_kg_usd']:.2f}")

    print(f"\nRecovered Materials:")
    total_recovered = results['total_recovered_materials']
    print(f"  Glass: {total_recovered['glass_kg']:,.1f} kg")
    print(f"  Aluminum: {total_recovered['aluminum_kg']:,.1f} kg")
    print(f"  Silicon: {total_recovered['silicon_kg']:,.1f} kg")
    print(f"  Silver: {total_recovered['silver_kg']:,.3f} kg")
    print(f"  Copper: {total_recovered['copper_kg']:,.1f} kg")


def example_reuse_assessment():
    """Example: Reuse Assessment & Second-Life Applications."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Reuse Assessment & Second-Life Applications")
    print("="*80)

    # Define test results for a used module
    test_results = ModuleTestResults(
        visual_inspection_passed=True,
        electrical_test_passed=True,
        insulation_test_passed=True,
        current_power_w=340,      # 85% of rated power
        rated_power_w=400,
        voltage_v=32.5,
        current_a=10.5,
        fill_factor=0.78,
        insulation_resistance_mohm=50.0,
        defects=[DefectType.DISCOLORATION],
        condition=ModuleCondition.GOOD
    )

    print(f"\nModule Test Results:")
    print(f"  Current power: {test_results.current_power_w}W")
    print(f"  Rated power: {test_results.rated_power_w}W")
    print(f"  Capacity retention: {test_results.capacity_retention*100:.1f}%")
    print(f"  Condition: {test_results.condition.value}")

    # Initialize analyzer
    analyzer = ReuseAnalyzer()

    # Assess reuse eligibility
    eligibility = analyzer.module_testing(test_results, age_years=8)

    print(f"\nReuse Eligibility:")
    print(f"  Eligible: {eligibility.is_eligible}")
    print(f"  Confidence: {eligibility.confidence_score*100:.1f}%")
    print(f"  Estimated remaining life: {eligibility.estimated_remaining_life_years:.1f} years")
    print(f"  Warranty eligible: {eligibility.warranty_eligible}")
    print(f"\nReasons:")
    for reason in eligibility.reasons:
        print(f"  - {reason}")

    # Analyze second-life markets
    markets = analyzer.second_life_markets(
        capacity_retention=test_results.capacity_retention,
        available_quantity_kw=340.0,
        module_specs={"voltage": 32.5, "current": 10.5}
    )

    print(f"\nSecond-Life Market Opportunities:")
    for i, market in enumerate(markets[:3], 1):  # Show top 3
        print(f"\n  {i}. {market.market_name}")
        print(f"     Application: {market.application}")
        print(f"     Price: ${market.typical_price_per_watt:.2f}/W")
        print(f"     Demand: {market.demand_level}")
        print(f"     Market size: {market.market_size_mw:.0f} MW")


def example_repair_optimization():
    """Example: Repair Optimization & Maintenance Strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Repair Optimization & Maintenance Strategies")
    print("="*80)

    # Define detected defects
    defects = [
        Defect(
            defect_id="D001",
            defect_type="junction_box_failure",
            severity=DefectSeverity.HIGH,
            location="Module A1-15",
            power_loss_w=25.0,
            safety_risk=True,
            progression_rate=5.0
        ),
        Defect(
            defect_id="D002",
            defect_type="cell_crack",
            severity=DefectSeverity.MEDIUM,
            location="Module B2-08",
            power_loss_w=10.0,
            safety_risk=False,
            progression_rate=2.0
        ),
    ]

    # Initialize optimizer
    optimizer = RepairOptimizer()

    # Prioritize defects
    prioritized = optimizer.defect_prioritization(
        defects=defects,
        system_size_kw=100.0,
        age_years=10.0
    )

    print(f"\nDefect Prioritization:")
    for i, p in enumerate(prioritized, 1):
        print(f"\n  {i}. {p.defect.defect_type} ({p.defect.severity.value})")
        print(f"     Priority score: {p.priority_score:.1f}")
        print(f"     Recommended action: {p.recommended_action.value}")
        print(f"     Estimated cost: ${p.estimated_repair_cost:.2f}")
        print(f"     Urgency: {p.urgency_days} days")

    # Repair vs Replace analysis
    repair_costs = RepairCost(
        labor_cost=150.0,
        parts_cost=75.0,
        equipment_cost=25.0,
        downtime_cost=50.0
    )

    decision = optimizer.repair_vs_replace(
        module_age_years=10,
        current_power_w=340,
        rated_power_w=400,
        repair_costs=repair_costs,
        defects=defects
    )

    print(f"\nRepair vs Replace Decision:")
    print(f"  Decision: {decision.decision.upper()}")
    print(f"  Confidence: {decision.confidence*100:.1f}%")
    print(f"  Repair cost: ${decision.repair_cost:.2f}")
    print(f"  Replacement cost: ${decision.replacement_cost:.2f}")
    print(f"  Repair NPV: ${decision.repair_npv:.2f}")
    print(f"  Replace NPV: ${decision.replace_npv:.2f}")
    print(f"\nReasoning:")
    for reason in decision.reasoning:
        print(f"  - {reason}")


def example_recycling_economics():
    """Example: Recycling Economics & Material Value."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Recycling Economics & Material Value")
    print("="*80)

    # Initialize economics analyzer
    economics = RecyclingEconomics()

    # Define recovered materials
    recovered_materials = {
        "aluminum": 2.4,      # kg per module
        "glass": 14.0,        # kg per module
        "silicon": 0.425,     # kg per module
        "silver": 0.0045,     # kg per module
        "copper": 0.19        # kg per module
    }

    print(f"\nRecovered Materials (per module):")
    for material, amount in recovered_materials.items():
        print(f"  {material.capitalize()}: {amount:.4f} kg")

    # Calculate material pricing
    revenue = economics.material_pricing(recovered_materials)

    print(f"\nMaterial Revenue:")
    print(f"  Aluminum: ${revenue.aluminum_revenue:.2f}")
    print(f"  Glass: ${revenue.glass_revenue:.2f}")
    print(f"  Silicon: ${revenue.silicon_revenue:.2f}")
    print(f"  Silver: ${revenue.silver_revenue:.2f}")
    print(f"  Copper: ${revenue.copper_revenue:.2f}")
    print(f"  Total: ${revenue.total_revenue:.2f}")

    # Calculate ROI
    roi = economics.recycling_roi(
        num_modules=10000,
        avg_module_weight_kg=20.0,
        recycling_cost_per_module=15.0,
        recovered_materials=recovered_materials,
        facility_investment=5_000_000,
        annual_volume=50000
    )

    print(f"\nROI Analysis (10,000 modules):")
    print(f"  Total revenue: ${roi.total_revenue:,.2f}")
    print(f"  Total costs: ${roi.total_costs:,.2f}")
    print(f"  Environmental credits: ${roi.environmental_credits:,.2f}")
    print(f"  Net profit: ${roi.net_profit:,.2f}")
    print(f"  ROI: {roi.roi_percent:.2f}%")
    if roi.payback_period_years:
        print(f"  Payback period: {roi.payback_period_years:.1f} years")

    # Environmental credits
    env_credits = economics.environmental_credits(
        num_modules=10000,
        avg_module_weight_kg=20.0,
        region="EU"
    )

    print(f"\nEnvironmental Credits:")
    print(f"  Carbon credits: ${env_credits.carbon_credits_usd:,.2f}")
    print(f"  Recycling certificates: ${env_credits.recycling_certificates_usd:,.2f}")
    print(f"  EPR compliance: ${env_credits.epr_compliance_value_usd:,.2f}")
    print(f"  Carbon offset: {env_credits.carbon_offset_tons:,.1f} tons CO2eq")


def example_lca_analysis():
    """Example: Environmental Impact & LCA Analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Environmental Impact & LCA Analysis")
    print("="*80)

    # Initialize LCA analyzer
    lca = LCAAnalyzer()

    # Calculate carbon footprint
    carbon = lca.carbon_footprint(
        module_power_w=400,
        module_weight_kg=20.0,
        manufacturing_location="China",
        transportation_km=10000,
        lifetime_years=25,
        recycling_at_eol=True
    )

    print(f"\nCarbon Footprint Breakdown:")
    print(f"  Raw materials: {carbon.raw_materials_kg_co2eq:.1f} kg CO2eq")
    print(f"  Manufacturing: {carbon.manufacturing_kg_co2eq:.1f} kg CO2eq")
    print(f"  Transportation: {carbon.transportation_kg_co2eq:.1f} kg CO2eq")
    print(f"  Installation: {carbon.installation_kg_co2eq:.1f} kg CO2eq")
    print(f"  Operation: {carbon.operation_kg_co2eq:.1f} kg CO2eq")
    print(f"  Maintenance: {carbon.maintenance_kg_co2eq:.1f} kg CO2eq")
    print(f"  End-of-life: {carbon.end_of_life_kg_co2eq:.1f} kg CO2eq")
    print(f"  TOTAL: {carbon.total_kg_co2eq:.1f} kg CO2eq ({carbon.total_tons_co2eq:.2f} tons)")

    # Energy payback analysis
    energy = lca.energy_payback(
        module_power_w=400,
        module_weight_kg=20.0,
        annual_irradiation_kwh_per_m2=1800,
        module_area_m2=2.0,
        manufacturing_location="China"
    )

    print(f"\nEnergy Payback Analysis:")
    print(f"  Embodied energy: {energy.embodied_energy_kwh:,.0f} kWh")
    print(f"  Annual generation: {energy.annual_energy_generation_kwh:,.0f} kWh")
    print(f"  Energy payback time: {energy.energy_payback_time_years:.2f} years")
    print(f"  Energy ROI (EROI): {energy.energy_return_on_investment:.1f}x")
    print(f"  Lifetime net energy: {energy.lifetime_net_energy_kwh:,.0f} kWh")
    print(f"  Carbon payback time: {energy.carbon_payback_time_years:.2f} years")

    # Environmental indicators
    indicators = lca.environmental_indicators(
        module_power_w=400,
        module_weight_kg=20.0,
        manufacturing_location="China"
    )

    print(f"\nEnvironmental Indicators:")
    print(f"  Climate change: {indicators.climate_change_kg_co2eq:.1f} kg CO2eq")
    print(f"  Acidification: {indicators.acidification_kg_so2eq:.3f} kg SO2eq")
    print(f"  Eutrophication: {indicators.eutrophication_kg_po4eq:.3f} kg PO4eq")
    print(f"  Water consumption: {indicators.water_consumption_m3:.2f} m³")
    print(f"  Resource depletion: {indicators.resource_depletion_score:.1f}/100")


def example_circularity_dashboard():
    """Example: Circularity UI & Dashboard."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Circular Economy Score & Dashboard")
    print("="*80)

    # Initialize UI
    ui = CircularityUI()

    # Calculate circular economy score
    ce_score = ui.circular_economy_score(
        material_circularity_index=0.75,
        recovery_rate=0.85,
        reuse_rate=0.20,
        lifetime_extension_factor=1.3,
        carbon_footprint_kg=1500,
        roi_percent=12.5
    )

    print(f"\nCircular Economy Score:")
    print(f"  Overall score: {ce_score.overall_score:.1f}/100")
    print(f"  Rating: {ce_score.rating}")
    print(f"\nSub-scores:")
    print(f"  Material efficiency: {ce_score.material_efficiency_score:.1f}/100")
    print(f"  Product longevity: {ce_score.product_longevity_score:.1f}/100")
    print(f"  Recycling effectiveness: {ce_score.recycling_effectiveness_score:.1f}/100")
    print(f"  Environmental impact: {ce_score.environmental_impact_score:.1f}/100")
    print(f"  Economic viability: {ce_score.economic_viability_score:.1f}/100")

    print(f"\nRecommendations:")
    for i, rec in enumerate(ce_score.recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\nVisualizations generated successfully!")
    print(f"  - Material flow Sankey diagram")
    print(f"  - 3R metrics dashboard")
    print(f"  - Circular economy radar chart")


if __name__ == "__main__":
    """Run all examples."""
    print("\n" + "="*80)
    print("PV CIRCULARITY SIMULATOR - USAGE EXAMPLES")
    print("="*80)

    example_material_recovery()
    example_reuse_assessment()
    example_repair_optimization()
    example_recycling_economics()
    example_lca_analysis()
    example_circularity_dashboard()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
