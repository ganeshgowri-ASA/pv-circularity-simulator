"""Example usage of RepowerAnalyzer for PV system repower feasibility analysis.

This example demonstrates a complete workflow for analyzing whether and how to
repower an existing PV system, including:
1. Capacity upgrade analysis
2. Component replacement planning
3. Technical feasibility assessment
4. Economic viability analysis
"""

from datetime import date, timedelta

from pv_simulator import RepowerAnalyzer
from pv_simulator.core.enums import (
    ClimateZone,
    ComponentType,
    HealthStatus,
    ModuleTechnology,
    RepowerStrategy,
)
from pv_simulator.core.models import (
    ComponentHealth,
    CostBreakdown,
    Location,
    PVModule,
    PVSystem,
    RepowerScenario,
)


def create_example_system() -> PVSystem:
    """Create an example 10-year-old 100kW PV system for analysis."""

    # Define system location
    location = Location(
        latitude=37.7749,
        longitude=-122.4194,
        climate_zone=ClimateZone.TEMPERATE,
        avg_annual_irradiance=1800.0,
        avg_temperature=15.5,
        elevation=50.0,
    )

    # Define current module specs
    current_module = PVModule(
        technology=ModuleTechnology.MONO_SI,
        rated_power=350.0,
        efficiency=0.20,
        area=1.75,
        degradation_rate=0.005,
        temperature_coefficient=-0.4,
        warranty_years=25,
        cost_per_watt=0.35,
    )

    # Define component health status
    component_health = [
        ComponentHealth(
            component_type=ComponentType.MODULE,
            status=HealthStatus.GOOD,
            performance_ratio=0.88,
            age_years=10.0,
            expected_lifetime=25.0,
            failure_probability=0.05,
            maintenance_cost_annual=500.0,
            replacement_cost=35000.0,
        ),
        ComponentHealth(
            component_type=ComponentType.INVERTER,
            status=HealthStatus.FAIR,
            performance_ratio=0.92,
            age_years=10.0,
            expected_lifetime=15.0,
            failure_probability=0.15,
            maintenance_cost_annual=800.0,
            replacement_cost=12000.0,
        ),
        ComponentHealth(
            component_type=ComponentType.RACKING,
            status=HealthStatus.EXCELLENT,
            performance_ratio=1.0,
            age_years=10.0,
            expected_lifetime=30.0,
            failure_probability=0.01,
            maintenance_cost_annual=200.0,
            replacement_cost=15000.0,
        ),
        ComponentHealth(
            component_type=ComponentType.MONITORING,
            status=HealthStatus.POOR,
            performance_ratio=0.70,
            age_years=10.0,
            expected_lifetime=10.0,
            failure_probability=0.30,
            maintenance_cost_annual=150.0,
            replacement_cost=3000.0,
        ),
    ]

    # Create system
    num_modules = 286  # ~100 kW system
    dc_capacity = (num_modules * current_module.rated_power) / 1000

    return PVSystem(
        system_id="EXAMPLE-SYSTEM-001",
        installation_date=date.today() - timedelta(days=365 * 10),
        location=location,
        module=current_module,
        num_modules=num_modules,
        dc_capacity=dc_capacity,
        ac_capacity=95.0,
        inverter_efficiency=0.96,
        system_losses=0.14,
        component_health=component_health,
        current_performance_ratio=0.78,
        avg_annual_production=140000.0,
    )


def main():
    """Run complete repower analysis workflow."""

    print("=" * 80)
    print("PV SYSTEM REPOWER FEASIBILITY ANALYSIS")
    print("=" * 80)
    print()

    # Create example system
    system = create_example_system()
    print(f"Analyzing System: {system.system_id}")
    print(f"  Current Capacity: {system.dc_capacity:.2f} kW DC")
    print(f"  Installation Date: {system.installation_date}")
    print(f"  System Age: {(date.today() - system.installation_date).days / 365.25:.1f} years")
    print(f"  Current Performance Ratio: {system.current_performance_ratio:.2%}")
    print(f"  Annual Production: {system.avg_annual_production:,.0f} kWh/year")
    print()

    # Initialize analyzer
    analyzer = RepowerAnalyzer()

    # ========================================================================
    # STEP 1: CAPACITY UPGRADE ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("STEP 1: CAPACITY UPGRADE ANALYSIS")
    print("=" * 80)

    capacity_analysis = analyzer.capacity_upgrade_analysis(
        system,
        available_roof_area=300.0,  # 300 m² additional space available
    )

    print(f"\nCurrent Capacity: {capacity_analysis.current_capacity:.2f} kW")
    print(f"Maximum Additional Capacity: {capacity_analysis.max_additional_capacity:.2f} kW")
    print(f"Recommended Upgrade: {capacity_analysis.recommended_upgrade:.2f} kW")
    print(f"Limiting Factor: {capacity_analysis.limiting_factor}")
    print(f"\nAvailable Resources:")
    print(f"  Space: {capacity_analysis.space_available:.2f} m²")
    print(f"  Structural: {capacity_analysis.structural_capacity_available:.2f} kg")
    print(f"  Electrical: {capacity_analysis.electrical_capacity_available:.2f} kW")

    print(f"\nUpgrade Scenarios:")
    for scenario in capacity_analysis.upgrade_scenarios:
        print(
            f"  {scenario['upgrade_capacity_kw']:.2f} kW upgrade → "
            f"{scenario['total_capacity_kw']:.2f} kW total "
            f"({scenario['capacity_increase_pct']:.1f}% increase, "
            f"{scenario['num_additional_modules']} modules)"
        )

    # ========================================================================
    # STEP 2: COMPONENT REPLACEMENT PLANNING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: COMPONENT REPLACEMENT PLANNING")
    print("=" * 80)

    replacement_plan = analyzer.component_replacement_planning(
        system, planning_horizon_years=5
    )

    print(f"\nTotal Replacement Cost: ${replacement_plan.total_replacement_cost:,.2f}")
    print(f"\nImmediate Replacements ({len(replacement_plan.immediate_replacements)}):")
    for component in replacement_plan.immediate_replacements:
        print(
            f"  • {component.component_type.value}: {component.status.value} "
            f"(${component.replacement_cost:,.2f})"
        )

    print(f"\nShort-term Replacements (0-1 year, {len(replacement_plan.short_term_replacements)}):")
    for component in replacement_plan.short_term_replacements:
        print(
            f"  • {component.component_type.value}: {component.status.value} "
            f"(${component.replacement_cost:,.2f})"
        )

    print(f"\nMedium-term Replacements (1-3 years, {len(replacement_plan.medium_term_replacements)}):")
    for component in replacement_plan.medium_term_replacements:
        print(
            f"  • {component.component_type.value}: {component.status.value} "
            f"(${component.replacement_cost:,.2f})"
        )

    print(f"\nPriority Order:")
    for i, comp_type in enumerate(replacement_plan.priority_order[:5], 1):
        print(f"  {i}. {comp_type.value}")

    if replacement_plan.risk_mitigation_plan:
        print(f"\nRisk Mitigation Strategies:")
        for comp, strategy in replacement_plan.risk_mitigation_plan.items():
            print(f"  • {comp}: {strategy}")

    # ========================================================================
    # STEP 3: TECHNICAL FEASIBILITY ASSESSMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TECHNICAL FEASIBILITY ASSESSMENT")
    print("=" * 80)

    # Test with recommended upgrade capacity
    target_capacity = system.dc_capacity + capacity_analysis.recommended_upgrade

    # Define new generation module
    new_module = PVModule(
        technology=ModuleTechnology.PERC,
        rated_power=450.0,  # More powerful modern module
        efficiency=0.22,
        area=1.95,
        degradation_rate=0.004,  # Better degradation rate
        temperature_coefficient=-0.35,
        warranty_years=25,
        cost_per_watt=0.32,
    )

    feasibility = analyzer.technical_feasibility_check(
        system, target_capacity=target_capacity, new_module=new_module
    )

    print(f"\nTarget Capacity: {target_capacity:.2f} kW")
    print(f"Overall Feasibility: {'✓ FEASIBLE' if feasibility.is_feasible else '✗ NOT FEASIBLE'}")
    print(f"Feasibility Score: {feasibility.feasibility_score:.1f}/100")
    print(f"\nDetailed Scores:")
    print(f"  Structural:   {feasibility.structural_feasibility:.1f}/100")
    print(f"  Electrical:   {feasibility.electrical_feasibility:.1f}/100")
    print(f"  Spatial:      {feasibility.spatial_feasibility:.1f}/100")
    print(f"  Regulatory:   {feasibility.regulatory_feasibility:.1f}/100")
    print(f"  Integration:  {feasibility.integration_feasibility:.1f}/100")

    if feasibility.constraints:
        print(f"\nConstraints Identified:")
        for constraint in feasibility.constraints:
            print(f"  • {constraint}")

    if feasibility.risks:
        print(f"\nRisks Identified:")
        for risk in feasibility.risks:
            print(f"  • {risk}")

    if feasibility.recommendations:
        print(f"\nRecommendations:")
        for rec in feasibility.recommendations:
            print(f"  • {rec}")

    # ========================================================================
    # STEP 4: ECONOMIC VIABILITY ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: ECONOMIC VIABILITY ANALYSIS")
    print("=" * 80)

    # Create multiple repower scenarios to compare
    scenarios = []

    # Scenario 1: Module-only replacement (maintain current capacity)
    scenarios.append(
        RepowerScenario(
            scenario_id="SCENARIO-1-MODULE-ONLY",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=system.dc_capacity,
            capacity_increase=0.0,
            new_module=new_module,
            num_new_modules=system.num_modules,
            cost_breakdown=CostBreakdown(
                module_costs=system.dc_capacity * 1000 * 0.32,
                inverter_costs=0.0,
                bos_costs=3000.0,
                labor_costs=6000.0,
                permitting_costs=1000.0,
                engineering_costs=1500.0,
            ),
            estimated_annual_production=150000.0,  # 7% improvement from better modules
            performance_improvement=0.07,
            technical_feasibility_score=95.0,
        )
    )

    # Scenario 2: Module + Inverter replacement (maintain capacity)
    scenarios.append(
        RepowerScenario(
            scenario_id="SCENARIO-2-MODULE-INVERTER",
            strategy=RepowerStrategy.SELECTIVE,
            new_dc_capacity=system.dc_capacity,
            capacity_increase=0.0,
            new_module=new_module,
            num_new_modules=system.num_modules,
            cost_breakdown=CostBreakdown(
                module_costs=system.dc_capacity * 1000 * 0.32,
                inverter_costs=15000.0,
                bos_costs=3000.0,
                labor_costs=8000.0,
                permitting_costs=1200.0,
                engineering_costs=2000.0,
            ),
            estimated_annual_production=155000.0,  # 11% improvement
            performance_improvement=0.11,
            technical_feasibility_score=92.0,
        )
    )

    # Scenario 3: Capacity upgrade
    scenarios.append(
        RepowerScenario(
            scenario_id="SCENARIO-3-UPGRADE",
            strategy=RepowerStrategy.UPGRADE,
            new_dc_capacity=target_capacity,
            capacity_increase=capacity_analysis.recommended_upgrade / system.dc_capacity,
            new_module=new_module,
            num_new_modules=int((target_capacity * 1000) / new_module.rated_power),
            cost_breakdown=CostBreakdown(
                module_costs=target_capacity * 1000 * 0.32,
                inverter_costs=18000.0,
                bos_costs=8000.0,
                labor_costs=12000.0,
                permitting_costs=2000.0,
                engineering_costs=3000.0,
            ),
            estimated_annual_production=170000.0,
            performance_improvement=0.21,
            technical_feasibility_score=feasibility.feasibility_score,
        )
    )

    # Define incentives
    incentives = {
        "Federal ITC (30%)": target_capacity * 1000 * 0.32 * 0.30,
        "State Rebate": 5000.0,
    }

    # Analyze economic viability
    economics = analyzer.economic_viability_analysis(
        system,
        repower_scenarios=scenarios,
        electricity_rate=0.12,
        incentives=incentives,
    )

    print(f"\nEconomic Viability: {'✓ VIABLE' if economics.is_viable else '✗ NOT VIABLE'}")
    print(f"Viability Score: {economics.viability_score:.1f}/100")
    print(f"\nIncentives Available: ${sum(economics.incentives_available.values()):,.2f}")
    for name, amount in economics.incentives_available.items():
        print(f"  • {name}: ${amount:,.2f}")

    print(f"\n" + "-" * 80)
    print("SCENARIO COMPARISON")
    print("-" * 80)

    for i, scenario in enumerate(economics.scenarios_analyzed, 1):
        print(f"\nScenario {i}: {scenario.scenario_id}")
        print(f"  Strategy: {scenario.strategy.value}")
        print(f"  New Capacity: {scenario.new_dc_capacity:.2f} kW")
        print(f"  Capacity Increase: {scenario.capacity_increase:.1%}")
        print(f"  Total CAPEX: ${scenario.cost_breakdown.total_capex:,.2f}")

        if scenario.economic_metrics:
            m = scenario.economic_metrics
            print(f"\n  Economic Metrics:")
            print(f"    NPV:            ${m.npv:,.2f}")
            print(f"    IRR:            {m.irr:.2%}")
            print(f"    ROI:            {m.roi:.2%}")
            print(f"    Payback Period: {m.payback_period:.1f} years")
            print(f"    LCOE:           ${m.lcoe:.4f}/kWh")
            print(f"    Annual Value:   ${m.annual_energy_value:,.2f}")
            print(f"    Annual OPEX:    ${m.annual_opex:,.2f}")

    if economics.best_scenario:
        print(f"\n" + "=" * 80)
        print(f"RECOMMENDED SCENARIO: {economics.best_scenario.scenario_id}")
        print("=" * 80)
        print(f"Strategy: {economics.best_scenario.strategy.value}")
        print(f"New Capacity: {economics.best_scenario.new_dc_capacity:.2f} kW")
        print(f"Total Investment: ${economics.best_scenario.cost_breakdown.total_capex:,.2f}")

        if economics.best_scenario.economic_metrics:
            m = economics.best_scenario.economic_metrics
            print(f"NPV: ${m.npv:,.2f}")
            print(f"Payback Period: {m.payback_period:.1f} years")

    # Sensitivity Analysis
    if economics.sensitivity_analysis:
        print(f"\n" + "-" * 80)
        print("SENSITIVITY ANALYSIS (NPV Impact)")
        print("-" * 80)

        for variable, impacts in economics.sensitivity_analysis.items():
            print(f"\n{variable.replace('_', ' ').title()}:")
            for variation, npv_change in impacts.items():
                print(f"  {variation:>5}: ${npv_change:>+12,.2f}")

    # Break-even Analysis
    if economics.break_even_scenarios:
        print(f"\n" + "-" * 80)
        print("BREAK-EVEN ANALYSIS")
        print("-" * 80)
        for metric, value in economics.break_even_scenarios.items():
            if "rate" in metric:
                print(f"{metric}: ${value:.4f}/kWh")
            elif "years" in metric:
                print(f"{metric}: {value:.1f} years")
            else:
                print(f"{metric}: {value:,.0f}")

    # Financing Options
    if economics.financing_options:
        print(f"\n" + "-" * 80)
        print("FINANCING OPTIONS")
        print("-" * 80)
        for i, option in enumerate(economics.financing_options, 1):
            print(f"\n{i}. {option['type']}")
            print(f"   Description: {option['description']}")
            print(f"   Pros: {option['pros']}")
            print(f"   Cons: {option['cons']}")

    # ========================================================================
    # SUMMARY AND RECOMMENDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY AND RECOMMENDATION")
    print("=" * 80)

    print(f"\nSystem Status:")
    print(f"  • Current performance ratio: {system.current_performance_ratio:.1%}")
    print(f"  • Maximum upgrade potential: {capacity_analysis.max_additional_capacity:.2f} kW")
    print(
        f"  • Components needing replacement: "
        f"{len(replacement_plan.immediate_replacements) + len(replacement_plan.short_term_replacements)}"
    )

    print(f"\nFeasibility:")
    print(f"  • Technical feasibility: {'✓' if feasibility.is_feasible else '✗'} ({feasibility.feasibility_score:.1f}/100)")
    print(f"  • Economic viability: {'✓' if economics.is_viable else '✗'} ({economics.viability_score:.1f}/100)")

    if economics.best_scenario:
        print(f"\nRecommended Action:")
        print(f"  • Strategy: {economics.best_scenario.strategy.value}")
        print(f"  • Investment: ${economics.best_scenario.cost_breakdown.total_capex:,.2f}")
        if economics.best_scenario.economic_metrics:
            m = economics.best_scenario.economic_metrics
            print(f"  • Expected Return: ${m.npv:,.2f} NPV over {m.analysis_period} years")
            print(f"  • Payback: {m.payback_period:.1f} years")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
