"""
Basic example of using the PV System Optimization Engine.

This script demonstrates how to:
1. Configure a PV system
2. Set optimization constraints and objectives
3. Run different optimization algorithms
4. Analyze results
"""

from src.models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    OptimizationObjectives,
)
from src.optimization.system_optimizer import SystemOptimizer
from src.optimization.energy_yield_optimizer import EnergyYieldOptimizer
from src.optimization.economic_optimizer import EconomicOptimizer
from src.optimization.layout_optimizer import LayoutOptimizer
from src.optimization.design_space_explorer import DesignSpaceExplorer, ParameterRange


def main() -> None:
    """Run basic optimization example."""

    # Step 1: Configure PV system parameters
    print("=" * 80)
    print("PV SYSTEM OPTIMIZATION ENGINE - BASIC EXAMPLE")
    print("=" * 80)

    parameters = PVSystemParameters(
        # Module specifications
        module_power=450.0,  # 450W modules
        module_efficiency=0.20,  # 20% efficiency
        module_area=2.5,  # 2.5 m² per module
        module_cost=150.0,  # $150 per module
        bifacial=True,  # Use bifacial modules
        bifaciality=0.7,  # 70% bifaciality

        # Tracker configuration
        tracker_type="single_axis",  # Single-axis tracking

        # Site parameters
        latitude=35.0,  # California latitude
        longitude=-120.0,
        available_land_acres=100.0,
        land_cost_per_acre=5000.0,
        albedo=0.25,  # Desert albedo

        # System sizing (initial guess)
        num_modules=10000,  # 4.5 MW system
        gcr=0.4,
        dc_ac_ratio=1.25,

        # Economic parameters
        discount_rate=0.08,
        project_lifetime=25,
        degradation_rate=0.005,
        om_cost_per_kw_year=15.0,
    )

    # Step 2: Define optimization constraints
    constraints = OptimizationConstraints(
        min_gcr=0.2,
        max_gcr=0.6,
        min_dc_ac_ratio=1.1,
        max_dc_ac_ratio=1.5,
        min_tilt=10.0,
        max_tilt=40.0,
        max_land_use_acres=100.0,
        max_shading_loss=0.15,
    )

    # Step 3: Define optimization objectives
    objectives = OptimizationObjectives(
        maximize_energy=0.8,
        minimize_lcoe=1.0,
        minimize_land_use=0.3,
        maximize_npv=0.7,
        minimize_shading=0.5,
        maximize_bifacial_gain=0.4,
    )

    print("\nSystem Configuration:")
    print(f"  Capacity: {parameters.num_modules * parameters.module_power / 1e6:.2f} MW DC")
    print(f"  Modules: {parameters.num_modules:,} × {parameters.module_power}W")
    print(f"  Site: {parameters.latitude}°N, {parameters.longitude}°E")
    print(f"  Tracker: {parameters.tracker_type}")

    # Step 4: Run genetic algorithm optimization
    print("\n" + "=" * 80)
    print("RUNNING GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 80)

    system_optimizer = SystemOptimizer(parameters, constraints, objectives)

    ga_result = system_optimizer.genetic_algorithm_optimizer(
        population_size=50,
        num_generations=30,
    )

    print(f"\nOptimization Status: {'✓ Success' if ga_result.success else '✗ Failed'}")
    print(f"Execution Time: {ga_result.execution_time_seconds:.2f} seconds")
    print(f"Function Evaluations: {ga_result.num_evaluations:,}")

    print("\nOptimal Design:")
    best = ga_result.best_solution
    print(f"  GCR: {best.gcr:.3f}")
    print(f"  DC/AC Ratio: {best.dc_ac_ratio:.2f}")
    print(f"  Tilt Angle: {best.tilt_angle:.1f}°")

    print("\nPerformance Metrics:")
    print(f"  Annual Energy: {best.annual_energy_kwh/1e6:.2f} GWh")
    print(f"  LCOE: ${best.lcoe:.4f}/kWh")
    print(f"  NPV: ${best.npv/1e6:.2f}M")
    print(f"  Land Use: {best.land_use_acres:.1f} acres")
    print(f"  Shading Loss: {best.shading_loss*100:.1f}%")

    # Step 5: Energy yield optimization
    print("\n" + "=" * 80)
    print("ENERGY YIELD OPTIMIZATION")
    print("=" * 80)

    energy_optimizer = EnergyYieldOptimizer(parameters, constraints)

    max_energy, energy_params = energy_optimizer.maximize_annual_energy(method="gradient")

    print(f"\nMaximum Annual Energy: {max_energy/1e6:.2f} GWh")
    print(f"Optimal Tilt: {energy_params['tilt_angle']:.1f}°")
    print(f"Optimal Azimuth: {energy_params['azimuth']:.1f}°")

    # Bifacial gain optimization
    if parameters.bifacial:
        bifacial_gain, bifacial_params = energy_optimizer.optimize_bifacial_gain()
        print(f"\nBifacial Gain: {bifacial_gain*100:.1f}%")
        print(f"Optimal Tilt for Bifacial: {bifacial_params['optimal_tilt_for_bifacial']:.1f}°")

    # Step 6: Economic optimization
    print("\n" + "=" * 80)
    print("ECONOMIC OPTIMIZATION")
    print("=" * 80)

    econ_optimizer = EconomicOptimizer(parameters, constraints)

    min_lcoe, lcoe_params = econ_optimizer.minimize_lcoe(
        vary_dc_ac_ratio=True,
        vary_gcr=True,
    )

    print(f"\nMinimum LCOE: ${min_lcoe:.4f}/kWh")
    print(f"Optimal GCR: {lcoe_params['gcr']:.3f}")
    print(f"Optimal DC/AC: {lcoe_params['dc_ac_ratio']:.2f}")

    max_npv, npv_params = econ_optimizer.maximize_npv(electricity_price=0.06)

    print(f"\nMaximum NPV: ${npv_params['npv']/1e6:.2f}M")
    print(f"Optimal Capacity: {npv_params['capacity_mw']:.2f} MW")

    # Step 7: Layout optimization
    print("\n" + "=" * 80)
    print("LAYOUT OPTIMIZATION")
    print("=" * 80)

    layout_optimizer = LayoutOptimizer(parameters, constraints)

    optimal_gcr, gcr_params = layout_optimizer.optimize_gcr(objective="energy_per_area")

    print(f"\nOptimal GCR: {optimal_gcr:.3f}")
    print(f"Energy per Acre: {gcr_params['energy_per_acre']/1000:.0f} MWh/acre/year")
    print(f"Shading Loss: {gcr_params['shading_loss']*100:.1f}%")

    string_config = layout_optimizer.optimize_string_configuration()

    print("\nString Configuration:")
    print(f"  Modules per String: {string_config['modules_per_string']}")
    print(f"  Number of Strings: {string_config['num_strings']:,}")
    print(f"  Number of Inverters: {string_config['num_inverters']}")
    print(f"  String Voltage: {string_config['string_voltage_nominal']:.0f}V")

    # Step 8: Design space exploration
    print("\n" + "=" * 80)
    print("DESIGN SPACE EXPLORATION")
    print("=" * 80)

    explorer = DesignSpaceExplorer(parameters, constraints)

    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity_results = explorer.sensitivity_analysis(
        parameters_to_vary=["gcr", "dc_ac_ratio", "module_efficiency"],
        variation_percent=10.0,
    )

    print("\nSensitivity Analysis Results:")
    lcoe_sensitivities = [r for r in sensitivity_results if "lcoe" in r.parameter_name]
    sorted_sensitivities = sorted(lcoe_sensitivities, key=lambda x: x.sensitivity_index, reverse=True)

    for result in sorted_sensitivities[:3]:
        param = result.parameter_name.replace("_lcoe", "")
        print(f"  {param}: Sensitivity = {result.sensitivity_index:.3f}, "
              f"Correlation = {result.correlation:+.3f}")

    # Parameter sweep
    print("\nRunning parameter sweep (GCR)...")
    sweep_results = explorer.parameter_sweep(
        [ParameterRange("gcr", 0.2, 0.6, 15)],
        output_metric="lcoe",
        parallel=False,
    )

    best_idx = sweep_results['results'].index(min(sweep_results['results']))
    best_gcr = sweep_results['values'][best_idx]
    best_lcoe = sweep_results['results'][best_idx]

    print(f"  Best GCR from sweep: {best_gcr:.3f}")
    print(f"  Corresponding LCOE: ${best_lcoe:.4f}/kWh")

    # Step 9: Multi-objective optimization
    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 80)

    print("\nRunning NSGA-II to find Pareto frontier...")

    mo_result = system_optimizer.multi_objective_optimization(
        population_size=50,
        num_generations=20,
    )

    print(f"\nPareto Front Size: {len(mo_result.pareto_front)} solutions")
    print(f"Execution Time: {mo_result.execution_time_seconds:.2f} seconds")

    print("\nSample Pareto-Optimal Solutions:")
    print(f"{'#':<4} {'LCOE':>10} {'Energy':>12} {'Land':>10} {'GCR':>8} {'DC/AC':>8}")
    print("-" * 60)

    for i, sol in enumerate(mo_result.pareto_front[:5]):
        design = sol.design
        print(f"{i+1:<4} ${design.lcoe:>9.4f} {design.annual_energy_kwh/1e6:>10.2f} GWh "
              f"{design.land_use_acres:>8.1f} ac {design.gcr:>7.3f} {design.dc_ac_ratio:>7.2f}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nAll optimization algorithms completed successfully!")
    print("Results demonstrate trade-offs between energy, cost, and land use.")


if __name__ == "__main__":
    main()
