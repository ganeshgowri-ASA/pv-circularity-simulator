"""
Example usage of Griddler Pro Integration & Metallization Optimization Module

This script demonstrates the key features of the griddler_integration module.
"""

import sys
sys.path.insert(0, '/home/user/pv-circularity-simulator')

from src.modules.griddler_integration import (
    GriddlerInterface,
    MetallizationParameters,
    GridPatternType,
    BusbarConfiguration,
    OptimizationObjective,
    MetallizationType,
    compare_patterns,
    calculate_module_level_impact
)


def example_basic_design():
    """Example 1: Basic finger pattern design."""
    print("=" * 80)
    print("Example 1: Basic Finger Pattern Design")
    print("=" * 80)

    griddler = GriddlerInterface()

    cell_params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'finger_count': 100,
        'finger_width': 50.0,
        'busbar_count': 3,
        'busbar_width': 1500.0
    }

    pattern = griddler.design_finger_pattern(cell_params)

    print(f"Pattern Type: {pattern.pattern_type.value}")
    print(f"Busbar Configuration: {pattern.busbar_config.value}")
    print(f"Number of Fingers: {len(pattern.finger_positions)}")
    print(f"Number of Busbars: {len(pattern.busbar_positions)}")
    print(f"Shading Fraction: {pattern.shading_fraction:.4f} ({pattern.shading_fraction*100:.2f}%)")
    print(f"Total Finger Length: {pattern.total_finger_length:.2f} mm")
    print(f"Total Busbar Length: {pattern.total_busbar_length:.2f} mm")
    print()


def example_busbar_optimization():
    """Example 2: Busbar width optimization."""
    print("=" * 80)
    print("Example 2: Busbar Width Optimization")
    print("=" * 80)

    griddler = GriddlerInterface()

    params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'busbar_count': 5,
        'current_density': 0.042,
        'voltage': 0.65,
        'height': 20.0
    }

    optimal_width = griddler.optimize_busbar_width(params)

    print(f"Optimal Busbar Width: {optimal_width:.2f} µm")
    print(f"For {params['busbar_count']} busbars (5BB)")
    print(f"Cell dimensions: {params['cell_width']:.2f} × {params['cell_length']:.2f} mm")
    print()


def example_resistance_calculation():
    """Example 3: Series resistance calculation."""
    print("=" * 80)
    print("Example 3: Series Resistance Calculation")
    print("=" * 80)

    griddler = GriddlerInterface()

    cell_params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'finger_count': 100,
        'finger_width': 50.0,
        'busbar_count': 5,
        'busbar_width': 1200.0
    }

    pattern = griddler.design_finger_pattern(cell_params)

    params = MetallizationParameters(
        finger_width=50.0,
        finger_count=100,
        busbar_count=5,
        busbar_width=1200.0,
        cell_width=156.75,
        cell_length=156.75
    )

    rs = griddler.calculate_series_resistance(pattern, params)

    print(f"Total Series Resistance: {rs:.4f} Ω·cm²")
    print(f"  Finger Resistance: {pattern.finger_resistance:.4f} Ω·cm²")
    print(f"  Busbar Resistance: {pattern.busbar_resistance:.4f} Ω·cm²")
    print(f"  Contact Resistance: {pattern.contact_resistance_total:.4f} Ω·cm²")
    print(f"  Emitter Resistance: {pattern.emitter_resistance:.4f} Ω·cm²")
    print()


def example_metallization_optimization():
    """Example 4: Full metallization optimization."""
    print("=" * 80)
    print("Example 4: Metallization Optimization")
    print("=" * 80)

    griddler = GriddlerInterface(silver_price_per_gram=0.75)

    cell_design = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'jsc': 0.042,  # A/cm²
        'voc': 0.68,   # V
        'busbar_config': BusbarConfiguration.BB5
    }

    print("Optimizing for BALANCED objective...")
    optimized = griddler.optimize_metallization(
        cell_design,
        objective=OptimizationObjective.BALANCED
    )

    print(f"\nOptimization Results:")
    print(f"  Objective Value: {optimized.objective_value:.6f}")
    print(f"  Finger Width: {optimized.parameters.finger_width:.2f} µm")
    print(f"  Finger Count: {optimized.parameters.finger_count}")
    print(f"  Busbar Width: {optimized.parameters.busbar_width:.2f} µm")
    print(f"  Busbar Count: {optimized.parameters.busbar_count}")
    print(f"\nPerformance Metrics:")
    print(f"  Optical Efficiency: {optimized.optical_efficiency:.4f} ({optimized.optical_efficiency*100:.2f}%)")
    print(f"  Electrical Efficiency: {optimized.electrical_efficiency:.4f} ({optimized.electrical_efficiency*100:.2f}%)")
    print(f"  Combined Efficiency: {optimized.combined_efficiency:.4f} ({optimized.combined_efficiency*100:.2f}%)")
    print(f"  Series Resistance: {optimized.pattern.series_resistance:.4f} Ω·cm²")
    print(f"  Shading Fraction: {optimized.pattern.shading_fraction:.4f} ({optimized.pattern.shading_fraction*100:.2f}%)")
    print(f"\nCost Metrics:")
    print(f"  Silver Mass: {optimized.pattern.silver_mass:.2f} mg/cell")
    print(f"  Silver Cost: ${optimized.silver_cost_per_cell:.4f}/cell")
    print(f"  Processing Cost: ${optimized.processing_cost_per_cell:.4f}/cell")
    print(f"  Total Cost: ${optimized.silver_cost_per_cell + optimized.processing_cost_per_cell:.4f}/cell")
    print(f"  Performance/Cost Ratio: {optimized.performance_to_cost_ratio:.2f}")
    print()


def example_advanced_patterns():
    """Example 5: Advanced pattern generation."""
    print("=" * 80)
    print("Example 5: Advanced Pattern Generation")
    print("=" * 80)

    griddler = GriddlerInterface()

    cell_params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'busbar_count': 12,
        'finger_count': 120,
        'busbar_width': 400.0
    }

    # Multi-busbar pattern
    mbb_pattern = griddler.generate_advanced_pattern(
        GridPatternType.MULTI_BUSBAR,
        cell_params
    )
    print(f"Multi-Busbar Pattern:")
    print(f"  Type: {mbb_pattern.pattern_type.value}")
    print(f"  Busbars: {len(mbb_pattern.busbar_positions)}")
    print(f"  Shading Fraction: {mbb_pattern.shading_fraction:.4f}")

    # IBC pattern
    ibc_pattern = griddler.generate_advanced_pattern(
        GridPatternType.IBC,
        cell_params
    )
    print(f"\nIBC Pattern:")
    print(f"  Type: {ibc_pattern.pattern_type.value}")
    print(f"  Fingers: {len(ibc_pattern.finger_positions)}")
    print(f"  Front-side Shading: {ibc_pattern.shading_fraction:.4f}")

    # Bifacial pattern
    bifacial_pattern = griddler.generate_advanced_pattern(
        GridPatternType.BIFACIAL,
        cell_params
    )
    print(f"\nBifacial Pattern:")
    print(f"  Type: {bifacial_pattern.pattern_type.value}")
    print(f"  Optimized for front & rear capture")
    print(f"  Shading Fraction: {bifacial_pattern.shading_fraction:.4f}")
    print()


def example_cost_analysis():
    """Example 6: Detailed cost analysis."""
    print("=" * 80)
    print("Example 6: Cost Analysis")
    print("=" * 80)

    griddler = GriddlerInterface(silver_price_per_gram=0.75)

    cell_params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'finger_count': 100,
        'finger_width': 50.0,
        'busbar_count': 5,
        'busbar_width': 1200.0
    }

    pattern = griddler.design_finger_pattern(cell_params)
    params = MetallizationParameters(
        finger_width=50.0,
        finger_count=100,
        busbar_count=5,
        busbar_width=1200.0
    )

    # Screen printing cost analysis
    sp_cost = griddler.calculate_cost_analysis(
        pattern, params, MetallizationType.SCREEN_PRINTING
    )

    print("Screen Printing Cost Analysis:")
    print(f"  Silver Mass: {sp_cost.silver_mass_mg:.2f} mg/cell")
    print(f"  Silver Price: ${sp_cost.silver_price_per_gram:.2f}/g")
    print(f"  Silver Cost: ${sp_cost.silver_cost:.4f}/cell")
    print(f"  Screen Printing: ${sp_cost.screen_printing_cost:.4f}/cell")
    print(f"  Firing Cost: ${sp_cost.firing_cost:.4f}/cell")
    print(f"  Total Cost: ${sp_cost.total_cost:.4f}/cell")
    print(f"  Paste Consumption: {sp_cost.paste_consumption_mg_per_cell:.2f} mg/cell")

    # Copper plating cost analysis
    cp_cost = griddler.calculate_cost_analysis(
        pattern, params, MetallizationType.COPPER_PLATING
    )

    print(f"\nCopper Plating Cost Analysis:")
    print(f"  Total Cost: ${cp_cost.total_cost:.4f}/cell")
    print(f"  Plating Cost: ${cp_cost.alternative_process_cost:.4f}/cell")
    print(f"  Cost Saving vs Screen Printing: ${sp_cost.total_cost - cp_cost.total_cost:.4f}/cell")
    print()


def example_cad_export():
    """Example 7: CAD export."""
    print("=" * 80)
    print("Example 7: CAD Export")
    print("=" * 80)

    griddler = GriddlerInterface()

    cell_params = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'finger_count': 50,  # Fewer fingers for cleaner export
        'finger_width': 50.0,
        'busbar_count': 3,
        'busbar_width': 1500.0
    }

    pattern = griddler.design_finger_pattern(cell_params)
    params = MetallizationParameters(
        finger_width=50.0,
        finger_count=50,
        busbar_count=3,
        busbar_width=1500.0
    )

    # Export to JSON
    json_export = griddler.export_to_cad(pattern, params, "JSON")
    print("JSON Export (first 500 chars):")
    print(json_export[:500])
    print("...")

    # Export to SVG
    svg_export = griddler.export_to_cad(pattern, params, "SVG")
    print(f"\nSVG Export generated ({len(svg_export)} bytes)")
    print("First 300 chars:")
    print(svg_export[:300])
    print("...")
    print()


def example_pattern_comparison():
    """Example 8: Compare multiple optimization objectives."""
    print("=" * 80)
    print("Example 8: Pattern Comparison")
    print("=" * 80)

    griddler = GriddlerInterface(silver_price_per_gram=0.75)

    cell_design = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'jsc': 0.042,
        'voc': 0.68,
        'busbar_config': BusbarConfiguration.BB5
    }

    objectives = [
        OptimizationObjective.MINIMIZE_RESISTANCE,
        OptimizationObjective.MINIMIZE_SHADING,
        OptimizationObjective.MINIMIZE_SILVER,
        OptimizationObjective.BALANCED
    ]

    patterns = []
    for obj in objectives:
        print(f"Optimizing for {obj.value}...")
        optimized = griddler.optimize_metallization(cell_design, objective=obj)
        patterns.append(optimized)

    comparison = compare_patterns(patterns)

    print(f"\nComparison of {comparison['count']} patterns:")
    print(f"{'Objective':<25} {'Efficiency':>12} {'Resistance':>12} {'Silver (mg)':>12} {'Cost ($)':>10}")
    print("-" * 75)

    for i, obj in enumerate(objectives):
        p = comparison['patterns'][i]
        total_cost = patterns[i].silver_cost_per_cell + patterns[i].processing_cost_per_cell
        print(f"{obj.value:<25} {p['combined_efficiency']:>12.4f} {p['series_resistance']:>12.4f} "
              f"{p['silver_mass_mg']:>12.2f} {total_cost:>10.4f}")

    print(f"\nBest Efficiency: {objectives[comparison['best_efficiency']].value}")
    print(f"Lowest Cost: {objectives[comparison['lowest_cost']].value}")
    print(f"Best Performance/Cost: {objectives[comparison['best_performance_to_cost']].value}")
    print()


def example_module_level_impact():
    """Example 9: Module-level impact analysis."""
    print("=" * 80)
    print("Example 9: Module-Level Impact")
    print("=" * 80)

    griddler = GriddlerInterface(silver_price_per_gram=0.75)

    cell_design = {
        'cell_width': 156.75,
        'cell_length': 156.75,
        'jsc': 0.042,
        'voc': 0.68,
        'busbar_config': BusbarConfiguration.BB5
    }

    optimized = griddler.optimize_metallization(
        cell_design,
        objective=OptimizationObjective.BALANCED
    )

    module_config = {
        'cells_in_series': 60,
        'cells_in_parallel': 1
    }

    module_impact = calculate_module_level_impact(optimized, module_config)

    print(f"Module Configuration: {module_config['cells_in_series']}S{module_config['cells_in_parallel']}P")
    print(f"  Total Cells: {module_impact['cells_count']}")
    print(f"  Module Series Resistance: {module_impact['module_series_resistance_ohm']:.4f} Ω")
    print(f"  Total Silver Mass: {module_impact['total_silver_mass_g']:.2f} g")
    print(f"  Total Metallization Cost: ${module_impact['total_metallization_cost_usd']:.2f}")
    print(f"  Module Efficiency Loss: {module_impact['module_efficiency_loss_percent']:.2f}%")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Griddler Pro Integration - Example Suite" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    example_basic_design()
    example_busbar_optimization()
    example_resistance_calculation()
    example_metallization_optimization()
    example_advanced_patterns()
    example_cost_analysis()
    example_cad_export()
    example_pattern_comparison()
    example_module_level_impact()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
