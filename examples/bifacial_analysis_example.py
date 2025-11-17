"""
Bifacial Module Analysis Example

This example demonstrates comprehensive bifacial PV system analysis including:
- System configuration
- Backside irradiance calculation
- Bifacial gain analysis
- Row spacing optimization
- Performance simulation
- Comparison of different configurations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modules.bifacial_model import (
    BifacialModuleModel,
    BifacialModuleParams,
    BifacialSystemConfig,
    MountingStructure,
    GroundSurface,
    TMY,
    AlbedoType,
    MountingType,
    ViewFactorModel,
    validate_bifacial_system,
    ALBEDO_VALUES,
)


def example_1_basic_calculation():
    """Example 1: Basic backside irradiance calculation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Backside Irradiance Calculation")
    print("="*80)

    model = BifacialModuleModel()

    # Calculate backside irradiance for different albedo values
    albedo_types = [AlbedoType.GRASS, AlbedoType.CONCRETE, AlbedoType.WHITE_MEMBRANE, AlbedoType.SNOW]

    print("\nBackside irradiance for different ground surfaces:")
    print(f"{'Surface':<20} {'Albedo':<10} {'Back Irr (W/m²)':<20} {'Bifacial Gain':<15}")
    print("-" * 80)

    for albedo_type in albedo_types:
        albedo = ALBEDO_VALUES[albedo_type]

        back_irr = model.calculate_backside_irradiance(
            ground_albedo=albedo,
            tilt=30.0,
            clearance=1.0,
            front_poa_global=1000.0,
            front_poa_beam=700.0,
            front_poa_diffuse=300.0,
            dhi=100.0
        )

        gain = model.calculate_bifacial_gain(1000.0, back_irr, 0.70)

        print(f"{albedo_type.value:<20} {albedo:<10.2f} {back_irr:<20.1f} {gain*100:<15.1f}%")


def example_2_view_factor_comparison():
    """Example 2: Compare different view factor models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: View Factor Model Comparison")
    print("="*80)

    structure = MountingStructure(
        mounting_type=MountingType.FIXED_TILT,
        tilt=30.0,
        clearance_height=1.0,
        row_spacing=4.0,
        row_width=1.1,
        n_rows=10
    )

    model = BifacialModuleModel()

    models = [ViewFactorModel.SIMPLE, ViewFactorModel.PEREZ, ViewFactorModel.DURUSOY]

    print(f"\n{'Model':<15} {'F_ground':<15} {'F_sky':<15} {'F_row':<15}")
    print("-" * 60)

    for vf_model in models:
        vf_results = model.model_view_factors(structure, vf_model)

        print(f"{vf_model.value:<15} "
              f"{vf_results['average_f_gnd_beam']:<15.3f} "
              f"{vf_results['average_f_sky']:<15.3f} "
              f"{vf_results['average_f_row']:<15.3f}")


def example_3_row_spacing_optimization():
    """Example 3: Optimize row spacing for maximum energy yield."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Row Spacing Optimization")
    print("="*80)

    model = BifacialModuleModel()

    # Optimize for different ground surfaces
    albedo_scenarios = [
        ("Grass", 0.20),
        ("Concrete", 0.30),
        ("White Membrane", 0.70)
    ]

    print(f"\n{'Surface':<20} {'Optimal GCR':<15} {'Spacing (m)':<15} {'Bifacial Gain':<15}")
    print("-" * 80)

    for surface_name, albedo in albedo_scenarios:
        results = model.optimize_row_spacing(
            module_width=1.1,
            tilt=30.0,
            ground_albedo=albedo,
            clearance=1.0,
            latitude=35.0,
            n_points=15
        )

        print(f"{surface_name:<20} "
              f"{results['optimal_gcr']:<15.2f} "
              f"{results['optimal_spacing']:<15.2f} "
              f"{results['optimal_bifacial_gain']*100:<15.1f}%")


def example_4_clearance_height_study():
    """Example 4: Study effect of clearance height on bifacial gain."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Clearance Height Impact Analysis")
    print("="*80)

    model = BifacialModuleModel()

    clearance_heights = np.linspace(0.5, 3.0, 11)
    albedo = 0.25

    print(f"\n{'Clearance (m)':<20} {'Back Irr (W/m²)':<20} {'Bifacial Gain':<15}")
    print("-" * 60)

    results = []
    for clearance in clearance_heights:
        back_irr = model.calculate_backside_irradiance(
            ground_albedo=albedo,
            tilt=30.0,
            clearance=clearance,
            front_poa_global=1000.0,
            front_poa_beam=700.0,
            front_poa_diffuse=300.0,
            dhi=100.0,
            row_spacing=4.0,
            row_width=1.1
        )

        gain = model.calculate_bifacial_gain(1000.0, back_irr, 0.70)
        results.append({'clearance': clearance, 'back_irr': back_irr, 'gain': gain})

        print(f"{clearance:<20.1f} {back_irr:<20.1f} {gain*100:<15.1f}%")

    print(f"\nOptimal clearance: {results[np.argmax([r['gain'] for r in results])]['clearance']:.1f} m")


def example_5_complete_system_simulation():
    """Example 5: Complete system performance simulation."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete System Performance Simulation")
    print("="*80)

    # Create system configuration
    module = BifacialModuleParams(
        bifaciality=0.75,
        front_efficiency=0.22,
        glass_transmission_front=0.91,
        glass_transmission_rear=0.88,
        temp_coeff_pmax=-0.0037,
        module_width=1.1,
        module_length=2.3
    )

    structure = MountingStructure(
        mounting_type=MountingType.FIXED_TILT,
        tilt=35.0,
        clearance_height=1.5,
        row_spacing=5.0,
        row_width=1.1,
        n_rows=20
    )

    ground = GroundSurface(
        albedo=0.0,
        albedo_type=AlbedoType.GRASS
    )

    config = BifacialSystemConfig(
        module=module,
        structure=structure,
        ground=ground,
        location_latitude=40.0,
        location_longitude=-105.0
    )

    # Validate configuration
    warnings = validate_bifacial_system(config)
    if warnings:
        print("\nConfiguration Warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    # Create TMY data (simplified - one day)
    hours = 24
    tmy = TMY(
        ghi=[0, 0, 0, 0, 0, 50, 200, 400, 600, 800, 900, 950,
             950, 900, 800, 600, 400, 200, 50, 0, 0, 0, 0, 0],
        dni=[0, 0, 0, 0, 0, 100, 300, 500, 700, 800, 850, 900,
             900, 850, 800, 700, 500, 300, 100, 0, 0, 0, 0, 0],
        dhi=[0, 0, 0, 0, 0, 30, 80, 120, 150, 180, 200, 200,
             200, 200, 180, 150, 120, 80, 30, 0, 0, 0, 0, 0],
        temp_air=[15, 14, 13, 13, 14, 16, 18, 21, 24, 27, 29, 31,
                  32, 31, 30, 28, 26, 23, 20, 18, 17, 16, 15, 15],
        wind_speed=[2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                    5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    )

    system = {
        'module': module,
        'structure': structure,
        'ground': ground,
        'latitude': config.location_latitude,
        'longitude': config.location_longitude
    }

    # Run simulation
    model = BifacialModuleModel(config)
    results = model.simulate_bifacial_performance(system, tmy, detailed_output=True)

    # Display results
    print("\nHourly Performance Summary:")
    print("-" * 80)
    print(results[['timestamp', 'front_poa_global', 'back_irradiance',
                   'bifacial_gain', 'cell_temperature', 'power_output']].head(12))

    # Summary statistics
    print("\nDaily Summary:")
    print("-" * 80)
    total_energy = results['power_output'].sum() / 1000.0  # kWh
    avg_bifacial_gain = results[results['power_output'] > 0]['bifacial_gain'].mean()
    max_cell_temp = results['cell_temperature'].max()

    print(f"Total daily energy: {total_energy:.2f} kWh")
    print(f"Average bifacial gain: {avg_bifacial_gain*100:.1f}%")
    print(f"Peak cell temperature: {max_cell_temp:.1f}°C")


def example_6_tracker_vs_fixed_comparison():
    """Example 6: Compare tracker vs fixed tilt systems."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Tracker vs Fixed Tilt Comparison")
    print("="*80)

    model = BifacialModuleModel()

    configurations = [
        {
            'name': 'Fixed Tilt 30°',
            'tilt': 30.0,
            'clearance': 1.0,
            'spacing': 4.0,
            'width': 1.1
        },
        {
            'name': 'Fixed Tilt 20°',
            'tilt': 20.0,
            'clearance': 1.0,
            'spacing': 4.0,
            'width': 1.1
        },
        {
            'name': 'Single-Axis Tracker',
            'tilt': 0.0,  # Flat at noon
            'clearance': 2.0,
            'spacing': 6.0,
            'width': 2.0
        }
    ]

    print(f"\n{'Configuration':<25} {'Back Irr (W/m²)':<20} {'Bifacial Gain':<15}")
    print("-" * 70)

    for config in configurations:
        back_irr = model.calculate_backside_irradiance(
            ground_albedo=0.25,
            tilt=config['tilt'],
            clearance=config['clearance'],
            front_poa_global=1000.0,
            front_poa_beam=700.0,
            front_poa_diffuse=300.0,
            dhi=100.0,
            row_spacing=config['spacing'],
            row_width=config['width']
        )

        gain = model.calculate_bifacial_gain(1000.0, back_irr, 0.70)

        print(f"{config['name']:<25} {back_irr:<20.1f} {gain*100:<15.1f}%")


def example_7_temperature_effects():
    """Example 7: Temperature and wind effects on performance."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Temperature and Wind Effects")
    print("="*80)

    model = BifacialModuleModel()

    scenarios = [
        {'name': 'Cool & Windy', 'temp': 15, 'wind': 5.0},
        {'name': 'Moderate', 'temp': 25, 'wind': 2.0},
        {'name': 'Hot & Calm', 'temp': 35, 'wind': 0.5},
    ]

    print(f"\n{'Scenario':<20} {'Ambient (°C)':<15} {'Wind (m/s)':<15} "
          f"{'Cell Temp (°C)':<20} {'Temp Loss':<15}")
    print("-" * 90)

    for scenario in scenarios:
        cell_temp, temp_loss = model.calculate_temperature_effect(
            front_irr=1000.0,
            back_irr=200.0,
            ambient_temp=scenario['temp'],
            wind_speed=scenario['wind']
        )

        print(f"{scenario['name']:<20} {scenario['temp']:<15} {scenario['wind']:<15.1f} "
              f"{cell_temp:<20.1f} {(1-temp_loss)*100:<15.1f}%")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BIFACIAL MODULE MODELING - COMPREHENSIVE EXAMPLES")
    print("="*80)

    try:
        example_1_basic_calculation()
        example_2_view_factor_comparison()
        example_3_row_spacing_optimization()
        example_4_clearance_height_study()
        example_5_complete_system_simulation()
        example_6_tracker_vs_fixed_comparison()
        example_7_temperature_effects()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
