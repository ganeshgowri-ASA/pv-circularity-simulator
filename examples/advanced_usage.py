"""
Advanced usage examples for SCAPS wrapper.

This script demonstrates:
1. Batch processing
2. Temperature coefficient calculation
3. Efficiency optimization
4. Custom device creation
"""

from pathlib import Path

from src.modules import (
    CellArchitecture,
    CellTemplates,
    Contact,
    ContactType,
    DeviceParams,
    DopingProfile,
    DopingType,
    InterfaceProperties,
    Layer,
    MaterialProperties,
    MaterialType,
    OpticalProperties,
    SCAPSInterface,
    SimulationSettings,
)


def example_batch_processing(scaps: SCAPSInterface):
    """Demonstrate batch processing with parametric sweeps."""
    print("\n" + "=" * 70)
    print("Example 1: Batch Processing")
    print("=" * 70)

    print("\nRunning batch simulation: Temperature sweep (280K - 340K)")

    # Create batch of simulations
    simulations = []
    temperatures = range(280, 341, 10)

    for temp in temperatures:
        params = CellTemplates.create_perc_cell()
        params_dict = params.model_dump()
        params_dict['settings']['temperature'] = float(temp)
        simulations.append(params_dict)

    # Run batch
    results = scaps.execute_scaps_batch(simulations, max_workers=4)

    # Display results
    print(f"\n{'Temp (K)':<12} {'Voc (V)':<12} {'Jsc (mA/cm²)':<15} {'FF':<10} {'Eff (%)'}")
    print("-" * 70)
    for temp, result in zip(temperatures, results):
        print(f"{temp:<12} {result.voc:<12.4f} {result.jsc:<15.4f} "
              f"{result.ff:<10.4f} {result.efficiency*100:.2f}")


def example_temperature_coefficients(scaps: SCAPSInterface):
    """Demonstrate temperature coefficient calculation."""
    print("\n" + "=" * 70)
    print("Example 2: Temperature Coefficients")
    print("=" * 70)

    print("\nCalculating temperature coefficients for PERC cell...")

    perc = CellTemplates.create_perc_cell()

    coefficients = scaps.calculate_temperature_coefficients(
        params=perc,
        temp_range=(273.0, 343.0),
        temp_step=5.0
    )

    print("\nTemperature Coefficients:")
    print(f"  TC Voc:  {coefficients['temperature_coefficient_voc']*1000:.3f} mV/K")
    print(f"  TC Jsc:  {coefficients['temperature_coefficient_jsc']:.5f} mA/cm²/K")
    print(f"  TC Eff:  {coefficients['temperature_coefficient_efficiency']*100:.5f} %/K")

    print("\nTemperature-dependent performance:")
    print(f"{'Temp (K)':<12} {'Voc (V)':<12} {'Jsc (mA/cm²)':<15} {'Eff (%)'}")
    print("-" * 60)
    for t, v, j, e in zip(
        coefficients['temperatures'][:5],
        coefficients['voc_values'][:5],
        coefficients['jsc_values'][:5],
        coefficients['efficiency_values'][:5]
    ):
        print(f"{t:<12.1f} {v:<12.4f} {j:<15.4f} {e*100:.2f}")
    print("...")


def example_optimization(scaps: SCAPSInterface):
    """Demonstrate efficiency optimization."""
    print("\n" + "=" * 70)
    print("Example 3: Efficiency Optimization")
    print("=" * 70)

    print("\nOptimizing PERC cell parameters...")
    print("Parameters to optimize:")
    print("  - Emitter doping concentration")
    print("  - BSF doping concentration")
    print("  - Emitter thickness")

    base_params = CellTemplates.create_perc_cell()

    # Define optimization bounds
    optimization_params = {
        'layers.0.doping.concentration': (5e18, 5e19),  # Emitter doping
        'layers.2.doping.concentration': (1e18, 1e19),  # BSF doping
        'layers.0.thickness': (300.0, 800.0),  # Emitter thickness (nm)
    }

    print("\nRunning optimization (this may take a while)...")

    opt_params, best_results = scaps.optimize_efficiency(
        base_params=base_params,
        optimization_params=optimization_params,
        max_iterations=20  # Reduced for example
    )

    print("\nOptimization Results:")
    print(f"  Original efficiency:  {scaps.run_simulation(base_params).efficiency*100:.2f}%")
    print(f"  Optimized efficiency: {best_results.efficiency*100:.2f}%")
    print(f"  Improvement:          {(best_results.efficiency - scaps.run_simulation(base_params).efficiency)*100:.2f}%")

    print("\nOptimized parameters:")
    print(f"  Emitter doping:  {opt_params.layers[0].doping.concentration:.2e} cm⁻³")
    print(f"  BSF doping:      {opt_params.layers[2].doping.concentration:.2e} cm⁻³")
    print(f"  Emitter thickness: {opt_params.layers[0].thickness:.1f} nm")


def example_custom_device(scaps: SCAPSInterface):
    """Demonstrate custom device creation."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Device Creation")
    print("=" * 70)

    print("\nCreating custom bifacial solar cell...")

    # Silicon properties
    si_props = MaterialProperties(
        material=MaterialType.SILICON,
        bandgap=1.12,
        electron_affinity=4.05,
        dielectric_constant=11.7,
        electron_mobility=1400.0,
        hole_mobility=450.0,
        nc=2.8e19,
        nv=1.04e19,
        electron_lifetime=2e-3,
        hole_lifetime=2e-3,
        auger_electron=2.8e-31,
        auger_hole=9.9e-32,
    )

    # Create symmetric structure
    layers = [
        # Front FSF (n+)
        Layer(
            name="Front FSF",
            thickness=300.0,
            material_properties=si_props.model_copy(update={'electron_lifetime': 1e-6}),
            doping=DopingProfile(
                doping_type=DopingType.N_TYPE,
                concentration=1e20,
                uniform=False,
                profile_type="gaussian",
                characteristic_length=80.0
            )
        ),
        # Base (n-type for bifacial)
        Layer(
            name="n-type base",
            thickness=160000.0,
            material_properties=si_props,
            doping=DopingProfile(
                doping_type=DopingType.N_TYPE,
                concentration=1e15,
                uniform=True
            )
        ),
        # Rear FSF (n+)
        Layer(
            name="Rear FSF",
            thickness=300.0,
            material_properties=si_props.model_copy(update={'electron_lifetime': 1e-6}),
            doping=DopingProfile(
                doping_type=DopingType.N_TYPE,
                concentration=1e20,
                uniform=False,
                profile_type="gaussian",
                characteristic_length=80.0
            )
        ),
    ]

    # Interfaces
    interfaces = [
        InterfaceProperties(
            name="front-base",
            layer1_index=0,
            layer2_index=1,
            sn=1e2,
            sp=1e2,
        ),
        InterfaceProperties(
            name="base-rear",
            layer1_index=1,
            layer2_index=2,
            sn=1e2,
            sp=1e2,
        ),
    ]

    # Symmetric contacts
    front_contact = Contact(
        contact_type=ContactType.FRONT,
        work_function=4.3,
        surface_recombination_electron=1e5,
        surface_recombination_hole=1e5,
        series_resistance=0.3,
        shunt_resistance=1e10,
    )

    back_contact = Contact(
        contact_type=ContactType.BACK,
        work_function=4.3,
        surface_recombination_electron=1e5,
        surface_recombination_hole=1e5,
        series_resistance=0.3,
        shunt_resistance=1e10,
    )

    # Optics for bifacial
    optics = OpticalProperties(
        arc_enabled=True,
        arc_thickness=75.0,
        arc_refractive_index=2.0,
        illumination_spectrum="AM1.5G",
        light_intensity=1000.0,
        front_reflection=0.03,
        back_reflection=0.05,  # Low reflection for bifacial
    )

    # Create device
    bifacial_device = DeviceParams(
        architecture=CellArchitecture.PERT,
        device_name="Bifacial n-type Cell",
        description="Symmetric bifacial design with front and rear FSF",
        layers=layers,
        interfaces=interfaces,
        front_contact=front_contact,
        back_contact=back_contact,
        optics=optics,
        settings=SimulationSettings(
            temperature=300.0,
            voltage_min=-0.1,
            voltage_max=0.8,
            voltage_step=0.01,
        )
    )

    # Run simulation
    print("\nRunning simulation...")
    results = scaps.run_simulation(bifacial_device)

    print("\nCustom Bifacial Cell Results:")
    print(f"  Voc:        {results.voc:.4f} V")
    print(f"  Jsc:        {results.jsc:.4f} mA/cm²")
    print(f"  Fill Factor: {results.ff:.4f}")
    print(f"  Efficiency:  {results.efficiency*100:.2f}%")
    print(f"  Pmax:       {results.pmax:.4f} mW/cm²")

    # Export
    output_file = Path("./example_outputs/bifacial_custom.json")
    output_file.parent.mkdir(exist_ok=True)
    scaps.export_results(results, output_file, format="json")
    print(f"\n  Results exported to: {output_file}")


def main():
    """Run advanced examples."""
    print("=" * 70)
    print("SCAPS-1D Python Wrapper - Advanced Usage Examples")
    print("=" * 70)

    # Initialize SCAPS interface
    scaps = SCAPSInterface(
        working_directory=Path("./advanced_simulations"),
        cache_directory=Path("./.advanced_cache"),
        enable_cache=True
    )

    # Run examples
    example_batch_processing(scaps)
    example_temperature_coefficients(scaps)
    example_optimization(scaps)
    example_custom_device(scaps)

    print("\n" + "=" * 70)
    print("All advanced examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
