"""
CTM Loss Model Demonstration and Validation Script.

This script demonstrates the full functionality of the CTM Loss Modeling Engine
with various module architectures and generates visualizations.
"""

import sys
sys.path.insert(0, '/home/user/pv-circularity-simulator')

from src.modules.ctm_loss_model import (
    CTMLossModel,
    CellParameters,
    ModuleParameters,
    ModuleType,
    EncapsulantType,
)
import matplotlib.pyplot as plt
import numpy as np


def demo_standard_module():
    """Demonstrate standard 60-cell PERC module analysis."""
    print("\n" + "="*80)
    print("DEMO 1: Standard 60-Cell PERC Module")
    print("="*80)

    # Define cell parameters (typical PERC cell)
    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
        front_grid_coverage=2.5,
        temp_coeff_power=-0.40,
        lid_factor=1.5,
    )

    # Define module parameters
    module = ModuleParameters(
        module_type=ModuleType.STANDARD,
        cells_in_series=60,
        cells_in_parallel=1,
        glass_ar_coating=True,
        encapsulant_type=EncapsulantType.STANDARD_EVA,
    )

    # Create model and calculate
    model = CTMLossModel(cell, module)
    k_factors = model.calculate_all_k_factors()
    module_power = model.calculate_module_power()
    ctm_ratio = model.get_ctm_ratio()

    # Print report
    print(model.generate_report())

    # Generate waterfall chart
    fig = model.generate_loss_waterfall(title="Standard 60-Cell PERC Module - CTM Analysis")
    plt.savefig('/tmp/ctm_standard_waterfall.png', dpi=150, bbox_inches='tight')
    print(f"\nWaterfall chart saved to: /tmp/ctm_standard_waterfall.png")

    return model


def demo_half_cut_module():
    """Demonstrate half-cut module advantages."""
    print("\n" + "="*80)
    print("DEMO 2: Half-Cut 120-Cell Module Comparison")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
    )

    # Standard module
    module_std = ModuleParameters(
        module_type=ModuleType.STANDARD,
        cells_in_series=60,
        cells_in_parallel=1,
    )
    model_std = CTMLossModel(cell, module_std)

    # Half-cut module
    module_hc = ModuleParameters(
        module_type=ModuleType.HALF_CUT,
        cells_in_series=60,
        cells_in_parallel=2,
    )
    model_hc = CTMLossModel(cell, module_hc)

    # Calculate both
    power_std = model_std.calculate_module_power()
    power_hc = model_hc.calculate_module_power()
    ctm_std = model_std.get_ctm_ratio()
    ctm_hc = model_hc.get_ctm_ratio()

    print(f"\nStandard Module:")
    print(f"  Power: {power_std:.2f} W")
    print(f"  CTM Ratio: {ctm_std:.4f} ({(ctm_std-1)*100:+.2f}%)")

    print(f"\nHalf-Cut Module:")
    print(f"  Power: {power_hc:.2f} W")
    print(f"  CTM Ratio: {ctm_hc:.4f} ({(ctm_hc-1)*100:+.2f}%)")

    print(f"\nImprovement: {power_hc - power_std:.2f} W ({(power_hc/power_std-1)*100:+.2f}%)")

    # Compare key k-factors
    print("\nKey K-Factor Comparison:")
    k_std = model_std.calculate_all_k_factors()
    k_hc = model_hc.calculate_all_k_factors()

    compare_factors = ['k12_resistive', 'k9_internal_mismatch', 'k8_cell_gaps']
    for factor in compare_factors:
        print(f"  {factor:25s} Standard: {k_std[factor]:.5f}  Half-Cut: {k_hc[factor]:.5f}")


def demo_bifacial_module():
    """Demonstrate bifacial module with rear gain."""
    print("\n" + "="*80)
    print("DEMO 3: Bifacial Module with Rear Irradiance")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
    )

    # Monofacial
    module_mono = ModuleParameters(
        cells_in_series=60,
        is_bifacial=False,
    )
    model_mono = CTMLossModel(cell, module_mono)

    # Bifacial with glass-glass
    module_bif = ModuleParameters(
        cells_in_series=60,
        is_bifacial=True,
        bifaciality_factor=0.75,
        rear_glass=True,
    )
    model_bif = CTMLossModel(cell, module_bif)

    power_mono = model_mono.calculate_module_power()
    power_bif = model_bif.calculate_module_power()

    print(f"\nMonofacial Module Power: {power_mono:.2f} W")
    print(f"Bifacial Module Power:   {power_bif:.2f} W")
    print(f"Bifacial Gain:           {power_bif - power_mono:.2f} W ({(power_bif/power_mono-1)*100:+.2f}%)")

    k7_mono = model_mono.calculate_k7_rear_optical_properties()
    k7_bif = model_bif.calculate_k7_rear_optical_properties()
    print(f"\nk7 (rear optical) - Mono: {k7_mono:.4f}, Bifacial: {k7_bif:.4f}")


def demo_shingled_module():
    """Demonstrate shingled cell module advantages."""
    print("\n" + "="*80)
    print("DEMO 4: Shingled Cell Module")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
    )

    module = ModuleParameters(
        module_type=ModuleType.SHINGLED,
        cells_in_series=60,
    )
    model = CTMLossModel(cell, module)

    k_factors = model.calculate_all_k_factors()
    power = model.calculate_module_power()

    print(f"\nShingled Module Power: {power:.2f} W")
    print(f"CTM Ratio: {model.get_ctm_ratio():.4f}")

    print("\nKey Shingled Advantages:")
    print(f"  k3 (grid correction):  {k_factors['k3_grid_correction']:.5f} (reduced shading)")
    print(f"  k8 (cell gaps):        {k_factors['k8_cell_gaps']:.5f} (no gaps!)")
    print(f"  k13 (interconnection): {k_factors['k13_interconnection']:.5f}")


def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis capabilities."""
    print("\n" + "="*80)
    print("DEMO 5: Sensitivity Analysis")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
    )
    module = ModuleParameters(cells_in_series=60)
    model = CTMLossModel(cell, module)

    # Analyze multiple parameters
    parameters = [
        'cell.efficiency',
        'module.glass_thickness',
        'module.cell_gap',
        'cell.temp_coeff_power',
    ]

    results = model.multi_parameter_sensitivity(parameters, (0.9, 1.1), 20)

    # Plot sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CTM Sensitivity Analysis', fontsize=16, fontweight='bold')

    for idx, param in enumerate(parameters):
        ax = axes[idx // 2, idx % 2]
        data = results[param]

        # Plot module power vs parameter
        ax.plot(data['parameter_values'], data['module_power'], 'b-', linewidth=2)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('Module Power (W)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sensitivity to {param}')

        # Add baseline
        baseline_idx = len(data['parameter_values']) // 2
        ax.axvline(data['parameter_values'][baseline_idx], color='r',
                   linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/ctm_sensitivity.png', dpi=150, bbox_inches='tight')
    print(f"\nSensitivity analysis saved to: /tmp/ctm_sensitivity.png")


def demo_environmental_effects():
    """Demonstrate environmental factor effects."""
    print("\n" + "="*80)
    print("DEMO 6: Environmental Effects on Module Power")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
        temp_coeff_power=-0.40,
    )

    # Temperature sweep
    temps = np.linspace(15, 75, 13)
    powers_temp = []

    for temp in temps:
        module = ModuleParameters(
            cells_in_series=60,
            operating_temperature=temp,
        )
        model = CTMLossModel(cell, module)
        powers_temp.append(model.calculate_module_power())

    # Irradiance sweep
    irradiances = np.linspace(200, 1000, 9)
    powers_irr = []

    for irr in irradiances:
        module = ModuleParameters(
            cells_in_series=60,
            irradiance=irr,
        )
        model = CTMLossModel(cell, module)
        powers_irr.append(model.calculate_module_power())

    # AOI sweep
    aoi_angles = np.linspace(0, 80, 17)
    powers_aoi = []

    for aoi in aoi_angles:
        module = ModuleParameters(
            cells_in_series=60,
            aoi_angle=aoi,
        )
        model = CTMLossModel(cell, module)
        powers_aoi.append(model.calculate_module_power())

    # Plot environmental effects
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Temperature
    axes[0].plot(temps, powers_temp, 'r-', linewidth=2)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Module Power (W)')
    axes[0].set_title('Temperature Effect (k21)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(25, color='k', linestyle='--', alpha=0.3, label='STC')

    # Irradiance
    axes[1].plot(irradiances, powers_irr, 'g-', linewidth=2)
    axes[1].set_xlabel('Irradiance (W/m²)')
    axes[1].set_ylabel('Module Power (W)')
    axes[1].set_title('Irradiance Effect (k22)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(1000, color='k', linestyle='--', alpha=0.3, label='STC')

    # AOI
    axes[2].plot(aoi_angles, powers_aoi, 'b-', linewidth=2)
    axes[2].set_xlabel('Angle of Incidence (°)')
    axes[2].set_ylabel('Module Power (W)')
    axes[2].set_title('AOI Effect (k24)')
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/ctm_environmental.png', dpi=150, bbox_inches='tight')
    print(f"\nEnvironmental effects plot saved to: /tmp/ctm_environmental.png")

    # Print some values
    print(f"\nPower at 25°C (STC):  {powers_temp[2]:.2f} W")
    print(f"Power at 65°C:        {powers_temp[-2]:.2f} W")
    print(f"Temperature loss:     {(powers_temp[-2]/powers_temp[2]-1)*100:.2f}%")


def demo_module_type_comparison():
    """Compare all module architectures."""
    print("\n" + "="*80)
    print("DEMO 7: Module Architecture Comparison")
    print("="*80)

    cell = CellParameters(
        power_stc=5.25,
        voltage_mpp=0.650,
        current_mpp=8.08,
        voltage_oc=0.720,
        current_sc=8.60,
        efficiency=22.8,
        width=166,
        height=166,
    )

    architectures = {
        'Standard': (ModuleType.STANDARD, 60, 1),
        'Half-Cut': (ModuleType.HALF_CUT, 60, 2),
        'Shingled': (ModuleType.SHINGLED, 60, 1),
        'IBC': (ModuleType.IBC, 60, 1),
    }

    results = {}
    for name, (mod_type, series, parallel) in architectures.items():
        module = ModuleParameters(
            module_type=mod_type,
            cells_in_series=series,
            cells_in_parallel=parallel,
        )
        model = CTMLossModel(cell, module)
        power = model.calculate_module_power()
        ctm = model.get_ctm_ratio()
        results[name] = (power, ctm)

    print("\n{:15s} {:>12s} {:>12s} {:>12s}".format(
        "Architecture", "Power (W)", "CTM Ratio", "vs. Standard"))
    print("-" * 60)

    baseline_power = results['Standard'][0]
    for name, (power, ctm) in results.items():
        diff = (power / baseline_power - 1) * 100
        print("{:15s} {:12.2f} {:12.4f} {:>11s}".format(
            name, power, ctm, f"{diff:+.2f}%"))

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    powers = [results[n][0] for n in names]
    colors = ['blue', 'green', 'orange', 'red']

    bars = ax.bar(names, powers, color=colors, alpha=0.7)
    ax.set_ylabel('Module Power (W)', fontsize=12)
    ax.set_title('Module Power by Architecture Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{power:.1f}W', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/tmp/ctm_architecture_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nArchitecture comparison saved to: /tmp/ctm_architecture_comparison.png")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("CTM LOSS MODELING ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("Fraunhofer ISE SmartCalc Methodology Implementation")
    print("="*80)

    # Run all demos
    demo_standard_module()
    demo_half_cut_module()
    demo_bifacial_module()
    demo_shingled_module()
    demo_sensitivity_analysis()
    demo_environmental_effects()
    demo_module_type_comparison()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - /tmp/ctm_standard_waterfall.png")
    print("  - /tmp/ctm_sensitivity.png")
    print("  - /tmp/ctm_environmental.png")
    print("  - /tmp/ctm_architecture_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
