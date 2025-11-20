"""
Module Builder Demonstration

This script demonstrates all features of the Module Configuration Builder:
1. Creating module configurations
2. Calculating specifications
3. Generating PVsyst PAN files
4. Validating designs
5. Optimizing layouts
6. Exporting to JSON and CSV
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules import (
    CellType,
    LayoutType,
    CellDesign,
    ModuleConfigBuilder,
    create_standard_module,
)


def demo_basic_module_creation():
    """Demonstrate basic module creation"""
    print("=" * 80)
    print("1. BASIC MODULE CREATION")
    print("=" * 80)

    # Create a cell design
    cell = CellDesign(
        cell_type=CellType.MONO_PERC,
        efficiency=0.225,
        area=0.0244,  # 156mm x 156mm M6 cell
        voltage_oc=0.68,
        current_sc=10.3,
        voltage_mpp=0.58,
        current_mpp=9.8,
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        temp_coeff_pmax=-0.35,
        series_resistance=0.005,
        shunt_resistance=500,
        ideality_factor=1.2,
        busbar_count=9
    )

    # Create module configuration
    builder = ModuleConfigBuilder()
    layout = {
        'layout_type': LayoutType.HALF_CUT,
        'cells_series': 120,
        'cells_parallel': 2,
        'submodules': 2,
        'bypass_diodes': 3
    }

    module = builder.create_module_config(
        cell_design=cell,
        layout=layout,
        name="Demo 450W Half-Cut",
        manufacturer="Demo Solar Inc."
    )

    print(f"\nModule Created: {module.name}")
    print(f"Manufacturer: {module.manufacturer}")
    print(f"Total Cells: {module.layout.total_cells}")
    print(f"Layout Type: {module.layout.layout_type.value}")
    print(f"Dimensions: {module.length:.0f} x {module.width:.0f} mm")
    print(f"Weight: {module.weight:.1f} kg")
    print(f"Area: {module.area:.3f} m²")

    return module


def demo_specifications_calculation(module):
    """Demonstrate specifications calculation"""
    print("\n" + "=" * 80)
    print("2. MODULE SPECIFICATIONS CALCULATION")
    print("=" * 80)

    builder = ModuleConfigBuilder()
    specs = builder.calculate_module_specs(module)

    print(f"\nElectrical Parameters (STC):")
    print(f"  Pmax:       {specs.pmax:.1f} W")
    print(f"  Voc:        {specs.voc:.2f} V")
    print(f"  Isc:        {specs.isc:.2f} A")
    print(f"  Vmpp:       {specs.vmpp:.2f} V")
    print(f"  Impp:       {specs.impp:.2f} A")
    print(f"  Efficiency: {specs.efficiency*100:.2f} %")
    print(f"  Fill Factor: {specs.fill_factor:.3f}")

    print(f"\nTemperature Coefficients:")
    print(f"  Pmax: {specs.temp_coeff_pmax:+.2f} %/°C")
    print(f"  Voc:  {specs.temp_coeff_voc:+.2f} %/°C")
    print(f"  Isc:  {specs.temp_coeff_isc:+.2f} %/°C")

    print(f"\nThermal:")
    print(f"  NOCT: {specs.noct:.1f} °C")

    print(f"\nCTM Losses:")
    print(f"  Resistance:  {specs.ctm_loss_resistance:.1f} %")
    print(f"  Reflection:  {specs.ctm_loss_reflection:.1f} %")
    print(f"  Mismatch:    {specs.ctm_loss_mismatch:.1f} %")
    print(f"  Inactive:    {specs.ctm_loss_inactive:.1f} %")
    print(f"  Total CTM:   {specs.ctm_total_loss:.1f} %")

    print(f"\nAdvanced Features:")
    print(f"  Low Irradiance Loss (200W/m²): {specs.low_irradiance_loss:.1f} %")
    print(f"  IAM Loss (50°): {specs.iam_loss_50deg:.1f} %")

    return specs


def demo_pan_file_generation(module):
    """Demonstrate PAN file generation"""
    print("\n" + "=" * 80)
    print("3. PVSYST PAN FILE GENERATION")
    print("=" * 80)

    builder = ModuleConfigBuilder()
    pan_content = builder.generate_pvsyst_pan_file(module)

    print(f"\nGenerated PAN file ({len(pan_content)} characters):")
    print("\n--- First 40 lines of PAN file ---")
    lines = pan_content.split('\n')
    for line in lines[:40]:
        print(line)
    print("...")

    # Save to file
    output_file = Path(__file__).parent / "demo_module.PAN"
    output_file.write_text(pan_content)
    print(f"\nFull PAN file saved to: {output_file}")

    return pan_content


def demo_validation(module):
    """Demonstrate module validation"""
    print("\n" + "=" * 80)
    print("4. MODULE DESIGN VALIDATION")
    print("=" * 80)

    builder = ModuleConfigBuilder()
    report = builder.validate_module_design(module)

    print(f"\nValidation Status: {'PASS' if report.is_valid else 'FAIL'}")
    print(f"Design: {report.design_name}")
    print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")

    print(f"\nValidation Issues ({len(report.issues)}):")
    for i, issue in enumerate(report.issues, 1):
        print(f"\n  {i}. [{issue.level.value.upper()}] {issue.category}")
        print(f"     {issue.message}")
        if issue.recommendation:
            print(f"     → {issue.recommendation}")

    return report


def demo_layout_optimization():
    """Demonstrate layout optimization"""
    print("\n" + "=" * 80)
    print("5. LAYOUT OPTIMIZATION")
    print("=" * 80)

    # Create a cell design
    cell = CellDesign(
        cell_type=CellType.MONO_TOPCON,
        efficiency=0.245,
        area=0.0244,
        voltage_oc=0.70,
        current_sc=10.5,
        voltage_mpp=0.60,
        current_mpp=10.0,
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        temp_coeff_pmax=-0.30,
        series_resistance=0.004,
        shunt_resistance=600,
        ideality_factor=1.15,
        busbar_count=12
    )

    builder = ModuleConfigBuilder()

    # Optimize for different objectives
    objectives = ['efficiency', 'cost', 'performance']

    for objective in objectives:
        print(f"\n--- Optimizing for: {objective.upper()} ---")

        constraints = {
            'target_power': 550,
            'max_voltage': 50,
            'max_current': 15,
            'optimize_for': objective,
            'allow_half_cut': True,
            'allow_shingled': True
        }

        optimal = builder.optimize_cell_layout(cell, constraints)

        print(f"  Layout Type: {optimal.layout.layout_type.value}")
        print(f"  Total Cells: {optimal.layout.total_cells}")
        print(f"  Configuration: {optimal.layout.cells_series}S x {optimal.layout.cells_parallel}P")
        print(f"  Bypass Diodes: {optimal.layout.bypass_diodes}")
        print(f"  Efficiency Gain: {optimal.efficiency_gain:+.2f}%")
        print(f"  Cost Delta: {optimal.cost_delta:+.2f}%")
        print(f"  Performance Score: {optimal.performance_score:.1f}/100")

        print(f"  Notes:")
        for note in optimal.optimization_notes:
            print(f"    • {note}")


def demo_export_formats():
    """Demonstrate export to different formats"""
    print("\n" + "=" * 80)
    print("6. EXPORT TO MULTIPLE FORMATS")
    print("=" * 80)

    builder = ModuleConfigBuilder()

    # Create multiple modules for comparison
    modules = [
        create_standard_module(450, CellType.MONO_PERC, LayoutType.HALF_CUT, "Manufacturer A"),
        create_standard_module(500, CellType.MONO_TOPCON, LayoutType.HALF_CUT, "Manufacturer B"),
        create_standard_module(550, CellType.MONO_HJT, LayoutType.BIFACIAL, "Manufacturer C"),
    ]

    # JSON export
    print("\n--- JSON Export ---")
    json_file = Path(__file__).parent / "module_export.json"
    json_str = builder.export_to_json(modules[0], include_specs=True, filepath=json_file)
    print(f"Exported to: {json_file}")
    print(f"Size: {len(json_str)} characters")

    # CSV export
    print("\n--- CSV Export ---")
    csv_file = Path(__file__).parent / "modules_comparison.csv"
    csv_str = builder.export_to_csv(modules, filepath=csv_file)
    print(f"Exported to: {csv_file}")
    print("\nCSV Preview:")
    print(csv_str[:500] + "...")


def demo_standard_modules():
    """Demonstrate convenience function for standard modules"""
    print("\n" + "=" * 80)
    print("7. STANDARD MODULE CREATION (CONVENIENCE FUNCTION)")
    print("=" * 80)

    power_classes = [450, 500, 550]
    cell_types = [CellType.MONO_PERC, CellType.MONO_TOPCON, CellType.MONO_HJT]
    builder = ModuleConfigBuilder()

    print("\nComparison of Standard Modules:")
    print(f"\n{'Power':<8} {'Cell Type':<15} {'Pmax (W)':<10} {'Eff (%)':<10} {'Voc (V)':<10} {'Cells':<8}")
    print("-" * 75)

    for power, cell_type in zip(power_classes, cell_types):
        module = create_standard_module(power, cell_type)
        specs = builder.calculate_module_specs(module)

        print(f"{power:<8} {cell_type.value:<15} {specs.pmax:<10.1f} "
              f"{specs.efficiency*100:<10.2f} {specs.voc:<10.2f} {module.layout.total_cells:<8}")


def demo_bifacial_module():
    """Demonstrate bifacial module configuration"""
    print("\n" + "=" * 80)
    print("8. BIFACIAL MODULE CONFIGURATION")
    print("=" * 80)

    # Create bifacial cell design
    cell = CellDesign(
        cell_type=CellType.MONO_PERC,
        efficiency=0.235,
        area=0.0244,
        voltage_oc=0.69,
        current_sc=10.4,
        voltage_mpp=0.59,
        current_mpp=9.9,
        temp_coeff_voc=-0.28,
        temp_coeff_isc=0.05,
        temp_coeff_pmax=-0.34,
        series_resistance=0.005,
        shunt_resistance=550,
        ideality_factor=1.2,
        is_bifacial=True,
        bifacial_factor=0.75,  # 75% rear-to-front ratio
        busbar_count=9
    )

    builder = ModuleConfigBuilder()
    layout = {
        'layout_type': LayoutType.BIFACIAL,
        'cells_series': 132,
        'cells_parallel': 1,
        'submodules': 1,
        'bypass_diodes': 3
    }

    module = builder.create_module_config(
        cell_design=cell,
        layout=layout,
        name="Demo 500W Bifacial",
        manufacturer="Demo Solar Inc.",
        glass_thickness_rear=2.0  # Required for bifacial
    )

    specs = builder.calculate_module_specs(module)

    print(f"\nBifacial Module: {module.name}")
    print(f"  Front Power: {specs.pmax:.1f} W")
    print(f"  Bifacial Factor: {specs.bifacial_factor:.2f}")
    print(f"  Estimated Rear Power: {specs.pmax * specs.bifacial_factor:.1f} W")
    print(f"  Total Potential: {specs.pmax * (1 + specs.bifacial_factor * 0.7):.1f} W (assuming 70% ground albedo)")
    print(f"  Glass: {module.glass_thickness_front:.1f}mm front, {module.glass_thickness_rear:.1f}mm rear")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("MODULE CONFIGURATION BUILDER - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)

    # Run all demos
    module = demo_basic_module_creation()
    specs = demo_specifications_calculation(module)
    pan_content = demo_pan_file_generation(module)
    report = demo_validation(module)
    demo_layout_optimization()
    demo_export_formats()
    demo_standard_modules()
    demo_bifacial_module()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll features demonstrated successfully!")
    print("Check the 'examples' directory for generated files.")


if __name__ == "__main__":
    main()
