"""
Example usage of IEC 61215 Test Simulator.

This script demonstrates how to:
1. Configure a PV module
2. Run complete IEC 61215 test sequences
3. Generate qualification reports
4. Visualize results
5. Export reports to PDF and Excel
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModuleConfig, CellTechnology, ModuleType
from src.modules.iec_61215_simulator import IEC61215TestSimulator


def create_example_modules() -> dict:
    """Create example module configurations for testing."""

    # High-efficiency mono-Si PERC module
    mono_perc = ModuleConfig(
        name="MonoPERC-400W-Premium",
        technology=CellTechnology.PERC,
        module_type=ModuleType.STANDARD,
        rated_power=400.0,
        voc=49.5,
        isc=10.5,
        vmp=41.2,
        imp=9.71,
        efficiency=20.5,
        area=1.95,
        cells_in_series=72,
        cells_in_parallel=1,
        dimensions=[1980, 990, 40],
        weight=22.5,
        glass_thickness_front=3.2,
        glass_thickness_back=0.0,
        encapsulant_type="EVA",
        backsheet_type="Tedlar",
        frame_material="Aluminum",
        junction_box="IP68 Rated",
        bypass_diodes=3,
        temperature_coeff_pmax=-0.37,
        temperature_coeff_voc=-0.28,
        temperature_coeff_isc=0.048,
        noct=44.0,
        max_system_voltage=1500.0,
        series_fuse_rating=20.0,
    )

    # Bifacial glass-glass HJT module
    bifacial_hjt = ModuleConfig(
        name="HJT-450W-Bifacial",
        technology=CellTechnology.HJT,
        module_type=ModuleType.BIFACIAL,
        rated_power=450.0,
        voc=48.8,
        isc=11.8,
        vmp=40.5,
        imp=11.11,
        efficiency=22.0,
        area=2.05,
        cells_in_series=66,
        cells_in_parallel=1,
        dimensions=[2094, 1038, 35],
        weight=26.0,
        glass_thickness_front=2.5,
        glass_thickness_back=2.0,
        encapsulant_type="POE",
        backsheet_type=None,
        frame_material="Aluminum",
        junction_box="IP68 Smart",
        bypass_diodes=3,
        temperature_coeff_pmax=-0.26,
        temperature_coeff_voc=-0.24,
        temperature_coeff_isc=0.045,
        noct=41.0,
        max_system_voltage=1500.0,
        series_fuse_rating=25.0,
    )

    # Standard poly-Si module
    poly_standard = ModuleConfig(
        name="PolySi-330W-Standard",
        technology=CellTechnology.POLY_SI,
        module_type=ModuleType.STANDARD,
        rated_power=330.0,
        voc=46.2,
        isc=9.2,
        vmp=37.8,
        imp=8.73,
        efficiency=17.0,
        area=1.94,
        cells_in_series=60,
        cells_in_parallel=1,
        dimensions=[1956, 992, 40],
        weight=22.0,
        glass_thickness_front=3.2,
        glass_thickness_back=0.0,
        encapsulant_type="EVA",
        backsheet_type="Standard",
        frame_material="Aluminum",
        junction_box="Standard",
        bypass_diodes=3,
        temperature_coeff_pmax=-0.42,
        temperature_coeff_voc=-0.31,
        temperature_coeff_isc=0.053,
        noct=46.0,
        max_system_voltage=1000.0,
        series_fuse_rating=15.0,
    )

    return {
        "mono_perc": mono_perc,
        "bifacial_hjt": bifacial_hjt,
        "poly_standard": poly_standard,
    }


def run_complete_iec_61215_sequence(module: ModuleConfig, output_dir: Path) -> None:
    """
    Run complete IEC 61215 test sequence for a module.

    Args:
        module: Module configuration to test
        output_dir: Directory for output files
    """
    print(f"\n{'='*80}")
    print(f"IEC 61215 QUALIFICATION TEST SEQUENCE")
    print(f"Module: {module.name}")
    print(f"{'='*80}\n")

    # Create simulator instance
    simulator = IEC61215TestSimulator(
        module=module,
        random_seed=42,  # For reproducible results
        strictness_factor=1.0,  # Standard strictness
    )

    # Create output directory
    module_output_dir = output_dir / module.name.replace(" ", "_")
    module_output_dir.mkdir(parents=True, exist_ok=True)

    # List to store all test results
    all_tests = []

    # MQT-10: Thermal Cycling Test (200 cycles)
    print("Running MQT-10: Thermal Cycling Test...")
    thermal_cycling = simulator.simulate_thermal_cycling(module, cycles=200)
    all_tests.append(thermal_cycling)
    print(f"  Status: {thermal_cycling.status.value}")
    print(f"  Power Degradation: {thermal_cycling.power_degradation:.2f}%")
    print(f"  Visual Defects: {len(thermal_cycling.visual_defects)}")
    print()

    # MQT-11: Humidity Freeze Test (10 cycles)
    print("Running MQT-11: Humidity Freeze Test...")
    humidity_freeze = simulator.simulate_humidity_freeze(module, cycles=10)
    all_tests.append(humidity_freeze)
    print(f"  Status: {humidity_freeze.status.value}")
    print(f"  Power Degradation: {humidity_freeze.power_degradation:.2f}%")
    print(f"  Insulation Resistance: {humidity_freeze.insulation_resistance:.1f} MΩ·m²")
    print()

    # MQT-12: Damp Heat Test (1000 hours)
    print("Running MQT-12: Damp Heat Test...")
    damp_heat = simulator.simulate_damp_heat(module, hours=1000)
    all_tests.append(damp_heat)
    print(f"  Status: {damp_heat.status.value}")
    print(f"  Power Degradation: {damp_heat.power_degradation:.2f}%")
    print(f"  Wet Leakage Current: {damp_heat.wet_leakage_current:.2f} mA")
    print()

    # MQT-13: UV Preconditioning Test
    print("Running MQT-13: UV Preconditioning Test...")
    uv_test = simulator.simulate_uv_preconditioning(module, hours=48, dose=15.0)
    all_tests.append(uv_test)
    print(f"  Status: {uv_test.status.value}")
    print(f"  Power Degradation: {uv_test.power_degradation:.2f}%")
    print()

    # MQT-17: Hail Impact Test (standard conditions)
    print("Running MQT-17: Hail Impact Test...")
    hail_test = simulator.simulate_hail_impact(module, diameter=25.0, velocity=23.0)
    all_tests.append(hail_test)
    print(f"  Status: {hail_test.status.value}")
    print(f"  Power Degradation: {hail_test.power_degradation:.2f}%")
    print(f"  Impact Energy: {hail_test.test_parameters['impact_energy_J']:.2f} J")
    print()

    # MQT-18: Mechanical Load Test (±2400 Pa)
    print("Running MQT-18: Mechanical Load Test...")
    mechanical_test = simulator.simulate_mechanical_load(module, front_load=2400, back_load=2400)
    all_tests.append(mechanical_test)
    print(f"  Status: {mechanical_test.status.value}")
    print(f"  Power Degradation: {mechanical_test.power_degradation:.2f}%")
    print(f"  Deflection: {mechanical_test.test_parameters['deflection_m']*1000:.2f} mm")
    print()

    # Generate qualification report
    print("Generating Qualification Report...")
    report = simulator.generate_qualification_report(all_tests)
    print(f"\n{'='*80}")
    print("QUALIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Total Power Degradation: {report.total_power_degradation:.2f}%")
    print(f"Final Power Retention: {(1 - report.total_power_degradation/100)*100:.2f}%")
    print(f"\nCompliance Checks:")
    print(f"  Power Retention ≥95%: {'PASS' if report.power_retention_check else 'FAIL'}")
    print(f"  Visual Inspection: {'PASS' if report.visual_inspection_check else 'FAIL'}")
    print(f"  Insulation Resistance: {'PASS' if report.insulation_resistance_check else 'FAIL'}")
    print(f"  Safety (Leakage): {'PASS' if report.safety_check else 'FAIL'}")

    if report.critical_failures:
        print(f"\nCritical Failures:")
        for failure in report.critical_failures:
            print(f"  - {failure}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Power degradation timeline
    print("Creating power degradation timeline...")
    timeline_path = module_output_dir / "power_degradation_timeline.png"
    simulator.plot_power_degradation_timeline(all_tests, save_path=timeline_path)
    print(f"  Saved: {timeline_path}")

    # I-V curve comparisons for each test
    for test in all_tests:
        if test.iv_curve_before and test.iv_curve_after:
            print(f"Creating I-V curve comparison for {test.test_id}...")
            iv_path = module_output_dir / f"iv_comparison_{test.test_id}.png"
            simulator.plot_iv_curve_comparison(test, save_path=iv_path)
            print(f"  Saved: {iv_path}")

    # Export reports
    print(f"\n{'='*80}")
    print("EXPORTING REPORTS")
    print(f"{'='*80}\n")

    # Excel report
    excel_path = module_output_dir / "qualification_report.xlsx"
    print(f"Exporting to Excel...")
    simulator.export_report_to_excel(report, excel_path)
    print(f"  Saved: {excel_path}")

    # PDF report
    pdf_path = module_output_dir / "qualification_report.pdf"
    print(f"Exporting to PDF...")
    simulator.export_report_to_pdf(report, pdf_path)
    print(f"  Saved: {pdf_path}")

    print(f"\n{'='*80}")
    print(f"TEST SEQUENCE COMPLETE - {module.name}")
    print(f"All outputs saved to: {module_output_dir}")
    print(f"{'='*80}\n")


def run_comparative_analysis(modules: dict, output_dir: Path) -> None:
    """
    Run comparative analysis of multiple module designs.

    Args:
        modules: Dictionary of module configurations
        output_dir: Directory for output files
    """
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS - MULTIPLE MODULE DESIGNS")
    print(f"{'='*80}\n")

    results_summary = []

    for name, module in modules.items():
        print(f"Testing {name}...")
        simulator = IEC61215TestSimulator(module, random_seed=42)

        # Run key tests
        thermal = simulator.simulate_thermal_cycling(module, cycles=200)
        damp_heat = simulator.simulate_damp_heat(module, hours=1000)
        hail = simulator.simulate_hail_impact(module)

        all_tests = [thermal, damp_heat, hail]
        report = simulator.generate_qualification_report(all_tests)

        results_summary.append({
            'Module': module.name,
            'Technology': module.technology.value,
            'Type': module.module_type.value,
            'Rated Power (W)': module.rated_power,
            'Total Degradation (%)': f"{report.total_power_degradation:.2f}",
            'Status': report.overall_status.value,
            'Power Retention': report.power_retention_check,
            'Visual Check': report.visual_inspection_check,
            'Insulation Check': report.insulation_resistance_check,
        })

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    import pandas as pd
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = output_dir / "comparative_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparative analysis saved to: {csv_path}")


def demonstrate_custom_test_scenarios() -> None:
    """Demonstrate custom test scenarios and edge cases."""
    print(f"\n{'='*80}")
    print("CUSTOM TEST SCENARIOS")
    print(f"{'='*80}\n")

    # Create a test module
    test_module = ModuleConfig(
        name="TestModule-350W",
        technology=CellTechnology.MONO_SI,
        module_type=ModuleType.STANDARD,
        rated_power=350.0,
        voc=46.0,
        isc=9.8,
        vmp=38.5,
        imp=9.09,
        efficiency=18.0,
        area=1.94,
        cells_in_series=60,
        dimensions=[1960, 990, 40],
        weight=21.0,
    )

    simulator = IEC61215TestSimulator(test_module, random_seed=42)

    # Scenario 1: Extreme thermal cycling (400 cycles)
    print("Scenario 1: Extreme Thermal Cycling (400 cycles)")
    extreme_thermal = simulator.simulate_thermal_cycling(test_module, cycles=400)
    print(f"  Power Degradation: {extreme_thermal.power_degradation:.2f}%")
    print(f"  Status: {extreme_thermal.status.value}\n")

    # Scenario 2: Extended damp heat (2000 hours)
    print("Scenario 2: Extended Damp Heat (2000 hours)")
    extended_dh = simulator.simulate_damp_heat(test_module, hours=2000)
    print(f"  Power Degradation: {extended_dh.power_degradation:.2f}%")
    print(f"  Status: {extended_dh.status.value}\n")

    # Scenario 3: Severe hail (35mm @ 30 m/s)
    print("Scenario 3: Severe Hail Impact (35mm @ 30 m/s)")
    severe_hail = simulator.simulate_hail_impact(test_module, diameter=35.0, velocity=30.0)
    print(f"  Impact Energy: {severe_hail.test_parameters['impact_energy_J']:.2f} J")
    print(f"  Power Degradation: {severe_hail.power_degradation:.2f}%")
    print(f"  Status: {severe_hail.status.value}\n")

    # Scenario 4: High mechanical load (4800 Pa)
    print("Scenario 4: High Mechanical Load (4800 Pa)")
    high_load = simulator.simulate_mechanical_load(test_module, front_load=4800, back_load=4800)
    print(f"  Deflection: {high_load.test_parameters['deflection_m']*1000:.2f} mm")
    print(f"  Power Degradation: {high_load.power_degradation:.2f}%")
    print(f"  Status: {high_load.status.value}\n")


def main() -> None:
    """Main execution function."""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Create example modules
    modules = create_example_modules()

    # Run complete test sequence for each module
    for module in modules.values():
        run_complete_iec_61215_sequence(module, output_dir)

    # Run comparative analysis
    run_comparative_analysis(modules, output_dir)

    # Demonstrate custom scenarios
    demonstrate_custom_test_scenarios()

    print(f"\n{'='*80}")
    print("ALL SIMULATIONS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
