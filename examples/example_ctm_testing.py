"""
Example: IEC 63202 CTM Testing and Validation

This example demonstrates how to:
1. Configure an IEC 63202 CTM test
2. Input cell and module measurements
3. Calculate CTM power ratio
4. Analyze loss components
5. Generate compliance reports
6. Use the B03 loss model for predictions
"""

from datetime import datetime
import numpy as np

from pv_circularity_simulator.core.iec63202 import (
    CTMTestConfig,
    CellProperties,
    ModuleConfiguration,
    ReferenceDeviceData,
    FlashSimulatorData,
    IVCurveData,
    CellTechnology,
    FlashSimulatorType,
    IEC63202CTMTester,
    CTMPowerLossAnalyzer,
    ReferenceDeviceCalibration,
    CTMTestReport,
)
from pv_circularity_simulator.core.ctm.b03_ctm_loss_model import (
    B03CTMLossModel,
    B03CTMConfiguration,
)


def example_basic_ctm_test():
    """
    Example 1: Basic CTM testing workflow.

    This example shows the complete workflow for performing an IEC 63202
    CTM test with minimal configuration.
    """
    print("=" * 80)
    print("Example 1: Basic IEC 63202 CTM Test")
    print("=" * 80)

    # Step 1: Define cell properties
    cell_props = CellProperties(
        technology=CellTechnology.PERC,
        area=244.3,  # cm²
        efficiency=22.8,  # %
        voc=0.682,  # V
        isc=8.52,  # A
        vmp=0.580,  # V
        imp=8.24,  # A
        pmax=5.22,  # W
        temperature_coefficient_pmax=-0.39,  # %/°C
        temperature_coefficient_voc=-0.0029,  # V/°C
        temperature_coefficient_isc=0.0005,  # A/°C
    )

    # Step 2: Define module configuration
    module_config = ModuleConfiguration(
        num_cells_series=60,
        num_strings_parallel=1,
        bypass_diodes=3,
    )

    # Step 3: Create reference device
    ref_device = ReferenceDeviceData(
        device_id="REF-001",
        calibration_date=datetime(2025, 1, 15),
        calibration_lab="PV Test Lab",
        calibration_certificate="CAL-2025-001",
        short_circuit_current=8.52,
        responsivity=0.00852,
        temperature_coefficient=0.0005,
        uncertainty_isc=1.5,
        uncertainty_temperature=0.2,
        next_calibration_due=datetime(2026, 1, 15),
    )

    # Step 4: Create flash simulator specification
    flash_sim = FlashSimulatorData(
        simulator_type=FlashSimulatorType.LED,
        spatial_uniformity=98.5,
        temporal_stability=99.2,
    )

    # Step 5: Create test configuration
    test_config = CTMTestConfig(
        test_id="CTM-2025-001",
        laboratory="PV Testing Laboratory",
        operator="John Doe",
        cell_properties=cell_props,
        module_config=module_config,
        reference_device=ref_device,
        flash_simulator=flash_sim,
        acceptance_criteria_min=95.0,
        acceptance_criteria_max=102.0,
    )

    # Step 6: Generate sample IV curves (in practice, these come from measurements)
    cell_measurements = []
    for i in range(5):
        voltage = np.linspace(0, 0.682, 30)
        current = 8.52 * (1 - (voltage / 0.682) ** 3) * (1 + np.random.normal(0, 0.01))
        cell_measurements.append(IVCurveData(
            voltage=voltage.tolist(),
            current=current.tolist(),
            temperature=25.0,
            irradiance=1000.0,
        ))

    module_measurements = []
    for i in range(3):
        voltage = np.linspace(0, 40.92, 50)
        current = 8.52 * (1 - (voltage / 40.92) ** 3) * (1 + np.random.normal(0, 0.005))
        module_measurements.append(IVCurveData(
            voltage=voltage.tolist(),
            current=current.tolist(),
            temperature=25.0,
            irradiance=1000.0,
        ))

    # Step 7: Create tester and run CTM test
    loss_analyzer = CTMPowerLossAnalyzer()
    tester = IEC63202CTMTester(
        config=test_config,
        power_loss_analyzer=loss_analyzer,
    )

    result = tester.ctm_power_ratio_test(
        cell_measurements=cell_measurements,
        module_measurements=module_measurements,
    )

    # Step 8: Display results
    print(f"\nCTM Test Results:")
    print(f"  Test ID: {result.config.test_id}")
    print(f"  Cell Power (avg): {result.cell_power_avg:.3f} W")
    print(f"  Module Power (avg): {result.module_power_avg:.2f} W")
    print(f"  Expected Module Power: {result.expected_module_power:.2f} W")
    print(f"  CTM Ratio: {result.ctm_ratio:.2f}%")
    print(f"  Uncertainty: ±{result.ctm_ratio_uncertainty:.2f}%")
    print(f"  Total Loss: {result.loss_components.total_loss:.2f}%")
    print(f"  Compliance: {'PASS ✓' if result.compliance_status else 'FAIL ✗'}")

    print(f"\nLoss Breakdown:")
    print(f"  Optical Losses: {result.loss_components.total_optical_loss:.2f}%")
    print(f"    - Reflection: {result.loss_components.optical_reflection:.2f}%")
    print(f"    - Absorption: {result.loss_components.optical_absorption:.2f}%")
    print(f"    - Shading: {result.loss_components.optical_shading:.2f}%")
    print(f"  Electrical Losses: {result.loss_components.total_electrical_loss:.2f}%")
    print(f"    - Series R: {result.loss_components.electrical_series_resistance:.2f}%")
    print(f"    - Mismatch: {result.loss_components.electrical_mismatch:.2f}%")

    # Step 9: Generate certificate
    certificate = tester.generate_ctm_certificate(
        certified_by="IEC 63202 Testing Authority",
        validity_months=12
    )

    print(f"\nCertificate Generated:")
    print(f"  Certificate Number: {certificate.certificate_number}")
    print(f"  Issue Date: {certificate.issue_date.strftime('%Y-%m-%d')}")
    print(f"  Valid Until: {certificate.expiry_date.strftime('%Y-%m-%d')}")
    print(f"  Status: {'VALID ✓' if certificate.is_valid else 'EXPIRED ✗'}")

    return result


def example_b03_loss_model():
    """
    Example 2: B03 CTM Loss Model Analysis.

    This example demonstrates using the B03 model to predict CTM losses
    based on manufacturing quality parameters.
    """
    print("\n" + "=" * 80)
    print("Example 2: B03 CTM Loss Model Analysis")
    print("=" * 80)

    # Create B03 model
    model = B03CTMLossModel()

    # Compare different quality scenarios
    print("\nComparing Quality Scenarios:")
    scenarios = ["premium_quality", "standard_quality", "economy_quality"]

    for scenario in scenarios:
        config = B03CTMConfiguration.from_scenario(scenario)
        result = model.calculate_ctm_losses(config)

        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  CTM Ratio: {result.total_ctm_ratio_percent:.2f}%")
        print(f"  Total Loss: {result.total_loss_percent:.2f}%")

        breakdown = result.get_loss_breakdown()
        print(f"  Loss by Category:")
        for category, loss in breakdown.items():
            print(f"    - {category.replace('_', ' ').title()}: {loss:.3f}%")

    # Sensitivity analysis
    print("\n" + "-" * 80)
    print("Sensitivity Analysis: Interconnect Shading (k10)")
    print("-" * 80)

    base_config = B03CTMConfiguration.from_scenario("standard_quality")
    sensitivity = model.sensitivity_analysis(
        base_configuration=base_config,
        factor_to_vary="k10_interconnect_shading",
    )

    for quality_level, ctm_ratio in sensitivity.items():
        print(f"  {quality_level:20s}: CTM Ratio = {ctm_ratio:.2f}%")


def example_advanced_loss_analysis():
    """
    Example 3: Advanced Loss Analysis.

    This example shows detailed loss component analysis using
    the CTMPowerLossAnalyzer.
    """
    print("\n" + "=" * 80)
    print("Example 3: Advanced CTM Loss Analysis")
    print("=" * 80)

    # Create analyzer
    analyzer = CTMPowerLossAnalyzer()

    # Analyze optical losses
    print("\nOptical Losses Analysis:")
    optical = analyzer.optical_losses(
        glass_transmission=0.96,
        encapsulant_absorption=0.015,
        grid_coverage_ratio=0.025,
        num_busbars=5,
    )

    for component, value in optical.items():
        print(f"  {component:20s}: {value:.3f}%")

    # Analyze electrical losses
    print("\nElectrical Losses Analysis:")
    cell_props = CellProperties(
        technology=CellTechnology.PERC,
        area=244.3,
        efficiency=22.8,
        voc=0.682,
        isc=8.52,
        vmp=0.580,
        imp=8.24,
        pmax=5.22,
        temperature_coefficient_pmax=-0.39,
        temperature_coefficient_voc=-0.0029,
        temperature_coefficient_isc=0.0005,
    )

    module_config = ModuleConfiguration(
        num_cells_series=60,
        num_strings_parallel=1,
    )

    electrical = analyzer.electrical_losses(
        cell_properties=cell_props,
        module_config=module_config,
        ribbon_resistivity=1.8e-8,  # Copper
    )

    for component, value in electrical.items():
        print(f"  {component:25s}: {value:.3f}%")

    # Spectral mismatch analysis
    print("\nSpectral Mismatch Analysis:")
    simulator_spectrum = {
        400: 0.95,
        500: 1.75,
        600: 1.65,
        700: 1.35,
        800: 1.05,
        900: 0.75,
    }

    spectral_factor = analyzer.spectral_mismatch_factor(simulator_spectrum)
    spectral_loss = (1 - spectral_factor) * 100

    print(f"  Spectral Mismatch Factor: {spectral_factor:.4f}")
    print(f"  Spectral Loss: {spectral_loss:.3f}%")


def example_report_generation():
    """
    Example 4: Report Generation and Export.

    This example demonstrates generating and exporting CTM test reports
    in various formats.
    """
    print("\n" + "=" * 80)
    print("Example 4: Report Generation and Export")
    print("=" * 80)

    # First, run a CTM test (reusing example 1)
    # For brevity, we'll just show the report generation part

    print("\nReport generation capabilities:")
    print("  - HTML Report: Interactive with Plotly charts")
    print("  - Excel Report: Multi-sheet workbook with data tables")
    print("  - PDF Report: Professional certificate-style document")
    print("  - JSON Export: Machine-readable test results")

    print("\nVisualizations available:")
    print("  - IV Curve Comparison (Cell vs. Module)")
    print("  - Loss Waterfall Chart")
    print("  - Loss Breakdown Pie Chart")
    print("  - Compliance Dashboard with Gauges")

    # In practice, you would do:
    # report = CTMTestReport(test_result)
    # report.export_to_excel("ctm_report.xlsx")
    # report.export_to_pdf("ctm_report.pdf")
    # html = report.generate_html_report()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("IEC 63202 CTM Testing Examples")
    print("PV Circularity Simulator")
    print("=" * 80)

    # Run examples
    result = example_basic_ctm_test()
    example_b03_loss_model()
    example_advanced_loss_analysis()
    example_report_generation()

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
