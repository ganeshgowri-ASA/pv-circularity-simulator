"""Basic example of PV system validation.

This example demonstrates how to use the System Validation module
to validate a simple PV system design.
"""

from datetime import datetime

from src.b05_system_validation import SystemValidator
from src.b05_system_validation.documentation_generator import DocumentationGenerator
from src.models.validation_models import (
    PerformanceMetrics,
    SystemConfiguration,
    SystemType,
)


def main() -> None:
    """Run basic validation example."""
    print("=" * 80)
    print("PV System Validation - Basic Example")
    print("=" * 80)

    # Define system configuration
    config = SystemConfiguration(
        system_type=SystemType.COMMERCIAL,
        system_name="Downtown Commercial Solar Array",
        location="San Francisco, CA",
        capacity_kw=100.0,
        module_count=250,
        inverter_count=2,
        string_count=20,
        modules_per_string=12,
        system_voltage_vdc=600.0,
        max_voltage_voc=800.0,
        operating_voltage_vmp=650.0,
        max_current_isc=10.0,
        operating_current_imp=9.5,
        ambient_temp_min=-10.0,
        ambient_temp_max=45.0,
        wind_speed_max=40.0,
        snow_load=50.0,
        jurisdiction="San Francisco",
        applicable_codes=["NEC 2020", "IEC 60364", "IBC 2021", "IFC 2021"],
    )

    # Define performance metrics (optional)
    performance_metrics = PerformanceMetrics(
        annual_energy_yield_kwh=150000.0,
        specific_yield_kwh_kwp=1500.0,
        performance_ratio=0.82,
        capacity_factor=0.20,
        loss_temperature=5.0,
        loss_soiling=2.0,
        loss_shading=1.0,
        loss_mismatch=2.0,
        loss_wiring=2.0,
        loss_inverter=3.0,
        loss_degradation=0.5,
        total_losses=15.5,
        is_energy_yield_realistic=True,
        is_pr_in_range=True,
        is_loss_budget_valid=True,
    )

    # Create validator
    print("\n1. Creating system validator...")
    validator = SystemValidator(config, performance_metrics)

    # Run complete validation
    print("\n2. Running complete design validation...")
    report = validator.validate_complete_design()

    # Print results
    print("\n3. Validation Results:")
    print("-" * 80)
    print(f"Report ID: {report.report_id}")
    print(f"System: {report.system_config.system_name}")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Total Issues: {report.total_issues}")
    print(f"  - Critical: {report.critical_issues}")
    print(f"  - Errors: {report.errors}")
    print(f"  - Warnings: {report.warnings}")

    print(f"\nCompliance Checks: {len(report.code_compliance)}")
    passed = sum(1 for c in report.code_compliance if c.status.value == "passed")
    failed = sum(1 for c in report.code_compliance if c.status.value == "failed")
    print(f"  - Passed: {passed}")
    print(f"  - Failed: {failed}")

    if report.performance_metrics:
        print(f"\nPerformance Metrics:")
        print(f"  - Performance Ratio: {report.performance_metrics.performance_ratio:.2%}")
        print(f"  - Specific Yield: {report.performance_metrics.specific_yield_kwh_kwp:.0f} kWh/kWp")
        print(f"  - Total Losses: {report.performance_metrics.total_losses:.1f}%")

    print("\n4. Top Recommendations:")
    print("-" * 80)
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"{i}. {rec}")

    # Generate documentation
    print("\n5. Generating documentation package...")
    doc_gen = DocumentationGenerator(config, report, output_dir="./exports")
    package = doc_gen.generate_complete_package()

    print(f"\n✅ Complete package generated:")
    print(f"   Package ID: {package.package_id}")
    print(f"   Documents: {package.document_count}")
    print(f"   Size: {package.total_size_mb:.2f} MB")
    print(f"   Location: ./exports/{package.package_id}/")

    # Export JSON report
    print("\n6. Exporting validation report to JSON...")
    json_path = f"./exports/{package.package_id}/validation_report.json"
    validator.export_report_json(json_path)
    print(f"   Saved to: {json_path}")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
