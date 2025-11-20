"""Example usage of the ReuseAssessor for PV module reuse assessment.

This example demonstrates how to use the ReuseAssessor to perform comprehensive
reuse assessments on photovoltaic modules with different conditions and ages.
"""

from datetime import datetime, timedelta
from pv_circularity_simulator import ReuseAssessor
from pv_circularity_simulator.core.models import ModuleData, PerformanceMetrics
from pv_circularity_simulator.core.enums import DegradationType


def assess_new_module():
    """Assess a new, high-quality module."""
    print("=" * 80)
    print("EXAMPLE 1: New Module Assessment")
    print("=" * 80)

    # Create assessor instance
    assessor = ReuseAssessor(
        degradation_rate_per_year=0.005,  # 0.5% per year
        expected_lifetime_years=25.0,
        minimum_performance_threshold=0.70,
        base_module_price_per_watt=0.50,
    )

    # Define module data
    module = ModuleData(
        module_id="PV-NEW-2024-001",
        manufacturer="SolarTech Premium",
        model="ST-400-HE",
        nameplate_power_w=400.0,
        manufacture_date=datetime.now() - timedelta(days=180),
        age_years=0.5,
        visual_defects=[],
        degradation_types=[],
        location="Testing Facility",
    )

    # Define performance metrics
    performance = PerformanceMetrics(
        measured_power_w=395.0,  # 98.75% of nameplate
        open_circuit_voltage_v=48.5,
        short_circuit_current_a=10.2,
        max_power_voltage_v=40.0,
        max_power_current_a=9.88,
        fill_factor=0.80,
        efficiency_percent=19.8,
    )

    # Perform assessment
    result = assessor.assess_module(module, performance)

    # Display results
    print(f"\nModule ID: {result.module_id}")
    print(f"Reuse Potential: {result.reuse_potential.value.upper()}")
    print(f"Reusability Score: {result.reusability_score:.1f}/100")
    print(f"Condition: {result.condition_assessment.overall_condition.value}")
    print(f"Performance Level: {result.performance_level.value}")
    print(f"Remaining Lifetime: {result.remaining_lifetime_years:.1f} years")
    print(f"Market Value: ${result.market_value_usd:.2f}")
    print(f"Target Market: {result.market_segment.value}")
    print(f"\nRecommended Applications:")
    for app in result.recommended_applications:
        print(f"  - {app}")
    print(f"\nConfidence Level: {result.confidence_level:.2%}")


def assess_mid_life_module():
    """Assess a mid-life module with minor defects."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Mid-Life Module Assessment")
    print("=" * 80)

    assessor = ReuseAssessor()

    module = ModuleData(
        module_id="PV-MID-2014-042",
        manufacturer="SunPower Corp",
        model="SP-320-BLK",
        nameplate_power_w=320.0,
        manufacture_date=datetime.now() - timedelta(days=3650),
        age_years=10.0,
        visual_defects=["Minor discoloration"],
        degradation_types=[DegradationType.DISCOLORATION],
        location="Residential Rooftop - Phoenix, AZ",
    )

    performance = PerformanceMetrics(
        measured_power_w=275.0,  # 85.9% of nameplate
        open_circuit_voltage_v=46.2,
        short_circuit_current_a=8.8,
        max_power_voltage_v=38.0,
        max_power_current_a=7.24,
        fill_factor=0.76,
        efficiency_percent=17.2,
    )

    result = assessor.assess_module(module, performance)

    print(f"\nModule ID: {result.module_id}")
    print(f"Age: {module.age_years} years")
    print(f"Reuse Potential: {result.reuse_potential.value.upper()}")
    print(f"Reusability Score: {result.reusability_score:.1f}/100")
    print(f"Condition Score: {result.condition_assessment.condition_score:.1f}/100")
    print(f"Performance: {result.performance_level.value} ({performance.measured_power_w}W / {module.nameplate_power_w}W)")
    print(f"Remaining Lifetime: {result.remaining_lifetime_years:.1f} years")
    print(f"Market Value: ${result.market_value_usd:.2f}")
    print(f"\nRecommended Applications:")
    for app in result.recommended_applications[:3]:  # Show first 3
        print(f"  - {app}")


def assess_degraded_module():
    """Assess an old module with multiple defects."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Degraded Module Assessment")
    print("=" * 80)

    assessor = ReuseAssessor()

    module = ModuleData(
        module_id="PV-OLD-2004-158",
        manufacturer="First Solar",
        model="FS-75-CdTe",
        nameplate_power_w=250.0,
        manufacture_date=datetime.now() - timedelta(days=7300),
        age_years=20.0,
        visual_defects=[
            "Severe discoloration",
            "Frame corrosion",
            "Delamination edges",
        ],
        degradation_types=[
            DegradationType.DISCOLORATION,
            DegradationType.CORROSION,
            DegradationType.DELAMINATION,
        ],
        location="Desert Installation - Nevada",
        environmental_conditions="Harsh desert climate, high UV exposure",
    )

    performance = PerformanceMetrics(
        measured_power_w=140.0,  # 56% of nameplate
        open_circuit_voltage_v=40.5,
        short_circuit_current_a=6.2,
        max_power_voltage_v=32.8,
        max_power_current_a=4.27,
        fill_factor=0.68,
        efficiency_percent=12.5,
    )

    result = assessor.assess_module(module, performance)

    print(f"\nModule ID: {result.module_id}")
    print(f"Age: {module.age_years} years")
    print(f"Reuse Potential: {result.reuse_potential.value.upper()}")
    print(f"Reusability Score: {result.reusability_score:.1f}/100")
    print(f"Condition: {result.condition_assessment.overall_condition.value}")
    print(f"Performance: {result.performance_level.value}")
    print(f"Remaining Lifetime: {result.remaining_lifetime_years:.1f} years")
    print(f"Market Value: ${result.market_value_usd:.2f}")
    print(f"\nLimiting Factors:")
    for factor in result.limiting_factors:
        print(f"  - {factor}")
    print(f"\nRecommended Applications:")
    for app in result.recommended_applications:
        print(f"  - {app}")


def assess_failed_module():
    """Assess a failed module."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Failed Module Assessment")
    print("=" * 80)

    assessor = ReuseAssessor()

    module = ModuleData(
        module_id="PV-FAIL-2010-999",
        manufacturer="GenericSolar",
        model="GS-180",
        nameplate_power_w=180.0,
        age_years=14.0,
        visual_defects=[
            "Hot spots visible",
            "Multiple cell cracks",
            "Burned junction box",
        ],
        degradation_types=[
            DegradationType.HOT_SPOT,
            DegradationType.CELL_CRACK,
            DegradationType.JUNCTION_BOX,
            DegradationType.BYPASS_DIODE,
        ],
    )

    performance = PerformanceMetrics(
        measured_power_w=65.0,  # 36% of nameplate
        open_circuit_voltage_v=35.2,
        short_circuit_current_a=4.5,
        max_power_voltage_v=27.8,
        max_power_current_a=2.34,
        fill_factor=0.58,
        efficiency_percent=8.2,
    )

    result = assessor.assess_module(
        module, performance, visual_inspection_pass=False, electrical_safety_pass=False
    )

    print(f"\nModule ID: {result.module_id}")
    print(f"Reuse Potential: {result.reuse_potential.value.upper()}")
    print(f"Reusability Score: {result.reusability_score:.1f}/100")
    print(f"Safety Status: {'PASS' if result.condition_assessment.electrical_safety_pass else 'FAIL'}")
    print(f"Condition: {result.condition_assessment.overall_condition.value}")
    print(f"Performance: {result.performance_level.value}")
    print(f"Market Value: ${result.market_value_usd:.2f}")
    print(f"\nLimiting Factors:")
    for factor in result.limiting_factors:
        print(f"  - {factor}")
    print(f"\nRecommended Path:")
    for app in result.recommended_applications:
        print(f"  - {app}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PV MODULE REUSE ASSESSMENT EXAMPLES" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    assess_new_module()
    assess_mid_life_module()
    assess_degraded_module()
    assess_failed_module()

    print("\n" + "=" * 80)
    print("Assessment Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
