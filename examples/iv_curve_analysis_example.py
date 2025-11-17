"""
Example: IV Curve Analysis and Electrical Diagnostics for PV Modules

This example demonstrates how to use the IV curve analysis capabilities
to extract electrical parameters, detect degradation, identify faults,
and perform baseline comparisons.
"""

from datetime import datetime

import numpy as np

from pv_circularity_simulator.core.models import ElectricalParameters, IVCurveData
from pv_circularity_simulator.diagnostics.iv_curve import (
    CurveComparison,
    ElectricalDiagnostics,
    IVCurveAnalyzer,
)


def generate_iv_curve(
    isc=9.0,
    voc=38.0,
    fill_factor=0.78,
    temperature=25.0,
    irradiance=1000.0,
    num_points=100,
    add_noise=True,
    degradation_factor=1.0,
):
    """
    Generate synthetic IV curve data.

    Uses a simplified single-diode model for realistic IV curve generation.
    """
    # Generate voltage points
    v = np.linspace(0, voc, num_points)

    # Single diode model approximation
    # I = Isc * (1 - C1*(exp(V/(C2*Voc)) - 1))
    c1 = (1 - (fill_factor * voc * isc) / (voc * isc))
    c2 = (voc / np.log(1 / c1 + 1))

    i = isc * (1 - c1 * (np.exp(v / c2) - 1))

    # Apply degradation
    i = i * degradation_factor

    # Ensure no negative currents
    i = np.maximum(i, 0)

    # Add realistic noise
    if add_noise:
        noise = np.random.normal(0, isc * 0.01, num_points)  # 1% noise
        i = i + noise
        i = np.maximum(i, 0)

    # Create IV curve data
    iv_data = IVCurveData(
        voltage=v,
        current=i,
        temperature=temperature,
        irradiance=irradiance,
        timestamp=datetime.now(),
        module_id=f"PV-MODULE-{np.random.randint(1000, 9999)}",
        num_cells=60,
    )

    return iv_data


def example_iv_curve_analyzer():
    """Demonstrate IV curve analysis and parameter extraction."""
    print("\n" + "=" * 70)
    print("IV CURVE ANALYZER EXAMPLE")
    print("=" * 70)

    # Create analyzer
    analyzer = IVCurveAnalyzer(num_cells=60, cell_area_cm2=243.36)

    # Generate sample IV curve
    iv_data = generate_iv_curve(isc=9.2, voc=38.5, fill_factor=0.78)

    # 1. Curve Tracing
    print("\n1. Curve Tracing and Processing")
    print("-" * 70)
    v, i, p = analyzer.curve_tracing(iv_data, smooth=True, remove_outliers=True)
    print(f"Original data points: {len(iv_data.voltage)}")
    print(f"Processed data points: {len(v)}")
    print(f"Voltage range: {v[0]:.2f} V to {v[-1]:.2f} V")
    print(f"Current range: {i[-1]:.3f} A to {i[0]:.3f} A")
    print(f"Maximum power: {np.max(p):.2f} W")

    # 2. Parameter Extraction
    print("\n2. Electrical Parameter Extraction")
    print("-" * 70)
    params = analyzer.parameter_extraction(iv_data, extract_resistances=True)

    print(f"Voc (Open Circuit Voltage): {params.voc:.3f} V")
    print(f"Isc (Short Circuit Current): {params.isc:.3f} A")
    print(f"Vmp (Max Power Voltage): {params.vmp:.3f} V")
    print(f"Imp (Max Power Current): {params.imp:.3f} A")
    print(f"Pmp (Max Power): {params.pmp:.2f} W")
    print(f"Fill Factor: {params.fill_factor:.4f} ({params.fill_factor*100:.2f}%)")

    if params.efficiency is not None:
        print(f"Efficiency: {params.efficiency:.4f} ({params.efficiency*100:.2f}%)")
    if params.rs is not None:
        print(f"Series Resistance (Rs): {params.rs:.4f} Ω")
    if params.rsh is not None:
        print(f"Shunt Resistance (Rsh): {params.rsh:.2f} Ω")
    if params.ideality_factor is not None:
        print(f"Ideality Factor: {params.ideality_factor:.3f}")

    # 3. Mismatch Detection
    print("\n3. Cell Mismatch Detection")
    print("-" * 70)

    # Generate curve with mismatch
    v_mismatch = np.linspace(0, 38, 100)
    i_mismatch = 9.0 * (1 - v_mismatch / 38)
    # Add step to simulate mismatch
    i_mismatch[40:60] -= 1.0
    i_mismatch = np.maximum(i_mismatch, 0)

    iv_mismatch = IVCurveData(
        voltage=v_mismatch,
        current=i_mismatch,
        temperature=25.0,
        irradiance=1000.0,
        timestamp=datetime.now(),
    )

    mismatch_result = analyzer.mismatch_detection(iv_mismatch)
    print(f"Mismatch detected: {mismatch_result['mismatch_detected']}")
    print(f"Max step magnitude: {mismatch_result['max_step_magnitude']:.4f}")
    print(f"Confidence: {mismatch_result['confidence']:.2f}")
    print(f"Description: {mismatch_result['description']}")

    # 4. Degradation Analysis
    print("\n4. Degradation Analysis")
    print("-" * 70)

    # Create baseline parameters
    baseline_params = ElectricalParameters(
        voc=38.5,
        isc=9.2,
        vmp=31.5,
        imp=8.7,
        pmp=274.0,
        fill_factor=0.78,
    )

    # Create degraded parameters (10% power loss)
    degraded_params = ElectricalParameters(
        voc=38.0,  # 1.3% loss
        isc=9.0,  # 2.2% loss
        vmp=31.0,
        imp=7.95,
        pmp=246.0,  # 10% loss
        fill_factor=0.76,  # 2.6% loss
    )

    degradation = analyzer.degradation_analysis(degraded_params, baseline_params)

    print(f"Power degradation: {degradation.power_degradation_percent:.2f}%")
    print(f"Voc degradation: {degradation.voc_degradation_percent:.2f}%")
    print(f"Isc degradation: {degradation.isc_degradation_percent:.2f}%")
    print(f"Fill factor degradation: {degradation.ff_degradation_percent:.2f}%")
    print(f"Severity: {degradation.severity.value.upper()}")


def example_electrical_diagnostics():
    """Demonstrate electrical diagnostics capabilities."""
    print("\n" + "=" * 70)
    print("ELECTRICAL DIAGNOSTICS EXAMPLE")
    print("=" * 70)

    diagnostics = ElectricalDiagnostics()

    # 1. Cell Failure Detection
    print("\n1. Cell Failure Detection")
    print("-" * 70)

    # Generate healthy curve
    healthy_iv = generate_iv_curve(fill_factor=0.78)
    healthy_result = diagnostics.cell_failures(healthy_iv)
    print("Healthy Module:")
    print(f"  Failures detected: {healthy_result['failure_detected']}")
    print(f"  Overall severity: {healthy_result['overall_severity']}")

    # Generate curve with low fill factor (shunted cells)
    low_ff_iv = generate_iv_curve(fill_factor=0.55)
    low_ff_result = diagnostics.cell_failures(low_ff_iv)
    print("\nModule with Low Fill Factor:")
    print(f"  Failures detected: {low_ff_result['failure_detected']}")
    print(f"  Number of failures: {low_ff_result['failure_count']}")
    for failure in low_ff_result['failures']:
        print(f"  - {failure['type']}: {failure['description']}")

    # 2. Bypass Diode Issues
    print("\n2. Bypass Diode Issue Detection")
    print("-" * 70)

    normal_iv = generate_iv_curve()
    diode_result = diagnostics.bypass_diode_issues(normal_iv)
    print(f"Diode issues detected: {diode_result['diode_issues_detected']}")
    print(f"Number of issues: {diode_result['issue_count']}")
    print(f"Overall severity: {diode_result['overall_severity']}")

    # 3. String Underperformance
    print("\n3. String Underperformance Analysis")
    print("-" * 70)

    # Create string of modules with one underperforming
    string_data = []
    for i in range(10):
        if i == 5:
            # Module 5 is underperforming (15% power loss)
            iv = generate_iv_curve(degradation_factor=0.85)
        else:
            # Normal modules
            iv = generate_iv_curve()
        string_data.append(iv)

    # Expected parameters for healthy module
    expected_params = ElectricalParameters(
        voc=38.0,
        isc=9.0,
        vmp=31.0,
        imp=8.5,
        pmp=263.5,
        fill_factor=0.78,
    )

    string_result = diagnostics.string_underperformance(string_data, expected_params)

    print(f"Total modules in string: {string_result['total_modules']}")
    print(f"Underperforming modules: {string_result['underperforming_count']}")
    print(f"Module indices: {string_result['underperforming_modules']}")
    print(f"String health: {string_result['string_health']}")


def example_curve_comparison():
    """Demonstrate curve comparison and trend analysis."""
    print("\n" + "=" * 70)
    print("CURVE COMPARISON AND TREND ANALYSIS EXAMPLE")
    print("=" * 70)

    comparison = CurveComparison()

    # 1. Baseline Comparison
    print("\n1. Baseline Comparison")
    print("-" * 70)

    baseline_iv = generate_iv_curve(add_noise=False)
    current_iv = generate_iv_curve(degradation_factor=0.92, add_noise=False)  # 8% degradation

    baseline_result = comparison.baseline_comparison(current_iv, baseline_iv)

    print(f"Curve similarity score: {baseline_result['curve_similarity_score']:.4f}")
    print(f"NRMSD: {baseline_result['nrmsd']:.4f}")
    print(f"Overall health: {baseline_result['overall_health']}")

    print("\nDegradation Analysis:")
    deg = baseline_result['degradation_analysis']
    print(f"  Power degradation: {deg['power_degradation_percent']:.2f}%")
    print(f"  Voc degradation: {deg['voc_degradation_percent']:.2f}%")
    print(f"  Isc degradation: {deg['isc_degradation_percent']:.2f}%")
    print(f"  Fill factor degradation: {deg['ff_degradation_percent']:.2f}%")

    # 2. Trend Analysis
    print("\n2. Historical Trend Analysis")
    print("-" * 70)

    # Generate 5 years of data with 1% annual degradation
    historical_data = []
    base_time = datetime.now().timestamp()

    for year in range(6):
        degradation_factor = 1.0 - (year * 0.01)  # 1% per year
        iv = generate_iv_curve(degradation_factor=degradation_factor, add_noise=False)
        timestamp = base_time + (year * 365.25 * 24 * 3600)
        historical_data.append((timestamp, iv))

    trend_result = comparison.trend_analysis(historical_data)

    print(f"Number of measurements: {trend_result['num_measurements']}")
    print(f"Time span: {trend_result['time_span_years']:.1f} years")
    print(f"Power degradation rate: {trend_result['power_degradation_rate_per_year']:.3f}%/year")
    print(f"Voc degradation rate: {trend_result['voc_degradation_rate_per_year']:.3f}%/year")
    print(f"Isc degradation rate: {trend_result['isc_degradation_rate_per_year']:.3f}%/year")
    print(f"Fill factor degradation rate: {trend_result['ff_degradation_rate_per_year']:.3f}%/year")

    if trend_result['estimated_remaining_life_years'] is not None:
        print(f"Estimated remaining life: {trend_result['estimated_remaining_life_years']:.1f} years")

    print(f"Trend confidence (R²): {trend_result['trend_confidence']:.4f}")
    print(f"Severity: {trend_result['severity']}")

    # 3. Anomaly Detection
    print("\n3. Anomaly Detection")
    print("-" * 70)

    # Create historical data (normal operation)
    historical = [generate_iv_curve(add_noise=False) for _ in range(10)]

    # Normal current measurement
    current_normal = generate_iv_curve(add_noise=False)
    normal_anomaly = comparison.anomaly_detection(current_normal, historical)

    print("Normal Measurement:")
    print(f"  Is anomaly: {normal_anomaly['is_anomaly']}")
    print(f"  Anomaly score: {normal_anomaly['anomaly_score']:.3f}")
    print(f"  Severity: {normal_anomaly['severity']}")

    # Anomalous measurement (40% power drop)
    current_anomalous = generate_iv_curve(degradation_factor=0.60, add_noise=False)
    anomalous_result = comparison.anomaly_detection(current_anomalous, historical)

    print("\nAnomalous Measurement (40% power drop):")
    print(f"  Is anomaly: {anomalous_result['is_anomaly']}")
    print(f"  Anomaly score: {anomalous_result['anomaly_score']:.3f}")
    print(f"  Severity: {anomalous_result['severity']}")
    print(f"  Detected anomalies:")
    for anomaly in anomalous_result['anomalies']:
        print(f"    - {anomaly['parameter']}: Z-score = {anomaly['z_score']:.2f}")


def example_complete_analysis():
    """Demonstrate a complete real-world analysis workflow."""
    print("\n" + "=" * 70)
    print("COMPLETE ANALYSIS WORKFLOW EXAMPLE")
    print("=" * 70)

    print("\nScenario: Annual performance check of a 5-year-old PV module")
    print("-" * 70)

    # Create baseline (commissioning) IV curve
    print("\n1. Load baseline IV curve (from commissioning)")
    baseline_iv = generate_iv_curve(
        isc=9.2, voc=38.5, fill_factor=0.78, add_noise=False
    )
    analyzer = IVCurveAnalyzer()
    baseline_params = analyzer.parameter_extraction(baseline_iv)
    print(f"   Baseline Pmp: {baseline_params.pmp:.2f} W")
    print(f"   Baseline FF: {baseline_params.fill_factor:.4f}")

    # Create current IV curve (5 years later, 5% degradation + slight cell damage)
    print("\n2. Measure current IV curve")
    current_iv = generate_iv_curve(
        isc=9.0, voc=38.0, fill_factor=0.74, degradation_factor=0.95
    )
    current_params = analyzer.parameter_extraction(current_iv)
    print(f"   Current Pmp: {current_params.pmp:.2f} W")
    print(f"   Current FF: {current_params.fill_factor:.4f}")

    # Perform diagnostics
    print("\n3. Run electrical diagnostics")
    diagnostics = ElectricalDiagnostics()
    cell_result = diagnostics.cell_failures(current_iv)

    print(f"   Failures detected: {cell_result['failure_detected']}")
    if cell_result['failure_detected']:
        print(f"   Number of issues: {cell_result['failure_count']}")
        for failure in cell_result['failures']:
            print(f"   - {failure['type']}: {failure.get('description', 'N/A')}")

    # Compare to baseline
    print("\n4. Compare to baseline")
    comparison = CurveComparison()
    comparison_result = comparison.baseline_comparison(current_iv, baseline_iv)

    deg = comparison_result['degradation_analysis']
    print(f"   Power degradation: {deg['power_degradation_percent']:.2f}%")
    print(f"   Overall health: {comparison_result['overall_health']}")

    # Generate report
    print("\n5. PERFORMANCE ASSESSMENT REPORT")
    print("=" * 70)
    print(f"Module ID: PV-MODULE-001")
    print(f"Inspection Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Years in operation: 5")
    print()
    print(f"CURRENT PERFORMANCE:")
    print(f"  Max Power: {current_params.pmp:.2f} W (rated: {baseline_params.pmp:.2f} W)")
    print(f"  Fill Factor: {current_params.fill_factor:.4f}")
    print(f"  Voc: {current_params.voc:.2f} V")
    print(f"  Isc: {current_params.isc:.2f} A")
    print()
    print(f"DEGRADATION:")
    print(f"  Total power loss: {deg['power_degradation_percent']:.2f}%")
    print(f"  Annual degradation rate: ~{deg['power_degradation_percent']/5:.2f}%/year")
    print()
    print(f"HEALTH ASSESSMENT: {comparison_result['overall_health'].upper()}")
    print()
    if cell_result['failure_detected']:
        print(f"ISSUES IDENTIFIED:")
        for failure in cell_result['failures']:
            print(f"  ⚠ {failure['type']}: {failure.get('description', 'N/A')}")
    else:
        print(f"✓ No critical issues detected")
    print()
    print(f"RECOMMENDATION:")
    if deg['power_degradation_percent'] > 10:
        print("  ⚠ Module shows significant degradation. Consider replacement or")
        print("    detailed inspection to identify root cause.")
    elif deg['power_degradation_percent'] > 5:
        print("  Continue monitoring. Schedule next inspection in 6 months.")
    else:
        print("  ✓ Module performance is within expected range for its age.")
        print("    Continue annual monitoring.")


def main():
    """Run all IV curve analysis examples."""
    print("\n" + "#" * 70)
    print("# PV IV CURVE ANALYSIS - COMPREHENSIVE EXAMPLES")
    print("#" * 70)

    # Run examples
    example_iv_curve_analyzer()
    example_electrical_diagnostics()
    example_curve_comparison()
    example_complete_analysis()

    print("\n" + "#" * 70)
    print("# EXAMPLES COMPLETED")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
