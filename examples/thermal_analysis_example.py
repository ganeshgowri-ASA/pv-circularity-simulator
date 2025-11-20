"""
Example: Thermal Imaging Analysis for PV Modules

This example demonstrates how to use the thermal imaging analysis capabilities
to detect hotspots, analyze temperature distribution, and identify potential
failures in PV modules.
"""

from datetime import datetime

import numpy as np

from pv_circularity_simulator.core.models import ThermalImageData, ThermalImageMetadata
from pv_circularity_simulator.diagnostics.thermal import (
    HotspotSeverityClassifier,
    IRImageProcessing,
    ThermalImageAnalyzer,
)


def create_synthetic_thermal_image():
    """Create a synthetic thermal image with various defects for demonstration."""
    # Create base temperature field (healthy module at ~45°C)
    temps = np.random.normal(45.0, 1.5, (200, 150))

    # Add hotspot 1: Severe hotspot (cracked cell)
    temps[50:70, 50:70] += 25.0

    # Add hotspot 2: Moderate hotspot (partial shading)
    temps[120:135, 80:95] += 15.0

    # Add cold spot: Potential disconnection
    temps[80:95, 120:135] -= 12.0

    # Add edge heating pattern
    temps[:5, :] += 8.0
    temps[-5:, :] += 8.0

    # Create metadata
    metadata = ThermalImageMetadata(
        timestamp=datetime.now(),
        camera_model="FLIR E95",
        ambient_temp=28.0,
        measurement_distance=4.0,
        emissivity=0.90,
        wind_speed=2.5,
        irradiance=1000.0,
        module_id="PV-MODULE-001",
        notes="Routine inspection - sunny day",
    )

    # Create thermal image data
    thermal_data = ThermalImageData(
        temperature_matrix=temps,
        metadata=metadata,
        width=150,
        height=200,
    )

    return thermal_data


def example_ir_image_processing():
    """Demonstrate IR image processing capabilities."""
    print("\n" + "=" * 70)
    print("IR IMAGE PROCESSING EXAMPLE")
    print("=" * 70)

    # Create processor
    processor = IRImageProcessing(default_emissivity=0.90)

    # Create sample raw temperature data
    raw_temps = np.random.normal(45.0, 2.0, (100, 100))

    # 1. Temperature Calibration
    print("\n1. Temperature Calibration")
    print("-" * 70)
    calibrated = processor.temperature_calibration(
        raw_temperature=raw_temps,
        ambient_temp=25.0,
        distance=5.0,
        emissivity=0.90,
    )
    print(f"Raw temperature mean: {np.mean(raw_temps):.2f}°C")
    print(f"Calibrated temperature mean: {np.mean(calibrated):.2f}°C")

    # 2. Emissivity Correction
    print("\n2. Emissivity Correction")
    print("-" * 70)
    corrected = processor.emissivity_correction(
        temperature=raw_temps,
        measured_emissivity=0.90,
        actual_emissivity=0.85,
        ambient_temp=25.0,
    )
    print(f"Original mean: {np.mean(raw_temps):.2f}°C")
    print(f"Corrected mean: {np.mean(corrected):.2f}°C")

    # 3. Background Subtraction
    print("\n3. Background Subtraction")
    print("-" * 70)
    background_subtracted = processor.background_subtraction(
        temperature=raw_temps,
        background=np.ones_like(raw_temps) * 30.0,
        adaptive=True,
    )
    print(f"Background-subtracted mean: {np.mean(background_subtracted):.2f}°C")

    # 4. Denoising
    print("\n4. Image Denoising")
    print("-" * 70)
    noisy_temps = raw_temps + np.random.normal(0, 2.0, raw_temps.shape)
    denoised = processor.denoise_thermal_image(noisy_temps, sigma=1.5, method="gaussian")
    print(f"Noisy std dev: {np.std(noisy_temps):.2f}°C")
    print(f"Denoised std dev: {np.std(denoised):.2f}°C")
    print(f"Noise reduction: {(1 - np.std(denoised)/np.std(noisy_temps))*100:.1f}%")


def example_hotspot_severity_classification():
    """Demonstrate hotspot severity classification."""
    print("\n" + "=" * 70)
    print("HOTSPOT SEVERITY CLASSIFICATION EXAMPLE")
    print("=" * 70)

    classifier = HotspotSeverityClassifier()

    # Test different temperature deltas
    test_deltas = [5.0, 15.0, 25.0, 35.0]

    print("\n1. Severity Level Classification")
    print("-" * 70)
    for delta in test_deltas:
        severity = classifier.severity_levels(delta)
        print(f"Temperature delta: {delta:5.1f}°C → Severity: {severity.value.upper()}")

    # Power loss estimation
    print("\n2. Power Loss Estimation")
    print("-" * 70)
    power_loss = classifier.power_loss_estimation(
        temperature_delta=25.0,
        hotspot_area_fraction=0.08,  # 8% of module area
        module_power_rating=350.0,
    )
    print(f"Hotspot temperature delta: 25.0°C")
    print(f"Affected area: 8% of module")
    print(f"Power loss: {power_loss['power_loss_watts']:.2f} W ({power_loss['power_loss_percent']:.2f}%)")
    print(f"Annual energy loss: {power_loss['annual_energy_loss_kwh']:.2f} kWh")

    # Failure prediction
    print("\n3. Failure Prediction")
    print("-" * 70)
    from pv_circularity_simulator.core.models import SeverityLevel

    prediction = classifier.failure_prediction(
        severity=SeverityLevel.SEVERE,
        temperature_delta=28.0,
        duration_days=45,
    )
    print(f"Severity: {prediction['severity'].upper()}")
    print(f"Failure probability: {prediction['failure_probability']*100:.1f}%")
    print(f"Mean time to failure: {prediction['mean_time_to_failure_days']} days")
    print(f"Estimated remaining time: {prediction['estimated_remaining_days']} days")
    print(f"Recommended action: {prediction['recommended_action']}")


def example_thermal_image_analysis():
    """Demonstrate complete thermal image analysis."""
    print("\n" + "=" * 70)
    print("COMPLETE THERMAL IMAGE ANALYSIS EXAMPLE")
    print("=" * 70)

    # Create synthetic thermal image
    thermal_data = create_synthetic_thermal_image()

    # Create analyzer
    analyzer = ThermalImageAnalyzer(
        hotspot_threshold=10.0,
        min_hotspot_area=15,
    )

    # 1. Hotspot Detection
    print("\n1. Hotspot Detection")
    print("-" * 70)
    hotspots = analyzer.hotspot_detection(thermal_data, method="threshold")
    print(f"Number of hotspots detected: {len(hotspots)}")

    for i, hotspot in enumerate(hotspots, 1):
        print(f"\nHotspot #{i}:")
        print(f"  Location (row, col): {hotspot.location}")
        print(f"  Temperature: {hotspot.temperature:.2f}°C")
        print(f"  Temperature delta: {hotspot.temperature_delta:.2f}°C")
        print(f"  Area: {hotspot.area_pixels} pixels")
        print(f"  Severity: {hotspot.severity.value.upper()}")
        print(f"  Confidence: {hotspot.confidence:.2f}")

    # 2. Temperature Distribution Analysis
    print("\n2. Temperature Distribution Analysis")
    print("-" * 70)
    temp_stats = analyzer.temperature_distribution_analysis(thermal_data)
    print(f"Mean temperature: {temp_stats['mean_temperature']:.2f}°C")
    print(f"Median temperature: {temp_stats['median_temperature']:.2f}°C")
    print(f"Std deviation: {temp_stats['std_temperature']:.2f}°C")
    print(f"Min temperature: {temp_stats['min_temperature']:.2f}°C")
    print(f"Max temperature: {temp_stats['max_temperature']:.2f}°C")
    print(f"Temperature range: {temp_stats['temperature_range']:.2f}°C")
    print(f"Uniformity index: {temp_stats['uniformity_index']:.3f}")

    # 3. Thermal Anomaly Identification
    print("\n3. Thermal Anomaly Identification")
    print("-" * 70)
    anomalies = analyzer.thermal_anomaly_identification(thermal_data)
    print(f"Number of anomalies detected: {len(anomalies)}")

    for i, anomaly in enumerate(anomalies, 1):
        print(f"\nAnomaly #{i}:")
        print(f"  Type: {anomaly['type']}")
        print(f"  Severity: {anomaly['severity']}")
        if 'location' in anomaly:
            print(f"  Location: {anomaly['location']}")
        if 'temperature_delta' in anomaly:
            print(f"  Temperature delta: {anomaly['temperature_delta']:.2f}°C")

    # 4. Bypass Diode Failure Detection
    print("\n4. Bypass Diode Failure Detection")
    print("-" * 70)
    diode_failures = analyzer.bypass_diode_failures(thermal_data)
    print(f"Number of suspected failures: {len(diode_failures)}")

    for i, failure in enumerate(diode_failures, 1):
        print(f"\nSuspected Failure #{i}:")
        print(f"  Type: {failure['type']}")
        print(f"  Section: {failure['section_index'] + 1}")
        print(f"  Temperature delta: {failure['temperature_delta']:.2f}°C")
        print(f"  Confidence: {failure['confidence']:.2f}")
        print(f"  Description: {failure['description']}")

    # 5. Complete Analysis
    print("\n5. Complete Thermal Analysis Report")
    print("-" * 70)
    result = analyzer.analyze(thermal_data)

    print(f"\nOVERALL THERMAL HEALTH: {result.overall_severity.value.upper()}")
    print(f"Analysis confidence: {result.confidence:.2f}")
    print(f"\nSummary:")
    print(f"  - Hotspots detected: {len(result.hotspots)}")
    print(f"  - Bypass diode issues: {len(result.bypass_diode_failures)}")
    print(f"  - Mean temperature: {result.mean_temperature:.2f}°C")
    print(f"  - Temperature uniformity: {result.temperature_uniformity:.3f}")


def main():
    """Run all thermal analysis examples."""
    print("\n" + "#" * 70)
    print("# PV THERMAL IMAGING ANALYSIS - COMPREHENSIVE EXAMPLES")
    print("#" * 70)

    # Run examples
    example_ir_image_processing()
    example_hotspot_severity_classification()
    example_thermal_image_analysis()

    print("\n" + "#" * 70)
    print("# EXAMPLES COMPLETED")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
