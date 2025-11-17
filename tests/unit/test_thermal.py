"""
Unit tests for thermal imaging analysis module.
"""

import numpy as np
import pytest

from pv_circularity_simulator.core.exceptions import CalibrationError, InvalidThermalDataError
from pv_circularity_simulator.core.models import SeverityLevel
from pv_circularity_simulator.diagnostics.thermal import (
    HotspotSeverityClassifier,
    IRImageProcessing,
    ThermalImageAnalyzer,
)


class TestIRImageProcessing:
    """Test IR image processing functionality."""

    def test_init(self):
        """Test initialization."""
        processor = IRImageProcessing(default_emissivity=0.85)
        assert processor.default_emissivity == 0.85

    def test_init_invalid_emissivity(self):
        """Test initialization with invalid emissivity."""
        with pytest.raises(ValueError):
            IRImageProcessing(default_emissivity=1.5)

    def test_temperature_calibration(self):
        """Test temperature calibration."""
        processor = IRImageProcessing()
        raw_temps = np.array([[30.0, 32.0], [31.0, 33.0]])

        calibrated = processor.temperature_calibration(
            raw_temps, ambient_temp=25.0, distance=5.0, emissivity=0.90
        )

        assert calibrated.shape == raw_temps.shape
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_emissivity_correction(self):
        """Test emissivity correction."""
        processor = IRImageProcessing()
        temps = np.array([[35.0, 37.0], [36.0, 38.0]])

        corrected = processor.emissivity_correction(
            temps, measured_emissivity=0.90, actual_emissivity=0.85, ambient_temp=25.0
        )

        assert corrected.shape == temps.shape
        assert not np.any(np.isnan(corrected))

    def test_emissivity_correction_invalid(self):
        """Test emissivity correction with invalid values."""
        processor = IRImageProcessing()
        temps = np.array([[35.0, 37.0], [36.0, 38.0]])

        with pytest.raises(CalibrationError):
            processor.emissivity_correction(
                temps, measured_emissivity=1.5, actual_emissivity=0.85
            )

    def test_background_subtraction(self):
        """Test background subtraction."""
        processor = IRImageProcessing()
        module_temps = np.array([[35.0, 37.0], [36.0, 38.0]])
        background = np.array([[30.0, 30.0], [30.0, 30.0]])

        result = processor.background_subtraction(module_temps, background, adaptive=False)

        assert result.shape == module_temps.shape
        assert np.all(result >= 0)

    def test_denoise_gaussian(self):
        """Test Gaussian denoising."""
        processor = IRImageProcessing()
        noisy_temps = np.random.rand(20, 20) * 5 + 30

        denoised = processor.denoise_thermal_image(noisy_temps, sigma=1.0, method="gaussian")

        assert denoised.shape == noisy_temps.shape
        # Denoised should have lower std dev
        assert np.std(denoised) <= np.std(noisy_temps)

    def test_denoise_median(self):
        """Test median denoising."""
        processor = IRImageProcessing()
        noisy_temps = np.random.rand(20, 20) * 5 + 30

        denoised = processor.denoise_thermal_image(noisy_temps, sigma=2.0, method="median")

        assert denoised.shape == noisy_temps.shape


class TestHotspotSeverityClassifier:
    """Test hotspot severity classification."""

    def test_severity_levels(self):
        """Test severity level classification."""
        classifier = HotspotSeverityClassifier()

        assert classifier.severity_levels(5.0) == SeverityLevel.NORMAL
        assert classifier.severity_levels(15.0) == SeverityLevel.WARNING
        assert classifier.severity_levels(25.0) == SeverityLevel.SEVERE
        assert classifier.severity_levels(35.0) == SeverityLevel.CRITICAL

    def test_power_loss_estimation(self):
        """Test power loss estimation."""
        classifier = HotspotSeverityClassifier()

        result = classifier.power_loss_estimation(
            temperature_delta=20.0, hotspot_area_fraction=0.05, module_power_rating=300.0
        )

        assert "power_loss_watts" in result
        assert "power_loss_percent" in result
        assert "annual_energy_loss_kwh" in result
        assert result["power_loss_watts"] >= 0

    def test_failure_prediction(self):
        """Test failure probability prediction."""
        classifier = HotspotSeverityClassifier()

        result = classifier.failure_prediction(
            severity=SeverityLevel.SEVERE, temperature_delta=25.0, duration_days=30
        )

        assert "failure_probability" in result
        assert "mean_time_to_failure_days" in result
        assert "recommended_action" in result
        assert 0 <= result["failure_probability"] <= 1.0


class TestThermalImageAnalyzer:
    """Test thermal image analysis."""

    def test_init(self):
        """Test initialization."""
        analyzer = ThermalImageAnalyzer(hotspot_threshold=15.0, min_hotspot_area=20)
        assert analyzer.hotspot_threshold == 15.0
        assert analyzer.min_hotspot_area == 20

    def test_hotspot_detection_threshold(self, sample_thermal_image):
        """Test hotspot detection using threshold method."""
        analyzer = ThermalImageAnalyzer(hotspot_threshold=10.0, min_hotspot_area=10)

        hotspots = analyzer.hotspot_detection(sample_thermal_image, method="threshold")

        assert isinstance(hotspots, list)
        # Should detect the synthetic hotspot we added
        assert len(hotspots) > 0

        # Check hotspot properties
        hotspot = hotspots[0]
        assert hotspot.temperature > 0
        assert hotspot.temperature_delta > 10.0
        assert hotspot.area_pixels >= 10
        assert hotspot.severity in SeverityLevel

    def test_hotspot_detection_zscore(self, sample_thermal_image):
        """Test hotspot detection using Z-score method."""
        analyzer = ThermalImageAnalyzer()

        hotspots = analyzer.hotspot_detection(sample_thermal_image, method="zscore")

        assert isinstance(hotspots, list)
        assert len(hotspots) > 0

    def test_hotspot_detection_clustering(self, sample_thermal_image):
        """Test hotspot detection using clustering method."""
        analyzer = ThermalImageAnalyzer()

        hotspots = analyzer.hotspot_detection(sample_thermal_image, method="clustering")

        assert isinstance(hotspots, list)

    def test_temperature_distribution_analysis(self, sample_thermal_image):
        """Test temperature distribution analysis."""
        analyzer = ThermalImageAnalyzer()

        stats = analyzer.temperature_distribution_analysis(sample_thermal_image)

        assert "mean_temperature" in stats
        assert "median_temperature" in stats
        assert "std_temperature" in stats
        assert "uniformity_index" in stats

        assert 0 <= stats["uniformity_index"] <= 1.0

    def test_thermal_anomaly_identification(self, sample_thermal_image):
        """Test thermal anomaly identification."""
        analyzer = ThermalImageAnalyzer()

        anomalies = analyzer.thermal_anomaly_identification(sample_thermal_image)

        assert isinstance(anomalies, list)

    def test_bypass_diode_failures(self, sample_thermal_image):
        """Test bypass diode failure detection."""
        analyzer = ThermalImageAnalyzer()

        failures = analyzer.bypass_diode_failures(sample_thermal_image)

        assert isinstance(failures, list)

    def test_analyze_complete(self, sample_thermal_image):
        """Test complete thermal analysis."""
        analyzer = ThermalImageAnalyzer()

        result = analyzer.analyze(sample_thermal_image)

        assert result.mean_temperature > 0
        assert result.median_temperature > 0
        assert 0 <= result.temperature_uniformity <= 1.0
        assert result.overall_severity in SeverityLevel
        assert 0 <= result.confidence <= 1.0
        assert isinstance(result.hotspots, list)
        assert isinstance(result.bypass_diode_failures, list)

    def test_analyze_empty_data(self, sample_thermal_metadata):
        """Test analysis with empty temperature data."""
        from pv_circularity_simulator.core.models import ThermalImageData

        empty_thermal = ThermalImageData(
            temperature_matrix=np.array([[]]),
            metadata=sample_thermal_metadata,
            width=0,
            height=0,
        )

        analyzer = ThermalImageAnalyzer()

        with pytest.raises(InvalidThermalDataError):
            analyzer.analyze(empty_thermal)
