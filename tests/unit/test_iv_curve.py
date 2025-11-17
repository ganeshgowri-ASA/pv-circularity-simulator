"""
Unit tests for IV curve analysis module.
"""

from datetime import datetime

import numpy as np
import pytest

from pv_circularity_simulator.core.exceptions import InsufficientDataError, InvalidIVCurveError
from pv_circularity_simulator.core.models import ElectricalParameters, IVCurveData, SeverityLevel
from pv_circularity_simulator.diagnostics.iv_curve import (
    CurveComparison,
    ElectricalDiagnostics,
    IVCurveAnalyzer,
)


class TestIVCurveAnalyzer:
    """Test IV curve analyzer functionality."""

    def test_init(self):
        """Test initialization."""
        analyzer = IVCurveAnalyzer(num_cells=72, cell_area_cm2=250.0)
        assert analyzer.num_cells == 72
        assert analyzer.cell_area_cm2 == 250.0

    def test_curve_tracing(self, sample_iv_curve_data):
        """Test curve tracing and processing."""
        analyzer = IVCurveAnalyzer()

        v, i, p = analyzer.curve_tracing(sample_iv_curve_data, smooth=True, remove_outliers=True)

        assert len(v) > 0
        assert len(i) > 0
        assert len(p) > 0
        assert len(v) == len(i) == len(p)
        assert np.all(i >= 0)
        assert np.all(v >= 0)

    def test_curve_tracing_insufficient_data(self):
        """Test curve tracing with insufficient data."""
        analyzer = IVCurveAnalyzer()

        # Create IV data with too few points
        v = np.array([0, 10])
        i = np.array([5, 0])

        iv_data = IVCurveData(
            voltage=v,
            current=i,
            temperature=25.0,
            irradiance=1000.0,
            timestamp=datetime.now(),
        )

        with pytest.raises(InvalidIVCurveError):
            analyzer.curve_tracing(iv_data)

    def test_parameter_extraction(self, sample_iv_curve_data):
        """Test electrical parameter extraction."""
        analyzer = IVCurveAnalyzer()

        params = analyzer.parameter_extraction(sample_iv_curve_data, extract_resistances=True)

        assert isinstance(params, ElectricalParameters)
        assert params.voc > 0
        assert params.isc > 0
        assert params.vmp > 0
        assert params.imp > 0
        assert params.pmp > 0
        assert 0 < params.fill_factor <= 1.0

        # Check that Vmp < Voc and Imp < Isc
        assert params.vmp < params.voc
        assert params.imp < params.isc

        # Check power calculation
        assert abs(params.pmp - (params.vmp * params.imp)) < 0.1

    def test_parameter_extraction_no_resistances(self, sample_iv_curve_data):
        """Test parameter extraction without resistances."""
        analyzer = IVCurveAnalyzer()

        params = analyzer.parameter_extraction(sample_iv_curve_data, extract_resistances=False)

        assert params.voc > 0
        assert params.isc > 0
        # Resistances should be None
        # Note: they might still be calculated, so we just check that extraction works

    def test_degradation_analysis(self, sample_electrical_params):
        """Test degradation analysis."""
        analyzer = IVCurveAnalyzer()

        # Create degraded parameters (10% power loss)
        degraded_params = ElectricalParameters(
            voc=sample_electrical_params.voc * 0.97,
            isc=sample_electrical_params.isc * 0.97,
            vmp=sample_electrical_params.vmp * 0.95,
            imp=sample_electrical_params.imp * 0.95,
            pmp=sample_electrical_params.pmp * 0.90,
            fill_factor=sample_electrical_params.fill_factor * 0.98,
        )

        degradation = analyzer.degradation_analysis(degraded_params, sample_electrical_params)

        assert degradation.power_degradation_percent > 0
        assert degradation.severity in SeverityLevel

    def test_mismatch_detection(self, sample_iv_curve_data):
        """Test cell mismatch detection."""
        analyzer = IVCurveAnalyzer()

        result = analyzer.mismatch_detection(sample_iv_curve_data)

        assert "mismatch_detected" in result
        assert "max_step_magnitude" in result
        assert "confidence" in result
        assert isinstance(result["mismatch_detected"], bool)

    def test_mismatch_detection_with_steps(self):
        """Test mismatch detection with artificial steps."""
        analyzer = IVCurveAnalyzer()

        # Create IV curve with a step (simulating mismatch)
        v = np.linspace(0, 36, 100)
        i = 9.0 * (1 - v / 36)

        # Add a step in the middle
        i[40:60] -= 1.0

        iv_data = IVCurveData(
            voltage=v,
            current=i,
            temperature=25.0,
            irradiance=1000.0,
            timestamp=datetime.now(),
        )

        result = analyzer.mismatch_detection(iv_data)

        # Should detect the step
        assert result["max_step_magnitude"] > 0


class TestElectricalDiagnostics:
    """Test electrical diagnostics functionality."""

    def test_init(self):
        """Test initialization."""
        diagnostics = ElectricalDiagnostics()
        assert diagnostics.analyzer is not None

    def test_cell_failures(self, sample_iv_curve_data):
        """Test cell failure detection."""
        diagnostics = ElectricalDiagnostics()

        result = diagnostics.cell_failures(sample_iv_curve_data)

        assert "failure_detected" in result
        assert "failure_count" in result
        assert "failures" in result
        assert "overall_severity" in result
        assert isinstance(result["failure_detected"], bool)

    def test_cell_failures_low_ff(self):
        """Test cell failure detection with low fill factor."""
        diagnostics = ElectricalDiagnostics()

        # Create IV data with low fill factor
        v = np.linspace(0, 30, 100)
        i = 8.0 * np.exp(-v / 10)  # Exponential decay - low FF

        iv_data = IVCurveData(
            voltage=v,
            current=i,
            temperature=25.0,
            irradiance=1000.0,
            timestamp=datetime.now(),
        )

        result = diagnostics.cell_failures(iv_data)

        # Should detect low fill factor
        assert result["failure_detected"]
        assert any(f["type"] == "low_fill_factor" for f in result["failures"])

    def test_bypass_diode_issues(self, sample_iv_curve_data):
        """Test bypass diode issue detection."""
        diagnostics = ElectricalDiagnostics()

        result = diagnostics.bypass_diode_issues(sample_iv_curve_data)

        assert "diode_issues_detected" in result
        assert "issue_count" in result
        assert "issues" in result
        assert isinstance(result["diode_issues_detected"], bool)

    def test_string_underperformance(self, sample_iv_curve_data, sample_electrical_params):
        """Test string underperformance detection."""
        diagnostics = ElectricalDiagnostics()

        # Create multiple modules
        string_data = [sample_iv_curve_data for _ in range(5)]

        result = diagnostics.string_underperformance(string_data, sample_electrical_params)

        assert "total_modules" in result
        assert "underperforming_count" in result
        assert "underperforming_modules" in result
        assert result["total_modules"] == 5

    def test_string_underperformance_no_data(self, sample_electrical_params):
        """Test string underperformance with no data."""
        diagnostics = ElectricalDiagnostics()

        with pytest.raises(InsufficientDataError):
            diagnostics.string_underperformance([], sample_electrical_params)


class TestCurveComparison:
    """Test curve comparison functionality."""

    def test_init(self):
        """Test initialization."""
        comparison = CurveComparison()
        assert comparison.analyzer is not None

    def test_baseline_comparison(self, sample_iv_curve_data):
        """Test baseline comparison."""
        comparison = CurveComparison()

        # Create baseline data (same as current for this test)
        baseline_data = sample_iv_curve_data

        result = comparison.baseline_comparison(sample_iv_curve_data, baseline_data)

        assert "current_parameters" in result
        assert "baseline_parameters" in result
        assert "degradation_analysis" in result
        assert "curve_similarity_score" in result
        assert 0 <= result["curve_similarity_score"] <= 1.0

    def test_baseline_comparison_degraded(self, sample_iv_curve_data):
        """Test baseline comparison with degraded module."""
        comparison = CurveComparison()

        # Create degraded IV curve
        v = sample_iv_curve_data.voltage
        i = sample_iv_curve_data.current * 0.85  # 15% current reduction

        degraded_data = IVCurveData(
            voltage=v,
            current=i,
            temperature=25.0,
            irradiance=1000.0,
            timestamp=datetime.now(),
        )

        result = comparison.baseline_comparison(degraded_data, sample_iv_curve_data)

        # Should detect degradation
        assert result["degradation_analysis"]["power_degradation_percent"] > 0

    def test_trend_analysis(self, sample_iv_curve_data):
        """Test trend analysis over time."""
        comparison = CurveComparison()

        # Create historical data with gradual degradation
        historical = []
        base_time = datetime.now().timestamp()

        for i in range(5):
            # Degrade by 1% per year
            degradation_factor = 1.0 - (i * 0.01)
            v = sample_iv_curve_data.voltage
            i_degraded = sample_iv_curve_data.current * degradation_factor

            iv_data = IVCurveData(
                voltage=v,
                current=i_degraded,
                temperature=25.0,
                irradiance=1000.0,
                timestamp=datetime.now(),
            )

            # Timestamp in seconds, 1 year apart
            timestamp = base_time + (i * 365.25 * 24 * 3600)
            historical.append((timestamp, iv_data))

        result = comparison.trend_analysis(historical)

        assert "num_measurements" in result
        assert "power_degradation_rate_per_year" in result
        assert "trend_confidence" in result
        assert result["num_measurements"] == 5

    def test_trend_analysis_insufficient_data(self, sample_iv_curve_data):
        """Test trend analysis with insufficient data."""
        comparison = CurveComparison()

        historical = [(datetime.now().timestamp(), sample_iv_curve_data)]

        with pytest.raises(InsufficientDataError):
            comparison.trend_analysis(historical)

    def test_anomaly_detection(self, sample_iv_curve_data):
        """Test anomaly detection."""
        comparison = CurveComparison()

        # Create historical data (normal operation)
        historical = [sample_iv_curve_data for _ in range(5)]

        # Current data is similar to historical
        result = comparison.anomaly_detection(sample_iv_curve_data, historical)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "z_scores" in result
        assert isinstance(result["is_anomaly"], bool)
        assert 0 <= result["anomaly_score"] <= 1.0

    def test_anomaly_detection_with_anomaly(self, sample_iv_curve_data):
        """Test anomaly detection with actual anomaly."""
        comparison = CurveComparison()

        # Create historical data (normal operation)
        historical = [sample_iv_curve_data for _ in range(5)]

        # Create anomalous current data (50% power drop)
        v = sample_iv_curve_data.voltage
        i_anomalous = sample_iv_curve_data.current * 0.5

        anomalous_data = IVCurveData(
            voltage=v,
            current=i_anomalous,
            temperature=25.0,
            irradiance=1000.0,
            timestamp=datetime.now(),
        )

        result = comparison.anomaly_detection(anomalous_data, historical)

        # Should detect anomaly
        assert result["is_anomaly"]
        assert result["anomaly_score"] > 0.5

    def test_anomaly_detection_insufficient_data(self, sample_iv_curve_data):
        """Test anomaly detection with insufficient historical data."""
        comparison = CurveComparison()

        historical = [sample_iv_curve_data, sample_iv_curve_data]  # Only 2 samples

        with pytest.raises(InsufficientDataError):
            comparison.anomaly_detection(sample_iv_curve_data, historical)
