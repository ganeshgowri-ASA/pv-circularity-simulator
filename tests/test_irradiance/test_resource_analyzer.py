"""Tests for SolarResourceAnalyzer."""

import pandas as pd
import pytest
import numpy as np

from src.irradiance.resource_analyzer import SolarResourceAnalyzer
from src.irradiance.models import ResourceStatistics


@pytest.fixture
def annual_data():
    """Generate synthetic annual irradiance data."""
    times = pd.date_range(start="2024-01-01", end="2024-12-31 23:00", freq="h", tz="UTC")

    # Simplified solar pattern
    day_of_year = times.dayofyear
    hour = times.hour

    # Seasonal variation (higher in summer)
    seasonal = 500 + 400 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Diurnal variation (peak at noon)
    diurnal = np.maximum(0, np.cos(2 * np.pi * (hour - 12) / 24))

    # Combine and add noise
    irradiance = seasonal * diurnal + np.random.normal(0, 50, len(times))
    irradiance = np.maximum(0, irradiance)  # No negative values

    return pd.Series(irradiance, index=times)


class TestSolarResourceAnalyzer:
    """Test suite for SolarResourceAnalyzer."""

    def test_initialization(self, annual_data):
        """Test analyzer initialization."""
        analyzer = SolarResourceAnalyzer(annual_data, data_label="GHI")
        assert len(analyzer.data) == len(annual_data)
        assert analyzer.data_label == "GHI"

    def test_monthly_averages(self, annual_data):
        """Test monthly statistics calculation."""
        analyzer = SolarResourceAnalyzer(annual_data)
        monthly_stats = analyzer.monthly_averages(aggregation="mean")

        # Should have 12 months
        assert len(monthly_stats) == 12

        # Check columns
        assert "Mean" in monthly_stats.columns
        assert "Median" in monthly_stats.columns
        assert "Std" in monthly_stats.columns
        assert "P90" in monthly_stats.columns

        # All values should be non-negative
        assert all(monthly_stats["Mean"] >= 0)

    def test_seasonal_patterns(self, annual_data):
        """Test seasonal pattern analysis."""
        analyzer = SolarResourceAnalyzer(annual_data)

        # Test both hemispheres
        for hemisphere in ["north", "south"]:
            seasonal_stats = analyzer.seasonal_patterns(hemishere=hemisphere)

            # Should have 4 seasons
            assert len(seasonal_stats) == 4

            # Check that seasons exist
            assert "Winter" in seasonal_stats.index
            assert "Summer" in seasonal_stats.index

            # All values should be non-negative
            assert all(seasonal_stats["Mean"] >= 0)

    def test_interannual_variability(self, annual_data):
        """Test interannual variability analysis."""
        analyzer = SolarResourceAnalyzer(annual_data)
        summary, yearly_detail = analyzer.interannual_variability()

        # Check summary metrics
        assert "Mean Annual Total" in summary["Metric"].values
        assert "CoeffVariation" in summary["Metric"].values

        # Check yearly details
        assert "Annual Total" in yearly_detail.columns
        assert "Daily Mean" in yearly_detail.columns

    def test_p50_p90_analysis(self, annual_data):
        """Test P50/P90 exceedance analysis."""
        analyzer = SolarResourceAnalyzer(annual_data)

        # Test different aggregations
        for aggregation in ["daily", "monthly"]:
            p_analysis = analyzer.p50_p90_analysis(time_aggregation=aggregation)

            # Check dictionary keys
            assert "summary" in p_analysis
            assert "distribution" in p_analysis
            assert "time_series" in p_analysis

            # Check summary
            summary = p_analysis["summary"]
            assert "P50" in summary["Percentile"].values
            assert "P90" in summary["Percentile"].values

            # P90 should be less than P50 (conservative estimate)
            p50_val = summary[summary["Percentile"] == "P50"]["Value"].values[0]
            p90_val = summary[summary["Percentile"] == "P90"]["Value"].values[0]
            assert p90_val < p50_val

    def test_solar_resource_maps(self, annual_data):
        """Test resource map data generation."""
        analyzer = SolarResourceAnalyzer(annual_data)
        maps = analyzer.solar_resource_maps()

        # Check map types
        assert "hourly_by_month" in maps
        assert "daily_hourly" in maps
        assert "availability" in maps

        # Check hourly_by_month dimensions
        hourly_by_month = maps["hourly_by_month"]
        assert hourly_by_month.shape[0] == 24  # 24 hours
        assert hourly_by_month.shape[1] == 12  # 12 months

        # All values should be non-negative
        assert all(hourly_by_month.min() >= 0)

    def test_generate_resource_summary(self, annual_data):
        """Test resource statistics summary."""
        analyzer = SolarResourceAnalyzer(annual_data)
        summary = analyzer.generate_resource_summary()

        # Check that it's a ResourceStatistics object
        assert isinstance(summary, ResourceStatistics)

        # Check key metrics exist
        assert summary.mean >= 0
        assert summary.p50 >= 0
        assert summary.p90 >= 0
        assert 0 <= summary.coefficient_of_variation <= 5  # Reasonable CV range

    def test_capacity_factor_range(self, annual_data):
        """Test capacity factor estimation."""
        # Resample to get reasonable annual totals
        daily_data = annual_data.resample("h").mean()

        analyzer = SolarResourceAnalyzer(daily_data)
        cf_range = analyzer.capacity_factor_range(system_capacity_kw=100.0)

        # Check keys
        assert "p50" in cf_range
        assert "p90" in cf_range
        assert "p50_annual_kwh" in cf_range
        assert "p90_annual_kwh" in cf_range

        # Capacity factors should be between 0 and 1
        assert 0 <= cf_range["p50"] <= 1
        assert 0 <= cf_range["p90"] <= 1

        # P90 should be more conservative (lower) than P50
        assert cf_range["p90"] <= cf_range["p50"]

    def test_identify_resource_anomalies(self, annual_data):
        """Test anomaly detection."""
        analyzer = SolarResourceAnalyzer(annual_data)

        # Inject some anomalies
        anomaly_data = annual_data.copy()
        anomaly_data.iloc[1000:1010] = 2000  # Very high values

        analyzer_with_anomalies = SolarResourceAnalyzer(anomaly_data)
        anomalies = analyzer_with_anomalies.identify_resource_anomalies(threshold_std=2.0)

        # Should detect some anomalies
        assert len(anomalies) > 0

        # Check columns
        assert "Timestamp" in anomalies.columns
        assert "Z-Score" in anomalies.columns
        assert "Deviation" in anomalies.columns
