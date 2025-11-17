"""Unit tests for IRImageProcessing class.

This module provides comprehensive tests for IR image processing and
time-series analysis methods.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_circularity.processing import IRImageProcessing
from pv_circularity.utils.validators import TimeSeriesData


@pytest.fixture
def sample_ir_image():
    """Create sample IR thermal image for testing."""
    # Create synthetic thermal image (256x256)
    np.random.seed(42)
    image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)

    # Add some hot spots
    image[50:70, 50:70] = 240  # Hot spot 1
    image[150:165, 150:165] = 235  # Hot spot 2
    image[100:110, 200:210] = 230  # Hot spot 3

    return image


@pytest.fixture
def sample_temperature_map():
    """Create sample temperature map for testing."""
    np.random.seed(42)
    # Temperature map in Celsius (256x256)
    temp_map = np.random.randn(256, 256) * 5 + 30  # Mean 30°C, std 5°C

    # Add hot spots with higher temperatures
    temp_map[50:70, 50:70] = 80  # Hot spot 1
    temp_map[150:165, 150:165] = 75  # Hot spot 2
    temp_map[100:110, 200:210] = 70  # Hot spot 3

    return temp_map


@pytest.fixture
def sample_time_series_for_ir():
    """Create sample time series for IR processing tests."""
    np.random.seed(42)
    n = 60  # 60 days of data
    timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

    # Create synthetic temperature data with trend and seasonality
    t = np.arange(n)
    trend = 0.1 * t  # Gradual warming
    seasonal = 5 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
    noise = np.random.randn(n) * 1
    values = (25 + trend + seasonal + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency="D",
        name="temperature_series",
    )


class TestIRImageProcessingInitialization:
    """Tests for IRImageProcessing initialization."""

    def test_init_basic(self, sample_ir_image):
        """Test basic initialization."""
        processor = IRImageProcessing(image=sample_ir_image)
        assert processor.image is not None
        assert processor.image.shape == (256, 256)

    def test_init_with_temperature_map(self, sample_ir_image, sample_temperature_map):
        """Test initialization with temperature map."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )
        assert processor.temperature_map is not None
        assert processor.temperature_map.shape == (256, 256)

    def test_init_with_metadata(self, sample_ir_image):
        """Test initialization with metadata."""
        metadata = {"timestamp": datetime.now(), "sensor": "FLIR"}
        processor = IRImageProcessing(image=sample_ir_image, metadata=metadata)
        assert processor.metadata == metadata

    def test_init_empty(self):
        """Test initialization without image."""
        processor = IRImageProcessing()
        assert processor.image is None


class TestHotSpotDetection:
    """Tests for hot spot detection."""

    def test_detect_hot_spots_basic(self, sample_ir_image):
        """Test basic hot spot detection."""
        processor = IRImageProcessing(image=sample_ir_image)
        hot_spots = processor.detect_hot_spots(threshold_percentile=95)

        assert isinstance(hot_spots, list)
        assert len(hot_spots) > 0
        # Should detect at least the 3 hot spots we created
        assert len(hot_spots) >= 3

    def test_detect_hot_spots_properties(self, sample_ir_image):
        """Test hot spot properties."""
        processor = IRImageProcessing(image=sample_ir_image)
        hot_spots = processor.detect_hot_spots(threshold_percentile=95, min_area=100)

        for hot_spot in hot_spots:
            assert "id" in hot_spot
            assert "centroid" in hot_spot
            assert "area" in hot_spot
            assert "max_intensity" in hot_spot
            assert "mean_intensity" in hot_spot
            assert "bbox" in hot_spot
            assert hot_spot["area"] >= 100

    def test_detect_hot_spots_with_temperature(
        self, sample_ir_image, sample_temperature_map
    ):
        """Test hot spot detection with temperature map."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )
        hot_spots = processor.detect_hot_spots(threshold_percentile=95)

        for hot_spot in hot_spots:
            assert "max_temperature" in hot_spot
            assert "mean_temperature" in hot_spot
            assert hot_spot["max_temperature"] > 0

    def test_detect_hot_spots_no_image(self):
        """Test hot spot detection without image."""
        processor = IRImageProcessing()
        with pytest.raises(ValueError):
            processor.detect_hot_spots()

    def test_detect_hot_spots_min_area(self, sample_ir_image):
        """Test hot spot detection with minimum area filter."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Small min_area should find more hot spots
        hot_spots_small = processor.detect_hot_spots(
            threshold_percentile=95, min_area=10
        )

        # Large min_area should find fewer hot spots
        hot_spots_large = processor.detect_hot_spots(
            threshold_percentile=95, min_area=500
        )

        assert len(hot_spots_small) >= len(hot_spots_large)


class TestTemperatureSeriesExtraction:
    """Tests for temperature series extraction."""

    def test_extract_temperature_series_full_image(
        self, sample_ir_image, sample_temperature_map
    ):
        """Test extracting temperature from full image."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )
        temps = processor.extract_temperature_series(aggregation="mean")

        assert len(temps) == 1
        assert 20 < temps[0] < 40  # Should be around 30°C with hot spots

    def test_extract_temperature_series_region(
        self, sample_ir_image, sample_temperature_map
    ):
        """Test extracting temperature from specific region."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )

        # Extract from hot spot region
        temps = processor.extract_temperature_series(
            region=(50, 50, 70, 70),
            aggregation="mean",
        )

        assert len(temps) == 1
        assert temps[0] > 70  # Hot spot temperature

    def test_extract_temperature_series_aggregations(
        self, sample_ir_image, sample_temperature_map
    ):
        """Test different aggregation methods."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )

        mean_temp = processor.extract_temperature_series(aggregation="mean")
        max_temp = processor.extract_temperature_series(aggregation="max")
        median_temp = processor.extract_temperature_series(aggregation="median")

        assert mean_temp[0] < max_temp[0]
        assert median_temp[0] > 0

    def test_extract_temperature_series_without_temp_map(self, sample_ir_image):
        """Test extraction without temperature map (uses image intensities)."""
        processor = IRImageProcessing(image=sample_ir_image, verbose=True)
        temps = processor.extract_temperature_series(aggregation="mean")

        assert len(temps) == 1
        assert 0 <= temps[0] <= 255

    def test_extract_temperature_series_no_data(self):
        """Test extraction without any data."""
        processor = IRImageProcessing()
        with pytest.raises(ValueError):
            processor.extract_temperature_series()


class TestSeasonalDecomposition:
    """Tests for seasonal decomposition."""

    def test_seasonal_decomposition_basic(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test basic seasonal decomposition."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.seasonal_decomposition(
            time_series_data=sample_time_series_for_ir,
            period=7,
        )

        assert len(result.trend) == len(sample_time_series_for_ir.values)
        assert len(result.seasonal) == len(sample_time_series_for_ir.values)
        assert len(result.residual) == len(sample_time_series_for_ir.values)
        assert len(result.timestamps) == len(sample_time_series_for_ir.values)

    def test_seasonal_decomposition_multiplicative(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test multiplicative decomposition."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.seasonal_decomposition(
            time_series_data=sample_time_series_for_ir,
            period=7,
            model="multiplicative",
        )

        assert len(result.trend) > 0
        assert len(result.seasonal) > 0

    def test_seasonal_decomposition_with_series(self, sample_ir_image):
        """Test decomposition with pandas Series."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Create Series
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        values = 25 + 5 * np.sin(2 * np.pi * np.arange(60) / 7) + np.random.randn(60)
        series = pd.Series(values, index=dates)

        result = processor.seasonal_decomposition(series, period=7)

        assert len(result.trend) == 60

    def test_seasonal_decomposition_with_array(self, sample_ir_image):
        """Test decomposition with numpy array."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Create array
        values = 25 + 5 * np.sin(2 * np.pi * np.arange(60) / 7) + np.random.randn(60)

        result = processor.seasonal_decomposition(values, period=7)

        assert len(result.trend) == 60

    def test_seasonal_decomposition_invalid_period(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test decomposition with invalid period."""
        processor = IRImageProcessing(image=sample_ir_image)

        with pytest.raises(ValueError):
            processor.seasonal_decomposition(sample_time_series_for_ir, period=1)

    def test_seasonal_decomposition_insufficient_data(self, sample_ir_image):
        """Test decomposition with insufficient data."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Create short series
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        values = list(range(5))
        short_series = TimeSeriesData(timestamps=timestamps, values=values)

        with pytest.raises(ValueError):
            processor.seasonal_decomposition(short_series, period=7)

    def test_seasonal_decomposition_to_dataframe(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test conversion of decomposition result to DataFrame."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.seasonal_decomposition(sample_time_series_for_ir, period=7)

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "trend" in df.columns
        assert "seasonal" in df.columns
        assert "residual" in df.columns


class TestTrendAnalysis:
    """Tests for trend analysis."""

    def test_trend_analysis_linear(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test linear trend analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.trend_analysis(
            time_series_data=sample_time_series_for_ir,
            method="linear",
        )

        assert "slope" in result
        assert "intercept" in result
        assert "r_squared" in result
        assert "p_value" in result
        assert "trend_values" in result
        assert "confidence_intervals" in result
        assert result["slope"] > 0  # Upward trend in our data

    def test_trend_analysis_polynomial(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test polynomial trend analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.trend_analysis(
            time_series_data=sample_time_series_for_ir,
            method="polynomial",
        )

        assert "coefficients" in result
        assert "r_squared" in result
        assert "trend_values" in result
        assert len(result["coefficients"]) == 3  # Degree 2 polynomial

    def test_trend_analysis_lowess(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test LOWESS trend analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.trend_analysis(
            time_series_data=sample_time_series_for_ir,
            method="lowess",
        )

        assert "r_squared" in result
        assert "trend_values" in result
        assert result["method"] == "lowess"

    def test_trend_analysis_with_series(self, sample_ir_image):
        """Test trend analysis with pandas Series."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Create Series with clear upward trend
        values = np.arange(50) + np.random.randn(50) * 2
        series = pd.Series(values)

        result = processor.trend_analysis(series, method="linear")

        assert result["slope"] > 0  # Upward trend
        assert result["r_squared"] > 0.9  # Strong fit

    def test_trend_analysis_invalid_method(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test trend analysis with invalid method."""
        processor = IRImageProcessing(image=sample_ir_image)

        with pytest.raises(ValueError):
            processor.trend_analysis(sample_time_series_for_ir, method="invalid")

    def test_trend_analysis_insufficient_data(self, sample_ir_image):
        """Test trend analysis with insufficient data."""
        processor = IRImageProcessing(image=sample_ir_image)

        timestamps = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = [1.0, 2.0]
        short_series = TimeSeriesData(timestamps=timestamps, values=values)

        with pytest.raises(ValueError):
            processor.trend_analysis(short_series)


class TestResidualAnalysis:
    """Tests for residual analysis."""

    def test_residual_analysis_basic(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test basic residual analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.residual_analysis(
            time_series_data=sample_time_series_for_ir,
        )

        assert "residuals" in result
        assert "mean" in result
        assert "std" in result
        assert "median" in result
        assert "normality_test" in result
        assert "autocorrelation_lag1" in result
        assert "outliers" in result
        assert "statistics" in result

    def test_residual_analysis_with_decomposition(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test residual analysis with seasonal decomposition."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.residual_analysis(
            time_series_data=sample_time_series_for_ir,
            decomposition_period=7,
        )

        assert "residuals" in result
        assert len(result["residuals"]) == len(sample_time_series_for_ir.values)

    def test_residual_analysis_normality_test(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test normality test in residual analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.residual_analysis(sample_time_series_for_ir)

        assert "normality_test" in result
        normality = result["normality_test"]
        if "p_value" in normality:
            assert "is_normal" in normality
            assert isinstance(normality["is_normal"], bool)

    def test_residual_analysis_outliers(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test outlier detection in residual analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.residual_analysis(sample_time_series_for_ir)

        assert "outliers" in result
        outliers = result["outliers"]
        assert "indices" in outliers
        assert "count" in outliers
        assert "percentage" in outliers
        assert isinstance(outliers["indices"], list)

    def test_residual_analysis_statistics(
        self, sample_ir_image, sample_time_series_for_ir
    ):
        """Test statistical measures in residual analysis."""
        processor = IRImageProcessing(image=sample_ir_image)
        result = processor.residual_analysis(sample_time_series_for_ir)

        stats = result["statistics"]
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats

    def test_residual_analysis_with_array(self, sample_ir_image):
        """Test residual analysis with numpy array."""
        processor = IRImageProcessing(image=sample_ir_image)

        # Create array
        values = 25 + 0.1 * np.arange(60) + np.random.randn(60)

        result = processor.residual_analysis(values)

        assert "residuals" in result
        assert len(result["residuals"]) == 60

    def test_residual_analysis_insufficient_data(self, sample_ir_image):
        """Test residual analysis with insufficient data."""
        processor = IRImageProcessing(image=sample_ir_image)

        timestamps = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = [1.0, 2.0]
        short_series = TimeSeriesData(timestamps=timestamps, values=values)

        with pytest.raises(ValueError):
            processor.residual_analysis(short_series)


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_repr_with_image(self, sample_ir_image):
        """Test string representation with image."""
        processor = IRImageProcessing(image=sample_ir_image)
        repr_str = repr(processor)

        assert "IRImageProcessing" in repr_str
        assert "(256, 256)" in repr_str

    def test_repr_without_image(self):
        """Test string representation without image."""
        processor = IRImageProcessing()
        repr_str = repr(processor)

        assert "IRImageProcessing" in repr_str
        assert "no image loaded" in repr_str

    def test_repr_with_temperature_map(
        self, sample_ir_image, sample_temperature_map
    ):
        """Test string representation with temperature map."""
        processor = IRImageProcessing(
            image=sample_ir_image,
            temperature_map=sample_temperature_map,
        )
        repr_str = repr(processor)

        assert "has_temp_map=True" in repr_str


class TestDecompositionResultValidation:
    """Tests for DecompositionResult validation."""

    def test_valid_decomposition_result(self):
        """Test creation of valid DecompositionResult."""
        from pv_circularity.utils.validators import DecompositionResult

        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        trend = list(range(10))
        seasonal = [0.0] * 10
        residual = [0.0] * 10

        result = DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            timestamps=timestamps,
        )

        assert len(result.trend) == 10

    def test_invalid_decomposition_result_lengths(self):
        """Test DecompositionResult with mismatched lengths."""
        from pv_circularity.utils.validators import DecompositionResult

        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        trend = list(range(10))
        seasonal = [0.0] * 5  # Wrong length
        residual = [0.0] * 10

        with pytest.raises(ValueError):
            DecompositionResult(
                trend=trend,
                seasonal=seasonal,
                residual=residual,
                timestamps=timestamps,
            )
