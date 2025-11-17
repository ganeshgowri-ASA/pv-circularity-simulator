"""IR Image Processing Module with Time-Series Analysis.

This module provides comprehensive IR (Infrared) image processing capabilities
for photovoltaic systems, including thermal analysis and time-series decomposition
for temperature trend analysis.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from statsmodels.tsa.seasonal import seasonal_decompose

from pv_circularity.utils.validators import DecompositionResult, TimeSeriesData


class IRImageProcessing:
    """Production-ready IR image processing with time-series analysis.

    This class provides comprehensive tools for processing infrared thermal images
    of photovoltaic systems, including hot-spot detection, temperature mapping,
    and time-series analysis of thermal data.

    Attributes:
        image: Current IR image being processed.
        temperature_map: Optional temperature map corresponding to the image.
        metadata: Optional metadata dictionary.
        verbose: Whether to print detailed progress information.

    Example:
        >>> from pv_circularity.processing import IRImageProcessing
        >>>
        >>> # Load IR image
        >>> processor = IRImageProcessing.from_file("thermal_image.png")
        >>>
        >>> # Detect hot spots
        >>> hot_spots = processor.detect_hot_spots(threshold_percentile=95)
        >>>
        >>> # Extract temperature time series
        >>> temp_series = processor.extract_temperature_series(region=(100, 100, 200, 200))
        >>>
        >>> # Perform seasonal decomposition
        >>> decomposition = processor.seasonal_decomposition(temp_series, period=24)
    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        temperature_map: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """Initialize the IRImageProcessing instance.

        Args:
            image: IR image array (grayscale or thermal).
            temperature_map: Optional temperature map in Celsius.
            metadata: Optional metadata dictionary.
            verbose: Whether to print detailed progress information.
        """
        self.image = image
        self.temperature_map = temperature_map
        self.metadata = metadata or {}
        self.verbose = verbose
        self._time_series_cache: Dict[str, TimeSeriesData] = {}

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        temperature_map_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> "IRImageProcessing":
        """Load IR image from file.

        Args:
            filepath: Path to the IR image file.
            temperature_map_path: Optional path to temperature map file.
            verbose: Whether to print detailed progress information.

        Returns:
            IRImageProcessing instance with loaded image.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be loaded.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")

        # Load image (support grayscale and thermal)
        image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from: {filepath}")

        # Load temperature map if provided
        temperature_map = None
        if temperature_map_path:
            temp_path = Path(temperature_map_path)
            if temp_path.exists():
                temperature_map = np.load(str(temp_path))

        metadata = {"filepath": str(filepath), "shape": image.shape}

        if verbose:
            print(f"Loaded IR image: {filepath} (shape: {image.shape})")

        return cls(
            image=image,
            temperature_map=temperature_map,
            metadata=metadata,
            verbose=verbose,
        )

    def detect_hot_spots(
        self,
        threshold_percentile: float = 95.0,
        min_area: int = 10,
        gaussian_sigma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Detect hot spots in the IR image.

        Hot spots indicate potential defects or issues in PV modules.

        Args:
            threshold_percentile: Percentile threshold for hot spot detection.
            min_area: Minimum area (in pixels) for a hot spot.
            gaussian_sigma: Sigma for Gaussian smoothing.

        Returns:
            List of dictionaries containing hot spot information.

        Raises:
            ValueError: If no image is loaded.

        Example:
            >>> hot_spots = processor.detect_hot_spots(threshold_percentile=95)
            >>> print(f"Found {len(hot_spots)} hot spots")
        """
        if self.image is None:
            raise ValueError("No image loaded. Use from_file() or set image attribute.")

        if self.verbose:
            print("Detecting hot spots...")

        # Smooth image
        smoothed = gaussian_filter(self.image.astype(float), sigma=gaussian_sigma)

        # Calculate threshold
        threshold = np.percentile(smoothed, threshold_percentile)

        # Create binary mask
        binary_mask = smoothed > threshold

        # Remove small regions
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)

        # Label connected components
        labeled_mask = morphology.label(cleaned_mask)

        # Extract hot spot properties
        hot_spots = []
        for region_id in range(1, labeled_mask.max() + 1):
            region_mask = labeled_mask == region_id
            region_coords = np.argwhere(region_mask)

            if len(region_coords) == 0:
                continue

            # Calculate properties
            y_coords, x_coords = region_coords[:, 0], region_coords[:, 1]
            centroid = (int(x_coords.mean()), int(y_coords.mean()))
            area = len(region_coords)
            max_temp = smoothed[region_mask].max()
            mean_temp = smoothed[region_mask].mean()

            hot_spot = {
                "id": region_id,
                "centroid": centroid,
                "area": area,
                "max_intensity": float(max_temp),
                "mean_intensity": float(mean_temp),
                "bbox": (int(x_coords.min()), int(y_coords.min()),
                        int(x_coords.max()), int(y_coords.max())),
            }

            if self.temperature_map is not None:
                hot_spot["max_temperature"] = float(self.temperature_map[region_mask].max())
                hot_spot["mean_temperature"] = float(self.temperature_map[region_mask].mean())

            hot_spots.append(hot_spot)

        if self.verbose:
            print(f"Detected {len(hot_spots)} hot spots")

        return hot_spots

    def extract_temperature_series(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        aggregation: str = "mean",
    ) -> np.ndarray:
        """Extract temperature values from a region.

        Args:
            region: Region of interest as (x1, y1, x2, y2). If None, uses entire image.
            aggregation: Aggregation method ('mean', 'max', 'median').

        Returns:
            Array of temperature values.

        Raises:
            ValueError: If temperature map is not available.

        Example:
            >>> temps = processor.extract_temperature_series(region=(100, 100, 200, 200))
        """
        if self.temperature_map is None:
            if self.verbose:
                print("Warning: No temperature map available, using image intensities")
            data = self.image
        else:
            data = self.temperature_map

        if data is None:
            raise ValueError("No image or temperature data available")

        # Extract region
        if region is not None:
            x1, y1, x2, y2 = region
            region_data = data[y1:y2, x1:x2]
        else:
            region_data = data

        # Aggregate
        if aggregation == "mean":
            return np.array([region_data.mean()])
        elif aggregation == "max":
            return np.array([region_data.max()])
        elif aggregation == "median":
            return np.array([np.median(region_data)])
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def seasonal_decomposition(
        self,
        time_series_data: Union[TimeSeriesData, pd.Series, np.ndarray],
        period: int,
        model: str = "additive",
        extrapolate_trend: str = "freq",
    ) -> DecompositionResult:
        """Perform seasonal decomposition of temperature time series.

        This method decomposes a time series into trend, seasonal, and residual
        components using classical seasonal decomposition.

        Args:
            time_series_data: Time series data to decompose.
            period: Period of the seasonal component.
            model: Type of decomposition ('additive' or 'multiplicative').
            extrapolate_trend: How to extrapolate the trend component.

        Returns:
            DecompositionResult containing trend, seasonal, and residual components.

        Raises:
            ValueError: If time series is too short or period is invalid.

        Example:
            >>> # Daily temperature readings with weekly seasonality
            >>> decomposition = processor.seasonal_decomposition(
            ...     time_series_data=temp_series,
            ...     period=7,  # Weekly pattern
            ...     model='additive'
            ... )
            >>> print(f"Trend: {decomposition.trend[:5]}")
        """
        if period < 2:
            raise ValueError(f"Period must be >= 2, got {period}")

        if self.verbose:
            print(f"Performing seasonal decomposition (period={period}, model={model})...")

        # Convert to pandas Series
        if isinstance(time_series_data, TimeSeriesData):
            series = time_series_data.to_series()
        elif isinstance(time_series_data, np.ndarray):
            series = pd.Series(time_series_data)
        elif isinstance(time_series_data, pd.Series):
            series = time_series_data
        else:
            raise TypeError(
                f"time_series_data must be TimeSeriesData, Series, or ndarray, "
                f"got {type(time_series_data)}"
            )

        if len(series) < 2 * period:
            raise ValueError(
                f"Time series too short for decomposition. "
                f"Need at least {2 * period} observations, got {len(series)}"
            )

        # Perform decomposition
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = seasonal_decompose(
                series,
                model=model,
                period=period,
                extrapolate_trend=extrapolate_trend,
            )

        # Extract components (handle NaN values)
        trend = result.trend.fillna(method="bfill").fillna(method="ffill").tolist()
        seasonal = result.seasonal.fillna(0).tolist()
        residual = result.resid.fillna(0).tolist()

        # Get timestamps
        if isinstance(series.index, pd.DatetimeIndex):
            timestamps = series.index.to_pydatetime().tolist()
        else:
            # Create synthetic timestamps
            timestamps = [datetime(2020, 1, 1) + pd.Timedelta(days=i) for i in range(len(series))]

        if self.verbose:
            print(f"Decomposition complete. Components have {len(trend)} values each.")

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            timestamps=timestamps,
        )

    def trend_analysis(
        self,
        time_series_data: Union[TimeSeriesData, pd.Series, np.ndarray],
        method: str = "linear",
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Analyze the trend component of temperature time series.

        This method fits a trend model to the time series and provides
        statistical analysis of the trend.

        Args:
            time_series_data: Time series data to analyze.
            method: Trend fitting method ('linear', 'polynomial', 'lowess').
            confidence_level: Confidence level for trend intervals.

        Returns:
            Dictionary containing trend analysis results including:
            - slope: Trend slope (for linear trend)
            - intercept: Trend intercept (for linear trend)
            - r_squared: R-squared value
            - p_value: Statistical significance
            - trend_values: Fitted trend values
            - confidence_intervals: Confidence intervals for trend

        Raises:
            ValueError: If method is invalid or data is insufficient.

        Example:
            >>> trend_info = processor.trend_analysis(
            ...     time_series_data=temp_series,
            ...     method='linear'
            ... )
            >>> print(f"Trend slope: {trend_info['slope']:.4f}")
            >>> print(f"R-squared: {trend_info['r_squared']:.4f}")
        """
        if self.verbose:
            print(f"Analyzing trend using {method} method...")

        # Convert to numpy array
        if isinstance(time_series_data, TimeSeriesData):
            values = np.array(time_series_data.values)
        elif isinstance(time_series_data, pd.Series):
            values = time_series_data.values
        elif isinstance(time_series_data, np.ndarray):
            values = time_series_data
        else:
            raise TypeError(
                f"time_series_data must be TimeSeriesData, Series, or ndarray, "
                f"got {type(time_series_data)}"
            )

        n = len(values)
        if n < 3:
            raise ValueError(f"Insufficient data for trend analysis. Need at least 3 points, got {n}")

        x = np.arange(n)

        if method == "linear":
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            trend_values = slope * x + intercept

            # Confidence intervals
            residuals = values - trend_values
            mse = np.sum(residuals**2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((x - x.mean()) ** 2))

            t_stat = stats.t.ppf((1 + confidence_level) / 2, n - 2)
            margin = t_stat * se_slope * np.sqrt(np.sum((x - x.mean()) ** 2))

            result = {
                "method": "linear",
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "trend_values": trend_values.tolist(),
                "confidence_intervals": {
                    "lower": (trend_values - margin).tolist(),
                    "upper": (trend_values + margin).tolist(),
                },
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "is_significant": p_value < (1 - confidence_level),
            }

        elif method == "polynomial":
            # Polynomial regression (degree 2)
            coeffs = np.polyfit(x, values, deg=2)
            trend_values = np.polyval(coeffs, x)

            # R-squared
            ss_res = np.sum((values - trend_values) ** 2)
            ss_tot = np.sum((values - values.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            result = {
                "method": "polynomial",
                "coefficients": coeffs.tolist(),
                "r_squared": float(r_squared),
                "trend_values": trend_values.tolist(),
                "trend_direction": "non-linear",
            }

        elif method == "lowess":
            # LOWESS smoothing
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(values, x, frac=0.3)
            trend_values = smoothed[:, 1]

            # Calculate R-squared
            ss_res = np.sum((values - trend_values) ** 2)
            ss_tot = np.sum((values - values.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            result = {
                "method": "lowess",
                "r_squared": float(r_squared),
                "trend_values": trend_values.tolist(),
                "trend_direction": "non-parametric",
            }

        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear', 'polynomial', or 'lowess'")

        if self.verbose:
            print(f"Trend analysis complete. R² = {result['r_squared']:.4f}")

        return result

    def residual_analysis(
        self,
        time_series_data: Union[TimeSeriesData, pd.Series, np.ndarray],
        decomposition_period: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze residuals from seasonal decomposition or detrending.

        This method provides comprehensive statistical analysis of residuals,
        including normality tests, autocorrelation, and outlier detection.

        Args:
            time_series_data: Time series data to analyze.
            decomposition_period: If provided, performs seasonal decomposition first.

        Returns:
            Dictionary containing residual analysis results including:
            - residuals: Residual values
            - mean: Mean of residuals
            - std: Standard deviation of residuals
            - normality_test: Results of normality test (Shapiro-Wilk)
            - autocorrelation: Lag-1 autocorrelation
            - outliers: Indices of detected outliers
            - statistics: Additional statistical measures

        Raises:
            ValueError: If data is insufficient.

        Example:
            >>> residual_info = processor.residual_analysis(
            ...     time_series_data=temp_series,
            ...     decomposition_period=7
            ... )
            >>> print(f"Residual mean: {residual_info['mean']:.4f}")
            >>> print(f"Number of outliers: {len(residual_info['outliers'])}")
        """
        if self.verbose:
            print("Performing residual analysis...")

        # Convert to numpy array
        if isinstance(time_series_data, TimeSeriesData):
            values = np.array(time_series_data.values)
            series = time_series_data.to_series()
        elif isinstance(time_series_data, pd.Series):
            values = time_series_data.values
            series = time_series_data
        elif isinstance(time_series_data, np.ndarray):
            values = time_series_data
            series = pd.Series(values)
        else:
            raise TypeError(
                f"time_series_data must be TimeSeriesData, Series, or ndarray, "
                f"got {type(time_series_data)}"
            )

        if len(values) < 3:
            raise ValueError(f"Insufficient data for residual analysis. Need at least 3 points")

        # Get residuals
        if decomposition_period is not None:
            # Perform decomposition to extract residuals
            if len(values) < 2 * decomposition_period:
                raise ValueError(
                    f"Time series too short for decomposition with period={decomposition_period}"
                )

            decomp_result = self.seasonal_decomposition(
                series, period=decomposition_period, model="additive"
            )
            residuals = np.array(decomp_result.residual)
        else:
            # Detrend using linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            trend = slope * x + intercept
            residuals = values - trend

        # Remove NaN values
        residuals_clean = residuals[~np.isnan(residuals)]

        if len(residuals_clean) < 3:
            raise ValueError("Too few valid residuals after removing NaN values")

        # Basic statistics
        mean_resid = float(np.mean(residuals_clean))
        std_resid = float(np.std(residuals_clean))
        median_resid = float(np.median(residuals_clean))

        # Normality test (Shapiro-Wilk)
        if len(residuals_clean) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(residuals_clean)
            normality_test = {
                "test": "Shapiro-Wilk",
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_normal": shapiro_p > 0.05,
            }
        else:
            normality_test = {"test": "Shapiro-Wilk", "note": "Insufficient data"}

        # Autocorrelation (lag-1)
        if len(residuals_clean) > 1:
            autocorr = float(np.corrcoef(residuals_clean[:-1], residuals_clean[1:])[0, 1])
        else:
            autocorr = 0.0

        # Outlier detection (using IQR method)
        q1 = np.percentile(residuals_clean, 25)
        q3 = np.percentile(residuals_clean, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()

        # Additional statistics
        skewness = float(stats.skew(residuals_clean))
        kurtosis = float(stats.kurtosis(residuals_clean))

        result = {
            "residuals": residuals.tolist(),
            "mean": mean_resid,
            "std": std_resid,
            "median": median_resid,
            "normality_test": normality_test,
            "autocorrelation_lag1": autocorr,
            "outliers": {
                "indices": outlier_indices,
                "count": len(outlier_indices),
                "percentage": 100 * len(outlier_indices) / len(residuals),
            },
            "statistics": {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min": float(np.min(residuals_clean)),
                "max": float(np.max(residuals_clean)),
                "range": float(np.max(residuals_clean) - np.min(residuals_clean)),
            },
        }

        if self.verbose:
            print(f"Residual analysis complete:")
            print(f"  Mean: {mean_resid:.4f}, Std: {std_resid:.4f}")
            print(f"  Outliers: {len(outlier_indices)} ({result['outliers']['percentage']:.1f}%)")
            print(f"  Normality: {'Yes' if normality_test.get('is_normal', False) else 'No'}")

        return result

    def visualize_decomposition(
        self,
        decomposition: DecompositionResult,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Visualize seasonal decomposition results.

        Args:
            decomposition: DecompositionResult to visualize.
            figsize: Figure size as (width, height).
            save_path: Optional path to save the figure.

        Example:
            >>> decomposition = processor.seasonal_decomposition(temp_series, period=7)
            >>> processor.visualize_decomposition(decomposition, save_path="decomposition.png")
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Original (reconstructed from components)
        original = np.array(decomposition.trend) + np.array(decomposition.seasonal)
        axes[0].plot(decomposition.timestamps, original, color='blue')
        axes[0].set_ylabel('Original')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.timestamps, decomposition.trend, color='orange')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.timestamps, decomposition.seasonal, color='green')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.timestamps, decomposition.residual, color='red')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved decomposition plot to: {save_path}")

        plt.show()

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        if self.image is not None:
            return f"IRImageProcessing(shape={self.image.shape}, has_temp_map={self.temperature_map is not None})"
        return "IRImageProcessing(no image loaded)"
