"""Example usage of IRImageProcessing.

This example demonstrates how to use the IRImageProcessing class
for thermal image analysis and time-series decomposition.
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from pv_circularity.processing import IRImageProcessing
from pv_circularity.utils.validators import TimeSeriesData


def create_synthetic_ir_image(
    size: tuple = (512, 512),
    n_hot_spots: int = 5,
) -> tuple:
    """Create synthetic IR thermal image with hot spots.

    Args:
        size: Image dimensions (height, width).
        n_hot_spots: Number of hot spots to create.

    Returns:
        Tuple of (image, temperature_map).
    """
    np.random.seed(42)
    h, w = size

    # Base temperature map (normal operating temperature)
    temperature_map = np.random.randn(h, w) * 3 + 40  # ~40°C mean

    # Add hot spots (defective cells)
    for i in range(n_hot_spots):
        # Random location
        cy = np.random.randint(50, h - 50)
        cx = np.random.randint(50, w - 50)

        # Random size
        radius = np.random.randint(10, 30)

        # Create hot spot
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        # Hot spot temperature (70-90°C)
        hot_temp = np.random.uniform(70, 90)
        temperature_map[mask] = hot_temp

    # Convert to grayscale image (0-255)
    temp_min, temp_max = temperature_map.min(), temperature_map.max()
    image = ((temperature_map - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)

    return image, temperature_map


def create_temperature_time_series(n_days: int = 90) -> TimeSeriesData:
    """Create synthetic temperature time series.

    Args:
        n_days: Number of days of data.

    Returns:
        TimeSeriesData with temperature measurements.
    """
    np.random.seed(42)
    timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Create realistic temperature pattern
    t = np.arange(n_days)

    # Long-term trend (degradation causing increased temperature)
    trend = 0.05 * t  # Gradual temperature increase

    # Weekly seasonality (ambient temperature variation)
    seasonal = 5 * np.sin(2 * np.pi * t / 7)

    # Random fluctuations
    noise = np.random.randn(n_days) * 2

    # Base temperature + components
    values = (45 + trend + seasonal + noise).tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        frequency="D",
        name="module_temperature_celsius",
    )


def example_hot_spot_detection():
    """Example: Detecting hot spots in IR images."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Hot Spot Detection")
    print("=" * 80)

    # Create synthetic IR image
    image, temperature_map = create_synthetic_ir_image(size=(512, 512), n_hot_spots=5)

    # Initialize processor
    processor = IRImageProcessing(
        image=image,
        temperature_map=temperature_map,
        metadata={"timestamp": datetime.now(), "module_id": "PV-001"},
        verbose=True,
    )

    print(f"\nImage shape: {image.shape}")
    print(f"Temperature range: {temperature_map.min():.1f}°C - {temperature_map.max():.1f}°C")

    # Detect hot spots
    hot_spots = processor.detect_hot_spots(
        threshold_percentile=95,
        min_area=50,
        gaussian_sigma=2.0,
    )

    print(f"\nDetected {len(hot_spots)} hot spots:")
    for i, spot in enumerate(hot_spots[:5], 1):  # Show first 5
        print(f"\nHot Spot {i}:")
        print(f"  Centroid: {spot['centroid']}")
        print(f"  Area: {spot['area']} pixels")
        print(f"  Max Temperature: {spot['max_temperature']:.1f}°C")
        print(f"  Mean Temperature: {spot['mean_temperature']:.1f}°C")
        print(f"  Bounding Box: {spot['bbox']}")

    # Visualize
    visualize_hot_spots(image, hot_spots)

    return processor, hot_spots


def visualize_hot_spots(image: np.ndarray, hot_spots: list):
    """Visualize detected hot spots.

    Args:
        image: IR thermal image.
        hot_spots: List of detected hot spots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original image
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original IR Image")
    ax1.axis("off")

    # Image with hot spots marked
    ax2.imshow(image, cmap="hot")

    # Draw bounding boxes
    for spot in hot_spots:
        x1, y1, x2, y2 = spot["bbox"]
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
        ax2.add_patch(rect)

        # Mark centroid
        cx, cy = spot["centroid"]
        ax2.plot(cx, cy, "c+", markersize=10, markeredgewidth=2)

    ax2.set_title(f"Detected Hot Spots ({len(hot_spots)})")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("hot_spots_detection.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to: hot_spots_detection.png")
    plt.close()


def example_seasonal_decomposition():
    """Example: Seasonal decomposition of temperature data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Seasonal Decomposition")
    print("=" * 80)

    # Create synthetic IR image (required for IRImageProcessing)
    image, _ = create_synthetic_ir_image()
    processor = IRImageProcessing(image=image, verbose=True)

    # Create temperature time series
    temp_series = create_temperature_time_series(n_days=90)

    print(f"\nTime series length: {len(temp_series.values)} days")
    print(f"Temperature range: {min(temp_series.values):.1f}°C - {max(temp_series.values):.1f}°C")

    # Perform seasonal decomposition (weekly pattern)
    decomposition = processor.seasonal_decomposition(
        time_series_data=temp_series,
        period=7,  # Weekly seasonality
        model="additive",
    )

    print("\nDecomposition complete!")
    print(f"  Trend component length: {len(decomposition.trend)}")
    print(f"  Seasonal component length: {len(decomposition.seasonal)}")
    print(f"  Residual component length: {len(decomposition.residual)}")

    # Visualize decomposition
    processor.visualize_decomposition(
        decomposition,
        figsize=(12, 10),
        save_path="temperature_decomposition.png",
    )

    return processor, decomposition


def example_trend_analysis():
    """Example: Trend analysis of temperature data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Trend Analysis")
    print("=" * 80)

    # Create synthetic IR image
    image, _ = create_synthetic_ir_image()
    processor = IRImageProcessing(image=image, verbose=True)

    # Create temperature time series
    temp_series = create_temperature_time_series(n_days=90)

    # Linear trend analysis
    print("\n--- Linear Trend Analysis ---")
    linear_trend = processor.trend_analysis(
        time_series_data=temp_series,
        method="linear",
        confidence_level=0.95,
    )

    print(f"Slope: {linear_trend['slope']:.4f}°C/day")
    print(f"Intercept: {linear_trend['intercept']:.2f}°C")
    print(f"R-squared: {linear_trend['r_squared']:.4f}")
    print(f"P-value: {linear_trend['p_value']:.4e}")
    print(f"Trend direction: {linear_trend['trend_direction']}")
    print(f"Statistically significant: {linear_trend['is_significant']}")

    # Polynomial trend analysis
    print("\n--- Polynomial Trend Analysis ---")
    poly_trend = processor.trend_analysis(
        time_series_data=temp_series,
        method="polynomial",
    )

    print(f"Coefficients: {poly_trend['coefficients']}")
    print(f"R-squared: {poly_trend['r_squared']:.4f}")

    # LOWESS trend analysis
    print("\n--- LOWESS Trend Analysis ---")
    lowess_trend = processor.trend_analysis(
        time_series_data=temp_series,
        method="lowess",
    )

    print(f"R-squared: {lowess_trend['r_squared']:.4f}")

    # Visualize trends
    visualize_trend_analysis(temp_series, linear_trend, poly_trend, lowess_trend)

    return processor, linear_trend


def visualize_trend_analysis(
    data: TimeSeriesData,
    linear_trend: dict,
    poly_trend: dict,
    lowess_trend: dict,
):
    """Visualize trend analysis results.

    Args:
        data: Original time series data.
        linear_trend: Linear trend results.
        poly_trend: Polynomial trend results.
        lowess_trend: LOWESS trend results.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Linear trend
    axes[0].plot(data.timestamps, data.values, "o", alpha=0.5, label="Data")
    axes[0].plot(data.timestamps, linear_trend["trend_values"], "r-", linewidth=2, label="Linear Trend")
    if "confidence_intervals" in linear_trend:
        axes[0].fill_between(
            data.timestamps,
            linear_trend["confidence_intervals"]["lower"],
            linear_trend["confidence_intervals"]["upper"],
            alpha=0.3,
            color="red",
            label="95% CI",
        )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Linear Trend (R² = {linear_trend['r_squared']:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Polynomial trend
    axes[1].plot(data.timestamps, data.values, "o", alpha=0.5, label="Data")
    axes[1].plot(data.timestamps, poly_trend["trend_values"], "g-", linewidth=2, label="Polynomial Trend")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title(f"Polynomial Trend (R² = {poly_trend['r_squared']:.4f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LOWESS trend
    axes[2].plot(data.timestamps, data.values, "o", alpha=0.5, label="Data")
    axes[2].plot(data.timestamps, lowess_trend["trend_values"], "b-", linewidth=2, label="LOWESS Trend")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Date")
    axes[2].set_title(f"LOWESS Trend (R² = {lowess_trend['r_squared']:.4f})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trend_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved trend analysis to: trend_analysis.png")
    plt.close()


def example_residual_analysis():
    """Example: Residual analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Residual Analysis")
    print("=" * 80)

    # Create synthetic IR image
    image, _ = create_synthetic_ir_image()
    processor = IRImageProcessing(image=image, verbose=True)

    # Create temperature time series
    temp_series = create_temperature_time_series(n_days=90)

    # Perform residual analysis with decomposition
    residuals = processor.residual_analysis(
        time_series_data=temp_series,
        decomposition_period=7,  # Weekly pattern
    )

    print("\nResidual Statistics:")
    print(f"  Mean: {residuals['mean']:.4f}°C")
    print(f"  Std Dev: {residuals['std']:.4f}°C")
    print(f"  Median: {residuals['median']:.4f}°C")

    print("\nNormality Test:")
    normality = residuals["normality_test"]
    if "p_value" in normality:
        print(f"  Test: {normality['test']}")
        print(f"  P-value: {normality['p_value']:.4f}")
        print(f"  Is Normal: {normality['is_normal']}")

    print(f"\nAutocorrelation (lag-1): {residuals['autocorrelation_lag1']:.4f}")

    print("\nOutliers:")
    outliers = residuals["outliers"]
    print(f"  Count: {outliers['count']}")
    print(f"  Percentage: {outliers['percentage']:.2f}%")

    print("\nAdditional Statistics:")
    stats = residuals["statistics"]
    print(f"  Skewness: {stats['skewness']:.4f}")
    print(f"  Kurtosis: {stats['kurtosis']:.4f}")
    print(f"  Range: {stats['range']:.4f}°C")

    # Visualize residuals
    visualize_residuals(temp_series, residuals)

    return processor, residuals


def visualize_residuals(data: TimeSeriesData, residuals: dict):
    """Visualize residual analysis results.

    Args:
        data: Original time series data.
        residuals: Residual analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plot
    axes[0, 0].plot(data.timestamps, residuals["residuals"], "b-", alpha=0.7)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Residual (°C)")
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(residuals["residuals"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(x=residuals["mean"], color="r", linestyle="--", label=f"Mean: {residuals['mean']:.2f}")
    axes[0, 1].set_xlabel("Residual (°C)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats as sp_stats
    residuals_clean = [r for r in residuals["residuals"] if not np.isnan(r)]
    sp_stats.probplot(residuals_clean, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # ACF plot (simple version)
    lags = range(1, min(21, len(residuals_clean)))
    acf_values = [
        np.corrcoef(residuals_clean[:-lag], residuals_clean[lag:])[0, 1] if lag < len(residuals_clean) else 0
        for lag in lags
    ]
    axes[1, 1].bar(lags, acf_values, alpha=0.7)
    axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].set_title("Autocorrelation Function")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("residual_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved residual analysis to: residual_analysis.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("PV CIRCULARITY SIMULATOR - IR IMAGE PROCESSING EXAMPLES")
    print("=" * 80)

    # Run examples
    example_hot_spot_detection()
    example_seasonal_decomposition()
    example_trend_analysis()
    example_residual_analysis()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
