"""
Utility functions for PV circularity simulator.
"""

from typing import Tuple, Union

import numpy as np
from scipy import stats

from pv_circularity_simulator.core.constants import (
    STANDARD_IRRADIANCE,
    STANDARD_TEMPERATURE,
    TEMP_COEFF_ISC,
    TEMP_COEFF_VOC,
)


def normalize_to_stc(
    value: float,
    current_temp: float,
    current_irradiance: float,
    is_voltage: bool = False,
) -> float:
    """
    Normalize a measurement to Standard Test Conditions (STC).

    Args:
        value: The measured value to normalize
        current_temp: Current temperature in Celsius
        current_irradiance: Current irradiance in W/m²
        is_voltage: True if normalizing voltage, False if normalizing current

    Returns:
        Normalized value at STC conditions

    Examples:
        >>> normalize_to_stc(5.0, 35, 800, is_voltage=False)
        6.3...  # Normalized current
    """
    temp_delta = current_temp - STANDARD_TEMPERATURE
    irradiance_ratio = STANDARD_IRRADIANCE / current_irradiance

    if is_voltage:
        # Voltage: compensate for temperature, minimal irradiance effect
        return value - (temp_delta * TEMP_COEFF_VOC)
    else:
        # Current: compensate for both temperature and irradiance
        temp_correction = 1 + (temp_delta * TEMP_COEFF_ISC)
        return value * irradiance_ratio * temp_correction


def calculate_temperature_uniformity(temperature_matrix: np.ndarray) -> float:
    """
    Calculate temperature uniformity index for a thermal image.

    The uniformity index ranges from 0 (non-uniform) to 1 (perfectly uniform).
    It is calculated as 1 - (std_dev / mean), normalized to [0, 1].

    Args:
        temperature_matrix: 2D array of temperature values

    Returns:
        Uniformity index between 0 and 1

    Examples:
        >>> temps = np.array([[25.0, 25.1], [24.9, 25.0]])
        >>> calculate_temperature_uniformity(temps)
        0.998...
    """
    if temperature_matrix.size == 0:
        return 0.0

    mean_temp = np.mean(temperature_matrix)
    std_temp = np.std(temperature_matrix)

    if mean_temp == 0:
        return 0.0

    # Coefficient of variation: std/mean
    cv = std_temp / abs(mean_temp)

    # Convert to uniformity index (higher is better)
    # Use exponential to map CV to [0, 1] range
    uniformity = np.exp(-cv)

    return float(np.clip(uniformity, 0.0, 1.0))


def detect_outliers_zscore(
    data: np.ndarray, threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Z-score method.

    Args:
        data: 1D array of data points
        threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        Tuple of (outlier_mask, z_scores) where outlier_mask is a boolean array

    Examples:
        >>> data = np.array([1, 2, 2, 3, 2, 1, 2, 50])
        >>> mask, scores = detect_outliers_zscore(data)
        >>> mask[-1]  # Last value is an outlier
        True
    """
    z_scores = np.abs(stats.zscore(data))
    outlier_mask = z_scores > threshold
    return outlier_mask, z_scores


def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Args:
        data: 1D array of data points
        factor: IQR multiplier for outlier bounds (default: 1.5)

    Returns:
        Boolean array indicating outliers

    Examples:
        >>> data = np.array([1, 2, 2, 3, 2, 1, 2, 50])
        >>> mask = detect_outliers_iqr(data)
        >>> mask[-1]  # Last value is an outlier
        True
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate moving average of data.

    Args:
        data: 1D array of data points
        window_size: Size of the moving window

    Returns:
        Smoothed data array

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> moving_average(data, 3)
        array([1. , 2. , 3. , 4. , 5. ])
    """
    if window_size < 1:
        return data

    # Use numpy convolve for moving average
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode="same")


def calculate_curve_quality(
    voltage: np.ndarray, current: np.ndarray, num_points_threshold: int = 50
) -> float:
    """
    Calculate quality score of an IV curve measurement.

    Quality is based on:
    - Number of data points
    - Smoothness of the curve
    - Coverage of voltage range

    Args:
        voltage: Voltage measurements
        current: Current measurements
        num_points_threshold: Minimum desired number of points

    Returns:
        Quality score between 0 and 1

    Examples:
        >>> v = np.linspace(0, 30, 100)
        >>> i = 5 * (1 - v/30)
        >>> calculate_curve_quality(v, i)
        0.9...
    """
    scores = []

    # 1. Number of points score
    num_points = len(voltage)
    points_score = min(num_points / num_points_threshold, 1.0)
    scores.append(points_score)

    # 2. Smoothness score (based on second derivative)
    if num_points > 3:
        second_derivative = np.diff(current, 2)
        smoothness = 1.0 / (1.0 + np.std(second_derivative))
        scores.append(smoothness)

    # 3. Voltage range coverage (should span from ~0 to Voc)
    voltage_range = np.max(voltage) - np.min(voltage)
    expected_range = np.max(voltage)  # Assume max voltage is near Voc
    if expected_range > 0:
        coverage_score = min(voltage_range / expected_range, 1.0)
        scores.append(coverage_score)

    return float(np.mean(scores))


def interpolate_curve(
    voltage: np.ndarray, current: np.ndarray, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate IV curve to a uniform grid.

    Args:
        voltage: Original voltage measurements
        current: Original current measurements
        num_points: Number of points in interpolated curve

    Returns:
        Tuple of (interpolated_voltage, interpolated_current)

    Examples:
        >>> v = np.array([0, 10, 20, 30])
        >>> i = np.array([5, 4, 2, 0])
        >>> v_interp, i_interp = interpolate_curve(v, i, num_points=10)
        >>> len(v_interp)
        10
    """
    # Sort by voltage
    sort_idx = np.argsort(voltage)
    v_sorted = voltage[sort_idx]
    i_sorted = current[sort_idx]

    # Create uniform voltage grid
    v_interp = np.linspace(v_sorted[0], v_sorted[-1], num_points)

    # Interpolate current
    i_interp = np.interp(v_interp, v_sorted, i_sorted)

    return v_interp, i_interp
