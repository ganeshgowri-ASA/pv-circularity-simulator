"""
B14-S04: Utilities & Helpers
Production-ready utility functions for data processing, conversions, and common operations.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from enum import Enum


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================

class EnergyUnit(str, Enum):
    """Energy unit types."""
    WH = "wh"
    KWH = "kwh"
    MWH = "mwh"
    GWH = "gwh"
    J = "j"
    KJ = "kj"
    MJ = "mj"
    GJ = "gj"


class PowerUnit(str, Enum):
    """Power unit types."""
    W = "w"
    KW = "kw"
    MW = "mw"
    GW = "gw"


class CurrencyUnit(str, Enum):
    """Currency units."""
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    CNY = "cny"


# Energy conversion factors to kWh
ENERGY_CONVERSION = {
    EnergyUnit.WH: 0.001,
    EnergyUnit.KWH: 1.0,
    EnergyUnit.MWH: 1000.0,
    EnergyUnit.GWH: 1_000_000.0,
    EnergyUnit.J: 2.77778e-7,
    EnergyUnit.KJ: 2.77778e-4,
    EnergyUnit.MJ: 0.277778,
    EnergyUnit.GJ: 277.778,
}

# Power conversion factors to kW
POWER_CONVERSION = {
    PowerUnit.W: 0.001,
    PowerUnit.KW: 1.0,
    PowerUnit.MW: 1000.0,
    PowerUnit.GW: 1_000_000.0,
}


def convert_energy(value: float, from_unit: Union[str, EnergyUnit],
                   to_unit: Union[str, EnergyUnit]) -> float:
    """
    Convert energy between different units.

    Args:
        value: Energy value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted energy value
    """
    if isinstance(from_unit, str):
        from_unit = EnergyUnit(from_unit.lower())
    if isinstance(to_unit, str):
        to_unit = EnergyUnit(to_unit.lower())

    # Convert to kWh first, then to target unit
    kwh_value = value * ENERGY_CONVERSION[from_unit]
    return kwh_value / ENERGY_CONVERSION[to_unit]


def convert_power(value: float, from_unit: Union[str, PowerUnit],
                  to_unit: Union[str, PowerUnit]) -> float:
    """
    Convert power between different units.

    Args:
        value: Power value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted power value
    """
    if isinstance(from_unit, str):
        from_unit = PowerUnit(from_unit.lower())
    if isinstance(to_unit, str):
        to_unit = PowerUnit(to_unit.lower())

    # Convert to kW first, then to target unit
    kw_value = value * POWER_CONVERSION[from_unit]
    return kw_value / POWER_CONVERSION[to_unit]


def energy_to_power(energy: float, duration_hours: float) -> float:
    """
    Convert energy (kWh) to average power (kW) over a duration.

    Args:
        energy: Energy in kWh
        duration_hours: Duration in hours

    Returns:
        Average power in kW
    """
    if duration_hours <= 0:
        raise ValueError("Duration must be positive")
    return energy / duration_hours


def power_to_energy(power: float, duration_hours: float) -> float:
    """
    Convert power (kW) to energy (kWh) over a duration.

    Args:
        power: Power in kW
        duration_hours: Duration in hours

    Returns:
        Energy in kWh
    """
    if duration_hours < 0:
        raise ValueError("Duration must be non-negative")
    return power * duration_hours


# ============================================================================
# FINANCIAL UTILITIES
# ============================================================================

def present_value(future_value: float, discount_rate: float, periods: int) -> float:
    """
    Calculate present value of a future cash flow.

    Args:
        future_value: Future cash flow value
        discount_rate: Discount rate (e.g., 0.08 for 8%)
        periods: Number of periods

    Returns:
        Present value
    """
    if discount_rate < -1:
        raise ValueError("Discount rate must be >= -1")
    return future_value / ((1 + discount_rate) ** periods)


def future_value(present_value_amt: float, growth_rate: float, periods: int) -> float:
    """
    Calculate future value given present value and growth rate.

    Args:
        present_value_amt: Present value
        growth_rate: Annual growth rate
        periods: Number of periods

    Returns:
        Future value
    """
    return present_value_amt * ((1 + growth_rate) ** periods)


def annuity_payment(principal: float, interest_rate: float, periods: int) -> float:
    """
    Calculate periodic payment for an annuity (loan payment).

    Args:
        principal: Loan principal
        interest_rate: Interest rate per period
        periods: Number of periods

    Returns:
        Payment amount per period
    """
    if interest_rate == 0:
        return principal / periods
    return principal * (interest_rate * (1 + interest_rate) ** periods) / \
           ((1 + interest_rate) ** periods - 1)


def real_to_nominal(real_rate: float, inflation_rate: float) -> float:
    """
    Convert real rate to nominal rate.

    Args:
        real_rate: Real interest/discount rate
        inflation_rate: Inflation rate

    Returns:
        Nominal rate
    """
    return (1 + real_rate) * (1 + inflation_rate) - 1


def nominal_to_real(nominal_rate: float, inflation_rate: float) -> float:
    """
    Convert nominal rate to real rate.

    Args:
        nominal_rate: Nominal interest/discount rate
        inflation_rate: Inflation rate

    Returns:
        Real rate
    """
    return (1 + nominal_rate) / (1 + inflation_rate) - 1


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_statistics(data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a dataset.

    Args:
        data: Input data array

    Returns:
        Dictionary of statistical metrics
    """
    arr = np.array(data)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "range": float(np.max(arr) - np.min(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0,
        "count": len(arr)
    }


def moving_average(data: Union[List[float], np.ndarray], window: int) -> np.ndarray:
    """
    Calculate moving average of data.

    Args:
        data: Input data
        window: Window size for moving average

    Returns:
        Moving average array
    """
    arr = np.array(data)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def exponential_moving_average(data: Union[List[float], np.ndarray],
                               alpha: float) -> np.ndarray:
    """
    Calculate exponential moving average.

    Args:
        data: Input data
        alpha: Smoothing factor (0 < alpha <= 1)

    Returns:
        Exponential moving average
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    arr = np.array(data)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]

    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

    return ema


def percentile_range(data: Union[List[float], np.ndarray],
                     lower: float = 5, upper: float = 95) -> Tuple[float, float]:
    """
    Calculate percentile range of data.

    Args:
        data: Input data
        lower: Lower percentile (default 5%)
        upper: Upper percentile (default 95%)

    Returns:
        Tuple of (lower_value, upper_value)
    """
    arr = np.array(data)
    return (float(np.percentile(arr, lower)),
            float(np.percentile(arr, upper)))


# ============================================================================
# TIME SERIES UTILITIES
# ============================================================================

def generate_time_series(start_date: datetime, end_date: datetime,
                         frequency: str = "1H") -> pd.DatetimeIndex:
    """
    Generate time series index.

    Args:
        start_date: Start datetime
        end_date: End datetime
        frequency: Pandas frequency string (e.g., '1H', '15min', '1D')

    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=frequency)


def resample_timeseries(data: pd.Series, target_frequency: str,
                        method: str = "mean") -> pd.Series:
    """
    Resample time series data.

    Args:
        data: Input time series (must have DatetimeIndex)
        target_frequency: Target frequency
        method: Aggregation method ('mean', 'sum', 'min', 'max')

    Returns:
        Resampled series
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    resampler = data.resample(target_frequency)

    if method == "mean":
        return resampler.mean()
    elif method == "sum":
        return resampler.sum()
    elif method == "min":
        return resampler.min()
    elif method == "max":
        return resampler.max()
    else:
        raise ValueError(f"Unknown method: {method}")


def fill_missing_values(data: pd.Series, method: str = "interpolate") -> pd.Series:
    """
    Fill missing values in time series.

    Args:
        data: Input series with potential NaN values
        method: Fill method ('interpolate', 'forward', 'backward', 'mean')

    Returns:
        Series with filled values
    """
    if method == "interpolate":
        return data.interpolate(method='time')
    elif method == "forward":
        return data.fillna(method='ffill')
    elif method == "backward":
        return data.fillna(method='bfill')
    elif method == "mean":
        return data.fillna(data.mean())
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# DATA EXPORT UTILITIES
# ============================================================================

def export_to_json(data: Any, file_path: Union[str, Path],
                   indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Export data to JSON file.

    Args:
        data: Data to export (must be JSON-serializable)
        file_path: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)


def export_to_csv(df: pd.DataFrame, file_path: Union[str, Path],
                  **kwargs) -> None:
    """
    Export DataFrame to CSV file.

    Args:
        df: DataFrame to export
        file_path: Output file path
        **kwargs: Additional arguments for pd.to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, **kwargs)


def export_to_excel(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    file_path: Union[str, Path],
                    sheet_name: str = "Sheet1") -> None:
    """
    Export data to Excel file.

    Args:
        data: DataFrame or dict of DataFrames
        file_path: Output file path
        sheet_name: Sheet name (if single DataFrame)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name)
    else:
        data.to_excel(file_path, sheet_name=sheet_name)


# ============================================================================
# ARRAY PROCESSING UTILITIES
# ============================================================================

def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize array values.

    Args:
        arr: Input array
        method: Normalization method ('minmax', 'zscore', 'l2')

    Returns:
        Normalized array
    """
    if method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std

    elif method == "l2":
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_data(data: np.ndarray, window_size: int = 5,
                method: str = "moving_average") -> np.ndarray:
    """
    Smooth data using various methods.

    Args:
        data: Input data
        window_size: Window size for smoothing
        method: Smoothing method

    Returns:
        Smoothed data
    """
    if method == "moving_average":
        return moving_average(data, window_size)
    elif method == "exponential":
        alpha = 2 / (window_size + 1)
        return exponential_moving_average(data, alpha)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_outliers(data: np.ndarray, method: str = "iqr",
                    threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data.

    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean array indicating outliers
    """
    if method == "iqr":
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        return (data < lower) | (data > upper)

    elif method == "zscore":
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_range(value: float, min_val: Optional[float] = None,
                   max_val: Optional[float] = None, name: str = "value") -> None:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of value for error message

    Raises:
        ValueError: If validation fails
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


def validate_positive(value: float, name: str = "value", strict: bool = True) -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        name: Name of value for error message
        strict: If True, must be > 0; if False, must be >= 0

    Raises:
        ValueError: If validation fails
    """
    if strict and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    elif not strict and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability [0, 1].

    Args:
        value: Value to validate
        name: Name of value for error message

    Raises:
        ValueError: If validation fails
    """
    validate_range(value, 0.0, 1.0, name)


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_currency(value: float, currency: str = "USD",
                    decimals: int = 2) -> str:
    """
    Format value as currency.

    Args:
        value: Numeric value
        currency: Currency code
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "CNY": "¥"}
    symbol = symbols.get(currency.upper(), currency)

    if abs(value) >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.{decimals}f}K"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Value (e.g., 0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float, decimals: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


__all__ = [
    # Unit Conversions
    "EnergyUnit",
    "PowerUnit",
    "CurrencyUnit",
    "convert_energy",
    "convert_power",
    "energy_to_power",
    "power_to_energy",
    # Financial
    "present_value",
    "future_value",
    "annuity_payment",
    "real_to_nominal",
    "nominal_to_real",
    # Statistics
    "calculate_statistics",
    "moving_average",
    "exponential_moving_average",
    "percentile_range",
    # Time Series
    "generate_time_series",
    "resample_timeseries",
    "fill_missing_values",
    # Export
    "export_to_json",
    "export_to_csv",
    "export_to_excel",
    # Array Processing
    "normalize_array",
    "smooth_data",
    "detect_outliers",
    # Validation
    "validate_range",
    "validate_positive",
    "validate_probability",
    # Formatting
    "format_currency",
    "format_percentage",
    "format_large_number",
]
