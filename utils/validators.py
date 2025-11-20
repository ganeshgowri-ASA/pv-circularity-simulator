"""
Validators Module
=================
Comprehensive validation functions for PV Circularity Simulator.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
from datetime import datetime
import pandas as pd
import numpy as np

from .constants import VALIDATION_RANGES, ERROR_MESSAGES


# ============================================================================
# NUMERIC VALIDATORS
# ============================================================================

def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    parameter_name: str = "Value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        parameter_name: Name of parameter for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{parameter_name} must be a number"

    if np.isnan(value) or np.isinf(value):
        return False, f"{parameter_name} must be a finite number"

    if value < min_val or value > max_val:
        return False, f"{parameter_name} must be between {min_val} and {max_val}, got {value}"

    return True, None


def validate_positive(value: float, parameter_name: str = "Value") -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        parameter_name: Name of parameter

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{parameter_name} must be a number"

    if value <= 0:
        return False, f"{parameter_name} must be positive, got {value}"

    return True, None


def validate_non_negative(value: float, parameter_name: str = "Value") -> Tuple[bool, Optional[str]]:
    """Validate that a value is non-negative."""
    if not isinstance(value, (int, float)):
        return False, f"{parameter_name} must be a number"

    if value < 0:
        return False, f"{parameter_name} must be non-negative, got {value}"

    return True, None


def validate_percentage(value: float, parameter_name: str = "Percentage") -> Tuple[bool, Optional[str]]:
    """Validate that a value is a valid percentage (0-100)."""
    return validate_range(value, 0, 100, parameter_name)


# ============================================================================
# PV SYSTEM VALIDATORS
# ============================================================================

def validate_efficiency(efficiency: float) -> Tuple[bool, Optional[str]]:
    """Validate solar cell/module efficiency."""
    ranges = VALIDATION_RANGES['efficiency']
    return validate_range(efficiency, ranges['min'], ranges['max'], "Efficiency")


def validate_performance_ratio(pr: float) -> Tuple[bool, Optional[str]]:
    """Validate performance ratio."""
    ranges = VALIDATION_RANGES['performance_ratio']
    return validate_range(pr, ranges['min'], ranges['max'], "Performance Ratio")


def validate_irradiance(irradiance: float) -> Tuple[bool, Optional[str]]:
    """Validate irradiance value."""
    ranges = VALIDATION_RANGES['irradiance']
    return validate_range(irradiance, ranges['min'], ranges['max'], "Irradiance")


def validate_temperature(temp: float, temp_type: str = "ambient") -> Tuple[bool, Optional[str]]:
    """
    Validate temperature value.

    Args:
        temp: Temperature in Celsius
        temp_type: 'ambient' or 'module'

    Returns:
        Tuple of (is_valid, error_message)
    """
    if temp_type == "ambient":
        ranges = VALIDATION_RANGES['ambient_temp']
    elif temp_type == "module":
        ranges = VALIDATION_RANGES['module_temp']
    else:
        return False, f"Unknown temperature type: {temp_type}"

    return validate_range(temp, ranges['min'], ranges['max'], f"{temp_type.capitalize()} Temperature")


def validate_power(power: float, unit: str = "kW") -> Tuple[bool, Optional[str]]:
    """Validate power value."""
    ranges = VALIDATION_RANGES['power']
    return validate_range(power, ranges['min'], ranges['max'], f"Power ({unit})")


def validate_voltage(voltage: float) -> Tuple[bool, Optional[str]]:
    """Validate voltage value."""
    ranges = VALIDATION_RANGES['voltage']
    return validate_range(voltage, ranges['min'], ranges['max'], "Voltage")


def validate_current(current: float) -> Tuple[bool, Optional[str]]:
    """Validate current value."""
    ranges = VALIDATION_RANGES['current']
    return validate_range(current, ranges['min'], ranges['max'], "Current")


def validate_system_capacity(capacity_kw: float) -> Tuple[bool, Optional[str]]:
    """Validate system capacity."""
    is_valid, msg = validate_positive(capacity_kw, "System Capacity")
    if not is_valid:
        return is_valid, msg

    if capacity_kw > 100000:  # 100 MW
        return False, "System capacity exceeds maximum allowed (100 MW)"

    return True, None


def validate_module_configuration(
    modules_per_string: int,
    num_strings: int,
    total_modules: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate module configuration consistency.

    Args:
        modules_per_string: Modules in series per string
        num_strings: Number of parallel strings
        total_modules: Total module count

    Returns:
        Tuple of (is_valid, error_message)
    """
    if modules_per_string <= 0:
        return False, "Modules per string must be positive"

    if num_strings <= 0:
        return False, "Number of strings must be positive"

    expected_total = modules_per_string * num_strings
    if total_modules != expected_total:
        return False, f"Module count mismatch: {total_modules} != {modules_per_string} × {num_strings} = {expected_total}"

    return True, None


# ============================================================================
# WEATHER DATA VALIDATORS
# ============================================================================

def validate_weather_data(weather_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate weather data DataFrame.

    Args:
        weather_df: Weather data DataFrame

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    required_columns = ['timestamp', 'ghi', 'ambient_temp']
    missing_columns = [col for col in required_columns if col not in weather_df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors

    # Validate data types
    if not pd.api.types.is_datetime64_any_dtype(weather_df['timestamp']):
        errors.append("Timestamp column must be datetime type")

    # Validate GHI values
    if 'ghi' in weather_df.columns:
        invalid_ghi = weather_df[(weather_df['ghi'] < 0) | (weather_df['ghi'] > 1500)]
        if len(invalid_ghi) > 0:
            errors.append(f"Found {len(invalid_ghi)} invalid GHI values (must be 0-1500 W/m²)")

    # Validate temperature values
    if 'ambient_temp' in weather_df.columns:
        invalid_temp = weather_df[(weather_df['ambient_temp'] < -50) | (weather_df['ambient_temp'] > 60)]
        if len(invalid_temp) > 0:
            errors.append(f"Found {len(invalid_temp)} invalid temperature values (must be -50 to 60°C)")

    # Check for missing data
    missing_data_pct = (weather_df.isnull().sum() / len(weather_df) * 100)
    high_missing = missing_data_pct[missing_data_pct > 10]
    if len(high_missing) > 0:
        errors.append(f"High missing data in columns: {dict(high_missing)}")

    return len(errors) == 0, errors


def validate_location(latitude: float, longitude: float) -> Tuple[bool, Optional[str]]:
    """
    Validate geographic coordinates.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees

    Returns:
        Tuple of (is_valid, error_message)
    """
    lat_valid, lat_msg = validate_range(latitude, -90, 90, "Latitude")
    if not lat_valid:
        return lat_valid, lat_msg

    lon_valid, lon_msg = validate_range(longitude, -180, 180, "Longitude")
    if not lon_valid:
        return lon_valid, lon_msg

    return True, None


# ============================================================================
# FINANCIAL VALIDATORS
# ============================================================================

def validate_financial_parameters(
    capex: float,
    opex: float,
    electricity_price: float,
    discount_rate: float,
    project_lifetime: int
) -> Tuple[bool, List[str]]:
    """
    Validate financial parameters.

    Args:
        capex: Capital expenditure
        opex: Operating expenditure
        electricity_price: Electricity price
        discount_rate: Discount rate
        project_lifetime: Project lifetime in years

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Validate CAPEX
    is_valid, msg = validate_positive(capex, "CAPEX")
    if not is_valid:
        errors.append(msg)

    # Validate OPEX
    is_valid, msg = validate_non_negative(opex, "OPEX")
    if not is_valid:
        errors.append(msg)

    # Validate electricity price
    is_valid, msg = validate_positive(electricity_price, "Electricity Price")
    if not is_valid:
        errors.append(msg)

    # Validate discount rate
    is_valid, msg = validate_range(discount_rate, 0, 30, "Discount Rate")
    if not is_valid:
        errors.append(msg)

    # Validate project lifetime
    if not isinstance(project_lifetime, int):
        errors.append("Project lifetime must be an integer")
    elif project_lifetime < 1 or project_lifetime > 50:
        errors.append("Project lifetime must be between 1 and 50 years")

    return len(errors) == 0, errors


def validate_irr(irr: float) -> Tuple[bool, Optional[str]]:
    """Validate Internal Rate of Return."""
    if not isinstance(irr, (int, float)):
        return False, "IRR must be a number"

    if irr < -100 or irr > 1000:
        return False, f"IRR value seems unrealistic: {irr}%"

    return True, None


def validate_dscr(dscr: float) -> Tuple[bool, Optional[str]]:
    """Validate Debt Service Coverage Ratio."""
    is_valid, msg = validate_non_negative(dscr, "DSCR")
    if not is_valid:
        return is_valid, msg

    if dscr < 1.0:
        return False, f"Warning: DSCR below 1.0 ({dscr:.2f}) indicates insufficient cash flow"

    return True, None


# ============================================================================
# DATA QUALITY VALIDATORS
# ============================================================================

def validate_data_completeness(
    data: pd.DataFrame,
    required_columns: List[str],
    max_missing_pct: float = 5.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate data completeness.

    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        max_missing_pct: Maximum allowed missing data percentage

    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        'total_rows': len(data),
        'missing_columns': [],
        'completeness_by_column': {},
        'overall_completeness': 0.0
    }

    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    report['missing_columns'] = missing_cols

    if missing_cols:
        return False, report

    # Calculate completeness for each column
    for col in required_columns:
        completeness = (1 - data[col].isnull().sum() / len(data)) * 100
        report['completeness_by_column'][col] = completeness

    # Calculate overall completeness
    report['overall_completeness'] = np.mean(list(report['completeness_by_column'].values()))

    # Check if completeness meets threshold
    is_valid = report['overall_completeness'] >= (100 - max_missing_pct)

    return is_valid, report


def validate_data_consistency(
    current_value: float,
    historical_values: List[float],
    std_dev_threshold: float = 3.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate data consistency using statistical methods.

    Args:
        current_value: Current value to validate
        historical_values: Historical values for comparison
        std_dev_threshold: Number of standard deviations for outlier detection

    Returns:
        Tuple of (is_valid, warning_message)
    """
    if not historical_values:
        return True, None

    mean = np.mean(historical_values)
    std = np.std(historical_values)

    if std == 0:
        return True, None

    z_score = abs((current_value - mean) / std)

    if z_score > std_dev_threshold:
        return False, f"Value {current_value} is {z_score:.1f} standard deviations from mean ({mean:.2f})"

    return True, None


# ============================================================================
# TIME SERIES VALIDATORS
# ============================================================================

def validate_time_series(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    check_monotonic: bool = True,
    check_gaps: bool = True,
    max_gap_minutes: int = 60
) -> Tuple[bool, List[str]]:
    """
    Validate time series data.

    Args:
        df: DataFrame with time series data
        timestamp_col: Name of timestamp column
        check_monotonic: Check if timestamps are monotonically increasing
        check_gaps: Check for data gaps
        max_gap_minutes: Maximum allowed gap in minutes

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if timestamp_col not in df.columns:
        issues.append(f"Timestamp column '{timestamp_col}' not found")
        return False, issues

    # Check if column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        issues.append(f"Column '{timestamp_col}' must be datetime type")
        return False, issues

    # Check monotonic
    if check_monotonic and not df[timestamp_col].is_monotonic_increasing:
        issues.append("Timestamps are not monotonically increasing")

    # Check for duplicates
    duplicates = df[timestamp_col].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate timestamps")

    # Check for gaps
    if check_gaps and len(df) > 1:
        time_diff = df[timestamp_col].diff()
        max_diff = time_diff.max()

        if pd.notnull(max_diff):
            max_diff_minutes = max_diff.total_seconds() / 60
            if max_diff_minutes > max_gap_minutes:
                issues.append(f"Found data gap of {max_diff_minutes:.1f} minutes (max allowed: {max_gap_minutes})")

    return len(issues) == 0, issues


# ============================================================================
# I-V CURVE VALIDATORS
# ============================================================================

def validate_iv_curve(
    voltage: np.ndarray,
    current: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Validate I-V curve data.

    Args:
        voltage: Voltage array
        current: Current array

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check array lengths
    if len(voltage) != len(current):
        errors.append(f"Voltage and current arrays must have same length ({len(voltage)} != {len(current)})")
        return False, errors

    # Check for minimum number of points
    if len(voltage) < 10:
        errors.append(f"I-V curve must have at least 10 points, got {len(voltage)}")

    # Check voltage range (should start near 0 and end near Voc)
    if voltage[0] < 0:
        errors.append("First voltage point should be non-negative")

    if not np.all(np.diff(voltage) >= 0):
        errors.append("Voltage array should be monotonically increasing")

    # Check current range (should start near Isc and decrease to 0)
    if current[0] < 0:
        errors.append("First current point (Isc) should be positive")

    if current[-1] > current[0] * 0.1:
        errors.append("Last current point should be near zero")

    # Calculate and validate fill factor
    voc = voltage[np.argmin(np.abs(current))]
    isc = current[0]
    power = voltage * current
    max_power = np.max(power)
    fill_factor = max_power / (voc * isc) if (voc * isc) > 0 else 0

    if fill_factor < 0.5 or fill_factor > 0.9:
        errors.append(f"Fill factor out of reasonable range: {fill_factor:.3f}")

    return len(errors) == 0, errors


# ============================================================================
# STRING VALIDATORS
# ============================================================================

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True, None
    return False, "Invalid email format"


def validate_project_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate project name."""
    if not name or len(name.strip()) == 0:
        return False, "Project name cannot be empty"

    if len(name) > 100:
        return False, "Project name too long (max 100 characters)"

    return True, None


# ============================================================================
# BATCH VALIDATOR
# ============================================================================

class ValidationReport:
    """Comprehensive validation report."""

    def __init__(self):
        """Initialize validation report."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add info message."""
        self.info.append(message)

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'valid': self.is_valid(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info),
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }

    def __str__(self) -> str:
        """String representation of report."""
        lines = ["Validation Report", "=" * 50]

        if self.errors:
            lines.append(f"\nERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")

        if self.warnings:
            lines.append(f"\nWARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        if self.info:
            lines.append(f"\nINFO ({len(self.info)}):")
            for i, info in enumerate(self.info, 1):
                lines.append(f"  {i}. {info}")

        lines.append(f"\nResult: {'PASS' if self.is_valid() else 'FAIL'}")
        return "\n".join(lines)


# Export main validators
__all__ = [
    'validate_range',
    'validate_positive',
    'validate_non_negative',
    'validate_percentage',
    'validate_efficiency',
    'validate_performance_ratio',
    'validate_irradiance',
    'validate_temperature',
    'validate_power',
    'validate_voltage',
    'validate_current',
    'validate_system_capacity',
    'validate_module_configuration',
    'validate_weather_data',
    'validate_location',
    'validate_financial_parameters',
    'validate_irr',
    'validate_dscr',
    'validate_data_completeness',
    'validate_data_consistency',
    'validate_time_series',
    'validate_iv_curve',
    'validate_email',
    'validate_project_name',
    'ValidationReport'
]
