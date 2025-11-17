"""
Validators Utilities
====================

Input validation and data quality checks.
"""

from typing import Any, List, Tuple, Optional, Union
import re
from datetime import datetime
import pandas as pd


def validate_latitude(lat: float) -> Tuple[bool, str]:
    """
    Validate latitude value.

    Args:
        lat: Latitude in degrees

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(lat, (int, float)):
        return False, "Latitude must be a number"

    if lat < -90 or lat > 90:
        return False, "Latitude must be between -90 and 90 degrees"

    return True, ""


def validate_longitude(lon: float) -> Tuple[bool, str]:
    """
    Validate longitude value.

    Args:
        lon: Longitude in degrees

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(lon, (int, float)):
        return False, "Longitude must be a number"

    if lon < -180 or lon > 180:
        return False, "Longitude must be between -180 and 180 degrees"

    return True, ""


def validate_angle(angle: float, min_val: float = 0, max_val: float = 360) -> Tuple[bool, str]:
    """
    Validate angle value.

    Args:
        angle: Angle in degrees
        min_val: Minimum valid angle
        max_val: Maximum valid angle

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(angle, (int, float)):
        return False, "Angle must be a number"

    if angle < min_val or angle > max_val:
        return False, f"Angle must be between {min_val} and {max_val} degrees"

    return True, ""


def validate_efficiency(efficiency: float) -> Tuple[bool, str]:
    """
    Validate efficiency percentage.

    Args:
        efficiency: Efficiency value (%)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(efficiency, (int, float)):
        return False, "Efficiency must be a number"

    if efficiency < 0 or efficiency > 100:
        return False, "Efficiency must be between 0 and 100%"

    # Check for reasonable PV efficiency ranges
    if efficiency > 30:
        return False, "Efficiency exceeds current technological limits (>30%)"

    if efficiency < 5:
        return False, "Efficiency is unreasonably low (<5%)"

    return True, ""


def validate_power(power: float, min_power: float = 0) -> Tuple[bool, str]:
    """
    Validate power value.

    Args:
        power: Power in Watts
        min_power: Minimum valid power

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(power, (int, float)):
        return False, "Power must be a number"

    if power < min_power:
        return False, f"Power must be greater than {min_power} W"

    return True, ""


def validate_voltage(voltage: float, min_voltage: float = 0, max_voltage: float = 1500) -> Tuple[bool, str]:
    """
    Validate voltage value.

    Args:
        voltage: Voltage in Volts
        min_voltage: Minimum valid voltage
        max_voltage: Maximum valid voltage

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(voltage, (int, float)):
        return False, "Voltage must be a number"

    if voltage < min_voltage or voltage > max_voltage:
        return False, f"Voltage must be between {min_voltage} and {max_voltage} V"

    return True, ""


def validate_current(current: float, min_current: float = 0) -> Tuple[bool, str]:
    """
    Validate current value.

    Args:
        current: Current in Amperes
        min_current: Minimum valid current

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(current, (int, float)):
        return False, "Current must be a number"

    if current < min_current:
        return False, f"Current must be greater than {min_current} A"

    return True, ""


def validate_temperature(temp: float, min_temp: float = -50, max_temp: float = 100) -> Tuple[bool, str]:
    """
    Validate temperature value.

    Args:
        temp: Temperature in Celsius
        min_temp: Minimum valid temperature
        max_temp: Maximum valid temperature

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(temp, (int, float)):
        return False, "Temperature must be a number"

    if temp < min_temp or temp > max_temp:
        return False, f"Temperature must be between {min_temp} and {max_temp} °C"

    return True, ""


def validate_irradiance(irradiance: float) -> Tuple[bool, str]:
    """
    Validate irradiance value.

    Args:
        irradiance: Irradiance in W/m²

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(irradiance, (int, float)):
        return False, "Irradiance must be a number"

    if irradiance < 0:
        return False, "Irradiance cannot be negative"

    if irradiance > 1500:
        return False, "Irradiance exceeds typical maximum (>1500 W/m²)"

    return True, ""


def validate_percentage(value: float, allow_zero: bool = True, allow_hundred: bool = True) -> Tuple[bool, str]:
    """
    Validate percentage value.

    Args:
        value: Percentage value
        allow_zero: Whether 0% is valid
        allow_hundred: Whether 100% is valid

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, "Value must be a number"

    min_val = 0 if allow_zero else 0.001
    max_val = 100 if allow_hundred else 99.999

    if value < min_val or value > max_val:
        return False, f"Percentage must be between {min_val} and {max_val}%"

    return True, ""


def validate_k_factor(k_factor: float) -> Tuple[bool, str]:
    """
    Validate CTM loss k-factor.

    Args:
        k_factor: K-factor value (0-1)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(k_factor, (int, float)):
        return False, "K-factor must be a number"

    if k_factor < 0 or k_factor > 1:
        return False, "K-factor must be between 0 and 1"

    if k_factor < 0.5:
        return False, "K-factor is unreasonably low (<0.5), indicating >50% loss"

    return True, ""


def validate_date_range(start_date: datetime, end_date: datetime) -> Tuple[bool, str]:
    """
    Validate date range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        return False, "Dates must be datetime objects"

    if start_date >= end_date:
        return False, "Start date must be before end date"

    if end_date > datetime.now():
        return False, "End date cannot be in the future"

    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format.

    Args:
        email: Email address string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(email, str):
        return False, "Email must be a string"

    # Simple email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, email):
        return False, "Invalid email format"

    return True, ""


def validate_positive_integer(value: Any) -> Tuple[bool, str]:
    """
    Validate positive integer.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, int):
        return False, "Value must be an integer"

    if value <= 0:
        return False, "Value must be positive"

    return True, ""


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    return True, ""


def validate_module_design(design_data: dict) -> Tuple[bool, List[str]]:
    """
    Validate complete module design data.

    Args:
        design_data: Dictionary containing module design parameters

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    required_fields = ['cell_type', 'num_cells', 'cell_efficiency', 'module_power']

    for field in required_fields:
        if field not in design_data:
            errors.append(f"Missing required field: {field}")

    # Validate individual parameters
    if 'cell_efficiency' in design_data:
        is_valid, msg = validate_efficiency(design_data['cell_efficiency'])
        if not is_valid:
            errors.append(f"Cell efficiency: {msg}")

    if 'num_cells' in design_data:
        is_valid, msg = validate_positive_integer(design_data['num_cells'])
        if not is_valid:
            errors.append(f"Number of cells: {msg}")

    if 'module_power' in design_data:
        is_valid, msg = validate_power(design_data['module_power'])
        if not is_valid:
            errors.append(f"Module power: {msg}")

    # Validate electrical parameters if present
    electrical_params = ['voc', 'isc', 'vmpp', 'impp']
    for param in electrical_params:
        if param in design_data:
            if 'v' in param.lower():
                is_valid, msg = validate_voltage(design_data[param])
            else:
                is_valid, msg = validate_current(design_data[param])

            if not is_valid:
                errors.append(f"{param}: {msg}")

    return len(errors) == 0, errors


def validate_system_design(design_data: dict) -> Tuple[bool, List[str]]:
    """
    Validate complete system design data.

    Args:
        design_data: Dictionary containing system design parameters

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check site information
    if 'site' in design_data:
        site = design_data['site']

        if 'latitude' in site:
            is_valid, msg = validate_latitude(site['latitude'])
            if not is_valid:
                errors.append(f"Latitude: {msg}")

        if 'longitude' in site:
            is_valid, msg = validate_longitude(site['longitude'])
            if not is_valid:
                errors.append(f"Longitude: {msg}")

    # Check array configuration
    if 'array' in design_data:
        array = design_data['array']

        if 'tilt_angle' in array:
            is_valid, msg = validate_angle(array['tilt_angle'], 0, 90)
            if not is_valid:
                errors.append(f"Tilt angle: {msg}")

        if 'azimuth' in array:
            is_valid, msg = validate_angle(array['azimuth'], 0, 360)
            if not is_valid:
                errors.append(f"Azimuth: {msg}")

        if 'num_modules' in array:
            is_valid, msg = validate_positive_integer(array['num_modules'])
            if not is_valid:
                errors.append(f"Number of modules: {msg}")

    # Check inverter configuration
    if 'inverter' in design_data:
        inverter = design_data['inverter']

        if 'efficiency' in inverter:
            is_valid, msg = validate_efficiency(inverter['efficiency'])
            if not is_valid:
                errors.append(f"Inverter efficiency: {msg}")

        if 'dc_ac_ratio' in inverter:
            dc_ac = inverter['dc_ac_ratio']
            if dc_ac < 0.8 or dc_ac > 2.0:
                errors.append("DC/AC ratio should typically be between 0.8 and 2.0")

    return len(errors) == 0, errors


def validate_financial_inputs(
    capex: float,
    opex: float,
    discount_rate: float,
    lifetime: int
) -> Tuple[bool, List[str]]:
    """
    Validate financial analysis inputs.

    Args:
        capex: Capital expenditure
        opex: Operating expenditure
        discount_rate: Discount rate (decimal)
        lifetime: Project lifetime (years)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if capex < 0:
        errors.append("CAPEX cannot be negative")

    if opex < 0:
        errors.append("OPEX cannot be negative")

    if discount_rate < 0 or discount_rate > 1:
        errors.append("Discount rate must be between 0 and 1 (decimal)")

    if lifetime <= 0 or lifetime > 50:
        errors.append("Project lifetime must be between 1 and 50 years")

    return len(errors) == 0, errors


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Remove leading/trailing periods and spaces
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


def validate_csv_data(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> Tuple[bool, List[str]]:
    """
    Validate CSV data structure.

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        min_rows: Minimum number of rows required

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(df, pd.DataFrame):
        errors.append("Data must be a pandas DataFrame")
        return False, errors

    if len(df) < min_rows:
        errors.append(f"Data must have at least {min_rows} row(s)")

    if expected_columns:
        is_valid, msg = validate_dataframe_columns(df, expected_columns)
        if not is_valid:
            errors.append(msg)

    # Check for empty DataFrame
    if df.empty:
        errors.append("Data is empty")

    return len(errors) == 0, errors
