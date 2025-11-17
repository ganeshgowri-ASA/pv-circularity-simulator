"""
Data Validation Helpers for PV Circularity Simulator.

This module provides Pydantic-based validators and utility functions for
validating data commonly used in photovoltaic system analysis.
"""

from typing import Any, Optional, Union, List, Dict, Callable
from datetime import datetime, date
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError,
    ConfigDict,
)
import re


class PositiveValue(BaseModel):
    """Validates that a value is positive (greater than zero)."""

    value: float = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)


class NonNegativeValue(BaseModel):
    """Validates that a value is non-negative (greater than or equal to zero)."""

    value: float = Field(..., ge=0)

    model_config = ConfigDict(frozen=True)


class PercentageValue(BaseModel):
    """Validates that a value is a valid percentage (0-100)."""

    value: float = Field(..., ge=0, le=100)

    model_config = ConfigDict(frozen=True)


class EfficiencyValue(BaseModel):
    """Validates that a value is a valid efficiency (0-1 as decimal)."""

    value: float = Field(..., ge=0, le=1)

    model_config = ConfigDict(frozen=True)


class TemperatureValue(BaseModel):
    """Validates temperature values with optional unit and range checking."""

    value: float
    unit: str = Field(default="C")
    min_value: Optional[float] = Field(default=-273.15)  # Absolute zero in Celsius
    max_value: Optional[float] = Field(default=None)

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate temperature unit."""
        allowed_units = {"C", "F", "K"}
        if v not in allowed_units:
            raise ValueError(f"Unit must be one of {allowed_units}")
        return v

    @model_validator(mode="after")
    def validate_range(self) -> "TemperatureValue":
        """Validate temperature is within acceptable range."""
        if self.min_value is not None and self.value < self.min_value:
            raise ValueError(f"Temperature {self.value} is below minimum {self.min_value}")
        if self.max_value is not None and self.value > self.max_value:
            raise ValueError(f"Temperature {self.value} exceeds maximum {self.max_value}")
        return self


class DateRangeValue(BaseModel):
    """Validates a date range with start and end dates."""

    start_date: Union[datetime, date]
    end_date: Union[datetime, date]

    @model_validator(mode="after")
    def validate_date_range(self) -> "DateRangeValue":
        """Ensure end date is after start date."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class PVModuleSpecs(BaseModel):
    """Validates PV module specifications."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=100)
    power_rating_w: float = Field(..., gt=0, description="Rated power in Watts")
    efficiency: float = Field(..., gt=0, le=1, description="Module efficiency (0-1)")
    area_m2: float = Field(..., gt=0, description="Module area in square meters")
    voltage_voc: float = Field(..., gt=0, description="Open circuit voltage in V")
    current_isc: float = Field(..., gt=0, description="Short circuit current in A")
    temperature_coeff_power: float = Field(
        ..., description="Temperature coefficient of power (%/°C)"
    )
    warranty_years: int = Field(..., ge=0, le=50, description="Warranty period in years")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate module name format."""
        if not v or v.isspace():
            raise ValueError("Name cannot be empty or whitespace")
        return v

    @model_validator(mode="after")
    def validate_efficiency_vs_power(self) -> "PVModuleSpecs":
        """Validate that efficiency matches power and area."""
        # Calculate theoretical efficiency
        # Assuming ~1000 W/m² standard test conditions
        theoretical_efficiency = self.power_rating_w / (self.area_m2 * 1000)

        # Allow 5% tolerance for measurement differences
        if abs(theoretical_efficiency - self.efficiency) > 0.05:
            raise ValueError(
                f"Efficiency {self.efficiency:.2%} doesn't match "
                f"power rating {self.power_rating_w}W and area {self.area_m2}m² "
                f"(calculated: {theoretical_efficiency:.2%})"
            )
        return self


class EnergyProductionData(BaseModel):
    """Validates energy production data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime
    energy_kwh: float = Field(..., ge=0, description="Energy produced in kWh")
    power_kw: float = Field(..., ge=0, description="Instantaneous power in kW")
    irradiance_w_m2: Optional[float] = Field(
        default=None, ge=0, le=1500, description="Solar irradiance in W/m²"
    )
    temperature_c: Optional[float] = Field(
        default=None, ge=-50, le=100, description="Module temperature in °C"
    )
    performance_ratio: Optional[float] = Field(
        default=None, ge=0, le=1, description="Performance ratio (0-1)"
    )


class MaterialComposition(BaseModel):
    """Validates material composition data for circular economy analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    material_name: str = Field(..., min_length=1)
    mass_kg: float = Field(..., ge=0)
    recyclability_rate: float = Field(..., ge=0, le=1, description="Recyclability (0-1)")
    toxicity_level: Optional[str] = Field(default=None)
    cost_per_kg: Optional[float] = Field(default=None, ge=0)

    @field_validator("toxicity_level")
    @classmethod
    def validate_toxicity(cls, v: Optional[str]) -> Optional[str]:
        """Validate toxicity level."""
        if v is not None:
            allowed_levels = {"low", "medium", "high", "very_high"}
            if v.lower() not in allowed_levels:
                raise ValueError(f"Toxicity level must be one of {allowed_levels}")
            return v.lower()
        return v


def validate_positive(value: Union[int, float], field_name: str = "value") -> float:
    """
    Validate that a value is positive (greater than zero).

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value as float

    Raises:
        ValueError: If value is not positive

    Examples:
        >>> validate_positive(5.0)
        5.0
        >>> validate_positive(-1)
        Traceback (most recent call last):
        ...
        ValueError: value must be positive (greater than zero)
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive (greater than zero)")
    return float(value)


def validate_non_negative(value: Union[int, float], field_name: str = "value") -> float:
    """
    Validate that a value is non-negative (greater than or equal to zero).

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value as float

    Raises:
        ValueError: If value is negative

    Examples:
        >>> validate_non_negative(0)
        0.0
        >>> validate_non_negative(5.5)
        5.5
        >>> validate_non_negative(-1)
        Traceback (most recent call last):
        ...
        ValueError: value cannot be negative
    """
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return float(value)


def validate_percentage(value: Union[int, float], field_name: str = "value") -> float:
    """
    Validate that a value is a valid percentage (0-100).

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value as float

    Raises:
        ValueError: If value is not in range [0, 100]

    Examples:
        >>> validate_percentage(50)
        50.0
        >>> validate_percentage(101)
        Traceback (most recent call last):
        ...
        ValueError: value must be between 0 and 100
    """
    if not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return float(value)


def validate_efficiency(value: Union[int, float], field_name: str = "value") -> float:
    """
    Validate that a value is a valid efficiency (0-1).

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value as float

    Raises:
        ValueError: If value is not in range [0, 1]

    Examples:
        >>> validate_efficiency(0.25)
        0.25
        >>> validate_efficiency(1.5)
        Traceback (most recent call last):
        ...
        ValueError: value must be between 0 and 1
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return float(value)


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: str = "value",
) -> float:
    """
    Validate that a value is within a specified range.

    Args:
        value: The value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages

    Returns:
        The validated value as float

    Raises:
        ValueError: If value is outside the specified range

    Examples:
        >>> validate_range(5, 0, 10)
        5.0
        >>> validate_range(15, 0, 10)
        Traceback (most recent call last):
        ...
        ValueError: value must be between 0 and 10
    """
    if min_value is not None and value < min_value:
        if max_value is not None:
            raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
        else:
            raise ValueError(f"{field_name} must be at least {min_value}")

    if max_value is not None and value > max_value:
        if min_value is not None:
            raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
        else:
            raise ValueError(f"{field_name} must be at most {max_value}")

    return float(value)


def validate_list_not_empty(value: List[Any], field_name: str = "list") -> List[Any]:
    """
    Validate that a list is not empty.

    Args:
        value: The list to validate
        field_name: Name of the field for error messages

    Returns:
        The validated list

    Raises:
        ValueError: If list is empty

    Examples:
        >>> validate_list_not_empty([1, 2, 3])
        [1, 2, 3]
        >>> validate_list_not_empty([])
        Traceback (most recent call last):
        ...
        ValueError: list cannot be empty
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def validate_dict_keys(
    value: Dict[str, Any],
    required_keys: List[str],
    field_name: str = "dictionary",
) -> Dict[str, Any]:
    """
    Validate that a dictionary contains all required keys.

    Args:
        value: The dictionary to validate
        required_keys: List of required key names
        field_name: Name of the field for error messages

    Returns:
        The validated dictionary

    Raises:
        ValueError: If any required keys are missing

    Examples:
        >>> validate_dict_keys({"a": 1, "b": 2}, ["a", "b"])
        {'a': 1, 'b': 2}
        >>> validate_dict_keys({"a": 1}, ["a", "b"])
        Traceback (most recent call last):
        ...
        ValueError: dictionary is missing required keys: b
    """
    missing_keys = set(required_keys) - set(value.keys())
    if missing_keys:
        raise ValueError(f"{field_name} is missing required keys: {', '.join(missing_keys)}")
    return value


def validate_email(email: str) -> str:
    """
    Validate email address format.

    Args:
        email: The email address to validate

    Returns:
        The validated email address

    Raises:
        ValueError: If email format is invalid

    Examples:
        >>> validate_email("user@example.com")
        'user@example.com'
        >>> validate_email("invalid-email")
        Traceback (most recent call last):
        ...
        ValueError: Invalid email format
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        raise ValueError("Invalid email format")
    return email


def validate_date_range(
    start_date: Union[datetime, date],
    end_date: Union[datetime, date],
) -> tuple[Union[datetime, date], Union[datetime, date]]:
    """
    Validate that end date is after start date.

    Args:
        start_date: The start date
        end_date: The end date

    Returns:
        Tuple of (start_date, end_date)

    Raises:
        ValueError: If end_date is before start_date

    Examples:
        >>> from datetime import date
        >>> validate_date_range(date(2024, 1, 1), date(2024, 12, 31))
        (datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
        >>> validate_date_range(date(2024, 12, 31), date(2024, 1, 1))
        Traceback (most recent call last):
        ...
        ValueError: end_date must be after start_date
    """
    if end_date < start_date:
        raise ValueError("end_date must be after start_date")
    return start_date, end_date


def safe_validate(
    validator_func: Callable[[Any], Any],
    value: Any,
    default: Any = None,
) -> Any:
    """
    Safely validate a value, returning default if validation fails.

    Args:
        validator_func: The validation function to use
        value: The value to validate
        default: Default value to return if validation fails

    Returns:
        The validated value or default if validation fails

    Examples:
        >>> safe_validate(validate_positive, 5)
        5.0
        >>> safe_validate(validate_positive, -1, default=0)
        0
    """
    try:
        return validator_func(value)
    except (ValueError, ValidationError):
        return default


def batch_validate(
    values: List[Any],
    validator_func: Callable[[Any], Any],
) -> tuple[List[Any], List[tuple[int, str]]]:
    """
    Validate a list of values and return both valid values and errors.

    Args:
        values: List of values to validate
        validator_func: The validation function to use

    Returns:
        Tuple of (valid_values, errors) where errors is list of (index, error_message)

    Examples:
        >>> valid, errors = batch_validate([1, -1, 5, -3], validate_positive)
        >>> valid
        [1.0, 5.0]
        >>> len(errors)
        2
    """
    valid_values = []
    errors = []

    for i, value in enumerate(values):
        try:
            valid_value = validator_func(value)
            valid_values.append(valid_value)
        except (ValueError, ValidationError) as e:
            errors.append((i, str(e)))

    return valid_values, errors
