"""
Unit Conversion Utilities for PV Circularity Simulator.

This module provides comprehensive unit conversion functions for various
physical quantities commonly used in photovoltaic system analysis.
"""

from typing import Union, Literal

# Type aliases for clarity
Number = Union[int, float]
EnergyUnit = Literal["Wh", "kWh", "MWh", "GWh", "J", "kJ", "MJ", "GJ"]
PowerUnit = Literal["W", "kW", "MW", "GW"]
AreaUnit = Literal["m2", "cm2", "mm2", "km2", "ha"]
MassUnit = Literal["kg", "g", "mg", "ton", "lb", "oz"]
LengthUnit = Literal["m", "cm", "mm", "km", "inch", "ft", "yard", "mile"]
TemperatureUnit = Literal["C", "F", "K"]
EfficiencyUnit = Literal["percent", "decimal", "ppm"]


# Energy Conversion Factors (to Wh)
_ENERGY_TO_WH = {
    "Wh": 1.0,
    "kWh": 1_000.0,
    "MWh": 1_000_000.0,
    "GWh": 1_000_000_000.0,
    "J": 1 / 3600.0,
    "kJ": 1_000 / 3600.0,
    "MJ": 1_000_000 / 3600.0,
    "GJ": 1_000_000_000 / 3600.0,
}

# Power Conversion Factors (to W)
_POWER_TO_W = {
    "W": 1.0,
    "kW": 1_000.0,
    "MW": 1_000_000.0,
    "GW": 1_000_000_000.0,
}

# Area Conversion Factors (to m²)
_AREA_TO_M2 = {
    "m2": 1.0,
    "cm2": 0.0001,
    "mm2": 0.000001,
    "km2": 1_000_000.0,
    "ha": 10_000.0,  # hectare
}

# Mass Conversion Factors (to kg)
_MASS_TO_KG = {
    "kg": 1.0,
    "g": 0.001,
    "mg": 0.000001,
    "ton": 1_000.0,
    "lb": 0.453592,
    "oz": 0.0283495,
}

# Length Conversion Factors (to m)
_LENGTH_TO_M = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "km": 1_000.0,
    "inch": 0.0254,
    "ft": 0.3048,
    "yard": 0.9144,
    "mile": 1_609.344,
}

# Efficiency Conversion Factors (to decimal)
_EFFICIENCY_TO_DECIMAL = {
    "decimal": 1.0,
    "percent": 0.01,
    "ppm": 0.000001,  # parts per million
}


def convert_energy(
    value: Number, from_unit: EnergyUnit, to_unit: EnergyUnit
) -> float:
    """
    Convert energy between different units.

    Supported units: Wh, kWh, MWh, GWh, J, kJ, MJ, GJ

    Args:
        value: The energy value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted energy value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_energy(1000, "Wh", "kWh")
        1.0
        >>> convert_energy(1, "kWh", "MJ")
        3.6
        >>> convert_energy(3600, "J", "Wh")
        1.0
    """
    if from_unit not in _ENERGY_TO_WH:
        raise ValueError(f"Unsupported energy unit: {from_unit}")
    if to_unit not in _ENERGY_TO_WH:
        raise ValueError(f"Unsupported energy unit: {to_unit}")

    # Convert to Wh, then to target unit
    wh = value * _ENERGY_TO_WH[from_unit]
    return wh / _ENERGY_TO_WH[to_unit]


def convert_power(
    value: Number, from_unit: PowerUnit, to_unit: PowerUnit
) -> float:
    """
    Convert power between different units.

    Supported units: W, kW, MW, GW

    Args:
        value: The power value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted power value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_power(1000, "W", "kW")
        1.0
        >>> convert_power(5, "MW", "kW")
        5000.0
        >>> convert_power(0.5, "kW", "W")
        500.0
    """
    if from_unit not in _POWER_TO_W:
        raise ValueError(f"Unsupported power unit: {from_unit}")
    if to_unit not in _POWER_TO_W:
        raise ValueError(f"Unsupported power unit: {to_unit}")

    # Convert to W, then to target unit
    watts = value * _POWER_TO_W[from_unit]
    return watts / _POWER_TO_W[to_unit]


def convert_area(
    value: Number, from_unit: AreaUnit, to_unit: AreaUnit
) -> float:
    """
    Convert area between different units.

    Supported units: m2, cm2, mm2, km2, ha (hectare)

    Args:
        value: The area value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted area value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_area(10000, "cm2", "m2")
        1.0
        >>> convert_area(1, "ha", "m2")
        10000.0
        >>> convert_area(1000000, "mm2", "m2")
        1.0
    """
    if from_unit not in _AREA_TO_M2:
        raise ValueError(f"Unsupported area unit: {from_unit}")
    if to_unit not in _AREA_TO_M2:
        raise ValueError(f"Unsupported area unit: {to_unit}")

    # Convert to m², then to target unit
    m2 = value * _AREA_TO_M2[from_unit]
    return m2 / _AREA_TO_M2[to_unit]


def convert_mass(
    value: Number, from_unit: MassUnit, to_unit: MassUnit
) -> float:
    """
    Convert mass between different units.

    Supported units: kg, g, mg, ton, lb (pound), oz (ounce)

    Args:
        value: The mass value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted mass value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_mass(1000, "g", "kg")
        1.0
        >>> convert_mass(1, "ton", "kg")
        1000.0
        >>> convert_mass(1, "lb", "kg")
        0.453592
    """
    if from_unit not in _MASS_TO_KG:
        raise ValueError(f"Unsupported mass unit: {from_unit}")
    if to_unit not in _MASS_TO_KG:
        raise ValueError(f"Unsupported mass unit: {to_unit}")

    # Convert to kg, then to target unit
    kg = value * _MASS_TO_KG[from_unit]
    return kg / _MASS_TO_KG[to_unit]


def convert_length(
    value: Number, from_unit: LengthUnit, to_unit: LengthUnit
) -> float:
    """
    Convert length between different units.

    Supported units: m, cm, mm, km, inch, ft (foot), yard, mile

    Args:
        value: The length value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted length value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_length(100, "cm", "m")
        1.0
        >>> convert_length(1, "km", "m")
        1000.0
        >>> convert_length(12, "inch", "ft")
        1.0
    """
    if from_unit not in _LENGTH_TO_M:
        raise ValueError(f"Unsupported length unit: {from_unit}")
    if to_unit not in _LENGTH_TO_M:
        raise ValueError(f"Unsupported length unit: {to_unit}")

    # Convert to m, then to target unit
    meters = value * _LENGTH_TO_M[from_unit]
    return meters / _LENGTH_TO_M[to_unit]


def convert_temperature(
    value: Number, from_unit: TemperatureUnit, to_unit: TemperatureUnit
) -> float:
    """
    Convert temperature between different units.

    Supported units: C (Celsius), F (Fahrenheit), K (Kelvin)

    Args:
        value: The temperature value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted temperature value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_temperature(0, "C", "F")
        32.0
        >>> convert_temperature(273.15, "K", "C")
        0.0
        >>> convert_temperature(100, "C", "K")
        373.15
    """
    # Convert to Celsius first
    if from_unit == "C":
        celsius = float(value)
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {from_unit}")

    # Convert from Celsius to target unit
    if to_unit == "C":
        return celsius
    elif to_unit == "F":
        return celsius * 9 / 5 + 32
    elif to_unit == "K":
        return celsius + 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {to_unit}")


def convert_efficiency(
    value: Number, from_unit: EfficiencyUnit, to_unit: EfficiencyUnit
) -> float:
    """
    Convert efficiency between different representations.

    Supported units: decimal (0-1), percent (0-100), ppm (parts per million)

    Args:
        value: The efficiency value to convert
        from_unit: The source unit
        to_unit: The target unit

    Returns:
        The converted efficiency value

    Raises:
        ValueError: If an unsupported unit is provided

    Examples:
        >>> convert_efficiency(20, "percent", "decimal")
        0.2
        >>> convert_efficiency(0.15, "decimal", "percent")
        15.0
        >>> convert_efficiency(1000, "ppm", "percent")
        0.1
    """
    if from_unit not in _EFFICIENCY_TO_DECIMAL:
        raise ValueError(f"Unsupported efficiency unit: {from_unit}")
    if to_unit not in _EFFICIENCY_TO_DECIMAL:
        raise ValueError(f"Unsupported efficiency unit: {to_unit}")

    # Convert to decimal, then to target unit
    decimal = value * _EFFICIENCY_TO_DECIMAL[from_unit]
    return decimal / _EFFICIENCY_TO_DECIMAL[to_unit]


def calculate_energy_from_power(
    power: Number, power_unit: PowerUnit, duration_hours: Number
) -> float:
    """
    Calculate energy from power and duration.

    Args:
        power: The power value
        power_unit: The unit of power (W, kW, MW, GW)
        duration_hours: Duration in hours

    Returns:
        Energy in Wh

    Examples:
        >>> calculate_energy_from_power(1000, "W", 5)
        5000.0
        >>> calculate_energy_from_power(2, "kW", 10)
        20000.0
    """
    power_w = convert_power(power, power_unit, "W")
    return power_w * duration_hours


def calculate_power_from_energy(
    energy: Number, energy_unit: EnergyUnit, duration_hours: Number
) -> float:
    """
    Calculate power from energy and duration.

    Args:
        energy: The energy value
        energy_unit: The unit of energy (Wh, kWh, MWh, etc.)
        duration_hours: Duration in hours

    Returns:
        Power in W

    Raises:
        ValueError: If duration_hours is zero

    Examples:
        >>> calculate_power_from_energy(5000, "Wh", 5)
        1000.0
        >>> calculate_power_from_energy(20, "kWh", 10)
        2000.0
    """
    if duration_hours == 0:
        raise ValueError("Duration cannot be zero")

    energy_wh = convert_energy(energy, energy_unit, "Wh")
    return energy_wh / duration_hours


def calculate_specific_yield(
    energy: Number, energy_unit: EnergyUnit, power: Number, power_unit: PowerUnit
) -> float:
    """
    Calculate specific yield (energy per unit power, typically kWh/kWp).

    Specific yield is a key performance metric for PV systems, representing
    the energy output per unit of installed power capacity.

    Args:
        energy: The energy produced
        energy_unit: The unit of energy
        power: The installed power capacity
        power_unit: The unit of power

    Returns:
        Specific yield in kWh/kWp

    Raises:
        ValueError: If power is zero

    Examples:
        >>> calculate_specific_yield(1000, "kWh", 5, "kW")
        200.0
        >>> calculate_specific_yield(5000, "Wh", 1000, "W")
        5.0
    """
    if power == 0:
        raise ValueError("Power cannot be zero")

    energy_kwh = convert_energy(energy, energy_unit, "kWh")
    power_kw = convert_power(power, power_unit, "kW")
    return energy_kwh / power_kw
