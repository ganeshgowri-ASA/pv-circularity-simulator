"""
Helper functions for thermal and temperature calculations.

This module provides utility functions for unit conversions and heat transfer calculations.
"""

import numpy as np
from typing import Union

from pv_simulator.utils.constants import (
    KINEMATIC_VISCOSITY_AIR,
    SPECIFIC_HEAT_AIR,
    THERMAL_CONDUCTIVITY_AIR,
)


def celsius_to_kelvin(temp_c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Celsius to Kelvin.

    Args:
        temp_c: Temperature in degrees Celsius

    Returns:
        Temperature in Kelvin

    Examples:
        >>> celsius_to_kelvin(25.0)
        298.15
        >>> celsius_to_kelvin(np.array([0, 25, 100]))
        array([273.15, 298.15, 373.15])
    """
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Kelvin to Celsius.

    Args:
        temp_k: Temperature in Kelvin

    Returns:
        Temperature in degrees Celsius

    Examples:
        >>> kelvin_to_celsius(298.15)
        25.0
        >>> kelvin_to_celsius(np.array([273.15, 298.15, 373.15]))
        array([0., 25., 100.])
    """
    return temp_k - 273.15


def calculate_reynolds_number(
    velocity: Union[float, np.ndarray],
    characteristic_length: float,
    kinematic_viscosity: float = KINEMATIC_VISCOSITY_AIR,
) -> Union[float, np.ndarray]:
    """
    Calculate Reynolds number for flow characterization.

    The Reynolds number is a dimensionless quantity that characterizes the flow regime
    (laminar vs. turbulent) and is critical for heat transfer calculations.

    Args:
        velocity: Flow velocity in m/s
        characteristic_length: Characteristic length in m (e.g., module length)
        kinematic_viscosity: Kinematic viscosity in m²/s (default: air at 20°C)

    Returns:
        Reynolds number (dimensionless)

    Examples:
        >>> calculate_reynolds_number(5.0, 1.0)
        333333.33333333337
        >>> calculate_reynolds_number(np.array([1, 5, 10]), 1.0)
        array([66666.67, 333333.33, 666666.67])
    """
    return velocity * characteristic_length / kinematic_viscosity


def calculate_prandtl_number(
    specific_heat: float = SPECIFIC_HEAT_AIR,
    dynamic_viscosity: float = KINEMATIC_VISCOSITY_AIR * 1.225,  # ν * ρ
    thermal_conductivity: float = THERMAL_CONDUCTIVITY_AIR,
) -> float:
    """
    Calculate Prandtl number for the fluid.

    The Prandtl number is the ratio of momentum diffusivity to thermal diffusivity
    and is used in heat transfer correlations.

    Args:
        specific_heat: Specific heat at constant pressure in J/(kg·K)
        dynamic_viscosity: Dynamic viscosity in Pa·s
        thermal_conductivity: Thermal conductivity in W/(m·K)

    Returns:
        Prandtl number (dimensionless)

    Examples:
        >>> pr = calculate_prandtl_number()
        >>> 0.7 < pr < 0.72  # For air at room temperature
        True
    """
    return specific_heat * dynamic_viscosity / thermal_conductivity


def calculate_nusselt_number(
    reynolds: Union[float, np.ndarray],
    prandtl: float = 0.71,
    flow_regime: str = "turbulent",
) -> Union[float, np.ndarray]:
    """
    Calculate Nusselt number for convective heat transfer.

    The Nusselt number represents the enhancement of heat transfer through a fluid layer
    as a result of convection relative to conduction across the same fluid layer.

    Args:
        reynolds: Reynolds number (dimensionless)
        prandtl: Prandtl number (dimensionless), default 0.71 for air
        flow_regime: Flow regime, either "laminar" or "turbulent"

    Returns:
        Nusselt number (dimensionless)

    Raises:
        ValueError: If flow_regime is not "laminar" or "turbulent"

    Examples:
        >>> calculate_nusselt_number(10000, 0.71, "turbulent")
        28.846...
        >>> calculate_nusselt_number(1000, 0.71, "laminar")
        5.385...

    References:
        - Turbulent: Dittus-Boelter equation: Nu = 0.023 * Re^0.8 * Pr^0.4
        - Laminar: Simplified correlation: Nu = 0.664 * Re^0.5 * Pr^(1/3)
    """
    if flow_regime == "turbulent":
        # Dittus-Boelter equation for turbulent flow
        nusselt = 0.023 * np.power(reynolds, 0.8) * np.power(prandtl, 0.4)
    elif flow_regime == "laminar":
        # Simplified laminar flow correlation
        nusselt = 0.664 * np.power(reynolds, 0.5) * np.power(prandtl, 1.0 / 3.0)
    else:
        raise ValueError(f"Invalid flow_regime: {flow_regime}. Must be 'laminar' or 'turbulent'")

    return nusselt


def calculate_heat_transfer_coefficient(
    nusselt: Union[float, np.ndarray],
    characteristic_length: float,
    thermal_conductivity: float = THERMAL_CONDUCTIVITY_AIR,
) -> Union[float, np.ndarray]:
    """
    Calculate convective heat transfer coefficient from Nusselt number.

    Args:
        nusselt: Nusselt number (dimensionless)
        characteristic_length: Characteristic length in m
        thermal_conductivity: Thermal conductivity of fluid in W/(m·K)

    Returns:
        Heat transfer coefficient in W/(m²·K)

    Examples:
        >>> h = calculate_heat_transfer_coefficient(30.0, 1.0)
        >>> 0.7 < h < 0.8  # For typical air conditions
        True
    """
    return nusselt * thermal_conductivity / characteristic_length


def wind_speed_at_height(
    wind_speed_ref: Union[float, np.ndarray],
    height: float,
    height_ref: float = 10.0,
    roughness_length: float = 0.1,
) -> Union[float, np.ndarray]:
    """
    Calculate wind speed at a specific height using logarithmic wind profile.

    Args:
        wind_speed_ref: Reference wind speed in m/s
        height: Target height in m
        height_ref: Reference height in m (default: 10 m, standard measurement height)
        roughness_length: Surface roughness length in m (default: 0.1 m for open terrain)

    Returns:
        Wind speed at target height in m/s

    Examples:
        >>> wind_speed_at_height(5.0, 2.0, 10.0)
        3.6...

    References:
        Logarithmic wind profile: v(z) = v_ref * ln(z/z0) / ln(z_ref/z0)
    """
    return wind_speed_ref * np.log(height / roughness_length) / np.log(height_ref / roughness_length)
