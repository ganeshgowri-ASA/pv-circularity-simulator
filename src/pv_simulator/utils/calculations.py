"""
Calculation Helpers for PV Circularity Simulator.

This module provides comprehensive calculation utilities for statistical,
financial, and technical computations commonly used in photovoltaic system
analysis and circular economy modeling.
"""

from typing import List, Union, Optional, Tuple
import math
from statistics import mean, median, stdev, variance

Number = Union[int, float]


# ============================================================================
# Statistical Calculations
# ============================================================================


def calculate_mean(values: List[Number]) -> float:
    """
    Calculate the arithmetic mean (average) of a list of values.

    Args:
        values: List of numeric values

    Returns:
        The mean value

    Raises:
        ValueError: If values list is empty

    Examples:
        >>> calculate_mean([1, 2, 3, 4, 5])
        3.0
        >>> calculate_mean([10, 20, 30])
        20.0
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return mean(values)


def calculate_median(values: List[Number]) -> float:
    """
    Calculate the median of a list of values.

    Args:
        values: List of numeric values

    Returns:
        The median value

    Raises:
        ValueError: If values list is empty

    Examples:
        >>> calculate_median([1, 2, 3, 4, 5])
        3
        >>> calculate_median([1, 2, 3, 4])
        2.5
    """
    if not values:
        raise ValueError("Cannot calculate median of empty list")
    return median(values)


def calculate_standard_deviation(
    values: List[Number], sample: bool = True
) -> float:
    """
    Calculate the standard deviation of a list of values.

    Args:
        values: List of numeric values
        sample: If True, use sample standard deviation (n-1); if False, use population (n)

    Returns:
        The standard deviation

    Raises:
        ValueError: If values list has fewer than 2 elements

    Examples:
        >>> calculate_standard_deviation([2, 4, 4, 4, 5, 5, 7, 9])
        2.138...
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values to calculate standard deviation")
    return stdev(values) if sample else math.sqrt(variance(values))


def calculate_variance(values: List[Number], sample: bool = True) -> float:
    """
    Calculate the variance of a list of values.

    Args:
        values: List of numeric values
        sample: If True, use sample variance (n-1); if False, use population (n)

    Returns:
        The variance

    Raises:
        ValueError: If values list has fewer than 2 elements

    Examples:
        >>> calculate_variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.571...
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values to calculate variance")
    return variance(values)


def calculate_percentile(values: List[Number], percentile: float) -> float:
    """
    Calculate a percentile of a list of values.

    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value

    Raises:
        ValueError: If values list is empty or percentile is out of range

    Examples:
        >>> calculate_percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)
        5.5
        >>> calculate_percentile([1, 2, 3, 4, 5], 95)
        4.8
    """
    if not values:
        raise ValueError("Cannot calculate percentile of empty list")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100)
    floor = math.floor(k)
    ceil = math.ceil(k)

    if floor == ceil:
        return sorted_values[int(k)]

    d0 = sorted_values[floor] * (ceil - k)
    d1 = sorted_values[ceil] * (k - floor)
    return d0 + d1


def calculate_weighted_average(
    values: List[Number], weights: List[Number]
) -> float:
    """
    Calculate weighted average of values.

    Args:
        values: List of numeric values
        weights: List of weights corresponding to values

    Returns:
        The weighted average

    Raises:
        ValueError: If lists are empty, different lengths, or total weight is zero

    Examples:
        >>> calculate_weighted_average([10, 20, 30], [1, 2, 3])
        23.333...
        >>> calculate_weighted_average([85, 90, 92], [0.5, 0.3, 0.2])
        88.3
    """
    if not values or not weights:
        raise ValueError("Values and weights cannot be empty")
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")

    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ============================================================================
# Financial Calculations
# ============================================================================


def calculate_npv(
    cash_flows: List[Number],
    discount_rate: float,
    initial_investment: Number = 0,
) -> float:
    """
    Calculate Net Present Value (NPV) of cash flows.

    Args:
        cash_flows: List of cash flows for each period
        discount_rate: Discount rate (as decimal, e.g., 0.1 for 10%)
        initial_investment: Initial investment (will be subtracted from NPV)

    Returns:
        The net present value

    Examples:
        >>> calculate_npv([100, 100, 100], 0.1, 250)
        -1.30...
        >>> calculate_npv([1000, 2000, 3000], 0.05)
        5596.87...
    """
    npv = -initial_investment
    for i, cash_flow in enumerate(cash_flows, start=1):
        npv += cash_flow / ((1 + discount_rate) ** i)
    return npv


def calculate_irr(
    cash_flows: List[Number],
    initial_investment: Number,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Calculate Internal Rate of Return (IRR) using Newton-Raphson method.

    Args:
        cash_flows: List of cash flows for each period
        initial_investment: Initial investment (positive number)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        The internal rate of return as decimal, or None if no solution found

    Examples:
        >>> irr = calculate_irr([100, 100, 100], 250)
        >>> irr is not None and 0.09 < irr < 0.10
        True
    """
    # Initial guess
    rate = 0.1

    for _ in range(max_iterations):
        # Calculate NPV and its derivative
        npv = -initial_investment
        npv_derivative = 0

        for i, cash_flow in enumerate(cash_flows, start=1):
            discount_factor = (1 + rate) ** i
            npv += cash_flow / discount_factor
            npv_derivative -= i * cash_flow / ((1 + rate) ** (i + 1))

        # Check for convergence
        if abs(npv) < tolerance:
            return rate

        # Newton-Raphson update
        if abs(npv_derivative) < 1e-10:
            return None  # Derivative too small

        rate = rate - npv / npv_derivative

    return None  # Did not converge


def calculate_payback_period(
    cash_flows: List[Number], initial_investment: Number
) -> Optional[float]:
    """
    Calculate the payback period (time to recover initial investment).

    Args:
        cash_flows: List of cash flows for each period
        initial_investment: Initial investment (positive number)

    Returns:
        Payback period in number of periods, or None if not recovered

    Examples:
        >>> calculate_payback_period([100, 100, 100], 250)
        2.5
        >>> calculate_payback_period([50, 50, 50], 200)
        4.0
    """
    cumulative = 0
    for i, cash_flow in enumerate(cash_flows):
        cumulative += cash_flow
        if cumulative >= initial_investment:
            # Interpolate to find exact period
            if i == 0:
                return initial_investment / cash_flow
            previous_cumulative = cumulative - cash_flow
            remaining = initial_investment - previous_cumulative
            return i + (remaining / cash_flow)

    return None  # Investment not recovered


def calculate_lcoe(
    total_costs: Number,
    total_energy_production: Number,
    discount_rate: float,
    lifetime_years: int,
) -> float:
    """
    Calculate Levelized Cost of Energy (LCOE) for a PV system.

    LCOE is the average cost per unit of energy produced over the lifetime
    of the system, accounting for time value of money.

    Args:
        total_costs: Total system costs (installation + O&M)
        total_energy_production: Total energy production over lifetime (kWh)
        discount_rate: Discount rate (as decimal)
        lifetime_years: System lifetime in years

    Returns:
        LCOE in currency units per kWh

    Raises:
        ValueError: If total_energy_production is zero

    Examples:
        >>> calculate_lcoe(10000, 50000, 0.05, 25)
        0.14145...
    """
    if total_energy_production == 0:
        raise ValueError("Total energy production cannot be zero")

    # Calculate present value denominator
    pv_denominator = sum(
        1 / ((1 + discount_rate) ** year) for year in range(1, lifetime_years + 1)
    )

    lcoe = total_costs / (total_energy_production * pv_denominator)
    return lcoe


# ============================================================================
# PV System Technical Calculations
# ============================================================================


def calculate_panel_efficiency(
    power_output_w: Number,
    area_m2: Number,
    irradiance_w_m2: Number = 1000,
) -> float:
    """
    Calculate PV panel efficiency.

    Args:
        power_output_w: Power output in Watts
        area_m2: Panel area in square meters
        irradiance_w_m2: Solar irradiance in W/m² (default: 1000 for STC)

    Returns:
        Efficiency as decimal (0-1)

    Raises:
        ValueError: If area or irradiance is zero

    Examples:
        >>> calculate_panel_efficiency(300, 1.6, 1000)
        0.1875
        >>> calculate_panel_efficiency(400, 2.0, 1000)
        0.2
    """
    if area_m2 == 0:
        raise ValueError("Area cannot be zero")
    if irradiance_w_m2 == 0:
        raise ValueError("Irradiance cannot be zero")

    return power_output_w / (area_m2 * irradiance_w_m2)


def calculate_temperature_derating(
    module_temp_c: Number,
    stc_temp_c: Number = 25,
    temp_coeff_percent_per_c: Number = -0.4,
) -> float:
    """
    Calculate power derating factor due to temperature.

    Args:
        module_temp_c: Actual module temperature in °C
        stc_temp_c: Standard Test Condition temperature (default: 25°C)
        temp_coeff_percent_per_c: Temperature coefficient of power (%/°C)

    Returns:
        Derating factor (e.g., 0.95 means 5% power loss)

    Examples:
        >>> calculate_temperature_derating(45, 25, -0.4)
        0.92
        >>> calculate_temperature_derating(25, 25, -0.4)
        1.0
    """
    temp_diff = module_temp_c - stc_temp_c
    derating_percent = temp_coeff_percent_per_c * temp_diff
    return 1 + (derating_percent / 100)


def calculate_performance_ratio(
    actual_energy_kwh: Number,
    theoretical_energy_kwh: Number,
) -> float:
    """
    Calculate Performance Ratio (PR) of a PV system.

    PR is the ratio of actual to theoretical energy production,
    accounting for all system losses.

    Args:
        actual_energy_kwh: Actual energy produced
        theoretical_energy_kwh: Theoretical energy (based on rated power and irradiance)

    Returns:
        Performance ratio as decimal (0-1)

    Raises:
        ValueError: If theoretical_energy_kwh is zero

    Examples:
        >>> calculate_performance_ratio(8500, 10000)
        0.85
        >>> calculate_performance_ratio(9200, 10000)
        0.92
    """
    if theoretical_energy_kwh == 0:
        raise ValueError("Theoretical energy cannot be zero")

    return actual_energy_kwh / theoretical_energy_kwh


def calculate_capacity_factor(
    actual_energy_kwh: Number,
    rated_power_kw: Number,
    period_hours: Number,
) -> float:
    """
    Calculate Capacity Factor of a PV system.

    Capacity factor is the ratio of actual energy production to the
    maximum possible production if the system ran at full capacity.

    Args:
        actual_energy_kwh: Actual energy produced
        rated_power_kw: Rated system power
        period_hours: Time period in hours

    Returns:
        Capacity factor as decimal (0-1)

    Raises:
        ValueError: If rated_power_kw or period_hours is zero

    Examples:
        >>> calculate_capacity_factor(1000, 5, 8760)  # 1 year
        0.0228...
        >>> calculate_capacity_factor(500, 2, 720)  # 1 month
        0.347...
    """
    if rated_power_kw == 0:
        raise ValueError("Rated power cannot be zero")
    if period_hours == 0:
        raise ValueError("Period hours cannot be zero")

    max_possible_kwh = rated_power_kw * period_hours
    return actual_energy_kwh / max_possible_kwh


def calculate_degradation_factor(
    degradation_rate_percent_per_year: Number,
    years: Number,
) -> float:
    """
    Calculate cumulative degradation factor over time.

    Args:
        degradation_rate_percent_per_year: Annual degradation rate (%)
        years: Number of years

    Returns:
        Degradation factor (e.g., 0.95 means 5% capacity loss)

    Examples:
        >>> calculate_degradation_factor(0.5, 10)
        0.951...
        >>> calculate_degradation_factor(1.0, 25)
        0.778...
    """
    annual_factor = 1 - (degradation_rate_percent_per_year / 100)
    return annual_factor ** years


# ============================================================================
# Circular Economy Calculations
# ============================================================================


def calculate_material_recovery_rate(
    recovered_mass_kg: Number,
    total_mass_kg: Number,
) -> float:
    """
    Calculate material recovery rate for recycling analysis.

    Args:
        recovered_mass_kg: Mass of recovered material
        total_mass_kg: Total mass of material in product

    Returns:
        Recovery rate as decimal (0-1)

    Raises:
        ValueError: If total_mass_kg is zero

    Examples:
        >>> calculate_material_recovery_rate(85, 100)
        0.85
        >>> calculate_material_recovery_rate(120, 150)
        0.8
    """
    if total_mass_kg == 0:
        raise ValueError("Total mass cannot be zero")

    return recovered_mass_kg / total_mass_kg


def calculate_circular_economy_score(
    recyclability: float,
    reusability: float,
    renewable_content: float,
    weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    """
    Calculate overall circular economy score (0-100).

    Combines recyclability, reusability, and renewable content into
    a single score representing circularity performance.

    Args:
        recyclability: Recyclability score (0-1)
        reusability: Reusability score (0-1)
        renewable_content: Renewable content score (0-1)
        weights: Optional tuple of weights for (recyclability, reusability, renewable)
                Default: (0.4, 0.3, 0.3)

    Returns:
        Circular economy score (0-100)

    Examples:
        >>> calculate_circular_economy_score(0.8, 0.6, 0.4)
        64.0
        >>> calculate_circular_economy_score(0.9, 0.7, 0.5, (0.5, 0.3, 0.2))
        75.0
    """
    if weights is None:
        weights = (0.4, 0.3, 0.3)

    score = (
        recyclability * weights[0]
        + reusability * weights[1]
        + renewable_content * weights[2]
    )

    return score * 100


def calculate_carbon_footprint_reduction(
    virgin_material_emissions_kg_co2: Number,
    recycled_material_emissions_kg_co2: Number,
    recycling_rate: float,
) -> float:
    """
    Calculate carbon footprint reduction from using recycled materials.

    Args:
        virgin_material_emissions_kg_co2: CO2 emissions from virgin material
        recycled_material_emissions_kg_co2: CO2 emissions from recycled material
        recycling_rate: Percentage of recycled content (0-1)

    Returns:
        CO2 reduction in kg

    Examples:
        >>> calculate_carbon_footprint_reduction(100, 20, 0.5)
        40.0
        >>> calculate_carbon_footprint_reduction(500, 100, 0.8)
        320.0
    """
    baseline_emissions = virgin_material_emissions_kg_co2
    actual_emissions = (
        virgin_material_emissions_kg_co2 * (1 - recycling_rate)
        + recycled_material_emissions_kg_co2 * recycling_rate
    )

    return baseline_emissions - actual_emissions


# ============================================================================
# General Math Utilities
# ============================================================================


def clamp(value: Number, min_value: Number, max_value: Number) -> Number:
    """
    Clamp a value between minimum and maximum bounds.

    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Clamped value

    Examples:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_value, min(value, max_value))


def linear_interpolation(
    x: Number, x1: Number, y1: Number, x2: Number, y2: Number
) -> float:
    """
    Perform linear interpolation between two points.

    Args:
        x: The x value to interpolate
        x1: First point x coordinate
        y1: First point y coordinate
        x2: Second point x coordinate
        y2: Second point y coordinate

    Returns:
        Interpolated y value

    Raises:
        ValueError: If x1 equals x2

    Examples:
        >>> linear_interpolation(5, 0, 0, 10, 100)
        50.0
        >>> linear_interpolation(7.5, 5, 20, 10, 40)
        30.0
    """
    if x1 == x2:
        raise ValueError("x1 and x2 cannot be equal")

    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def round_to_significant_figures(value: Number, sig_figs: int) -> float:
    """
    Round a number to specified significant figures.

    Args:
        value: Number to round
        sig_figs: Number of significant figures

    Returns:
        Rounded value

    Examples:
        >>> round_to_significant_figures(12345, 3)
        12300.0
        >>> round_to_significant_figures(0.0012345, 2)
        0.0012
    """
    if value == 0:
        return 0.0

    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)
