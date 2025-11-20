"""
Calculations Utilities
======================

Mathematical calculations and PV-specific formulas.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd


def calculate_module_power(
    cell_efficiency: float,
    num_cells: int,
    cell_area: float,
    ctm_ratio: float = 0.95
) -> float:
    """
    Calculate module power output.

    Args:
        cell_efficiency: Cell efficiency (%)
        num_cells: Number of cells in module
        cell_area: Area of each cell (m²)
        ctm_ratio: Cell-to-module ratio

    Returns:
        Module power (W)
    """
    stc_irradiance = 1000  # W/m²
    total_cell_area = num_cells * cell_area
    cell_power = (cell_efficiency / 100) * stc_irradiance * total_cell_area
    module_power = cell_power * ctm_ratio

    return module_power


def calculate_ctm_ratio(k_factors: dict) -> float:
    """
    Calculate total Cell-to-Module ratio from k-factors.

    Args:
        k_factors: Dictionary of k-factor values

    Returns:
        Total CTM ratio
    """
    total_ratio = 1.0

    for key, value in k_factors.items():
        if key.startswith('k') and isinstance(value, (int, float)):
            total_ratio *= value

    return total_ratio


def calculate_performance_ratio(
    actual_energy: float,
    nameplate_capacity: float,
    irradiation: float,
    stc_irradiance: float = 1000
) -> float:
    """
    Calculate Performance Ratio (PR).

    Args:
        actual_energy: Actual energy produced (kWh)
        nameplate_capacity: System nameplate capacity (kWp)
        irradiation: Total irradiation (kWh/m²)
        stc_irradiance: STC irradiance (W/m²)

    Returns:
        Performance ratio (%)
    """
    expected_energy = nameplate_capacity * irradiation / (stc_irradiance / 1000)
    pr = (actual_energy / expected_energy) * 100 if expected_energy > 0 else 0

    return pr


def calculate_capacity_factor(
    actual_energy: float,
    nameplate_capacity: float,
    hours: float
) -> float:
    """
    Calculate Capacity Factor.

    Args:
        actual_energy: Actual energy produced (kWh)
        nameplate_capacity: System nameplate capacity (kW)
        hours: Number of hours in period

    Returns:
        Capacity factor (%)
    """
    max_possible_energy = nameplate_capacity * hours
    cf = (actual_energy / max_possible_energy) * 100 if max_possible_energy > 0 else 0

    return cf


def calculate_specific_yield(
    energy: float,
    nameplate_capacity: float
) -> float:
    """
    Calculate Specific Yield.

    Args:
        energy: Energy produced (kWh)
        nameplate_capacity: System nameplate capacity (kWp)

    Returns:
        Specific yield (kWh/kWp)
    """
    return energy / nameplate_capacity if nameplate_capacity > 0 else 0


def calculate_degradation_rate(
    pr_values: List[float],
    timestamps: List[datetime]
) -> Tuple[float, float]:
    """
    Calculate degradation rate from PR time series.

    Args:
        pr_values: List of performance ratio values (%)
        timestamps: Corresponding timestamps

    Returns:
        Tuple of (degradation_rate %/year, r_squared)
    """
    if len(pr_values) < 2:
        return 0.0, 0.0

    # Convert timestamps to days since first measurement
    days = [(ts - timestamps[0]).days for ts in timestamps]
    years = [d / 365.25 for d in days]

    # Linear regression
    coeffs = np.polyfit(years, pr_values, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Calculate R²
    pr_pred = [slope * y + intercept for y in years]
    ss_res = sum((pr - pr_p) ** 2 for pr, pr_p in zip(pr_values, pr_pred))
    ss_tot = sum((pr - np.mean(pr_values)) ** 2 for pr in pr_values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Degradation rate (absolute value of slope)
    degradation_rate = abs(slope)

    return degradation_rate, r_squared


def calculate_temperature_corrected_power(
    power_stc: float,
    module_temp: float,
    temp_coefficient: float,
    stc_temp: float = 25
) -> float:
    """
    Calculate power output corrected for temperature.

    Args:
        power_stc: Power at STC (W)
        module_temp: Actual module temperature (°C)
        temp_coefficient: Temperature coefficient (%/°C)
        stc_temp: STC temperature (°C)

    Returns:
        Temperature-corrected power (W)
    """
    temp_diff = module_temp - stc_temp
    power_factor = 1 + (temp_coefficient / 100) * temp_diff
    corrected_power = power_stc * power_factor

    return corrected_power


def calculate_module_temperature(
    ambient_temp: float,
    irradiance: float,
    wind_speed: float,
    noct: float = 45,
    mounting_type: str = "open_rack"
) -> float:
    """
    Calculate module temperature using NOCT model.

    Args:
        ambient_temp: Ambient temperature (°C)
        irradiance: Irradiance (W/m²)
        wind_speed: Wind speed (m/s)
        noct: Nominal Operating Cell Temperature (°C)
        mounting_type: Type of mounting ("open_rack", "close_roof", "insulated_back")

    Returns:
        Module temperature (°C)
    """
    # Temperature rise factor based on mounting type
    mounting_factors = {
        "open_rack": 1.0,
        "close_roof": 1.2,
        "insulated_back": 1.4
    }

    factor = mounting_factors.get(mounting_type, 1.0)

    # Wind speed correction
    wind_correction = max(1 - 0.04 * (wind_speed - 1), 0.7)

    # Module temperature
    temp_rise = ((noct - 20) / 800) * irradiance * factor * wind_correction
    module_temp = ambient_temp + temp_rise

    return module_temp


def calculate_dc_ac_ratio(
    dc_capacity: float,
    ac_capacity: float
) -> float:
    """
    Calculate DC/AC ratio (inverter loading ratio).

    Args:
        dc_capacity: DC capacity (kWp)
        ac_capacity: AC capacity (kW)

    Returns:
        DC/AC ratio
    """
    return dc_capacity / ac_capacity if ac_capacity > 0 else 0


def calculate_lcoe(
    capex: float,
    opex_annual: float,
    energy_annual: float,
    lifetime: int,
    discount_rate: float,
    degradation_rate: float = 0.5
) -> float:
    """
    Calculate Levelized Cost of Energy (LCOE).

    Args:
        capex: Capital expenditure ($)
        opex_annual: Annual operating expenditure ($)
        energy_annual: Annual energy production (kWh)
        lifetime: Project lifetime (years)
        discount_rate: Discount rate (decimal, e.g., 0.08 for 8%)
        degradation_rate: Annual degradation rate (%/year)

    Returns:
        LCOE ($/kWh)
    """
    # Calculate NPV of costs
    npv_costs = capex
    for year in range(1, lifetime + 1):
        npv_costs += opex_annual / ((1 + discount_rate) ** year)

    # Calculate NPV of energy
    npv_energy = 0
    for year in range(1, lifetime + 1):
        energy_year = energy_annual * ((1 - degradation_rate / 100) ** (year - 1))
        npv_energy += energy_year / ((1 + discount_rate) ** year)

    lcoe = npv_costs / npv_energy if npv_energy > 0 else 0

    return lcoe


def calculate_npv(
    capex: float,
    revenue_annual: float,
    opex_annual: float,
    lifetime: int,
    discount_rate: float,
    degradation_rate: float = 0.5
) -> float:
    """
    Calculate Net Present Value (NPV).

    Args:
        capex: Capital expenditure ($)
        revenue_annual: Annual revenue ($)
        opex_annual: Annual operating expenditure ($)
        lifetime: Project lifetime (years)
        discount_rate: Discount rate (decimal)
        degradation_rate: Annual degradation rate (%/year)

    Returns:
        NPV ($)
    """
    npv = -capex

    for year in range(1, lifetime + 1):
        # Revenue decreases with degradation
        revenue_year = revenue_annual * ((1 - degradation_rate / 100) ** (year - 1))
        cash_flow = revenue_year - opex_annual
        npv += cash_flow / ((1 + discount_rate) ** year)

    return npv


def calculate_payback_period(
    capex: float,
    revenue_annual: float,
    opex_annual: float,
    degradation_rate: float = 0.5
) -> float:
    """
    Calculate simple payback period.

    Args:
        capex: Capital expenditure ($)
        revenue_annual: Annual revenue ($)
        opex_annual: Annual operating expenditure ($)
        degradation_rate: Annual degradation rate (%/year)

    Returns:
        Payback period (years)
    """
    cumulative_cash_flow = -capex
    year = 0

    while cumulative_cash_flow < 0 and year < 50:  # Max 50 years
        year += 1
        revenue_year = revenue_annual * ((1 - degradation_rate / 100) ** (year - 1))
        annual_cash_flow = revenue_year - opex_annual
        cumulative_cash_flow += annual_cash_flow

    return year


def calculate_string_voltage(
    modules_per_string: int,
    vmp: float,
    voc: float,
    temperature: float = 25,
    temp_coeff_voc: float = -0.28,
    temp_coeff_vmp: float = -0.35
) -> Tuple[float, float]:
    """
    Calculate string voltage at given temperature.

    Args:
        modules_per_string: Number of modules in string
        vmp: Module Vmp at STC (V)
        voc: Module Voc at STC (V)
        temperature: Operating temperature (°C)
        temp_coeff_voc: Temperature coefficient of Voc (%/°C)
        temp_coeff_vmp: Temperature coefficient of Vmp (%/°C)

    Returns:
        Tuple of (string_vmp, string_voc) at given temperature
    """
    # Temperature correction
    temp_diff = temperature - 25  # STC temperature

    vmp_corrected = vmp * (1 + temp_coeff_vmp / 100 * temp_diff)
    voc_corrected = voc * (1 + temp_coeff_voc / 100 * temp_diff)

    string_vmp = vmp_corrected * modules_per_string
    string_voc = voc_corrected * modules_per_string

    return string_vmp, string_voc


def calculate_array_losses(
    soiling: float = 2.0,
    shading: float = 3.0,
    snow: float = 0.0,
    mismatch: float = 2.0,
    wiring: float = 2.0,
    connections: float = 0.5,
    lid: float = 1.5,
    nameplate: float = 1.0,
    age: float = 0.5
) -> Tuple[float, dict]:
    """
    Calculate total array losses.

    Args:
        soiling: Soiling losses (%)
        shading: Shading losses (%)
        snow: Snow losses (%)
        mismatch: Mismatch losses (%)
        wiring: DC wiring losses (%)
        connections: Connection losses (%)
        lid: Light-induced degradation (%)
        nameplate: Nameplate rating tolerance (%)
        age: Age-related degradation (%)

    Returns:
        Tuple of (total_loss_factor, loss_breakdown_dict)
    """
    losses = {
        'soiling': soiling,
        'shading': shading,
        'snow': snow,
        'mismatch': mismatch,
        'wiring': wiring,
        'connections': connections,
        'lid': lid,
        'nameplate': nameplate,
        'age': age
    }

    # Compound losses
    total_factor = 1.0
    for loss_pct in losses.values():
        total_factor *= (1 - loss_pct / 100)

    total_loss = (1 - total_factor) * 100

    return total_factor, losses
