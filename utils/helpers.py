"""
Utility helper functions for PV Circularity Simulator.

This module provides common helper functions used across the application
including calculations, data processing, and visualization utilities.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# PV PERFORMANCE CALCULATIONS
# ============================================================================

def calculate_performance_ratio(
    actual_energy: float,
    expected_energy: float
) -> float:
    """
    Calculate Performance Ratio (PR).

    Args:
        actual_energy: Actual energy produced (kWh)
        expected_energy: Expected energy at rated conditions (kWh)

    Returns:
        Performance ratio (0-1)
    """
    if expected_energy <= 0:
        return 0.0
    return min(actual_energy / expected_energy, 1.5)  # Cap at 150%


def calculate_specific_yield(
    energy_kwh: float,
    capacity_kwp: float,
    days: int = 1
) -> float:
    """
    Calculate specific yield (kWh/kWp/day).

    Args:
        energy_kwh: Energy produced (kWh)
        capacity_kwp: System capacity (kWp)
        days: Number of days

    Returns:
        Specific yield (kWh/kWp/day)
    """
    if capacity_kwp <= 0 or days <= 0:
        return 0.0
    return energy_kwh / (capacity_kwp * days)


def calculate_capacity_factor(
    actual_energy: float,
    rated_capacity: float,
    hours: float = 24
) -> float:
    """
    Calculate capacity factor.

    Args:
        actual_energy: Actual energy produced (kWh)
        rated_capacity: Rated capacity (kW)
        hours: Time period (hours)

    Returns:
        Capacity factor (0-1)
    """
    if rated_capacity <= 0 or hours <= 0:
        return 0.0
    max_energy = rated_capacity * hours
    return actual_energy / max_energy


def temperature_corrected_power(
    power_stc: float,
    temp_actual: float,
    temp_stc: float = 25.0,
    temp_coeff: float = -0.45
) -> float:
    """
    Calculate temperature-corrected power output.

    Args:
        power_stc: Power at STC (W)
        temp_actual: Actual cell temperature (°C)
        temp_stc: STC temperature (°C)
        temp_coeff: Temperature coefficient (%/°C)

    Returns:
        Temperature-corrected power (W)
    """
    temp_diff = temp_actual - temp_stc
    correction_factor = 1 + (temp_coeff / 100) * temp_diff
    return power_stc * correction_factor


def calculate_noct_temperature(
    ambient_temp: float,
    irradiance: float,
    noct: float = 45.0,
    irradiance_noct: float = 800.0
) -> float:
    """
    Calculate module temperature using NOCT.

    Args:
        ambient_temp: Ambient temperature (°C)
        irradiance: Irradiance (W/m²)
        noct: Nominal Operating Cell Temperature (°C)
        irradiance_noct: NOCT reference irradiance (W/m²)

    Returns:
        Module temperature (°C)
    """
    return ambient_temp + (noct - 20) * (irradiance / irradiance_noct)


def calculate_poa_irradiance(
    ghi: float,
    dni: float,
    dhi: float,
    solar_zenith: float,
    solar_azimuth: float,
    tilt: float,
    azimuth: float
) -> float:
    """
    Calculate plane-of-array (POA) irradiance.

    Args:
        ghi: Global horizontal irradiance (W/m²)
        dni: Direct normal irradiance (W/m²)
        dhi: Diffuse horizontal irradiance (W/m²)
        solar_zenith: Solar zenith angle (degrees)
        solar_azimuth: Solar azimuth angle (degrees)
        tilt: Module tilt angle (degrees)
        azimuth: Module azimuth angle (degrees)

    Returns:
        POA irradiance (W/m²)
    """
    # Convert to radians
    zenith_rad = np.radians(solar_zenith)
    azimuth_diff_rad = np.radians(solar_azimuth - azimuth)
    tilt_rad = np.radians(tilt)

    # Angle of incidence
    cos_aoi = (np.cos(zenith_rad) * np.cos(tilt_rad) +
               np.sin(zenith_rad) * np.sin(tilt_rad) * np.cos(azimuth_diff_rad))

    # Direct component
    direct = max(0, dni * cos_aoi)

    # Diffuse component (isotropic sky model)
    diffuse = dhi * (1 + np.cos(tilt_rad)) / 2

    # Ground-reflected component (albedo = 0.2)
    reflected = ghi * 0.2 * (1 - np.cos(tilt_rad)) / 2

    return direct + diffuse + reflected


# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_lcoe(
    total_capex: float,
    annual_energy: float,
    annual_opex: float,
    discount_rate: float,
    lifetime: int,
    degradation_rate: float = 0.005
) -> float:
    """
    Calculate Levelized Cost of Energy (LCOE).

    Args:
        total_capex: Total capital expenditure ($)
        annual_energy: First-year energy production (kWh)
        annual_opex: Annual operating expenditure ($)
        discount_rate: Discount rate (fraction)
        lifetime: Project lifetime (years)
        degradation_rate: Annual degradation rate (fraction)

    Returns:
        LCOE ($/kWh)
    """
    # Present value of costs
    pv_capex = total_capex
    pv_opex = sum(annual_opex / ((1 + discount_rate) ** year)
                  for year in range(1, lifetime + 1))
    pv_costs = pv_capex + pv_opex

    # Present value of energy
    pv_energy = sum(annual_energy * ((1 - degradation_rate) ** year) / ((1 + discount_rate) ** year)
                    for year in range(1, lifetime + 1))

    if pv_energy <= 0:
        return float('inf')

    return pv_costs / pv_energy


def calculate_npv(
    cash_flows: List[float],
    discount_rate: float,
    initial_investment: float
) -> float:
    """
    Calculate Net Present Value (NPV).

    Args:
        cash_flows: Annual cash flows ($)
        discount_rate: Discount rate (fraction)
        initial_investment: Initial investment ($)

    Returns:
        NPV ($)
    """
    pv = sum(cf / ((1 + discount_rate) ** (i + 1))
             for i, cf in enumerate(cash_flows))
    return pv - initial_investment


def calculate_irr(
    cash_flows: List[float],
    initial_investment: float,
    max_iter: int = 1000
) -> Optional[float]:
    """
    Calculate Internal Rate of Return (IRR) using Newton-Raphson method.

    Args:
        cash_flows: Annual cash flows ($)
        initial_investment: Initial investment ($)
        max_iter: Maximum iterations

    Returns:
        IRR (fraction) or None if not found
    """
    cash_flows_full = [-initial_investment] + cash_flows

    # Initial guess
    irr = 0.1

    for _ in range(max_iter):
        # NPV calculation
        npv = sum(cf / ((1 + irr) ** i) for i, cf in enumerate(cash_flows_full))

        # Derivative of NPV
        dnpv = sum(-i * cf / ((1 + irr) ** (i + 1)) for i, cf in enumerate(cash_flows_full))

        if abs(dnpv) < 1e-10:
            return None

        # Newton-Raphson update
        irr_new = irr - npv / dnpv

        if abs(irr_new - irr) < 1e-6:
            return irr_new

        irr = irr_new

    return None


def calculate_payback_period(
    initial_investment: float,
    annual_cash_flows: List[float]
) -> Optional[float]:
    """
    Calculate simple payback period.

    Args:
        initial_investment: Initial investment ($)
        annual_cash_flows: Annual cash flows ($)

    Returns:
        Payback period (years) or None if not achieved
    """
    cumulative = 0
    for year, cash_flow in enumerate(annual_cash_flows, start=1):
        cumulative += cash_flow
        if cumulative >= initial_investment:
            # Interpolate for fractional year
            previous_cumulative = cumulative - cash_flow
            fraction = (initial_investment - previous_cumulative) / cash_flow
            return year - 1 + fraction

    return None


# ============================================================================
# CIRCULARITY CALCULATIONS
# ============================================================================

def calculate_circularity_score(
    reuse_potential: float,
    repair_feasibility: float,
    recyclability: float,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    Calculate overall circularity score.

    Args:
        reuse_potential: Reuse potential score (0-100)
        repair_feasibility: Repair feasibility score (0-100)
        recyclability: Recyclability score (0-100)
        weights: Weights for (reuse, repair, recycle)

    Returns:
        Circularity score (0-100)
    """
    w_reuse, w_repair, w_recycle = weights
    score = (w_reuse * reuse_potential +
             w_repair * repair_feasibility +
             w_recycle * recyclability)
    return min(max(score, 0), 100)


def estimate_material_recovery(
    module_weight: float,
    composition: Dict[str, float],
    recovery_rates: Dict[str, float]
) -> Dict[str, float]:
    """
    Estimate recoverable materials from module recycling.

    Args:
        module_weight: Module weight (kg)
        composition: Material composition by weight fraction
        recovery_rates: Recovery rate for each material (0-1)

    Returns:
        Dictionary of recovered materials (kg)
    """
    recovered = {}
    for material, fraction in composition.items():
        material_weight = module_weight * fraction
        recovery_rate = recovery_rates.get(material, 0.8)
        recovered[material] = material_weight * recovery_rate

    return recovered


def calculate_recycling_revenue(
    recovered_materials: Dict[str, float],
    material_prices: Dict[str, float],
    recycling_cost_per_kg: float = 0.30
) -> Tuple[float, float]:
    """
    Calculate net recycling revenue.

    Args:
        recovered_materials: Recovered materials (kg)
        material_prices: Material prices ($/kg)
        recycling_cost_per_kg: Recycling cost ($/kg)

    Returns:
        Tuple of (gross_revenue, net_revenue) in $
    """
    gross_revenue = sum(
        weight * material_prices.get(material, 0)
        for material, weight in recovered_materials.items()
    )

    total_weight = sum(recovered_materials.values())
    recycling_cost = total_weight * recycling_cost_per_kg

    net_revenue = gross_revenue - recycling_cost

    return gross_revenue, net_revenue


# ============================================================================
# DATA PROCESSING
# ============================================================================

def resample_timeseries(
    df: pd.DataFrame,
    freq: str = 'H',
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """
    Resample time series data.

    Args:
        df: DataFrame with datetime index
        freq: Resampling frequency ('H', 'D', 'M', etc.)
        agg_func: Aggregation function ('mean', 'sum', 'max', 'min')

    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    return df.resample(freq).agg(agg_func)


def calculate_moving_average(
    data: Union[List[float], np.ndarray, pd.Series],
    window: int
) -> np.ndarray:
    """
    Calculate moving average.

    Args:
        data: Input data
        window: Window size

    Returns:
        Moving average array
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=window, center=True).mean().values
    else:
        data_array = np.array(data)
        return np.convolve(data_array, np.ones(window) / window, mode='same')


def detect_outliers(
    data: Union[List[float], np.ndarray, pd.Series],
    method: str = 'iqr',
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in data.

    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean array indicating outliers
    """
    data_array = np.array(data)

    if method == 'iqr':
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data_array < lower_bound) | (data_array > upper_bound)

    elif method == 'zscore':
        mean = np.mean(data_array)
        std = np.std(data_array)
        z_scores = np.abs((data_array - mean) / std)
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def create_performance_chart(
    dates: List[datetime],
    values: List[float],
    title: str,
    y_label: str,
    color: str = '#2ECC71'
) -> go.Figure:
    """
    Create a standard performance chart.

    Args:
        dates: List of datetime objects
        values: List of values
        title: Chart title
        y_label: Y-axis label
        color: Line color

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=6),
        name=y_label
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_comparison_bar_chart(
    categories: List[str],
    values1: List[float],
    values2: List[float],
    label1: str,
    label2: str,
    title: str
) -> go.Figure:
    """
    Create a comparison bar chart.

    Args:
        categories: Category labels
        values1: First set of values
        values2: Second set of values
        label1: Label for first set
        label2: Label for second set
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=values1,
        name=label1,
        marker_color='#3498DB'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=values2,
        name=label2,
        marker_color='#E74C3C'
    ))

    fig.update_layout(
        title=title,
        barmode='group',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
    colors: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a pie chart.

    Args:
        labels: Slice labels
        values: Slice values
        title: Chart title
        colors: Optional custom colors

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors) if colors else None,
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )

    return fig


def create_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    colorscale: str = 'RdYlGn'
) -> go.Figure:
    """
    Create a heatmap.

    Args:
        data: 2D array of values
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Chart title
        colorscale: Plotly colorscale

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        text=data,
        texttemplate='%{text:.1f}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500
    )

    return fig


def format_number(value: float, decimals: int = 2, unit: str = '') -> str:
    """
    Format number with appropriate suffix (K, M, B).

    Args:
        value: Number to format
        decimals: Number of decimal places
        unit: Unit suffix

    Returns:
        Formatted string
    """
    abs_value = abs(value)

    if abs_value >= 1e9:
        formatted = f"{value / 1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        formatted = f"{value / 1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        formatted = f"{value / 1e3:.{decimals}f}K"
    else:
        formatted = f"{value:.{decimals}f}"

    return f"{formatted}{unit}" if unit else formatted


def generate_date_range(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'D'
) -> List[datetime]:
    """
    Generate date range.

    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D', 'H', 'M', etc.)

    Returns:
        List of datetime objects
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq).tolist()
