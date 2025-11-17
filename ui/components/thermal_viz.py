"""
Thermal visualization components for temperature charts and analysis.

This module provides interactive Plotly charts for thermal modeling results,
including temperature comparisons, cooling analysis, and heat transfer visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any


def plot_temperature_comparison(
    results: Dict[str, Any],
    conditions: Any,
) -> go.Figure:
    """
    Create bar chart comparing cell temperatures from different models.

    Args:
        results: Dictionary of model results {model_name: ThermalModelOutput}
        conditions: TemperatureConditions object

    Returns:
        Plotly figure object
    """
    model_names = list(results.keys())
    cell_temps = [result.cell_temperature for result in results.values()]
    module_temps = [result.module_temperature for result in results.values()]

    fig = go.Figure()

    # Cell temperature bars
    fig.add_trace(go.Bar(
        name='Cell Temperature',
        x=model_names,
        y=cell_temps,
        marker_color='#ff6b6b',
        text=[f"{temp:.1f}°C" for temp in cell_temps],
        textposition='outside',
    ))

    # Module temperature bars
    fig.add_trace(go.Bar(
        name='Module Temperature',
        x=model_names,
        y=module_temps,
        marker_color='#4ecdc4',
        text=[f"{temp:.1f}°C" for temp in module_temps],
        textposition='outside',
    ))

    # Add ambient temperature line
    fig.add_hline(
        y=conditions.ambient_temp,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Ambient: {conditions.ambient_temp}°C",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"Temperature Model Comparison<br><sub>Irradiance: {conditions.irradiance}W/m² | Wind: {conditions.wind_speed}m/s</sub>",
        xaxis_title="Model",
        yaxis_title="Temperature (°C)",
        barmode='group',
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def plot_wind_speed_effects(wind_results: pd.DataFrame) -> go.Figure:
    """
    Plot the effect of wind speed on temperature and heat transfer.

    Args:
        wind_results: DataFrame with wind speed analysis results

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Heat Transfer Coefficient vs Wind Speed",
            "Temperature Reduction vs Wind Speed"
        ),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Plot 1: Heat transfer coefficients
    fig.add_trace(
        go.Scatter(
            x=wind_results['wind_speed_ms'],
            y=wind_results['h_conv_front'],
            mode='lines+markers',
            name='Convective h',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=wind_results['wind_speed_ms'],
            y=wind_results['h_total_front'],
            mode='lines+markers',
            name='Total h',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
        ),
        row=1, col=1
    )

    # Plot 2: Temperature reduction
    fig.add_trace(
        go.Scatter(
            x=wind_results['wind_speed_ms'],
            y=wind_results['temp_reduction_c'],
            mode='lines+markers',
            name='Temp Reduction',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.2)',
        ),
        row=2, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Heat Transfer Coeff (W/m²·K)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature Reduction (°C)", row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
    )

    return fig


def plot_mounting_configuration_effects(mounting_results: pd.DataFrame) -> go.Figure:
    """
    Compare module temperatures across different mounting configurations.

    Args:
        mounting_results: DataFrame with mounting configuration comparison

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Sandia model
    fig.add_trace(go.Bar(
        name='Sandia',
        x=mounting_results['mounting_type'],
        y=mounting_results['cell_temp_sandia_c'],
        marker_color='#e74c3c',
    ))

    # PVsyst model
    fig.add_trace(go.Bar(
        name='PVsyst',
        x=mounting_results['mounting_type'],
        y=mounting_results['cell_temp_pvsyst_c'],
        marker_color='#3498db',
    ))

    # NOCT model
    fig.add_trace(go.Bar(
        name='NOCT-based',
        x=mounting_results['mounting_type'],
        y=mounting_results['cell_temp_noct_c'],
        marker_color='#2ecc71',
    ))

    # Average line
    fig.add_trace(go.Scatter(
        name='Average',
        x=mounting_results['mounting_type'],
        y=mounting_results['avg_cell_temp_c'],
        mode='lines+markers',
        line=dict(color='black', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
    ))

    fig.update_layout(
        title="Cell Temperature by Mounting Configuration",
        xaxis_title="Mounting Type",
        yaxis_title="Cell Temperature (°C)",
        barmode='group',
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def plot_temperature_time_series(ts_df: pd.DataFrame) -> go.Figure:
    """
    Plot temperature time series with environmental conditions.

    Args:
        ts_df: DataFrame with time series data (hour, temps, irradiance, etc.)

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Cell Temperature Over Time",
            "Solar Irradiance",
            "Wind Speed"
        ),
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )

    # Plot 1: Cell temperatures
    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['cell_temp_sandia'],
            mode='lines',
            name='Sandia',
            line=dict(color='#e74c3c', width=2),
        ),
        row=1, col=1, secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['cell_temp_pvsyst'],
            mode='lines',
            name='PVsyst',
            line=dict(color='#3498db', width=2),
        ),
        row=1, col=1, secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['cell_temp_noct'],
            mode='lines',
            name='NOCT',
            line=dict(color='#2ecc71', width=2),
        ),
        row=1, col=1, secondary_y=False
    )

    # Ambient temperature on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['ambient_temp'],
            mode='lines',
            name='Ambient',
            line=dict(color='gray', width=2, dash='dash'),
        ),
        row=1, col=1, secondary_y=True
    )

    # Plot 2: Irradiance
    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['irradiance'],
            mode='lines',
            name='Irradiance',
            line=dict(color='#f39c12', width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.3)',
            showlegend=False,
        ),
        row=2, col=1
    )

    # Plot 3: Wind speed
    fig.add_trace(
        go.Scatter(
            x=ts_df['hour'],
            y=ts_df['wind_speed'],
            mode='lines',
            name='Wind Speed',
            line=dict(color='#16a085', width=2),
            fill='tozeroy',
            fillcolor='rgba(22, 160, 133, 0.3)',
            showlegend=False,
        ),
        row=3, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
    fig.update_yaxes(title_text="Cell Temp (°C)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Ambient (°C)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Irradiance (W/m²)", row=2, col=1)
    fig.update_yaxes(title_text="Wind Speed (m/s)", row=3, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5
        ),
    )

    return fig


def plot_heat_transfer_coefficients(coeffs: Any) -> go.Figure:
    """
    Create breakdown chart of heat transfer coefficients.

    Args:
        coeffs: HeatTransferCoefficients object

    Returns:
        Plotly figure object
    """
    categories = ['Front\nConvective', 'Front\nRadiative', 'Back\nConvective', 'Back\nRadiative']
    values = [
        coeffs.convective_front,
        coeffs.radiative_front,
        coeffs.convective_back,
        coeffs.radiative_back,
    ]
    colors = ['#ff6b6b', '#ff8787', '#4ecdc4', '#6eddd6']

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
    )])

    # Add total lines
    fig.add_hline(
        y=coeffs.total_front,
        line_dash="dash",
        line_color="#ff6b6b",
        annotation_text=f"Front Total: {coeffs.total_front:.2f}",
        annotation_position="right",
    )

    fig.add_hline(
        y=coeffs.total_back,
        line_dash="dash",
        line_color="#4ecdc4",
        annotation_text=f"Back Total: {coeffs.total_back:.2f}",
        annotation_position="left",
    )

    fig.update_layout(
        title="Heat Transfer Coefficient Breakdown",
        xaxis_title="Surface / Type",
        yaxis_title="Heat Transfer Coefficient (W/m²·K)",
        height=500,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_thermal_time_constants(tau_results: Dict[str, float]) -> go.Figure:
    """
    Visualize thermal time constants and response times.

    Args:
        tau_results: Dictionary with time constant results

    Returns:
        Plotly figure object
    """
    categories = ['Heating τ', 'Cooling τ', '63% Response', '95% Response']
    values_minutes = [
        tau_results['tau_heating_minutes'],
        tau_results['tau_cooling_minutes'],
        tau_results['response_time_63'] / 60,
        tau_results['response_time_95'] / 60,
    ]
    colors = ['#ff6b6b', '#4ecdc4', '#feca57', '#ff9ff3']

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values_minutes,
        marker_color=colors,
        text=[f"{v:.1f} min" for v in values_minutes],
        textposition='outside',
    )])

    fig.update_layout(
        title="Thermal Response Time Constants",
        xaxis_title="Time Constant Type",
        yaxis_title="Time (minutes)",
        height=450,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_cooling_rate_analysis(
    initial_temp: float,
    ambient_temp: float,
    tau_cooling: float,
    duration_minutes: float = 60.0,
) -> go.Figure:
    """
    Plot module cooling curve over time.

    Args:
        initial_temp: Initial module temperature (°C)
        ambient_temp: Ambient temperature (°C)
        tau_cooling: Cooling time constant (seconds)
        duration_minutes: Duration to simulate (minutes)

    Returns:
        Plotly figure object
    """
    time_seconds = np.linspace(0, duration_minutes * 60, 500)
    time_minutes = time_seconds / 60

    # Exponential cooling: T(t) = T_amb + (T0 - T_amb) * exp(-t/τ)
    temp_curve = ambient_temp + (initial_temp - ambient_temp) * np.exp(-time_seconds / tau_cooling)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_minutes,
        y=temp_curve,
        mode='lines',
        name='Module Temperature',
        line=dict(color='#e74c3c', width=3),
        fill='tonexty',
    ))

    # Add ambient line
    fig.add_hline(
        y=ambient_temp,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Ambient: {ambient_temp}°C",
    )

    # Add time constant markers
    tau_minutes = tau_cooling / 60
    temp_at_tau = ambient_temp + (initial_temp - ambient_temp) * np.exp(-1)

    fig.add_trace(go.Scatter(
        x=[tau_minutes],
        y=[temp_at_tau],
        mode='markers',
        name=f'τ = {tau_minutes:.1f} min',
        marker=dict(size=15, color='orange', symbol='star'),
    ))

    fig.update_layout(
        title="Module Cooling Curve",
        xaxis_title="Time (minutes)",
        yaxis_title="Temperature (°C)",
        height=500,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_performance_vs_temperature(
    temperatures: np.ndarray,
    rated_power: float,
    temp_coeff: float,
    ref_temp: float = 25.0,
) -> go.Figure:
    """
    Plot power output vs temperature for a module.

    Args:
        temperatures: Array of temperatures (°C)
        rated_power: Rated power at STC (W)
        temp_coeff: Power temperature coefficient (1/°C)
        ref_temp: Reference temperature (°C)

    Returns:
        Plotly figure object
    """
    # Calculate power at each temperature
    # P(T) = P_ref * (1 + temp_coeff * (T - T_ref))
    power_output = rated_power * (1 + temp_coeff * (temperatures - ref_temp))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temperatures,
        y=power_output,
        mode='lines',
        name='Power Output',
        line=dict(color='#2ecc71', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)',
    ))

    # Add STC reference
    fig.add_vline(
        x=ref_temp,
        line_dash="dash",
        line_color="red",
        annotation_text=f"STC ({ref_temp}°C)",
    )

    fig.update_layout(
        title=f"Power Output vs Temperature<br><sub>Rated: {rated_power}W | Temp Coeff: {temp_coeff*100:.3f}%/°C</sub>",
        xaxis_title="Cell Temperature (°C)",
        yaxis_title="Power Output (W)",
        height=500,
        template="plotly_white",
        showlegend=False,
    )

    return fig
