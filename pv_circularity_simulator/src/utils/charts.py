"""
Charts Utilities
================

Plotting and visualization utilities using Plotly and Matplotlib.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt


def create_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a line chart using Plotly.

    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: List of column names for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for col in y_columns:
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[col],
            mode='lines+markers',
            name=col
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        **kwargs
    )

    return fig


def create_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a bar chart using Plotly.

    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color: Column name for color coding
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = px.bar(
        data,
        x=x_column,
        y=y_column,
        color=color,
        title=title,
        labels={x_column: x_label, y_column: y_label},
        template='plotly_white',
        **kwargs
    )

    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a pie chart using Plotly.

    Args:
        labels: Category labels
        values: Values for each category
        title: Chart title
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3  # Makes it a donut chart
    )])

    fig.update_layout(
        title=title,
        template='plotly_white',
        **kwargs
    )

    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colorscale: str = "Viridis",
    **kwargs
) -> go.Figure:
    """
    Create a heatmap using Plotly.

    Args:
        data: DataFrame containing the data
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        colorscale: Color scale for heatmap
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        **kwargs
    )

    return fig


def create_waterfall_chart(
    categories: List[str],
    values: List[float],
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a waterfall chart for loss analysis.

    Args:
        categories: Category names
        values: Values for each category (positive or negative)
        title: Chart title
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Waterfall(
        name="Losses",
        orientation="v",
        measure=["relative"] * (len(categories) - 1) + ["total"],
        x=categories,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        template='plotly_white',
        **kwargs
    )

    return fig


def create_sankey_diagram(
    source: List[int],
    target: List[int],
    value: List[float],
    labels: List[str],
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a Sankey diagram for material/energy flow.

    Args:
        source: List of source node indices
        target: List of target node indices
        value: List of flow values
        labels: Node labels
        title: Chart title
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(
        title=title,
        font_size=10,
        **kwargs
    )

    return fig


def create_gauge_chart(
    value: float,
    max_value: float = 100,
    title: str = "",
    thresholds: Optional[Dict[str, float]] = None,
    **kwargs
) -> go.Figure:
    """
    Create a gauge chart for KPI display.

    Args:
        value: Current value
        max_value: Maximum value for gauge
        title: Chart title
        thresholds: Dictionary of threshold colors
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    if thresholds is None:
        thresholds = {
            'red': max_value * 0.5,
            'yellow': max_value * 0.75,
            'green': max_value
        }

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds.get('red', max_value * 0.5)], 'color': "lightcoral"},
                {'range': [thresholds.get('red', max_value * 0.5),
                          thresholds.get('yellow', max_value * 0.75)], 'color': "lightyellow"},
                {'range': [thresholds.get('yellow', max_value * 0.75), max_value], 'color': "lightgreen"}
            ],
        }
    ))

    fig.update_layout(**kwargs)

    return fig


def create_time_series_forecast(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "",
    confidence_intervals: bool = True,
    **kwargs
) -> go.Figure:
    """
    Create a time series chart with historical and forecast data.

    Args:
        historical_data: DataFrame with historical data
        forecast_data: DataFrame with forecast data
        date_column: Column name for dates
        value_column: Column name for values
        title: Chart title
        confidence_intervals: Whether to show confidence intervals
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data[date_column],
        y=historical_data[value_column],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))

    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data[date_column],
        y=forecast_data[value_column],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Confidence intervals
    if confidence_intervals and 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data[date_column],
            y=forecast_data['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_data[date_column],
            y=forecast_data['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0)',
            name='95% Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=value_column,
        template='plotly_white',
        **kwargs
    )

    return fig


def create_solar_position_chart(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    title: str = "Solar Position",
    **kwargs
) -> go.Figure:
    """
    Create a polar plot of solar position (sun path).

    Args:
        azimuth: Array of azimuth angles (degrees)
        elevation: Array of elevation angles (degrees)
        title: Chart title
        **kwargs: Additional Plotly figure parameters

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=90 - elevation,  # Convert elevation to radius (zenith angle)
        theta=azimuth,
        mode='markers+lines',
        name='Sun Path'
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 90],
                angle=90,
                tickvals=[0, 30, 60, 90],
                ticktext=['90°', '60°', '30°', '0°']
            ),
            angularaxis=dict(
                direction='clockwise',
                period=360
            )
        ),
        **kwargs
    )

    return fig
