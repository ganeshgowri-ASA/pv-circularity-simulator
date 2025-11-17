"""
Chart templates for PV Circularity visualizations.

This module provides pre-configured chart templates optimized for
photovoltaic system data visualization, including time series,
performance metrics, lifecycle analysis, and circular economy modeling.
"""

from typing import Optional, List, Dict, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import pandas as pd
import numpy as np


class ChartTemplates:
    """
    Pre-configured chart templates for PV system visualization.

    This class provides ready-to-use chart templates optimized for common
    PV circularity data visualization scenarios, including energy production,
    degradation analysis, lifecycle assessment, and circularity metrics.

    Examples:
        >>> templates = ChartTemplates()
        >>> fig = templates.time_series(
        ...     data=df,
        ...     x_col='timestamp',
        ...     y_cols=['power_output', 'temperature'],
        ...     title='PV System Performance'
        ... )
    """

    def __init__(self) -> None:
        """Initialize chart templates with default configurations."""
        self.default_height: int = 500
        self.default_width: int = 800

    def time_series(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: Union[str, List[str]],
        title: str = "Time Series Analysis",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        show_markers: bool = False,
        fill_area: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a time series line chart.

        Args:
            data: DataFrame containing time series data
            x_col: Column name for x-axis (time)
            y_cols: Column name(s) for y-axis data
            title: Chart title
            x_label: X-axis label (uses x_col if None)
            y_label: Y-axis label
            show_markers: Whether to show data point markers
            fill_area: Whether to fill area under line
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure

        Examples:
            >>> templates = ChartTemplates()
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2024-01-01', periods=100),
            ...     'power': np.random.rand(100) * 1000
            ... })
            >>> fig = templates.time_series(df, 'date', 'power')
        """
        fig = go.Figure()

        # Ensure y_cols is a list
        if isinstance(y_cols, str):
            y_cols = [y_cols]

        # Add traces for each y column
        for y_col in y_cols:
            mode = "lines+markers" if show_markers else "lines"
            fill = "tozeroy" if fill_area and len(y_cols) == 1 else None

            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    name=y_col,
                    mode=mode,
                    fill=fill,
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + f"{x_label or x_col}: %{{x}}<br>"
                    + f"{y_label or 'Value'}: %{{y:.2f}}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label or x_col,
            yaxis_title=y_label or "Value",
            height=height or self.default_height,
            width=width or self.default_width,
            hovermode="x unified",
            showlegend=len(y_cols) > 1,
        )

        return fig

    def bar_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Bar Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        orientation: str = "v",
        color_col: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a bar chart.

        Args:
            data: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            orientation: Bar orientation ('v' for vertical, 'h' for horizontal)
            color_col: Column name for color coding
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure

        Examples:
            >>> templates = ChartTemplates()
            >>> df = pd.DataFrame({
            ...     'module_type': ['Type A', 'Type B', 'Type C'],
            ...     'efficiency': [18.5, 20.2, 19.1]
            ... })
            >>> fig = templates.bar_chart(df, 'module_type', 'efficiency')
        """
        fig = go.Figure()

        if color_col and color_col in data.columns:
            # Grouped or colored bars
            for category in data[color_col].unique():
                subset = data[data[color_col] == category]
                fig.add_trace(
                    go.Bar(
                        x=subset[x_col] if orientation == "v" else subset[y_col],
                        y=subset[y_col] if orientation == "v" else subset[x_col],
                        name=str(category),
                        orientation=orientation,
                    )
                )
        else:
            # Simple bar chart
            fig.add_trace(
                go.Bar(
                    x=data[x_col] if orientation == "v" else data[y_col],
                    y=data[y_col] if orientation == "v" else data[x_col],
                    orientation=orientation,
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label or x_col,
            yaxis_title=y_label or y_col,
            height=height or self.default_height,
            width=width or self.default_width,
            showlegend=color_col is not None,
        )

        return fig

    def heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        title: str = "Heatmap",
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        colorscale: str = "Viridis",
        show_values: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a heatmap visualization.

        Args:
            data: 2D array or DataFrame for heatmap
            title: Chart title
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            colorscale: Plotly colorscale name
            show_values: Whether to show values in cells
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure

        Examples:
            >>> templates = ChartTemplates()
            >>> matrix = np.random.rand(10, 10)
            >>> fig = templates.heatmap(matrix, title='Correlation Matrix')
        """
        if isinstance(data, pd.DataFrame):
            z_data = data.values
            x_labels = x_labels or data.columns.tolist()
            y_labels = y_labels or data.index.tolist()
        else:
            z_data = data

        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                hoverongaps=False,
                text=z_data if show_values else None,
                texttemplate="%{text:.2f}" if show_values else None,
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
        )

        return fig

    def scatter_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Scatter Plot",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size_col: Optional[str] = None,
        color_col: Optional[str] = None,
        trendline: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a scatter plot.

        Args:
            data: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            size_col: Column for marker size
            color_col: Column for marker color
            trendline: Whether to add a trendline
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure

        Examples:
            >>> templates = ChartTemplates()
            >>> df = pd.DataFrame({
            ...     'irradiance': np.random.rand(50) * 1000,
            ...     'power': np.random.rand(50) * 500
            ... })
            >>> fig = templates.scatter_plot(df, 'irradiance', 'power')
        """
        fig = go.Figure()

        # Prepare marker configuration
        marker_config: Dict[str, Any] = {}
        if size_col and size_col in data.columns:
            marker_config["size"] = data[size_col]
            marker_config["sizemode"] = "diameter"
        if color_col and color_col in data.columns:
            marker_config["color"] = data[color_col]
            marker_config["colorscale"] = "Viridis"
            marker_config["showscale"] = True

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode="markers",
                marker=marker_config if marker_config else None,
                name="Data",
            )
        )

        # Add trendline if requested
        if trendline:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            y_trend = p(x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    name="Trend",
                    line=dict(dash="dash", color="red"),
                )
            )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label or x_col,
            yaxis_title=y_label or y_col,
            height=height or self.default_height,
            width=width or self.default_width,
        )

        return fig

    def pie_chart(
        self,
        data: pd.DataFrame,
        values_col: str,
        names_col: str,
        title: str = "Distribution",
        hole: float = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a pie or donut chart.

        Args:
            data: DataFrame containing data
            values_col: Column name for values
            names_col: Column name for labels
            title: Chart title
            hole: Size of center hole (0 for pie, >0 for donut)
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure

        Examples:
            >>> templates = ChartTemplates()
            >>> df = pd.DataFrame({
            ...     'category': ['Recycled', 'Reused', 'Disposed'],
            ...     'percentage': [45, 30, 25]
            ... })
            >>> fig = templates.pie_chart(df, 'percentage', 'category')
        """
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=data[names_col],
                    values=data[values_col],
                    hole=hole,
                    textposition="auto",
                    textinfo="label+percent",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
        )

        return fig

    def multi_axis_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y1_cols: Union[str, List[str]],
        y2_cols: Union[str, List[str]],
        title: str = "Multi-Axis Chart",
        y1_label: str = "Primary Axis",
        y2_label: str = "Secondary Axis",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a chart with dual y-axes.

        Args:
            data: DataFrame containing data
            x_col: Column name for x-axis
            y1_cols: Column(s) for primary y-axis
            y2_cols: Column(s) for secondary y-axis
            title: Chart title
            y1_label: Primary y-axis label
            y2_label: Secondary y-axis label
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Configured Plotly figure with dual axes

        Examples:
            >>> templates = ChartTemplates()
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2024-01-01', periods=30),
            ...     'power': np.random.rand(30) * 1000,
            ...     'temperature': np.random.rand(30) * 40 + 10
            ... })
            >>> fig = templates.multi_axis_chart(df, 'date', 'power', 'temperature')
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Ensure columns are lists
        if isinstance(y1_cols, str):
            y1_cols = [y1_cols]
        if isinstance(y2_cols, str):
            y2_cols = [y2_cols]

        # Add traces for primary y-axis
        for col in y1_cols:
            fig.add_trace(
                go.Scatter(x=data[x_col], y=data[col], name=col, mode="lines"),
                secondary_y=False,
            )

        # Add traces for secondary y-axis
        for col in y2_cols:
            fig.add_trace(
                go.Scatter(
                    x=data[x_col], y=data[col], name=col, mode="lines", line=dict(dash="dash")
                ),
                secondary_y=True,
            )

        # Update axes
        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y1_label, secondary_y=False)
        fig.update_yaxes(title_text=y2_label, secondary_y=True)

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
            hovermode="x unified",
        )

        return fig

    def dashboard_grid(
        self,
        figures: List[go.Figure],
        rows: int,
        cols: int,
        title: str = "Dashboard",
        subplot_titles: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a dashboard with multiple charts in a grid.

        Args:
            figures: List of Plotly figures to combine
            rows: Number of rows in grid
            cols: Number of columns in grid
            title: Dashboard title
            subplot_titles: Titles for each subplot
            height: Total dashboard height
            width: Total dashboard width

        Returns:
            Combined Plotly figure with subplot grid

        Examples:
            >>> templates = ChartTemplates()
            >>> fig1 = templates.time_series(df1, 'date', 'power')
            >>> fig2 = templates.bar_chart(df2, 'type', 'efficiency')
            >>> dashboard = templates.dashboard_grid([fig1, fig2], 1, 2)
        """
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Add each figure to the grid
        for idx, source_fig in enumerate(figures):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            # Add all traces from the source figure
            for trace in source_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
            height=height or (self.default_height * rows),
            width=width or (self.default_width * cols),
            showlegend=True,
        )

        return fig


# Global chart templates instance
_global_chart_templates = ChartTemplates()


def get_chart_templates() -> ChartTemplates:
    """
    Get the global chart templates instance.

    Returns:
        Global ChartTemplates instance

    Examples:
        >>> from pv_circularity.visualization.templates import get_chart_templates
        >>> templates = get_chart_templates()
        >>> fig = templates.time_series(df, 'date', 'power')
    """
    return _global_chart_templates


def chart_templates() -> ChartTemplates:
    """
    Access the chart templates system.

    This is a convenience function that returns the global chart templates,
    providing access to all pre-configured chart types.

    Returns:
        ChartTemplates instance for creating charts

    Examples:
        >>> from pv_circularity.visualization import chart_templates
        >>> templates = chart_templates()
        >>> fig = templates.bar_chart(df, 'category', 'value')
    """
    return get_chart_templates()
