"""
B15-S04: Data Visualization Library
Production-ready visualization library with chart templates and interactive plots.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import altair as alt

from ..core.data_models import ChartConfiguration


class VisualizationLib:
    """
    Comprehensive visualization library for PV circularity simulator.
    """

    def __init__(self):
        """Initialize visualization library."""
        self.default_colors = px.colors.qualitative.Plotly
        self.theme = "plotly"

    def chart_templates(self, chart_config: ChartConfiguration) -> go.Figure:
        """
        Create chart from configuration.

        Args:
            chart_config: Chart configuration

        Returns:
            Plotly figure
        """
        if chart_config.chart_type == "line":
            return self._create_line_chart(chart_config)
        elif chart_config.chart_type == "bar":
            return self._create_bar_chart(chart_config)
        elif chart_config.chart_type == "scatter":
            return self._create_scatter_chart(chart_config)
        elif chart_config.chart_type == "area":
            return self._create_area_chart(chart_config)
        elif chart_config.chart_type == "pie":
            return self._create_pie_chart(chart_config)
        elif chart_config.chart_type == "heatmap":
            return self._create_heatmap(chart_config)
        elif chart_config.chart_type == "sankey":
            return self._create_sankey(chart_config)
        else:
            raise ValueError(f"Unknown chart type: {chart_config.chart_type}")

    def _create_line_chart(self, config: ChartConfiguration) -> go.Figure:
        """Create line chart."""
        fig = go.Figure()

        for series in config.data_series:
            fig.add_trace(go.Scatter(
                x=series.get('x', []),
                y=series.get('y', []),
                mode='lines',
                name=series.get('name', 'Series'),
                line=dict(color=series.get('color'))
            ))

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            template=config.color_scheme or self.theme
        )

        return fig

    def _create_bar_chart(self, config: ChartConfiguration) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure()

        for series in config.data_series:
            fig.add_trace(go.Bar(
                x=series.get('x', []),
                y=series.get('y', []),
                name=series.get('name', 'Series'),
                marker_color=series.get('color')
            ))

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            template=config.color_scheme or self.theme
        )

        return fig

    def _create_scatter_chart(self, config: ChartConfiguration) -> go.Figure:
        """Create scatter chart."""
        fig = go.Figure()

        for series in config.data_series:
            fig.add_trace(go.Scatter(
                x=series.get('x', []),
                y=series.get('y', []),
                mode='markers',
                name=series.get('name', 'Series'),
                marker=dict(
                    size=series.get('size', 8),
                    color=series.get('color')
                )
            ))

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            template=config.color_scheme or self.theme
        )

        return fig

    def _create_area_chart(self, config: ChartConfiguration) -> go.Figure:
        """Create area chart."""
        fig = go.Figure()

        for series in config.data_series:
            fig.add_trace(go.Scatter(
                x=series.get('x', []),
                y=series.get('y', []),
                mode='lines',
                name=series.get('name', 'Series'),
                fill='tonexty' if series.get('fill') else None,
                line=dict(color=series.get('color'))
            ))

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            template=config.color_scheme or self.theme
        )

        return fig

    def _create_pie_chart(self, config: ChartConfiguration) -> go.Figure:
        """Create pie chart."""
        if config.data_series:
            series = config.data_series[0]
            fig = go.Figure(data=[go.Pie(
                labels=series.get('labels', []),
                values=series.get('values', []),
                hole=series.get('hole', 0)
            )])

            fig.update_layout(
                title=config.title,
                template=config.color_scheme or self.theme
            )

            return fig
        return go.Figure()

    def _create_heatmap(self, config: ChartConfiguration) -> go.Figure:
        """Create heatmap."""
        if config.data_series:
            series = config.data_series[0]
            fig = go.Figure(data=go.Heatmap(
                z=series.get('z', [[]]),
                x=series.get('x', []),
                y=series.get('y', []),
                colorscale=series.get('colorscale', 'Viridis')
            ))

            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_axis_label,
                yaxis_title=config.y_axis_label,
                template=config.color_scheme or self.theme
            )

            return fig
        return go.Figure()

    def _create_sankey(self, config: ChartConfiguration) -> go.Figure:
        """Create Sankey diagram."""
        if config.data_series:
            series = config.data_series[0]

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    label=series.get('node_labels', []),
                    color=series.get('node_colors', [])
                ),
                link=dict(
                    source=series.get('source', []),
                    target=series.get('target', []),
                    value=series.get('value', [])
                )
            )])

            fig.update_layout(
                title=config.title,
                font_size=12
            )

            return fig
        return go.Figure()

    def interactive_plots(self,
                         data: pd.DataFrame,
                         plot_type: str = "scatter_matrix") -> go.Figure:
        """
        Create interactive exploratory plots.

        Args:
            data: Input DataFrame
            plot_type: Type of plot

        Returns:
            Interactive Plotly figure
        """
        if plot_type == "scatter_matrix":
            fig = px.scatter_matrix(data)
        elif plot_type == "parallel_coordinates":
            fig = px.parallel_coordinates(data)
        elif plot_type == "box":
            fig = px.box(data)
        elif plot_type == "violin":
            fig = px.violin(data)
        else:
            fig = go.Figure()

        return fig

    def export_graphics(self,
                       fig: go.Figure,
                       filename: str,
                       formats: Optional[List[str]] = None) -> List[str]:
        """
        Export graphics to multiple formats.

        Args:
            fig: Plotly figure
            filename: Base filename (without extension)
            formats: List of formats to export

        Returns:
            List of created file paths
        """
        if formats is None:
            formats = ["png", "svg", "pdf"]

        created_files = []

        for fmt in formats:
            if fmt in ["png", "svg", "pdf", "jpg", "webp"]:
                filepath = f"{filename}.{fmt}"
                try:
                    fig.write_image(filepath)
                    created_files.append(filepath)
                except Exception as e:
                    print(f"Failed to export {fmt}: {e}")
            elif fmt == "html":
                filepath = f"{filename}.html"
                fig.write_html(filepath)
                created_files.append(filepath)
            elif fmt == "json":
                filepath = f"{filename}.json"
                fig.write_json(filepath)
                created_files.append(filepath)

        return created_files

    def create_dashboard_grid(self,
                             figures: List[go.Figure],
                             rows: int,
                             cols: int,
                             titles: Optional[List[str]] = None) -> go.Figure:
        """
        Create dashboard with multiple plots in grid layout.

        Args:
            figures: List of figures to display
            rows: Number of rows
            cols: Number of columns
            titles: Subplot titles

        Returns:
            Combined figure
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles
        )

        for idx, plot_fig in enumerate(figures):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            for trace in plot_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(height=300 * rows, showlegend=True)

        return fig

    def create_time_series_plot(self,
                               timestamps: List,
                               data: Dict[str, np.ndarray],
                               title: str = "Time Series") -> go.Figure:
        """
        Create time series plot with multiple series.

        Args:
            timestamps: Time points
            data: Dictionary of series name -> data array
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, values in data.items():
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=name
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )

        return fig

    def create_comparison_chart(self,
                               categories: List[str],
                               scenario_data: Dict[str, List[float]],
                               title: str = "Scenario Comparison") -> go.Figure:
        """
        Create comparison chart for multiple scenarios.

        Args:
            categories: Category labels
            scenario_data: Dict of scenario name -> values
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for scenario_name, values in scenario_data.items():
            fig.add_trace(go.Bar(
                name=scenario_name,
                x=categories,
                y=values
            ))

        fig.update_layout(
            title=title,
            barmode='group',
            xaxis_title="Category",
            yaxis_title="Value"
        )

        return fig


__all__ = ["VisualizationLib"]
