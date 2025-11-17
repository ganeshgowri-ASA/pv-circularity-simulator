"""
Core visualization library for PV Circularity Simulator.

This module provides the main VisualizationLib class that integrates all
visualization capabilities including chart templates, interactive plots,
export functionality, and custom theming.
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import altair as alt

from pv_circularity.visualization.templates import ChartTemplates, chart_templates
from pv_circularity.visualization.interactive import InteractivePlots, interactive_plots
from pv_circularity.visualization.exports import ExportManager, export_functionality
from pv_circularity.visualization.themes import ThemeManager, custom_themes


class VisualizationLib:
    """
    Comprehensive visualization library for PV circularity analysis.

    This is the main interface for all visualization capabilities, providing
    integrated access to chart templates, interactive plots, export functions,
    and custom theming. It's designed specifically for photovoltaic system
    lifecycle data visualization.

    Attributes:
        templates: Chart templates for common visualization patterns
        interactive: Interactive plotting capabilities
        exporter: Export functionality for multiple formats
        themes: Theme management and customization

    Examples:
        >>> from pv_circularity.visualization import VisualizationLib
        >>> viz = VisualizationLib()
        >>>
        >>> # Set a theme
        >>> viz.themes.set_theme('solar')
        >>>
        >>> # Create a time series chart
        >>> fig = viz.templates.time_series(df, 'date', 'power')
        >>>
        >>> # Make it interactive
        >>> interactive_fig = viz.interactive.create_interactive_timeseries(
        ...     df, 'date', 'power'
        ... )
        >>>
        >>> # Export the result
        >>> viz.exporter.export_plotly(fig, 'output.png', width=1600, height=900)
    """

    def __init__(self, default_theme: str = "default") -> None:
        """
        Initialize the visualization library.

        Args:
            default_theme: Name of the default theme to use
                Available themes: 'default', 'solar', 'circularity',
                'performance', 'technical', 'dark'

        Examples:
            >>> viz = VisualizationLib(default_theme='solar')
            >>> viz.themes.get_current_theme()
        """
        self._templates = chart_templates()
        self._interactive = interactive_plots()
        self._exporter = export_functionality()
        self._themes = custom_themes()

        # Set default theme
        if default_theme != "default":
            self._themes.set_theme(default_theme)

    @property
    def templates(self) -> ChartTemplates:
        """
        Access chart templates.

        Returns:
            ChartTemplates instance with pre-configured charts

        Examples:
            >>> viz = VisualizationLib()
            >>> fig = viz.templates.bar_chart(df, 'category', 'value')
        """
        return self._templates

    @property
    def interactive(self) -> InteractivePlots:
        """
        Access interactive plotting capabilities.

        Returns:
            InteractivePlots instance for interactive visualizations

        Examples:
            >>> viz = VisualizationLib()
            >>> fig = viz.interactive.create_animated_scatter(
            ...     df, 'x', 'y', 'year'
            ... )
        """
        return self._interactive

    @property
    def exporter(self) -> ExportManager:
        """
        Access export functionality.

        Returns:
            ExportManager instance for exporting visualizations

        Examples:
            >>> viz = VisualizationLib()
            >>> viz.exporter.export_plotly(fig, 'chart.png')
        """
        return self._exporter

    @property
    def themes(self) -> ThemeManager:
        """
        Access theme management.

        Returns:
            ThemeManager instance for theme customization

        Examples:
            >>> viz = VisualizationLib()
            >>> viz.themes.set_theme('solar')
            >>> print(viz.themes.list_themes())
        """
        return self._themes

    def quick_plot(
        self,
        data: pd.DataFrame,
        chart_type: str = "auto",
        x: Optional[str] = None,
        y: Optional[Union[str, List[str]]] = None,
        title: Optional[str] = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create a quick plot with automatic chart type detection.

        This convenience method automatically selects an appropriate chart
        type based on data characteristics or uses the specified type.

        Args:
            data: DataFrame to visualize
            chart_type: Type of chart ('auto', 'line', 'bar', 'scatter',
                'pie', 'heatmap')
            x: Column name for x-axis (auto-detected if None)
            y: Column name(s) for y-axis (auto-detected if None)
            title: Chart title
            interactive: Whether to create an interactive version
            **kwargs: Additional arguments passed to chart creation

        Returns:
            Plotly figure

        Examples:
            >>> viz = VisualizationLib()
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2024-01-01', periods=30),
            ...     'power': np.random.rand(30) * 1000
            ... })
            >>> fig = viz.quick_plot(df, x='date', y='power')
        """
        # Auto-detect columns if not specified
        if x is None and y is None:
            # Use first column as x, second as y
            cols = data.columns.tolist()
            if len(cols) >= 2:
                x = cols[0]
                y = cols[1]
            elif len(cols) == 1:
                x = data.index.name or "index"
                y = cols[0]
            else:
                raise ValueError("DataFrame must have at least one column")

        # Auto-detect chart type if needed
        if chart_type == "auto":
            chart_type = self._auto_detect_chart_type(data, x, y)

        # Generate default title if not provided
        if title is None:
            y_label = y if isinstance(y, str) else ", ".join(y)
            title = f"{y_label} vs {x}"

        # Create chart based on type
        if chart_type == "line" or chart_type == "timeseries":
            if interactive:
                return self._interactive.create_interactive_timeseries(
                    data, x, y, title, **kwargs
                )
            else:
                return self._templates.time_series(data, x, y, title, **kwargs)

        elif chart_type == "bar":
            if isinstance(y, list):
                y = y[0]  # Bar chart works with single y
            return self._templates.bar_chart(data, x, y, title, **kwargs)

        elif chart_type == "scatter":
            if isinstance(y, list):
                y = y[0]  # Scatter works with single y
            return self._templates.scatter_plot(data, x, y, title, **kwargs)

        elif chart_type == "pie":
            if isinstance(y, list):
                y = y[0]
            return self._templates.pie_chart(data, y, x, title, **kwargs)

        elif chart_type == "heatmap":
            return self._templates.heatmap(data, title, **kwargs)

        else:
            raise ValueError(
                f"Unsupported chart type: {chart_type}. "
                f"Supported types: line, bar, scatter, pie, heatmap"
            )

    def _auto_detect_chart_type(
        self,
        data: pd.DataFrame,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
    ) -> str:
        """
        Automatically detect appropriate chart type based on data.

        Args:
            data: DataFrame to analyze
            x: X-axis column
            y: Y-axis column(s)

        Returns:
            Suggested chart type
        """
        # Check if x is datetime
        if x and pd.api.types.is_datetime64_any_dtype(data[x]):
            return "line"

        # Check if x is categorical with few unique values
        if x and data[x].dtype == "object" and data[x].nunique() < 20:
            return "bar"

        # Check if both x and y are numeric
        y_col = y if isinstance(y, str) else (y[0] if y else None)
        if (
            x
            and y_col
            and pd.api.types.is_numeric_dtype(data[x])
            and pd.api.types.is_numeric_dtype(data[y_col])
        ):
            # Check correlation to decide between line and scatter
            if len(data) > 50:
                return "scatter"
            else:
                return "line"

        # Default to line chart
        return "line"

    def create_pv_performance_dashboard(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        power_col: str = "power",
        irradiance_col: Optional[str] = None,
        temperature_col: Optional[str] = None,
        title: str = "PV System Performance Dashboard",
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create a comprehensive PV performance monitoring dashboard.

        This specialized method creates a multi-panel dashboard specifically
        designed for photovoltaic system performance monitoring.

        Args:
            data: DataFrame with PV system data
            timestamp_col: Column name for timestamps
            power_col: Column name for power output
            irradiance_col: Column name for solar irradiance (optional)
            temperature_col: Column name for temperature (optional)
            title: Dashboard title
            **kwargs: Additional customization parameters

        Returns:
            Multi-panel Plotly figure

        Examples:
            >>> viz = VisualizationLib()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            ...     'power': np.random.rand(100) * 1000,
            ...     'irradiance': np.random.rand(100) * 1000,
            ...     'temperature': np.random.rand(100) * 40 + 10
            ... })
            >>> dashboard = viz.create_pv_performance_dashboard(df)
        """
        # Create individual charts
        charts = []

        # Main power chart
        power_chart = self._templates.time_series(
            data, timestamp_col, power_col, "Power Output", show_markers=False
        )
        charts.append(power_chart)

        # Multi-axis chart if temperature or irradiance available
        if irradiance_col and temperature_col:
            multi_chart = self._templates.multi_axis_chart(
                data,
                timestamp_col,
                irradiance_col,
                temperature_col,
                "Environmental Conditions",
                "Irradiance (W/m²)",
                "Temperature (°C)",
            )
            charts.append(multi_chart)
        elif irradiance_col:
            irr_chart = self._templates.time_series(
                data, timestamp_col, irradiance_col, "Solar Irradiance"
            )
            charts.append(irr_chart)
        elif temperature_col:
            temp_chart = self._templates.time_series(
                data, timestamp_col, temperature_col, "Module Temperature"
            )
            charts.append(temp_chart)

        # Combine into dashboard
        rows = len(charts)
        dashboard = self._templates.dashboard_grid(
            charts,
            rows=rows,
            cols=1,
            title=title,
            subplot_titles=[c.layout.title.text for c in charts],
            **kwargs,
        )

        return dashboard

    def create_circularity_analysis(
        self,
        data: pd.DataFrame,
        categories: str,
        values: str,
        title: str = "Circular Economy Analysis",
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create a circular economy analysis visualization.

        This method creates visualizations specifically for analyzing
        circular economy metrics like recycling rates, material recovery,
        and lifecycle stages.

        Args:
            data: DataFrame with circularity data
            categories: Column name for categories (e.g., lifecycle stages)
            values: Column name for values (e.g., material quantities)
            title: Chart title
            **kwargs: Additional parameters

        Returns:
            Plotly figure optimized for circularity analysis

        Examples:
            >>> viz = VisualizationLib()
            >>> df = pd.DataFrame({
            ...     'stage': ['Manufacturing', 'Use', 'Collection', 'Recycling'],
            ...     'material_kg': [1000, 950, 800, 700]
            ... })
            >>> fig = viz.create_circularity_analysis(df, 'stage', 'material_kg')
        """
        # Create a waterfall-style visualization
        fig = go.Figure()

        # Calculate differences
        cumulative = 0
        x_vals = []
        y_vals = []
        measures = []
        texts = []

        for idx, row in data.iterrows():
            x_vals.append(row[categories])
            current_val = row[values]

            if idx == 0:
                # First value is absolute
                y_vals.append(current_val)
                measures.append("relative")
                cumulative = current_val
            else:
                # Subsequent values show change
                diff = current_val - cumulative
                y_vals.append(diff)
                measures.append("relative")
                cumulative = current_val

            texts.append(f"{current_val:.0f}")

        # Add total
        x_vals.append("Net Flow")
        y_vals.append(cumulative)
        measures.append("total")
        texts.append(f"{cumulative:.0f}")

        fig.add_trace(
            go.Waterfall(
                name=title,
                orientation="v",
                x=x_vals,
                y=y_vals,
                text=texts,
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            showlegend=False,
            **kwargs,
        )

        return fig

    def apply_theme(self, fig: go.Figure, theme_name: Optional[str] = None) -> go.Figure:
        """
        Apply a theme to an existing figure.

        Args:
            fig: Plotly figure to style
            theme_name: Name of theme to apply (uses current if None)

        Returns:
            Styled figure

        Examples:
            >>> viz = VisualizationLib()
            >>> fig = go.Figure(data=[go.Bar(x=[1,2,3], y=[4,5,6])])
            >>> styled_fig = viz.apply_theme(fig, 'solar')
        """
        return self._themes.apply_theme_to_plotly(fig, theme_name)

    def export(
        self,
        fig: go.Figure,
        filepath: str,
        **kwargs: Any,
    ) -> None:
        """
        Quick export of a figure.

        Args:
            fig: Figure to export
            filepath: Output file path
            **kwargs: Additional export parameters

        Examples:
            >>> viz = VisualizationLib()
            >>> fig = viz.templates.bar_chart(df, 'x', 'y')
            >>> viz.export(fig, 'output.png', width=1600, height=900)
        """
        self._exporter.export_plotly(fig, filepath, **kwargs)

    def set_theme(self, theme_name: str) -> None:
        """
        Set the active theme for all visualizations.

        Args:
            theme_name: Name of theme to activate

        Examples:
            >>> viz = VisualizationLib()
            >>> viz.set_theme('solar')
        """
        self._themes.set_theme(theme_name)

    def list_themes(self) -> List[str]:
        """
        List all available themes.

        Returns:
            List of theme names

        Examples:
            >>> viz = VisualizationLib()
            >>> themes = viz.list_themes()
            >>> print(themes)
        """
        return self._themes.list_themes()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the visualization library.

        Returns:
            Dictionary with library information and capabilities

        Examples:
            >>> viz = VisualizationLib()
            >>> info = viz.get_info()
            >>> print(info['version'])
        """
        return {
            "version": "0.1.0",
            "name": "PV Circularity Visualization Library",
            "current_theme": self._themes._current_theme,
            "available_themes": self.list_themes(),
            "capabilities": {
                "chart_templates": [
                    "time_series",
                    "bar_chart",
                    "scatter_plot",
                    "heatmap",
                    "pie_chart",
                    "multi_axis_chart",
                    "dashboard_grid",
                ],
                "interactive_plots": [
                    "interactive_timeseries",
                    "crossfilter_dashboard",
                    "drill_down_chart",
                    "animated_scatter",
                    "linked_brushing",
                    "real_time_plot",
                ],
                "export_formats": ["png", "jpg", "svg", "pdf", "html", "json"],
                "themes": self.list_themes(),
            },
        }


# Convenience functions for backward compatibility and ease of use
def chart_templates() -> ChartTemplates:
    """Get the global chart templates instance."""
    from pv_circularity.visualization.templates import chart_templates as _ct
    return _ct()


def interactive_plots() -> InteractivePlots:
    """Get the global interactive plots instance."""
    from pv_circularity.visualization.interactive import interactive_plots as _ip
    return _ip()


def export_functionality() -> ExportManager:
    """Get the global export manager instance."""
    from pv_circularity.visualization.exports import export_functionality as _ef
    return _ef()


def custom_themes() -> ThemeManager:
    """Get the global theme manager instance."""
    from pv_circularity.visualization.themes import custom_themes as _ct
    return _ct()
