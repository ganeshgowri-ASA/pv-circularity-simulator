"""
Interactive plotting functionality for PV Circularity visualizations.

This module provides interactive plotting capabilities with advanced features
like zooming, panning, hover tooltips, filtering, and real-time updates.
"""

from typing import Optional, List, Dict, Any, Callable, Union
import plotly.graph_objects as go
import altair as alt
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


class InteractivePlots:
    """
    Interactive visualization components with advanced interactivity.

    This class provides methods for creating highly interactive visualizations
    with features like dynamic filtering, cross-filtering, drill-down
    capabilities, and real-time data updates.

    Examples:
        >>> interactive = InteractivePlots()
        >>> fig = interactive.create_interactive_timeseries(
        ...     data=df,
        ...     x='date',
        ...     y='power',
        ...     enable_rangeslider=True
        ... )
    """

    def __init__(self) -> None:
        """Initialize interactive plots with default configurations."""
        self.default_height: int = 600
        self.default_width: int = 1000

    def create_interactive_timeseries(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        title: str = "Interactive Time Series",
        enable_rangeslider: bool = True,
        enable_rangeselector: bool = True,
        show_annotations: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create an interactive time series with range controls.

        Args:
            data: DataFrame with time series data
            x: Column name for x-axis (time)
            y: Column name(s) for y-axis
            title: Chart title
            enable_rangeslider: Show range slider below chart
            enable_rangeselector: Show quick date range buttons
            show_annotations: Show data point annotations
            height: Chart height in pixels
            width: Chart width in pixels

        Returns:
            Interactive Plotly figure with range controls

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2024-01-01', periods=365),
            ...     'power': np.random.rand(365) * 1000
            ... })
            >>> fig = interactive.create_interactive_timeseries(df, 'date', 'power')
        """
        fig = go.Figure()

        # Ensure y is a list
        if isinstance(y, str):
            y = [y]

        # Add traces
        for y_col in y:
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y_col],
                    name=y_col,
                    mode="lines",
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + f"{x}: %{{x}}<br>"
                    + "Value: %{y:.2f}<extra></extra>",
                )
            )

        # Configure range slider
        rangeslider_config = dict(visible=True) if enable_rangeslider else None

        # Configure range selector buttons
        rangeselector_config = None
        if enable_rangeselector:
            rangeselector_config = dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(label="All", step="all"),
                    ]
                )
            )

        # Update layout
        fig.update_xaxes(
            rangeslider=rangeslider_config,
            rangeselector=rangeselector_config,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
            hovermode="x unified",
            xaxis_title=x,
            yaxis_title="Value",
        )

        return fig

    def create_crossfilter_dashboard(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        measure: str,
        title: str = "Cross-Filter Dashboard",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> alt.Chart:
        """
        Create an Altair dashboard with cross-filtering capabilities.

        Args:
            data: DataFrame with data to visualize
            dimensions: List of dimension columns for filtering
            measure: Measure column to aggregate
            title: Dashboard title
            height: Chart height
            width: Chart width

        Returns:
            Altair chart with cross-filtering

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'category': ['A', 'B', 'C'] * 10,
            ...     'region': ['North', 'South'] * 15,
            ...     'value': np.random.rand(30) * 100
            ... })
            >>> chart = interactive.create_crossfilter_dashboard(
            ...     df, ['category', 'region'], 'value'
            ... )
        """
        # Create selection
        selection = alt.selection_multi(fields=dimensions, bind="legend")

        # Create base chart
        base = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{measure}:Q", title=measure),
            y=alt.Y(f"{dimensions[0]}:N", title=dimensions[0]),
            color=alt.condition(
                selection,
                alt.Color(f"{dimensions[0]}:N", legend=None),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
            tooltip=[alt.Tooltip(dim, title=dim) for dim in dimensions]
            + [alt.Tooltip(measure, title=measure, format=".2f")],
        ).add_selection(selection)

        # Configure size
        chart_height = height or 400
        chart_width = width or 600

        return base.properties(
            title=title,
            height=chart_height,
            width=chart_width,
        ).interactive()

    def create_drill_down_chart(
        self,
        data: pd.DataFrame,
        hierarchy: List[str],
        measure: str,
        title: str = "Drill-Down Chart",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a chart with drill-down capability through data hierarchy.

        Args:
            data: DataFrame with hierarchical data
            hierarchy: List of columns representing hierarchy levels
            measure: Column to measure/aggregate
            title: Chart title
            height: Chart height
            width: Chart width

        Returns:
            Plotly figure with drill-down capability

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'country': ['USA'] * 4,
            ...     'state': ['CA', 'CA', 'TX', 'TX'],
            ...     'city': ['LA', 'SF', 'Houston', 'Dallas'],
            ...     'sales': [100, 150, 120, 90]
            ... })
            >>> fig = interactive.create_drill_down_chart(
            ...     df, ['country', 'state', 'city'], 'sales'
            ... )
        """
        # Aggregate data at each level
        aggregated_data = []
        for level_idx, level_col in enumerate(hierarchy):
            group_cols = hierarchy[: level_idx + 1]
            level_data = (
                data.groupby(group_cols)[measure]
                .sum()
                .reset_index()
                .assign(level=level_idx)
            )
            aggregated_data.append(level_data)

        # Create sunburst chart (natural drill-down visualization)
        all_data = pd.concat(aggregated_data, ignore_index=True)

        # Prepare labels and parents for sunburst
        labels = []
        parents = []
        values = []

        for _, row in all_data.iterrows():
            level = row["level"]
            if level == 0:
                label = row[hierarchy[0]]
                parent = ""
            else:
                label = row[hierarchy[level]]
                parent = row[hierarchy[level - 1]]

            labels.append(label)
            parents.append(parent)
            values.append(row[measure])

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>"
                + f"{measure}: %{{value:.2f}}<br>"
                + "Percentage: %{percentParent:.1%}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
        )

        return fig

    def create_animated_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        animation_frame: str,
        size: Optional[str] = None,
        color: Optional[str] = None,
        title: str = "Animated Scatter Plot",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create an animated scatter plot.

        Args:
            data: DataFrame with data
            x: Column for x-axis
            y: Column for y-axis
            animation_frame: Column for animation frames (e.g., time)
            size: Column for marker size
            color: Column for marker color
            title: Chart title
            height: Chart height
            width: Chart width

        Returns:
            Animated Plotly scatter plot

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'year': [2020] * 10 + [2021] * 10,
            ...     'efficiency': np.random.rand(20) * 25,
            ...     'cost': np.random.rand(20) * 1000,
            ...     'capacity': np.random.rand(20) * 100
            ... })
            >>> fig = interactive.create_animated_scatter(
            ...     df, 'efficiency', 'cost', 'year', size='capacity'
            ... )
        """
        fig = go.Figure()

        # Get unique animation frames
        frames_data = []
        for frame_value in sorted(data[animation_frame].unique()):
            frame_data = data[data[animation_frame] == frame_value]

            # Prepare marker config
            marker_config: Dict[str, Any] = {}
            if size:
                marker_config["size"] = frame_data[size]
                marker_config["sizemode"] = "diameter"
                marker_config["sizeref"] = 2.0 * frame_data[size].max() / (40.0**2)
            if color:
                marker_config["color"] = frame_data[color]
                marker_config["colorscale"] = "Viridis"
                marker_config["showscale"] = True

            trace = go.Scatter(
                x=frame_data[x],
                y=frame_data[y],
                mode="markers",
                marker=marker_config if marker_config else None,
                text=frame_data[color] if color else None,
                hovertemplate=f"<b>{x}: %{{x:.2f}}</b><br>"
                + f"{y}: %{{y:.2f}}<br>"
                + ("<b>%{text}</b>" if color else "")
                + "<extra></extra>",
            )

            frames_data.append(
                go.Frame(data=[trace], name=str(frame_value))
            )

        # Add initial frame
        fig.add_trace(frames_data[0].data[0])

        # Add frames
        fig.frames = frames_data

        # Add play/pause buttons and slider
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f.name],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=500, redraw=True),
                                ),
                            ],
                            label=f.name,
                        )
                        for f in frames_data
                    ],
                    x=0.1,
                    y=0,
                    len=0.9,
                    xanchor="left",
                    yanchor="top",
                )
            ],
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x,
            yaxis_title=y,
            height=height or self.default_height,
            width=width or self.default_width,
        )

        return fig

    def create_linked_brushing(
        self,
        data: pd.DataFrame,
        chart_configs: List[Dict[str, Any]],
        title: str = "Linked Brushing Dashboard",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> alt.Chart:
        """
        Create multiple linked charts with brushing selection.

        Args:
            data: DataFrame with data
            chart_configs: List of chart configurations
                Each config should have: {'type', 'x', 'y', 'title'}
            title: Overall dashboard title
            height: Chart height
            width: Chart width

        Returns:
            Altair chart with linked brushing

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'x1': np.random.rand(100),
            ...     'y1': np.random.rand(100),
            ...     'x2': np.random.rand(100),
            ...     'y2': np.random.rand(100)
            ... })
            >>> configs = [
            ...     {'type': 'scatter', 'x': 'x1', 'y': 'y1', 'title': 'Chart 1'},
            ...     {'type': 'scatter', 'x': 'x2', 'y': 'y2', 'title': 'Chart 2'}
            ... ]
            >>> chart = interactive.create_linked_brushing(df, configs)
        """
        # Create brush selection
        brush = alt.selection_interval()

        charts = []
        for config in chart_configs:
            chart_type = config.get("type", "scatter")
            x_col = config["x"]
            y_col = config["y"]
            chart_title = config.get("title", f"{x_col} vs {y_col}")

            if chart_type == "scatter":
                chart = (
                    alt.Chart(data)
                    .mark_point()
                    .encode(
                        x=alt.X(f"{x_col}:Q", title=x_col),
                        y=alt.Y(f"{y_col}:Q", title=y_col),
                        color=alt.condition(
                            brush, alt.value("steelblue"), alt.value("lightgray")
                        ),
                        opacity=alt.condition(brush, alt.value(0.8), alt.value(0.3)),
                    )
                    .properties(title=chart_title, width=300, height=250)
                    .add_selection(brush)
                )
            elif chart_type == "bar":
                chart = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{x_col}:N", title=x_col),
                        y=alt.Y(f"{y_col}:Q", title=y_col),
                        color=alt.condition(
                            brush, alt.value("steelblue"), alt.value("lightgray")
                        ),
                    )
                    .properties(title=chart_title, width=300, height=250)
                    .add_selection(brush)
                )
            else:
                # Default to scatter
                chart = (
                    alt.Chart(data)
                    .mark_point()
                    .encode(x=f"{x_col}:Q", y=f"{y_col}:Q")
                    .properties(title=chart_title, width=300, height=250)
                    .add_selection(brush)
                )

            charts.append(chart)

        # Combine charts horizontally or vertically
        if len(charts) == 1:
            combined = charts[0]
        elif len(charts) == 2:
            combined = charts[0] | charts[1]
        else:
            # Create grid layout
            rows = []
            for i in range(0, len(charts), 2):
                if i + 1 < len(charts):
                    rows.append(charts[i] | charts[i + 1])
                else:
                    rows.append(charts[i])

            combined = rows[0]
            for row in rows[1:]:
                combined = combined & row

        return combined.properties(title=title).configure_title(fontSize=16, anchor="middle")

    def create_real_time_plot(
        self,
        initial_data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Real-Time Plot",
        max_points: int = 100,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Create a plot configured for real-time updates.

        Args:
            initial_data: Initial DataFrame
            x: Column for x-axis
            y: Column for y-axis
            title: Chart title
            max_points: Maximum number of points to display
            height: Chart height
            width: Chart width

        Returns:
            Plotly figure configured for real-time updates

        Note:
            This returns a static figure. For true real-time updates,
            use with Plotly Dash or similar framework.

        Examples:
            >>> interactive = InteractivePlots()
            >>> df = pd.DataFrame({
            ...     'time': pd.date_range('2024-01-01', periods=50, freq='1min'),
            ...     'value': np.random.rand(50)
            ... })
            >>> fig = interactive.create_real_time_plot(df, 'time', 'value')
        """
        # Use only the last max_points
        if len(initial_data) > max_points:
            plot_data = initial_data.tail(max_points)
        else:
            plot_data = initial_data

        fig = go.Figure(
            data=go.Scatter(
                x=plot_data[x],
                y=plot_data[y],
                mode="lines+markers",
                name=y,
                line=dict(color="steelblue", width=2),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x,
            yaxis_title=y,
            height=height or self.default_height,
            width=width or self.default_width,
            xaxis=dict(range=[plot_data[x].min(), plot_data[x].max()]),
            yaxis=dict(
                range=[
                    plot_data[y].min() * 0.9,
                    plot_data[y].max() * 1.1,
                ]
            ),
            hovermode="x unified",
        )

        return fig


# Global interactive plots instance
_global_interactive_plots = InteractivePlots()


def get_interactive_plots() -> InteractivePlots:
    """
    Get the global interactive plots instance.

    Returns:
        Global InteractivePlots instance

    Examples:
        >>> from pv_circularity.visualization.interactive import get_interactive_plots
        >>> interactive = get_interactive_plots()
        >>> fig = interactive.create_interactive_timeseries(df, 'date', 'power')
    """
    return _global_interactive_plots


def interactive_plots() -> InteractivePlots:
    """
    Access the interactive plotting system.

    This is a convenience function that returns the global interactive plots,
    providing access to all interactive visualization capabilities.

    Returns:
        InteractivePlots instance for creating interactive visualizations

    Examples:
        >>> from pv_circularity.visualization import interactive_plots
        >>> interactive = interactive_plots()
        >>> fig = interactive.create_animated_scatter(df, 'x', 'y', 'year')
    """
    return get_interactive_plots()
