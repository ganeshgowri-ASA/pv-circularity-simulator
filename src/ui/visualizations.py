"""Plotly-based visualizations for solar resource analysis."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..irradiance.models import POAComponents, ResourceStatistics


class SolarResourceVisualizer:
    """Production-ready visualizations for solar resource assessment.

    Provides interactive Plotly charts for:
    - Time series irradiance data
    - Resource heat maps (hour-by-month, day-of-year)
    - Annual profiles and patterns
    - Statistical distributions
    - Component breakdowns

    All charts are designed for professional reporting and analysis.
    """

    def __init__(self, theme: str = "plotly_white"):
        """Initialize the visualizer.

        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn')
        """
        self.theme = theme

    def plot_irradiance_timeseries(
        self,
        data: Union[pd.Series, pd.DataFrame],
        title: str = "Solar Irradiance Time Series",
        ylabel: str = "Irradiance (W/m²)",
        show_average: bool = True,
        height: int = 500,
    ) -> go.Figure:
        """Create interactive time series plot of irradiance data.

        Args:
            data: Time series data (Series or DataFrame with multiple columns)
            title: Chart title
            ylabel: Y-axis label
            show_average: Add horizontal line showing average
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> viz = SolarResourceVisualizer()
            >>> fig = viz.plot_irradiance_timeseries(ghi_data)
            >>> fig.show()
        """
        fig = go.Figure()

        # Handle Series or DataFrame
        if isinstance(data, pd.Series):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode="lines",
                    name=data.name or "Irradiance",
                    line=dict(color="#2E86DE", width=1),
                )
            )

            if show_average:
                avg = data.mean()
                fig.add_hline(
                    y=avg,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {avg:.1f} W/m²",
                    annotation_position="right",
                )

        else:  # DataFrame
            colors = ["#2E86DE", "#00B894", "#FDCB6E", "#E17055", "#6C5CE7"]
            for i, col in enumerate(data.columns):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode="lines",
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=ylabel,
            template=self.theme,
            height=height,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def plot_poa_components(
        self,
        poa_components: POAComponents,
        title: str = "POA Irradiance Components",
        height: int = 600,
    ) -> go.Figure:
        """Create stacked area chart showing POA irradiance components.

        Args:
            poa_components: POAComponents object with component breakdown
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> fig = viz.plot_poa_components(poa_components)
            >>> fig.write_html("poa_components.html")
        """
        fig = go.Figure()

        # Add components as stacked areas
        fig.add_trace(
            go.Scatter(
                x=poa_components.poa_direct.index,
                y=poa_components.poa_direct,
                name="Direct Beam",
                mode="lines",
                stackgroup="one",
                fillcolor="rgba(255, 193, 7, 0.6)",
                line=dict(width=0.5, color="rgba(255, 193, 7, 1)"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=poa_components.poa_diffuse.index,
                y=poa_components.poa_diffuse,
                name="Sky Diffuse",
                mode="lines",
                stackgroup="one",
                fillcolor="rgba(33, 150, 243, 0.6)",
                line=dict(width=0.5, color="rgba(33, 150, 243, 1)"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=poa_components.poa_ground.index,
                y=poa_components.poa_ground,
                name="Ground Reflected",
                mode="lines",
                stackgroup="one",
                fillcolor="rgba(76, 175, 80, 0.6)",
                line=dict(width=0.5, color="rgba(76, 175, 80, 1)"),
            )
        )

        # Add total line
        fig.add_trace(
            go.Scatter(
                x=poa_components.poa_global.index,
                y=poa_components.poa_global,
                name="Total POA",
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Irradiance (W/m²)",
            template=self.theme,
            height=height,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def plot_resource_heatmap(
        self,
        matrix_data: pd.DataFrame,
        title: str = "Solar Resource Heat Map",
        xlabel: str = "Month",
        ylabel: str = "Hour of Day",
        colorscale: str = "YlOrRd",
        height: int = 600,
    ) -> go.Figure:
        """Create heat map visualization of solar resource patterns.

        Args:
            matrix_data: DataFrame with time dimensions (e.g., hour x month)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorscale: Plotly colorscale name
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> maps = analyzer.solar_resource_maps()
            >>> fig = viz.plot_resource_heatmap(maps['hourly_by_month'])
            >>> fig.show()
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix_data.values,
                x=matrix_data.columns,
                y=matrix_data.index,
                colorscale=colorscale,
                colorbar=dict(title="W/m²"),
                hoverongaps=False,
                hovertemplate=ylabel
                + ": %{y}<br>"
                + xlabel
                + ": %{x}<br>"
                + "Irradiance: %{z:.1f} W/m²<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template=self.theme,
            height=height,
            xaxis=dict(side="bottom"),
        )

        return fig

    def plot_annual_profile(
        self,
        daily_data: pd.Series,
        title: str = "Annual Irradiance Profile",
        show_smoothed: bool = True,
        height: int = 500,
    ) -> go.Figure:
        """Create annual profile chart showing daily totals throughout the year.

        Args:
            daily_data: Daily aggregated irradiance data
            title: Chart title
            show_smoothed: Add smoothed trend line
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> daily_ghi = ghi.resample('D').sum()
            >>> fig = viz.plot_annual_profile(daily_ghi)
            >>> fig.show()
        """
        fig = go.Figure()

        # Plot daily values
        fig.add_trace(
            go.Scatter(
                x=daily_data.index.dayofyear,
                y=daily_data.values,
                mode="markers",
                name="Daily Total",
                marker=dict(size=4, color="#2E86DE", opacity=0.6),
            )
        )

        if show_smoothed:
            # Add smoothed trend using rolling average
            smoothed = daily_data.rolling(window=30, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=smoothed.index.dayofyear,
                    y=smoothed.values,
                    mode="lines",
                    name="30-Day Average",
                    line=dict(color="red", width=2),
                )
            )

        # Add month markers
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig.update_layout(
            title=title,
            xaxis_title="Day of Year",
            yaxis_title="Daily Irradiance (Wh/m²)",
            template=self.theme,
            height=height,
            xaxis=dict(tickmode="array", tickvals=month_starts, ticktext=month_names),
            hovermode="x",
        )

        return fig

    def plot_monthly_boxplot(
        self,
        data: pd.Series,
        title: str = "Monthly Irradiance Distribution",
        ylabel: str = "Irradiance (W/m²)",
        height: int = 500,
    ) -> go.Figure:
        """Create box plot showing monthly distributions.

        Args:
            data: Time series data
            title: Chart title
            ylabel: Y-axis label
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> fig = viz.plot_monthly_boxplot(ghi_data)
            >>> fig.show()
        """
        fig = go.Figure()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for month in range(1, 13):
            month_data = data[data.index.month == month]
            fig.add_trace(
                go.Box(
                    y=month_data,
                    name=month_names[month - 1],
                    boxmean="sd",
                    marker_color="#2E86DE",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title=ylabel,
            template=self.theme,
            height=height,
            showlegend=False,
        )

        return fig

    def plot_p50_p90_analysis(
        self,
        p_analysis: Dict[str, pd.DataFrame],
        title: str = "P50/P90 Exceedance Analysis",
        height: int = 600,
    ) -> go.Figure:
        """Create visualization of P50/P90 exceedance probability analysis.

        Args:
            p_analysis: Output from SolarResourceAnalyzer.p50_p90_analysis()
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly Figure object with histogram and percentile markers

        Example:
            >>> p_analysis = analyzer.p50_p90_analysis()
            >>> fig = viz.plot_p50_p90_analysis(p_analysis)
            >>> fig.show()
        """
        time_series = p_analysis["time_series"]
        summary = p_analysis["summary"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Probability Distribution", "Exceedance Curve"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=time_series,
                nbinsx=50,
                name="Distribution",
                marker_color="#2E86DE",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Add P-value markers
        colors = {"P10": "#00B894", "P50": "#FDCB6E", "P90": "#E17055"}

        for _, row in summary.iterrows():
            if row["Percentile"] in ["P10", "P50", "P90"]:
                fig.add_vline(
                    x=row["Value"],
                    line_dash="dash",
                    line_color=colors.get(row["Percentile"], "gray"),
                    annotation_text=f"{row['Percentile']}: {row['Value']:.1f}",
                    annotation_position="top",
                    row=1,
                    col=1,
                )

        # Exceedance curve
        sorted_data = np.sort(time_series)
        exceedance_prob = np.arange(len(sorted_data), 0, -1) / len(sorted_data) * 100

        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=exceedance_prob,
                mode="lines",
                name="Exceedance",
                line=dict(color="#6C5CE7", width=2),
            ),
            row=2,
            col=1,
        )

        # Add P50 and P90 points
        for percentile in ["P50", "P90"]:
            p_value = summary[summary["Percentile"] == percentile]["Value"].values[0]
            exceedance = summary[summary["Percentile"] == percentile][
                "Exceedance Probability"
            ].values[0] * 100

            fig.add_trace(
                go.Scatter(
                    x=[p_value],
                    y=[exceedance],
                    mode="markers",
                    name=percentile,
                    marker=dict(size=12, color=colors[percentile], symbol="diamond"),
                ),
                row=2,
                col=1,
            )

        fig.update_xaxes(title_text="Irradiance (Wh/m²)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Exceedance Probability (%)", row=2, col=1)

        fig.update_layout(
            title_text=title, template=self.theme, height=height, showlegend=True
        )

        return fig

    def plot_comparison_chart(
        self,
        data_dict: Dict[str, pd.Series],
        title: str = "Irradiance Comparison",
        ylabel: str = "Irradiance (W/m²)",
        chart_type: str = "line",
        height: int = 500,
    ) -> go.Figure:
        """Create comparison chart for multiple data series.

        Args:
            data_dict: Dictionary of {label: data_series}
            title: Chart title
            ylabel: Y-axis label
            chart_type: Chart type ('line', 'bar', 'area')
            height: Chart height in pixels

        Returns:
            Plotly Figure object

        Example:
            >>> comparison = {
            ...     'GHI': ghi_data,
            ...     'POA': poa_data,
            ...     'POA with losses': poa_effective
            ... }
            >>> fig = viz.plot_comparison_chart(comparison)
            >>> fig.show()
        """
        fig = go.Figure()

        colors = ["#2E86DE", "#00B894", "#FDCB6E", "#E17055", "#6C5CE7"]

        for i, (label, data) in enumerate(data_dict.items()):
            if chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data,
                        mode="lines",
                        name=label,
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )
            elif chart_type == "bar":
                fig.add_trace(
                    go.Bar(x=data.index, y=data, name=label, marker_color=colors[i % len(colors)])
                )
            elif chart_type == "area":
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data,
                        mode="lines",
                        name=label,
                        fill="tozeroy",
                        fillcolor=f"rgba{tuple(list(int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)) + [0.3])}",
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=ylabel,
            template=self.theme,
            height=height,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def create_dashboard(
        self,
        data: pd.Series,
        poa_components: Optional[POAComponents] = None,
        resource_stats: Optional[ResourceStatistics] = None,
    ) -> go.Figure:
        """Create comprehensive dashboard with multiple charts.

        Args:
            data: Primary time series data
            poa_components: Optional POA component breakdown
            resource_stats: Optional resource statistics

        Returns:
            Plotly Figure with multiple subplots

        Example:
            >>> fig = viz.create_dashboard(ghi_data, poa_components, stats)
            >>> fig.write_html("solar_resource_dashboard.html")
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Time Series",
                "Monthly Distribution",
                "Daily Profile",
                "Statistics Summary",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # 1. Time series (recent data)
        recent = data.tail(24 * 30)  # Last 30 days
        fig.add_trace(
            go.Scatter(
                x=recent.index, y=recent, mode="lines", name="Irradiance", line=dict(color="#2E86DE")
            ),
            row=1,
            col=1,
        )

        # 2. Monthly box plots
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for month in range(1, 13):
            month_data = data[data.index.month == month]
            if len(month_data) > 0:
                fig.add_trace(
                    go.Box(y=month_data, name=month_names[month - 1], marker_color="#2E86DE"),
                    row=1,
                    col=2,
                )

        # 3. Average daily profile
        hourly_avg = data.groupby(data.index.hour).mean()
        fig.add_trace(
            go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg,
                mode="lines+markers",
                name="Hourly Avg",
                line=dict(color="#00B894", width=2),
            ),
            row=2,
            col=1,
        )

        # 4. Statistics table
        if resource_stats:
            stats_table = [
                ["Metric", "Value"],
                ["Mean", f"{resource_stats.mean:.1f} W/m²"],
                ["Median", f"{resource_stats.median:.1f} W/m²"],
                ["Std Dev", f"{resource_stats.std:.1f} W/m²"],
                ["P90", f"{resource_stats.p90:.1f} W/m²"],
                ["P50", f"{resource_stats.p50:.1f} W/m²"],
                ["CV", f"{resource_stats.coefficient_of_variation:.3f}"],
            ]
        else:
            stats_table = [
                ["Metric", "Value"],
                ["Mean", f"{data.mean():.1f} W/m²"],
                ["Median", f"{data.median():.1f} W/m²"],
                ["Std Dev", f"{data.std():.1f} W/m²"],
            ]

        fig.add_trace(
            go.Table(
                header=dict(values=stats_table[0], fill_color="#2E86DE", font=dict(color="white")),
                cells=dict(values=list(zip(*stats_table[1:])), fill_color="lavender"),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Solar Resource Dashboard",
            template=self.theme,
            height=800,
            showlegend=False,
        )

        return fig
