"""Interactive Visualizations Module.

This module provides interactive visualization capabilities using Plotly and Altair:
- Monthly production charts
- Loss breakdown Sankey diagrams
- Weather correlation plots
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt

from ..models.eya_models import WeatherData, EnergyOutput


class InteractiveVisualizations:
    """Interactive visualization engine for Energy Yield Analysis.

    This class provides comprehensive visualization capabilities including:
    - Monthly and annual production charts
    - Loss breakdown Sankey diagrams
    - Weather correlation plots
    - Performance heatmaps
    - Time series analysis

    Uses Plotly for interactive charts and Altair for declarative visualizations.
    """

    def __init__(self):
        """Initialize the interactive visualizations engine."""
        self.color_scheme = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9800",
            "info": "#17a2b8",
        }

    def monthly_production_charts(
        self, monthly_data: pd.DataFrame, title: str = "Monthly Energy Production"
    ) -> go.Figure:
        """Generate interactive monthly production charts.

        Args:
            monthly_data: DataFrame with monthly production data
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Monthly AC Energy Production",
                "Specific Yield",
                "Capacity Factor",
                "DC vs AC Energy",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Monthly AC Energy (Bar Chart)
        fig.add_trace(
            go.Bar(
                x=monthly_data["month"],
                y=monthly_data["ac_energy"],
                name="AC Energy",
                marker_color=self.color_scheme["primary"],
                text=[f"{val:,.0f}" for val in monthly_data["ac_energy"]],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # Specific Yield (Line Chart)
        fig.add_trace(
            go.Scatter(
                x=monthly_data["month"],
                y=monthly_data["specific_yield"],
                name="Specific Yield",
                mode="lines+markers",
                line=dict(color=self.color_scheme["success"], width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=2,
        )

        # Capacity Factor (Bar Chart)
        fig.add_trace(
            go.Bar(
                x=monthly_data["month"],
                y=monthly_data["capacity_factor"] * 100,
                name="Capacity Factor",
                marker_color=self.color_scheme["warning"],
                text=[f"{val:.1f}%" for val in monthly_data["capacity_factor"] * 100],
                textposition="outside",
            ),
            row=2,
            col=1,
        )

        # DC vs AC Energy (Grouped Bar)
        fig.add_trace(
            go.Bar(
                x=monthly_data["month"],
                y=monthly_data["dc_energy"],
                name="DC Energy",
                marker_color=self.color_scheme["info"],
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=monthly_data["month"],
                y=monthly_data["ac_energy"],
                name="AC Energy",
                marker_color=self.color_scheme["primary"],
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)

        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="kWh/kWp", row=1, col=2)

        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Capacity Factor (%)", row=2, col=1)

        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Energy (kWh)", row=2, col=2)

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=title,
            title_font_size=20,
            hovermode="x unified",
        )

        return fig

    def loss_breakdown_sankey(
        self, waterfall_data: pd.DataFrame, title: str = "Energy Loss Waterfall"
    ) -> go.Figure:
        """Generate Sankey diagram for loss breakdown.

        Args:
            waterfall_data: DataFrame with waterfall loss data
            title: Chart title

        Returns:
            Plotly Figure with Sankey diagram
        """
        # Prepare Sankey data
        labels = waterfall_data["stage"].tolist()

        # Create source, target, and value lists for Sankey
        sources = []
        targets = []
        values = []
        colors = []

        # Color palette for losses
        loss_colors = [
            "rgba(255, 127, 14, 0.6)",  # Orange
            "rgba(214, 39, 40, 0.6)",  # Red
            "rgba(148, 103, 189, 0.6)",  # Purple
            "rgba(140, 86, 75, 0.6)",  # Brown
        ]

        for i in range(len(waterfall_data) - 1):
            sources.append(i)
            targets.append(i + 1)

            if "loss_value" in waterfall_data.columns and i > 0:
                values.append(waterfall_data.iloc[i + 1]["value"])
                colors.append(loss_colors[i % len(loss_colors)])
            else:
                values.append(waterfall_data.iloc[i]["value"] - waterfall_data.iloc[i + 1]["value"])
                colors.append("rgba(31, 119, 180, 0.4)")

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color="rgba(31, 119, 180, 0.8)",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=colors,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=title,
            title_font_size=20,
            font=dict(size=12),
            height=600,
        )

        return fig

    def weather_correlation_plots(
        self, weather_data: List[WeatherData], energy_outputs: List[EnergyOutput]
    ) -> go.Figure:
        """Generate weather correlation plots.

        Args:
            weather_data: List of weather data points
            energy_outputs: List of energy outputs

        Returns:
            Plotly Figure with correlation plots
        """
        # Convert to DataFrames
        weather_df = pd.DataFrame([w.model_dump() for w in weather_data])
        energy_df = pd.DataFrame([e.model_dump() for e in energy_outputs])

        # Merge datasets
        merged_df = pd.merge(
            weather_df[["timestamp", "ghi", "temperature", "wind_speed"]],
            energy_df[["timestamp", "ac_energy", "dc_energy"]],
            on="timestamp",
        )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Energy vs GHI",
                "Energy vs Temperature",
                "Energy vs Wind Speed",
                "Daily Energy Profile",
            ),
        )

        # Energy vs GHI (Scatter)
        fig.add_trace(
            go.Scatter(
                x=merged_df["ghi"],
                y=merged_df["ac_energy"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=merged_df["temperature"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Temp (°C)", x=0.46),
                ),
                name="AC Energy",
            ),
            row=1,
            col=1,
        )

        # Energy vs Temperature (Scatter)
        fig.add_trace(
            go.Scatter(
                x=merged_df["temperature"],
                y=merged_df["ac_energy"],
                mode="markers",
                marker=dict(size=5, color=self.color_scheme["danger"]),
                name="Energy vs Temp",
            ),
            row=1,
            col=2,
        )

        # Energy vs Wind Speed (Scatter)
        fig.add_trace(
            go.Scatter(
                x=merged_df["wind_speed"],
                y=merged_df["ac_energy"],
                mode="markers",
                marker=dict(size=5, color=self.color_scheme["info"]),
                name="Energy vs Wind",
            ),
            row=2,
            col=1,
        )

        # Daily Energy Profile (aggregate by hour)
        merged_df["hour"] = pd.to_datetime(merged_df["timestamp"]).dt.hour
        hourly_avg = merged_df.groupby("hour")["ac_energy"].mean()

        fig.add_trace(
            go.Bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                marker_color=self.color_scheme["success"],
                name="Avg Hourly Energy",
            ),
            row=2,
            col=2,
        )

        # Update axes
        fig.update_xaxes(title_text="GHI (W/m²)", row=1, col=1)
        fig.update_yaxes(title_text="AC Energy (kWh)", row=1, col=1)

        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="AC Energy (kWh)", row=1, col=2)

        fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
        fig.update_yaxes(title_text="AC Energy (kWh)", row=2, col=1)

        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Avg Energy (kWh)", row=2, col=2)

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Weather Correlation Analysis",
            title_font_size=20,
        )

        return fig

    def performance_heatmap(
        self, monthly_data: pd.DataFrame, title: str = "Performance Heatmap"
    ) -> go.Figure:
        """Generate performance heatmap.

        Args:
            monthly_data: DataFrame with monthly data
            title: Chart title

        Returns:
            Plotly Figure with heatmap
        """
        # Reshape data for heatmap (assuming we have year-month data)
        # For demo, create a simple heatmap of key metrics

        metrics = ["ac_energy", "specific_yield", "capacity_factor"]
        metric_labels = ["AC Energy", "Specific Yield", "Capacity Factor"]

        # Normalize data for heatmap
        heatmap_data = []
        for metric in metrics:
            values = monthly_data[metric].values
            # Normalize to 0-100 scale
            normalized = (values - values.min()) / (values.max() - values.min()) * 100
            heatmap_data.append(normalized)

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=monthly_data["month"],
                y=metric_labels,
                colorscale="RdYlGn",
                text=[[f"{val:.1f}" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Normalized<br>Performance"),
            )
        )

        fig.update_layout(
            title=title,
            title_font_size=20,
            xaxis_title="Month",
            yaxis_title="Metric",
            height=400,
        )

        return fig

    def sensitivity_chart(
        self, sensitivity_df: pd.DataFrame, title: str = "Sensitivity Analysis"
    ) -> go.Figure:
        """Generate sensitivity analysis tornado chart.

        Args:
            sensitivity_df: DataFrame with sensitivity analysis results
            title: Chart title

        Returns:
            Plotly Figure with tornado chart
        """
        # Group by parameter and calculate range
        param_groups = sensitivity_df.groupby("Parameter")

        tornado_data = []
        for param, group in param_groups:
            min_change = group["Energy Change (%)"].min()
            max_change = group["Energy Change (%)"].max()
            range_val = max_change - min_change

            tornado_data.append({
                "Parameter": param,
                "Min": min_change,
                "Max": max_change,
                "Range": range_val,
            })

        tornado_df = pd.DataFrame(tornado_data).sort_values("Range", ascending=True)

        # Create tornado chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=tornado_df["Parameter"],
                x=tornado_df["Min"],
                orientation="h",
                name="Downside",
                marker_color=self.color_scheme["danger"],
            )
        )

        fig.add_trace(
            go.Bar(
                y=tornado_df["Parameter"],
                x=tornado_df["Max"],
                orientation="h",
                name="Upside",
                marker_color=self.color_scheme["success"],
            )
        )

        fig.update_layout(
            title=title,
            title_font_size=20,
            xaxis_title="Energy Change (%)",
            yaxis_title="Parameter",
            barmode="overlay",
            height=400,
            showlegend=True,
        )

        return fig

    def p50_p90_p99_chart(
        self, prob_data: pd.DataFrame, title: str = "Probabilistic Analysis (P50/P90/P99)"
    ) -> go.Figure:
        """Generate P50/P90/P99 probabilistic analysis chart.

        Args:
            prob_data: DataFrame with probabilistic data
            title: Chart title

        Returns:
            Plotly Figure with probabilistic analysis
        """
        fig = go.Figure()

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=prob_data["Probability Level"],
                y=prob_data["Annual Energy (kWh)"],
                marker_color=[
                    self.color_scheme["danger"],
                    self.color_scheme["warning"],
                    self.color_scheme["info"],
                    self.color_scheme["success"],
                    self.color_scheme["primary"],
                ],
                text=[f"{val:,.0f}" for val in prob_data["Annual Energy (kWh)"]],
                textposition="outside",
            )
        )

        # Add P50 reference line
        p50_value = prob_data[prob_data["Probability Level"].str.contains("P50")][
            "Annual Energy (kWh)"
        ].values[0]

        fig.add_hline(
            y=p50_value,
            line_dash="dash",
            line_color="red",
            annotation_text="P50 Reference",
            annotation_position="right",
        )

        fig.update_layout(
            title=title,
            title_font_size=20,
            xaxis_title="Probability Level",
            yaxis_title="Annual Energy (kWh)",
            height=500,
            showlegend=False,
        )

        return fig

    def annual_degradation_chart(
        self, degradation_df: pd.DataFrame, title: str = "Annual Degradation Projection"
    ) -> go.Figure:
        """Generate annual degradation projection chart.

        Args:
            degradation_df: DataFrame with degradation projections
            title: Chart title

        Returns:
            Plotly Figure with degradation analysis
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Annual Energy Production", "Cumulative Energy Production"),
        )

        # Annual Energy
        fig.add_trace(
            go.Scatter(
                x=degradation_df["year"],
                y=degradation_df["annual_energy_kwh"],
                mode="lines+markers",
                name="Annual Energy",
                line=dict(color=self.color_scheme["primary"], width=3),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.2)",
            ),
            row=1,
            col=1,
        )

        # Cumulative Energy
        fig.add_trace(
            go.Scatter(
                x=degradation_df["year"],
                y=degradation_df["cumulative_energy_kwh"],
                mode="lines+markers",
                name="Cumulative Energy",
                line=dict(color=self.color_scheme["success"], width=3),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.2)",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Annual Energy (kWh)", row=1, col=1)

        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Energy (kWh)", row=1, col=2)

        fig.update_layout(
            title=title,
            title_font_size=20,
            height=500,
            showlegend=True,
        )

        return fig

    def create_altair_chart(self, data: pd.DataFrame, x: str, y: str, chart_type: str = "bar") -> alt.Chart:
        """Create Altair chart for declarative visualizations.

        Args:
            data: DataFrame with data to visualize
            x: X-axis column name
            y: Y-axis column name
            chart_type: Chart type ('bar', 'line', 'scatter')

        Returns:
            Altair Chart object
        """
        if chart_type == "bar":
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x=alt.X(x, title=x),
                    y=alt.Y(y, title=y),
                    tooltip=[x, y],
                )
            )
        elif chart_type == "line":
            chart = (
                alt.Chart(data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(x, title=x),
                    y=alt.Y(y, title=y),
                    tooltip=[x, y],
                )
            )
        elif chart_type == "scatter":
            chart = (
                alt.Chart(data)
                .mark_circle(size=60)
                .encode(
                    x=alt.X(x, title=x),
                    y=alt.Y(y, title=y),
                    tooltip=[x, y],
                )
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        return chart.properties(width=600, height=400).interactive()
