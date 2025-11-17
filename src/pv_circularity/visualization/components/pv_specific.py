"""
PV-specific custom visualization components.

This module provides specialized visualization components for photovoltaic
system analysis, including I-V curves, efficiency heatmaps, degradation
analysis, and material flow diagrams.
"""

from typing import Optional, List, Dict, Any, Tuple
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


class IVCurveVisualizer:
    """
    Visualize current-voltage (I-V) characteristics of PV cells/modules.

    This component creates I-V and P-V curve visualizations commonly used
    in photovoltaic system characterization and analysis.

    Examples:
        >>> iv_viz = IVCurveVisualizer()
        >>> fig = iv_viz.plot_iv_curve(
        ...     voltage=np.linspace(0, 40, 100),
        ...     current=current_data,
        ...     title='PV Module I-V Characteristics'
        ... )
    """

    def __init__(self) -> None:
        """Initialize the I-V curve visualizer."""
        self.default_height: int = 600
        self.default_width: int = 900

    def plot_iv_curve(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        title: str = "I-V Characteristic Curve",
        voc: Optional[float] = None,
        isc: Optional[float] = None,
        vmp: Optional[float] = None,
        imp: Optional[float] = None,
        include_power: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Plot I-V curve with optional maximum power point marking.

        Args:
            voltage: Array of voltage values
            current: Array of current values
            title: Chart title
            voc: Open-circuit voltage (for annotation)
            isc: Short-circuit current (for annotation)
            vmp: Voltage at maximum power point
            imp: Current at maximum power point
            include_power: Whether to include P-V curve
            height: Figure height in pixels
            width: Figure width in pixels

        Returns:
            Plotly figure with I-V (and optionally P-V) curves

        Examples:
            >>> v = np.linspace(0, 40, 100)
            >>> i = 8 * (1 - v/40)  # Simplified linear model
            >>> viz = IVCurveVisualizer()
            >>> fig = viz.plot_iv_curve(v, i, voc=40, isc=8)
        """
        if include_power:
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # I-V curve
            fig.add_trace(
                go.Scatter(
                    x=voltage,
                    y=current,
                    mode="lines",
                    name="Current",
                    line=dict(color="steelblue", width=3),
                ),
                secondary_y=False,
            )

            # P-V curve
            power = voltage * current
            fig.add_trace(
                go.Scatter(
                    x=voltage,
                    y=power,
                    mode="lines",
                    name="Power",
                    line=dict(color="orangered", width=3, dash="dash"),
                ),
                secondary_y=True,
            )

            # Mark maximum power point
            if vmp is not None and imp is not None:
                pmp = vmp * imp
                fig.add_trace(
                    go.Scatter(
                        x=[vmp],
                        y=[imp],
                        mode="markers",
                        name="MPP (I-V)",
                        marker=dict(size=12, color="green", symbol="star"),
                        showlegend=False,
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[vmp],
                        y=[pmp],
                        mode="markers",
                        name="MPP (P-V)",
                        marker=dict(size=12, color="green", symbol="star"),
                    ),
                    secondary_y=True,
                )

            # Update axes titles
            fig.update_yaxes(title_text="Current (A)", secondary_y=False)
            fig.update_yaxes(title_text="Power (W)", secondary_y=True)

        else:
            # Simple I-V curve
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=voltage,
                    y=current,
                    mode="lines",
                    name="I-V Curve",
                    line=dict(color="steelblue", width=3),
                )
            )

            # Mark key points
            if voc is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[voc],
                        y=[0],
                        mode="markers+text",
                        name="Voc",
                        marker=dict(size=10, color="red"),
                        text=["Voc"],
                        textposition="top center",
                    )
                )

            if isc is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[isc],
                        mode="markers+text",
                        name="Isc",
                        marker=dict(size=10, color="red"),
                        text=["Isc"],
                        textposition="middle right",
                    )
                )

            fig.update_yaxes(title_text="Current (A)")

        # Common layout updates
        fig.update_xaxes(title_text="Voltage (V)")
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=height or self.default_height,
            width=width or self.default_width,
            hovermode="x unified",
            legend=dict(x=0.7, y=0.95),
        )

        return fig

    def compare_iv_curves(
        self,
        curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "I-V Curve Comparison",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        """
        Compare multiple I-V curves on the same plot.

        Args:
            curves_data: Dictionary mapping curve names to (voltage, current) tuples
            title: Chart title
            height: Figure height
            width: Figure width

        Returns:
            Plotly figure comparing multiple I-V curves

        Examples:
            >>> viz = IVCurveVisualizer()
            >>> curves = {
            ...     'Standard': (v1, i1),
            ...     'High Temp': (v2, i2),
            ...     'Low Irradiance': (v3, i3)
            ... }
            >>> fig = viz.compare_iv_curves(curves)
        """
        fig = go.Figure()

        colors = ["steelblue", "orangered", "green", "purple", "brown"]

        for idx, (name, (voltage, current)) in enumerate(curves_data.items()):
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=voltage,
                    y=current,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="Voltage (V)",
            yaxis_title="Current (A)",
            height=height or self.default_height,
            width=width or self.default_width,
            hovermode="x unified",
            legend=dict(x=0.7, y=0.95),
        )

        return fig


class EfficiencyHeatmap:
    """
    Create efficiency heatmaps for PV system analysis.

    This component generates heatmap visualizations showing efficiency
    variations across different conditions (temperature, irradiance, etc.).

    Examples:
        >>> heatmap = EfficiencyHeatmap()
        >>> fig = heatmap.create_2d_efficiency_map(
        ...     temperatures=[25, 35, 45],
        ...     irradiances=[200, 400, 600, 800, 1000],
        ...     efficiency_matrix=efficiency_data
        ... )
    """

    def __init__(self) -> None:
        """Initialize the efficiency heatmap visualizer."""
        self.default_colorscale: str = "RdYlGn"

    def create_2d_efficiency_map(
        self,
        x_values: List[float],
        y_values: List[float],
        efficiency_matrix: np.ndarray,
        x_label: str = "Parameter X",
        y_label: str = "Parameter Y",
        title: str = "Efficiency Heatmap",
        show_values: bool = True,
        colorscale: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a 2D efficiency heatmap.

        Args:
            x_values: Values for x-axis (e.g., temperatures)
            y_values: Values for y-axis (e.g., irradiances)
            efficiency_matrix: 2D array of efficiency values
            x_label: Label for x-axis
            y_label: Label for y-axis
            title: Chart title
            show_values: Whether to display values in cells
            colorscale: Plotly colorscale name

        Returns:
            Plotly heatmap figure

        Examples:
            >>> temps = [25, 30, 35, 40, 45]
            >>> irrad = [200, 400, 600, 800, 1000]
            >>> eff = np.random.rand(5, 5) * 20 + 10  # 10-30% efficiency
            >>> viz = EfficiencyHeatmap()
            >>> fig = viz.create_2d_efficiency_map(temps, irrad, eff)
        """
        colorscale = colorscale or self.default_colorscale

        fig = go.Figure(
            data=go.Heatmap(
                z=efficiency_matrix,
                x=x_values,
                y=y_values,
                colorscale=colorscale,
                text=efficiency_matrix,
                texttemplate="%{text:.1f}%" if show_values else None,
                textfont={"size": 10},
                colorbar=dict(title="Efficiency (%)"),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=500,
            width=700,
        )

        return fig


class DegradationAnalyzer:
    """
    Analyze and visualize PV system degradation over time.

    This component creates visualizations for degradation analysis,
    including performance loss trends and degradation rate calculations.

    Examples:
        >>> analyzer = DegradationAnalyzer()
        >>> fig = analyzer.plot_degradation_trend(
        ...     time_years=years,
        ...     performance_ratio=pr_data,
        ...     calculate_rate=True
        ... )
    """

    def __init__(self) -> None:
        """Initialize the degradation analyzer."""
        pass

    def plot_degradation_trend(
        self,
        time_years: np.ndarray,
        performance_ratio: np.ndarray,
        title: str = "PV System Degradation Analysis",
        calculate_rate: bool = True,
        add_forecast: bool = False,
        forecast_years: int = 5,
    ) -> go.Figure:
        """
        Plot degradation trend with optional rate calculation and forecast.

        Args:
            time_years: Array of years
            performance_ratio: Array of performance ratios (0-100)
            title: Chart title
            calculate_rate: Whether to calculate and show degradation rate
            add_forecast: Whether to add forecast line
            forecast_years: Number of years to forecast

        Returns:
            Plotly figure showing degradation analysis

        Examples:
            >>> years = np.arange(0, 10)
            >>> pr = 100 - 0.5 * years + np.random.randn(10) * 0.5
            >>> analyzer = DegradationAnalyzer()
            >>> fig = analyzer.plot_degradation_trend(years, pr)
        """
        fig = go.Figure()

        # Plot actual data
        fig.add_trace(
            go.Scatter(
                x=time_years,
                y=performance_ratio,
                mode="markers+lines",
                name="Measured PR",
                marker=dict(size=8, color="steelblue"),
                line=dict(color="steelblue", width=2),
            )
        )

        # Calculate degradation rate
        if calculate_rate and len(time_years) > 1:
            # Linear regression
            coeffs = np.polyfit(time_years, performance_ratio, 1)
            trend_line = np.poly1d(coeffs)

            # Degradation rate (negative slope)
            degradation_rate = -coeffs[0]

            # Plot trend line
            fig.add_trace(
                go.Scatter(
                    x=time_years,
                    y=trend_line(time_years),
                    mode="lines",
                    name=f"Trend (Rate: {degradation_rate:.2f}%/yr)",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            # Add forecast
            if add_forecast:
                forecast_time = np.arange(
                    time_years[-1] + 1, time_years[-1] + forecast_years + 1
                )
                forecast_pr = trend_line(forecast_time)

                fig.add_trace(
                    go.Scatter(
                        x=forecast_time,
                        y=forecast_pr,
                        mode="lines",
                        name="Forecast",
                        line=dict(color="orange", width=2, dash="dot"),
                    )
                )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="Time (years)",
            yaxis_title="Performance Ratio (%)",
            height=500,
            width=800,
            hovermode="x unified",
        )

        return fig


class SankeyFlowDiagram:
    """
    Create Sankey diagrams for material and energy flow analysis.

    This component creates Sankey flow diagrams useful for circular
    economy analysis and material/energy flow visualization.

    Examples:
        >>> sankey = SankeyFlowDiagram()
        >>> fig = sankey.create_material_flow(
        ...     sources=['Production', 'Production', 'Use'],
        ...     targets=['Use', 'Waste', 'Recycling'],
        ...     values=[1000, 50, 800]
        ... )
    """

    def __init__(self) -> None:
        """Initialize the Sankey diagram creator."""
        pass

    def create_material_flow(
        self,
        sources: List[str],
        targets: List[str],
        values: List[float],
        title: str = "Material Flow Diagram",
        node_colors: Optional[List[str]] = None,
        link_colors: Optional[List[str]] = None,
    ) -> go.Figure:
        """
        Create a Sankey diagram for material flow analysis.

        Args:
            sources: List of source node names
            targets: List of target node names
            values: List of flow values
            title: Chart title
            node_colors: List of colors for nodes
            link_colors: List of colors for links

        Returns:
            Plotly Sankey diagram

        Examples:
            >>> sankey = SankeyFlowDiagram()
            >>> sources = ['Manufacturing', 'Manufacturing', 'Use Phase', 'Collection']
            >>> targets = ['Use Phase', 'Loss', 'Collection', 'Recycling']
            >>> values = [1000, 50, 900, 800]
            >>> fig = sankey.create_material_flow(sources, targets, values)
        """
        # Get unique nodes
        all_nodes = list(set(sources + targets))
        node_indices = {node: idx for idx, node in enumerate(all_nodes)}

        # Map sources and targets to indices
        source_indices = [node_indices[s] for s in sources]
        target_indices = [node_indices[t] for t in targets]

        # Default colors
        if node_colors is None:
            node_colors = [
                "#1F77B4",
                "#FF7F0E",
                "#2CA02C",
                "#D62728",
                "#9467BD",
                "#8C564B",
            ] * (len(all_nodes) // 6 + 1)
            node_colors = node_colors[: len(all_nodes)]

        if link_colors is None:
            link_colors = ["rgba(31, 119, 180, 0.4)"] * len(values)

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color=node_colors,
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values,
                        color=link_colors,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            font=dict(size=12),
            height=600,
            width=1000,
        )

        return fig

    def create_energy_balance(
        self,
        input_energy: float,
        output_useful: float,
        losses: Dict[str, float],
        title: str = "Energy Balance Diagram",
    ) -> go.Figure:
        """
        Create an energy balance Sankey diagram.

        Args:
            input_energy: Total input energy
            output_useful: Useful energy output
            losses: Dictionary of loss categories and values
            title: Chart title

        Returns:
            Plotly Sankey diagram showing energy balance

        Examples:
            >>> sankey = SankeyFlowDiagram()
            >>> losses = {
            ...     'Thermal Loss': 150,
            ...     'Optical Loss': 50,
            ...     'Electrical Loss': 100
            ... }
            >>> fig = sankey.create_energy_balance(1000, 700, losses)
        """
        sources = []
        targets = []
        values = []

        # Input to output
        sources.append("Input Energy")
        targets.append("Useful Output")
        values.append(output_useful)

        # Input to losses
        for loss_type, loss_value in losses.items():
            sources.append("Input Energy")
            targets.append(loss_type)
            values.append(loss_value)

        return self.create_material_flow(
            sources=sources, targets=targets, values=values, title=title
        )
