"""
Circularity UI & 3R Dashboard (B11-S06)

This module provides visualization and dashboard components for circularity
analysis, including material flow diagrams, 3R metrics, and circular economy scores.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ThreeRMetrics(BaseModel):
    """Metrics for Reduce, Reuse, Recycle analysis."""

    reduce_virgin_material_pct: float = Field(ge=0, le=100, description="Virgin material reduction %")
    reduce_energy_consumption_pct: float = Field(ge=0, le=100, description="Energy consumption reduction %")
    reduce_waste_generation_pct: float = Field(ge=0, le=100, description="Waste generation reduction %")
    reuse_modules_count: int = Field(ge=0, description="Number of modules reused")
    reuse_capacity_retention_avg: float = Field(ge=0, le=100, description="Average capacity retention %")
    reuse_market_value_usd: float = Field(ge=0, description="Total reuse market value")
    recycle_recovery_rate_pct: float = Field(ge=0, le=100, description="Material recovery rate %")
    recycle_material_value_usd: float = Field(ge=0, description="Recycled material value")
    recycle_environmental_benefit_kg_co2eq: float = Field(ge=0, description="Environmental benefit")


class CircularEconomyScore(BaseModel):
    """Comprehensive circular economy scoring."""

    overall_score: float = Field(ge=0, le=100, description="Overall circularity score (0-100)")
    material_efficiency_score: float = Field(ge=0, le=100, description="Material efficiency score")
    product_longevity_score: float = Field(ge=0, le=100, description="Product longevity score")
    recycling_effectiveness_score: float = Field(ge=0, le=100, description="Recycling effectiveness")
    environmental_impact_score: float = Field(ge=0, le=100, description="Environmental impact score")
    economic_viability_score: float = Field(ge=0, le=100, description="Economic viability score")
    rating: str = Field(description="Letter rating (A+ to F)")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class MaterialFlow(BaseModel):
    """Material flow data for Sankey diagrams."""

    source_nodes: List[str] = Field(description="Source nodes")
    target_nodes: List[str] = Field(description="Target nodes")
    values: List[float] = Field(description="Flow values")
    labels: List[str] = Field(description="All node labels")


class CircularityUI:
    """
    Visualization and dashboard components for circularity analysis.

    Provides methods for creating material flow diagrams, 3R metrics dashboards,
    and circular economy score visualizations.
    """

    def __init__(self, theme: str = "plotly"):
        """
        Initialize CircularityUI.

        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn')
        """
        self.theme = theme

    def material_flow_diagrams(
        self,
        input_materials: Dict[str, float],
        recovered_materials: Dict[str, float],
        waste_materials: Dict[str, float],
        reused_materials: Optional[Dict[str, float]] = None,
        title: str = "PV Module Material Flow Analysis"
    ) -> go.Figure:
        """
        Create Sankey diagram showing material flows through lifecycle.

        Args:
            input_materials: Input materials by type (kg)
            recovered_materials: Recovered materials by type (kg)
            waste_materials: Waste materials by type (kg)
            reused_materials: Reused materials by type (kg)
            title: Diagram title

        Returns:
            Plotly Figure with Sankey diagram
        """
        # Build material flow
        flow = self._build_material_flow(
            input_materials,
            recovered_materials,
            waste_materials,
            reused_materials
        )

        # Create color scheme
        colors = self._generate_flow_colors(len(flow.labels))

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=flow.labels,
                color=colors
            ),
            link=dict(
                source=[flow.labels.index(s) for s in flow.source_nodes],
                target=[flow.labels.index(t) for t in flow.target_nodes],
                value=flow.values,
                color="rgba(0,0,0,0.2)"
            )
        )])

        fig.update_layout(
            title=title,
            font=dict(size=12),
            height=600,
            template=self.theme
        )

        return fig

    def three_r_metrics(
        self,
        metrics: ThreeRMetrics,
        show_details: bool = True
    ) -> go.Figure:
        """
        Create comprehensive 3R (Reduce, Reuse, Recycle) metrics visualization.

        Args:
            metrics: ThreeRMetrics object with all metrics
            show_details: Whether to show detailed breakdown

        Returns:
            Plotly Figure with 3R metrics dashboard
        """
        # Create subplots
        if show_details:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "Reduce Metrics",
                    "Reuse Metrics",
                    "Recycle Metrics",
                    "Material Reduction",
                    "Reuse Value",
                    "Recycling Impact"
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
                ]
            )

            # Row 1: Key indicators
            # Reduce indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=metrics.reduce_virgin_material_pct,
                title={"text": "Virgin Material Reduction"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 33], "color": "lightgray"},
                        {"range": [33, 66], "color": "gray"},
                        {"range": [66, 100], "color": "lightgreen"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80
                    }
                }
            ), row=1, col=1)

            # Reuse indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.reuse_capacity_retention_avg,
                title={"text": "Avg Reuse Capacity"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 70], "color": "lightgray"},
                        {"range": [70, 85], "color": "lightblue"},
                        {"range": [85, 100], "color": "blue"}
                    ]
                }
            ), row=1, col=2)

            # Recycle indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.recycle_recovery_rate_pct,
                title={"text": "Material Recovery Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkorange"},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 80], "color": "lightyellow"},
                        {"range": [80, 100], "color": "orange"}
                    ]
                }
            ), row=1, col=3)

            # Row 2: Detailed metrics
            # Reduce details
            reduce_data = pd.DataFrame({
                "Metric": ["Virgin Material", "Energy", "Waste"],
                "Reduction %": [
                    metrics.reduce_virgin_material_pct,
                    metrics.reduce_energy_consumption_pct,
                    metrics.reduce_waste_generation_pct
                ]
            })
            fig.add_trace(go.Bar(
                x=reduce_data["Metric"],
                y=reduce_data["Reduction %"],
                marker_color="green",
                name="Reduce"
            ), row=2, col=1)

            # Reuse details
            fig.add_trace(go.Bar(
                x=["Modules Reused", "Market Value (k$)"],
                y=[metrics.reuse_modules_count, metrics.reuse_market_value_usd / 1000],
                marker_color="blue",
                name="Reuse"
            ), row=2, col=2)

            # Recycle details
            fig.add_trace(go.Bar(
                x=["Material Value (k$)", "CO2 Avoided (tons)"],
                y=[
                    metrics.recycle_material_value_usd / 1000,
                    metrics.recycle_environmental_benefit_kg_co2eq / 1000
                ],
                marker_color="orange",
                name="Recycle"
            ), row=2, col=3)

            fig.update_layout(
                height=800,
                showlegend=False,
                template=self.theme,
                title_text="3R (Reduce, Reuse, Recycle) Metrics Dashboard"
            )

        else:
            # Simple 3-bar comparison
            fig = go.Figure()

            categories = ["Reduce", "Reuse", "Recycle"]
            values = [
                (metrics.reduce_virgin_material_pct + metrics.reduce_energy_consumption_pct + metrics.reduce_waste_generation_pct) / 3,
                metrics.reuse_capacity_retention_avg,
                metrics.recycle_recovery_rate_pct
            ]

            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=["green", "blue", "orange"],
                text=[f"{v:.1f}%" for v in values],
                textposition="outside"
            ))

            fig.update_layout(
                title="3R Performance Summary",
                yaxis_title="Performance (%)",
                yaxis_range=[0, 100],
                template=self.theme,
                height=400
            )

        return fig

    def circular_economy_score(
        self,
        material_circularity_index: float,
        recovery_rate: float,
        reuse_rate: float,
        lifetime_extension_factor: float,
        carbon_footprint_kg: float,
        roi_percent: float
    ) -> CircularEconomyScore:
        """
        Calculate comprehensive circular economy score.

        Args:
            material_circularity_index: MCI score (0-1)
            recovery_rate: End-of-life recovery rate (0-1)
            reuse_rate: Module reuse rate (0-1)
            lifetime_extension_factor: Lifetime extension (>=1)
            carbon_footprint_kg: Total carbon footprint
            roi_percent: Economic ROI percentage

        Returns:
            CircularEconomyScore with detailed scoring and recommendations
        """
        # Calculate sub-scores (0-100 scale)
        material_efficiency = material_circularity_index * 100

        # Product longevity score
        longevity_score = min(100, (lifetime_extension_factor - 1) * 100 + 50)

        # Recycling effectiveness score
        recycling_score = recovery_rate * 100

        # Environmental impact score (inverse of carbon footprint, normalized)
        # Lower carbon footprint = higher score
        # Assume baseline of 2000 kg CO2eq for a module lifecycle
        baseline_carbon = 2000
        if carbon_footprint_kg <= baseline_carbon:
            env_score = 50 + (baseline_carbon - carbon_footprint_kg) / baseline_carbon * 50
        else:
            env_score = max(0, 50 - (carbon_footprint_kg - baseline_carbon) / baseline_carbon * 50)

        # Economic viability score
        # Positive ROI = good, >20% = excellent
        if roi_percent >= 20:
            econ_score = 100
        elif roi_percent >= 0:
            econ_score = 50 + (roi_percent / 20) * 50
        else:
            econ_score = max(0, 50 + roi_percent * 2.5)

        # Calculate weighted overall score
        weights = {
            "material_efficiency": 0.25,
            "longevity": 0.20,
            "recycling": 0.20,
            "environmental": 0.20,
            "economic": 0.15
        }

        overall = (
            material_efficiency * weights["material_efficiency"] +
            longevity_score * weights["longevity"] +
            recycling_score * weights["recycling"] +
            env_score * weights["environmental"] +
            econ_score * weights["economic"]
        )

        # Determine rating
        if overall >= 90:
            rating = "A+"
        elif overall >= 80:
            rating = "A"
        elif overall >= 70:
            rating = "B"
        elif overall >= 60:
            rating = "C"
        elif overall >= 50:
            rating = "D"
        else:
            rating = "F"

        # Generate recommendations
        recommendations = []
        if material_efficiency < 60:
            recommendations.append("Increase use of recycled materials in manufacturing")
        if longevity_score < 60:
            recommendations.append("Implement preventive maintenance to extend product lifetime")
        if recycling_score < 70:
            recommendations.append("Improve end-of-life collection and recycling processes")
        if env_score < 60:
            recommendations.append("Reduce carbon footprint through cleaner manufacturing")
        if econ_score < 50:
            recommendations.append("Optimize economics through improved material recovery or reduced costs")
        if reuse_rate < 0.2:
            recommendations.append("Develop second-life markets to increase module reuse")

        return CircularEconomyScore(
            overall_score=overall,
            material_efficiency_score=material_efficiency,
            product_longevity_score=longevity_score,
            recycling_effectiveness_score=recycling_score,
            environmental_impact_score=env_score,
            economic_viability_score=econ_score,
            rating=rating,
            recommendations=recommendations
        )

    def plot_circular_economy_score(
        self,
        score: CircularEconomyScore
    ) -> go.Figure:
        """
        Visualize circular economy score with radar chart.

        Args:
            score: CircularEconomyScore object

        Returns:
            Plotly Figure with radar chart
        """
        categories = [
            "Material<br>Efficiency",
            "Product<br>Longevity",
            "Recycling<br>Effectiveness",
            "Environmental<br>Impact",
            "Economic<br>Viability"
        ]

        values = [
            score.material_efficiency_score,
            score.product_longevity_score,
            score.recycling_effectiveness_score,
            score.environmental_impact_score,
            score.economic_viability_score
        ]

        fig = go.Figure()

        # Add the score
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Score',
            line_color='blue'
        ))

        # Add benchmark (80% for all categories)
        benchmark = [80] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=benchmark,
            theta=categories,
            fill='toself',
            name='Target (80%)',
            line_color='green',
            opacity=0.3
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title=f"Circular Economy Score: {score.overall_score:.1f}/100 (Rating: {score.rating})",
            template=self.theme,
            height=500
        )

        return fig

    def create_dashboard_data(
        self,
        modules_data: List[Dict],
        recovery_data: Dict,
        economic_data: Dict,
        environmental_data: Dict
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive dashboard data.

        Args:
            modules_data: List of module information
            recovery_data: Material recovery data
            economic_data: Economic analysis data
            environmental_data: Environmental impact data

        Returns:
            Dictionary with dashboard data and visualizations
        """
        # Calculate 3R metrics
        total_modules = len(modules_data)
        reused_modules = sum(1 for m in modules_data if m.get("reused", False))
        recycled_modules = sum(1 for m in modules_data if m.get("recycled", False))

        three_r = ThreeRMetrics(
            reduce_virgin_material_pct=recovery_data.get("virgin_reduction_pct", 0),
            reduce_energy_consumption_pct=environmental_data.get("energy_reduction_pct", 0),
            reduce_waste_generation_pct=recovery_data.get("waste_reduction_pct", 0),
            reuse_modules_count=reused_modules,
            reuse_capacity_retention_avg=np.mean([
                m.get("capacity_retention", 0) for m in modules_data if m.get("reused", False)
            ]) if reused_modules > 0 else 0,
            reuse_market_value_usd=economic_data.get("reuse_value", 0),
            recycle_recovery_rate_pct=recovery_data.get("recovery_rate_pct", 0),
            recycle_material_value_usd=economic_data.get("material_value", 0),
            recycle_environmental_benefit_kg_co2eq=environmental_data.get("carbon_avoided", 0)
        )

        # Calculate circular economy score
        ce_score = self.circular_economy_score(
            material_circularity_index=recovery_data.get("mci", 0.5),
            recovery_rate=recovery_data.get("recovery_rate_pct", 0) / 100,
            reuse_rate=reused_modules / total_modules if total_modules > 0 else 0,
            lifetime_extension_factor=recovery_data.get("lifetime_extension", 1.0),
            carbon_footprint_kg=environmental_data.get("carbon_footprint_kg", 1500),
            roi_percent=economic_data.get("roi_percent", 0)
        )

        # Create visualizations
        material_flow_fig = self.material_flow_diagrams(
            input_materials=recovery_data.get("input_materials", {}),
            recovered_materials=recovery_data.get("recovered_materials", {}),
            waste_materials=recovery_data.get("waste_materials", {}),
            reused_materials=recovery_data.get("reused_materials", {})
        )

        three_r_fig = self.three_r_metrics(three_r, show_details=True)
        ce_score_fig = self.plot_circular_economy_score(ce_score)

        return {
            "summary": {
                "total_modules": total_modules,
                "reused_modules": reused_modules,
                "recycled_modules": recycled_modules,
                "circular_economy_score": ce_score.overall_score,
                "ce_rating": ce_score.rating
            },
            "metrics": {
                "three_r": three_r.model_dump(),
                "circular_economy": ce_score.model_dump()
            },
            "visualizations": {
                "material_flow": material_flow_fig,
                "three_r_metrics": three_r_fig,
                "circular_economy_score": ce_score_fig
            }
        }

    @staticmethod
    def _build_material_flow(
        input_materials: Dict[str, float],
        recovered_materials: Dict[str, float],
        waste_materials: Dict[str, float],
        reused_materials: Optional[Dict[str, float]]
    ) -> MaterialFlow:
        """Build material flow data for Sankey diagram."""
        sources = []
        targets = []
        values = []

        # Collect all unique material types
        all_materials = set(
            list(input_materials.keys()) +
            list(recovered_materials.keys()) +
            list(waste_materials.keys()) +
            (list(reused_materials.keys()) if reused_materials else [])
        )

        # Create node labels
        labels = ["Input"]
        material_labels = [f"Material: {m}" for m in all_materials]
        labels.extend(material_labels)
        labels.extend(["Processing", "Recovery", "Waste", "Reuse"])

        # Input to materials
        for material, amount in input_materials.items():
            sources.append("Input")
            targets.append(f"Material: {material}")
            values.append(amount)

        # Materials to processing
        for material, amount in input_materials.items():
            sources.append(f"Material: {material}")
            targets.append("Processing")
            values.append(amount)

        # Processing to recovery/waste/reuse
        total_input = sum(input_materials.values())
        total_recovered = sum(recovered_materials.values())
        total_waste = sum(waste_materials.values())
        total_reused = sum(reused_materials.values()) if reused_materials else 0

        if total_recovered > 0:
            sources.append("Processing")
            targets.append("Recovery")
            values.append(total_recovered)

        if total_waste > 0:
            sources.append("Processing")
            targets.append("Waste")
            values.append(total_waste)

        if total_reused > 0:
            sources.append("Processing")
            targets.append("Reuse")
            values.append(total_reused)

        return MaterialFlow(
            source_nodes=sources,
            target_nodes=targets,
            values=values,
            labels=labels
        )

    @staticmethod
    def _generate_flow_colors(num_nodes: int) -> List[str]:
        """Generate color scheme for flow diagram."""
        # Use a color palette
        base_colors = [
            "#2E86AB",  # Blue
            "#A23B72",  # Purple
            "#F18F01",  # Orange
            "#C73E1D",  # Red
            "#6A994E",  # Green
            "#BC4B51",  # Dark red
            "#8CB369",  # Light green
            "#F4A259"   # Light orange
        ]

        # Repeat colors if needed
        colors = []
        for i in range(num_nodes):
            colors.append(base_colors[i % len(base_colors)])

        return colors


# Streamlit dashboard functions (only if Streamlit is available)
if STREAMLIT_AVAILABLE:
    def create_streamlit_dashboard(
        dashboard_data: Dict[str, Any],
        title: str = "PV Circularity & 3R Dashboard"
    ) -> None:
        """
        Create interactive Streamlit dashboard.

        Args:
            dashboard_data: Dashboard data from create_dashboard_data()
            title: Dashboard title
        """
        st.set_page_config(page_title=title, layout="wide")
        st.title(title)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Modules",
                dashboard_data["summary"]["total_modules"]
            )

        with col2:
            st.metric(
                "Reused Modules",
                dashboard_data["summary"]["reused_modules"]
            )

        with col3:
            st.metric(
                "Recycled Modules",
                dashboard_data["summary"]["recycled_modules"]
            )

        with col4:
            st.metric(
                "CE Score",
                f"{dashboard_data['summary']['circular_economy_score']:.1f}",
                delta=dashboard_data["summary"]["ce_rating"]
            )

        # Visualizations
        st.header("Material Flow Analysis")
        st.plotly_chart(
            dashboard_data["visualizations"]["material_flow"],
            use_container_width=True
        )

        st.header("3R Metrics")
        st.plotly_chart(
            dashboard_data["visualizations"]["three_r_metrics"],
            use_container_width=True
        )

        st.header("Circular Economy Score")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(
                dashboard_data["visualizations"]["circular_economy_score"],
                use_container_width=True
            )

        with col2:
            st.subheader("Recommendations")
            for i, rec in enumerate(dashboard_data["metrics"]["circular_economy"]["recommendations"], 1):
                st.write(f"{i}. {rec}")
