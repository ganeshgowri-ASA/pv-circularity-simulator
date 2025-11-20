"""
Materials Engineering Database Module (Branch B01).

Comprehensive materials database for PV technologies including:
- Silicon (c-Si, poly-Si, mono-Si, bifacial)
- Perovskite (single-junction, tandem)
- Thin-films (CIGS, CdTe, a-Si)
- Advanced materials (Tandem, HIT, multi-junction)
- Material property lookup and comparison
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constants import MATERIAL_PROPERTIES, MATERIAL_COLORS
from utils.validators import MaterialProperties


class MaterialsDatabase:
    """Comprehensive materials engineering database."""

    def __init__(self):
        """Initialize materials database."""
        self.materials = MATERIAL_PROPERTIES
        self.material_names = list(self.materials.keys())

    def get_material(self, material_name: str) -> Optional[Dict]:
        """
        Get material properties by name.

        Args:
            material_name: Material identifier

        Returns:
            Dictionary of material properties
        """
        return self.materials.get(material_name)

    def get_all_materials(self) -> pd.DataFrame:
        """
        Get all materials as DataFrame.

        Returns:
            DataFrame with all material properties
        """
        data = []
        for mat_id, props in self.materials.items():
            row = {
                'Material ID': mat_id,
                'Name': props['name'],
                'Type': props['type'],
                'Bandgap (eV)': props['bandgap'],
                'Efficiency Min (%)': props['efficiency_range'][0],
                'Efficiency Max (%)': props['efficiency_range'][1],
                'Cost ($/Wp)': props['cost_per_wp'],
                'Degradation (%/year)': props['degradation_rate'],
                'Recyclability (%)': props['recyclability'],
                'Temp Coefficient (%/°C)': props['temp_coefficient']
            }
            data.append(row)

        return pd.DataFrame(data)

    def compare_materials(
        self,
        material_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple materials across key metrics.

        Args:
            material_ids: List of material identifiers
            metrics: Optional list of specific metrics to compare

        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = ['efficiency_range', 'cost_per_wp', 'degradation_rate', 'recyclability']

        comparison = {}
        for mat_id in material_ids:
            if mat_id in self.materials:
                mat_props = self.materials[mat_id]
                comparison[mat_props['name']] = {
                    metric: mat_props.get(metric, 'N/A')
                    for metric in metrics
                }

        return pd.DataFrame(comparison).T

    def get_materials_by_type(self, material_type: str) -> List[str]:
        """
        Get materials by type.

        Args:
            material_type: Type filter (semiconductor, thin-film, tandem, etc.)

        Returns:
            List of material IDs
        """
        return [
            mat_id for mat_id, props in self.materials.items()
            if props.get('type') == material_type
        ]

    def get_best_materials(
        self,
        criterion: str = 'efficiency',
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top materials by criterion.

        Args:
            criterion: Ranking criterion (efficiency, cost, recyclability)
            top_n: Number of top materials to return

        Returns:
            List of (material_id, value) tuples
        """
        material_values = []

        for mat_id, props in self.materials.items():
            if criterion == 'efficiency':
                value = props['efficiency_range'][1]  # Max efficiency
            elif criterion == 'cost':
                value = -props['cost_per_wp']  # Lower is better
            elif criterion == 'recyclability':
                value = props['recyclability']
            elif criterion == 'degradation':
                value = -props['degradation_rate']  # Lower is better
            else:
                continue

            material_values.append((mat_id, value))

        # Sort and return top N
        material_values.sort(key=lambda x: x[1], reverse=True)
        return material_values[:top_n]

    def calculate_material_score(
        self,
        material_id: str,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overall material score based on multiple factors.

        Args:
            material_id: Material identifier
            weights: Optional custom weights for scoring

        Returns:
            Overall score (0-100)
        """
        if material_id not in self.materials:
            return 0.0

        if weights is None:
            weights = {
                'efficiency': 0.35,
                'cost': 0.25,
                'degradation': 0.20,
                'recyclability': 0.20
            }

        props = self.materials[material_id]

        # Normalize metrics to 0-100 scale
        efficiency_score = (props['efficiency_range'][1] / 35.0) * 100  # Max theoretical ~35%
        cost_score = (1 - min(props['cost_per_wp'] / 1.0, 1.0)) * 100  # Lower cost = higher score
        degradation_score = (1 - min(props['degradation_rate'] / 5.0, 1.0)) * 100
        recyclability_score = props['recyclability']

        # Weighted sum
        total_score = (
            weights['efficiency'] * efficiency_score +
            weights['cost'] * cost_score +
            weights['degradation'] * degradation_score +
            weights['recyclability'] * recyclability_score
        )

        return min(max(total_score, 0), 100)

    def create_comparison_radar(self, material_ids: List[str]) -> go.Figure:
        """
        Create radar chart comparing materials.

        Args:
            material_ids: List of material IDs to compare

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        categories = ['Efficiency', 'Cost Effectiveness', 'Stability', 'Recyclability']

        for mat_id in material_ids:
            if mat_id not in self.materials:
                continue

            props = self.materials[mat_id]

            # Normalize values to 0-100 scale
            values = [
                (props['efficiency_range'][1] / 35.0) * 100,
                (1 - min(props['cost_per_wp'] / 1.0, 1.0)) * 100,
                (1 - min(props['degradation_rate'] / 5.0, 1.0)) * 100,
                props['recyclability']
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=props['name'],
                line=dict(color=props.get('color', '#3498DB'))
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Material Comparison Radar Chart",
            height=500
        )

        return fig

    def create_efficiency_cost_scatter(self) -> go.Figure:
        """
        Create efficiency vs cost scatter plot.

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for mat_id, props in self.materials.items():
            fig.add_trace(go.Scatter(
                x=[props['cost_per_wp']],
                y=[props['efficiency_range'][1]],
                mode='markers+text',
                marker=dict(
                    size=props['recyclability'] / 2,  # Size by recyclability
                    color=props.get('color', '#3498DB'),
                    opacity=0.7
                ),
                text=[props['name']],
                textposition='top center',
                name=props['name'],
                hovertemplate=(
                    f"<b>{props['name']}</b><br>"
                    f"Efficiency: {props['efficiency_range'][1]}%<br>"
                    f"Cost: ${props['cost_per_wp']}/Wp<br>"
                    f"Recyclability: {props['recyclability']}%<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title="Material Efficiency vs Cost Analysis",
            xaxis_title="Cost ($/Wp)",
            yaxis_title="Peak Efficiency (%)",
            showlegend=False,
            hovermode='closest',
            height=500,
            template='plotly_white'
        )

        return fig


def render_materials_database():
    """Render materials database interface in Streamlit."""
    st.header("🔬 Materials Engineering Database")
    st.markdown("Comprehensive database of PV materials with properties and performance metrics.")

    db = MaterialsDatabase()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Materials Overview",
        "🔍 Material Comparison",
        "📈 Performance Analysis",
        "🎯 Material Selection"
    ])

    with tab1:
        st.subheader("All Materials in Database")

        # Display all materials
        df = db.get_all_materials()
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Materials",
                len(db.materials),
                help="Number of materials in database"
            )

        with col2:
            avg_eff = df['Efficiency Max (%)'].mean()
            st.metric(
                "Avg Max Efficiency",
                f"{avg_eff:.1f}%",
                help="Average maximum efficiency"
            )

        with col3:
            avg_cost = df['Cost ($/Wp)'].mean()
            st.metric(
                "Avg Cost",
                f"${avg_cost:.2f}/Wp",
                help="Average cost per Watt-peak"
            )

        with col4:
            avg_recycle = df['Recyclability (%)'].mean()
            st.metric(
                "Avg Recyclability",
                f"{avg_recycle:.0f}%",
                help="Average recyclability"
            )

    with tab2:
        st.subheader("Material Comparison")

        # Material selection
        selected_materials = st.multiselect(
            "Select materials to compare:",
            options=db.material_names,
            default=["c-Si", "perovskite", "CIGS"]
        )

        if len(selected_materials) >= 2:
            # Comparison table
            comparison_df = db.compare_materials(selected_materials)
            st.dataframe(comparison_df, use_container_width=True)

            # Radar chart
            st.plotly_chart(
                db.create_comparison_radar(selected_materials),
                use_container_width=True
            )

            # Detailed metrics
            st.subheader("Detailed Material Scores")
            cols = st.columns(len(selected_materials))

            for idx, mat_id in enumerate(selected_materials):
                with cols[idx]:
                    score = db.calculate_material_score(mat_id)
                    mat_name = db.materials[mat_id]['name']
                    st.metric(
                        mat_name,
                        f"{score:.1f}/100",
                        help="Overall material score"
                    )
        else:
            st.info("Please select at least 2 materials to compare.")

    with tab3:
        st.subheader("Performance Analysis")

        # Efficiency vs Cost scatter
        st.plotly_chart(
            db.create_efficiency_cost_scatter(),
            use_container_width=True
        )

        # Top materials by criterion
        st.subheader("Top Materials by Criterion")

        col1, col2 = st.columns(2)

        with col1:
            criterion = st.selectbox(
                "Select criterion:",
                ["efficiency", "cost", "recyclability", "degradation"]
            )

        with col2:
            top_n = st.slider("Number of materials:", 3, 10, 5)

        top_materials = db.get_best_materials(criterion, top_n)

        # Display results
        for rank, (mat_id, value) in enumerate(top_materials, 1):
            mat = db.materials[mat_id]
            st.write(f"**{rank}. {mat['name']}**")

            if criterion == 'efficiency':
                st.caption(f"Max Efficiency: {mat['efficiency_range'][1]}%")
            elif criterion == 'cost':
                st.caption(f"Cost: ${mat['cost_per_wp']}/Wp")
            elif criterion == 'recyclability':
                st.caption(f"Recyclability: {mat['recyclability']}%")
            elif criterion == 'degradation':
                st.caption(f"Degradation: {mat['degradation_rate']}%/year")

    with tab4:
        st.subheader("🎯 Material Selection Tool")
        st.markdown("Configure priorities to find the best material for your application.")

        # Custom weights
        st.write("**Set Priority Weights:**")

        col1, col2 = st.columns(2)

        with col1:
            w_efficiency = st.slider("Efficiency Priority", 0, 100, 35) / 100
            w_cost = st.slider("Cost Priority", 0, 100, 25) / 100

        with col2:
            w_degradation = st.slider("Stability Priority", 0, 100, 20) / 100
            w_recyclability = st.slider("Recyclability Priority", 0, 100, 20) / 100

        weights = {
            'efficiency': w_efficiency,
            'cost': w_cost,
            'degradation': w_degradation,
            'recyclability': w_recyclability
        }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate scores for all materials
        material_scores = [
            (mat_id, db.calculate_material_score(mat_id, weights))
            for mat_id in db.material_names
        ]
        material_scores.sort(key=lambda x: x[1], reverse=True)

        # Display recommendations
        st.subheader("📋 Recommended Materials")

        for rank, (mat_id, score) in enumerate(material_scores[:5], 1):
            mat = db.materials[mat_id]

            with st.expander(f"**{rank}. {mat['name']}** - Score: {score:.1f}/100"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Efficiency", f"{mat['efficiency_range'][1]}%")
                    st.metric("Cost", f"${mat['cost_per_wp']}/Wp")

                with col2:
                    st.metric("Degradation", f"{mat['degradation_rate']}%/year")
                    st.metric("Recyclability", f"{mat['recyclability']}%")

                with col3:
                    st.metric("Bandgap", f"{mat['bandgap']} eV")
                    st.metric("Type", mat['type'])

                st.caption(f"**Thermal Conductivity:** {mat['thermal_conductivity']} W/(m·K)")
                st.caption(f"**Temperature Coefficient:** {mat['temp_coefficient']}%/°C")

    # Footer
    st.divider()
    st.info("💡 **Materials Database** - Branch B01 | 5 Sessions Integrated")
