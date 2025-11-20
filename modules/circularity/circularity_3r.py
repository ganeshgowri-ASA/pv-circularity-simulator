"""
Circularity Assessment Module - 3R Framework (Branch B11).

Features:
- Reuse potential assessment (capacity testing, physical inspection)
- Repair feasibility analysis (component replacement, cost estimation)
- Recycling value calculation (material recovery, revenue estimation)
- Circular economy scoring (weighted 3R metrics)
- Material composition and recovery rates
- End-of-life pathway optimization
- Circular business model analysis
- Environmental impact assessment
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from utils.constants import (
    CIRCULARITY_METRICS,
    MODULE_COMPOSITION,
    MATERIAL_PROPERTIES
)
from utils.validators import CircularityAssessment
from utils.helpers import calculate_npv


class CircularityAnalyzer:
    """Comprehensive circular economy assessment using 3R framework."""

    def __init__(self):
        """Initialize circularity analyzer."""
        self.reuse_criteria = {
            'capacity_threshold': 80,  # % of nameplate
            'age_threshold': 15,  # years
            'physical_condition': ['excellent', 'good']
        }
        self.repair_cost_range = (50, 200)  # $ per module
        self.recycling_cost_per_kg = 0.30

    def assess_reuse_potential(
        self,
        remaining_capacity: float,
        age: float,
        physical_condition: str,
        visual_defects: List[str],
        electrical_test_passed: bool
    ) -> Dict[str, any]:
        """
        Assess module reuse potential.

        Args:
            remaining_capacity: Remaining capacity (% of nameplate)
            age: Module age (years)
            physical_condition: Physical condition rating
            visual_defects: List of visual defects
            electrical_test_passed: Electrical testing result

        Returns:
            Reuse assessment results
        """
        # Calculate reuse score components
        capacity_score = min(100, (remaining_capacity / 80) * 100) if remaining_capacity > 0 else 0
        age_score = max(0, (1 - age / 25) * 100)

        condition_scores = {
            'excellent': 100,
            'good': 75,
            'fair': 50,
            'poor': 25
        }
        condition_score = condition_scores.get(physical_condition, 0)

        defect_penalty = len(visual_defects) * 5
        electrical_bonus = 20 if electrical_test_passed else 0

        # Overall reuse score (weighted average)
        reuse_score = (
            capacity_score * 0.40 +
            age_score * 0.25 +
            condition_score * 0.25 +
            electrical_bonus * 0.10
        ) - defect_penalty

        reuse_score = max(0, min(100, reuse_score))

        # Determine reuse category
        if reuse_score >= 80:
            reuse_category = 'prime_reuse'
            applications = ['Grid-connected systems', 'Commercial installations', 'Utility-scale projects']
            value_retention = 0.65
        elif reuse_score >= 60:
            reuse_category = 'secondary_reuse'
            applications = ['Off-grid systems', 'Residential installations', 'Backup power']
            value_retention = 0.45
        elif reuse_score >= 40:
            reuse_category = 'tertiary_reuse'
            applications = ['Low-power applications', 'DIY projects', 'Educational purposes']
            value_retention = 0.25
        else:
            reuse_category = 'not_suitable'
            applications = ['Not suitable for reuse']
            value_retention = 0.0

        return {
            'reuse_score': reuse_score,
            'capacity_score': capacity_score,
            'age_score': age_score,
            'condition_score': condition_score,
            'defect_penalty': defect_penalty,
            'electrical_bonus': electrical_bonus,
            'reuse_category': reuse_category,
            'suitable_applications': applications,
            'value_retention': value_retention,
            'is_reusable': reuse_score >= 40
        }

    def assess_repair_feasibility(
        self,
        defect_types: List[str],
        module_value: float,
        labor_cost_per_hour: float = 75,
        parts_markup: float = 1.3
    ) -> Dict[str, any]:
        """
        Assess module repair feasibility and cost.

        Args:
            defect_types: List of defect types
            module_value: Current module value ($)
            labor_cost_per_hour: Labor cost ($/hour)
            parts_markup: Parts cost markup multiplier

        Returns:
            Repair feasibility analysis
        """
        # Repair database
        repair_catalog = {
            'bypass_diode_failure': {'labor_hours': 0.5, 'parts_cost': 15, 'success_rate': 0.95, 'life_extension': 5},
            'junction_box_damage': {'labor_hours': 0.75, 'parts_cost': 25, 'success_rate': 0.90, 'life_extension': 7},
            'cable_damage': {'labor_hours': 0.5, 'parts_cost': 20, 'success_rate': 0.98, 'life_extension': 8},
            'connector_corrosion': {'labor_hours': 0.25, 'parts_cost': 10, 'success_rate': 0.95, 'life_extension': 5},
            'frame_damage': {'labor_hours': 1.5, 'parts_cost': 50, 'success_rate': 0.80, 'life_extension': 3},
            'glass_crack': {'labor_hours': 3.0, 'parts_cost': 150, 'success_rate': 0.60, 'life_extension': 4},
            'backsheet_damage': {'labor_hours': 2.0, 'parts_cost': 80, 'success_rate': 0.70, 'life_extension': 4},
            'cell_interconnect': {'labor_hours': 2.5, 'parts_cost': 100, 'success_rate': 0.65, 'life_extension': 3}
        }

        total_labor_hours = 0
        total_parts_cost = 0
        repair_actions = []
        total_life_extension = 0
        overall_success_rate = 1.0

        for defect in defect_types:
            if defect in repair_catalog:
                repair = repair_catalog[defect]
                total_labor_hours += repair['labor_hours']
                total_parts_cost += repair['parts_cost'] * parts_markup
                repair_actions.append(defect.replace('_', ' ').title())
                total_life_extension = max(total_life_extension, repair['life_extension'])
                overall_success_rate *= repair['success_rate']

        # Calculate total cost
        labor_cost = total_labor_hours * labor_cost_per_hour
        total_repair_cost = labor_cost + total_parts_cost

        # Feasibility determination
        cost_benefit_ratio = total_repair_cost / module_value if module_value > 0 else 999
        is_feasible = (cost_benefit_ratio < 0.5) and (overall_success_rate > 0.6)

        # Economic analysis
        if is_feasible:
            repaired_value = module_value * 0.8  # Repaired modules worth 80% of intact used modules
            net_benefit = repaired_value - total_repair_cost
            roi = (net_benefit / total_repair_cost * 100) if total_repair_cost > 0 else 0
        else:
            repaired_value = 0
            net_benefit = -total_repair_cost
            roi = -100

        return {
            'is_feasible': is_feasible,
            'total_repair_cost': total_repair_cost,
            'labor_cost': labor_cost,
            'parts_cost': total_parts_cost,
            'cost_benefit_ratio': cost_benefit_ratio,
            'repair_actions': repair_actions,
            'estimated_life_extension': total_life_extension,
            'success_rate': overall_success_rate,
            'repaired_value': repaired_value,
            'net_benefit': net_benefit,
            'roi': roi,
            'recommendation': 'Repair recommended' if is_feasible else 'Replacement or recycling recommended'
        }

    def calculate_recycling_value(
        self,
        module_weight_kg: float,
        module_type: str = 'c-Si',
        quantity: int = 1
    ) -> Dict[str, any]:
        """
        Calculate recycling value and material recovery.

        Args:
            module_weight_kg: Module weight (kg)
            module_type: Module technology type
            quantity: Number of modules

        Returns:
            Recycling value analysis
        """
        # Material recovery database (kg per kg of module)
        recovery_rates = {
            'glass': {'rate': 0.95, 'price_per_kg': 0.05},
            'aluminum': {'rate': 0.95, 'price_per_kg': 1.80},
            'silicon': {'rate': 0.85, 'price_per_kg': 2.50},
            'copper': {'rate': 0.90, 'price_per_kg': 6.50},
            'silver': {'rate': 0.80, 'price_per_kg': 650.00},
            'eva_polymer': {'rate': 0.70, 'price_per_kg': 0.80},
            'backsheet': {'rate': 0.60, 'price_per_kg': 0.30}
        }

        total_weight = module_weight_kg * quantity

        # Calculate material composition
        materials_recovered = {}
        total_revenue = 0
        total_recyclable_weight = 0

        for material, composition_fraction in MODULE_COMPOSITION.items():
            if material in recovery_rates:
                recovery = recovery_rates[material]
                material_weight = total_weight * composition_fraction
                recovered_weight = material_weight * recovery['rate']
                revenue = recovered_weight * recovery['price_per_kg']

                materials_recovered[material] = {
                    'total_weight_kg': material_weight,
                    'recovered_weight_kg': recovered_weight,
                    'recovery_rate': recovery['rate'],
                    'revenue': revenue
                }

                total_revenue += revenue
                total_recyclable_weight += recovered_weight

        # Recycling costs
        recycling_cost = total_weight * self.recycling_cost_per_kg
        transportation_cost = total_weight * 0.10  # $0.10 per kg for transport
        total_cost = recycling_cost + transportation_cost

        # Net value
        net_revenue = total_revenue - total_cost

        # Environmental impact
        co2_avoided = total_recyclable_weight * 1.5  # kg CO2 per kg material recycled
        energy_saved = total_recyclable_weight * 15  # MJ per kg

        return {
            'quantity': quantity,
            'total_weight_kg': total_weight,
            'recyclable_weight_kg': total_recyclable_weight,
            'recovery_rate_pct': (total_recyclable_weight / total_weight * 100),
            'materials_recovered': materials_recovered,
            'gross_revenue': total_revenue,
            'recycling_cost': recycling_cost,
            'transportation_cost': transportation_cost,
            'total_cost': total_cost,
            'net_revenue': net_revenue,
            'revenue_per_module': net_revenue / quantity,
            'co2_avoided_kg': co2_avoided,
            'energy_saved_mj': energy_saved,
            'environmental_benefit_value': co2_avoided * 0.05  # $0.05 per kg CO2
        }

    def calculate_circularity_score(
        self,
        reuse_assessment: Dict,
        repair_assessment: Dict,
        recycling_assessment: Dict,
        weights: Dict[str, float] = None
    ) -> Dict[str, any]:
        """
        Calculate overall circularity score.

        Args:
            reuse_assessment: Reuse assessment results
            repair_assessment: Repair assessment results
            recycling_assessment: Recycling assessment results
            weights: Custom weights for 3R components

        Returns:
            Overall circularity score and analysis
        """
        if weights is None:
            weights = {'reuse': 0.45, 'repair': 0.35, 'recycle': 0.20}

        # Normalize scores to 0-100
        reuse_score = reuse_assessment['reuse_score']

        # Repair score based on feasibility and ROI
        if repair_assessment['is_feasible']:
            repair_score = min(100, 50 + repair_assessment['roi'] / 2)
        else:
            repair_score = 20  # Base score for having repair option

        # Recycle score based on recovery rate and net value
        recovery_rate = recycling_assessment['recovery_rate_pct']
        recycle_score = min(100, recovery_rate)

        # Weighted overall score
        overall_score = (
            reuse_score * weights['reuse'] +
            repair_score * weights['repair'] +
            recycle_score * weights['recycle']
        )

        # Determine circularity grade
        if overall_score >= 80:
            grade = 'A'
            description = 'Excellent circularity - High reuse and recovery potential'
        elif overall_score >= 70:
            grade = 'B'
            description = 'Good circularity - Strong secondary market value'
        elif overall_score >= 60:
            grade = 'C'
            description = 'Fair circularity - Moderate recovery options'
        elif overall_score >= 50:
            grade = 'D'
            description = 'Limited circularity - Primarily recycling pathway'
        else:
            grade = 'F'
            description = 'Poor circularity - Challenging end-of-life management'

        # Recommended pathway
        pathway_scores = {
            'Reuse': reuse_score,
            'Repair then Reuse': (reuse_score * 0.7 + repair_score * 0.3),
            'Recycle': recycle_score
        }
        optimal_pathway = max(pathway_scores, key=pathway_scores.get)

        return {
            'overall_score': overall_score,
            'reuse_score': reuse_score,
            'repair_score': repair_score,
            'recycle_score': recycle_score,
            'circularity_grade': grade,
            'description': description,
            'optimal_pathway': optimal_pathway,
            'pathway_scores': pathway_scores,
            'weights_used': weights
        }

    def analyze_circular_business_models(
        self,
        module_count: int,
        avg_module_value: float,
        reuse_potential: float,
        repair_feasibility: bool
    ) -> Dict[str, Dict]:
        """
        Analyze circular business model opportunities.

        Args:
            module_count: Number of modules
            avg_module_value: Average module value ($)
            reuse_potential: Reuse score (0-100)
            repair_feasibility: Whether repair is feasible

        Returns:
            Business model analysis
        """
        business_models = {}

        # Model 1: Resale Market
        if reuse_potential >= 60:
            resale_price = avg_module_value * (reuse_potential / 100) * 0.6
            total_revenue = resale_price * module_count
            operating_costs = total_revenue * 0.15  # 15% for testing, cleaning, logistics
            net_revenue = total_revenue - operating_costs

            business_models['resale'] = {
                'name': 'Secondary Market Resale',
                'description': 'Sell tested modules in secondary markets',
                'price_per_module': resale_price,
                'total_revenue': total_revenue,
                'operating_costs': operating_costs,
                'net_revenue': net_revenue,
                'margin_pct': (net_revenue / total_revenue * 100) if total_revenue > 0 else 0,
                'viability': 'high' if reuse_potential >= 70 else 'medium'
            }

        # Model 2: Refurbishment Service
        if repair_feasibility:
            refurb_price = avg_module_value * 0.7
            refurb_cost = avg_module_value * 0.25  # Repair + certification
            total_revenue = refurb_price * module_count
            total_costs = refurb_cost * module_count
            net_revenue = total_revenue - total_costs

            business_models['refurbishment'] = {
                'name': 'Refurbishment & Certification',
                'description': 'Repair, certify, and resell with warranty',
                'price_per_module': refurb_price,
                'total_revenue': total_revenue,
                'operating_costs': total_costs,
                'net_revenue': net_revenue,
                'margin_pct': (net_revenue / total_revenue * 100) if total_revenue > 0 else 0,
                'viability': 'high'
            }

        # Model 3: Material Recovery
        recycling_value_per_module = 15  # Average net recycling value
        total_revenue = recycling_value_per_module * module_count
        processing_costs = total_revenue * 0.40
        net_revenue = total_revenue - processing_costs

        business_models['material_recovery'] = {
            'name': 'Material Recovery & Recycling',
            'description': 'Extract and sell recovered materials',
            'price_per_module': recycling_value_per_module,
            'total_revenue': total_revenue,
            'operating_costs': processing_costs,
            'net_revenue': net_revenue,
            'margin_pct': (net_revenue / total_revenue * 100) if total_revenue > 0 else 0,
            'viability': 'medium'
        }

        # Model 4: Component Harvesting
        if reuse_potential < 60:
            component_value = avg_module_value * 0.40  # Value of salvageable components
            total_revenue = component_value * module_count
            labor_costs = total_revenue * 0.25
            net_revenue = total_revenue - labor_costs

            business_models['component_harvesting'] = {
                'name': 'Component Harvesting',
                'description': 'Extract valuable components (junction boxes, cables, frames)',
                'price_per_module': component_value,
                'total_revenue': total_revenue,
                'operating_costs': labor_costs,
                'net_revenue': net_revenue,
                'margin_pct': (net_revenue / total_revenue * 100) if total_revenue > 0 else 0,
                'viability': 'medium'
            }

        return business_models


def render_circularity_3r():
    """Render circularity assessment interface in Streamlit."""
    st.header("♻️ Circularity Assessment (3R Framework)")
    st.markdown("Comprehensive circular economy analysis: Reuse, Repair, Recycle.")

    analyzer = CircularityAnalyzer()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Reuse Assessment",
        "🔧 Repair Analysis",
        "♻️ Recycling Value",
        "📊 Circularity Score",
        "💼 Business Models"
    ])

    with tab1:
        st.subheader("Reuse Potential Assessment")

        col1, col2 = st.columns(2)

        with col1:
            remaining_capacity = st.slider("Remaining Capacity (% of nameplate):", 0, 100, 85, 1)
            age = st.number_input("Module Age (years):", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
            physical_condition = st.selectbox("Physical Condition:", ['excellent', 'good', 'fair', 'poor'])

        with col2:
            visual_defects = st.multiselect(
                "Visual Defects:",
                ['Discoloration', 'Delamination', 'Cell cracks', 'Frame corrosion',
                 'Junction box damage', 'Backsheet damage', 'Glass cracks', 'None']
            )
            electrical_test_passed = st.checkbox("Electrical Testing Passed", value=True)

        if st.button("🔍 Assess Reuse Potential", key="assess_reuse"):
            with st.spinner("Analyzing reuse potential..."):
                reuse_assessment = analyzer.assess_reuse_potential(
                    remaining_capacity,
                    age,
                    physical_condition,
                    visual_defects if 'None' not in visual_defects else [],
                    electrical_test_passed
                )

            st.session_state['reuse_assessment'] = reuse_assessment

            st.success(f"✅ Reuse assessment completed - Score: {reuse_assessment['reuse_score']:.1f}/100")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Overall Reuse Score", f"{reuse_assessment['reuse_score']:.1f}/100")
                st.metric("Capacity Score", f"{reuse_assessment['capacity_score']:.1f}/100")

            with col2:
                st.metric("Age Score", f"{reuse_assessment['age_score']:.1f}/100")
                st.metric("Condition Score", f"{reuse_assessment['condition_score']:.1f}/100")

            with col3:
                category_icon = '🟢' if reuse_assessment['is_reusable'] else '🔴'
                st.metric("Reuse Category", f"{category_icon} {reuse_assessment['reuse_category'].replace('_', ' ').title()}")
                st.metric("Value Retention", f"{reuse_assessment['value_retention']*100:.0f}%")

            # Score breakdown
            st.subheader("Score Breakdown")

            score_components = {
                'Capacity': reuse_assessment['capacity_score'],
                'Age': reuse_assessment['age_score'],
                'Condition': reuse_assessment['condition_score'],
                'Electrical Bonus': reuse_assessment['electrical_bonus']
            }

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=list(score_components.keys()),
                y=list(score_components.values()),
                marker_color=['#3498DB', '#2ECC71', '#F39C12', '#9B59B6'],
                text=[f"{v:.1f}" for v in score_components.values()],
                textposition='auto'
            ))

            if reuse_assessment['defect_penalty'] > 0:
                fig.add_trace(go.Bar(
                    x=['Defect Penalty'],
                    y=[-reuse_assessment['defect_penalty']],
                    marker_color='#E74C3C',
                    text=[f"-{reuse_assessment['defect_penalty']:.1f}"],
                    textposition='auto'
                ))

            fig.update_layout(
                title="Reuse Score Components",
                xaxis_title="Component",
                yaxis_title="Score",
                height=400,
                showlegend=False,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Suitable applications
            st.subheader("Suitable Applications")

            for app in reuse_assessment['suitable_applications']:
                st.write(f"✓ {app}")

            # Reuse potential gauge
            st.subheader("Reuse Potential Indicator")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reuse_assessment['reuse_score'],
                title={'text': "Reuse Potential Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ECC71"},
                    'steps': [
                        {'range': [0, 40], 'color': "#E74C3C"},
                        {'range': [40, 60], 'color': "#F39C12"},
                        {'range': [60, 80], 'color': "#F4D03F"},
                        {'range': [80, 100], 'color': "#2ECC71"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))

            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Repair Feasibility Analysis")

        col1, col2 = st.columns(2)

        with col1:
            defect_types = st.multiselect(
                "Identified Defects:",
                ['bypass_diode_failure', 'junction_box_damage', 'cable_damage',
                 'connector_corrosion', 'frame_damage', 'glass_crack',
                 'backsheet_damage', 'cell_interconnect']
            )
            module_value = st.number_input("Current Module Value ($):", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)

        with col2:
            labor_cost = st.number_input("Labor Cost ($/hour):", min_value=0.0, max_value=200.0, value=75.0, step=5.0)
            parts_markup = st.slider("Parts Cost Markup:", 1.0, 2.0, 1.3, 0.1)

        if st.button("🔧 Analyze Repair Feasibility", key="analyze_repair"):
            with st.spinner("Analyzing repair options..."):
                repair_assessment = analyzer.assess_repair_feasibility(
                    defect_types,
                    module_value,
                    labor_cost,
                    parts_markup
                )

            st.session_state['repair_assessment'] = repair_assessment

            feasibility_icon = '✅' if repair_assessment['is_feasible'] else '❌'
            st.success(f"{feasibility_icon} Repair analysis completed")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Repair Cost", f"${repair_assessment['total_repair_cost']:.2f}")
                st.metric("Labor Cost", f"${repair_assessment['labor_cost']:.2f}")

            with col2:
                st.metric("Parts Cost", f"${repair_assessment['parts_cost']:.2f}")
                st.metric("Cost/Benefit Ratio", f"{repair_assessment['cost_benefit_ratio']:.2f}")

            with col3:
                st.metric("Success Rate", f"{repair_assessment['success_rate']*100:.1f}%")
                st.metric("Life Extension", f"{repair_assessment['estimated_life_extension']} years")

            with col4:
                st.metric("Repaired Value", f"${repair_assessment['repaired_value']:.2f}")
                st.metric("ROI", f"{repair_assessment['roi']:.1f}%")

            # Recommendation
            if repair_assessment['is_feasible']:
                st.success(f"✅ {repair_assessment['recommendation']}")
            else:
                st.error(f"❌ {repair_assessment['recommendation']}")

            # Repair actions
            if repair_assessment['repair_actions']:
                st.subheader("Required Repair Actions")

                for action in repair_assessment['repair_actions']:
                    st.write(f"• {action}")

            # Cost breakdown
            st.subheader("Cost Breakdown")

            fig = go.Figure()

            costs = {
                'Labor': repair_assessment['labor_cost'],
                'Parts': repair_assessment['parts_cost']
            }

            fig.add_trace(go.Pie(
                labels=list(costs.keys()),
                values=list(costs.values()),
                marker=dict(colors=['#3498DB', '#2ECC71'])
            ))

            fig.update_layout(
                title="Repair Cost Distribution",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Economic analysis
            st.subheader("Economic Analysis")

            metrics = ['Module Value', 'Repair Cost', 'Repaired Value', 'Net Benefit']
            values = [
                module_value,
                repair_assessment['total_repair_cost'],
                repair_assessment['repaired_value'],
                repair_assessment['net_benefit']
            ]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                marker_color=['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6' if repair_assessment['net_benefit'] > 0 else '#E74C3C'],
                text=[f"${v:.2f}" for v in values],
                textposition='auto'
            ))

            fig.update_layout(
                title="Repair Economics",
                xaxis_title="Metric",
                yaxis_title="Value ($)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Recycling Value Calculation")

        col1, col2 = st.columns(2)

        with col1:
            module_weight = st.number_input("Module Weight (kg):", min_value=1.0, max_value=50.0, value=18.5, step=0.5)
            module_type = st.selectbox("Module Type:", list(MATERIAL_PROPERTIES.keys()))

        with col2:
            quantity = st.number_input("Number of Modules:", min_value=1, max_value=100000, value=100, step=10)

        if st.button("♻️ Calculate Recycling Value", key="calc_recycling"):
            with st.spinner("Calculating recycling value..."):
                recycling_assessment = analyzer.calculate_recycling_value(
                    module_weight,
                    module_type,
                    quantity
                )

            st.session_state['recycling_assessment'] = recycling_assessment

            st.success(f"✅ Recycling analysis completed - Net Revenue: ${recycling_assessment['net_revenue']:,.2f}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Weight", f"{recycling_assessment['total_weight_kg']:.1f} kg")
                st.metric("Recyclable Weight", f"{recycling_assessment['recyclable_weight_kg']:.1f} kg")

            with col2:
                st.metric("Recovery Rate", f"{recycling_assessment['recovery_rate_pct']:.1f}%")
                st.metric("Gross Revenue", f"${recycling_assessment['gross_revenue']:,.2f}")

            with col3:
                st.metric("Total Costs", f"${recycling_assessment['total_cost']:,.2f}")
                st.metric("Net Revenue", f"${recycling_assessment['net_revenue']:,.2f}")

            with col4:
                st.metric("Revenue/Module", f"${recycling_assessment['revenue_per_module']:.2f}")
                st.metric("CO2 Avoided", f"{recycling_assessment['co2_avoided_kg']:.1f} kg")

            # Material recovery breakdown
            st.subheader("Material Recovery Breakdown")

            materials = list(recycling_assessment['materials_recovered'].keys())
            recovered_weights = [recycling_assessment['materials_recovered'][m]['recovered_weight_kg'] for m in materials]
            revenues = [recycling_assessment['materials_recovered'][m]['revenue'] for m in materials]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Recovered Weight by Material', 'Revenue by Material'),
                specs=[[{'type': 'pie'}, {'type': 'pie'}]]
            )

            fig.add_trace(
                go.Pie(labels=materials, values=recovered_weights, name='Weight'),
                row=1, col=1
            )

            fig.add_trace(
                go.Pie(labels=materials, values=revenues, name='Revenue'),
                row=1, col=2
            )

            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # Detailed material table
            st.subheader("Detailed Material Analysis")

            material_data = []
            for material, data in recycling_assessment['materials_recovered'].items():
                material_data.append({
                    'Material': material.replace('_', ' ').title(),
                    'Total Weight (kg)': f"{data['total_weight_kg']:.2f}",
                    'Recovered Weight (kg)': f"{data['recovered_weight_kg']:.2f}",
                    'Recovery Rate': f"{data['recovery_rate']*100:.1f}%",
                    'Revenue': f"${data['revenue']:.2f}"
                })

            st.dataframe(pd.DataFrame(material_data), use_container_width=True)

            # Environmental impact
            st.subheader("Environmental Impact")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("CO2 Avoided", f"{recycling_assessment['co2_avoided_kg']:,.0f} kg")

            with col2:
                st.metric("Energy Saved", f"{recycling_assessment['energy_saved_mj']:,.0f} MJ")

            with col3:
                st.metric("Environmental Benefit", f"${recycling_assessment['environmental_benefit_value']:.2f}")

            # Cost breakdown
            st.subheader("Cost Analysis")

            fig = go.Figure()

            categories = ['Gross Revenue', 'Recycling Cost', 'Transport Cost', 'Net Revenue']
            values = [
                recycling_assessment['gross_revenue'],
                -recycling_assessment['recycling_cost'],
                -recycling_assessment['transportation_cost'],
                recycling_assessment['net_revenue']
            ]

            fig.add_trace(go.Waterfall(
                x=categories,
                y=values,
                text=[f"${abs(v):,.2f}" for v in values],
                textposition='outside',
                connector={'line': {'color': 'rgb(63, 63, 63)'}},
                decreasing={'marker': {'color': '#E74C3C'}},
                increasing={'marker': {'color': '#2ECC71'}},
                totals={'marker': {'color': '#3498DB'}}
            ))

            fig.update_layout(
                title="Recycling Value Waterfall",
                yaxis_title="Value ($)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Overall Circularity Score")

        if 'reuse_assessment' not in st.session_state or 'repair_assessment' not in st.session_state or 'recycling_assessment' not in st.session_state:
            st.warning("⚠️ Please complete assessments in all three tabs first (Reuse, Repair, Recycling)")
        else:
            st.write("### Customize 3R Weights")

            col1, col2, col3 = st.columns(3)

            with col1:
                reuse_weight = st.slider("Reuse Weight:", 0.0, 1.0, 0.45, 0.05)

            with col2:
                repair_weight = st.slider("Repair Weight:", 0.0, 1.0, 0.35, 0.05)

            with col3:
                recycle_weight = st.slider("Recycle Weight:", 0.0, 1.0, 0.20, 0.05)

            # Normalize weights
            total_weight = reuse_weight + repair_weight + recycle_weight
            weights = {
                'reuse': reuse_weight / total_weight,
                'repair': repair_weight / total_weight,
                'recycle': recycle_weight / total_weight
            }

            if st.button("📊 Calculate Circularity Score", key="calc_circularity"):
                circularity_score = analyzer.calculate_circularity_score(
                    st.session_state['reuse_assessment'],
                    st.session_state['repair_assessment'],
                    st.session_state['recycling_assessment'],
                    weights
                )

                st.session_state['circularity_score'] = circularity_score

                st.success(f"✅ Circularity Score: {circularity_score['overall_score']:.1f}/100 (Grade: {circularity_score['circularity_grade']})")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Overall Score", f"{circularity_score['overall_score']:.1f}/100")
                    st.metric("Circularity Grade", circularity_score['circularity_grade'])

                with col2:
                    st.metric("Reuse Score", f"{circularity_score['reuse_score']:.1f}/100")

                with col3:
                    st.metric("Repair Score", f"{circularity_score['repair_score']:.1f}/100")

                with col4:
                    st.metric("Recycle Score", f"{circularity_score['recycle_score']:.1f}/100")

                st.info(f"📝 {circularity_score['description']}")
                st.success(f"🎯 **Optimal Pathway:** {circularity_score['optimal_pathway']}")

                # 3R Score comparison
                st.subheader("3R Component Scores")

                fig = go.Figure()

                components = ['Reuse', 'Repair', 'Recycle']
                scores = [
                    circularity_score['reuse_score'],
                    circularity_score['repair_score'],
                    circularity_score['recycle_score']
                ]

                fig.add_trace(go.Bar(
                    x=components,
                    y=scores,
                    marker_color=['#2ECC71', '#3498DB', '#F39C12'],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="3R Framework Scores",
                    xaxis_title="Component",
                    yaxis_title="Score (0-100)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Pathway comparison
                st.subheader("Pathway Comparison")

                pathway_labels = list(circularity_score['pathway_scores'].keys())
                pathway_values = list(circularity_score['pathway_scores'].values())

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=pathway_labels,
                    y=pathway_values,
                    marker_color=['#2ECC71', '#3498DB', '#F39C12'],
                    text=[f"{v:.1f}" for v in pathway_values],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="End-of-Life Pathway Scores",
                    xaxis_title="Pathway",
                    yaxis_title="Score (0-100)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Radar chart
                st.subheader("Circularity Performance Radar")

                categories = ['Reuse<br>Potential', 'Repair<br>Feasibility', 'Recycling<br>Value',
                             'Environmental<br>Impact', 'Economic<br>Value']
                values = [
                    circularity_score['reuse_score'],
                    circularity_score['repair_score'],
                    circularity_score['recycle_score'],
                    min(100, st.session_state['recycling_assessment']['recovery_rate_pct']),
                    min(100, (st.session_state['reuse_assessment']['value_retention'] * 100))
                ]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color='#2ECC71',
                    name='Performance'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    title="Multi-Dimensional Circularity Performance",
                    height=500,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Circular Business Models")

        if 'reuse_assessment' not in st.session_state or 'repair_assessment' not in st.session_state:
            st.warning("⚠️ Please complete Reuse and Repair assessments first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                module_count = st.number_input("Number of Modules:", min_value=1, max_value=100000, value=1000, step=100)
                avg_module_value = st.number_input("Average Module Value ($):", min_value=1.0, max_value=1000.0, value=200.0, step=10.0)

            with col2:
                st.write("**Current Assessments:**")
                st.write(f"Reuse Score: {st.session_state['reuse_assessment']['reuse_score']:.1f}/100")
                st.write(f"Repair Feasible: {'Yes' if st.session_state['repair_assessment']['is_feasible'] else 'No'}")

            if st.button("💼 Analyze Business Models", key="analyze_business"):
                business_models = analyzer.analyze_circular_business_models(
                    module_count,
                    avg_module_value,
                    st.session_state['reuse_assessment']['reuse_score'],
                    st.session_state['repair_assessment']['is_feasible']
                )

                st.success(f"✅ Analyzed {len(business_models)} business models")

                # Business model cards
                for model_key, model in business_models.items():
                    with st.expander(f"📊 {model['name']} - {model['viability'].upper()} Viability", expanded=True):
                        st.write(f"**{model['description']}**")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Price/Module", f"${model['price_per_module']:.2f}")

                        with col2:
                            st.metric("Total Revenue", f"${model['total_revenue']:,.0f}")

                        with col3:
                            st.metric("Operating Costs", f"${model['operating_costs']:,.0f}")

                        with col4:
                            st.metric("Net Revenue", f"${model['net_revenue']:,.0f}")

                        st.metric("Profit Margin", f"{model['margin_pct']:.1f}%")

                # Comparison chart
                st.subheader("Business Model Comparison")

                model_names = [m['name'] for m in business_models.values()]
                net_revenues = [m['net_revenue'] for m in business_models.values()]
                margins = [m['margin_pct'] for m in business_models.values()]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Net Revenue by Model', 'Profit Margin by Model')
                )

                fig.add_trace(
                    go.Bar(x=model_names, y=net_revenues, name='Net Revenue',
                           marker_color='#2ECC71'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(x=model_names, y=margins, name='Margin %',
                           marker_color='#3498DB'),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Model", row=1, col=1)
                fig.update_xaxes(title_text="Model", row=1, col=2)
                fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
                fig.update_yaxes(title_text="Margin (%)", row=1, col=2)

                fig.update_layout(height=400, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Recommended model
                best_model = max(business_models.values(), key=lambda x: x['net_revenue'])
                st.success(f"🎯 **Recommended Model:** {best_model['name']} with ${best_model['net_revenue']:,.0f} net revenue")

    st.divider()
    st.info("💡 **Circularity Assessment (3R Framework)** - Branch B11 | Comprehensive End-of-Life Analysis")
