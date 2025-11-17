"""
Revamp & Repower Planning Module (Branch B10).

Features:
- System age assessment and remaining lifetime estimation
- Revamp vs repower strategy selection and comparison
- Component replacement planning (modules, inverters, trackers)
- Performance upgrade analysis with multiple scenarios
- Cost-benefit analysis of different strategies
- Payback period calculations with NPV analysis
- Energy production comparison (existing vs upgraded)
- Implementation roadmap and phased approach planning
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import (
    INVERTER_TYPES,
    MOUNTING_TYPES,
    FINANCIAL_DEFAULTS,
    MATERIAL_PROPERTIES
)
from utils.validators import RevampRepowerPlan, SystemConfiguration
from utils.helpers import (
    calculate_performance_ratio,
    calculate_specific_yield,
    calculate_npv,
    calculate_irr
)


class RevampRepowerPlanner:
    """Comprehensive revamp and repower planning and analysis."""

    def __init__(self):
        """Initialize revamp and repower planner."""
        self.strategies = ["full_repower", "partial_repower", "revamp", "augmentation"]
        self.component_types = ["modules", "inverters", "trackers", "wiring", "transformers"]

    def assess_system_age(
        self,
        installation_year: int,
        nameplate_capacity: float,
        current_capacity: float,
        degradation_rate: float = 0.5
    ) -> Dict[str, any]:
        """
        Assess system age and remaining lifetime.

        Args:
            installation_year: Year of installation
            nameplate_capacity: Original nameplate capacity (kW)
            current_capacity: Current measured capacity (kW)
            degradation_rate: Annual degradation rate (%/year)

        Returns:
            Age assessment results
        """
        current_year = datetime.now().year
        age = current_year - installation_year

        # Calculate actual degradation
        actual_degradation = ((nameplate_capacity - current_capacity) / nameplate_capacity) * 100
        effective_degradation_rate = actual_degradation / age if age > 0 else 0

        # Predict remaining lifetime (until 80% capacity)
        capacity_retained = (current_capacity / nameplate_capacity) * 100
        capacity_to_threshold = capacity_retained - 80.0
        years_to_threshold = capacity_to_threshold / effective_degradation_rate if effective_degradation_rate > 0 else 25

        # Expected end of life
        expected_eol_year = current_year + int(years_to_threshold)

        # Remaining value
        remaining_lifetime_fraction = max(0, years_to_threshold / 25)
        condition_score = min(100, capacity_retained)

        return {
            'age_years': age,
            'capacity_retained_pct': capacity_retained,
            'actual_degradation_pct': actual_degradation,
            'effective_degradation_rate': effective_degradation_rate,
            'years_to_threshold': max(0, years_to_threshold),
            'expected_eol_year': expected_eol_year,
            'remaining_lifetime_fraction': remaining_lifetime_fraction,
            'condition_score': condition_score,
            'intervention_urgency': 'high' if years_to_threshold < 2 else ('medium' if years_to_threshold < 5 else 'low')
        }

    def analyze_component_replacement(
        self,
        component_type: str,
        age_years: float,
        quantity: int,
        unit_cost: float
    ) -> Dict[str, any]:
        """
        Analyze component replacement needs and costs.

        Args:
            component_type: Type of component (modules, inverters, etc.)
            age_years: Component age in years
            quantity: Number of components
            unit_cost: Cost per unit ($)

        Returns:
            Component replacement analysis
        """
        # Expected lifetimes
        expected_lifetimes = {
            'modules': 25,
            'inverters': 15,
            'trackers': 20,
            'wiring': 30,
            'transformers': 25
        }

        expected_life = expected_lifetimes.get(component_type, 20)
        remaining_life = max(0, expected_life - age_years)
        life_consumed_pct = (age_years / expected_life) * 100

        # Replacement priority
        if life_consumed_pct > 90:
            priority = 'critical'
        elif life_consumed_pct > 70:
            priority = 'high'
        elif life_consumed_pct > 50:
            priority = 'medium'
        else:
            priority = 'low'

        # Cost analysis
        total_cost = quantity * unit_cost
        annual_cost_savings = total_cost * 0.03  # Assume 3% O&M savings from new equipment

        return {
            'component_type': component_type,
            'age_years': age_years,
            'expected_life_years': expected_life,
            'remaining_life_years': remaining_life,
            'life_consumed_pct': life_consumed_pct,
            'replacement_priority': priority,
            'quantity': quantity,
            'unit_cost': unit_cost,
            'total_replacement_cost': total_cost,
            'annual_cost_savings': annual_cost_savings
        }

    def compare_strategies(
        self,
        current_capacity: float,
        target_capacity: float,
        module_cost_per_kw: float = 450,
        inverter_cost_per_kw: float = 100,
        bos_cost_per_kw: float = 200,
        installation_cost_per_kw: float = 150
    ) -> Dict[str, Dict[str, any]]:
        """
        Compare different revamp/repower strategies.

        Args:
            current_capacity: Current system capacity (kW)
            target_capacity: Target capacity after upgrade (kW)
            module_cost_per_kw: Module cost ($/kW)
            inverter_cost_per_kw: Inverter cost ($/kW)
            bos_cost_per_kw: Balance of system cost ($/kW)
            installation_cost_per_kw: Installation cost ($/kW)

        Returns:
            Strategy comparison results
        """
        strategies = {}

        # Full Repower - Replace everything
        full_repower_capacity = target_capacity
        full_repower_cost = full_repower_capacity * (
            module_cost_per_kw + inverter_cost_per_kw +
            bos_cost_per_kw + installation_cost_per_kw
        )
        full_repower_gain = ((target_capacity - current_capacity) / current_capacity) * 100

        strategies['full_repower'] = {
            'name': 'Full Repower',
            'description': 'Complete system replacement with new equipment',
            'capacity_added_kw': target_capacity - current_capacity,
            'modules_replaced_pct': 100,
            'inverters_replaced_pct': 100,
            'total_cost': full_repower_cost,
            'cost_per_kw': full_repower_cost / full_repower_capacity,
            'performance_gain_pct': full_repower_gain,
            'implementation_time_months': 6,
            'downtime_days': 45
        }

        # Partial Repower - Replace 50% of modules, all inverters
        partial_capacity_added = (target_capacity - current_capacity) * 0.5
        partial_repower_cost = (
            partial_capacity_added * module_cost_per_kw +
            current_capacity * inverter_cost_per_kw +
            partial_capacity_added * installation_cost_per_kw
        )
        partial_repower_gain = ((current_capacity + partial_capacity_added - current_capacity) / current_capacity) * 100

        strategies['partial_repower'] = {
            'name': 'Partial Repower',
            'description': 'Replace 50% of modules and all inverters',
            'capacity_added_kw': partial_capacity_added,
            'modules_replaced_pct': 50,
            'inverters_replaced_pct': 100,
            'total_cost': partial_repower_cost,
            'cost_per_kw': partial_repower_cost / (current_capacity + partial_capacity_added),
            'performance_gain_pct': partial_repower_gain,
            'implementation_time_months': 4,
            'downtime_days': 30
        }

        # Revamp - Upgrade inverters and optimize existing modules
        revamp_cost = (
            current_capacity * inverter_cost_per_kw +
            current_capacity * 50  # Optimization costs
        )
        revamp_gain = 15  # Typical gain from inverter upgrade

        strategies['revamp'] = {
            'name': 'Revamp',
            'description': 'Upgrade inverters and optimize existing modules',
            'capacity_added_kw': 0,
            'modules_replaced_pct': 0,
            'inverters_replaced_pct': 100,
            'total_cost': revamp_cost,
            'cost_per_kw': revamp_cost / current_capacity,
            'performance_gain_pct': revamp_gain,
            'implementation_time_months': 2,
            'downtime_days': 14
        }

        # Augmentation - Add new capacity alongside existing
        augment_capacity = target_capacity - current_capacity
        augment_cost = augment_capacity * (
            module_cost_per_kw + inverter_cost_per_kw +
            bos_cost_per_kw + installation_cost_per_kw
        )
        augment_gain = ((target_capacity - current_capacity) / current_capacity) * 100

        strategies['augmentation'] = {
            'name': 'Augmentation',
            'description': 'Add new capacity alongside existing system',
            'capacity_added_kw': augment_capacity,
            'modules_replaced_pct': 0,
            'inverters_replaced_pct': 0,
            'total_cost': augment_cost,
            'cost_per_kw': augment_cost / augment_capacity,
            'performance_gain_pct': augment_gain,
            'implementation_time_months': 5,
            'downtime_days': 0
        }

        return strategies

    def calculate_financial_metrics(
        self,
        strategy_cost: float,
        current_production: float,
        upgraded_production: float,
        electricity_price: float = 0.12,
        discount_rate: float = 0.08,
        project_lifetime: int = 15
    ) -> Dict[str, float]:
        """
        Calculate financial metrics for a strategy.

        Args:
            strategy_cost: Total cost of strategy ($)
            current_production: Current annual production (kWh)
            upgraded_production: Upgraded annual production (kWh)
            electricity_price: Electricity price ($/kWh)
            discount_rate: Discount rate (fraction)
            project_lifetime: Analysis period (years)

        Returns:
            Financial metrics
        """
        # Annual revenue increase
        production_increase = upgraded_production - current_production
        annual_revenue = production_increase * electricity_price

        # Simple payback
        simple_payback = strategy_cost / annual_revenue if annual_revenue > 0 else 999

        # NPV calculation
        cash_flows = [-strategy_cost]
        for year in range(1, project_lifetime + 1):
            annual_revenue_escalated = annual_revenue * ((1 + 0.025) ** year)  # 2.5% escalation
            o_and_m = strategy_cost * 0.01  # 1% annual O&M
            net_cash_flow = annual_revenue_escalated - o_and_m
            cash_flows.append(net_cash_flow)

        npv = calculate_npv(cash_flows, discount_rate)

        # IRR calculation
        irr = calculate_irr(cash_flows)

        # LCOE
        total_energy = upgraded_production * project_lifetime
        total_cost = strategy_cost + (strategy_cost * 0.01 * project_lifetime)
        lcoe = total_cost / total_energy if total_energy > 0 else 0

        # ROI
        total_revenue = sum([annual_revenue * ((1 + 0.025) ** year) for year in range(1, project_lifetime + 1)])
        roi = ((total_revenue - strategy_cost) / strategy_cost) * 100 if strategy_cost > 0 else 0

        return {
            'annual_production_increase_kwh': production_increase,
            'annual_revenue_increase': annual_revenue,
            'simple_payback_years': simple_payback,
            'npv': npv,
            'irr': irr,
            'lcoe': lcoe,
            'roi_pct': roi,
            'total_lifetime_savings': total_revenue - strategy_cost
        }

    def create_implementation_roadmap(
        self,
        strategy: str,
        implementation_months: int,
        start_date: datetime
    ) -> pd.DataFrame:
        """
        Create detailed implementation roadmap.

        Args:
            strategy: Selected strategy
            implementation_months: Implementation duration (months)
            start_date: Project start date

        Returns:
            Implementation roadmap DataFrame
        """
        phases = []

        if strategy == 'full_repower':
            phases = [
                {'phase': 'Planning & Design', 'duration_months': 2, 'completion_pct': 10},
                {'phase': 'Procurement', 'duration_months': 1, 'completion_pct': 20},
                {'phase': 'Site Preparation', 'duration_months': 0.5, 'completion_pct': 30},
                {'phase': 'Module Removal', 'duration_months': 0.5, 'completion_pct': 40},
                {'phase': 'Structural Upgrades', 'duration_months': 0.5, 'completion_pct': 50},
                {'phase': 'New Module Installation', 'duration_months': 1, 'completion_pct': 70},
                {'phase': 'Inverter Installation', 'duration_months': 0.5, 'completion_pct': 85},
                {'phase': 'Commissioning & Testing', 'duration_months': 0.5, 'completion_pct': 100}
            ]
        elif strategy == 'partial_repower':
            phases = [
                {'phase': 'Planning & Assessment', 'duration_months': 1, 'completion_pct': 15},
                {'phase': 'Procurement', 'duration_months': 1, 'completion_pct': 30},
                {'phase': 'Module Replacement (50%)', 'duration_months': 1, 'completion_pct': 60},
                {'phase': 'Inverter Replacement', 'duration_months': 0.5, 'completion_pct': 80},
                {'phase': 'System Integration', 'duration_months': 0.5, 'completion_pct': 95},
                {'phase': 'Testing & Handover', 'duration_months': 0.5, 'completion_pct': 100}
            ]
        elif strategy == 'revamp':
            phases = [
                {'phase': 'System Assessment', 'duration_months': 0.5, 'completion_pct': 20},
                {'phase': 'Inverter Procurement', 'duration_months': 0.5, 'completion_pct': 40},
                {'phase': 'Module Testing & Optimization', 'duration_months': 0.5, 'completion_pct': 60},
                {'phase': 'Inverter Installation', 'duration_months': 0.5, 'completion_pct': 85},
                {'phase': 'Commissioning', 'duration_months': 0.5, 'completion_pct': 100}
            ]
        else:  # augmentation
            phases = [
                {'phase': 'Site Survey & Design', 'duration_months': 1.5, 'completion_pct': 15},
                {'phase': 'Equipment Procurement', 'duration_months': 1, 'completion_pct': 25},
                {'phase': 'Site Preparation', 'duration_months': 1, 'completion_pct': 40},
                {'phase': 'New Array Installation', 'duration_months': 1.5, 'completion_pct': 70},
                {'phase': 'Electrical Integration', 'duration_months': 0.5, 'completion_pct': 85},
                {'phase': 'Testing & Optimization', 'duration_months': 0.5, 'completion_pct': 100}
            ]

        # Calculate dates
        current_date = start_date
        roadmap_data = []

        for phase in phases:
            duration_days = int(phase['duration_months'] * 30)
            end_date = current_date + timedelta(days=duration_days)

            roadmap_data.append({
                'Phase': phase['phase'],
                'Start Date': current_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Duration (months)': phase['duration_months'],
                'Completion (%)': phase['completion_pct']
            })

            current_date = end_date

        return pd.DataFrame(roadmap_data)


def render_revamp_repower():
    """Render revamp and repower planning interface in Streamlit."""
    st.header("🔄 Revamp & Repower Planning")
    st.markdown("Strategic planning for system upgrades, replacements, and life extension.")

    planner = RevampRepowerPlanner()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Age Assessment",
        "🔧 Component Analysis",
        "📊 Strategy Comparison",
        "💰 Financial Analysis",
        "🗺️ Implementation Roadmap"
    ])

    with tab1:
        st.subheader("System Age Assessment")

        col1, col2 = st.columns(2)

        with col1:
            installation_year = st.number_input("Installation Year:", min_value=1990, max_value=2024, value=2010)
            nameplate_capacity = st.number_input("Nameplate Capacity (kW):", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)

        with col2:
            current_capacity = st.number_input("Current Measured Capacity (kW):", min_value=1.0, max_value=10000.0, value=92.0, step=1.0)
            degradation_rate = st.slider("Expected Degradation Rate (%/year):", 0.1, 2.0, 0.5, 0.1)

        if st.button("🔍 Assess System Age", key="assess_age"):
            with st.spinner("Analyzing system age..."):
                assessment = planner.assess_system_age(
                    installation_year,
                    nameplate_capacity,
                    current_capacity,
                    degradation_rate
                )

            st.session_state['age_assessment'] = assessment
            st.session_state['current_capacity'] = current_capacity

            st.success("✅ Age assessment completed")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("System Age", f"{assessment['age_years']} years")
                st.metric("Capacity Retained", f"{assessment['capacity_retained_pct']:.1f}%")

            with col2:
                st.metric("Degradation", f"{assessment['actual_degradation_pct']:.1f}%")
                st.metric("Degradation Rate", f"{assessment['effective_degradation_rate']:.2f}%/year")

            with col3:
                st.metric("Years to 80% Threshold", f"{assessment['years_to_threshold']:.1f} years")
                st.metric("Expected EOL", str(assessment['expected_eol_year']))

            with col4:
                st.metric("Condition Score", f"{assessment['condition_score']:.0f}/100")
                urgency_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                urgency_icon = urgency_colors[assessment['intervention_urgency']]
                st.metric("Intervention Urgency", f"{urgency_icon} {assessment['intervention_urgency'].upper()}")

            # Degradation visualization
            st.subheader("Capacity Degradation Over Time")

            years = list(range(installation_year, datetime.now().year + int(assessment['years_to_threshold']) + 5))
            capacity = [nameplate_capacity * (1 - degradation_rate/100) ** (y - installation_year) for y in years]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=years,
                y=capacity,
                mode='lines',
                name='Projected Capacity',
                line=dict(color='#3498DB', width=3)
            ))

            fig.add_hline(
                y=nameplate_capacity * 0.8,
                line_dash="dash",
                line_color="red",
                annotation_text="80% Threshold",
                annotation_position="right"
            )

            fig.add_vline(
                x=datetime.now().year,
                line_dash="dot",
                line_color="green",
                annotation_text="Current Year",
                annotation_position="top"
            )

            fig.update_layout(
                title="Capacity Degradation Projection",
                xaxis_title="Year",
                yaxis_title="Capacity (kW)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Condition gauge
            st.subheader("System Condition")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=assessment['condition_score'],
                title={'text': "Overall Condition Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ECC71"},
                    'steps': [
                        {'range': [0, 50], 'color': "#E74C3C"},
                        {'range': [50, 75], 'color': "#F39C12"},
                        {'range': [75, 100], 'color': "#2ECC71"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))

            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Component Replacement Analysis")

        st.write("### Analyze Individual Components")

        col1, col2 = st.columns(2)

        with col1:
            component_type = st.selectbox("Component Type:", planner.component_types)
            age_years = st.number_input("Component Age (years):", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

        with col2:
            quantity = st.number_input("Quantity:", min_value=1, max_value=100000, value=100, step=10)
            unit_cost = st.number_input("Unit Cost ($):", min_value=1.0, max_value=10000.0, value=500.0, step=50.0)

        if st.button("📋 Analyze Component", key="analyze_component"):
            analysis = planner.analyze_component_replacement(
                component_type,
                age_years,
                quantity,
                unit_cost
            )

            st.session_state['component_analysis'] = analysis

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Expected Life", f"{analysis['expected_life_years']} years")
                st.metric("Remaining Life", f"{analysis['remaining_life_years']:.1f} years")

            with col2:
                st.metric("Life Consumed", f"{analysis['life_consumed_pct']:.1f}%")
                priority_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                priority_icon = priority_colors[analysis['replacement_priority']]
                st.metric("Priority", f"{priority_icon} {analysis['replacement_priority'].upper()}")

            with col3:
                st.metric("Total Cost", f"${analysis['total_replacement_cost']:,.0f}")
                st.metric("Annual Savings", f"${analysis['annual_cost_savings']:,.0f}")

            # Life consumption bar
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Life Consumed', 'Remaining Life'],
                y=[analysis['life_consumed_pct'], 100 - analysis['life_consumed_pct']],
                marker_color=['#E74C3C', '#2ECC71'],
                text=[f"{analysis['life_consumed_pct']:.1f}%", f"{100-analysis['life_consumed_pct']:.1f}%"],
                textposition='auto'
            ))

            fig.update_layout(
                title=f"{component_type.title()} Life Cycle Status",
                yaxis_title="Percentage (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Multi-component analysis
        st.divider()
        st.write("### Multi-Component Analysis")

        if st.button("🔍 Analyze All Components", key="analyze_all"):
            components_data = []

            component_configs = [
                ('modules', 10, 500, 200),
                ('inverters', 8, 20, 5000),
                ('trackers', 10, 50, 3000),
                ('wiring', 10, 1000, 5),
                ('transformers', 10, 2, 50000)
            ]

            for comp_type, age, qty, cost in component_configs:
                analysis = planner.analyze_component_replacement(comp_type, age, qty, cost)
                components_data.append(analysis)

            df = pd.DataFrame(components_data)

            st.dataframe(
                df[['component_type', 'age_years', 'remaining_life_years',
                    'life_consumed_pct', 'replacement_priority', 'total_replacement_cost']],
                use_container_width=True
            )

            # Priority distribution
            fig = go.Figure()

            priority_counts = df['replacement_priority'].value_counts()

            fig.add_trace(go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                marker=dict(colors=['#E74C3C', '#F39C12', '#F39C12', '#2ECC71'])
            ))

            fig.update_layout(
                title="Component Replacement Priority Distribution",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Strategy Comparison")

        col1, col2 = st.columns(2)

        with col1:
            current_cap = st.number_input("Current System Capacity (kW):", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="strat_current")
            target_cap = st.number_input("Target Capacity (kW):", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)

        with col2:
            module_cost = st.number_input("Module Cost ($/kW):", min_value=100, max_value=1000, value=450, step=50)
            inverter_cost = st.number_input("Inverter Cost ($/kW):", min_value=50, max_value=300, value=100, step=10)

        if st.button("🔄 Compare Strategies", key="compare_strat"):
            with st.spinner("Analyzing strategies..."):
                strategies = planner.compare_strategies(
                    current_cap,
                    target_cap,
                    module_cost,
                    inverter_cost
                )

            st.session_state['strategies'] = strategies
            st.session_state['current_cap_strat'] = current_cap

            st.success(f"✅ Analyzed {len(strategies)} strategies")

            # Strategy comparison table
            st.subheader("Strategy Comparison Overview")

            comparison_data = []
            for key, strat in strategies.items():
                comparison_data.append({
                    'Strategy': strat['name'],
                    'Description': strat['description'],
                    'Total Cost': f"${strat['total_cost']:,.0f}",
                    'Cost per kW': f"${strat['cost_per_kw']:.0f}",
                    'Performance Gain': f"{strat['performance_gain_pct']:.1f}%",
                    'Duration': f"{strat['implementation_time_months']} months",
                    'Downtime': f"{strat['downtime_days']} days"
                })

            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

            # Cost comparison chart
            st.subheader("Cost Comparison")

            fig = go.Figure()

            strategy_names = [s['name'] for s in strategies.values()]
            costs = [s['total_cost'] for s in strategies.values()]

            fig.add_trace(go.Bar(
                x=strategy_names,
                y=costs,
                marker_color=['#3498DB', '#2ECC71', '#F39C12', '#9B59B6'],
                text=[f"${c:,.0f}" for c in costs],
                textposition='auto'
            ))

            fig.update_layout(
                title="Total Cost by Strategy",
                xaxis_title="Strategy",
                yaxis_title="Total Cost ($)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Performance gain comparison
            st.subheader("Performance Gain Comparison")

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Performance Gain (%)', 'Cost Efficiency ($/kW)')
            )

            gains = [s['performance_gain_pct'] for s in strategies.values()]
            cost_per_kw = [s['cost_per_kw'] for s in strategies.values()]

            fig.add_trace(
                go.Bar(x=strategy_names, y=gains, name='Performance Gain',
                       marker_color='#2ECC71'),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(x=strategy_names, y=cost_per_kw, name='Cost per kW',
                       marker_color='#E74C3C'),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Strategy", row=1, col=1)
            fig.update_xaxes(title_text="Strategy", row=1, col=2)
            fig.update_yaxes(title_text="Gain (%)", row=1, col=1)
            fig.update_yaxes(title_text="Cost ($/kW)", row=1, col=2)

            fig.update_layout(height=400, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Financial Analysis")

        if 'strategies' not in st.session_state:
            st.warning("⚠️ Please compare strategies first in the Strategy Comparison tab")
        else:
            selected_strategy = st.selectbox(
                "Select Strategy for Analysis:",
                list(st.session_state['strategies'].keys()),
                format_func=lambda x: st.session_state['strategies'][x]['name']
            )

            col1, col2 = st.columns(2)

            with col1:
                current_production = st.number_input("Current Annual Production (kWh):", min_value=1000, max_value=100000000, value=150000, step=10000)
                electricity_price = st.number_input("Electricity Price ($/kWh):", min_value=0.01, max_value=1.0, value=0.12, step=0.01)

            with col2:
                discount_rate = st.slider("Discount Rate (%):", 1, 20, 8) / 100
                project_lifetime = st.slider("Analysis Period (years):", 5, 30, 15)

            if st.button("💰 Calculate Financial Metrics", key="calc_financial"):
                strategy = st.session_state['strategies'][selected_strategy]

                # Estimate upgraded production
                performance_gain = strategy['performance_gain_pct'] / 100
                upgraded_production = current_production * (1 + performance_gain)

                metrics = planner.calculate_financial_metrics(
                    strategy['total_cost'],
                    current_production,
                    upgraded_production,
                    electricity_price,
                    discount_rate,
                    project_lifetime
                )

                st.session_state['financial_metrics'] = metrics

                st.success("✅ Financial analysis completed")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("NPV", f"${metrics['npv']:,.0f}")
                    st.metric("IRR", f"{metrics['irr']*100:.2f}%" if metrics['irr'] > 0 else "N/A")

                with col2:
                    st.metric("Simple Payback", f"{metrics['simple_payback_years']:.1f} years")
                    st.metric("ROI", f"{metrics['roi_pct']:.1f}%")

                with col3:
                    st.metric("LCOE", f"${metrics['lcoe']:.4f}/kWh")
                    st.metric("Annual Revenue", f"${metrics['annual_revenue_increase']:,.0f}")

                with col4:
                    st.metric("Production Increase", f"{metrics['annual_production_increase_kwh']:,.0f} kWh")
                    st.metric("Lifetime Savings", f"${metrics['total_lifetime_savings']:,.0f}")

                # Cash flow projection
                st.subheader("Cash Flow Projection")

                years = list(range(project_lifetime + 1))
                cash_flows = []

                for year in years:
                    if year == 0:
                        cash_flows.append(-strategy['total_cost'])
                    else:
                        annual_revenue = metrics['annual_revenue_increase'] * ((1.025) ** year)
                        o_and_m = strategy['total_cost'] * 0.01
                        cash_flows.append(annual_revenue - o_and_m)

                cumulative_cash = np.cumsum(cash_flows)

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Annual Cash Flow', 'Cumulative Cash Flow')
                )

                fig.add_trace(
                    go.Bar(x=years, y=cash_flows, name='Annual Cash Flow',
                           marker_color=['#E74C3C' if cf < 0 else '#2ECC71' for cf in cash_flows]),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=years, y=cumulative_cash, name='Cumulative',
                              line=dict(color='#3498DB', width=3)),
                    row=1, col=2
                )

                fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

                fig.update_xaxes(title_text="Year", row=1, col=1)
                fig.update_xaxes(title_text="Year", row=1, col=2)
                fig.update_yaxes(title_text="Cash Flow ($)", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative ($)", row=1, col=2)

                fig.update_layout(height=400, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Sensitivity analysis
                st.subheader("Sensitivity Analysis")

                electricity_prices = np.linspace(electricity_price * 0.7, electricity_price * 1.3, 20)
                npvs = []

                for price in electricity_prices:
                    temp_metrics = planner.calculate_financial_metrics(
                        strategy['total_cost'],
                        current_production,
                        upgraded_production,
                        price,
                        discount_rate,
                        project_lifetime
                    )
                    npvs.append(temp_metrics['npv'])

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=electricity_prices,
                    y=npvs,
                    mode='lines',
                    line=dict(color='#3498DB', width=3),
                    fill='tozeroy'
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="red")

                fig.update_layout(
                    title="NPV Sensitivity to Electricity Price",
                    xaxis_title="Electricity Price ($/kWh)",
                    yaxis_title="NPV ($)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Implementation Roadmap")

        if 'strategies' not in st.session_state:
            st.warning("⚠️ Please compare strategies first in the Strategy Comparison tab")
        else:
            selected_strategy = st.selectbox(
                "Select Strategy:",
                list(st.session_state['strategies'].keys()),
                format_func=lambda x: st.session_state['strategies'][x]['name'],
                key="roadmap_strategy"
            )

            start_date = st.date_input("Project Start Date:", datetime.now())

            if st.button("🗺️ Generate Roadmap", key="gen_roadmap"):
                strategy = st.session_state['strategies'][selected_strategy]

                roadmap = planner.create_implementation_roadmap(
                    selected_strategy,
                    strategy['implementation_time_months'],
                    datetime.combine(start_date, datetime.min.time())
                )

                st.success(f"✅ Roadmap generated for {strategy['name']}")

                st.dataframe(roadmap, use_container_width=True)

                # Gantt chart
                st.subheader("Project Timeline (Gantt Chart)")

                fig = go.Figure()

                for idx, row in roadmap.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['Duration (months)']],
                        y=[row['Phase']],
                        orientation='h',
                        name=row['Phase'],
                        text=f"{row['Duration (months)']} months",
                        textposition='inside',
                        marker=dict(
                            color=f"rgba({50+idx*30}, {100+idx*20}, {200-idx*20}, 0.8)"
                        )
                    ))

                fig.update_layout(
                    title="Implementation Timeline",
                    xaxis_title="Duration (months)",
                    yaxis_title="Phase",
                    height=500,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Completion progress
                st.subheader("Project Completion Milestones")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=roadmap['Phase'],
                    y=roadmap['Completion (%)'],
                    mode='lines+markers',
                    line=dict(color='#2ECC71', width=3),
                    marker=dict(size=12)
                ))

                fig.update_layout(
                    title="Cumulative Completion Progress",
                    xaxis_title="Phase",
                    yaxis_title="Completion (%)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Key milestones
                st.subheader("Key Milestones")

                milestones = roadmap[roadmap['Completion (%)'].isin([30, 50, 75, 100])]

                for idx, row in milestones.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.write(f"**{row['Phase']}**")

                    with col2:
                        st.write(f"Complete by: {row['End Date']}")

                    with col3:
                        st.write(f"{row['Completion (%)']}%")

    st.divider()
    st.info("💡 **Revamp & Repower Planning** - Branch B10 | Strategic System Upgrade Analysis")
