"""
B13-S05: Financial Dashboard
Production-ready Streamlit financial dashboard with visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional

from ..core.data_models import ProjectFinancials, FinancingStructure
from .lcoe_calculator import LCOECalculator
from .npv_analyzer import NPVAnalyzer
from .irr_calculator import IRRCalculator
from .bankability_analyzer import BankabilityAnalyzer


class FinancialUI:
    """
    Interactive financial analysis dashboard.
    """

    def __init__(self):
        """Initialize financial UI."""
        self.project = None
        self.financing = None

    def financial_summary(self, project: ProjectFinancials,
                         annual_revenue: float) -> None:
        """
        Display financial summary metrics.

        Args:
            project: Project financial parameters
            annual_revenue: Annual revenue
        """
        st.header("📊 Financial Summary")

        # Calculate key metrics
        lcoe_calc = LCOECalculator(project)
        npv_analyzer = NPVAnalyzer(project)
        irr_calc = IRRCalculator(project)

        annual_generation = annual_revenue / 0.10  # Assume $0.10/kWh
        lcoe_result = lcoe_calc.levelized_costs(annual_generation)
        npv_result = npv_analyzer.project_valuation(annual_revenue)
        irr_result = irr_calc.internal_rate_of_return(annual_revenue)

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "LCOE",
                f"${lcoe_result.lcoe_usd_per_kwh:.3f}/kWh",
                help="Levelized Cost of Energy"
            )

        with col2:
            st.metric(
                "NPV",
                f"${npv_result.npv_usd:,.0f}",
                delta=f"{'Profitable' if npv_result.npv_usd > 0 else 'Not Profitable'}",
                help="Net Present Value"
            )

        with col3:
            st.metric(
                "IRR",
                f"{irr_result.irr_percent:.2f}%",
                delta=f"{irr_result.irr_percent - irr_result.hurdle_rate_percent:.2f}% vs Hurdle",
                help="Internal Rate of Return"
            )

        with col4:
            st.metric(
                "Payback",
                f"{npv_result.payback_period_years or 'N/A'} years" if npv_result.payback_period_years else "N/A",
                help="Simple Payback Period"
            )

    def cash_flow_waterfall(self, project: ProjectFinancials,
                            annual_revenue: float) -> go.Figure:
        """
        Create cash flow waterfall chart.

        Args:
            project: Project parameters
            annual_revenue: Annual revenue

        Returns:
            Plotly figure
        """
        npv_analyzer = NPVAnalyzer(project)
        cash_flows = npv_analyzer.cash_flow_projection(annual_revenue)

        # Prepare data for waterfall
        years = [cf.year for cf in cash_flows]
        cumulative_cf = [cf.cumulative_cash_flow for cf in cash_flows]

        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            x=['Initial'] + [f"Year {y}" for y in years],
            y=[-project.capex_usd] + [cf.free_cash_flow for cf in cash_flows],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="Project Cash Flow Waterfall",
            xaxis_title="Year",
            yaxis_title="Cash Flow ($)",
            showlegend=False,
            height=500
        )

        return fig

    def sensitivity_charts(self, project: ProjectFinancials,
                          annual_generation: float) -> go.Figure:
        """
        Create sensitivity analysis tornado chart.

        Args:
            project: Project parameters
            annual_generation: Annual generation

        Returns:
            Plotly figure
        """
        lcoe_calc = LCOECalculator(project)
        sensitivity = lcoe_calc.sensitivity_analysis(annual_generation)

        # Prepare data
        params = list(sensitivity.keys())
        low_values = [sensitivity[p]['lcoe_low'] for p in params]
        high_values = [sensitivity[p]['lcoe_high'] for p in params]
        base_lcoe = sensitivity[params[0]]['base_lcoe']

        # Sort by sensitivity
        sensitivity_magnitudes = [abs(high_values[i] - low_values[i]) for i in range(len(params))]
        sorted_indices = sorted(range(len(params)),
                              key=lambda i: sensitivity_magnitudes[i],
                              reverse=True)

        sorted_params = [params[i] for i in sorted_indices]
        sorted_low = [low_values[i] for i in sorted_indices]
        sorted_high = [high_values[i] for i in sorted_indices]

        # Create tornado chart
        fig = go.Figure()

        # Low values (left bars)
        fig.add_trace(go.Bar(
            name='Low (-20%)',
            y=sorted_params,
            x=[v - base_lcoe for v in sorted_low],
            orientation='h',
            marker=dict(color='lightblue')
        ))

        # High values (right bars)
        fig.add_trace(go.Bar(
            name='High (+20%)',
            y=sorted_params,
            x=[v - base_lcoe for v in sorted_high],
            orientation='h',
            marker=dict(color='lightcoral')
        ))

        fig.update_layout(
            title="LCOE Sensitivity Analysis (Tornado Chart)",
            xaxis_title="LCOE Change from Base ($/kWh)",
            barmode='overlay',
            height=400
        )

        return fig

    def render_dashboard(self) -> None:
        """Render complete financial dashboard."""
        st.title("💰 Financial Analysis Dashboard")

        # Sidebar inputs
        st.sidebar.header("Project Parameters")

        capacity_kw = st.sidebar.number_input("Capacity (kW)", value=10000, step=1000)
        capex_per_kw = st.sidebar.number_input("CAPEX ($/kW)", value=1000, step=50)
        opex_pct = st.sidebar.slider("O&M (% of CAPEX)", 0.5, 3.0, 1.0, 0.1)
        discount_rate = st.sidebar.slider("Discount Rate (%)", 4.0, 15.0, 8.0, 0.5) / 100
        lifetime = st.sidebar.number_input("Project Lifetime (years)", value=25, min_value=10, max_value=40)

        # Create project
        project = ProjectFinancials(
            project_name="Sample Project",
            capacity_kw=capacity_kw,
            capex_usd=capacity_kw * capex_per_kw,
            opex_annual_usd=capacity_kw * capex_per_kw * (opex_pct / 100),
            project_lifetime_years=lifetime,
            discount_rate=discount_rate,
            inflation_rate=0.02,
            tax_rate=0.21,
            degradation_rate=0.005
        )

        # Revenue calculation
        capacity_factor = st.sidebar.slider("Capacity Factor (%)", 10.0, 40.0, 20.0, 1.0) / 100
        electricity_price = st.sidebar.number_input("Electricity Price ($/kWh)", value=0.10, step=0.01)

        annual_generation = capacity_kw * 8760 * capacity_factor
        annual_revenue = annual_generation * electricity_price

        # Display summary
        self.financial_summary(project, annual_revenue)

        # Cash flow waterfall
        st.subheader("Cash Flow Analysis")
        waterfall_fig = self.cash_flow_waterfall(project, annual_revenue)
        st.plotly_chart(waterfall_fig, use_container_width=True)

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        sensitivity_fig = self.sensitivity_charts(project, annual_generation)
        st.plotly_chart(sensitivity_fig, use_container_width=True)

        # Detailed cash flows table
        st.subheader("Detailed Cash Flows")
        npv_analyzer = NPVAnalyzer(project)
        cash_flows = npv_analyzer.cash_flow_projection(annual_revenue)

        cf_df = pd.DataFrame([
            {
                'Year': cf.year,
                'Revenue': cf.revenue,
                'OpEx': cf.operating_expenses,
                'Depreciation': cf.depreciation,
                'Taxes': cf.taxes,
                'Free Cash Flow': cf.free_cash_flow,
                'Cumulative CF': cf.cumulative_cash_flow
            }
            for cf in cash_flows
        ])

        st.dataframe(cf_df.style.format({
            'Revenue': '${:,.0f}',
            'OpEx': '${:,.0f}',
            'Depreciation': '${:,.0f}',
            'Taxes': '${:,.0f}',
            'Free Cash Flow': '${:,.0f}',
            'Cumulative CF': '${:,.0f}'
        }))


__all__ = ["FinancialUI"]
