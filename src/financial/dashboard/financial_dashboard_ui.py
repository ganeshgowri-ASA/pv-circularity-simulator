"""
Financial Dashboard UI for PV Circularity Simulator.

Provides comprehensive interactive dashboard with LCOE calculator, cash flow
visualization, sensitivity analysis, and financial report generation capabilities.

This module integrates all financial analysis components into a user-friendly
Streamlit-based dashboard interface.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

from ..models.financial_models import (
    CostStructure,
    RevenueStream,
    CircularityMetrics,
    CashFlowModel,
    SensitivityParameter,
)
from ..calculators.lcoe_calculator import LCOECalculator, LCOEResult
from ..calculators.sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityMetric,
)
from ..visualization.charts import FinancialChartBuilder
from ..reporting.report_generator import FinancialReportGenerator


class FinancialDashboardUI:
    """
    Comprehensive Financial Analysis Dashboard for PV Systems.

    This class provides a complete Streamlit-based user interface for:
    - LCOE calculation and analysis
    - Cash flow visualization
    - Sensitivity analysis
    - Financial report generation

    Features:
    - Interactive parameter inputs
    - Real-time calculations
    - Professional visualizations using Plotly
    - Multi-format report export (PDF, Excel, HTML, CSV)
    - Circular economy impact analysis
    - Monte Carlo simulation
    """

    def __init__(
        self,
        project_name: str = "PV System Financial Analysis",
        company_name: str = "PV Circularity Simulator",
    ):
        """
        Initialize Financial Dashboard UI.

        Args:
            project_name: Name of the project for reports
            company_name: Company or organization name
        """
        self.project_name = project_name
        self.company_name = company_name
        self.chart_builder = FinancialChartBuilder()
        self.report_generator = FinancialReportGenerator(
            project_name=project_name,
            company_name=company_name
        )

        # Initialize session state
        if 'financial_model' not in st.session_state:
            st.session_state.financial_model = None
        if 'lcoe_result' not in st.session_state:
            st.session_state.lcoe_result = None

    def run(self):
        """
        Run the complete financial dashboard application.

        This is the main entry point that renders the full dashboard UI.
        """
        st.set_page_config(
            page_title="PV Financial Analysis Dashboard",
            page_icon="☀️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Header
        st.title("☀️ PV Financial Analysis Dashboard")
        st.markdown(f"**{self.company_name}** | Comprehensive Financial Modeling & Analysis")
        st.markdown("---")

        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select Analysis Module",
                [
                    "📊 Overview & Setup",
                    "💰 LCOE Calculator",
                    "📈 Cash Flow Analysis",
                    "🎯 Sensitivity Analysis",
                    "📄 Report Generator"
                ]
            )

            st.markdown("---")
            st.markdown("### Quick Stats")

            if st.session_state.lcoe_result is not None:
                result = st.session_state.lcoe_result
                st.metric("LCOE", f"${result.lcoe:.4f}/kWh")
                st.metric("Circularity Benefit",
                         f"{result.circularity_benefit/result.without_circularity*100:.1f}%")

        # Route to selected page
        if page == "📊 Overview & Setup":
            self._render_overview_page()
        elif page == "💰 LCOE Calculator":
            self.lcoe_calculator_display()
        elif page == "📈 Cash Flow Analysis":
            self.cashflow_visualization()
        elif page == "🎯 Sensitivity Analysis":
            self.sensitivity_analysis_ui()
        elif page == "📄 Report Generator":
            self.financial_reports_generator()

    def _render_overview_page(self):
        """Render overview and system setup page."""
        st.header("Project Overview & System Configuration")

        st.info("""
        **Welcome to the PV Financial Analysis Dashboard!**

        This comprehensive tool helps you analyze the financial viability of
        photovoltaic systems with integrated circular economy considerations.

        **Features:**
        - 💰 LCOE Calculator: Complete levelized cost analysis
        - 📈 Cash Flow Visualization: Interactive financial projections
        - 🎯 Sensitivity Analysis: Risk assessment and parameter optimization
        - 📄 Report Generator: Professional multi-format reports
        - ♻️ Circularity Impact: Quantify 3R benefits (Reduce, Reuse, Recycle)
        """)

        st.subheader("Quick Start Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### System Parameters")
            system_capacity = st.number_input(
                "System Capacity (kWp)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=10.0
            )

            annual_production = st.number_input(
                "Annual Energy Production (kWh)",
                min_value=1000.0,
                max_value=100000000.0,
                value=150000.0,
                step=1000.0,
                help="Expected first-year energy production"
            )

            lifetime_years = st.slider(
                "System Lifetime (years)",
                min_value=10,
                max_value=50,
                value=25
            )

        with col2:
            st.markdown("#### Financial Parameters")
            equipment_cost = st.number_input(
                "Equipment Cost ($)",
                min_value=1000.0,
                max_value=10000000.0,
                value=150000.0,
                step=1000.0
            )

            energy_price = st.number_input(
                "Energy Price ($/kWh)",
                min_value=0.01,
                max_value=1.0,
                value=0.12,
                step=0.01,
                format="%.3f"
            )

            discount_rate = st.slider(
                "Discount Rate (%)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5
            ) / 100

        if st.button("🚀 Initialize Financial Model", type="primary"):
            # Create default models
            cost_structure = CostStructure(
                initial_capex=equipment_cost * 1.3,
                equipment_cost=equipment_cost,
                installation_cost=equipment_cost * 0.2,
                soft_costs=equipment_cost * 0.1,
                annual_opex=equipment_cost * 0.01,
                maintenance_cost=equipment_cost * 0.01,
            )

            revenue_stream = RevenueStream(
                annual_energy_production=annual_production,
                energy_price=energy_price,
            )

            circularity_metrics = CircularityMetrics()

            cash_flow_model = CashFlowModel(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=lifetime_years,
                discount_rate=discount_rate,
            )

            # Calculate LCOE
            calculator = LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=lifetime_years,
                discount_rate=discount_rate,
            )

            lcoe_result = calculator.calculate_lcoe()

            # Store in session state
            st.session_state.financial_model = cash_flow_model
            st.session_state.lcoe_result = lcoe_result

            st.success("✅ Financial model initialized successfully!")
            st.balloons()

    def lcoe_calculator_display(self):
        """
        Display comprehensive LCOE calculator interface.

        Provides interactive inputs for all cost and revenue parameters,
        calculates LCOE with circularity impacts, and displays results
        with detailed breakdowns and visualizations.
        """
        st.header("💰 Levelized Cost of Energy (LCOE) Calculator")

        st.markdown("""
        The LCOE represents the per-kilowatt-hour cost of building and operating
        a PV system over its lifetime. This calculator incorporates circular
        economy benefits including end-of-life recovery value.
        """)

        # Input sections
        with st.expander("⚙️ Cost Structure Inputs", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Capital Expenditure (CAPEX)**")
                equipment_cost = st.number_input(
                    "Equipment Cost ($)",
                    value=150000.0,
                    key="lcoe_equipment"
                )
                installation_cost = st.number_input(
                    "Installation Cost ($)",
                    value=30000.0,
                    key="lcoe_installation"
                )
                soft_costs = st.number_input(
                    "Soft Costs ($)",
                    value=20000.0,
                    key="lcoe_soft"
                )

            with col2:
                st.markdown("**Operating Expenditure (OPEX)**")
                maintenance_cost = st.number_input(
                    "Annual Maintenance ($)",
                    value=1500.0,
                    key="lcoe_maintenance"
                )
                insurance_cost = st.number_input(
                    "Annual Insurance ($)",
                    value=500.0,
                    key="lcoe_insurance"
                )
                land_lease_cost = st.number_input(
                    "Annual Land Lease ($)",
                    value=0.0,
                    key="lcoe_lease"
                )

            with col3:
                st.markdown("**End-of-Life Costs**")
                decommissioning_cost = st.number_input(
                    "Decommissioning Cost ($)",
                    value=5000.0,
                    key="lcoe_decommission"
                )
                disposal_cost = st.number_input(
                    "Disposal Cost ($)",
                    value=2000.0,
                    key="lcoe_disposal"
                )

        with st.expander("🌞 Energy Production & Revenue", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                annual_energy = st.number_input(
                    "Annual Energy Production (kWh)",
                    value=150000.0,
                    key="lcoe_energy"
                )
                energy_price = st.number_input(
                    "Energy Price ($/kWh)",
                    value=0.12,
                    format="%.4f",
                    key="lcoe_price"
                )
                degradation_rate = st.slider(
                    "Annual Degradation Rate (%)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key="lcoe_degradation"
                ) / 100

            with col2:
                feed_in_tariff = st.number_input(
                    "Feed-in Tariff ($/kWh)",
                    value=0.0,
                    format="%.4f",
                    key="lcoe_fit"
                )
                tariff_duration = st.number_input(
                    "Tariff Duration (years)",
                    value=0,
                    key="lcoe_fit_years"
                )
                escalation_rate = st.slider(
                    "Energy Price Escalation Rate (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    key="lcoe_escalation"
                ) / 100

        with st.expander("♻️ Circularity Parameters", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                material_recovery_rate = st.slider(
                    "Material Recovery Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=90.0,
                    step=5.0,
                    key="lcoe_recovery"
                ) / 100

                system_weight = st.number_input(
                    "System Weight (kg)",
                    value=1000.0,
                    key="lcoe_weight"
                )

                refurbishment_potential = st.slider(
                    "Refurbishment Potential (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    step=5.0,
                    key="lcoe_refurb"
                ) / 100

            with col2:
                recycling_revenue = st.number_input(
                    "Recycling Revenue ($/kg)",
                    value=12.0,
                    key="lcoe_recycling_rev"
                )

                recycling_cost = st.number_input(
                    "Recycling Cost ($/kg)",
                    value=5.0,
                    key="lcoe_recycling_cost"
                )

                refurbishment_value = st.slider(
                    "Refurbishment Value Retention (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=5.0,
                    key="lcoe_refurb_value"
                ) / 100

        with st.expander("📊 Financial Parameters"):
            col1, col2, col3 = st.columns(3)

            with col1:
                lifetime_years = st.slider(
                    "System Lifetime (years)",
                    min_value=10,
                    max_value=50,
                    value=25,
                    key="lcoe_lifetime"
                )

            with col2:
                discount_rate = st.slider(
                    "Discount Rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=6.0,
                    step=0.5,
                    key="lcoe_discount"
                ) / 100

            with col3:
                inflation_rate = st.slider(
                    "Inflation Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.5,
                    step=0.1,
                    key="lcoe_inflation"
                ) / 100

        # Calculate LCOE button
        if st.button("🔄 Calculate LCOE", type="primary"):
            # Create models
            cost_structure = CostStructure(
                initial_capex=equipment_cost + installation_cost + soft_costs,
                equipment_cost=equipment_cost,
                installation_cost=installation_cost,
                soft_costs=soft_costs,
                annual_opex=maintenance_cost + insurance_cost + land_lease_cost,
                maintenance_cost=maintenance_cost,
                insurance_cost=insurance_cost,
                land_lease_cost=land_lease_cost,
                decommissioning_cost=decommissioning_cost,
                disposal_cost=disposal_cost,
            )

            revenue_stream = RevenueStream(
                annual_energy_production=annual_energy,
                energy_price=energy_price,
                feed_in_tariff=feed_in_tariff,
                tariff_duration=int(tariff_duration),
                degradation_rate=degradation_rate,
                escalation_rate=escalation_rate,
            )

            circularity_metrics = CircularityMetrics(
                material_recovery_rate=material_recovery_rate,
                system_weight=system_weight,
                refurbishment_potential=refurbishment_potential,
                refurbishment_value=refurbishment_value,
                recycling_revenue=recycling_revenue,
                recycling_cost=recycling_cost,
            )

            # Calculate LCOE
            calculator = LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=lifetime_years,
                discount_rate=discount_rate,
                inflation_rate=inflation_rate,
            )

            lcoe_result = calculator.calculate_lcoe()

            # Store in session
            cash_flow_model = CashFlowModel(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=circularity_metrics,
                lifetime_years=lifetime_years,
                discount_rate=discount_rate,
                inflation_rate=inflation_rate,
            )
            st.session_state.financial_model = cash_flow_model
            st.session_state.lcoe_result = lcoe_result

            # Display results
            st.success("✅ LCOE calculation completed!")

            # Key metrics
            st.markdown("### 📊 Key Results")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "LCOE (Nominal)",
                    f"${lcoe_result.lcoe:.4f}/kWh",
                    help="Levelized cost including all expenses"
                )

            with col2:
                st.metric(
                    "LCOE (Real)",
                    f"${lcoe_result.lcoe_real:.4f}/kWh",
                    help="Inflation-adjusted LCOE"
                )

            with col3:
                benefit_pct = (lcoe_result.circularity_benefit /
                              lcoe_result.without_circularity * 100)
                st.metric(
                    "Circularity Benefit",
                    f"{benefit_pct:.1f}%",
                    f"${lcoe_result.circularity_benefit:.4f}/kWh",
                    help="LCOE reduction from circular economy"
                )

            with col4:
                circ_score = circularity_metrics.get_circularity_score()
                st.metric(
                    "Circularity Score",
                    f"{circ_score:.1f}/100",
                    help="Overall circular economy performance"
                )

            # Comparison chart
            st.markdown("### 🔄 Circularity Impact")
            # Create simple comparison for LCOE with/without circularity
            calc_no_circ = LCOECalculator(
                cost_structure=cost_structure,
                revenue_stream=revenue_stream,
                circularity_metrics=CircularityMetrics(
                    material_recovery_rate=0.0,
                    refurbishment_potential=0.0,
                ),
                lifetime_years=lifetime_years,
                discount_rate=discount_rate,
            )
            lcoe_no_circ = calc_no_circ.calculate_lcoe()

            circ_chart = self.chart_builder.create_circularity_impact_chart(
                lcoe_result,
                lcoe_no_circ
            )
            st.plotly_chart(circ_chart, use_container_width=True)

            # Cost breakdown
            st.markdown("### 💵 Cost Breakdown")
            col1, col2 = st.columns(2)

            with col1:
                pie_chart = self.chart_builder.create_lcoe_breakdown_pie(lcoe_result)
                st.plotly_chart(pie_chart, use_container_width=True)

            with col2:
                waterfall_chart = self.chart_builder.create_lcoe_breakdown_waterfall(
                    lcoe_result
                )
                st.plotly_chart(waterfall_chart, use_container_width=True)

            # Detailed breakdown table
            st.markdown("### 📋 Detailed Cost Breakdown")
            breakdown_df = pd.DataFrame([
                {
                    'Category': k,
                    'Amount ($)': f'${v:,.2f}',
                    'Percentage': f'{v/lcoe_result.total_lifetime_cost*100:.1f}%'
                }
                for k, v in lcoe_result.cost_breakdown.items()
            ])
            st.dataframe(breakdown_df, use_container_width=True)

    def cashflow_visualization(self):
        """
        Display comprehensive cash flow analysis and visualization.

        Shows detailed cash flow projections over system lifetime with
        interactive charts for revenue, costs, and net cash flows.
        """
        st.header("📈 Cash Flow Analysis & Visualization")

        if st.session_state.financial_model is None:
            st.warning("⚠️ Please initialize the financial model first in the Overview page.")
            return

        cash_flow_model = st.session_state.financial_model

        # Generate cash flow data
        df = cash_flow_model.generate_cash_flow_series()

        # Key metrics
        st.markdown("### 💰 Financial Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            npv = cash_flow_model.calculate_npv()
            st.metric("Net Present Value", f"${npv:,.0f}")

        with col2:
            irr = cash_flow_model.calculate_irr()
            st.metric("Internal Rate of Return", f"{irr*100:.2f}%")

        with col3:
            payback = cash_flow_model.calculate_payback_period()
            st.metric("Payback Period", f"{payback:.1f} years")

        with col4:
            roi = cash_flow_model.calculate_roi()
            st.metric("Return on Investment", f"{roi:.1f}%")

        st.markdown("---")

        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Cash Flow Waterfall",
            "📈 Revenue vs Costs",
            "💹 Cumulative Analysis",
            "📋 Data Table"
        ])

        with tab1:
            st.markdown("### Annual Cash Flow")
            waterfall_chart = self.chart_builder.create_cash_flow_waterfall(
                cash_flow_model
            )
            st.plotly_chart(waterfall_chart, use_container_width=True)

        with tab2:
            st.markdown("### Revenue and Cost Trends")
            breakdown_chart = self.chart_builder.create_cash_flow_breakdown(
                cash_flow_model
            )
            st.plotly_chart(breakdown_chart, use_container_width=True)

        with tab3:
            st.markdown("### Cumulative Cash Flow")
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['year'],
                y=df['cumulative_cash_flow'],
                mode='lines+markers',
                name='Cumulative Cash Flow',
                line=dict(color='#1976D2', width=3),
                fill='tonexty',
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Cumulative Cash Flow Over Time",
                xaxis_title="Year",
                yaxis_title="Cumulative Cash Flow ($)",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### Detailed Cash Flow Data")

            # Format dataframe for display
            display_df = df.copy()
            display_df['revenue'] = display_df['revenue'].apply(lambda x: f'${x:,.2f}')
            display_df['costs'] = display_df['costs'].apply(lambda x: f'${x:,.2f}')
            display_df['net_cash_flow'] = display_df['net_cash_flow'].apply(
                lambda x: f'${x:,.2f}'
            )
            display_df['cumulative_cash_flow'] = display_df['cumulative_cash_flow'].apply(
                lambda x: f'${x:,.2f}'
            )

            st.dataframe(display_df, use_container_width=True)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cash Flow Data (CSV)",
                data=csv,
                file_name=f"cash_flow_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # Dashboard view
        st.markdown("---")
        st.markdown("### 📊 Financial Dashboard")

        if st.session_state.lcoe_result is not None:
            dashboard = self.chart_builder.create_financial_metrics_dashboard(
                cash_flow_model,
                st.session_state.lcoe_result
            )
            st.plotly_chart(dashboard, use_container_width=True)

    def sensitivity_analysis_ui(self):
        """
        Display comprehensive sensitivity analysis interface.

        Provides one-way sensitivity analysis, tornado diagrams,
        two-way analysis, and Monte Carlo simulation capabilities.
        """
        st.header("🎯 Sensitivity Analysis")

        if st.session_state.financial_model is None:
            st.warning("⚠️ Please initialize the financial model first in the Overview page.")
            return

        cash_flow_model = st.session_state.financial_model

        # Create analyzer
        analyzer = SensitivityAnalyzer(
            base_cost_structure=cash_flow_model.cost_structure,
            base_revenue_stream=cash_flow_model.revenue_stream,
            base_circularity_metrics=cash_flow_model.circularity_metrics,
            lifetime_years=cash_flow_model.lifetime_years,
            discount_rate=cash_flow_model.discount_rate,
        )

        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "One-Way Sensitivity",
                "Tornado Diagram",
                "Two-Way Sensitivity",
                "Monte Carlo Simulation"
            ]
        )

        if analysis_type == "One-Way Sensitivity":
            self._render_one_way_sensitivity(analyzer)
        elif analysis_type == "Tornado Diagram":
            self._render_tornado_analysis(analyzer)
        elif analysis_type == "Two-Way Sensitivity":
            self._render_two_way_sensitivity(analyzer)
        elif analysis_type == "Monte Carlo Simulation":
            self._render_monte_carlo(analyzer)

    def _render_one_way_sensitivity(self, analyzer: SensitivityAnalyzer):
        """Render one-way sensitivity analysis interface."""
        st.markdown("### 📊 One-Way Sensitivity Analysis")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Parameter Selection")

            param_name = st.selectbox(
                "Parameter to Analyze",
                [
                    "equipment_cost",
                    "annual_energy_production",
                    "energy_price",
                    "maintenance_cost",
                    "discount_rate",
                    "degradation_rate",
                    "material_recovery_rate",
                ]
            )

            # Get base value
            if hasattr(analyzer.base_cost_structure, param_name):
                base_value = getattr(analyzer.base_cost_structure, param_name)
            elif hasattr(analyzer.base_revenue_stream, param_name):
                base_value = getattr(analyzer.base_revenue_stream, param_name)
            else:
                base_value = getattr(analyzer.base_circularity_metrics, param_name)

            min_value = st.number_input("Minimum Value", value=base_value * 0.5)
            max_value = st.number_input("Maximum Value", value=base_value * 1.5)
            step = st.number_input("Step Size", value=(max_value - min_value) / 20)

            metric = st.selectbox(
                "Output Metric",
                ["LCOE", "NPV", "IRR", "Payback Period", "ROI"]
            )

            metric_map = {
                "LCOE": SensitivityMetric.LCOE,
                "NPV": SensitivityMetric.NPV,
                "IRR": SensitivityMetric.IRR,
                "Payback Period": SensitivityMetric.PAYBACK_PERIOD,
                "ROI": SensitivityMetric.ROI,
            }

        if st.button("📊 Run Analysis"):
            parameter = SensitivityParameter(
                name=param_name,
                base_value=base_value,
                min_value=min_value,
                max_value=max_value,
                step=step,
            )

            result = analyzer.one_way_sensitivity(parameter, metric_map[metric])

            with col2:
                st.markdown("#### Analysis Results")

                # Create chart
                sens_chart = self.chart_builder.create_sensitivity_chart(result)
                st.plotly_chart(sens_chart, use_container_width=True)

                # Display statistics
                st.markdown("##### Statistics")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Elasticity", f"{result.elasticity:.3f}")

                with col_b:
                    min_metric, max_metric = result.get_range_impact()
                    range_impact = max_metric - min_metric
                    st.metric("Range Impact", f"{range_impact:.4f}")

                with col_c:
                    pct_change = result.get_percentage_change()
                    max_pct = np.max(np.abs(pct_change))
                    st.metric("Max % Change", f"{max_pct:.1f}%")

                # Data table
                st.markdown("##### Data")
                result_df = result.to_dataframe()
                st.dataframe(result_df, use_container_width=True)

    def _render_tornado_analysis(self, analyzer: SensitivityAnalyzer):
        """Render tornado diagram analysis interface."""
        st.markdown("### 🌪️ Tornado Diagram Analysis")

        st.info("""
        Tornado diagrams show the sensitivity of the output metric to multiple
        parameters simultaneously, sorted by impact magnitude.
        """)

        variation_pct = st.slider(
            "Parameter Variation (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=5.0,
            help="Percentage to vary each parameter (+/-)"
        )

        metric = st.selectbox(
            "Output Metric",
            ["LCOE", "NPV", "IRR"],
            key="tornado_metric"
        )

        metric_map = {
            "LCOE": SensitivityMetric.LCOE,
            "NPV": SensitivityMetric.NPV,
            "IRR": SensitivityMetric.IRR,
        }

        if st.button("🌪️ Generate Tornado Diagram"):
            # Define parameters to analyze
            parameters = [
                SensitivityParameter(
                    "equipment_cost",
                    analyzer.base_cost_structure.equipment_cost,
                    analyzer.base_cost_structure.equipment_cost * 0.5,
                    analyzer.base_cost_structure.equipment_cost * 1.5,
                    1000
                ),
                SensitivityParameter(
                    "annual_energy_production",
                    analyzer.base_revenue_stream.annual_energy_production,
                    analyzer.base_revenue_stream.annual_energy_production * 0.5,
                    analyzer.base_revenue_stream.annual_energy_production * 1.5,
                    1000
                ),
                SensitivityParameter(
                    "energy_price",
                    analyzer.base_revenue_stream.energy_price,
                    analyzer.base_revenue_stream.energy_price * 0.5,
                    analyzer.base_revenue_stream.energy_price * 1.5,
                    0.01
                ),
                SensitivityParameter(
                    "maintenance_cost",
                    analyzer.base_cost_structure.maintenance_cost,
                    analyzer.base_cost_structure.maintenance_cost * 0.5,
                    analyzer.base_cost_structure.maintenance_cost * 1.5,
                    100
                ),
                SensitivityParameter(
                    "material_recovery_rate",
                    analyzer.base_circularity_metrics.material_recovery_rate,
                    0.5,
                    1.0,
                    0.05
                ),
            ]

            tornado_data = analyzer.tornado_analysis(
                parameters,
                metric_map[metric],
                variation_percent=variation_pct
            )

            # Display chart
            tornado_chart = self.chart_builder.create_tornado_diagram(tornado_data)
            st.plotly_chart(tornado_chart, use_container_width=True)

            # Display data
            st.markdown("##### Impact Data")
            tornado_df = tornado_data.to_dataframe()
            st.dataframe(tornado_df, use_container_width=True)

    def _render_two_way_sensitivity(self, analyzer: SensitivityAnalyzer):
        """Render two-way sensitivity analysis interface."""
        st.markdown("### 🎨 Two-Way Sensitivity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### First Parameter")
            param1_name = st.selectbox(
                "Parameter 1",
                ["equipment_cost", "annual_energy_production", "energy_price"],
                key="param1"
            )

        with col2:
            st.markdown("#### Second Parameter")
            param2_name = st.selectbox(
                "Parameter 2",
                ["discount_rate", "degradation_rate", "maintenance_cost"],
                key="param2"
            )

        if st.button("🎨 Generate Heatmap"):
            # Create parameters (simplified for demo)
            if param1_name == "equipment_cost":
                base1 = analyzer.base_cost_structure.equipment_cost
                param1 = SensitivityParameter(param1_name, base1, base1*0.7, base1*1.3, base1*0.1)
            elif param1_name == "annual_energy_production":
                base1 = analyzer.base_revenue_stream.annual_energy_production
                param1 = SensitivityParameter(param1_name, base1, base1*0.7, base1*1.3, base1*0.1)
            else:
                base1 = analyzer.base_revenue_stream.energy_price
                param1 = SensitivityParameter(param1_name, base1, base1*0.7, base1*1.3, base1*0.02)

            if param2_name == "discount_rate":
                param2 = SensitivityParameter(param2_name, 0.06, 0.03, 0.10, 0.01)
            elif param2_name == "degradation_rate":
                param2 = SensitivityParameter(param2_name, 0.005, 0.002, 0.01, 0.001)
            else:
                base2 = analyzer.base_cost_structure.maintenance_cost
                param2 = SensitivityParameter(param2_name, base2, base2*0.5, base2*1.5, base2*0.2)

            result_df = analyzer.two_way_sensitivity(
                param1,
                param2,
                SensitivityMetric.LCOE
            )

            # Display heatmap
            heatmap = self.chart_builder.create_2d_sensitivity_heatmap(
                result_df,
                param1_name,
                param2_name,
                "lcoe"
            )
            st.plotly_chart(heatmap, use_container_width=True)

    def _render_monte_carlo(self, analyzer: SensitivityAnalyzer):
        """Render Monte Carlo simulation interface."""
        st.markdown("### 🎲 Monte Carlo Simulation")

        st.info("""
        Monte Carlo simulation uses probability distributions to model
        uncertainty in input parameters and analyze risk.
        """)

        n_simulations = st.slider(
            "Number of Simulations",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )

        if st.button("🎲 Run Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                # Define distributions (normal distribution for demo)
                distributions = {
                    'equipment_cost': (
                        stats.norm,
                        {
                            'loc': analyzer.base_cost_structure.equipment_cost,
                            'scale': analyzer.base_cost_structure.equipment_cost * 0.1
                        }
                    ),
                    'energy_price': (
                        stats.norm,
                        {
                            'loc': analyzer.base_revenue_stream.energy_price,
                            'scale': analyzer.base_revenue_stream.energy_price * 0.15
                        }
                    ),
                }

                results_df = analyzer.monte_carlo_simulation(
                    distributions,
                    SensitivityMetric.LCOE,
                    n_simulations=n_simulations,
                    random_seed=42
                )

                # Display histogram
                histogram = self.chart_builder.create_monte_carlo_histogram(
                    results_df,
                    'lcoe'
                )
                st.plotly_chart(histogram, use_container_width=True)

                # Statistics
                st.markdown("##### Statistical Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean", f"${results_df['lcoe'].mean():.4f}/kWh")
                with col2:
                    st.metric("Median", f"${results_df['lcoe'].median():.4f}/kWh")
                with col3:
                    st.metric("Std Dev", f"${results_df['lcoe'].std():.4f}/kWh")
                with col4:
                    p90 = results_df['lcoe'].quantile(0.9)
                    st.metric("P90", f"${p90:.4f}/kWh")

    def financial_reports_generator(self):
        """
        Display financial report generation interface.

        Provides options to generate professional reports in multiple
        formats (PDF, Excel, HTML, CSV) with customizable content.
        """
        st.header("📄 Financial Report Generator")

        if st.session_state.financial_model is None or st.session_state.lcoe_result is None:
            st.warning("⚠️ Please initialize the financial model and calculate LCOE first.")
            return

        cash_flow_model = st.session_state.financial_model
        lcoe_result = st.session_state.lcoe_result

        st.markdown("""
        Generate comprehensive financial reports in multiple formats for
        stakeholders, investors, and documentation purposes.
        """)

        # Report configuration
        with st.expander("⚙️ Report Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                project_name = st.text_input(
                    "Project Name",
                    value=self.project_name
                )
                analyst_name = st.text_input(
                    "Analyst Name",
                    value="Financial Analysis Team"
                )

            with col2:
                company_name = st.text_input(
                    "Company Name",
                    value=self.company_name
                )
                include_charts = st.checkbox(
                    "Include Visualizations",
                    value=True
                )

        st.markdown("---")
        st.markdown("### 📑 Generate Reports")

        col1, col2, col3, col4 = st.columns(4)

        # PDF Report
        with col1:
            if st.button("📕 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    temp_dir = Path(tempfile.mkdtemp())
                    pdf_path = temp_dir / "financial_report.pdf"

                    report_gen = FinancialReportGenerator(
                        project_name=project_name,
                        company_name=company_name,
                        analyst_name=analyst_name
                    )

                    try:
                        report_gen.generate_executive_summary_pdf(
                            cash_flow_model,
                            lcoe_result,
                            pdf_path,
                            include_charts=include_charts
                        )

                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=f.read(),
                                file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )

                        st.success("✅ PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

        # Excel Report
        with col2:
            if st.button("📗 Generate Excel Report", use_container_width=True):
                with st.spinner("Generating Excel..."):
                    temp_dir = Path(tempfile.mkdtemp())
                    excel_path = temp_dir / "financial_report.xlsx"

                    report_gen = FinancialReportGenerator(
                        project_name=project_name,
                        company_name=company_name,
                        analyst_name=analyst_name
                    )

                    report_gen.generate_detailed_excel_report(
                        cash_flow_model,
                        lcoe_result,
                        excel_path
                    )

                    with open(excel_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=f.read(),
                            file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    st.success("✅ Excel report generated successfully!")

        # HTML Report
        with col3:
            if st.button("📘 Generate HTML Report", use_container_width=True):
                with st.spinner("Generating HTML..."):
                    temp_dir = Path(tempfile.mkdtemp())
                    html_path = temp_dir / "financial_report.html"

                    report_gen = FinancialReportGenerator(
                        project_name=project_name,
                        company_name=company_name,
                        analyst_name=analyst_name
                    )

                    report_gen.generate_html_report(
                        cash_flow_model,
                        lcoe_result,
                        html_path
                    )

                    with open(html_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=f.read(),
                            file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html"
                        )

                    st.success("✅ HTML report generated successfully!")

        # CSV Export
        with col4:
            if st.button("📊 Export Data (CSV)", use_container_width=True):
                temp_dir = Path(tempfile.mkdtemp())

                report_gen = FinancialReportGenerator(
                    project_name=project_name,
                    company_name=company_name
                )

                files = report_gen.export_data_to_csv(cash_flow_model, temp_dir)

                # Create zip of all CSV files
                import zipfile
                zip_path = temp_dir / "financial_data.zip"

                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_type, file_path in files.items():
                        zipf.write(file_path, file_path.name)

                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download CSV Bundle",
                        data=f.read(),
                        file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip"
                    )

                st.success("✅ CSV data exported successfully!")

        # Preview section
        st.markdown("---")
        st.markdown("### 👁️ Report Preview")

        tab1, tab2 = st.tabs(["📊 Executive Summary", "💵 Detailed Data"])

        with tab1:
            st.markdown("#### Key Financial Metrics")

            npv = cash_flow_model.calculate_npv()
            irr = cash_flow_model.calculate_irr()
            payback = cash_flow_model.calculate_payback_period()
            roi = cash_flow_model.calculate_roi()

            summary_df = pd.DataFrame([
                {"Metric": "LCOE ($/kWh)", "Value": f"${lcoe_result.lcoe:.4f}"},
                {"Metric": "NPV ($)", "Value": f"${npv:,.2f}"},
                {"Metric": "IRR (%)", "Value": f"{irr*100:.2f}%"},
                {"Metric": "Payback Period (years)", "Value": f"{payback:.1f}"},
                {"Metric": "ROI (%)", "Value": f"{roi:.1f}%"},
                {"Metric": "Circularity Score", "Value": f"{cash_flow_model.circularity_metrics.get_circularity_score():.1f}/100"},
            ])

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("#### Cash Flow Data")
            cf_df = cash_flow_model.generate_cash_flow_series()
            st.dataframe(cf_df, use_container_width=True)


# Convenience function to run dashboard
def run_dashboard():
    """
    Convenience function to run the financial dashboard.

    This function can be called directly to launch the Streamlit application.

    Example:
        >>> from financial.dashboard.financial_dashboard_ui import run_dashboard
        >>> run_dashboard()
    """
    dashboard = FinancialDashboardUI()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()
