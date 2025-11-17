"""
Circularity Assessment Dashboard UI for PV lifecycle analysis.

This module provides a comprehensive Streamlit-based dashboard for visualizing
and analyzing PV circularity metrics, including material flows, reuse/repair/recycling
strategies, impact assessments, and policy compliance tracking.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from ..core.data_models import (
    CircularityMetrics,
    MaterialFlow,
    ReuseMetrics,
    RepairMetrics,
    RecyclingMetrics,
    PolicyCompliance,
    ImpactScorecard,
    MaterialType,
    ProcessStage,
    ComplianceStatus,
)


class CircularityDashboardUI:
    """
    Production-ready Streamlit dashboard for PV circularity assessment.

    This class provides a comprehensive interface for visualizing and analyzing
    circularity metrics across the PV lifecycle, supporting decision-making
    for sustainable PV operations.

    Attributes:
        metrics: CircularityMetrics object containing assessment data
        title: Dashboard title
        theme: Color theme for visualizations
        cache_enabled: Whether to enable Streamlit caching

    Example:
        >>> from pv_circularity_simulator.dashboards import CircularityDashboardUI
        >>> from pv_circularity_simulator.core import CircularityMetrics
        >>>
        >>> metrics = CircularityMetrics(assessment_id="ASSESS-001")
        >>> dashboard = CircularityDashboardUI(metrics=metrics)
        >>> dashboard.render()
    """

    def __init__(
        self,
        metrics: Optional[CircularityMetrics] = None,
        title: str = "PV Circularity Assessment Dashboard",
        theme: str = "plotly",
        cache_enabled: bool = True
    ):
        """
        Initialize the Circularity Dashboard UI.

        Args:
            metrics: CircularityMetrics object with assessment data
            title: Dashboard title displayed at the top
            theme: Plotly theme for visualizations
            cache_enabled: Enable Streamlit caching for performance

        Raises:
            ValueError: If metrics is provided but invalid
        """
        self.metrics = metrics
        self.title = title
        self.theme = theme
        self.cache_enabled = cache_enabled

        # Initialize session state
        if 'selected_material' not in st.session_state:
            st.session_state.selected_material = None
        if 'selected_stage' not in st.session_state:
            st.session_state.selected_stage = None

    def render(self) -> None:
        """
        Render the complete dashboard interface.

        This is the main entry point for displaying the dashboard. It orchestrates
        all dashboard components including header, tabs, visualizations, and metrics.

        Side Effects:
            Modifies Streamlit's UI state by rendering widgets and components
        """
        self._render_header()

        if self.metrics is None or not self._validate_metrics():
            st.warning("⚠️ No valid circularity data available. Please load assessment data.")
            self._render_data_upload_section()
            return

        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔄 Material Flows",
            "♻️ 3R Strategies",
            "📋 Compliance & Impact"
        ])

        with tab1:
            self._render_overview_tab()

        with tab2:
            self.material_flow_visualizer()

        with tab3:
            self.reuse_repair_recycling_tabs()

        with tab4:
            col1, col2 = st.columns([1, 1])
            with col1:
                self.policy_compliance_tracker()
            with col2:
                self.impact_scorecards()

    def material_flow_visualizer(self) -> None:
        """
        Visualize material flows through the PV lifecycle.

        Creates interactive Sankey diagrams and flow charts showing material
        movement through manufacturing, operation, and end-of-life stages.
        Includes material-specific breakdowns and efficiency metrics.

        Features:
            - Interactive Sankey diagram of material flows
            - Material type filtering
            - Stage-by-stage efficiency analysis
            - Mass balance calculations
            - Loss identification and quantification

        Side Effects:
            Renders Plotly charts and Streamlit widgets to the dashboard
        """
        st.subheader("🔄 Material Flow Analysis")

        if not self.metrics or not self.metrics.material_flows:
            st.info("No material flow data available.")
            return

        # Convert material flows to DataFrame for analysis
        flows_data = []
        for flow in self.metrics.material_flows:
            flows_data.append({
                'Material': flow.material_type.value,
                'Stage': flow.stage.value,
                'Input (kg)': flow.input_mass_kg,
                'Output (kg)': flow.output_mass_kg,
                'Loss (kg)': flow.loss_mass_kg,
                'Efficiency (%)': flow.efficiency * 100,
                'Location': flow.location,
                'Timestamp': flow.timestamp
            })

        df_flows = pd.DataFrame(flows_data)

        # Material type filter
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_materials = st.multiselect(
                "Filter by Material",
                options=df_flows['Material'].unique(),
                default=df_flows['Material'].unique()
            )

        df_filtered = df_flows[df_flows['Material'].isin(selected_materials)]

        # Create Sankey diagram
        st.markdown("#### Material Flow Sankey Diagram")
        sankey_fig = self._create_sankey_diagram(df_filtered)
        st.plotly_chart(sankey_fig, use_container_width=True)

        # Flow metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_input = df_filtered['Input (kg)'].sum()
            st.metric("Total Input", f"{total_input:,.0f} kg")
        with col2:
            total_output = df_filtered['Output (kg)'].sum()
            st.metric("Total Output", f"{total_output:,.0f} kg")
        with col3:
            total_loss = df_filtered['Loss (kg)'].sum()
            loss_pct = (total_loss / total_input * 100) if total_input > 0 else 0
            st.metric("Total Loss", f"{total_loss:,.0f} kg", f"{loss_pct:.1f}%")
        with col4:
            avg_efficiency = df_filtered['Efficiency (%)'].mean()
            st.metric("Avg Efficiency", f"{avg_efficiency:.1f}%")

        # Stage-by-stage breakdown
        st.markdown("#### Flow by Lifecycle Stage")
        stage_summary = df_filtered.groupby('Stage').agg({
            'Input (kg)': 'sum',
            'Output (kg)': 'sum',
            'Loss (kg)': 'sum',
            'Efficiency (%)': 'mean'
        }).round(2)

        fig_stage = go.Figure()
        fig_stage.add_trace(go.Bar(
            name='Input',
            x=stage_summary.index,
            y=stage_summary['Input (kg)'],
            marker_color='lightblue'
        ))
        fig_stage.add_trace(go.Bar(
            name='Output',
            x=stage_summary.index,
            y=stage_summary['Output (kg)'],
            marker_color='green'
        ))
        fig_stage.add_trace(go.Bar(
            name='Loss',
            x=stage_summary.index,
            y=stage_summary['Loss (kg)'],
            marker_color='red'
        ))
        fig_stage.update_layout(
            barmode='group',
            title="Material Mass by Lifecycle Stage",
            xaxis_title="Stage",
            yaxis_title="Mass (kg)",
            height=400
        )
        st.plotly_chart(fig_stage, use_container_width=True)

        # Detailed data table
        with st.expander("📋 Detailed Flow Data"):
            st.dataframe(
                df_filtered.sort_values('Timestamp', ascending=False),
                use_container_width=True,
                hide_index=True
            )

    def reuse_repair_recycling_tabs(self) -> None:
        """
        Display reuse, repair, and recycling metrics in tabbed interface.

        Provides comprehensive visualization and analysis of the 3R strategies
        (Reduce, Reuse, Recycle) with detailed metrics, trends, and performance
        indicators for each strategy.

        Features:
            - Reuse: Success rates, quality grades, cost savings
            - Repair: Failure modes, success rates, performance recovery
            - Recycling: Recovery rates, material-specific yields, economics

        Side Effects:
            Renders Streamlit tabs with metrics, charts, and data tables
        """
        st.subheader("♻️ 3R Strategies: Reuse, Repair, Recycling")

        reuse_tab, repair_tab, recycling_tab = st.tabs([
            "🔄 Reuse",
            "🔧 Repair",
            "♻️ Recycling"
        ])

        # REUSE TAB
        with reuse_tab:
            if self.metrics and self.metrics.reuse_metrics:
                reuse = self.metrics.reuse_metrics

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Modules Collected",
                        f"{reuse.total_modules_collected:,}",
                        help="Total modules collected for potential reuse"
                    )
                with col2:
                    st.metric(
                        "Reuse Rate",
                        f"{reuse.reuse_rate:.1f}%",
                        help="Percentage of collected modules successfully reused"
                    )
                with col3:
                    st.metric(
                        "Avg Residual Capacity",
                        f"{reuse.avg_residual_capacity_pct:.1f}%",
                        help="Average remaining power capacity of reused modules"
                    )
                with col4:
                    st.metric(
                        "Life Extension",
                        f"{reuse.avg_extension_years:.1f} yrs",
                        help="Average operational life extension through reuse"
                    )

                # Economic and environmental impact
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "💰 Cost Savings",
                        f"${reuse.cost_savings_usd:,.0f}",
                        help="Total cost savings vs purchasing new modules"
                    )
                with col2:
                    st.metric(
                        "🌍 CO₂ Avoided",
                        f"{reuse.co2_avoided_kg:,.0f} kg",
                        help="CO₂ emissions avoided through module reuse"
                    )

                # Quality grade distribution
                if reuse.quality_grade_distribution:
                    st.markdown("#### Quality Grade Distribution")
                    grades_df = pd.DataFrame([
                        {'Grade': k, 'Count': v}
                        for k, v in reuse.quality_grade_distribution.items()
                    ])
                    fig_grades = px.pie(
                        grades_df,
                        values='Count',
                        names='Grade',
                        title="Reused Modules by Quality Grade"
                    )
                    st.plotly_chart(fig_grades, use_container_width=True)

                # Reuse funnel
                st.markdown("#### Reuse Funnel Analysis")
                funnel_data = pd.DataFrame({
                    'Stage': ['Collected', 'Suitable', 'Reused'],
                    'Count': [
                        reuse.total_modules_collected,
                        reuse.modules_suitable_for_reuse,
                        reuse.modules_reused
                    ]
                })
                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_data['Stage'],
                    x=funnel_data['Count'],
                    textinfo="value+percent initial"
                ))
                fig_funnel.update_layout(height=300)
                st.plotly_chart(fig_funnel, use_container_width=True)
            else:
                st.info("No reuse metrics available.")

        # REPAIR TAB
        with repair_tab:
            if self.metrics and self.metrics.repair_metrics:
                repair = self.metrics.repair_metrics

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Modules Assessed",
                        f"{repair.total_modules_assessed:,}",
                        help="Total modules assessed for repair feasibility"
                    )
                with col2:
                    st.metric(
                        "Repair Success Rate",
                        f"{repair.repair_success_rate:.1f}%",
                        help="Percentage of repair attempts that succeeded"
                    )
                with col3:
                    st.metric(
                        "Avg Repair Cost",
                        f"${repair.avg_repair_cost_usd:.0f}",
                        help="Average cost per module repair"
                    )
                with col4:
                    st.metric(
                        "Performance Recovery",
                        f"{repair.avg_performance_recovery_pct:.1f}%",
                        help="Average performance recovery after repair"
                    )

                # Operational metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "⏱️ Avg Repair Time",
                        f"{repair.repair_time_hours:.1f} hrs",
                        help="Average time to complete a repair"
                    )
                with col2:
                    st.metric(
                        "📅 Warranty Extension",
                        f"{repair.warranty_extension_months} months",
                        help="Average warranty extension period"
                    )

                # Common failure modes
                if repair.common_failure_modes:
                    st.markdown("#### Common Failure Modes")
                    failures_df = pd.DataFrame([
                        {'Failure Mode': k, 'Occurrences': v}
                        for k, v in repair.common_failure_modes.items()
                    ]).sort_values('Occurrences', ascending=True)

                    fig_failures = px.bar(
                        failures_df,
                        x='Occurrences',
                        y='Failure Mode',
                        orientation='h',
                        title="Most Common Failure Types"
                    )
                    st.plotly_chart(fig_failures, use_container_width=True)

                # Repair funnel
                st.markdown("#### Repair Funnel Analysis")
                repair_funnel = pd.DataFrame({
                    'Stage': ['Assessed', 'Repairable', 'Repaired'],
                    'Count': [
                        repair.total_modules_assessed,
                        repair.modules_repairable,
                        repair.modules_repaired
                    ]
                })
                fig_repair_funnel = go.Figure(go.Funnel(
                    y=repair_funnel['Stage'],
                    x=repair_funnel['Count'],
                    textinfo="value+percent initial"
                ))
                fig_repair_funnel.update_layout(height=300)
                st.plotly_chart(fig_repair_funnel, use_container_width=True)
            else:
                st.info("No repair metrics available.")

        # RECYCLING TAB
        with recycling_tab:
            if self.metrics and self.metrics.recycling_metrics:
                recycling = self.metrics.recycling_metrics

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Mass Processed",
                        f"{recycling.total_mass_processed_kg:,.0f} kg",
                        help="Total mass processed through recycling"
                    )
                with col2:
                    st.metric(
                        "Mass Recovered",
                        f"{recycling.total_mass_recovered_kg:,.0f} kg",
                        help="Total mass of materials recovered"
                    )
                with col3:
                    st.metric(
                        "Recovery Efficiency",
                        f"{recycling.recovery_efficiency:.1f}%",
                        help="Overall material recovery efficiency"
                    )
                with col4:
                    net_value = (recycling.revenue_per_kg - recycling.recycling_cost_per_kg) * recycling.total_mass_processed_kg
                    st.metric(
                        "Net Value",
                        f"${net_value:,.0f}",
                        help="Net economic value (revenue - cost)"
                    )

                # Material recovery rates
                if recycling.material_recovery_rates:
                    st.markdown("#### Material Recovery Rates")
                    recovery_df = pd.DataFrame([
                        {'Material': k, 'Recovery Rate (%)': v}
                        for k, v in recycling.material_recovery_rates.items()
                    ]).sort_values('Recovery Rate (%)', ascending=False)

                    fig_recovery = px.bar(
                        recovery_df,
                        x='Material',
                        y='Recovery Rate (%)',
                        title="Recovery Efficiency by Material Type",
                        color='Recovery Rate (%)',
                        color_continuous_scale='Greens'
                    )
                    fig_recovery.update_layout(height=400)
                    st.plotly_chart(fig_recovery, use_container_width=True)

                # Resource consumption
                st.markdown("#### Recycling Process Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "⚡ Energy Consumption",
                        f"{recycling.energy_consumption_kwh:,.0f} kWh",
                        help="Total energy consumed in recycling process"
                    )
                with col2:
                    st.metric(
                        "💧 Water Usage",
                        f"{recycling.water_usage_liters:,.0f} L",
                        help="Water consumption in recycling operations"
                    )
                with col3:
                    st.metric(
                        "⚠️ Hazardous Waste",
                        f"{recycling.hazardous_waste_kg:,.0f} kg",
                        help="Hazardous waste generated during recycling"
                    )

                # Economic analysis
                st.markdown("#### Economic Analysis")
                cost_per_kg = recycling.recycling_cost_per_kg
                revenue_per_kg = recycling.revenue_per_kg

                econ_df = pd.DataFrame({
                    'Category': ['Cost per kg', 'Revenue per kg'],
                    'Value (USD)': [cost_per_kg, revenue_per_kg]
                })

                fig_econ = px.bar(
                    econ_df,
                    x='Category',
                    y='Value (USD)',
                    title="Recycling Economics",
                    color='Category',
                    color_discrete_map={
                        'Cost per kg': 'red',
                        'Revenue per kg': 'green'
                    }
                )
                st.plotly_chart(fig_econ, use_container_width=True)
            else:
                st.info("No recycling metrics available.")

    def impact_scorecards(self) -> None:
        """
        Display environmental and economic impact scorecards.

        Presents comprehensive impact assessment across multiple categories
        including carbon footprint, resource consumption, waste generation,
        and economic performance. Compares baseline (linear) vs circular
        economy scenarios.

        Features:
            - Multi-category impact assessment
            - Baseline vs circular comparison
            - Target tracking and progress monitoring
            - Data quality indicators
            - Trend visualization

        Side Effects:
            Renders metric cards, comparison charts, and progress indicators
        """
        st.subheader("📊 Impact Scorecards")

        if not self.metrics or not self.metrics.impact_scorecards:
            st.info("No impact scorecard data available.")
            return

        # Overall circularity index
        st.metric(
            "🎯 Circularity Index",
            f"{self.metrics.circularity_index:.1f}/100",
            help="Overall circularity performance score"
        )

        st.markdown("---")

        # Display scorecards
        for scorecard in self.metrics.impact_scorecards:
            with st.container():
                st.markdown(f"#### {scorecard.category}")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Baseline",
                        f"{scorecard.baseline_value:,.1f} {scorecard.unit}",
                        help="Linear economy baseline value"
                    )

                with col2:
                    st.metric(
                        "Circular",
                        f"{scorecard.circular_value:,.1f} {scorecard.unit}",
                        help="Circular economy value"
                    )

                with col3:
                    delta_color = "normal" if scorecard.improvement_pct >= 0 else "inverse"
                    st.metric(
                        "Improvement",
                        f"{scorecard.improvement_pct:.1f}%",
                        delta=f"{scorecard.improvement_pct:.1f}%",
                        delta_color=delta_color,
                        help="Percentage improvement from baseline"
                    )

                with col4:
                    if scorecard.target_value:
                        progress = (scorecard.circular_value / scorecard.target_value) * 100
                        st.metric(
                            f"Target Progress",
                            f"{progress:.0f}%",
                            help=f"Progress toward {scorecard.target_year} target"
                        )
                    else:
                        st.metric("Data Quality", f"{scorecard.data_quality}/5 ⭐")

                # Comparison chart
                comparison_df = pd.DataFrame({
                    'Scenario': ['Baseline', 'Circular', 'Target'],
                    'Value': [
                        scorecard.baseline_value,
                        scorecard.circular_value,
                        scorecard.target_value if scorecard.target_value else scorecard.circular_value
                    ]
                })

                fig = px.bar(
                    comparison_df,
                    x='Scenario',
                    y='Value',
                    title=f"{scorecard.category} Comparison",
                    color='Scenario',
                    color_discrete_map={
                        'Baseline': 'red',
                        'Circular': 'green',
                        'Target': 'blue'
                    }
                )
                fig.update_layout(
                    yaxis_title=scorecard.unit,
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Sub-metrics breakdown
                if scorecard.sub_metrics:
                    with st.expander("📊 Detailed Breakdown"):
                        sub_df = pd.DataFrame([
                            {'Metric': k, 'Value': v}
                            for k, v in scorecard.sub_metrics.items()
                        ])
                        st.dataframe(sub_df, use_container_width=True, hide_index=True)

                st.markdown("---")

        # Summary table
        with st.expander("📋 Complete Impact Summary"):
            summary_data = []
            for sc in self.metrics.impact_scorecards:
                summary_data.append({
                    'Category': sc.category,
                    'Baseline': f"{sc.baseline_value:.1f}",
                    'Circular': f"{sc.circular_value:.1f}",
                    'Improvement (%)': f"{sc.improvement_pct:.1f}",
                    'Unit': sc.unit,
                    'Target': f"{sc.target_value:.1f}" if sc.target_value else "N/A",
                    'Target Year': sc.target_year if sc.target_year else "N/A",
                    'Data Quality': f"{sc.data_quality}/5"
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    def policy_compliance_tracker(self) -> None:
        """
        Track compliance with environmental policies and regulations.

        Monitors adherence to various PV recycling and circularity regulations
        across different jurisdictions (EU WEEE, US regulations, etc.). Tracks
        collection rates, recovery targets, and compliance deadlines.

        Features:
            - Multi-jurisdiction compliance monitoring
            - Collection and recovery rate tracking
            - Compliance status visualization
            - Deadline tracking and alerts
            - Penalty and risk assessment

        Side Effects:
            Renders compliance status cards, gauges, and alert messages
        """
        st.subheader("📋 Policy Compliance Tracker")

        if not self.metrics or not self.metrics.policy_compliance:
            st.info("No policy compliance data available.")
            return

        # Overall compliance summary
        total_policies = len(self.metrics.policy_compliance)
        compliant = sum(1 for p in self.metrics.policy_compliance
                       if p.compliance_status == ComplianceStatus.COMPLIANT)
        compliance_rate = (compliant / total_policies * 100) if total_policies > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Policies", total_policies)
        with col2:
            st.metric("Compliant", f"{compliant}/{total_policies}")
        with col3:
            color = "🟢" if compliance_rate >= 80 else "🟡" if compliance_rate >= 50 else "🔴"
            st.metric("Compliance Rate", f"{color} {compliance_rate:.0f}%")

        st.markdown("---")

        # Individual policy tracking
        for policy in self.metrics.policy_compliance:
            with st.container():
                # Header with status indicator
                status_emoji = {
                    ComplianceStatus.COMPLIANT: "✅",
                    ComplianceStatus.PARTIALLY_COMPLIANT: "⚠️",
                    ComplianceStatus.NON_COMPLIANT: "❌",
                    ComplianceStatus.NOT_APPLICABLE: "➖"
                }

                st.markdown(f"### {status_emoji.get(policy.compliance_status, '❓')} {policy.policy_name}")
                st.caption(f"Jurisdiction: {policy.jurisdiction}")

                # Collection and recovery rates
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Collection Rate**")
                    collection_gap = policy.actual_collection_rate_pct - policy.required_collection_rate_pct

                    fig_collection = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=policy.actual_collection_rate_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Collection (%)"},
                        delta={
                            'reference': policy.required_collection_rate_pct,
                            'increasing': {'color': "green"},
                            'decreasing': {'color': "red"}
                        },
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, policy.required_collection_rate_pct], 'color': 'lightgray'},
                                {'range': [policy.required_collection_rate_pct, 100], 'color': 'lightgreen'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': policy.required_collection_rate_pct
                            }
                        }
                    ))
                    fig_collection.update_layout(height=250)
                    st.plotly_chart(fig_collection, use_container_width=True)

                    st.caption(f"Required: {policy.required_collection_rate_pct:.1f}% | "
                             f"Actual: {policy.actual_collection_rate_pct:.1f}% | "
                             f"Gap: {collection_gap:+.1f}%")

                with col2:
                    st.markdown("**Recovery Rate**")
                    recovery_gap = policy.actual_recovery_rate_pct - policy.required_recovery_rate_pct

                    fig_recovery = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=policy.actual_recovery_rate_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Recovery (%)"},
                        delta={
                            'reference': policy.required_recovery_rate_pct,
                            'increasing': {'color': "green"},
                            'decreasing': {'color': "red"}
                        },
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, policy.required_recovery_rate_pct], 'color': 'lightgray'},
                                {'range': [policy.required_recovery_rate_pct, 100], 'color': 'lightgreen'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': policy.required_recovery_rate_pct
                            }
                        }
                    ))
                    fig_recovery.update_layout(height=250)
                    st.plotly_chart(fig_recovery, use_container_width=True)

                    st.caption(f"Required: {policy.required_recovery_rate_pct:.1f}% | "
                             f"Actual: {policy.actual_recovery_rate_pct:.1f}% | "
                             f"Gap: {recovery_gap:+.1f}%")

                # Compliance details
                col1, col2 = st.columns(2)
                with col1:
                    if policy.compliance_deadline:
                        days_until = (policy.compliance_deadline - datetime.now()).days
                        if days_until < 0:
                            st.error(f"⏰ Deadline passed {abs(days_until)} days ago")
                        elif days_until < 30:
                            st.warning(f"⏰ Deadline in {days_until} days")
                        else:
                            st.info(f"📅 Deadline: {policy.compliance_deadline.strftime('%Y-%m-%d')}")

                with col2:
                    if policy.penalties_usd > 0:
                        st.error(f"💰 Penalties: ${policy.penalties_usd:,.0f}")

                if policy.notes:
                    st.info(f"📝 {policy.notes}")

                st.markdown("---")

    def _render_header(self) -> None:
        """Render the dashboard header with title and metadata."""
        st.set_page_config(
            page_title=self.title,
            page_icon="♻️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title(self.title)

        if self.metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Assessment ID: {self.metrics.assessment_id}")
            with col2:
                st.caption(f"Timestamp: {self.metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            with col3:
                st.caption(f"Circularity Index: {self.metrics.circularity_index:.1f}/100")

        st.markdown("---")

    def _render_overview_tab(self) -> None:
        """Render the overview tab with high-level metrics and charts."""
        st.markdown("## Dashboard Overview")

        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if self.metrics.reuse_metrics:
                st.metric(
                    "♻️ Reuse Rate",
                    f"{self.metrics.reuse_metrics.reuse_rate:.1f}%"
                )
            else:
                st.metric("♻️ Reuse Rate", "N/A")

        with col2:
            if self.metrics.repair_metrics:
                st.metric(
                    "🔧 Repair Success",
                    f"{self.metrics.repair_metrics.repair_success_rate:.1f}%"
                )
            else:
                st.metric("🔧 Repair Success", "N/A")

        with col3:
            if self.metrics.recycling_metrics:
                st.metric(
                    "🔄 Recovery Efficiency",
                    f"{self.metrics.recycling_metrics.recovery_efficiency:.1f}%"
                )
            else:
                st.metric("🔄 Recovery Efficiency", "N/A")

        with col4:
            material_flows_count = len(self.metrics.material_flows) if self.metrics.material_flows else 0
            st.metric("📊 Material Flows", material_flows_count)

        st.markdown("---")

        # Quick insights
        st.markdown("### 🎯 Key Insights")

        insights = self._generate_insights()
        for insight in insights:
            st.info(insight)

    def _render_data_upload_section(self) -> None:
        """Render data upload interface for loading circularity data."""
        st.markdown("### 📤 Upload Circularity Data")
        st.info("Upload a JSON file containing circularity assessment data to visualize metrics.")

        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload circularity metrics in JSON format"
        )

        if uploaded_file is not None:
            st.success("File uploaded! Processing...")
            # Note: Actual file processing would be implemented here

    def _validate_metrics(self) -> bool:
        """
        Validate that metrics object contains usable data.

        Returns:
            bool: True if metrics are valid, False otherwise
        """
        if self.metrics is None:
            return False

        # Check if at least one data source is available
        has_data = (
            (self.metrics.material_flows and len(self.metrics.material_flows) > 0) or
            self.metrics.reuse_metrics is not None or
            self.metrics.repair_metrics is not None or
            self.metrics.recycling_metrics is not None or
            (self.metrics.impact_scorecards and len(self.metrics.impact_scorecards) > 0) or
            (self.metrics.policy_compliance and len(self.metrics.policy_compliance) > 0)
        )

        return has_data

    def _create_sankey_diagram(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a Sankey diagram for material flows.

        Args:
            df: DataFrame containing material flow data

        Returns:
            Plotly Figure object with Sankey diagram
        """
        # Create nodes and links for Sankey
        stages = df['Stage'].unique()
        materials = df['Material'].unique()

        # Build node labels
        nodes = []
        node_dict = {}
        idx = 0

        for stage in stages:
            for material in materials:
                label = f"{material}\n{stage}"
                nodes.append(label)
                node_dict[(material, stage)] = idx
                idx += 1

        # Build links
        sources = []
        targets = []
        values = []

        # Group by material and create flow links between stages
        for material in materials:
            material_df = df[df['Material'] == material].sort_values('Timestamp')
            stage_list = material_df['Stage'].tolist()

            for i in range(len(stage_list) - 1):
                source_stage = stage_list[i]
                target_stage = stage_list[i + 1]

                if (material, source_stage) in node_dict and (material, target_stage) in node_dict:
                    sources.append(node_dict[(material, source_stage)])
                    targets.append(node_dict[(material, target_stage)])

                    # Use output from source stage as value
                    flow_value = material_df[material_df['Stage'] == source_stage]['Output (kg)'].iloc[0]
                    values.append(flow_value)

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0,100,200,0.4)"
            )
        )])

        fig.update_layout(
            title="Material Flow Sankey Diagram",
            font_size=10,
            height=500
        )

        return fig

    def _generate_insights(self) -> List[str]:
        """
        Generate automated insights from circularity data.

        Returns:
            List of insight strings
        """
        insights = []

        if self.metrics.reuse_metrics:
            if self.metrics.reuse_metrics.reuse_rate > 60:
                insights.append(
                    f"✅ Strong reuse performance: {self.metrics.reuse_metrics.reuse_rate:.1f}% "
                    f"of collected modules are being reused, exceeding typical industry benchmarks."
                )
            elif self.metrics.reuse_metrics.reuse_rate < 30:
                insights.append(
                    f"⚠️ Reuse opportunity: Only {self.metrics.reuse_metrics.reuse_rate:.1f}% "
                    f"reuse rate. Consider improving collection and testing processes."
                )

        if self.metrics.recycling_metrics:
            if self.metrics.recycling_metrics.recovery_efficiency > 80:
                insights.append(
                    f"✅ Excellent recycling efficiency: {self.metrics.recycling_metrics.recovery_efficiency:.1f}% "
                    f"material recovery rate demonstrates best-in-class recycling operations."
                )

        if self.metrics.policy_compliance:
            non_compliant = sum(
                1 for p in self.metrics.policy_compliance
                if p.compliance_status == ComplianceStatus.NON_COMPLIANT
            )
            if non_compliant > 0:
                insights.append(
                    f"⚠️ Compliance alert: {non_compliant} policy requirements are not being met. "
                    f"Review compliance tracker for details."
                )

        if not insights:
            insights.append("📊 Dashboard contains comprehensive circularity metrics. Explore tabs for detailed analysis.")

        return insights
