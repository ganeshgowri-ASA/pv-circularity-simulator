"""
Integrated Analytics & Reporting Module.

Features:
- Cross-module KPI aggregation
- Executive dashboard
- Custom report generation (PDF export)
- Data export (CSV, Excel)
- Trend analysis and insights
- Comparative analysis across projects
- Performance benchmarking
- Automated alerts and notifications
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64

from utils.constants import PERFORMANCE_KPIS, COLOR_PALETTE
from utils.helpers import (
    calculate_performance_ratio,
    calculate_specific_yield,
    format_number,
    create_performance_chart,
    create_comparison_bar_chart
)


class AnalyticsReporter:
    """Integrated analytics and reporting system."""

    def __init__(self):
        """Initialize analytics reporter."""
        self.kpi_list = PERFORMANCE_KPIS
        self.color_palette = COLOR_PALETTE

    def aggregate_system_kpis(
        self,
        design_data: Optional[Dict] = None,
        performance_data: Optional[Dict] = None,
        financial_data: Optional[Dict] = None,
        circularity_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Aggregate KPIs across all modules.

        Args:
            design_data: Design module data
            performance_data: Performance monitoring data
            financial_data: Financial analysis data
            circularity_data: Circularity assessment data

        Returns:
            Dictionary with aggregated KPIs
        """
        kpis = {
            'timestamp': datetime.now(),
            'design': {},
            'performance': {},
            'financial': {},
            'circularity': {}
        }

        # Design KPIs
        if design_data:
            kpis['design'] = {
                'system_capacity_kw': design_data.get('capacity_kw', 0),
                'num_modules': design_data.get('num_modules', 0),
                'module_efficiency': design_data.get('module_efficiency', 0),
                'system_efficiency': design_data.get('system_efficiency', 0),
                'ctm_efficiency': design_data.get('ctm_efficiency', 0)
            }

        # Performance KPIs
        if performance_data:
            kpis['performance'] = {
                'performance_ratio': performance_data.get('performance_ratio', 0),
                'capacity_factor': performance_data.get('capacity_factor', 0),
                'specific_yield': performance_data.get('specific_yield', 0),
                'availability': performance_data.get('availability', 0.98),
                'energy_today_kwh': performance_data.get('energy_today', 0),
                'energy_total_kwh': performance_data.get('energy_total', 0)
            }

        # Financial KPIs
        if financial_data:
            kpis['financial'] = {
                'lcoe': financial_data.get('lcoe', 0),
                'npv': financial_data.get('npv', 0),
                'irr': financial_data.get('irr', 0),
                'payback_period': financial_data.get('payback_period', 0),
                'total_capex': financial_data.get('total_capex', 0),
                'annual_opex': financial_data.get('annual_opex', 0)
            }

        # Circularity KPIs
        if circularity_data:
            kpis['circularity'] = {
                'circularity_score': circularity_data.get('circularity_score', 0),
                'reuse_potential': circularity_data.get('reuse_potential', 0),
                'recyclability': circularity_data.get('recyclability', 0),
                'material_recovery_rate': circularity_data.get('material_recovery_rate', 0)
            }

        return kpis

    def create_executive_dashboard_data(
        self,
        aggregated_kpis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create executive dashboard summary.

        Args:
            aggregated_kpis: Aggregated KPI data

        Returns:
            Dictionary with dashboard data
        """
        design = aggregated_kpis.get('design', {})
        performance = aggregated_kpis.get('performance', {})
        financial = aggregated_kpis.get('financial', {})
        circularity = aggregated_kpis.get('circularity', {})

        # Overall system health score (0-100)
        health_components = []

        if performance.get('performance_ratio'):
            health_components.append(min(performance['performance_ratio'] * 100, 100))

        if performance.get('availability'):
            health_components.append(performance['availability'] * 100)

        if financial.get('irr'):
            health_components.append(min(financial['irr'] * 100 / 0.15, 100))  # Normalize to 15% IRR

        if circularity.get('circularity_score'):
            health_components.append(circularity['circularity_score'])

        overall_health = np.mean(health_components) if health_components else 0

        dashboard = {
            'overall_health_score': overall_health,
            'system_capacity_kw': design.get('system_capacity_kw', 0),
            'lifetime_energy_kwh': performance.get('energy_total_kwh', 0),
            'performance_ratio': performance.get('performance_ratio', 0),
            'lcoe': financial.get('lcoe', 0),
            'npv': financial.get('npv', 0),
            'irr': financial.get('irr', 0),
            'circularity_score': circularity.get('circularity_score', 0),
            'key_alerts': self._generate_alerts(aggregated_kpis),
            'recommendations': self._generate_recommendations(aggregated_kpis)
        }

        return dashboard

    def _generate_alerts(self, kpis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate automated alerts based on KPI thresholds."""
        alerts = []

        performance = kpis.get('performance', {})
        financial = kpis.get('financial', {})

        # Performance alerts
        pr = performance.get('performance_ratio', 0)
        if pr > 0 and pr < 0.75:
            alerts.append({
                'severity': 'high',
                'category': 'Performance',
                'message': f'Low Performance Ratio: {pr:.2%}. Expected > 80%',
                'action': 'Schedule system inspection and cleaning'
            })
        elif pr > 0 and pr < 0.80:
            alerts.append({
                'severity': 'medium',
                'category': 'Performance',
                'message': f'Performance Ratio below target: {pr:.2%}',
                'action': 'Review O&M procedures'
            })

        # Availability alerts
        availability = performance.get('availability', 1.0)
        if availability < 0.95:
            alerts.append({
                'severity': 'high',
                'category': 'Availability',
                'message': f'System availability: {availability:.2%}. Target > 98%',
                'action': 'Investigate downtime causes'
            })

        # Financial alerts
        irr = financial.get('irr', 0)
        if irr > 0 and irr < 0.08:
            alerts.append({
                'severity': 'medium',
                'category': 'Financial',
                'message': f'IRR below target: {irr:.2%}. Target > 10%',
                'action': 'Review revenue optimization strategies'
            })

        return alerts

    def _generate_recommendations(self, kpis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on KPI analysis."""
        recommendations = []

        performance = kpis.get('performance', {})
        financial = kpis.get('financial', {})
        circularity = kpis.get('circularity', {})

        # Performance recommendations
        pr = performance.get('performance_ratio', 0)
        if pr > 0 and pr < 0.80:
            recommendations.append("Consider implementing advanced O&M procedures to improve PR")

        capacity_factor = performance.get('capacity_factor', 0)
        if capacity_factor > 0 and capacity_factor < 0.18:
            recommendations.append("Evaluate potential for system optimization or capacity expansion")

        # Financial recommendations
        lcoe = financial.get('lcoe', 0)
        if lcoe > 0.10:
            recommendations.append("Explore cost reduction opportunities in O&M to improve LCOE")

        # Circularity recommendations
        circ_score = circularity.get('circularity_score', 0)
        if circ_score > 0 and circ_score < 60:
            recommendations.append("Develop circular economy strategies to improve sustainability score")

        if not recommendations:
            recommendations.append("System performance is within expected parameters")

        return recommendations

    def perform_trend_analysis(
        self,
        historical_data: pd.DataFrame,
        metric: str,
        window: int = 30
    ) -> Dict[str, Any]:
        """
        Perform trend analysis on historical data.

        Args:
            historical_data: Historical data DataFrame with datetime index
            metric: Metric column name
            window: Rolling window size (days)

        Returns:
            Dictionary with trend analysis results
        """
        if metric not in historical_data.columns:
            return {'error': f'Metric {metric} not found in data'}

        # Calculate trend indicators
        data = historical_data[metric].dropna()

        # Moving average
        ma = data.rolling(window=window).mean()

        # Linear trend
        x = np.arange(len(data))
        y = data.values

        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trend_line = coeffs[0] * x + coeffs[1]
            trend_slope = coeffs[0]
            trend_direction = 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
        else:
            trend_line = y
            trend_slope = 0
            trend_direction = 'insufficient_data'

        # Volatility
        volatility = data.std()

        # Recent vs historical average
        recent_avg = data.iloc[-window:].mean() if len(data) >= window else data.mean()
        historical_avg = data.mean()
        change_pct = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg != 0 else 0

        return {
            'metric': metric,
            'current_value': data.iloc[-1] if len(data) > 0 else 0,
            'moving_average': ma.iloc[-1] if len(ma) > 0 else 0,
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'volatility': volatility,
            'recent_avg': recent_avg,
            'historical_avg': historical_avg,
            'change_pct': change_pct
        }

    def benchmark_performance(
        self,
        current_system: Dict[str, float],
        benchmark_systems: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Benchmark current system against reference systems.

        Args:
            current_system: Current system metrics
            benchmark_systems: List of benchmark system metrics

        Returns:
            Dictionary with benchmark analysis
        """
        if not benchmark_systems:
            return {'error': 'No benchmark systems provided'}

        # Calculate percentiles
        metrics = list(current_system.keys())
        benchmarks = {}

        for metric in metrics:
            if metric in current_system:
                current_value = current_system[metric]

                # Collect values from benchmark systems
                benchmark_values = [
                    sys.get(metric, 0)
                    for sys in benchmark_systems
                    if metric in sys
                ]

                if benchmark_values:
                    benchmark_values.append(current_value)
                    benchmark_values.sort()

                    # Calculate percentile
                    percentile = (benchmark_values.index(current_value) / len(benchmark_values)) * 100

                    benchmarks[metric] = {
                        'current_value': current_value,
                        'percentile': percentile,
                        'min': min(benchmark_values),
                        'max': max(benchmark_values),
                        'median': np.median(benchmark_values),
                        'mean': np.mean(benchmark_values),
                        'std': np.std(benchmark_values)
                    }

        return benchmarks

    def generate_comparative_analysis(
        self,
        projects: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Generate comparative analysis across multiple projects.

        Args:
            projects: Dictionary of project data

        Returns:
            DataFrame with comparative analysis
        """
        comparison_data = []

        for project_name, metrics in projects.items():
            row = {'project': project_name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        return df

    def create_summary_report_data(
        self,
        aggregated_kpis: Dict[str, Any],
        project_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create comprehensive summary report data.

        Args:
            aggregated_kpis: Aggregated KPI data
            project_info: Project information

        Returns:
            Dictionary with report data
        """
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'project_info': project_info,
            'kpis': aggregated_kpis,
            'dashboard': self.create_executive_dashboard_data(aggregated_kpis),
            'summary_statistics': self._calculate_summary_statistics(aggregated_kpis)
        }

        return report

    def _calculate_summary_statistics(self, kpis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for reporting."""
        design = kpis.get('design', {})
        performance = kpis.get('performance', {})
        financial = kpis.get('financial', {})

        capacity_kw = design.get('system_capacity_kw', 0)
        energy_total = performance.get('energy_total_kwh', 0)
        capex = financial.get('total_capex', 0)

        stats = {
            'system_size_mw': capacity_kw / 1000,
            'lifetime_energy_mwh': energy_total / 1000,
            'total_investment_m': capex / 1_000_000,
            'energy_per_investment': energy_total / capex if capex > 0 else 0,
            'capacity_cost_per_kw': capex / capacity_kw if capacity_kw > 0 else 0
        }

        return stats

    def export_data_to_csv(self, data: pd.DataFrame) -> str:
        """
        Export data to CSV format.

        Args:
            data: DataFrame to export

        Returns:
            CSV string
        """
        return data.to_csv(index=False)

    def export_data_to_excel(self, data_dict: Dict[str, pd.DataFrame]) -> bytes:
        """
        Export multiple DataFrames to Excel with multiple sheets.

        Args:
            data_dict: Dictionary of sheet_name: DataFrame

        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)
        return output.getvalue()

    def create_download_link(self, data: bytes, filename: str, link_text: str) -> str:
        """
        Create download link for file.

        Args:
            data: File data as bytes
            filename: Filename for download
            link_text: Link display text

        Returns:
            HTML download link
        """
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'


def render_analytics_reporting():
    """Render analytics and reporting interface in Streamlit."""
    st.header("📊 Integrated Analytics & Reporting")
    st.markdown("Cross-module analytics, executive dashboards, and comprehensive reporting.")

    reporter = AnalyticsReporter()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Executive Dashboard",
        "📈 Trend Analysis",
        "📊 Benchmarking",
        "📋 Custom Reports",
        "💾 Data Export"
    ])

    with tab1:
        st.subheader("Executive Dashboard")

        st.write("### System Overview")

        # Simulate aggregated KPIs
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Design Parameters**")
            system_capacity = st.number_input("System Capacity (kW):", min_value=100, max_value=100000, value=5000)
            module_efficiency = st.slider("Module Efficiency (%):", 15, 25, 20) / 100

        with col2:
            st.write("**Performance Metrics**")
            performance_ratio = st.slider("Performance Ratio:", 0.70, 0.95, 0.82, 0.01)
            capacity_factor = st.slider("Capacity Factor:", 0.10, 0.30, 0.20, 0.01)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Financial Metrics**")
            lcoe = st.number_input("LCOE ($/kWh):", min_value=0.02, max_value=0.20, value=0.06, step=0.01)
            npv = st.number_input("NPV ($):", min_value=-1000000, max_value=10000000, value=2500000, step=100000)

        with col2:
            st.write("**Circularity Metrics**")
            circularity_score = st.slider("Circularity Score:", 0, 100, 75)
            recyclability = st.slider("Recyclability (%):", 50, 100, 85)

        if st.button("🎯 Generate Executive Dashboard", key="exec_dash"):
            # Aggregate KPIs
            aggregated_kpis = reporter.aggregate_system_kpis(
                design_data={
                    'capacity_kw': system_capacity,
                    'module_efficiency': module_efficiency,
                    'system_efficiency': module_efficiency * 0.85
                },
                performance_data={
                    'performance_ratio': performance_ratio,
                    'capacity_factor': capacity_factor,
                    'specific_yield': capacity_factor * 8760 / system_capacity,
                    'availability': 0.98,
                    'energy_total': system_capacity * capacity_factor * 8760 * 25
                },
                financial_data={
                    'lcoe': lcoe,
                    'npv': npv,
                    'irr': 0.12,
                    'payback_period': 8.5,
                    'total_capex': system_capacity * 1000 * 1.2
                },
                circularity_data={
                    'circularity_score': circularity_score,
                    'reuse_potential': 70,
                    'recyclability': recyclability,
                    'material_recovery_rate': recyclability / 100
                }
            )

            # Create dashboard
            dashboard = reporter.create_executive_dashboard_data(aggregated_kpis)

            st.session_state['dashboard'] = dashboard
            st.session_state['aggregated_kpis'] = aggregated_kpis

            st.success("✅ Executive Dashboard Generated")

            # Overall health score
            health_score = dashboard['overall_health_score']

            health_color = '#2ECC71' if health_score >= 80 else '#F39C12' if health_score >= 60 else '#E74C3C'
            health_status = 'Excellent' if health_score >= 80 else 'Good' if health_score >= 60 else 'Needs Attention'

            st.metric(
                "Overall System Health Score",
                f"{health_score:.1f}/100",
                delta=health_status,
                delta_color="normal" if health_score >= 60 else "inverse"
            )

            # Key metrics
            st.subheader("Key Performance Indicators")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("System Capacity", f"{dashboard['system_capacity_kw']:.0f} kW")

            with col2:
                st.metric("Performance Ratio", f"{dashboard['performance_ratio']:.2%}")

            with col3:
                st.metric("LCOE", f"${dashboard['lcoe']:.4f}/kWh")

            with col4:
                st.metric("Circularity Score", f"{dashboard['circularity_score']:.0f}/100")

            # Financial metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("NPV", f"${format_number(dashboard['npv'])}")

            with col2:
                st.metric("IRR", f"{dashboard['irr']:.2%}")

            with col3:
                lifetime_energy = dashboard['lifetime_energy_kwh']
                st.metric("Lifetime Energy", f"{format_number(lifetime_energy)} kWh")

            # Health score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': health_color},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Alerts
            if dashboard['key_alerts']:
                st.subheader("⚠️ Active Alerts")

                for alert in dashboard['key_alerts']:
                    severity_emoji = '🔴' if alert['severity'] == 'high' else '🟡'
                    st.warning(f"{severity_emoji} **{alert['category']}**: {alert['message']}\n\n**Action**: {alert['action']}")

            # Recommendations
            st.subheader("💡 Recommendations")

            for i, rec in enumerate(dashboard['recommendations'], 1):
                st.info(f"{i}. {rec}")

            # KPI breakdown chart
            st.subheader("KPI Category Breakdown")

            categories = ['Design', 'Performance', 'Financial', 'Circularity']
            scores = [
                85,  # Design score
                performance_ratio * 100,
                min(npv / 50000, 100) if npv > 0 else 0,  # Normalize NPV
                circularity_score
            ]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=categories,
                y=scores,
                text=[f"{s:.1f}" for s in scores],
                textposition='auto',
                marker_color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'],
                            COLOR_PALETTE['warning'], COLOR_PALETTE['info']]
            ))

            fig.update_layout(
                title="Performance by Category",
                yaxis_title="Score (0-100)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Trend Analysis")

        st.write("### Historical Performance Trends")

        # Generate synthetic historical data
        col1, col2 = st.columns(2)

        with col1:
            metric_to_analyze = st.selectbox(
                "Select Metric:",
                ["Performance Ratio", "Capacity Factor", "Energy Output", "Availability"]
            )

        with col2:
            trend_window = st.slider("Trend Window (days):", 7, 90, 30)

        if st.button("📈 Analyze Trends", key="trend_analysis"):
            with st.spinner("Analyzing trends..."):
                # Generate synthetic historical data
                dates = pd.date_range(end=datetime.now(), periods=180, freq='D')

                if metric_to_analyze == "Performance Ratio":
                    base_value = 0.82
                    noise = 0.03
                elif metric_to_analyze == "Capacity Factor":
                    base_value = 0.20
                    noise = 0.02
                elif metric_to_analyze == "Energy Output":
                    base_value = 15000
                    noise = 2000
                else:  # Availability
                    base_value = 0.98
                    noise = 0.01

                # Add seasonal variation and noise
                seasonal = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * noise
                random_noise = np.random.normal(0, noise / 2, len(dates))
                values = base_value + seasonal + random_noise

                hist_data = pd.DataFrame({
                    'date': dates,
                    metric_to_analyze: values
                })
                hist_data.set_index('date', inplace=True)

                # Perform trend analysis
                trend_result = reporter.perform_trend_analysis(
                    hist_data,
                    metric_to_analyze,
                    trend_window
                )

                st.session_state['trend_result'] = trend_result
                st.session_state['hist_data'] = hist_data

            st.success("✅ Trend Analysis Complete")

            # Display trend metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Value", f"{trend_result['current_value']:.4f}")

            with col2:
                st.metric("Moving Average", f"{trend_result['moving_average']:.4f}")

            with col3:
                direction_emoji = '📈' if trend_result['trend_direction'] == 'increasing' else '📉' if trend_result['trend_direction'] == 'decreasing' else '➡️'
                st.metric("Trend", f"{direction_emoji} {trend_result['trend_direction']}")

            with col4:
                st.metric("Change", f"{trend_result['change_pct']:+.2f}%")

            # Trend visualization
            fig = go.Figure()

            # Actual data
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data[metric_to_analyze],
                mode='lines',
                name='Actual',
                line=dict(color='#3498DB', width=1)
            ))

            # Moving average
            ma = hist_data[metric_to_analyze].rolling(window=trend_window).mean()
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=ma,
                mode='lines',
                name=f'{trend_window}-Day MA',
                line=dict(color='#E74C3C', width=2)
            ))

            # Historical average
            fig.add_hline(
                y=trend_result['historical_avg'],
                line_dash="dash",
                line_color="green",
                annotation_text="Historical Avg"
            )

            fig.update_layout(
                title=f"{metric_to_analyze} Trend Analysis",
                xaxis_title="Date",
                yaxis_title=metric_to_analyze,
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistical summary
            st.subheader("Statistical Summary")

            stats_df = pd.DataFrame({
                'Metric': [
                    'Current Value',
                    'Moving Average',
                    'Historical Average',
                    'Recent Average',
                    'Volatility (Std Dev)',
                    'Trend Slope',
                    'Change from Historical'
                ],
                'Value': [
                    f"{trend_result['current_value']:.4f}",
                    f"{trend_result['moving_average']:.4f}",
                    f"{trend_result['historical_avg']:.4f}",
                    f"{trend_result['recent_avg']:.4f}",
                    f"{trend_result['volatility']:.4f}",
                    f"{trend_result['trend_slope']:.6f}",
                    f"{trend_result['change_pct']:.2f}%"
                ]
            })

            st.dataframe(stats_df, use_container_width=True)

    with tab3:
        st.subheader("Performance Benchmarking")

        st.write("### Compare Against Industry Benchmarks")

        if st.button("📊 Generate Benchmark Analysis", key="benchmark"):
            # Create sample benchmark data
            current_system = {
                'Performance Ratio': 0.82,
                'Capacity Factor': 0.20,
                'LCOE ($/kWh)': 0.06,
                'Availability': 0.98,
                'Specific Yield (kWh/kWp/day)': 4.2
            }

            # Generate benchmark systems (simulated industry data)
            np.random.seed(42)
            benchmark_systems = []
            for _ in range(50):
                benchmark_systems.append({
                    'Performance Ratio': np.random.normal(0.80, 0.05),
                    'Capacity Factor': np.random.normal(0.19, 0.03),
                    'LCOE ($/kWh)': np.random.normal(0.065, 0.015),
                    'Availability': np.random.normal(0.97, 0.02),
                    'Specific Yield (kWh/kWp/day)': np.random.normal(4.0, 0.5)
                })

            benchmark_results = reporter.benchmark_performance(current_system, benchmark_systems)

            st.session_state['benchmark_results'] = benchmark_results

            st.success("✅ Benchmark Analysis Complete")

            # Percentile summary
            st.subheader("Percentile Rankings")

            percentile_data = []
            for metric, data in benchmark_results.items():
                percentile_data.append({
                    'Metric': metric,
                    'Current Value': f"{data['current_value']:.4f}",
                    'Percentile': f"{data['percentile']:.1f}%",
                    'Industry Median': f"{data['median']:.4f}",
                    'Industry Mean': f"{data['mean']:.4f}"
                })

            st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)

            # Percentile visualization
            fig = go.Figure()

            metrics = list(benchmark_results.keys())
            percentiles = [benchmark_results[m]['percentile'] for m in metrics]

            colors = ['#2ECC71' if p >= 50 else '#E74C3C' for p in percentiles]

            fig.add_trace(go.Bar(
                x=metrics,
                y=percentiles,
                text=[f"{p:.0f}%" for p in percentiles],
                textposition='auto',
                marker_color=colors
            ))

            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                annotation_text="Median"
            )

            fig.update_layout(
                title="Performance Percentile Rankings",
                yaxis_title="Percentile (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Box plots for each metric
            st.subheader("Distribution Analysis")

            for metric, data in list(benchmark_results.items())[:3]:  # Show first 3 metrics
                fig = go.Figure()

                # Box plot for industry
                fig.add_trace(go.Box(
                    y=[data['min'], data['median'], data['max']],
                    name='Industry Range',
                    marker_color='lightblue'
                ))

                # Current system point
                fig.add_trace(go.Scatter(
                    x=['Industry Range'],
                    y=[data['current_value']],
                    mode='markers',
                    name='Current System',
                    marker=dict(size=15, color='red', symbol='diamond')
                ))

                fig.update_layout(
                    title=f"{metric} - Industry Comparison",
                    yaxis_title=metric,
                    height=300,
                    template='plotly_white',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Custom Report Generation")

        st.write("### Configure Report Parameters")

        col1, col2 = st.columns(2)

        with col1:
            report_type = st.selectbox(
                "Report Type:",
                ["Executive Summary", "Technical Performance", "Financial Analysis", "Comprehensive"]
            )

            project_name = st.text_input("Project Name:", "Solar PV System")

        with col2:
            report_period = st.selectbox(
                "Report Period:",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Lifetime"]
            )

            location = st.text_input("Location:", "United States")

        include_sections = st.multiselect(
            "Include Sections:",
            ["Executive Summary", "System Overview", "Performance Analysis", "Financial Metrics",
             "Circularity Assessment", "Recommendations", "Detailed Data Tables"],
            default=["Executive Summary", "System Overview", "Performance Analysis"]
        )

        if st.button("📋 Generate Report", key="gen_report"):
            with st.spinner("Generating comprehensive report..."):
                # Create project info
                project_info = {
                    'project_name': project_name,
                    'location': location,
                    'report_type': report_type,
                    'report_period': report_period,
                    'report_date': datetime.now().strftime('%Y-%m-%d')
                }

                # Use aggregated KPIs if available
                if 'aggregated_kpis' in st.session_state:
                    agg_kpis = st.session_state['aggregated_kpis']
                else:
                    # Create sample data
                    agg_kpis = reporter.aggregate_system_kpis(
                        design_data={'capacity_kw': 5000, 'module_efficiency': 0.20},
                        performance_data={'performance_ratio': 0.82, 'capacity_factor': 0.20},
                        financial_data={'lcoe': 0.06, 'npv': 2500000, 'irr': 0.12},
                        circularity_data={'circularity_score': 75}
                    )

                # Create report data
                report_data = reporter.create_summary_report_data(agg_kpis, project_info)

                st.session_state['report_data'] = report_data

            st.success("✅ Report Generated Successfully")

            # Display report
            st.markdown("---")
            st.title(f"📊 {project_info['project_name']}")
            st.markdown(f"**Report Type:** {report_type} | **Period:** {report_period} | **Date:** {project_info['report_date']}")
            st.markdown(f"**Location:** {location}")
            st.markdown("---")

            if "Executive Summary" in include_sections:
                st.header("Executive Summary")

                dashboard = report_data['dashboard']

                st.write(f"""
                This report provides a comprehensive analysis of the {project_name} for the {report_period.lower()} period.
                The system demonstrates an overall health score of **{dashboard['overall_health_score']:.1f}/100**.

                **Key Highlights:**
                - System Capacity: {dashboard['system_capacity_kw']:.0f} kW
                - Performance Ratio: {dashboard['performance_ratio']:.2%}
                - Financial NPV: ${format_number(dashboard['npv'])}
                - Circularity Score: {dashboard['circularity_score']:.0f}/100
                """)

            if "System Overview" in include_sections:
                st.header("System Overview")

                design = agg_kpis.get('design', {})

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Design Specifications")
                    st.write(f"- System Capacity: {design.get('system_capacity_kw', 0):.0f} kW")
                    st.write(f"- Number of Modules: {design.get('num_modules', 0):,}")
                    st.write(f"- Module Efficiency: {design.get('module_efficiency', 0):.1%}")

                with col2:
                    st.subheader("Summary Statistics")
                    stats = report_data['summary_statistics']
                    st.write(f"- System Size: {stats['system_size_mw']:.2f} MW")
                    st.write(f"- Total Investment: ${stats['total_investment_m']:.2f}M")
                    st.write(f"- Cost per kW: ${stats['capacity_cost_per_kw']:.0f}/kW")

            if "Performance Analysis" in include_sections:
                st.header("Performance Analysis")

                performance = agg_kpis.get('performance', {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Performance Ratio", f"{performance.get('performance_ratio', 0):.2%}")

                with col2:
                    st.metric("Capacity Factor", f"{performance.get('capacity_factor', 0):.2%}")

                with col3:
                    st.metric("Availability", f"{performance.get('availability', 0):.2%}")

            if "Financial Metrics" in include_sections:
                st.header("Financial Metrics")

                financial = agg_kpis.get('financial', {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("LCOE", f"${financial.get('lcoe', 0):.4f}/kWh")

                with col2:
                    st.metric("NPV", f"${format_number(financial.get('npv', 0))}")

                with col3:
                    st.metric("IRR", f"{financial.get('irr', 0):.2%}")

            if "Recommendations" in include_sections:
                st.header("Recommendations")

                for i, rec in enumerate(dashboard['recommendations'], 1):
                    st.write(f"{i}. {rec}")

            st.markdown("---")
            st.caption(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PV Circularity Simulator")

    with tab5:
        st.subheader("Data Export")

        st.write("### Export System Data")

        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "Export Format:",
                ["CSV", "Excel (XLSX)", "JSON"]
            )

        with col2:
            data_type = st.selectbox(
                "Data Type:",
                ["All Data", "Performance Metrics", "Financial Data", "KPI Summary"]
            )

        if st.button("💾 Prepare Export", key="prep_export"):
            # Generate sample data for export
            if data_type == "Performance Metrics":
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                export_df = pd.DataFrame({
                    'date': dates,
                    'performance_ratio': np.random.normal(0.82, 0.03, 90),
                    'capacity_factor': np.random.normal(0.20, 0.02, 90),
                    'energy_kwh': np.random.normal(15000, 2000, 90),
                    'availability': np.random.normal(0.98, 0.01, 90)
                })
            elif data_type == "Financial Data":
                years = list(range(1, 26))
                export_df = pd.DataFrame({
                    'year': years,
                    'revenue': [50000 * (1.025 ** (y-1)) for y in years],
                    'opex': [10000] * 25,
                    'cash_flow': [40000 * (1.02 ** (y-1)) for y in years]
                })
            else:  # All Data or KPI Summary
                export_df = pd.DataFrame({
                    'metric': ['System Capacity', 'Performance Ratio', 'LCOE', 'NPV', 'Circularity Score'],
                    'value': [5000, 0.82, 0.06, 2500000, 75],
                    'unit': ['kW', '%', '$/kWh', '$', 'score']
                })

            st.session_state['export_df'] = export_df

            st.success("✅ Export Data Prepared")

            # Display preview
            st.subheader("Data Preview")
            st.dataframe(export_df.head(10), use_container_width=True)

            st.metric("Total Records", len(export_df))

            # Generate download based on format
            if export_format == "CSV":
                csv_data = reporter.export_data_to_csv(export_df)

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"pv_system_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            elif export_format == "Excel (XLSX)":
                # Create multi-sheet export
                data_dict = {
                    'Summary': export_df,
                    'Metadata': pd.DataFrame({
                        'parameter': ['Export Date', 'Data Type', 'Total Records'],
                        'value': [datetime.now().strftime('%Y-%m-%d'), data_type, len(export_df)]
                    })
                }

                excel_data = reporter.export_data_to_excel(data_dict)

                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"pv_system_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif export_format == "JSON":
                json_data = export_df.to_json(orient='records', indent=2)

                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"pv_system_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

    st.divider()
    st.info("💡 **Integrated Analytics & Reporting** | Complete Analytics & Reporting Suite")
