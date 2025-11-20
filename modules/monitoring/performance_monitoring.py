"""
Real-time Performance Monitoring Module (Branch B07).

Features:
- Real-time KPI tracking (PR, capacity factor, specific yield, system efficiency)
- Power and energy monitoring (DC, AC, daily, monthly, annual)
- Inverter performance tracking
- System availability calculations
- Performance benchmarking
- Alerts and threshold monitoring
- Historical data analysis
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import PERFORMANCE_KPIS, INVERTER_TYPES
from utils.validators import PerformanceMetrics, SystemConfiguration
from utils.helpers import (
    calculate_performance_ratio,
    calculate_specific_yield,
    calculate_capacity_factor,
    temperature_corrected_power,
    calculate_noct_temperature
)


class PerformanceMonitor:
    """Real-time performance monitoring and analysis."""

    def __init__(self):
        """Initialize performance monitor."""
        self.kpi_names = PERFORMANCE_KPIS
        self.alert_thresholds = {
            'pr_min': 0.75,
            'efficiency_min': 0.10,
            'availability_min': 0.95,
            'inverter_efficiency_min': 0.92
        }

    def generate_real_time_data(
        self,
        system_capacity: float,
        current_irradiance: float,
        module_temp: float,
        ambient_temp: float,
        inverter_efficiency: float = 0.98,
        num_hours: int = 24
    ) -> pd.DataFrame:
        """
        Generate simulated real-time monitoring data.

        Args:
            system_capacity: System DC capacity (kW)
            current_irradiance: Current irradiance (W/m²)
            module_temp: Module temperature (°C)
            ambient_temp: Ambient temperature (°C)
            inverter_efficiency: Inverter efficiency (fraction)
            num_hours: Number of hours to generate

        Returns:
            DataFrame with real-time monitoring data
        """
        timestamps = [datetime.now() - timedelta(hours=num_hours-i-1) for i in range(num_hours)]

        data = {
            'timestamp': timestamps,
            'irradiance': [],
            'temp_module': [],
            'temp_ambient': [],
            'power_dc': [],
            'power_ac': [],
            'voltage_dc': [],
            'current_dc': [],
            'voltage_ac': [],
            'current_ac': [],
            'inverter_efficiency': [],
            'energy_kwh': []
        }

        for i, ts in enumerate(timestamps):
            hour = ts.hour

            # Simulate solar curve
            if 6 <= hour <= 18:
                hour_factor = np.sin(np.pi * (hour - 6) / 12)
                irr = current_irradiance * hour_factor * np.random.uniform(0.85, 1.0)
            else:
                irr = 0

            # Temperature variation
            temp_mod = ambient_temp + (45 - 20) * (irr / 800) + np.random.normal(0, 2)
            temp_amb = ambient_temp + 5 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 1)

            # DC power with temperature derating
            temp_coeff = -0.004
            temp_derating = 1 + temp_coeff * (temp_mod - 25)
            dc_power = system_capacity * (irr / 1000) * temp_derating * np.random.uniform(0.95, 1.0)
            dc_power = max(0, dc_power)

            # AC power with inverter efficiency
            inv_eff = inverter_efficiency * np.random.uniform(0.98, 1.0)
            ac_power = dc_power * inv_eff

            # Voltage and current (typical values)
            dc_voltage = 600 + np.random.normal(0, 10) if dc_power > 0 else 0
            dc_current = dc_power * 1000 / dc_voltage if dc_voltage > 0 else 0

            ac_voltage = 480 + np.random.normal(0, 5) if ac_power > 0 else 0
            ac_current = ac_power * 1000 / ac_voltage if ac_voltage > 0 else 0

            # Energy (kWh for 1-hour interval)
            energy = ac_power

            data['irradiance'].append(irr)
            data['temp_module'].append(temp_mod)
            data['temp_ambient'].append(temp_amb)
            data['power_dc'].append(dc_power)
            data['power_ac'].append(ac_power)
            data['voltage_dc'].append(dc_voltage)
            data['current_dc'].append(dc_current)
            data['voltage_ac'].append(ac_voltage)
            data['current_ac'].append(ac_current)
            data['inverter_efficiency'].append(inv_eff)
            data['energy_kwh'].append(energy)

        return pd.DataFrame(data)

    def calculate_kpis(
        self,
        performance_data: pd.DataFrame,
        system_capacity: float
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance KPIs.

        Args:
            performance_data: Performance monitoring data
            system_capacity: System DC capacity (kW)

        Returns:
            Dictionary of KPIs
        """
        # Energy metrics
        total_energy = performance_data['energy_kwh'].sum()
        daily_energy = total_energy / (len(performance_data) / 24)
        peak_power_dc = performance_data['power_dc'].max()
        peak_power_ac = performance_data['power_ac'].max()
        avg_power_dc = performance_data['power_dc'].mean()
        avg_power_ac = performance_data['power_ac'].mean()

        # Capacity factor
        hours = len(performance_data)
        capacity_factor = calculate_capacity_factor(total_energy, system_capacity, hours)

        # Specific yield
        days = hours / 24
        specific_yield = calculate_specific_yield(total_energy, system_capacity, days)

        # Performance ratio
        total_insolation = performance_data['irradiance'].sum() / 1000  # kWh/m²
        reference_yield = total_insolation
        final_yield = total_energy / system_capacity
        pr = final_yield / reference_yield if reference_yield > 0 else 0

        # Temperature-corrected PR
        avg_temp = performance_data['temp_module'].mean()
        temp_loss_factor = 1 + (-0.004) * (avg_temp - 25)
        pr_temp_corrected = pr / temp_loss_factor if temp_loss_factor > 0 else pr

        # System efficiency
        total_irradiance_energy = performance_data['irradiance'].sum() * system_capacity / 1000
        system_efficiency = total_energy / total_irradiance_energy if total_irradiance_energy > 0 else 0

        # Inverter efficiency
        total_dc_energy = performance_data['power_dc'].sum()
        inverter_efficiency = total_energy / total_dc_energy if total_dc_energy > 0 else 0

        # DC/AC ratio
        dc_ac_ratio = peak_power_dc / peak_power_ac if peak_power_ac > 0 else 0

        # Availability (assume 100% for now, in real system would track downtime)
        availability = 1.0

        return {
            'performance_ratio': pr,
            'pr_temp_corrected': pr_temp_corrected,
            'capacity_factor': capacity_factor,
            'specific_yield': specific_yield,
            'system_efficiency': system_efficiency,
            'inverter_efficiency': inverter_efficiency,
            'availability': availability,
            'total_energy_kwh': total_energy,
            'daily_energy_kwh': daily_energy,
            'peak_power_dc_kw': peak_power_dc,
            'peak_power_ac_kw': peak_power_ac,
            'avg_power_dc_kw': avg_power_dc,
            'avg_power_ac_kw': avg_power_ac,
            'dc_ac_ratio': dc_ac_ratio
        }

    def check_alerts(
        self,
        kpis: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """
        Check for performance alerts based on thresholds.

        Args:
            kpis: Calculated KPIs

        Returns:
            List of alerts
        """
        alerts = []

        if kpis['performance_ratio'] < self.alert_thresholds['pr_min']:
            alerts.append({
                'severity': 'high',
                'metric': 'Performance Ratio',
                'value': f"{kpis['performance_ratio']:.3f}",
                'threshold': f"{self.alert_thresholds['pr_min']:.3f}",
                'message': f"PR below threshold: {kpis['performance_ratio']:.3f} < {self.alert_thresholds['pr_min']:.3f}"
            })

        if kpis['system_efficiency'] < self.alert_thresholds['efficiency_min']:
            alerts.append({
                'severity': 'medium',
                'metric': 'System Efficiency',
                'value': f"{kpis['system_efficiency']:.3f}",
                'threshold': f"{self.alert_thresholds['efficiency_min']:.3f}",
                'message': f"System efficiency below threshold: {kpis['system_efficiency']:.3f} < {self.alert_thresholds['efficiency_min']:.3f}"
            })

        if kpis['availability'] < self.alert_thresholds['availability_min']:
            alerts.append({
                'severity': 'high',
                'metric': 'Availability',
                'value': f"{kpis['availability']:.3f}",
                'threshold': f"{self.alert_thresholds['availability_min']:.3f}",
                'message': f"Availability below threshold: {kpis['availability']:.3f} < {self.alert_thresholds['availability_min']:.3f}"
            })

        if kpis['inverter_efficiency'] < self.alert_thresholds['inverter_efficiency_min']:
            alerts.append({
                'severity': 'medium',
                'metric': 'Inverter Efficiency',
                'value': f"{kpis['inverter_efficiency']:.3f}",
                'threshold': f"{self.alert_thresholds['inverter_efficiency_min']:.3f}",
                'message': f"Inverter efficiency below threshold: {kpis['inverter_efficiency']:.3f} < {self.alert_thresholds['inverter_efficiency_min']:.3f}"
            })

        return alerts

    def benchmark_performance(
        self,
        kpis: Dict[str, float],
        system_type: str = "utility_scale"
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark performance against industry standards.

        Args:
            kpis: Calculated KPIs
            system_type: Type of system (utility_scale, commercial, residential)

        Returns:
            Benchmarking analysis
        """
        # Industry benchmarks
        benchmarks = {
            'utility_scale': {
                'pr': (0.75, 0.85),
                'capacity_factor': (0.15, 0.25),
                'system_efficiency': (0.13, 0.17),
                'inverter_efficiency': (0.96, 0.99),
                'availability': (0.95, 0.99)
            },
            'commercial': {
                'pr': (0.70, 0.82),
                'capacity_factor': (0.13, 0.22),
                'system_efficiency': (0.12, 0.16),
                'inverter_efficiency': (0.95, 0.98),
                'availability': (0.93, 0.98)
            },
            'residential': {
                'pr': (0.65, 0.78),
                'capacity_factor': (0.12, 0.20),
                'system_efficiency': (0.11, 0.15),
                'inverter_efficiency': (0.94, 0.97),
                'availability': (0.90, 0.97)
            }
        }

        benchmark = benchmarks.get(system_type, benchmarks['utility_scale'])

        results = {}

        # PR benchmark
        pr_min, pr_max = benchmark['pr']
        pr_actual = kpis['performance_ratio']
        pr_status = 'excellent' if pr_actual >= pr_max else ('good' if pr_actual >= pr_min else 'below_target')
        results['pr'] = {
            'value': pr_actual,
            'benchmark_min': pr_min,
            'benchmark_max': pr_max,
            'status': pr_status,
            'percentile': ((pr_actual - pr_min) / (pr_max - pr_min)) * 100 if pr_max > pr_min else 0
        }

        # Capacity factor benchmark
        cf_min, cf_max = benchmark['capacity_factor']
        cf_actual = kpis['capacity_factor']
        cf_status = 'excellent' if cf_actual >= cf_max else ('good' if cf_actual >= cf_min else 'below_target')
        results['capacity_factor'] = {
            'value': cf_actual,
            'benchmark_min': cf_min,
            'benchmark_max': cf_max,
            'status': cf_status,
            'percentile': ((cf_actual - cf_min) / (cf_max - cf_min)) * 100 if cf_max > cf_min else 0
        }

        # System efficiency benchmark
        eff_min, eff_max = benchmark['system_efficiency']
        eff_actual = kpis['system_efficiency']
        eff_status = 'excellent' if eff_actual >= eff_max else ('good' if eff_actual >= eff_min else 'below_target')
        results['system_efficiency'] = {
            'value': eff_actual,
            'benchmark_min': eff_min,
            'benchmark_max': eff_max,
            'status': eff_status,
            'percentile': ((eff_actual - eff_min) / (eff_max - eff_min)) * 100 if eff_max > eff_min else 0
        }

        return results

    def analyze_inverter_performance(
        self,
        performance_data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Detailed inverter performance analysis.

        Args:
            performance_data: Performance monitoring data

        Returns:
            Inverter performance analysis
        """
        # Filter data where inverter is operating
        operating_data = performance_data[performance_data['power_dc'] > 0].copy()

        if len(operating_data) == 0:
            return {
                'avg_efficiency': 0,
                'efficiency_at_loads': {},
                'total_losses_kwh': 0,
                'peak_efficiency': 0
            }

        # Average efficiency
        avg_efficiency = operating_data['inverter_efficiency'].mean()
        peak_efficiency = operating_data['inverter_efficiency'].max()

        # Efficiency at different load levels
        operating_data['load_fraction'] = operating_data['power_dc'] / operating_data['power_dc'].max()

        load_bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        efficiency_at_loads = {}

        for load_min, load_max in load_bins:
            mask = (operating_data['load_fraction'] >= load_min) & (operating_data['load_fraction'] < load_max)
            if mask.sum() > 0:
                efficiency_at_loads[f"{load_min*100:.0f}-{load_max*100:.0f}%"] = operating_data.loc[mask, 'inverter_efficiency'].mean()

        # Total losses
        total_dc_energy = performance_data['power_dc'].sum()
        total_ac_energy = performance_data['power_ac'].sum()
        total_losses = total_dc_energy - total_ac_energy

        return {
            'avg_efficiency': avg_efficiency,
            'peak_efficiency': peak_efficiency,
            'efficiency_at_loads': efficiency_at_loads,
            'total_losses_kwh': total_losses,
            'loss_percentage': (total_losses / total_dc_energy * 100) if total_dc_energy > 0 else 0
        }

    def calculate_availability(
        self,
        performance_data: pd.DataFrame,
        expected_operating_hours: int
    ) -> Dict[str, float]:
        """
        Calculate system availability metrics.

        Args:
            performance_data: Performance monitoring data
            expected_operating_hours: Expected operating hours (daylight hours)

        Returns:
            Availability metrics
        """
        # Count hours where system is producing power
        producing_hours = (performance_data['power_ac'] > 0).sum()

        # Total hours in dataset
        total_hours = len(performance_data)

        # Availability
        time_based_availability = producing_hours / total_hours
        energy_based_availability = time_based_availability  # Simplified

        # Downtime
        downtime_hours = total_hours - producing_hours

        # Mean time between failures (MTBF) - simplified
        mtbf = total_hours / max(1, downtime_hours)

        return {
            'time_based_availability': time_based_availability,
            'energy_based_availability': energy_based_availability,
            'producing_hours': producing_hours,
            'downtime_hours': downtime_hours,
            'mtbf_hours': mtbf,
            'uptime_percentage': time_based_availability * 100
        }


def render_performance_monitoring():
    """Render performance monitoring interface in Streamlit."""
    st.header("📊 Real-time Performance Monitoring")
    st.markdown("Comprehensive real-time monitoring and KPI tracking for PV systems.")

    monitor = PerformanceMonitor()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ Real-time Data",
        "📈 KPI Dashboard",
        "🔔 Alerts",
        "📊 Benchmarking",
        "🔌 Inverter Analysis"
    ])

    with tab1:
        st.subheader("Real-time System Data")

        col1, col2 = st.columns(2)

        with col1:
            system_capacity = st.number_input("System DC Capacity (kW):", min_value=1, max_value=10000, value=100, step=10)
            current_irradiance = st.slider("Current Irradiance (W/m²):", 0, 1200, 800, 50)

        with col2:
            module_temp = st.slider("Module Temperature (°C):", 0, 80, 45)
            ambient_temp = st.slider("Ambient Temperature (°C):", -10, 50, 25)

        num_hours = st.slider("Historical Hours to Display:", 1, 168, 24)

        if st.button("🔄 Generate Real-time Data", key="gen_realtime"):
            with st.spinner("Generating monitoring data..."):
                data = monitor.generate_real_time_data(
                    system_capacity,
                    current_irradiance,
                    module_temp,
                    ambient_temp,
                    num_hours=num_hours
                )

            st.success(f"✅ Generated {len(data)} data points")

            # Store in session state
            st.session_state['performance_data'] = data
            st.session_state['system_capacity'] = system_capacity

            # Current values
            latest = data.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("DC Power", f"{latest['power_dc']:.2f} kW")
                st.metric("DC Voltage", f"{latest['voltage_dc']:.1f} V")

            with col2:
                st.metric("AC Power", f"{latest['power_ac']:.2f} kW")
                st.metric("AC Voltage", f"{latest['voltage_ac']:.1f} V")

            with col3:
                st.metric("Irradiance", f"{latest['irradiance']:.0f} W/m²")
                st.metric("Module Temp", f"{latest['temp_module']:.1f} °C")

            with col4:
                st.metric("Inverter Eff.", f"{latest['inverter_efficiency']*100:.2f}%")
                st.metric("Energy Today", f"{data['energy_kwh'].sum():.1f} kWh")

            # Power chart
            st.subheader("Power Output")

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('DC vs AC Power', 'Irradiance', 'Module Temperature', 'Inverter Efficiency')
            )

            # DC vs AC Power
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['power_dc'],
                          name='DC Power', line=dict(color='#3498DB', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['power_ac'],
                          name='AC Power', line=dict(color='#2ECC71', width=2)),
                row=1, col=1
            )

            # Irradiance
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['irradiance'],
                          name='Irradiance', line=dict(color='#F39C12', width=2)),
                row=1, col=2
            )

            # Temperature
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['temp_module'],
                          name='Module Temp', line=dict(color='#E74C3C', width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['temp_ambient'],
                          name='Ambient Temp', line=dict(color='#9B59B6', width=2)),
                row=2, col=1
            )

            # Inverter efficiency
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['inverter_efficiency']*100,
                          name='Inverter Eff', line=dict(color='#1ABC9C', width=2)),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=2)

            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Irradiance (W/m²)", row=1, col=2)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

            fig.update_layout(height=600, showlegend=True, template='plotly_white')

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Performance KPI Dashboard")

        if 'performance_data' not in st.session_state:
            st.warning("⚠️ Please generate real-time data first in the Real-time Data tab")
        else:
            data = st.session_state['performance_data']
            capacity = st.session_state['system_capacity']

            kpis = monitor.calculate_kpis(data, capacity)

            st.session_state['kpis'] = kpis

            # Display KPIs
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Performance Ratio", f"{kpis['performance_ratio']:.3f}")
                st.metric("PR (Temp-Corrected)", f"{kpis['pr_temp_corrected']:.3f}")

            with col2:
                st.metric("Capacity Factor", f"{kpis['capacity_factor']*100:.2f}%")
                st.metric("Specific Yield", f"{kpis['specific_yield']:.2f} kWh/kWp/day")

            with col3:
                st.metric("System Efficiency", f"{kpis['system_efficiency']*100:.2f}%")
                st.metric("Inverter Efficiency", f"{kpis['inverter_efficiency']*100:.2f}%")

            with col4:
                st.metric("Availability", f"{kpis['availability']*100:.2f}%")
                st.metric("DC/AC Ratio", f"{kpis['dc_ac_ratio']:.3f}")

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Energy", f"{kpis['total_energy_kwh']:.1f} kWh")

            with col2:
                st.metric("Daily Average", f"{kpis['daily_energy_kwh']:.1f} kWh")

            with col3:
                st.metric("Peak Power", f"{kpis['peak_power_ac_kw']:.2f} kW")

            # KPI visualization
            st.subheader("KPI Summary")

            kpi_metrics = ['performance_ratio', 'capacity_factor', 'system_efficiency', 'inverter_efficiency', 'availability']
            kpi_labels = ['PR', 'Capacity Factor', 'System Eff.', 'Inverter Eff.', 'Availability']
            kpi_values = [kpis[k] * 100 if k != 'performance_ratio' else kpis[k] * 100 for k in kpi_metrics]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=kpi_labels,
                y=kpi_values,
                marker_color=['#2ECC71', '#3498DB', '#9B59B6', '#F39C12', '#E74C3C'],
                text=[f"{v:.1f}%" for v in kpi_values],
                textposition='auto'
            ))

            fig.update_layout(
                title="Key Performance Indicators",
                xaxis_title="KPI",
                yaxis_title="Value (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Energy breakdown
            st.subheader("Energy Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Daily energy profile
                data['date'] = data['timestamp'].dt.date
                daily_energy = data.groupby('date')['energy_kwh'].sum()

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=list(daily_energy.index),
                    y=daily_energy.values,
                    marker_color='#2ECC71',
                    name='Daily Energy'
                ))

                fig.update_layout(
                    title="Daily Energy Production",
                    xaxis_title="Date",
                    yaxis_title="Energy (kWh)",
                    height=300,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Power distribution
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=data['power_ac'],
                    nbinsx=30,
                    marker_color='#3498DB',
                    name='AC Power'
                ))

                fig.update_layout(
                    title="AC Power Distribution",
                    xaxis_title="Power (kW)",
                    yaxis_title="Frequency",
                    height=300,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Performance Alerts & Threshold Monitoring")

        if 'kpis' not in st.session_state:
            st.warning("⚠️ Please calculate KPIs first in the KPI Dashboard tab")
        else:
            st.write("### Alert Thresholds")

            col1, col2 = st.columns(2)

            with col1:
                pr_threshold = st.slider("PR Minimum:", 0.5, 0.9, 0.75, 0.05)
                eff_threshold = st.slider("System Efficiency Minimum:", 0.05, 0.20, 0.10, 0.01)

            with col2:
                avail_threshold = st.slider("Availability Minimum:", 0.85, 0.99, 0.95, 0.01)
                inv_eff_threshold = st.slider("Inverter Efficiency Minimum:", 0.85, 0.99, 0.92, 0.01)

            monitor.alert_thresholds = {
                'pr_min': pr_threshold,
                'efficiency_min': eff_threshold,
                'availability_min': avail_threshold,
                'inverter_efficiency_min': inv_eff_threshold
            }

            alerts = monitor.check_alerts(st.session_state['kpis'])

            if len(alerts) == 0:
                st.success("✅ All metrics within acceptable ranges - No alerts")
            else:
                st.warning(f"⚠️ {len(alerts)} alert(s) detected")

                for alert in alerts:
                    severity_colors = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }

                    severity_icon = severity_colors.get(alert['severity'], '⚪')

                    st.error(f"{severity_icon} **{alert['metric']}**: {alert['message']}")

            # Alert history visualization
            st.subheader("Alert History")

            alert_data = {
                'Metric': [a['metric'] for a in alerts],
                'Severity': [a['severity'] for a in alerts],
                'Current Value': [a['value'] for a in alerts],
                'Threshold': [a['threshold'] for a in alerts]
            }

            if len(alerts) > 0:
                st.dataframe(pd.DataFrame(alert_data), use_container_width=True)

                # Alert severity distribution
                severity_counts = pd.Series([a['severity'] for a in alerts]).value_counts()

                fig = go.Figure()

                fig.add_trace(go.Pie(
                    labels=severity_counts.index,
                    values=severity_counts.values,
                    marker=dict(colors=['#E74C3C', '#F39C12', '#2ECC71'])
                ))

                fig.update_layout(
                    title="Alert Severity Distribution",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Performance Benchmarking")

        if 'kpis' not in st.session_state:
            st.warning("⚠️ Please calculate KPIs first in the KPI Dashboard tab")
        else:
            system_type = st.selectbox(
                "System Type:",
                ["utility_scale", "commercial", "residential"]
            )

            benchmark_results = monitor.benchmark_performance(
                st.session_state['kpis'],
                system_type
            )

            st.write("### Benchmark Comparison")

            for metric_name, metric_data in benchmark_results.items():
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Create gauge chart
                    fig = go.Figure()

                    fig.add_trace(go.Indicator(
                        mode="gauge+number+delta",
                        value=metric_data['value'] * 100,
                        title={'text': metric_name.replace('_', ' ').title()},
                        delta={'reference': metric_data['benchmark_max'] * 100},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2ECC71"},
                            'steps': [
                                {'range': [0, metric_data['benchmark_min'] * 100], 'color': "#E74C3C"},
                                {'range': [metric_data['benchmark_min'] * 100, metric_data['benchmark_max'] * 100], 'color': "#F39C12"},
                                {'range': [metric_data['benchmark_max'] * 100, 100], 'color': "#2ECC71"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': metric_data['benchmark_max'] * 100
                            }
                        }
                    ))

                    fig.update_layout(height=250, template='plotly_white')

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Status", metric_data['status'].replace('_', ' ').title())
                    st.metric("Percentile", f"{metric_data['percentile']:.1f}%")
                    st.metric("Benchmark Range",
                             f"{metric_data['benchmark_min']*100:.1f}% - {metric_data['benchmark_max']*100:.1f}%")

                st.divider()

    with tab5:
        st.subheader("Inverter Performance Analysis")

        if 'performance_data' not in st.session_state:
            st.warning("⚠️ Please generate performance data first")
        else:
            data = st.session_state['performance_data']

            inv_analysis = monitor.analyze_inverter_performance(data)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Efficiency", f"{inv_analysis['avg_efficiency']*100:.2f}%")

            with col2:
                st.metric("Peak Efficiency", f"{inv_analysis['peak_efficiency']*100:.2f}%")

            with col3:
                st.metric("Total Losses", f"{inv_analysis['total_losses_kwh']:.2f} kWh")

            # Efficiency at different loads
            st.subheader("Efficiency vs Load")

            if inv_analysis['efficiency_at_loads']:
                load_labels = list(inv_analysis['efficiency_at_loads'].keys())
                load_values = [v * 100 for v in inv_analysis['efficiency_at_loads'].values()]

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=load_labels,
                    y=load_values,
                    marker_color='#3498DB',
                    text=[f"{v:.2f}%" for v in load_values],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Inverter Efficiency at Different Load Levels",
                    xaxis_title="Load Level (%)",
                    yaxis_title="Efficiency (%)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Availability analysis
            st.subheader("System Availability")

            expected_hours = len(data)
            availability = monitor.calculate_availability(data, expected_hours)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Time-Based Availability", f"{availability['time_based_availability']*100:.2f}%")

            with col2:
                st.metric("Producing Hours", f"{availability['producing_hours']} hrs")

            with col3:
                st.metric("Downtime", f"{availability['downtime_hours']} hrs")

            # Availability pie chart
            fig = go.Figure()

            fig.add_trace(go.Pie(
                labels=['Producing', 'Downtime'],
                values=[availability['producing_hours'], availability['downtime_hours']],
                marker=dict(colors=['#2ECC71', '#E74C3C'])
            ))

            fig.update_layout(
                title="System Availability Breakdown",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.info("💡 **Real-time Performance Monitoring** - Branch B07 | Comprehensive System Monitoring & KPI Tracking")
