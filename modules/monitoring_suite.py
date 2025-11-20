"""
Monitoring Suite Module - Branches B07-B09

This module provides functionality for:
- B07: Real-time Performance Monitoring
- B08: Fault Detection & Diagnostics
- B09: Energy Forecasting (ML-based)

Author: PV Circularity Simulator Team
Version: 1.0 (71 Sessions Integrated)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    FAULT_TYPES,
    SENSOR_TYPES,
    PERFORMANCE_THRESHOLDS,
    COLOR_PALETTE,
)
from utils.validators import (
    PerformanceData,
    FaultDetection,
    EnergyForecast,
)


# ============================================================================
# BRANCH 07: REAL-TIME PERFORMANCE MONITORING
# ============================================================================

def render_performance_monitoring() -> None:
    """
    Render the Real-time Performance Monitoring interface.

    Features:
    - Live system metrics
    - SCADA integration display
    - Performance ratio calculation
    - String-level monitoring
    - Alarm management
    - Historical trending
    """
    st.header("📊 Real-time Performance Monitoring")
    st.markdown("*SCADA integration and live system analytics*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Live Dashboard",
        "📈 Historical Trends",
        "⚡ String Monitoring",
        "🔔 Alarms & Events"
    ])

    # Tab 1: Live Dashboard
    with tab1:
        st.subheader("Live System Performance")

        # Simulate real-time data
        current_time = datetime.now()

        # Generate current performance data
        irradiance = max(0, 1000 * np.sin(np.pi * (current_time.hour - 6) / 12) ** 1.5) if 6 <= current_time.hour <= 18 else 0
        module_temp = 25 + irradiance / 40 + np.random.normal(0, 2)
        ambient_temp = 25 + np.random.normal(0, 3)
        wind_speed = max(0, 3 + np.random.normal(0, 1.5))

        dc_power = irradiance * 0.095 + np.random.normal(0, 0.5)  # Simplified
        ac_power = dc_power * 0.98

        # Display current metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Current Power",
                f"{ac_power:.2f} kW",
                delta=f"{np.random.normal(0.15, 0.05):.2f} kW"
            )

        with col2:
            energy_today = ac_power * 6.5  # Simplified
            st.metric("Today's Yield", f"{energy_today:.1f} kWh")

        with col3:
            efficiency = ac_power / (irradiance * 0.1 + 0.001) if irradiance > 0 else 0
            st.metric("System Efficiency", f"{min(efficiency * 100, 18):.1f}%")

        with col4:
            pr = (ac_power / (irradiance / 1000 * 100)) if irradiance > 100 else 0
            pr = min(pr, 1.0)
            st.metric("Performance Ratio", f"{pr:.1%}", delta="Normal")

        with col5:
            st.metric("System Health", "98.5%", delta="+0.5%")

        st.divider()

        # Environmental conditions
        st.subheader("Environmental Conditions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Irradiance", f"{irradiance:.0f} W/m²")
        with col2:
            st.metric("Module Temp", f"{module_temp:.1f}°C")
        with col3:
            st.metric("Ambient Temp", f"{ambient_temp:.1f}°C")
        with col4:
            st.metric("Wind Speed", f"{wind_speed:.1f} m/s")

        # Validate performance data
        try:
            perf_data = PerformanceData(
                timestamp=current_time,
                dc_power_kw=dc_power,
                ac_power_kw=ac_power,
                irradiance_w_m2=irradiance,
                module_temp_c=module_temp,
                ambient_temp_c=ambient_temp,
                wind_speed_ms=wind_speed,
                energy_today_kwh=energy_today,
                performance_ratio=pr
            )

            # Display inverter efficiency
            st.info(f"**Inverter Efficiency:** {perf_data.inverter_efficiency:.2%}")

        except Exception as e:
            st.error(f"Data validation error: {str(e)}")

        # Real-time power gauge
        st.subheader("Current Power Output")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ac_power,
            delta={'reference': 9.0, 'increasing': {'color': COLOR_PALETTE['success']}},
            gauge={
                'axis': {'range': [None, 12]},
                'bar': {'color': COLOR_PALETTE['primary']},
                'steps': [
                    {'range': [0, 6], 'color': COLOR_PALETTE['light']},
                    {'range': [6, 10], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': COLOR_PALETTE['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 11
                }
            },
            title={'text': "AC Power (kW)"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Intra-day power curve
        st.subheader("Today's Power Profile")

        hours = np.arange(0, current_time.hour + 1, 0.5)
        power_profile = []

        for hour in hours:
            hour_irr = max(0, 1000 * np.sin(np.pi * (hour - 6) / 12) ** 1.5) if 6 <= hour <= 18 else 0
            hour_power = hour_irr * 0.095 + np.random.normal(0, 0.3)
            power_profile.append(max(0, hour_power))

        fig_today = go.Figure()
        fig_today.add_trace(go.Scatter(
            x=hours,
            y=power_profile,
            mode='lines',
            fill='tozeroy',
            line=dict(color=COLOR_PALETTE['primary'], width=2),
            name='AC Power'
        ))
        fig_today.update_layout(
            title="Power Generation Profile (Today)",
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            height=400
        )
        st.plotly_chart(fig_today, use_container_width=True)

    # Tab 2: Historical Trends
    with tab2:
        st.subheader("Historical Performance Trends")

        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )

        # Generate historical data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_energy = 45 + 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365) + np.random.normal(0, 5, len(date_range))
        daily_pr = 0.82 + 0.05 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365) + np.random.normal(0, 0.02, len(date_range))
        daily_pr = np.clip(daily_pr, 0.65, 0.95)

        hist_df = pd.DataFrame({
            'Date': date_range,
            'Energy (kWh)': daily_energy,
            'Performance Ratio': daily_pr,
            'Peak Power (kW)': 9.5 + np.random.normal(0, 0.5, len(date_range)),
            'Avg Irradiance (W/m²)': 600 + 200 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365) + np.random.normal(0, 50, len(date_range))
        })

        # Multi-parameter chart
        metric_to_plot = st.selectbox(
            "Select Metric",
            ['Energy (kWh)', 'Performance Ratio', 'Peak Power (kW)', 'Avg Irradiance (W/m²)']
        )

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df['Date'],
            y=hist_df[metric_to_plot],
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE['primary'], width=2),
            name=metric_to_plot
        ))

        # Add moving average
        window = 7
        if len(hist_df) >= window:
            ma = hist_df[metric_to_plot].rolling(window=window).mean()
            fig_hist.add_trace(go.Scatter(
                x=hist_df['Date'],
                y=ma,
                mode='lines',
                line=dict(color=COLOR_PALETTE['danger'], width=2, dash='dash'),
                name=f'{window}-Day Moving Average'
            ))

        fig_hist.update_layout(
            title=f"Historical {metric_to_plot}",
            xaxis_title="Date",
            yaxis_title=metric_to_plot,
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Energy", f"{hist_df['Energy (kWh)'].sum():,.0f} kWh")
        with col2:
            st.metric("Avg PR", f"{hist_df['Performance Ratio'].mean():.2%}")
        with col3:
            st.metric("Peak Day", f"{hist_df['Energy (kWh)'].max():.1f} kWh")
        with col4:
            st.metric("Avg Daily", f"{hist_df['Energy (kWh)'].mean():.1f} kWh")

        # Performance distribution
        col1, col2 = st.columns(2)

        with col1:
            fig_dist = go.Figure(data=[go.Histogram(
                x=hist_df['Performance Ratio'],
                nbinsx=20,
                marker_color=COLOR_PALETTE['secondary']
            )])
            fig_dist.update_layout(
                title="Performance Ratio Distribution",
                xaxis_title="Performance Ratio",
                yaxis_title="Frequency",
                height=350
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            fig_box = go.Figure(data=[go.Box(
                y=hist_df['Energy (kWh)'],
                marker_color=COLOR_PALETTE['success'],
                name='Daily Energy'
            )])
            fig_box.update_layout(
                title="Daily Energy Box Plot",
                yaxis_title="Energy (kWh)",
                height=350
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Tab 3: String Monitoring
    with tab3:
        st.subheader("String-Level Performance Analysis")

        # Generate string data
        num_strings = st.slider("Number of Strings", 3, 20, 10)

        string_data = []
        for i in range(1, num_strings + 1):
            base_current = 10.5
            current = base_current + np.random.normal(0, 0.3)

            # Simulate some string issues
            if i == 5:
                current *= 0.75  # Underperforming string
            if i == 12 and num_strings >= 12:
                current *= 0.5  # Fault

            string_data.append({
                'String': f'String {i}',
                'Current (A)': current,
                'Voltage (V)': 450 + np.random.normal(0, 10),
                'Power (kW)': current * 450 / 1000,
                'Status': 'Normal' if current > base_current * 0.9 else 'Warning' if current > base_current * 0.6 else 'Fault'
            })

        string_df = pd.DataFrame(string_data)

        # String current comparison
        fig_strings = go.Figure()

        colors = [
            COLOR_PALETTE['success'] if status == 'Normal' else
            COLOR_PALETTE['warning'] if status == 'Warning' else
            COLOR_PALETTE['danger']
            for status in string_df['Status']
        ]

        fig_strings.add_trace(go.Bar(
            x=string_df['String'],
            y=string_df['Current (A)'],
            marker_color=colors,
            text=string_df['Current (A)'].round(2),
            textposition='outside'
        ))

        # Add expected current line
        fig_strings.add_hline(
            y=10.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Expected Current"
        )

        fig_strings.update_layout(
            title="String Current Comparison",
            xaxis_title="String",
            yaxis_title="Current (A)",
            height=500
        )
        st.plotly_chart(fig_strings, use_container_width=True)

        # String status summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            normal_count = len(string_df[string_df['Status'] == 'Normal'])
            st.metric("Normal Strings", normal_count, delta=f"{normal_count/num_strings:.0%}")
        with col2:
            warning_count = len(string_df[string_df['Status'] == 'Warning'])
            st.metric("Warning", warning_count, delta=f"-{warning_count}", delta_color="inverse")
        with col3:
            fault_count = len(string_df[string_df['Status'] == 'Fault'])
            st.metric("Faulted", fault_count, delta=f"-{fault_count}", delta_color="inverse")
        with col4:
            total_power = string_df['Power (kW)'].sum()
            st.metric("Total String Power", f"{total_power:.2f} kW")

        # Detailed string table
        st.markdown("**Detailed String Data:**")

        def highlight_status(row):
            if row['Status'] == 'Fault':
                return [f'background-color: {COLOR_PALETTE["danger"]}40'] * len(row)
            elif row['Status'] == 'Warning':
                return [f'background-color: {COLOR_PALETTE["warning"]}40'] * len(row)
            else:
                return [''] * len(row)

        styled_string_df = string_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_string_df, use_container_width=True)

    # Tab 4: Alarms & Events
    with tab4:
        st.subheader("System Alarms & Events")

        # Generate alarm data
        alarm_data = [
            {
                'Timestamp': datetime.now() - timedelta(hours=2),
                'Severity': 'High',
                'Type': 'String Fault',
                'Description': 'String 12 current below 50% of expected',
                'Status': 'Active',
                'Location': 'String 12'
            },
            {
                'Timestamp': datetime.now() - timedelta(hours=5),
                'Severity': 'Medium',
                'Type': 'Performance',
                'Description': 'Performance ratio below 75%',
                'Status': 'Acknowledged',
                'Location': 'System-wide'
            },
            {
                'Timestamp': datetime.now() - timedelta(hours=12),
                'Severity': 'Low',
                'Type': 'Communication',
                'Description': 'Inverter 2 communication timeout',
                'Status': 'Cleared',
                'Location': 'Inverter 2'
            },
            {
                'Timestamp': datetime.now() - timedelta(days=1),
                'Severity': 'High',
                'Type': 'Grid',
                'Description': 'Grid voltage out of range',
                'Status': 'Cleared',
                'Location': 'Grid Connection'
            },
        ]

        alarm_df = pd.DataFrame(alarm_data)

        # Filter alarms
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                ['High', 'Medium', 'Low'],
                default=['High', 'Medium', 'Low']
            )
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                ['Active', 'Acknowledged', 'Cleared'],
                default=['Active', 'Acknowledged']
            )
        with col3:
            type_filter = st.multiselect(
                "Filter by Type",
                alarm_df['Type'].unique(),
                default=alarm_df['Type'].unique()
            )

        # Apply filters
        filtered_alarms = alarm_df[
            (alarm_df['Severity'].isin(severity_filter)) &
            (alarm_df['Status'].isin(status_filter)) &
            (alarm_df['Type'].isin(type_filter))
        ]

        # Display alarms
        st.markdown(f"**Active Alarms ({len(filtered_alarms)} of {len(alarm_df)}):**")

        for idx, alarm in filtered_alarms.iterrows():
            severity_color = {
                'High': 'error',
                'Medium': 'warning',
                'Low': 'info'
            }[alarm['Severity']]

            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{alarm['Type']}**: {alarm['Description']}")
                    st.caption(f"{alarm['Timestamp'].strftime('%Y-%m-%d %H:%M')} | {alarm['Location']}")
                with col2:
                    if severity_color == 'error':
                        st.error(alarm['Severity'])
                    elif severity_color == 'warning':
                        st.warning(alarm['Severity'])
                    else:
                        st.info(alarm['Severity'])
                with col3:
                    st.write(f"Status: {alarm['Status']}")

                st.divider()

        # Alarm statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            active_alarms = len(alarm_df[alarm_df['Status'] == 'Active'])
            st.metric("Active Alarms", active_alarms)
        with col2:
            high_severity = len(alarm_df[alarm_df['Severity'] == 'High'])
            st.metric("High Severity", high_severity)
        with col3:
            avg_resolution_time = 4.5  # hours (simulated)
            st.metric("Avg Resolution", f"{avg_resolution_time:.1f} hrs")
        with col4:
            mtbf = 120  # hours (simulated)
            st.metric("MTBF", f"{mtbf} hrs")


# ============================================================================
# BRANCH 08: FAULT DETECTION & DIAGNOSTICS
# ============================================================================

def render_fault_diagnostics() -> None:
    """
    Render the Fault Detection & Diagnostics interface.

    Features:
    - AI-powered defect detection
    - IR thermography analysis
    - IV curve diagnostics
    - EL imaging interpretation
    - Root cause analysis
    - Maintenance recommendations
    """
    st.header("🔍 Fault Detection & Diagnostics")
    st.markdown("*AI-powered defect detection with multi-modal analysis*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Active Faults",
        "🌡️ Thermal Analysis",
        "📉 IV Curve Analysis",
        "💡 Recommendations"
    ])

    # Tab 1: Active Faults
    with tab1:
        st.subheader("Detected Faults & Issues")

        # Display all fault types with simulated detections
        detected_faults = []

        # Simulate some detected faults
        fault_instances = [
            ('Hot Spot', 'String 2, Module 15', 2),
            ('Soiling', 'System-wide', 0.5),
            ('Cell Crack', 'String 5, Module 8', 1),
        ]

        for fault_name, location, days_ago in fault_instances:
            fault_info = FAULT_TYPES[fault_name]

            fault_obj = FaultDetection(
                fault_type=fault_name,
                severity=fault_info['severity'],
                detection_method=fault_info['detection_method'],
                location=location,
                timestamp=datetime.now() - timedelta(days=days_ago),
                power_loss_pct=fault_info['power_loss_pct'],
                recommended_action=fault_info['action'],
                status='Open' if days_ago < 1 else 'In Progress'
            )

            detected_faults.append(fault_obj)

        # Display faults
        for fault in detected_faults:
            severity_color = {
                'Low': 'info',
                'Medium': 'warning',
                'High': 'error',
                'Critical': 'error'
            }.get(fault.severity, 'info')

            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**{fault.fault_type}** ({fault.location})")
                    st.caption(f"Detected: {fault.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Method: {fault.detection_method}")

                with col2:
                    if severity_color == 'error':
                        st.error(f"{fault.severity} Severity")
                    elif severity_color == 'warning':
                        st.warning(f"{fault.severity} Severity")
                    else:
                        st.info(f"{fault.severity} Severity")

                    st.metric("Power Loss", f"{fault.power_loss_pct}%")

                with col3:
                    st.write(f"**Action Required:**")
                    st.write(fault.recommended_action)
                    st.write(f"Status: **{fault.status}**")

                st.divider()

        # Fault summary
        st.subheader("Fault Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Faults", len(detected_faults))
        with col2:
            high_severity = len([f for f in detected_faults if f.severity in ['High', 'Critical']])
            st.metric("High/Critical", high_severity)
        with col3:
            total_power_loss = sum(f.power_loss_pct for f in detected_faults)
            st.metric("Total Power Loss", f"{total_power_loss:.1f}%")
        with col4:
            open_faults = len([f for f in detected_faults if f.status == 'Open'])
            st.metric("Open Faults", open_faults)

        # Fault distribution
        fault_counts = {}
        for fault_type in FAULT_TYPES.keys():
            count = len([f for f in detected_faults if f.fault_type == fault_type])
            if count > 0:
                fault_counts[fault_type] = count

        if fault_counts:
            fig_faults = go.Figure(data=[go.Pie(
                labels=list(fault_counts.keys()),
                values=list(fault_counts.values()),
                hole=.3
            )])
            fig_faults.update_layout(
                title="Fault Type Distribution",
                height=400
            )
            st.plotly_chart(fig_faults, use_container_width=True)

    # Tab 2: Thermal Analysis
    with tab2:
        st.subheader("IR Thermography Analysis")

        st.markdown("""
        **Thermal imaging detects:**
        - Hot spots (cell-level failures)
        - Bypass diode failures
        - Connection issues
        - Delamination
        """)

        # Simulate thermal image data
        module_rows = st.slider("Module Rows", 3, 10, 6)
        module_cols = st.slider("Module Columns", 3, 15, 10)

        # Generate synthetic thermal map
        thermal_data = np.random.normal(45, 5, (module_rows, module_cols))

        # Add some hot spots
        thermal_data[2, 5] = 75  # Hot spot
        thermal_data[4, 3] = 70  # Another hot spot

        fig_thermal = go.Figure(data=go.Heatmap(
            z=thermal_data,
            colorscale='RdYlGn_r',
            text=thermal_data.round(1),
            texttemplate='%{text}°C',
            textfont={"size": 10},
            colorbar=dict(title="Temperature (°C)")
        ))

        fig_thermal.update_layout(
            title="Module Thermal Map (IR Imaging)",
            xaxis_title="Module Column",
            yaxis_title="Module Row",
            height=500
        )

        st.plotly_chart(fig_thermal, use_container_width=True)

        # Temperature statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Temp", f"{thermal_data.max():.1f}°C")
        with col2:
            st.metric("Avg Temp", f"{thermal_data.mean():.1f}°C")
        with col3:
            st.metric("Std Dev", f"{thermal_data.std():.1f}°C")
        with col4:
            hot_spots = np.sum(thermal_data > 65)
            st.metric("Hot Spots (>65°C)", hot_spots)

        if hot_spots > 0:
            st.warning(f"⚠️ {hot_spots} potential hot spots detected. Immediate inspection recommended.")

    # Tab 3: IV Curve Analysis
    with tab3:
        st.subheader("IV Curve Diagnostics")

        # Generate normal and faulty IV curves
        voltage = np.linspace(0, 50, 100)
        current_normal = 11 * (1 - np.exp((voltage - 49) / 3.9))
        current_normal = np.maximum(current_normal, 0)

        # Faulty curves
        current_bypass_fail = current_normal.copy()
        current_bypass_fail[voltage > 25] *= 0.65  # Bypass diode failure

        current_shading = current_normal * 0.75  # Shading

        current_soiling = current_normal * 0.95  # Soiling (slight reduction)

        fig_iv = go.Figure()

        fig_iv.add_trace(go.Scatter(
            x=voltage,
            y=current_normal,
            mode='lines',
            name='Normal (Reference)',
            line=dict(color=COLOR_PALETTE['success'], width=3)
        ))

        fig_iv.add_trace(go.Scatter(
            x=voltage,
            y=current_bypass_fail,
            mode='lines',
            name='Bypass Diode Failure',
            line=dict(color=COLOR_PALETTE['danger'], width=2, dash='dash')
        ))

        fig_iv.add_trace(go.Scatter(
            x=voltage,
            y=current_shading,
            mode='lines',
            name='Shading',
            line=dict(color=COLOR_PALETTE['warning'], width=2, dash='dot')
        ))

        fig_iv.add_trace(go.Scatter(
            x=voltage,
            y=current_soiling,
            mode='lines',
            name='Soiling',
            line=dict(color=COLOR_PALETTE['info'], width=2)
        ))

        fig_iv.update_layout(
            title="IV Curve Comparison - Fault Signatures",
            xaxis_title="Voltage (V)",
            yaxis_title="Current (A)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_iv, use_container_width=True)

        # IV curve interpretation
        st.markdown("""
        **IV Curve Fault Signatures:**
        - **Bypass Diode Failure**: Step reduction in current at mid-voltage
        - **Shading**: Overall parallel shift downward
        - **Soiling**: Slight reduction in Isc with minimal Voc change
        - **Series Resistance**: Decreased fill factor, reduced slope
        - **Shunt Resistance**: Increased slope near Voc
        """)

    # Tab 4: Recommendations
    with tab4:
        st.subheader("Maintenance Recommendations")

        # Priority-based recommendations
        recommendations = [
            {
                'Priority': 'Critical',
                'Issue': 'Hot Spot Detected (String 2, Module 15)',
                'Action': 'Immediate inspection and potential module replacement',
                'Timeline': 'Within 24 hours',
                'Cost': '$500-800',
                'Impact': 'Prevents fire hazard, restores 15% power loss'
            },
            {
                'Priority': 'High',
                'Issue': 'String 12 Current Deviation',
                'Action': 'Check connections, inspect for shading/soiling',
                'Timeline': 'Within 1 week',
                'Cost': '$200-400',
                'Impact': 'Restores 8-10% string power'
            },
            {
                'Priority': 'Medium',
                'Issue': 'System-wide Soiling Loss',
                'Action': 'Schedule cleaning of all modules',
                'Timeline': 'Within 2 weeks',
                'Cost': '$300-500',
                'Impact': 'Improves system output by 3-5%'
            },
            {
                'Priority': 'Low',
                'Issue': 'Cell Crack Detection',
                'Action': 'Monitor degradation trend, plan future replacement',
                'Timeline': 'Within 3 months',
                'Cost': '$600-1000',
                'Impact': 'Prevents future 5-10% module degradation'
            },
        ]

        for rec in recommendations:
            priority_color = {
                'Critical': 'error',
                'High': 'warning',
                'Medium': 'info',
                'Low': 'success'
            }[rec['Priority']]

            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{rec['Issue']}**")
                    st.write(f"**Action:** {rec['Action']}")
                    st.caption(f"Impact: {rec['Impact']}")

                with col2:
                    if priority_color == 'error':
                        st.error(rec['Priority'])
                    elif priority_color == 'warning':
                        st.warning(rec['Priority'])
                    else:
                        st.info(rec['Priority'])

                    st.metric("Timeline", rec['Timeline'])
                    st.metric("Est. Cost", rec['Cost'])

                st.divider()


# ============================================================================
# BRANCH 09: ENERGY FORECASTING (ML-BASED)
# ============================================================================

def render_energy_forecasting() -> None:
    """
    Render the Energy Forecasting interface.

    Features:
    - Short-term forecasting (1-7 days)
    - Long-term forecasting (monthly/annual)
    - ML ensemble models (Prophet + LSTM)
    - Weather integration
    - Uncertainty quantification
    - Forecast accuracy tracking
    """
    st.header("🔮 Energy Forecasting & Prediction")
    st.markdown("*ML-powered energy yield forecasting with uncertainty bounds*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Short-term Forecast",
        "📆 Long-term Forecast",
        "🎯 Model Performance",
        "⚙️ Forecast Configuration"
    ])

    # Tab 1: Short-term Forecast
    with tab1:
        st.subheader("7-Day Energy Forecast")

        # Generate forecast data
        forecast_days = 7
        forecast_dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')

        base_yield = 45
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)

        forecasts = []
        for i, date in enumerate(forecast_dates):
            forecast_kwh = base_yield * seasonal_factor + np.random.normal(0, 3)
            ci_width = 5 + i * 0.5  # Uncertainty increases with forecast horizon

            forecast = EnergyForecast(
                forecast_date=date,
                forecast_kwh=forecast_kwh,
                confidence_interval_lower=forecast_kwh - ci_width,
                confidence_interval_upper=forecast_kwh + ci_width,
                model_type="Ensemble (Prophet + LSTM)",
                irradiance_forecast_w_m2=600 + np.random.normal(0, 50),
                temp_forecast_c=25 + np.random.normal(0, 3)
            )
            forecasts.append(forecast)

        # Display forecast chart
        fig_forecast = go.Figure()

        # Forecast line
        fig_forecast.add_trace(go.Scatter(
            x=[f.forecast_date for f in forecasts],
            y=[f.forecast_kwh for f in forecasts],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=COLOR_PALETTE['primary'], width=3)
        ))

        # Confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=[f.forecast_date for f in forecasts],
            y=[f.confidence_interval_upper for f in forecasts],
            mode='lines',
            name='Upper CI (95%)',
            line=dict(width=0),
            showlegend=False
        ))

        fig_forecast.add_trace(go.Scatter(
            x=[f.forecast_date for f in forecasts],
            y=[f.confidence_interval_lower for f in forecasts],
            mode='lines',
            name='Lower CI (95%)',
            line=dict(width=0),
            fillcolor='rgba(46, 204, 113, 0.2)',
            fill='tonexty',
            showlegend=False
        ))

        fig_forecast.update_layout(
            title="7-Day Energy Production Forecast",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast table
        st.markdown("**Detailed Forecast:**")

        forecast_df = pd.DataFrame([
            {
                'Date': f.forecast_date.strftime('%Y-%m-%d'),
                'Day': f.forecast_date.strftime('%A'),
                'Forecast (kWh)': f"{f.forecast_kwh:.1f}",
                '95% CI': f"{f.confidence_interval_lower:.1f} - {f.confidence_interval_upper:.1f}",
                'Irradiance (W/m²)': f"{f.irradiance_forecast_w_m2:.0f}",
                'Temp (°C)': f"{f.temp_forecast_c:.1f}"
            }
            for f in forecasts
        ])

        st.dataframe(forecast_df, use_container_width=True)

        # Weekly summary
        col1, col2, col3 = st.columns(3)
        with col1:
            week_total = sum(f.forecast_kwh for f in forecasts)
            st.metric("Week Total", f"{week_total:.0f} kWh")
        with col2:
            daily_avg = week_total / 7
            st.metric("Daily Average", f"{daily_avg:.1f} kWh")
        with col3:
            best_day = max(forecasts, key=lambda f: f.forecast_kwh)
            st.metric("Best Day", best_day.forecast_date.strftime('%A'))

    # Tab 2: Long-term Forecast
    with tab2:
        st.subheader("Monthly & Annual Forecast")

        # Monthly forecast
        months_ahead = 12
        month_names = [(datetime.now() + timedelta(days=30*i)).strftime('%b %Y') for i in range(months_ahead)]

        base_monthly = 1350
        monthly_forecasts = []

        for i in range(months_ahead):
            month_date = datetime.now() + timedelta(days=30*i)
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * month_date.timetuple().tm_yday / 365)
            monthly_yield = base_monthly * seasonal + np.random.normal(0, 50)
            monthly_forecasts.append(monthly_yield)

        fig_monthly = go.Figure()

        fig_monthly.add_trace(go.Bar(
            x=month_names,
            y=monthly_forecasts,
            marker_color=COLOR_PALETTE['secondary'],
            name='Monthly Forecast'
        ))

        fig_monthly.update_layout(
            title="12-Month Energy Forecast",
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            height=500
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

        # Annual projection
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            annual_total = sum(monthly_forecasts)
            st.metric("Annual Forecast", f"{annual_total:,.0f} kWh")
        with col2:
            monthly_avg = annual_total / 12
            st.metric("Monthly Avg", f"{monthly_avg:,.0f} kWh")
        with col3:
            best_month = month_names[np.argmax(monthly_forecasts)]
            st.metric("Best Month", best_month)
        with col4:
            worst_month = month_names[np.argmin(monthly_forecasts)]
            st.metric("Lowest Month", worst_month)

    # Tab 3: Model Performance
    with tab3:
        st.subheader("Forecast Model Performance")

        # Simulate historical forecast vs actual
        past_days = 30
        past_dates = pd.date_range(end=datetime.now(), periods=past_days, freq='D')

        actual_values = 45 + 10 * np.sin(2 * np.pi * np.arange(past_days) / 365) + np.random.normal(0, 4, past_days)
        forecast_values = actual_values + np.random.normal(0, 3, past_days)

        fig_performance = go.Figure()

        fig_performance.add_trace(go.Scatter(
            x=past_dates,
            y=actual_values,
            mode='lines+markers',
            name='Actual',
            line=dict(color=COLOR_PALETTE['success'], width=2)
        ))

        fig_performance.add_trace(go.Scatter(
            x=past_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color=COLOR_PALETTE['primary'], width=2, dash='dash')
        ))

        fig_performance.update_layout(
            title="Forecast vs Actual (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_performance, use_container_width=True)

        # Accuracy metrics
        mae = np.mean(np.abs(actual_values - forecast_values))
        rmse = np.sqrt(np.mean((actual_values - forecast_values)**2))
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{mae:.2f} kWh")
        with col2:
            st.metric("RMSE", f"{rmse:.2f} kWh")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            accuracy = 100 - mape
            st.metric("Accuracy", f"{accuracy:.1f}%")

        # Error distribution
        errors = actual_values - forecast_values

        fig_error = go.Figure(data=[go.Histogram(
            x=errors,
            nbinsx=20,
            marker_color=COLOR_PALETTE['info']
        )])

        fig_error.update_layout(
            title="Forecast Error Distribution",
            xaxis_title="Error (kWh)",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig_error, use_container_width=True)

    # Tab 4: Forecast Configuration
    with tab4:
        st.subheader("Forecast Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Selection**")
            primary_model = st.selectbox(
                "Primary Model",
                ["Prophet", "LSTM", "Ensemble", "XGBoost", "Random Forest"]
            )

            if primary_model == "Ensemble":
                st.info("Ensemble combines Prophet (time series) + LSTM (deep learning)")

            forecast_horizon = st.slider("Default Forecast Horizon (days)", 1, 30, 7)

            update_frequency = st.selectbox(
                "Model Update Frequency",
                ["Real-time", "Hourly", "Daily", "Weekly"]
            )

        with col2:
            st.markdown("**Input Features**")
            use_weather = st.checkbox("Weather Forecast Integration", True)
            use_historical = st.checkbox("Historical Production Patterns", True)
            use_irradiance = st.checkbox("Satellite Irradiance Data", True)
            use_seasonal = st.checkbox("Seasonal Decomposition", True)

        st.markdown("**Advanced Settings**")

        col1, col2, col3 = st.columns(3)

        with col1:
            confidence_level = st.slider("Confidence Interval", 80, 99, 95)
        with col2:
            training_window = st.slider("Training Window (days)", 30, 730, 365)
        with col3:
            retraining_freq = st.slider("Retraining Frequency (days)", 1, 30, 7)

        if st.button("💾 Save Configuration"):
            st.success("✓ Forecast configuration saved successfully!")

        # Model status
        st.divider()
        st.subheader("Model Status")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Version", "v2.3.1")
        with col2:
            st.metric("Last Training", "2025-11-15")
        with col3:
            st.metric("Training Samples", "365 days")
