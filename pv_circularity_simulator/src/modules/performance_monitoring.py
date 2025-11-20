"""
Performance Monitoring Module
=============================

Real-time and historical performance monitoring.
Track actual vs. expected performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the performance monitoring module.

    Args:
        session: Session manager instance

    Features:
        - Real-time performance metrics
        - Expected vs. actual comparison
        - Performance ratio tracking
        - Availability monitoring
        - Downtime analysis
        - Alert generation
        - KPI dashboards
    """
    st.header("📈 Performance Monitoring")

    st.info("Monitor real-time and historical system performance.")

    # Time range selection
    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.selectbox("View Mode", ["Live", "Historical", "Comparison"])

    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"]
        )

    with col3:
        if time_range == "Custom":
            date_range = st.date_input("Select Date Range", [])

    # Live metrics
    st.markdown("---")
    st.subheader("⚡ Live Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    # Generate dummy live data
    current_power = np.random.uniform(800, 950)  # kW
    expected_power = 900  # kW
    pr_current = (current_power / expected_power) * 100

    with col1:
        st.metric(
            "Current Power",
            f"{current_power:.1f} kW",
            f"{((current_power - expected_power) / expected_power * 100):.1f}%"
        )

    with col2:
        st.metric("Today's Energy", "6,234 kWh", "+5.2%")

    with col3:
        st.metric("Performance Ratio", f"{pr_current:.1f}%", "-2.1%")

    with col4:
        st.metric("System Availability", "99.2%", "+0.3%")

    # Performance trends
    st.markdown("---")
    st.subheader("📊 Performance Trends")

    # Generate dummy time series data
    hours = 24
    timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')

    # Actual power with some variation and solar curve
    base_curve = [
        0, 0, 0, 0, 0, 0,  # Night (0-5)
        100, 300, 500, 700, 850, 950,  # Morning (6-11)
        1000, 980, 950, 900, 800, 600,  # Afternoon (12-17)
        400, 200, 50, 0, 0, 0  # Evening (18-23)
    ]

    actual_power = [p + np.random.uniform(-50, 50) if p > 0 else 0 for p in base_curve]
    expected_power_curve = [p * 1.05 for p in base_curve]  # Expected is 5% higher

    perf_df = pd.DataFrame({
        'Time': timestamps,
        'Actual Power (kW)': actual_power,
        'Expected Power (kW)': expected_power_curve
    })

    st.line_chart(perf_df.set_index('Time'))

    # Daily production comparison
    st.markdown("---")
    st.subheader("📅 Daily Production Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Last 7 days
        days = 7
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        actual_daily = [np.random.uniform(5000, 6000) for _ in range(days)]
        expected_daily = [5800 for _ in range(days)]

        daily_df = pd.DataFrame({
            'Date': dates.strftime('%m-%d'),
            'Actual (kWh)': actual_daily,
            'Expected (kWh)': expected_daily
        })

        st.bar_chart(daily_df.set_index('Date'))

    with col2:
        st.markdown("**Performance Summary (7 days)**")
        total_actual = sum(actual_daily)
        total_expected = sum(expected_daily)
        avg_pr = (total_actual / total_expected) * 100

        st.metric("Total Production", f"{total_actual:,.0f} kWh")
        st.metric("Expected Production", f"{total_expected:,.0f} kWh")
        st.metric("Average PR", f"{avg_pr:.1f}%")

        if avg_pr < 75:
            st.error("⚠️ Performance below threshold!")
        elif avg_pr < 85:
            st.warning("⚠️ Performance degraded")
        else:
            st.success("✅ Performance normal")

    # System health
    st.markdown("---")
    st.subheader("🏥 System Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Inverters**")
        st.progress(0.98, text="Inverter 1: 98%")
        st.progress(0.97, text="Inverter 2: 97%")
        st.progress(0.99, text="Inverter 3: 99%")

    with col2:
        st.markdown("**Module Temperature**")
        module_temp = np.random.uniform(45, 55)
        st.metric("Avg. Temperature", f"{module_temp:.1f}°C")
        if module_temp > 65:
            st.error("⚠️ High temperature!")
        else:
            st.success("✅ Normal")

    with col3:
        st.markdown("**Alerts**")
        st.warning("⚠️ 1 String underperforming")
        st.info("ℹ️ 2 Maintenance reminders")

    # Detailed metrics table
    st.markdown("---")
    st.subheader("📋 Detailed Metrics")

    detailed_metrics = pd.DataFrame({
        'Metric': [
            'DC Power', 'AC Power', 'Energy Today', 'Energy Month',
            'Energy Year', 'Peak Power Today', 'Inverter Efficiency',
            'Grid Voltage', 'Grid Frequency', 'Power Factor'
        ],
        'Current Value': [
            '920 kW', '900 kW', '6,234 kWh', '185,432 kWh',
            '2,145,678 kWh', '950 kW', '98.5%',
            '415 V', '50.0 Hz', '0.99'
        ],
        'Status': [
            '✅ Normal', '✅ Normal', '✅ Normal', '✅ Normal',
            '✅ Normal', '✅ Normal', '✅ Normal',
            '✅ Normal', '✅ Normal', '✅ Normal'
        ]
    })

    st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)

    # Export data
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Performance Data"):
            st.info("Export functionality coming soon")

    with col2:
        if st.button("📊 Generate Report"):
            st.info("Report generation coming soon")

    with col3:
        if st.button("🔔 Configure Alerts"):
            st.info("Alert configuration coming soon")
