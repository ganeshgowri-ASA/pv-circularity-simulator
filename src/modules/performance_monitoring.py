"""
Performance Monitoring Module - Real-time system monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def render():
    """Render the Performance Monitoring module"""
    st.header("📈 Performance Monitoring")
    st.markdown("---")

    st.markdown("""
    ### Real-time System Performance Monitoring

    Monitor and analyze PV system performance in real-time.
    """)

    # Current status overview
    st.markdown("### System Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Power", "87.5 kW", delta="2.3 kW")
    with col2:
        st.metric("Today's Energy", "542 kWh", delta="12%")
    with col3:
        st.metric("Performance Ratio", "82.5%", delta="-1.2%")
    with col4:
        status = st.selectbox("Status", ["🟢 Online", "🟡 Warning", "🔴 Offline"], index=0, label_visibility="collapsed")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Live Data", "Historical", "Alarms", "Reports"])

    with tab1:
        st.subheader("Live Performance Data")

        # Time range selector
        col1, col2 = st.columns([3, 1])
        with col2:
            refresh_rate = st.selectbox("Refresh", ["Manual", "5 sec", "15 sec", "1 min", "5 min"])

        # Generate sample real-time data
        now = datetime.now()
        times = [now - timedelta(minutes=i*5) for i in range(12)]
        times.reverse()

        power_data = pd.DataFrame({
            'Time': times,
            'AC Power (kW)': [85 + np.random.rand()*10 for _ in range(12)],
            'DC Power (kW)': [87 + np.random.rand()*10 for _ in range(12)],
        })

        st.markdown("#### Power Output (Last Hour)")
        st.line_chart(power_data.set_index('Time'))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### String Performance")

            string_data = pd.DataFrame({
                'String': [f'String {i+1}' for i in range(8)],
                'Voltage (V)': [720 + np.random.rand()*20 for _ in range(8)],
                'Current (A)': [10.5 + np.random.rand()*2 for _ in range(8)],
                'Power (kW)': [7.5 + np.random.rand()*1.5 for _ in range(8)],
            })
            st.dataframe(string_data, use_container_width=True)

        with col2:
            st.markdown("#### Environmental Conditions")

            env_col1, env_col2 = st.columns(2)
            with env_col1:
                st.metric("Irradiance", "875 W/m²")
                st.metric("Module Temp", "42°C")
            with env_col2:
                st.metric("Ambient Temp", "28°C")
                st.metric("Wind Speed", "3.2 m/s")

            st.markdown("#### Inverter Status")
            st.metric("Efficiency", "98.2%")
            st.metric("Grid Frequency", "59.98 Hz")

    with tab2:
        st.subheader("Historical Performance")

        time_range = st.selectbox(
            "Time Period",
            ["Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Custom Range"]
        )

        if time_range == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")

        # Generate sample historical data
        st.markdown("#### Energy Production")
        days = pd.date_range(end=datetime.now(), periods=30, freq='D')
        daily_energy = pd.DataFrame({
            'Date': days,
            'Energy (kWh)': [450 + np.random.rand()*150 for _ in range(30)]
        })
        st.bar_chart(daily_energy.set_index('Date'))

        st.markdown("#### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Energy (30d)", "15,240 kWh")
            st.metric("Average Daily", "508 kWh/day")
        with col2:
            st.metric("Peak Power", "95.2 kW")
            st.metric("Average PR", "81.5%")
        with col3:
            st.metric("Availability", "99.2%")
            st.metric("Capacity Factor", "18.5%")

        # Performance ratio trend
        st.markdown("#### Performance Ratio Trend")
        pr_data = pd.DataFrame({
            'Date': days,
            'PR (%)': [80 + np.random.rand()*5 for _ in range(30)]
        })
        st.line_chart(pr_data.set_index('Date'))

    with tab3:
        st.subheader("Alarms & Notifications")

        # Alarm filters
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect(
                "Severity",
                ["Critical", "Warning", "Info"],
                default=["Critical", "Warning"]
            )
        with col2:
            status_filter = st.multiselect(
                "Status",
                ["Active", "Acknowledged", "Resolved"],
                default=["Active"]
            )
        with col3:
            category_filter = st.multiselect(
                "Category",
                ["System", "String", "Inverter", "Communication", "Environmental"],
                default=["System", "String", "Inverter"]
            )

        # Sample alarm data
        alarms = [
            {"Time": "2024-01-15 10:23", "Severity": "Warning", "Category": "String",
             "Message": "String 3: Low power output detected", "Status": "Active"},
            {"Time": "2024-01-15 09:15", "Severity": "Info", "Category": "System",
             "Message": "Daily performance report generated", "Status": "Resolved"},
            {"Time": "2024-01-14 16:45", "Severity": "Critical", "Category": "Inverter",
             "Message": "Inverter 1: Grid voltage out of range", "Status": "Acknowledged"},
            {"Time": "2024-01-14 14:30", "Severity": "Warning", "Category": "Environmental",
             "Message": "High module temperature detected (>65°C)", "Status": "Resolved"},
        ]

        st.markdown("#### Active Alarms")
        for alarm in alarms:
            if alarm['Status'] in status_filter and alarm['Severity'] in severity_filter:
                severity_icon = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}
                with st.expander(f"{severity_icon[alarm['Severity']]} {alarm['Time']} - {alarm['Message']}"):
                    st.write(f"**Category:** {alarm['Category']}")
                    st.write(f"**Status:** {alarm['Status']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Acknowledge", key=f"ack_{alarm['Time']}"):
                            st.info("Alarm acknowledged")
                    with col2:
                        if st.button("Resolve", key=f"resolve_{alarm['Time']}"):
                            st.success("Alarm resolved")

        st.markdown("---")
        st.markdown("#### Alarm Statistics (Last 30 Days)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alarms", "47")
        with col2:
            st.metric("Critical", "3", delta_color="inverse")
        with col3:
            st.metric("Avg Resolution Time", "2.5 hours")

    with tab4:
        st.subheader("Performance Reports")

        report_type = st.selectbox(
            "Report Type",
            ["Daily Summary", "Weekly Summary", "Monthly Summary",
             "Custom Performance Report", "Availability Report", "Production Forecast vs Actual"]
        )

        report_period = st.date_input("Report Period", value=datetime.now())

        st.markdown("#### Report Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.multiselect(
                "Include Metrics",
                ["Energy Production", "Performance Ratio", "Availability",
                 "Alarms Summary", "Environmental Data", "Financial Metrics"],
                default=["Energy Production", "Performance Ratio"]
            )
        with col2:
            st.selectbox("Report Format", ["PDF", "Excel", "CSV", "JSON"])
            st.checkbox("Email Report", value=False)
            st.checkbox("Auto-generate Daily", value=False)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Report", use_container_width=True):
                st.success("Report generated successfully!")
        with col2:
            if st.button("📧 Schedule Report", use_container_width=True):
                st.info("Report scheduling configured")

    st.markdown("---")
    if st.button("💾 Export Monitoring Data", use_container_width=True):
        st.success("Monitoring data exported successfully!")
