"""
Streamlit dashboard for real-time PV system monitoring.

This module provides an interactive web dashboard for monitoring PV system
performance with auto-refresh and real-time data visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import get_settings
from src.monitoring.realtime.monitor import RealTimeMonitor
from src.monitoring.metrics.performance import PerformanceMetrics
from src.monitoring.alerts.engine import AlertEngine

# Page configuration
st.set_page_config(
    page_title="PV Real-Time Monitoring",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load settings
settings = get_settings()

# Auto-refresh setup
refresh_interval = settings.streamlit.refresh_interval_sec * 1000  # Convert to milliseconds
st_autorefresh(interval=refresh_interval, limit=None, key="data_refresh")


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'monitor' not in st.session_state:
        st.session_state.monitor = None
    if 'metrics_calculator' not in st.session_state:
        st.session_state.metrics_calculator = PerformanceMetrics(settings)
    if 'alert_engine' not in st.session_state:
        st.session_state.alert_engine = AlertEngine(settings)
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = {
            'inverters': [],
            'scada': [],
            'metrics': [],
            'alerts': []
        }


def create_gauge_chart(value: float, title: str, max_value: float = 100, threshold: float = 80):
    """
    Create a gauge chart for metrics display.

    Args:
        value: Current value
        title: Chart title
        max_value: Maximum value for gauge
        threshold: Threshold for color coding

    Returns:
        Plotly figure object
    """
    # Determine color based on threshold
    if value >= threshold:
        color = "green"
    elif value >= threshold * 0.8:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': threshold, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 0.8], 'color': '#ffcccc'},
                {'range': [threshold * 0.8, threshold], 'color': '#ffffcc'},
                {'range': [threshold, max_value], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_power_time_series(data: list, title: str = "Power Output"):
    """
    Create time-series chart for power data.

    Args:
        data: List of data points with timestamp and power values
        title: Chart title

    Returns:
        Plotly figure object
    """
    if not data:
        return go.Figure()

    fig = go.Figure()

    # Extract timestamps and values
    timestamps = [d['timestamp'] for d in data]
    ac_power = [d.get('ac_power', 0) for d in data]
    dc_power = [d.get('dc_power', 0) for d in data]

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ac_power,
        mode='lines+markers',
        name='AC Power',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=dc_power,
        mode='lines+markers',
        name='DC Power',
        line=dict(color='orange', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_alert_table(alerts: list):
    """
    Create styled alert table.

    Args:
        alerts: List of alert dictionaries

    Returns:
        DataFrame styled for display
    """
    if not alerts:
        return None

    import pandas as pd

    df = pd.DataFrame([
        {
            'Time': a.get('timestamp', datetime.utcnow()).strftime('%H:%M:%S'),
            'Severity': a.get('severity', 'UNKNOWN'),
            'Type': a.get('alert_type', 'UNKNOWN'),
            'Message': a.get('message', ''),
            'Component': a.get('component_id', 'N/A')
        }
        for a in alerts
    ])

    # Color coding based on severity
    def color_severity(val):
        colors = {
            'CRITICAL': 'background-color: #ff4444',
            'HIGH': 'background-color: #ff8844',
            'MEDIUM': 'background-color: #ffaa44',
            'LOW': 'background-color: #ffdd44',
            'INFO': 'background-color: #44ddff'
        }
        return colors.get(val, '')

    styled_df = df.style.applymap(color_severity, subset=['Severity'])
    return styled_df


def main():
    """Main dashboard application."""
    init_session_state()

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown(f"**Site ID:** {settings.site_id}")
    st.sidebar.markdown(f"**Capacity:** {settings.site_capacity_kw} kW")
    st.sidebar.markdown(f"**Refresh Interval:** {settings.streamlit.refresh_interval_sec}s")

    # Monitoring controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Monitoring Controls")

    monitoring_enabled = st.sidebar.checkbox("Enable Real-Time Monitoring", value=False)

    if monitoring_enabled and st.session_state.monitor is None:
        st.sidebar.success("Starting monitoring...")
        # Note: In production, this would be handled by a background service
        st.session_state.monitor = RealTimeMonitor(settings)

    # Main dashboard
    st.title("☀️ PV Real-Time Performance Monitoring")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "⚡ Inverters",
        "📈 Performance",
        "🚨 Alerts"
    ])

    # Overview Tab
    with tab1:
        st.header("System Overview")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Power",
                f"{850.5:.1f} kW",
                delta=f"{12.3:.1f} kW",
                delta_color="normal"
            )

        with col2:
            st.metric(
                "Performance Ratio",
                f"{87.5:.1f}%",
                delta=f"{-2.1:.1f}%",
                delta_color="inverse"
            )

        with col3:
            st.metric(
                "Daily Energy",
                f"{4250.0:.0f} kWh",
                delta=f"{150.0:.0f} kWh"
            )

        with col4:
            st.metric(
                "Availability",
                f"{98.5:.1f}%",
                delta=f"{0.5:.1f}%"
            )

        # Gauge charts row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_pr = create_gauge_chart(87.5, "Performance Ratio (%)", 100, 80)
            st.plotly_chart(fig_pr, use_container_width=True)

        with col2:
            fig_cf = create_gauge_chart(35.2, "Capacity Factor (%)", 100, 30)
            st.plotly_chart(fig_cf, use_container_width=True)

        with col3:
            fig_avail = create_gauge_chart(98.5, "Availability (%)", 100, 95)
            st.plotly_chart(fig_avail, use_container_width=True)

        # Power time series
        st.markdown("---")
        st.subheader("Power Output Trend")

        # Sample data (in production, this would come from real-time data)
        sample_data = [
            {'timestamp': datetime.now() - timedelta(minutes=i*5), 'ac_power': 800 + i*10, 'dc_power': 820 + i*10}
            for i in range(20, 0, -1)
        ]

        fig_power = create_power_time_series(sample_data)
        st.plotly_chart(fig_power, use_container_width=True)

    # Inverters Tab
    with tab2:
        st.header("Inverter Monitoring")

        # Filter options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_inv = st.text_input("Search Inverter", placeholder="Enter inverter ID...")
        with col2:
            status_filter = st.selectbox("Status", ["All", "Online", "Offline", "Fault"])

        # Sample inverter data
        import pandas as pd
        inv_data = pd.DataFrame([
            {
                'ID': f'INV{i:03d}',
                'Status': 'Online' if i % 10 != 0 else 'Fault',
                'AC Power (kW)': round(45.5 + i * 0.5, 1),
                'DC Power (kW)': round(47.2 + i * 0.5, 1),
                'Efficiency (%)': round(96.5 - i * 0.1, 1),
                'Temperature (°C)': round(45.0 + i * 0.3, 1)
            }
            for i in range(1, 21)
        ])

        st.dataframe(inv_data, use_container_width=True, height=400)

        # Download button
        csv = inv_data.to_csv(index=False)
        st.download_button(
            "Download Inverter Data (CSV)",
            csv,
            "inverter_data.csv",
            "text/csv",
            key='download-csv'
        )

    # Performance Tab
    with tab3:
        st.header("Performance Metrics")

        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now())

        # Performance metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Specific Yield")
            sy_data = [
                {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 'value': 4.2 + i*0.1}
                for i in range(7, 0, -1)
            ]
            fig_sy = px.bar(
                sy_data,
                x='date',
                y='value',
                title='Daily Specific Yield (kWh/kWp)',
                labels={'date': 'Date', 'value': 'Specific Yield'}
            )
            st.plotly_chart(fig_sy, use_container_width=True)

        with col2:
            st.subheader("Performance Ratio Trend")
            pr_data = [
                {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 'value': 85.0 + i*0.5}
                for i in range(7, 0, -1)
            ]
            fig_pr_trend = px.line(
                pr_data,
                x='date',
                y='value',
                title='Performance Ratio (%)',
                labels={'date': 'Date', 'value': 'PR (%)'},
                markers=True
            )
            fig_pr_trend.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig_pr_trend, use_container_width=True)

    # Alerts Tab
    with tab4:
        st.header("System Alerts")

        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Critical", "2", delta="0")
        with col2:
            st.metric("High", "5", delta="+1")
        with col3:
            st.metric("Medium", "8", delta="-2")
        with col4:
            st.metric("Low", "12", delta="+3")

        st.markdown("---")

        # Recent alerts
        st.subheader("Recent Alerts")

        sample_alerts = [
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'severity': 'HIGH',
                'alert_type': 'UNDERPERFORMANCE',
                'message': 'Performance ratio below threshold',
                'component_id': 'INV005'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'severity': 'MEDIUM',
                'alert_type': 'EQUIPMENT_FAULT',
                'message': 'High temperature detected',
                'component_id': 'INV012'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'severity': 'CRITICAL',
                'alert_type': 'GRID_OUTAGE',
                'message': 'Grid frequency out of range',
                'component_id': 'SITE'
            }
        ]

        alert_table = create_alert_table(sample_alerts)
        if alert_table is not None:
            st.dataframe(alert_table, use_container_width=True)

        # Alert details expander
        for alert in sample_alerts:
            with st.expander(f"{alert['severity']} - {alert['message']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Type:** {alert['alert_type']}")
                with col2:
                    st.write(f"**Component:** {alert['component_id']}")
                    st.write(f"**Severity:** {alert['severity']}")

                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Acknowledge", key=f"ack_{alert['component_id']}"):
                        st.success("Alert acknowledged")
                with col2:
                    if st.button("Resolve", key=f"resolve_{alert['component_id']}"):
                        st.success("Alert resolved")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 12px;">
        PV Circularity Simulator - Real-Time Performance Monitoring System<br>
        Powered by Streamlit | Auto-refresh enabled
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
