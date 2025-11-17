"""System Planning page for PV Circularity Simulator."""

import streamlit as st


def render() -> None:
    """Render the System Planning page.

    This page provides tools for planning complete PV installations,
    including system sizing, layout design, and energy forecasting.
    """
    st.title("🏗️ System Planning")

    st.markdown("""
    Plan and design complete photovoltaic installations with system sizing,
    layout optimization, and energy production forecasting.
    """)

    # System parameters
    st.subheader("⚙️ System Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        system_size = st.number_input("System Size (kWp)", min_value=0.0, value=10.0, step=0.5)
    with col2:
        location = st.selectbox("Location", ["California, USA", "Berlin, Germany", "Tokyo, Japan", "Sydney, Australia"])
    with col3:
        tilt_angle = st.slider("Tilt Angle (°)", min_value=0, max_value=90, value=30)

    # Layout design
    with st.expander("📐 Layout Design"):
        st.markdown("**Array Configuration**")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Modules per String", min_value=1, value=10)
            st.number_input("Number of Strings", min_value=1, value=3)
        with col2:
            st.number_input("Row Spacing (m)", min_value=0.0, value=2.5)
            st.selectbox("Mounting Type", ["Fixed Tilt", "Single-axis Tracker", "Dual-axis Tracker"])

    # Energy forecasting
    st.subheader("📈 Energy Forecasting")

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating energy production forecast..."):
            import pandas as pd
            import numpy as np

            # Example forecast data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            production = [650, 750, 950, 1100, 1250, 1300, 1350, 1280, 1050, 850, 680, 600]

            forecast_data = pd.DataFrame({
                'Month': months,
                'Production (kWh)': production
            })

            st.line_chart(forecast_data.set_index('Month'))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Production", "11,810 kWh")
            with col2:
                st.metric("Capacity Factor", "13.4%")
            with col3:
                st.metric("Performance Ratio", "82%")

    # System economics
    with st.expander("💰 Economic Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input("System Cost ($/W)", min_value=0.0, value=2.5, step=0.1)
            st.number_input("Electricity Rate ($/kWh)", min_value=0.0, value=0.15, step=0.01)

        with col2:
            st.metric("Total Investment", "$25,000")
            st.metric("Payback Period", "7.2 years")
            st.metric("25-year ROI", "328%")
