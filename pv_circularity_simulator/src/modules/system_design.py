"""
System Design Module
===================

Complete PV system design and configuration.
Defines array layout, inverter selection, and system topology.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the system design module.

    Args:
        session: Session manager instance

    Features:
        - Site location and solar resource
        - Array configuration (tilt, azimuth, tracking)
        - Inverter selection and sizing
        - String/combiner box design
        - DC/AC ratio optimization
        - Shading analysis
        - System capacity and layout
    """
    st.header("🏗️ System Design")

    st.info("Design your complete PV system configuration.")

    # Site information
    st.subheader("📍 Site Information")
    col1, col2 = st.columns(2)

    with col1:
        site_name = st.text_input("Site Name", "Solar Plant 1")
        latitude = st.number_input("Latitude (°)", -90.0, 90.0, 28.6139, 0.0001)
        longitude = st.number_input("Longitude (°)", -180.0, 180.0, 77.2090, 0.0001)
        altitude = st.number_input("Altitude (m)", 0, 5000, 200)

    with col2:
        st.markdown("**Climate Data**")
        avg_ghi = st.number_input("Avg. GHI (kWh/m²/day)", 0.0, 10.0, 5.5, 0.1)
        avg_temp = st.number_input("Avg. Temperature (°C)", -20, 50, 25)
        wind_speed = st.number_input("Avg. Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.1)

    # Array configuration
    st.markdown("---")
    st.subheader("☀️ Array Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        mounting_type = st.selectbox(
            "Mounting Type",
            ["Fixed Tilt", "Single-axis Tracking", "Dual-axis Tracking", "Rooftop"]
        )
        tilt_angle = st.slider("Tilt Angle (°)", 0, 90, 25)
        azimuth = st.slider("Azimuth (°)", 0, 360, 180)

    with col2:
        num_modules = st.number_input("Number of Modules", 1, 100000, 1000)
        module_power = st.number_input("Module Power (W)", 100, 700, 400)
        system_capacity = (num_modules * module_power) / 1000
        st.metric("System Capacity", f"{system_capacity:.2f} kWp")

    with col3:
        modules_per_string = st.number_input("Modules per String", 1, 50, 20)
        num_strings = st.number_input("Number of Strings", 1, 1000, 50)
        ground_coverage = st.slider("Ground Coverage Ratio", 0.1, 0.8, 0.4, 0.05)

    # Inverter configuration
    st.markdown("---")
    st.subheader("🔌 Inverter Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        inverter_type = st.selectbox(
            "Inverter Type",
            ["String Inverter", "Central Inverter", "Micro Inverter", "Power Optimizer"]
        )
        inverter_capacity = st.number_input("Inverter Capacity (kW)", 1, 10000, 900)
        num_inverters = st.number_input("Number of Inverters", 1, 100, 1)

    with col2:
        inverter_efficiency = st.slider("Inverter Efficiency (%)", 90.0, 99.0, 98.5, 0.1)
        dc_ac_ratio = system_capacity / (inverter_capacity * num_inverters)
        st.metric("DC/AC Ratio", f"{dc_ac_ratio:.2f}")

        if dc_ac_ratio < 1.1:
            st.warning("DC/AC ratio is low. Consider optimizing.")
        elif dc_ac_ratio > 1.5:
            st.warning("DC/AC ratio is high. May lead to clipping.")
        else:
            st.success("DC/AC ratio is optimal.")

    with col3:
        max_dc_voltage = st.number_input("Max DC Voltage (V)", 600, 1500, 1000)
        mppt_channels = st.number_input("MPPT Channels", 1, 20, 8)
        strings_per_mppt = num_strings // mppt_channels
        st.metric("Strings per MPPT", strings_per_mppt)

    # System layout
    st.markdown("---")
    st.subheader("📐 System Layout")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Modules", f"{num_modules:,}")
    with col2:
        st.metric("Total Strings", num_strings)
    with col3:
        st.metric("System DC Capacity", f"{system_capacity:.2f} kWp")
    with col4:
        total_ac = inverter_capacity * num_inverters
        st.metric("System AC Capacity", f"{total_ac:.2f} kW")

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Loss Factors**")
            soiling_loss = st.slider("Soiling Loss (%)", 0.0, 10.0, 2.0, 0.1)
            shading_loss = st.slider("Shading Loss (%)", 0.0, 20.0, 3.0, 0.1)
            snow_loss = st.slider("Snow Loss (%)", 0.0, 10.0, 0.0, 0.1)

        with col2:
            st.markdown("**System Losses**")
            dc_wiring_loss = st.slider("DC Wiring Loss (%)", 0.0, 5.0, 1.5, 0.1)
            ac_wiring_loss = st.slider("AC Wiring Loss (%)", 0.0, 5.0, 1.0, 0.1)
            transformer_loss = st.slider("Transformer Loss (%)", 0.0, 3.0, 1.0, 0.1)

    # Save design
    if st.button("Save System Design"):
        session.set('system_design_data', {
            'site': {
                'name': site_name,
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'avg_ghi': avg_ghi,
                'avg_temp': avg_temp,
                'wind_speed': wind_speed
            },
            'array': {
                'mounting_type': mounting_type,
                'tilt_angle': tilt_angle,
                'azimuth': azimuth,
                'num_modules': num_modules,
                'module_power': module_power,
                'modules_per_string': modules_per_string,
                'num_strings': num_strings,
                'ground_coverage': ground_coverage
            },
            'inverter': {
                'type': inverter_type,
                'capacity': inverter_capacity,
                'num_inverters': num_inverters,
                'efficiency': inverter_efficiency,
                'dc_ac_ratio': dc_ac_ratio,
                'max_dc_voltage': max_dc_voltage,
                'mppt_channels': mppt_channels
            },
            'losses': {
                'soiling': soiling_loss,
                'shading': shading_loss,
                'snow': snow_loss,
                'dc_wiring': dc_wiring_loss,
                'ac_wiring': ac_wiring_loss,
                'transformer': transformer_loss
            }
        })
        st.success("System design saved!")
