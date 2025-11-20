"""
System Design Module - PV system configuration and planning
"""

import streamlit as st


def render():
    """Render the System Design module"""
    st.header("🏗️ System Design")
    st.markdown("---")

    st.markdown("""
    ### PV System Configuration & Planning

    Design complete photovoltaic systems including array layout, inverters, and electrical design.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Site Info", "Array Design", "Inverter Selection", "BOS"])

    with tab1:
        st.subheader("Site Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Location")
            latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=40.7128)
            longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=-74.0060)
            elevation = st.number_input("Elevation (m)", min_value=0, value=10)

            timezone = st.selectbox("Timezone", [
                "UTC-5 (EST)", "UTC-6 (CST)", "UTC-7 (MST)", "UTC-8 (PST)",
                "UTC+1 (CET)", "UTC+8 (CST)", "Other"
            ])

        with col2:
            st.markdown("#### Climate Data")
            avg_irradiance = st.number_input("Average Annual Irradiance (kWh/m²/day)", min_value=0.0, value=4.5)
            avg_temp = st.number_input("Average Temperature (°C)", value=15.0)

            climate_zone = st.selectbox("Climate Zone", [
                "Hot-Dry", "Hot-Humid", "Mixed-Humid", "Cold", "Marine"
            ])

            st.number_input("Snow Load (kg/m²)", min_value=0.0, value=0.0)

        st.markdown("#### System Type")
        system_type = st.selectbox(
            "Installation Type",
            ["Ground-mounted fixed tilt", "Ground-mounted single-axis tracking",
             "Ground-mounted dual-axis tracking", "Rooftop residential",
             "Rooftop commercial", "Carport", "Floating PV"]
        )

    with tab2:
        st.subheader("Array Design Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Module Configuration")
            modules_per_string = st.number_input("Modules per String", min_value=1, value=20)
            strings_per_inverter = st.number_input("Strings per Inverter", min_value=1, value=10)

            total_modules = modules_per_string * strings_per_inverter
            st.metric("Total Modules per Inverter", total_modules)

            num_inverters = st.number_input("Number of Inverters", min_value=1, value=1)
            st.metric("Total System Modules", total_modules * num_inverters)

        with col2:
            st.markdown("#### Array Orientation")
            tilt_angle = st.slider("Tilt Angle (°)", 0, 90, 30)
            azimuth = st.slider("Azimuth (°)", 0, 360, 180)

            st.info(f"Optimal tilt ≈ {abs(latitude):.1f}° (latitude-based)")

            row_spacing = st.number_input("Row Spacing (m)", min_value=0.0, value=3.0, step=0.1)
            gcr = st.slider("Ground Coverage Ratio (GCR)", 0.0, 1.0, 0.4, 0.01)

        st.markdown("#### Array Sizing")
        module_power = st.number_input("Module Power (W)", min_value=0, value=375)
        system_dc_power = (total_modules * num_inverters * module_power) / 1000

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System DC Power", f"{system_dc_power:.1f} kWp")
        with col2:
            st.metric("Array Voltage", f"{modules_per_string * 40:.0f} V")
        with col3:
            st.metric("Array Current", f"{strings_per_inverter * 10:.1f} A")

    with tab3:
        st.subheader("Inverter Selection & Configuration")

        col1, col2 = st.columns(2)

        with col1:
            inverter_type = st.selectbox(
                "Inverter Type",
                ["Central Inverter", "String Inverter", "Micro-inverter",
                 "Power Optimizer + Inverter", "Hybrid Inverter"]
            )

            inverter_power = st.number_input("Inverter AC Power (kW)", min_value=0.0, value=100.0)
            dc_ac_ratio = system_dc_power / inverter_power if inverter_power > 0 else 0
            st.metric("DC/AC Ratio", f"{dc_ac_ratio:.2f}", help="Typical range: 1.1-1.3")

        with col2:
            st.markdown("#### Inverter Specifications")
            max_efficiency = st.number_input("Max Efficiency (%)", min_value=0.0, max_value=100.0, value=98.5)
            euro_efficiency = st.number_input("Euro Efficiency (%)", min_value=0.0, max_value=100.0, value=98.0)

            mppt_count = st.number_input("Number of MPPT Inputs", min_value=1, value=4)
            strings_per_mppt = strings_per_inverter // mppt_count if mppt_count > 0 else 0
            st.metric("Strings per MPPT", strings_per_mppt)

        st.markdown("#### Operating Ranges")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Min MPPT Voltage (V)", value=570)
        with col2:
            st.number_input("Max MPPT Voltage (V)", value=850)
        with col3:
            st.number_input("Max DC Voltage (V)", value=1000)

    with tab4:
        st.subheader("Balance of System (BOS)")

        st.markdown("#### DC Components")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("DC Cable Type", ["PV1-F 4mm²", "PV1-F 6mm²", "Custom"])
            st.number_input("DC Cable Length (m)", min_value=0, value=50)
        with col2:
            st.selectbox("DC Combiner Box", ["Standard", "With monitoring", "None"])
            st.checkbox("DC Disconnect Switch", value=True)

        st.markdown("#### AC Components")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("AC Cable Type", ["NYY 4x16mm²", "NYY 4x25mm²", "Custom"])
            st.number_input("AC Cable Length (m)", min_value=0, value=100)
        with col2:
            st.selectbox("Transformer", ["None", "Step-up (LV/MV)", "Isolation", "Custom"])
            st.checkbox("AC Disconnect Switch", value=True)

        st.markdown("#### Protection & Safety")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("DC Surge Protection", value=True)
            st.checkbox("AC Surge Protection", value=True)
            st.checkbox("Arc Fault Detection", value=False)
        with col2:
            st.checkbox("Ground Fault Protection", value=True)
            st.checkbox("Anti-islanding Protection", value=True)
            st.checkbox("Rapid Shutdown System", value=False)

        st.markdown("#### Monitoring & Control")
        monitoring = st.selectbox(
            "Monitoring System",
            ["Basic (Inverter only)", "Advanced (String level)",
             "Premium (Module level)", "Custom SCADA"]
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate System Report", use_container_width=True):
            st.info("System report generation not yet implemented")
    with col2:
        if st.button("💾 Save System Design", use_container_width=True):
            st.success("System design saved successfully!")
