"""Energy Analysis Page - Annual and Monthly Energy Production Analysis."""

import streamlit as st
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType
from src.ui.dashboard import EYADashboard
from src.ui.visualizations import InteractiveVisualizations

st.set_page_config(page_title="Energy Analysis", page_icon="📊", layout="wide")

st.title("📊 Energy Production Analysis")
st.markdown("---")

# Initialize session state
if "project_configured" not in st.session_state:
    st.session_state.project_configured = False

# Configuration section
with st.expander("⚙️ Project Configuration", expanded=not st.session_state.project_configured):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Information")
        project_name = st.text_input("Project Name", value="Solar PV Project")
        location = st.text_input("Location", value="San Francisco, CA")
        latitude = st.number_input("Latitude", value=37.7749, min_value=-90.0, max_value=90.0, step=0.0001)
        longitude = st.number_input("Longitude", value=-122.4194, min_value=-180.0, max_value=180.0, step=0.0001)
        altitude = st.number_input("Altitude (m)", value=0.0, min_value=0.0, step=1.0)
        commissioning_date = st.date_input("Commissioning Date", value=datetime(2024, 1, 1))
        project_lifetime = st.number_input("Project Lifetime (years)", value=25, min_value=1, max_value=50)

    with col2:
        st.subheader("System Configuration")
        capacity_dc = st.number_input("DC Capacity (kWp)", value=1000.0, min_value=1.0, step=10.0)
        capacity_ac = st.number_input("AC Capacity (kWac)", value=850.0, min_value=1.0, step=10.0)
        module_type = st.selectbox("Module Type", options=["mono-Si", "poly-Si", "CdTe", "CIGS"])
        module_efficiency = st.number_input("Module Efficiency (%)", value=20.0, min_value=1.0, max_value=50.0) / 100
        module_count = st.number_input("Module Count", value=5000, min_value=1, step=1)
        inverter_efficiency = st.number_input("Inverter Efficiency (%)", value=98.0, min_value=80.0, max_value=100.0) / 100
        mounting_type = st.selectbox("Mounting Type", options=["fixed_tilt", "single_axis", "dual_axis", "roof_mounted"])
        tilt_angle = st.number_input("Tilt Angle (°)", value=30.0, min_value=0.0, max_value=90.0)
        azimuth_angle = st.number_input("Azimuth Angle (°)", value=180.0, min_value=0.0, max_value=360.0)

    if st.button("🚀 Configure & Calculate", type="primary"):
        st.session_state.project_configured = True
        st.rerun()

# Main analysis
if st.session_state.project_configured or st.button("Use Default Configuration"):
    try:
        # Create project info and system config
        project_info = ProjectInfo(
            project_name=project_name if "project_name" in locals() else "Solar PV Project",
            location=location if "location" in locals() else "San Francisco, CA",
            latitude=latitude if "latitude" in locals() else 37.7749,
            longitude=longitude if "longitude" in locals() else -122.4194,
            altitude=altitude if "altitude" in locals() else 0.0,
            commissioning_date=datetime.combine(commissioning_date, datetime.min.time()) if "commissioning_date" in locals() else datetime(2024, 1, 1),
            project_lifetime=project_lifetime if "project_lifetime" in locals() else 25,
        )

        system_config = SystemConfiguration(
            capacity_dc=capacity_dc if "capacity_dc" in locals() else 1000.0,
            capacity_ac=capacity_ac if "capacity_ac" in locals() else 850.0,
            module_type=ModuleType(module_type) if "module_type" in locals() else ModuleType.MONO_SI,
            module_efficiency=module_efficiency if "module_efficiency" in locals() else 0.20,
            module_count=module_count if "module_count" in locals() else 5000,
            inverter_efficiency=inverter_efficiency if "inverter_efficiency" in locals() else 0.98,
            mounting_type=MountingType(mounting_type) if "mounting_type" in locals() else MountingType.FIXED_TILT,
            tilt_angle=tilt_angle if "tilt_angle" in locals() else 30.0,
            azimuth_angle=azimuth_angle if "azimuth_angle" in locals() else 180.0,
        )

        # Initialize dashboard
        dashboard = EYADashboard(project_info, system_config)

        # Calculate energy output
        with st.spinner("Calculating energy production..."):
            energy_data = dashboard.annual_energy_output()

        st.success("✅ Analysis complete!")

        # Display annual totals
        st.markdown("### 📈 Annual Energy Production")

        col1, col2, col3, col4 = st.columns(4)

        annual_totals = energy_data["Annual Totals"]
        perf_indicators = energy_data["Performance Indicators"]

        with col1:
            st.metric("DC Energy", annual_totals["DC Energy"])

        with col2:
            st.metric("AC Energy", annual_totals["AC Energy"])

        with col3:
            st.metric("Specific Yield", perf_indicators["Specific Yield"])

        with col4:
            st.metric("Capacity Factor", perf_indicators["Capacity Factor"])

        # Performance indicators
        st.markdown("### 🎯 Performance Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Daily Energy", perf_indicators["Average Daily Energy"])

        with col2:
            st.metric("Peak Month Production", perf_indicators["Peak Month Production"])

        with col3:
            st.metric("Lowest Month Production", perf_indicators["Lowest Month Production"])

        # Monthly data visualization
        st.markdown("### 📊 Monthly Production Analysis")

        viz = InteractiveVisualizations()
        monthly_chart = viz.monthly_production_charts(energy_data["monthly_data"])
        st.plotly_chart(monthly_chart, use_container_width=True)

        # Monthly data table
        with st.expander("📋 View Monthly Data Table"):
            st.dataframe(
                energy_data["monthly_data"].style.format({
                    "dc_energy": "{:,.0f}",
                    "ac_energy": "{:,.0f}",
                    "exported_energy": "{:,.0f}",
                    "specific_yield": "{:.2f}",
                    "capacity_factor": "{:.2%}",
                }),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please configure the project parameters above to start the analysis.")
