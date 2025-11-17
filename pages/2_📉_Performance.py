"""Performance Analysis Page - Performance Ratio and System Efficiency."""

import streamlit as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType
from src.ui.dashboard import EYADashboard

st.set_page_config(page_title="Performance Analysis", page_icon="📉", layout="wide")

st.title("📉 Performance Analysis")
st.markdown("Analyze system performance ratio, yields, and efficiency metrics")
st.markdown("---")

# Quick configuration (simplified)
with st.sidebar:
    st.subheader("Quick Configuration")
    capacity_dc = st.number_input("DC Capacity (kWp)", value=1000.0, min_value=1.0)
    capacity_ac = st.number_input("AC Capacity (kWac)", value=850.0, min_value=1.0)
    tilt_angle = st.number_input("Tilt Angle (°)", value=30.0, min_value=0.0, max_value=90.0)

# Create default configuration
project_info = ProjectInfo(
    project_name="Solar PV Project",
    location="San Francisco, CA",
    latitude=37.7749,
    longitude=-122.4194,
    commissioning_date=datetime(2024, 1, 1),
)

system_config = SystemConfiguration(
    capacity_dc=capacity_dc,
    capacity_ac=capacity_ac,
    module_type=ModuleType.MONO_SI,
    module_efficiency=0.20,
    module_count=5000,
    tilt_angle=tilt_angle,
    azimuth_angle=180.0,
)

# Initialize dashboard
dashboard = EYADashboard(project_info, system_config)

# Calculate performance
with st.spinner("Calculating performance metrics..."):
    pr_data = dashboard.performance_ratio()

st.success("✅ Performance analysis complete!")

# Performance Ratio Overview
st.markdown("### 🎯 Performance Ratio")

col1, col2, col3 = st.columns(3)

pr_metrics = pr_data["Performance Ratio"]

with col1:
    pr_value = float(pr_metrics["PR"].replace("%", ""))
    st.metric(
        "Performance Ratio (PR)",
        pr_metrics["PR"],
        delta=f"{pr_value - 75:.1f}% vs 75% baseline",
        delta_color="normal",
    )

with col2:
    st.metric("Reference Yield", pr_metrics["Reference Yield"])

with col3:
    st.metric("Final Yield", pr_metrics["Final Yield"])

# Yield Analysis
st.markdown("### 📊 Yield Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Energy Flow")
    yield_data = pr_data["Performance Ratio"]

    st.markdown(f"""
    **Reference Yield**: {yield_data['Reference Yield']}
    ↓
    **Array Yield**: {yield_data['Array Yield']}
    ↓
    **Final Yield**: {yield_data['Final Yield']}
    """)

with col2:
    st.markdown("#### Losses")
    loss_data = pr_data["Yield Analysis"]

    st.markdown(f"""
    **Capture Losses**: {loss_data['Capture Losses']}
    **System Losses**: {loss_data['System Losses']}
    **Total Losses**: {loss_data['Total Losses']}
    """)

# Efficiency Metrics
st.markdown("### ⚡ Efficiency Metrics")

col1, col2, col3 = st.columns(3)

eff_metrics = pr_data["Efficiency Metrics"]

with col1:
    st.metric("Array Efficiency", eff_metrics["Array Efficiency"])

with col2:
    st.metric("System Efficiency", eff_metrics["System Efficiency"])

with col3:
    st.metric("Overall Efficiency", eff_metrics["Overall Efficiency"])

# Performance Breakdown
st.markdown("### 📋 Detailed Performance Breakdown")

breakdown_data = {
    "Metric": [
        "Reference Yield",
        "Capture Losses",
        "Array Yield",
        "System Losses",
        "Final Yield",
        "Performance Ratio",
    ],
    "Value": [
        pr_metrics["Reference Yield"],
        pr_data["Yield Analysis"]["Capture Losses"],
        pr_metrics["Array Yield"],
        pr_data["Yield Analysis"]["System Losses"],
        pr_metrics["Final Yield"],
        pr_metrics["PR"],
    ],
    "Description": [
        "Theoretical maximum based on irradiation",
        "Losses from irradiation to DC array",
        "DC energy at array output",
        "Losses from DC to AC conversion",
        "Final AC energy delivered",
        "Overall system performance",
    ],
}

import pandas as pd
st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

# Insights
st.markdown("### 💡 Performance Insights")

pr_val = float(pr_metrics["PR"].replace("%", ""))

if pr_val > 85:
    st.success(f"""
    ✅ **Excellent Performance**
    Your system's PR of {pr_metrics['PR']} is excellent and indicates a well-designed,
    high-performing PV system. This is above the typical range of 75-85%.
    """)
elif pr_val > 75:
    st.info(f"""
    ℹ️ **Good Performance**
    Your system's PR of {pr_metrics['PR']} is within the typical range for well-maintained
    PV systems (75-85%). There may be opportunities for minor optimizations.
    """)
else:
    st.warning(f"""
    ⚠️ **Performance Below Expectations**
    Your system's PR of {pr_metrics['PR']} is below the typical range (75-85%).
    Consider reviewing system losses, maintenance practices, and component efficiency.
    """)

# Recommendations
with st.expander("📌 Recommendations for Improvement"):
    st.markdown("""
    **To Improve Performance Ratio:**

    1. **Reduce Capture Losses**
       - Optimize tilt and azimuth angles for location
       - Minimize shading from nearby objects
       - Regular cleaning to reduce soiling losses

    2. **Reduce System Losses**
       - Use high-efficiency inverters
       - Minimize DC wiring losses with proper cable sizing
       - Ensure good module matching to reduce mismatch losses

    3. **Maintenance Best Practices**
       - Regular cleaning schedules
       - Periodic inspection for damage
       - Monitor performance for early fault detection
       - Keep vegetation trimmed to prevent shading

    4. **System Monitoring**
       - Install real-time monitoring systems
       - Track PR trends over time
       - Compare against weather-adjusted expectations
       - Investigate any significant deviations
    """)
