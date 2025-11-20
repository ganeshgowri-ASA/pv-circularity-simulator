"""Loss Analysis Page - Detailed System Loss Breakdown."""

import streamlit as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType, MountingType, LossBreakdown
from src.ui.dashboard import EYADashboard
from src.ui.visualizations import InteractiveVisualizations

st.set_page_config(page_title="Loss Analysis", page_icon="🔻", layout="wide")

st.title("🔻 System Loss Analysis")
st.markdown("Comprehensive breakdown of all system losses")
st.markdown("---")

# Custom loss configuration
with st.sidebar:
    st.subheader("Loss Configuration")

    with st.expander("Customize Loss Values"):
        soiling_loss = st.slider("Soiling Loss (%)", 0.0, 10.0, 2.0, 0.1)
        shading_loss = st.slider("Shading Loss (%)", 0.0, 15.0, 3.0, 0.1)
        snow_loss = st.slider("Snow Loss (%)", 0.0, 5.0, 0.5, 0.1)
        temperature_loss = st.slider("Temperature Loss (%)", 0.0, 15.0, 5.0, 0.1)
        mismatch_loss = st.slider("Mismatch Loss (%)", 0.0, 5.0, 2.0, 0.1)
        inverter_loss = st.slider("Inverter Loss (%)", 0.0, 5.0, 2.0, 0.1)

        loss_breakdown = LossBreakdown(
            soiling_loss=soiling_loss,
            shading_loss=shading_loss,
            snow_loss=snow_loss,
            temperature_loss=temperature_loss,
            mismatch_loss=mismatch_loss,
            inverter_loss=inverter_loss,
        )
    else:
        loss_breakdown = LossBreakdown()

# Create configuration
project_info = ProjectInfo(
    project_name="Solar PV Project",
    location="San Francisco, CA",
    latitude=37.7749,
    longitude=-122.4194,
    commissioning_date=datetime(2024, 1, 1),
)

system_config = SystemConfiguration(
    capacity_dc=1000.0,
    capacity_ac=850.0,
    module_type=ModuleType.MONO_SI,
    module_efficiency=0.20,
    module_count=5000,
    tilt_angle=30.0,
    azimuth_angle=180.0,
)

# Initialize dashboard with custom losses
dashboard = EYADashboard(project_info, system_config, loss_breakdown)

# Calculate losses
with st.spinner("Analyzing system losses..."):
    losses_data = dashboard.losses_waterfall()

st.success("✅ Loss analysis complete!")

# Total Loss Overview
st.markdown("### 📊 Total System Loss")

col1, col2, col3 = st.columns(3)

with col1:
    total_loss = float(losses_data["Total System Loss"].replace("%", ""))
    st.metric("Total Loss", losses_data["Total System Loss"], delta=f"{total_loss - 20:.1f}% vs 20% typical")

with col2:
    remaining = 100 - total_loss
    st.metric("Effective Efficiency", f"{remaining:.2f}%", delta=f"{remaining - 80:.1f}% vs 80% typical")

with col3:
    # Calculate energy impact
    energy_data = dashboard.annual_energy_output()
    annual_ac = float(energy_data["Annual Totals"]["AC Energy"].replace(" kWh", "").replace(",", ""))
    theoretical_max = annual_ac / (remaining / 100)
    lost_energy = theoretical_max - annual_ac
    st.metric("Lost Energy", f"{lost_energy:,.0f} kWh/year")

# Loss Categories
st.markdown("### 📋 Loss Categories")

categories = losses_data["Loss Categories"]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Optical Losses", categories["Optical Losses"])
    st.caption("Soiling, Shading, Snow")

with col2:
    st.metric("Electrical Losses", categories["Electrical Losses"])
    st.caption("Mismatch, Wiring, Inverter")

with col3:
    st.metric("Degradation Losses", categories["Degradation Losses"])
    st.caption("LID, Nameplate, Age")

with col4:
    st.metric("Environmental Losses", categories["Environmental Losses"])
    st.caption("Temperature effects")

with col5:
    st.metric("Availability Losses", categories["System Availability Losses"])
    st.caption("Downtime, Maintenance")

# Detailed Loss Breakdown
st.markdown("### 🔍 Detailed Loss Breakdown")

import pandas as pd

detailed_losses = losses_data["Detailed Breakdown"]

loss_table_data = {
    "Loss Type": [],
    "Value (%)": [],
    "Category": [],
}

loss_mapping = {
    "soiling_loss": ("Soiling", "Optical"),
    "shading_loss": ("Shading", "Optical"),
    "snow_loss": ("Snow", "Optical"),
    "mismatch_loss": ("Mismatch", "Electrical"),
    "wiring_loss": ("Wiring", "Electrical"),
    "connection_loss": ("Connections", "Electrical"),
    "inverter_loss": ("Inverter", "Electrical"),
    "transformer_loss": ("Transformer", "Electrical"),
    "lid_loss": ("Light-Induced Degradation", "Degradation"),
    "nameplate_loss": ("Nameplate Rating", "Degradation"),
    "age_loss": ("Age-related", "Degradation"),
    "temperature_loss": ("Temperature", "Environmental"),
    "availability_loss": ("Availability", "System"),
}

for key, (name, category) in loss_mapping.items():
    loss_table_data["Loss Type"].append(name)
    loss_table_data["Value (%)"].append(f"{detailed_losses[key]:.2f}%")
    loss_table_data["Category"].append(category)

df_losses = pd.DataFrame(loss_table_data)
st.dataframe(df_losses, use_container_width=True, hide_index=True)

# Waterfall Visualization
st.markdown("### 📉 Loss Waterfall (Energy Flow)")

viz = InteractiveVisualizations()
waterfall_fig = viz.loss_breakdown_sankey(losses_data["waterfall_data"])
st.plotly_chart(waterfall_fig, use_container_width=True)

# Loss Impact Analysis
st.markdown("### 💰 Loss Impact Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top 3 Loss Contributors")

    loss_values = {k: v for k, v in detailed_losses.items() if k != "total_loss"}
    sorted_losses = sorted(loss_values.items(), key=lambda x: x[1], reverse=True)[:3]

    for i, (loss_name, loss_value) in enumerate(sorted_losses, 1):
        display_name = loss_mapping.get(loss_name, (loss_name, "Unknown"))[0]
        st.markdown(f"**{i}. {display_name}**: {loss_value:.2f}%")

with col2:
    st.markdown("#### Energy Recovery Potential")

    # Calculate potential energy recovery for top losses
    for loss_name, loss_value in sorted_losses:
        display_name = loss_mapping.get(loss_name, (loss_name, "Unknown"))[0]
        recoverable_energy = annual_ac * (loss_value / 100) / (1 - loss_value / 100)
        st.markdown(f"**{display_name}**: {recoverable_energy:,.0f} kWh/year")

# Mitigation Strategies
with st.expander("🛠️ Loss Mitigation Strategies"):
    st.markdown("""
    ### How to Reduce System Losses

    #### Optical Losses
    - **Soiling**: Implement regular cleaning schedule (monthly in dusty areas)
    - **Shading**: Conduct shading analysis, trim vegetation, optimize layout
    - **Snow**: Use steeper tilt angles in snowy climates, consider snow guards

    #### Electrical Losses
    - **Mismatch**: Use module-level power electronics (MLPE) or optimizers
    - **Wiring**: Minimize cable runs, use appropriate wire gauge
    - **Inverter**: Select high-efficiency inverters (>98%), proper sizing
    - **Transformer**: Consider transformer-less inverters or high-efficiency units

    #### Degradation Losses
    - **LID**: Select modules with low LID characteristics, consider LID-free technologies
    - **Nameplate**: Flash test modules, buy from reputable manufacturers
    - **Age**: Use high-quality modules with low degradation rates (<0.5%/year)

    #### Environmental Losses
    - **Temperature**: Improve ventilation, consider elevated mounting
    - Use modules with low temperature coefficients

    #### System Availability
    - Implement robust O&M program
    - Use real-time monitoring for quick fault detection
    - Have spare parts inventory for critical components
    """)

# Comparison with Industry Standards
with st.expander("📊 Industry Benchmarks"):
    st.markdown("""
    ### Typical Loss Ranges

    | Loss Type | Typical Range | Your Value |
    |-----------|---------------|------------|
    | Soiling | 1-5% | """ + f"{detailed_losses['soiling_loss']:.2f}%" + """ |
    | Shading | 0-5% | """ + f"{detailed_losses['shading_loss']:.2f}%" + """ |
    | Temperature | 3-8% | """ + f"{detailed_losses['temperature_loss']:.2f}%" + """ |
    | Mismatch | 1-3% | """ + f"{detailed_losses['mismatch_loss']:.2f}%" + """ |
    | Inverter | 2-4% | """ + f"{detailed_losses['inverter_loss']:.2f}%" + """ |
    | Total System | 15-25% | """ + f"{detailed_losses['total_loss']:.2f}%" + """ |

    **Note**: Lower values are better. Values outside typical ranges may indicate issues or opportunities for improvement.
    """)
