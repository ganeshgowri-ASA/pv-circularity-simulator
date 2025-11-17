"""
Thermal Modeling Dashboard - Temperature prediction and thermal analysis for PV modules.

This page provides comprehensive thermal modeling capabilities including:
- Multiple temperature prediction models (Sandia, PVsyst, Faiman, NOCT)
- B03 NOCT database integration
- Heat transfer coefficient calculations
- Wind speed and mounting configuration effects
- Interactive temperature charts and cooling analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from pv_simulator.core.cell_temperature import CellTemperatureModel, ModuleTemperatureCalculator
from pv_simulator.models.thermal import (
    TemperatureConditions,
    ThermalParameters,
    MountingConfiguration,
    TemperatureCoefficients,
)
from pv_simulator.data.loaders import load_b03_noct_database

# Import visualization components
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))
from components.thermal_viz import (
    plot_temperature_comparison,
    plot_wind_speed_effects,
    plot_mounting_configuration_effects,
    plot_temperature_time_series,
    plot_heat_transfer_coefficients,
    plot_thermal_time_constants,
)

# Page config
st.set_page_config(
    page_title="Thermal Modeling | PV Circularity Simulator",
    page_icon="🌡️",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b6b;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #ff6b6b;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "noct_loader" not in st.session_state:
    try:
        st.session_state.noct_loader = load_b03_noct_database()
    except Exception as e:
        st.session_state.noct_loader = None
        st.warning(f"Could not load B03 NOCT database: {e}")

# Header
st.markdown('<h1 class="main-header">🌡️ Thermal Modeling & Temperature Prediction</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
    <h4>🎯 About Thermal Modeling</h4>
    <p>
    Accurate temperature prediction is crucial for PV performance estimation. Module temperature
    affects power output, efficiency, and long-term reliability. This dashboard provides multiple
    industry-standard thermal models and comprehensive analysis tools.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Input Parameters
with st.sidebar:
    st.header("⚙️ Configuration")

    # Data source selection
    st.subheader("1️⃣ Module Selection")
    data_source = st.radio(
        "Data Source",
        ["B03 Database", "Manual Input"],
        help="Select B03 verified modules or enter custom parameters"
    )

    selected_module = None
    if data_source == "B03 Database" and st.session_state.noct_loader:
        # Load modules
        modules = st.session_state.noct_loader.get_b03_verified_modules()
        module_options = {f"{m.module_id} - {m.manufacturer} {m.model_name}": m for m in modules}

        selected_key = st.selectbox(
            "Select Module",
            options=list(module_options.keys()),
            help="Choose from B03 verified modules"
        )
        selected_module = module_options[selected_key]

        # Display module info
        with st.expander("📋 Module Information"):
            st.write(f"**Manufacturer:** {selected_module.manufacturer}")
            st.write(f"**Model:** {selected_module.model_name}")
            st.write(f"**Technology:** {selected_module.technology}")
            st.write(f"**NOCT:** {selected_module.noct_spec.noct_celsius}°C")
            st.write(f"**Power (STC):** {selected_module.rated_power_stc}W")
            st.write(f"**Efficiency:** {selected_module.efficiency_stc}%")
            st.write(f"**Area:** {selected_module.module_area}m²")

    # Environmental Conditions
    st.subheader("2️⃣ Environmental Conditions")

    ambient_temp = st.slider(
        "Ambient Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0,
        step=0.5,
        help="Air temperature around the module"
    )

    irradiance = st.slider(
        "Solar Irradiance (W/m²)",
        min_value=0.0,
        max_value=1200.0,
        value=1000.0,
        step=10.0,
        help="Solar irradiance on module plane"
    )

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.5,
        help="Wind speed at module height"
    )

    # Mounting Configuration
    st.subheader("3️⃣ Mounting Configuration")

    mounting_type = st.selectbox(
        "Mounting Type",
        ["open_rack", "roof_mounted", "ground_mounted", "building_integrated"],
        help="Mounting configuration affects cooling"
    )

    tilt_angle = st.slider(
        "Tilt Angle (degrees)",
        min_value=0.0,
        max_value=90.0,
        value=30.0,
        step=5.0,
        help="Module tilt from horizontal"
    )

    # Model Selection
    st.subheader("4️⃣ Model Selection")

    models_to_run = st.multiselect(
        "Temperature Models",
        ["Sandia", "PVsyst", "Faiman", "NOCT-based"],
        default=["Sandia", "PVsyst", "Faiman", "NOCT-based"],
        help="Select models to compare"
    )

# Create conditions and mounting objects
conditions = TemperatureConditions(
    ambient_temp=ambient_temp,
    irradiance=irradiance,
    wind_speed=wind_speed,
)

mounting = MountingConfiguration(
    mounting_type=mounting_type,
    tilt_angle=tilt_angle,
)

# Get thermal parameters
if selected_module:
    thermal_params = ThermalParameters(
        heat_capacity=selected_module.heat_capacity,
        absorptivity=selected_module.absorptivity,
        emissivity=selected_module.emissivity,
        module_area=selected_module.module_area,
    )
    noct_value = selected_module.noct_spec.noct_celsius
else:
    thermal_params = ThermalParameters()
    noct_value = 45.0  # Default NOCT

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Temperature Prediction",
    "🌬️ Cooling Analysis",
    "🔍 Heat Transfer",
    "📈 Time Series Analysis"
])

# Tab 1: Temperature Prediction
with tab1:
    st.markdown('<div class="section-header">Temperature Prediction Results</div>', unsafe_allow_html=True)

    # Run calculations
    temp_model = CellTemperatureModel(
        conditions=conditions,
        thermal_params=thermal_params,
        mounting=mounting,
    )

    results = {}

    if "Sandia" in models_to_run:
        results["Sandia"] = temp_model.sandia_model()

    if "PVsyst" in models_to_run:
        results["PVsyst"] = temp_model.pvsyst_model()

    if "Faiman" in models_to_run:
        results["Faiman"] = temp_model.faiman_model()

    if "NOCT-based" in models_to_run:
        results["NOCT-based"] = temp_model.noct_based(noct=noct_value)

    # Display results in columns
    cols = st.columns(len(results))

    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-box">
                <h4>{model_name} Model</h4>
                <h2 style="color: #ff6b6b;">{result.cell_temperature:.1f}°C</h2>
                <p>Cell Temperature</p>
                <hr>
                <p><strong>Module Temp:</strong> {result.module_temperature:.1f}°C</p>
                <p><strong>ΔT from Ambient:</strong> {result.cell_temperature - ambient_temp:.1f}°C</p>
            </div>
            """, unsafe_allow_html=True)

    # Comparison chart
    st.markdown("### Model Comparison")
    fig_comparison = plot_temperature_comparison(results, conditions)
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Performance impact
    if selected_module:
        st.markdown("### 📉 Performance Impact Analysis")

        # Calculate power loss due to temperature
        avg_cell_temp = np.mean([r.cell_temperature for r in results.values()])
        temp_rise = avg_cell_temp - 25.0  # STC reference
        power_loss_pct = temp_rise * selected_module.temp_coeff_power * 100
        power_loss_w = selected_module.rated_power_stc * power_loss_pct / 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Cell Temp", f"{avg_cell_temp:.1f}°C", f"+{temp_rise:.1f}°C from STC")
        col2.metric("Power Loss", f"{abs(power_loss_pct):.2f}%", f"{abs(power_loss_w):.1f}W")
        col3.metric("Temp Coefficient", f"{selected_module.temp_coeff_power*100:.3f}%/°C")
        col4.metric("Expected Power", f"{selected_module.rated_power_stc + power_loss_w:.1f}W")

# Tab 2: Cooling Analysis
with tab2:
    st.markdown('<div class="section-header">Cooling & Wind Speed Analysis</div>', unsafe_allow_html=True)

    calculator = ModuleTemperatureCalculator(
        thermal_params=thermal_params,
        mounting=mounting,
        conditions=conditions,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌬️ Wind Speed Effects")
        wind_results = calculator.wind_speed_effects(
            wind_speed_range=np.linspace(0, 15, 30),
            base_temp=ambient_temp,
        )
        fig_wind = plot_wind_speed_effects(wind_results)
        st.plotly_chart(fig_wind, use_container_width=True)

        st.info(f"""
        **Current Wind Speed:** {wind_speed:.1f} m/s

        **Analysis:** Higher wind speeds increase convective cooling, reducing module temperature.
        At {wind_speed:.1f} m/s, the convective heat transfer coefficient is
        {calculator.heat_transfer_coefficients(wind_speed).convective_front:.1f} W/(m²·K).
        """)

    with col2:
        st.markdown("### 🏗️ Mounting Configuration Effects")
        mounting_results = calculator.mounting_configuration_effects(
            irradiance=irradiance,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
        )
        fig_mounting = plot_mounting_configuration_effects(mounting_results)
        st.plotly_chart(fig_mounting, use_container_width=True)

        # Display comparison table
        st.markdown("#### Configuration Comparison")
        display_df = mounting_results.copy()
        display_df.columns = ["Mounting", "Sandia (°C)", "PVsyst (°C)", "NOCT (°C)", "Average (°C)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# Tab 3: Heat Transfer Analysis
with tab3:
    st.markdown('<div class="section-header">Heat Transfer Coefficients</div>', unsafe_allow_html=True)

    coeffs = calculator.heat_transfer_coefficients(wind_speed)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Heat Transfer Breakdown")
        fig_coeffs = plot_heat_transfer_coefficients(coeffs)
        st.plotly_chart(fig_coeffs, use_container_width=True)

    with col2:
        st.markdown("### 📋 Detailed Coefficients")

        st.markdown(f"""
        <div class="metric-box">
            <h4>Front Surface</h4>
            <p><strong>Convective:</strong> {coeffs.convective_front:.2f} W/(m²·K)</p>
            <p><strong>Radiative:</strong> {coeffs.radiative_front:.2f} W/(m²·K)</p>
            <p><strong>Total:</strong> {coeffs.total_front:.2f} W/(m²·K)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h4>Back Surface</h4>
            <p><strong>Convective:</strong> {coeffs.convective_back:.2f} W/(m²·K)</p>
            <p><strong>Radiative:</strong> {coeffs.radiative_back:.2f} W/(m²·K)</p>
            <p><strong>Total:</strong> {coeffs.total_back:.2f} W/(m²·K)</p>
        </div>
        """, unsafe_allow_html=True)

        # Back surface temperature
        avg_cell_temp = np.mean([r.cell_temperature for r in results.values()]) if results else 50.0
        back_temp = calculator.back_surface_temperature(
            front_surface_temp=avg_cell_temp,
            irradiance=irradiance,
            ambient_temp=ambient_temp,
        )

        st.markdown(f"""
        <div class="metric-box">
            <h4>Temperature Gradient</h4>
            <p><strong>Front (Cell):</strong> {avg_cell_temp:.1f}°C</p>
            <p><strong>Back Surface:</strong> {back_temp:.1f}°C</p>
            <p><strong>ΔT (Front-Back):</strong> {avg_cell_temp - back_temp:.1f}°C</p>
        </div>
        """, unsafe_allow_html=True)

    # Thermal time constants
    st.markdown("### ⏱️ Thermal Response Time")

    tau_results = calculator.thermal_time_constants(wind_speed)
    fig_tau = plot_thermal_time_constants(tau_results)
    st.plotly_chart(fig_tau, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Heating Time Constant", f"{tau_results['tau_heating_minutes']:.1f} min")
    col2.metric("Cooling Time Constant", f"{tau_results['tau_cooling_minutes']:.1f} min")
    col3.metric("63% Response Time", f"{tau_results['response_time_63']/60:.1f} min")
    col4.metric("95% Response Time", f"{tau_results['response_time_95']/60:.1f} min")

    st.info("""
    **Thermal Time Constants** indicate how quickly the module responds to environmental changes:
    - **Heating τ**: Time to reach 63% of final temperature when irradiance increases
    - **Cooling τ**: Time to cool down to 37% of initial temperature difference
    - Lower values = faster response to changing conditions
    """)

# Tab 4: Time Series Analysis
with tab4:
    st.markdown('<div class="section-header">Time Series Temperature Analysis</div>', unsafe_allow_html=True)

    st.markdown("### 📅 Daily Temperature Profile")

    # Generate time series data
    hours = np.arange(0, 24, 0.5)

    # Simulate daily irradiance profile (simplified)
    irradiance_profile = np.maximum(0, 1000 * np.sin(np.pi * (hours - 6) / 12))

    # Ambient temperature profile (simplified)
    temp_profile = ambient_temp + 5 * np.sin(np.pi * (hours - 6) / 12) - 2

    # Wind speed variation
    wind_profile = wind_speed + 2 * np.sin(np.pi * hours / 12)
    wind_profile = np.maximum(0.5, wind_profile)

    # Calculate temperatures
    cell_temps_sandia = []
    cell_temps_pvsyst = []
    cell_temps_noct = []

    for hour, irr, temp, wind in zip(hours, irradiance_profile, temp_profile, wind_profile):
        cond = TemperatureConditions(
            ambient_temp=float(temp),
            irradiance=float(irr),
            wind_speed=float(wind),
        )
        model = CellTemperatureModel(cond, thermal_params, mounting)

        cell_temps_sandia.append(model.sandia_model().cell_temperature)
        cell_temps_pvsyst.append(model.pvsyst_model().cell_temperature)
        cell_temps_noct.append(model.noct_based(noct=noct_value).cell_temperature)

    # Create time series dataframe
    ts_df = pd.DataFrame({
        'hour': hours,
        'irradiance': irradiance_profile,
        'ambient_temp': temp_profile,
        'wind_speed': wind_profile,
        'cell_temp_sandia': cell_temps_sandia,
        'cell_temp_pvsyst': cell_temps_pvsyst,
        'cell_temp_noct': cell_temps_noct,
    })

    # Plot time series
    fig_ts = plot_temperature_time_series(ts_df)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Summary statistics
    st.markdown("### 📊 Daily Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Peak Cell Temp",
        f"{max(cell_temps_sandia):.1f}°C",
        f"at {hours[np.argmax(cell_temps_sandia)]:.1f}h"
    )
    col2.metric(
        "Avg Cell Temp",
        f"{np.mean(cell_temps_sandia):.1f}°C",
        f"(daylight hours)"
    )
    col3.metric(
        "Peak Irradiance",
        f"{max(irradiance_profile):.0f} W/m²",
        f"at {hours[np.argmax(irradiance_profile)]:.1f}h"
    )
    col4.metric(
        "Avg Wind Speed",
        f"{np.mean(wind_profile):.1f} m/s",
        f"±{np.std(wind_profile):.1f} m/s"
    )

    # Download data
    st.markdown("### 💾 Export Data")
    csv = ts_df.to_csv(index=False)
    st.download_button(
        label="Download Time Series Data (CSV)",
        data=csv,
        file_name=f"temperature_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <small>
    Thermal models based on: King et al. (2004), Faiman (2008), Mermoud (2012) |
    B03 NOCT Database | pvlib integration
    </small>
</div>
""", unsafe_allow_html=True)
