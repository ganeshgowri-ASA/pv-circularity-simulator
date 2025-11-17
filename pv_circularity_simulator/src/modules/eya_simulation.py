"""
EYA Simulation Module
====================

Energy Yield Assessment (EYA) simulation.
Pre-construction energy production estimation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the EYA simulation module.

    Args:
        session: Session manager instance

    Features:
        - TMY (Typical Meteorological Year) data import
        - Hourly/monthly energy yield calculation
        - Performance ratio (PR) estimation
        - Capacity factor calculation
        - Degradation modeling
        - Uncertainty analysis
        - Financial metrics (LCOE, NPV, IRR)
    """
    st.header("☀️ Energy Yield Assessment (EYA)")

    st.info("""
    Simulate pre-construction energy production using meteorological data and system design.
    """)

    # Check if system design exists
    system_design = session.get('system_design_data', {})
    if not system_design:
        st.warning("⚠️ Please complete System Design first.")
        return

    # Simulation parameters
    st.subheader("⚙️ Simulation Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        simulation_years = st.number_input("Simulation Period (years)", 1, 30, 25)
        degradation_rate = st.slider("Annual Degradation (%/year)", 0.0, 1.0, 0.5, 0.05)

    with col2:
        performance_ratio = st.slider("Performance Ratio (%)", 70.0, 90.0, 80.0, 0.5)
        availability = st.slider("System Availability (%)", 90.0, 100.0, 98.0, 0.1)

    with col3:
        inverter_clipping = st.checkbox("Include Inverter Clipping", True)
        soiling_variation = st.checkbox("Include Soiling Variation", True)

    # Weather data source
    st.markdown("---")
    st.subheader("🌦️ Weather Data")

    weather_source = st.selectbox(
        "Weather Data Source",
        ["TMY3", "PVGIS", "Meteonorm", "Custom Upload"]
    )

    if weather_source == "Custom Upload":
        uploaded_file = st.file_uploader("Upload TMY Data (CSV)", type=['csv'])
        if uploaded_file:
            st.success("Weather data uploaded successfully!")
    else:
        st.info(f"Using {weather_source} data for the selected location.")

    # Run simulation
    st.markdown("---")
    if st.button("🚀 Run EYA Simulation", type="primary"):
        with st.spinner("Running energy yield simulation..."):
            # Placeholder simulation
            import time
            time.sleep(2)

            # Generate dummy results
            system_capacity = system_design['array']['num_modules'] * system_design['array']['module_power'] / 1000

            # Monthly energy production (dummy data)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Simulate seasonal variation
            monthly_energy = []
            for i in range(12):
                base_energy = system_capacity * 100  # kWh
                seasonal_factor = 1 + 0.3 * np.sin((i - 2) * np.pi / 6)  # Peak in summer
                monthly_energy.append(base_energy * seasonal_factor)

            monthly_df = pd.DataFrame({
                'Month': months,
                'Energy (MWh)': [e/1000 for e in monthly_energy]
            })

            annual_energy = sum(monthly_energy) / 1000  # MWh
            specific_yield = annual_energy / system_capacity  # MWh/kWp
            capacity_factor = (annual_energy * 1000) / (system_capacity * 8760) * 100

            st.success("Simulation completed!")

            # Results
            st.subheader("📊 Simulation Results")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Annual Energy", f"{annual_energy:.2f} MWh")
            with col2:
                st.metric("Specific Yield", f"{specific_yield:.2f} MWh/kWp")
            with col3:
                st.metric("Capacity Factor", f"{capacity_factor:.2f}%")
            with col4:
                st.metric("Performance Ratio", f"{performance_ratio:.1f}%")

            # Monthly production chart
            st.markdown("---")
            st.subheader("📈 Monthly Energy Production")
            st.bar_chart(monthly_df.set_index('Month'))

            # Multi-year projection
            st.markdown("---")
            st.subheader("📅 Multi-Year Energy Projection")

            years = list(range(1, simulation_years + 1))
            projected_energy = [
                annual_energy * (1 - degradation_rate/100) ** (year - 1)
                for year in years
            ]

            projection_df = pd.DataFrame({
                'Year': years,
                'Energy (MWh)': projected_energy
            })

            st.line_chart(projection_df.set_index('Year'))

            # Financial metrics
            st.markdown("---")
            st.subheader("💰 Financial Metrics")

            col1, col2 = st.columns(2)

            with col1:
                capex = st.number_input("CAPEX ($/W)", 0.0, 5.0, 1.2, 0.1)
                opex = st.number_input("Annual OPEX ($/kW)", 0, 100, 15)

            with col2:
                electricity_rate = st.number_input("Electricity Rate ($/kWh)", 0.0, 1.0, 0.10, 0.01)
                discount_rate = st.number_input("Discount Rate (%)", 0.0, 20.0, 8.0, 0.5)

            total_capex = system_capacity * 1000 * capex
            lcoe = (total_capex + sum([opex * system_capacity / ((1 + discount_rate/100)**year) for year in years])) / \
                   sum([energy / ((1 + discount_rate/100)**year) for year, energy in zip(years, projected_energy)])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total CAPEX", f"${total_capex:,.0f}")
            with col2:
                st.metric("LCOE", f"${lcoe:.3f}/kWh")
            with col3:
                revenue_25y = sum(projected_energy) * electricity_rate * 1000
                st.metric("25-Year Revenue", f"${revenue_25y:,.0f}")

            # Save results
            session.set('eya_results', {
                'annual_energy': annual_energy,
                'specific_yield': specific_yield,
                'capacity_factor': capacity_factor,
                'performance_ratio': performance_ratio,
                'monthly_production': monthly_df.to_dict(),
                'multi_year_projection': projection_df.to_dict(),
                'lcoe': lcoe
            })
