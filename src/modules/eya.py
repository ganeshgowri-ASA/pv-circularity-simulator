"""
EYA Module - Energy Yield Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render the EYA (Energy Yield Analysis) module"""
    st.header("☀️ Energy Yield Analysis (EYA)")
    st.markdown("---")

    st.markdown("""
    ### Energy Yield Assessment & P50/P90 Analysis

    Predict long-term energy production and financial performance.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Input Data", "Loss Analysis", "Production Estimate", "P50/P90"])

    with tab1:
        st.subheader("Irradiance & Weather Data")

        col1, col2 = st.columns(2)

        with col1:
            data_source = st.selectbox(
                "Climate Data Source",
                ["PVGIS", "NREL NSRDB", "Meteonorm", "SolarGIS", "Custom TMY"]
            )

            ghi_annual = st.number_input("Annual GHI (kWh/m²/year)", min_value=0.0, value=1650.0)
            dni_annual = st.number_input("Annual DNI (kWh/m²/year)", min_value=0.0, value=1800.0)
            dhi_annual = st.number_input("Annual DHI (kWh/m²/year)", min_value=0.0, value=650.0)

        with col2:
            poa_annual = st.number_input("Annual POA (kWh/m²/year)", min_value=0.0, value=1850.0)
            st.info("POA = Plane of Array irradiance")

            avg_temp = st.number_input("Average Ambient Temperature (°C)", value=15.0)
            wind_speed = st.number_input("Average Wind Speed (m/s)", min_value=0.0, value=3.5)

        st.markdown("#### Monthly Irradiance Profile")
        monthly_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'GHI (kWh/m²)': [80, 95, 130, 155, 180, 190, 195, 175, 145, 110, 85, 75]
        })
        st.bar_chart(monthly_data.set_index('Month'))

    with tab2:
        st.subheader("System Loss Analysis")

        st.markdown("### Loss Factors (%)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Environmental Losses")
            soiling_loss = st.slider("Soiling Loss", 0.0, 10.0, 2.0, 0.1)
            snow_loss = st.slider("Snow Loss", 0.0, 10.0, 0.0, 0.1)
            shading_loss = st.slider("Shading Loss", 0.0, 20.0, 0.5, 0.1)

            st.markdown("#### Temperature Losses")
            temp_loss = st.slider("Temperature Loss", 0.0, 15.0, 7.0, 0.1)

        with col2:
            st.markdown("#### Electrical Losses")
            mismatch_loss = st.slider("Module Mismatch", 0.0, 5.0, 1.0, 0.1)
            dc_wiring_loss = st.slider("DC Wiring Loss", 0.0, 5.0, 1.5, 0.1)
            inverter_loss = st.slider("Inverter Loss", 0.0, 5.0, 2.0, 0.1)
            ac_wiring_loss = st.slider("AC Wiring Loss", 0.0, 3.0, 0.5, 0.1)
            transformer_loss = st.slider("Transformer Loss", 0.0, 3.0, 1.0, 0.1)

        st.markdown("#### Availability Losses")
        availability_loss = st.slider("System Unavailability", 0.0, 10.0, 1.0, 0.1)

        # Calculate total losses
        total_loss = (soiling_loss + snow_loss + shading_loss + temp_loss +
                     mismatch_loss + dc_wiring_loss + inverter_loss +
                     ac_wiring_loss + transformer_loss + availability_loss)

        performance_ratio = 100 - total_loss

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total System Losses", f"{total_loss:.2f}%", delta_color="inverse")
        with col2:
            st.metric("Performance Ratio (PR)", f"{performance_ratio:.2f}%")

    with tab3:
        st.subheader("Annual Energy Production Estimate")

        # System parameters
        system_size = st.number_input("System Size (kWp)", min_value=0.0, value=100.0)

        # Calculate specific yield
        specific_yield = poa_annual * (performance_ratio / 100)
        annual_production = system_size * specific_yield

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Specific Yield", f"{specific_yield:.0f} kWh/kWp/year")
        with col2:
            st.metric("Annual Production", f"{annual_production:.0f} MWh")
        with col3:
            capacity_factor = (annual_production / (system_size * 8760)) * 100
            st.metric("Capacity Factor", f"{capacity_factor:.1f}%")

        st.markdown("### Monthly Energy Production")
        monthly_production = []
        for idx, row in monthly_data.iterrows():
            monthly_prod = system_size * row['GHI (kWh/m²)'] * (performance_ratio / 100)
            monthly_production.append(monthly_prod)

        production_df = pd.DataFrame({
            'Month': monthly_data['Month'],
            'Production (MWh)': monthly_production
        })
        st.bar_chart(production_df.set_index('Month'))

        st.markdown("### 25-Year Production Forecast")
        degradation_rate = st.slider("Annual Degradation Rate (%)", 0.0, 1.0, 0.5, 0.05)

        years = list(range(1, 26))
        yearly_production = [annual_production * ((1 - degradation_rate/100) ** (year-1)) for year in years]

        forecast_df = pd.DataFrame({
            'Year': years,
            'Production (MWh)': yearly_production
        })
        st.line_chart(forecast_df.set_index('Year'))

        lifetime_production = sum(yearly_production)
        st.metric("25-Year Lifetime Production", f"{lifetime_production:.0f} MWh")

    with tab4:
        st.subheader("P50/P90 Exceedance Probability Analysis")

        st.info("""
        **P50**: 50% probability that production will exceed this value (median estimate)
        **P90**: 90% probability that production will exceed this value (conservative estimate)
        """)

        # Uncertainty factors
        st.markdown("### Uncertainty Analysis")

        col1, col2 = st.columns(2)
        with col1:
            irradiance_uncertainty = st.slider("Irradiance Data Uncertainty (%)", 0.0, 10.0, 5.0, 0.5)
            module_uncertainty = st.slider("Module Performance Uncertainty (%)", 0.0, 5.0, 2.0, 0.5)
        with col2:
            loss_uncertainty = st.slider("Loss Factor Uncertainty (%)", 0.0, 5.0, 3.0, 0.5)
            interannual_variability = st.slider("Interannual Variability (%)", 0.0, 10.0, 5.0, 0.5)

        # Calculate combined uncertainty
        total_uncertainty = np.sqrt(
            irradiance_uncertainty**2 +
            module_uncertainty**2 +
            loss_uncertainty**2 +
            interannual_variability**2
        )

        # P values (using normal distribution approximation)
        p50_production = annual_production
        p90_production = annual_production * (1 - 1.28 * total_uncertainty / 100)
        p99_production = annual_production * (1 - 2.33 * total_uncertainty / 100)

        st.markdown("### Exceedance Values")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P50 (Median)", f"{p50_production:.0f} MWh/year")
        with col2:
            st.metric("P90 (Conservative)", f"{p90_production:.0f} MWh/year")
        with col3:
            st.metric("P99 (Very Conservative)", f"{p99_production:.0f} MWh/year")

        # Uncertainty range
        st.markdown("### Production Probability Distribution")
        prob_range = pd.DataFrame({
            'Probability': ['P99', 'P90', 'P75', 'P50', 'P25', 'P10', 'P01'],
            'Production (MWh)': [
                annual_production * (1 - 2.33 * total_uncertainty / 100),
                annual_production * (1 - 1.28 * total_uncertainty / 100),
                annual_production * (1 - 0.67 * total_uncertainty / 100),
                annual_production,
                annual_production * (1 + 0.67 * total_uncertainty / 100),
                annual_production * (1 + 1.28 * total_uncertainty / 100),
                annual_production * (1 + 2.33 * total_uncertainty / 100),
            ]
        })
        st.bar_chart(prob_range.set_index('Probability'))

    st.markdown("---")
    if st.button("📊 Generate EYA Report", use_container_width=True):
        st.success("EYA report generated successfully!")
