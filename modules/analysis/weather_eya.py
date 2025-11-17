"""
Weather Data & Energy Yield Assessment Module (Branch B06).

Features:
- Location-based weather data
- GHI, DNI, DHI calculations
- P50/P90 energy yield assessment
- Climate zone analysis
- Soiling and degradation modeling
- Performance ratio predictions
- Long-term energy production forecasts
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import CLIMATE_ZONES
from utils.validators import LocationData, EnergyForecast
from utils.helpers import (
    calculate_poa_irradiance,
    calculate_performance_ratio,
    calculate_specific_yield,
    create_performance_chart
)


class WeatherEnergyAnalyzer:
    """Weather data analysis and energy yield assessment."""

    def __init__(self):
        """Initialize weather and energy analyzer."""
        self.climate_zones = CLIMATE_ZONES

    def generate_weather_data(
        self,
        latitude: float,
        longitude: float,
        climate_zone: str,
        num_days: int = 365
    ) -> pd.DataFrame:
        """
        Generate synthetic weather data based on location and climate.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            climate_zone: Climate zone classification
            num_days: Number of days to generate

        Returns:
            DataFrame with hourly weather data
        """
        # Get climate parameters
        climate_params = self.climate_zones.get(climate_zone, self.climate_zones["temperate"])

        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(num_days * 24)]

        # Initialize arrays
        ghi_values = []
        dni_values = []
        dhi_values = []
        temp_values = []
        wind_values = []

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday

            # Solar elevation (simplified)
            solar_declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
            hour_angle = 15 * (hour - 12)

            sin_elevation = (np.sin(np.radians(latitude)) * np.sin(np.radians(solar_declination)) +
                           np.cos(np.radians(latitude)) * np.cos(np.radians(solar_declination)) *
                           np.cos(np.radians(hour_angle)))

            solar_elevation = np.degrees(np.arcsin(max(0, sin_elevation)))

            # GHI based on solar elevation and climate
            if solar_elevation > 0:
                clear_sky_ghi = climate_params["avg_ghi"] / 365 / 5  # Daily average / peak sun hours
                ghi = clear_sky_ghi * (sin_elevation ** 1.5) * np.random.uniform(0.7, 1.0)

                # Cloud variation
                if np.random.random() < 0.3:  # 30% chance of clouds
                    ghi *= np.random.uniform(0.3, 0.7)
            else:
                ghi = 0

            ghi_values.append(ghi)

            # DNI and DHI decomposition
            if ghi > 0:
                clearness_index = min(1.0, ghi / (1000 * sin_elevation))
                if clearness_index > 0.65:
                    dni_fraction = 0.8
                else:
                    dni_fraction = 0.4

                dni = ghi * dni_fraction / max(0.1, sin_elevation)
                dhi = ghi - dni * sin_elevation
            else:
                dni = 0
                dhi = 0

            dni_values.append(dni)
            dhi_values.append(dhi)

            # Temperature (seasonal variation)
            temp_min, temp_max = climate_params["temp_range"]
            temp_avg = (temp_min + temp_max) / 2
            temp_amplitude = (temp_max - temp_min) / 2

            # Seasonal variation
            seasonal_temp = temp_avg + temp_amplitude * np.sin(np.radians((360/365) * (day_of_year - 80)))

            # Daily variation
            daily_variation = 5 * np.sin(np.radians(360 * (hour - 6) / 24))

            temp = seasonal_temp + daily_variation + np.random.normal(0, 2)
            temp_values.append(temp)

            # Wind speed
            wind = np.random.gamma(2, 2) + np.random.uniform(-1, 1)
            wind_values.append(max(0, wind))

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ghi': ghi_values,
            'dni': dni_values,
            'dhi': dhi_values,
            'temp_ambient': temp_values,
            'wind_speed': wind_values
        })

        return df

    def calculate_poa_irradiance_series(
        self,
        weather_df: pd.DataFrame,
        latitude: float,
        tilt: float,
        azimuth: float
    ) -> pd.DataFrame:
        """
        Calculate plane-of-array irradiance for entire time series.

        Args:
            weather_df: Weather data DataFrame
            latitude: Location latitude
            tilt: Panel tilt angle
            azimuth: Panel azimuth angle

        Returns:
            DataFrame with POA irradiance added
        """
        poa_values = []

        for idx, row in weather_df.iterrows():
            ts = row['timestamp']
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour + ts.minute / 60

            # Solar position (simplified)
            solar_declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
            hour_angle = 15 * (hour - 12)

            # Solar elevation and azimuth
            sin_elevation = (np.sin(np.radians(latitude)) * np.sin(np.radians(solar_declination)) +
                           np.cos(np.radians(latitude)) * np.cos(np.radians(solar_declination)) *
                           np.cos(np.radians(hour_angle)))

            solar_elevation = max(0, np.degrees(np.arcsin(max(-1, min(1, sin_elevation)))))

            # Simplified solar azimuth
            solar_azimuth = 180 if hour < 12 else 180

            # Calculate POA
            if solar_elevation > 0:
                poa = calculate_poa_irradiance(
                    row['ghi'], row['dni'], row['dhi'],
                    90 - solar_elevation, solar_azimuth,
                    tilt, azimuth
                )
            else:
                poa = 0

            poa_values.append(poa)

        weather_df['poa_irradiance'] = poa_values
        return weather_df

    def calculate_energy_yield(
        self,
        weather_df: pd.DataFrame,
        system_capacity: float,
        module_efficiency: float,
        inverter_efficiency: float,
        temp_coefficient: float = -0.004,
        soiling_loss: float = 0.02,
        degradation_rate: float = 0.005,
        system_age: int = 0
    ) -> Dict[str, any]:
        """
        Calculate energy yield from weather data.

        Args:
            weather_df: Weather data with POA irradiance
            system_capacity: System DC capacity (kW)
            module_efficiency: Module efficiency (fraction)
            inverter_efficiency: Inverter efficiency (fraction)
            temp_coefficient: Temperature coefficient (%/°C)
            soiling_loss: Soiling loss factor (fraction)
            degradation_rate: Annual degradation rate (fraction)
            system_age: System age (years)

        Returns:
            Energy yield analysis
        """
        # Calculate module temperature (NOCT model)
        noct = 45
        weather_df['temp_module'] = weather_df['temp_ambient'] + (noct - 20) * (weather_df['poa_irradiance'] / 800)

        # Temperature derating
        weather_df['temp_derating'] = 1 + temp_coefficient * (weather_df['temp_module'] - 25)

        # Power output calculation
        weather_df['dc_power'] = (
            system_capacity *
            (weather_df['poa_irradiance'] / 1000) *
            weather_df['temp_derating'] *
            (1 - soiling_loss) *
            ((1 - degradation_rate) ** system_age)
        )

        # AC power
        weather_df['ac_power'] = weather_df['dc_power'] * inverter_efficiency

        # Clip to inverter capacity
        inverter_capacity = system_capacity / 1.25
        weather_df['ac_power'] = weather_df['ac_power'].clip(upper=inverter_capacity)

        # Calculate energy (kWh)
        weather_df['ac_energy'] = weather_df['ac_power']  # 1-hour intervals

        # Summary statistics
        total_energy = weather_df['ac_energy'].sum()
        capacity_factor = total_energy / (system_capacity * len(weather_df))
        specific_yield = total_energy / system_capacity

        # Performance ratio
        total_insolation = weather_df['poa_irradiance'].sum() / 1000  # kWh/m²
        reference_yield = total_insolation
        final_yield = specific_yield
        pr = final_yield / reference_yield if reference_yield > 0 else 0

        return {
            'weather_df': weather_df,
            'total_energy_kwh': total_energy,
            'daily_average_kwh': total_energy / (len(weather_df) / 24),
            'monthly_energy_kwh': weather_df.groupby(weather_df['timestamp'].dt.month)['ac_energy'].sum().to_dict(),
            'capacity_factor': capacity_factor,
            'specific_yield_kwh_kwp': specific_yield,
            'performance_ratio': pr,
            'peak_power_kw': weather_df['ac_power'].max(),
            'avg_power_kw': weather_df['ac_power'].mean()
        }

    def calculate_p50_p90(
        self,
        base_yield: float,
        uncertainty_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate P50/P90 energy yield estimates.

        Args:
            base_yield: Base case energy yield (kWh)
            uncertainty_factors: Dictionary of uncertainty factors

        Returns:
            P50, P90, and P99 estimates
        """
        if uncertainty_factors is None:
            uncertainty_factors = {
                'irradiance': 0.05,  # 5% uncertainty
                'temperature': 0.02,
                'soiling': 0.03,
                'degradation': 0.02,
                'equipment': 0.03,
                'availability': 0.02
            }

        # Calculate total uncertainty (RSS - Root Sum Square)
        total_uncertainty = np.sqrt(sum(u**2 for u in uncertainty_factors.values()))

        # P-values (assuming normal distribution)
        # P50 = median (50th percentile) = base case
        # P90 = 90th percentile (1.28 standard deviations below mean)
        # P99 = 99th percentile (2.33 standard deviations below mean)

        p50 = base_yield
        p90 = base_yield * (1 - 1.28 * total_uncertainty)
        p99 = base_yield * (1 - 2.33 * total_uncertainty)
        p10 = base_yield * (1 + 1.28 * total_uncertainty)

        return {
            'P99': p99,
            'P90': p90,
            'P50': p50,
            'P10': p10,
            'uncertainty': total_uncertainty,
            'uncertainty_factors': uncertainty_factors
        }

    def model_degradation(
        self,
        initial_power: float,
        degradation_rate: float,
        years: int = 25
    ) -> Dict[str, List[float]]:
        """
        Model long-term degradation.

        Args:
            initial_power: Initial system power (kW)
            degradation_rate: Annual degradation rate (fraction)
            years: Number of years to model

        Returns:
            Degradation model data
        """
        years_array = np.arange(0, years + 1)

        # Linear degradation
        linear_power = initial_power * (1 - degradation_rate * years_array)

        # With LID in first year
        lid_loss = 0.02  # 2% LID in first year
        lid_power = initial_power * (1 - lid_loss) * (1 - degradation_rate * (years_array - 1))
        lid_power[0] = initial_power  # Before LID

        # Energy production over lifetime
        annual_energy = []
        cumulative_energy = []
        total = 0

        for year in years_array:
            if year == 0:
                annual = 0
            else:
                power = lid_power[year]
                annual = power * 8760 * 0.20  # Assuming 20% capacity factor
            annual_energy.append(annual)
            total += annual
            cumulative_energy.append(total)

        return {
            'years': years_array.tolist(),
            'linear_power': linear_power.tolist(),
            'power_with_lid': lid_power.tolist(),
            'annual_energy': annual_energy,
            'cumulative_energy': cumulative_energy
        }

    def forecast_energy_production(
        self,
        historical_weather: pd.DataFrame,
        forecast_days: int = 30,
        method: str = "statistical"
    ) -> Dict[str, any]:
        """
        Forecast future energy production.

        Args:
            historical_weather: Historical weather data
            forecast_days: Number of days to forecast
            method: Forecasting method

        Returns:
            Energy production forecast
        """
        # Simple statistical forecast based on historical patterns
        daily_energy = historical_weather.groupby(historical_weather['timestamp'].dt.date)['ac_energy'].sum()

        # Calculate seasonal pattern
        historical_weather['day_of_year'] = historical_weather['timestamp'].dt.dayofyear
        seasonal_pattern = historical_weather.groupby('day_of_year')['ac_energy'].mean()

        # Generate forecast
        last_date = historical_weather['timestamp'].max().date()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

        forecast_energy = []
        lower_bound = []
        upper_bound = []

        for date in forecast_dates:
            day_of_year = date.timetuple().tm_yday

            # Get seasonal baseline
            if day_of_year in seasonal_pattern.index:
                baseline = seasonal_pattern[day_of_year] * 24  # Convert hourly to daily
            else:
                baseline = seasonal_pattern.mean() * 24

            # Add uncertainty
            std_dev = daily_energy.std()
            forecast_energy.append(baseline)
            lower_bound.append(baseline - 1.96 * std_dev)
            upper_bound.append(baseline + 1.96 * std_dev)

        return {
            'forecast_dates': forecast_dates,
            'predicted_energy': forecast_energy,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': method,
            'confidence_level': 0.95
        }


def render_weather_eya():
    """Render weather and energy yield assessment interface in Streamlit."""
    st.header("🌤️ Weather Data & Energy Yield Assessment")
    st.markdown("Comprehensive weather analysis and long-term energy production forecasting.")

    analyzer = WeatherEnergyAnalyzer()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Location & Weather",
        "⚡ Energy Yield",
        "📊 P50/P90 Analysis",
        "📉 Degradation",
        "🔮 Forecast"
    ])

    with tab1:
        st.subheader("Location-Based Weather Data")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.slider("Latitude:", -90, 90, 35)
            longitude = st.slider("Longitude:", -180, 180, -95)

        with col2:
            climate_zone = st.selectbox("Climate Zone:", list(analyzer.climate_zones.keys()))
            num_days = st.slider("Simulation Days:", 30, 365, 365, 30)

        if st.button("🌤️ Generate Weather Data", key="gen_weather"):
            with st.spinner(f"Generating {num_days} days of weather data..."):
                weather_df = analyzer.generate_weather_data(latitude, longitude, climate_zone, num_days)

            st.success(f"✅ Generated {len(weather_df):,} hours of weather data")

            # Store in session state
            st.session_state['weather_df'] = weather_df

            # Climate zone info
            climate_info = analyzer.climate_zones[climate_zone]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average GHI", f"{climate_info['avg_ghi']} kWh/m²/year")

            with col2:
                temp_range = climate_info['temp_range']
                st.metric("Temperature Range", f"{temp_range[0]}°C to {temp_range[1]}°C")

            with col3:
                st.metric("Humidity", f"{climate_info['humidity'] * 100:.0f}%")

            # Weather summary statistics
            st.subheader("Weather Summary")

            summary_stats = {
                'Metric': ['GHI', 'DNI', 'DHI', 'Temperature', 'Wind Speed'],
                'Mean': [
                    f"{weather_df['ghi'].mean():.1f} W/m²",
                    f"{weather_df['dni'].mean():.1f} W/m²",
                    f"{weather_df['dhi'].mean():.1f} W/m²",
                    f"{weather_df['temp_ambient'].mean():.1f}°C",
                    f"{weather_df['wind_speed'].mean():.1f} m/s"
                ],
                'Max': [
                    f"{weather_df['ghi'].max():.1f} W/m²",
                    f"{weather_df['dni'].max():.1f} W/m²",
                    f"{weather_df['dhi'].max():.1f} W/m²",
                    f"{weather_df['temp_ambient'].max():.1f}°C",
                    f"{weather_df['wind_speed'].max():.1f} m/s"
                ]
            }

            st.dataframe(pd.DataFrame(summary_stats), use_container_width=True)

            # Plot weather data
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Global Horizontal Irradiance', 'Temperature',
                              'Wind Speed', 'Irradiance Components')
            )

            # Resample to daily for better visualization
            daily_weather = weather_df.resample('D', on='timestamp').mean()

            # GHI
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['ghi'],
                          name='GHI', line=dict(color='#F39C12')),
                row=1, col=1
            )

            # Temperature
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['temp_ambient'],
                          name='Temperature', line=dict(color='#E74C3C')),
                row=1, col=2
            )

            # Wind
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['wind_speed'],
                          name='Wind', line=dict(color='#3498DB')),
                row=2, col=1
            )

            # Irradiance components
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['ghi'],
                          name='GHI', line=dict(color='#F39C12')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['dni'],
                          name='DNI', line=dict(color='#E74C3C')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=daily_weather.index, y=daily_weather['dhi'],
                          name='DHI', line=dict(color='#3498DB')),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)

            fig.update_yaxes(title_text="GHI (W/m²)", row=1, col=1)
            fig.update_yaxes(title_text="Temp (°C)", row=1, col=2)
            fig.update_yaxes(title_text="Wind (m/s)", row=2, col=1)
            fig.update_yaxes(title_text="Irradiance (W/m²)", row=2, col=2)

            fig.update_layout(height=600, showlegend=True, template='plotly_white')

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Energy Yield Assessment")

        if 'weather_df' not in st.session_state:
            st.warning("⚠️ Please generate weather data first in the Location & Weather tab")
        else:
            col1, col2 = st.columns(2)

            with col1:
                system_capacity = st.number_input("System DC Capacity (kW):", min_value=1, max_value=10000, value=100, step=10)
                tilt = st.slider("Panel Tilt (degrees):", 0, 90, 35)
                azimuth = st.slider("Panel Azimuth (degrees):", 0, 360, 180)

            with col2:
                module_eff = st.slider("Module Efficiency (%):", 10, 25, 20) / 100
                inverter_eff = st.slider("Inverter Efficiency (%):", 90, 99, 98) / 100
                soiling = st.slider("Soiling Loss (%):", 0, 10, 2) / 100
                system_age = st.number_input("System Age (years):", 0, 30, 0)

            if st.button("⚡ Calculate Energy Yield", key="calc_yield"):
                with st.spinner("Calculating energy production..."):
                    # Add POA irradiance
                    weather_with_poa = analyzer.calculate_poa_irradiance_series(
                        st.session_state['weather_df'].copy(),
                        latitude, tilt, azimuth
                    )

                    # Calculate energy yield
                    results = analyzer.calculate_energy_yield(
                        weather_with_poa,
                        system_capacity,
                        module_eff,
                        inverter_eff,
                        soiling_loss=soiling,
                        system_age=system_age
                    )

                st.success("✅ Energy Yield Calculated")

                # Store results
                st.session_state['yield_results'] = results

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Annual Energy", f"{results['total_energy_kwh']:,.0f} kWh")

                with col2:
                    st.metric("Daily Average", f"{results['daily_average_kwh']:.1f} kWh")

                with col3:
                    st.metric("Capacity Factor", f"{results['capacity_factor'] * 100:.1f}%")

                with col4:
                    st.metric("Performance Ratio", f"{results['performance_ratio']:.2f}")

                # Monthly energy production
                st.subheader("Monthly Energy Production")

                months = list(results['monthly_energy_kwh'].keys())
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_values = [results['monthly_energy_kwh'][m] for m in months]

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=month_names[:len(months)],
                    y=monthly_values,
                    marker_color='#2ECC71',
                    text=[f"{v:,.0f}" for v in monthly_values],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Monthly Energy Production",
                    xaxis_title="Month",
                    yaxis_title="Energy (kWh)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Daily production profile
                st.subheader("Daily Production Profile")

                daily_prod = results['weather_df'].groupby(
                    results['weather_df']['timestamp'].dt.date
                )['ac_energy'].sum()

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=daily_prod.index,
                    y=daily_prod.values,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#3498DB', width=1),
                    name='Daily Energy'
                ))

                fig.update_layout(
                    title="Daily Energy Production Over Time",
                    xaxis_title="Date",
                    yaxis_title="Daily Energy (kWh)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("P50/P90 Energy Yield Analysis")
        st.markdown("Probabilistic energy yield assessment for bankability and financing.")

        if 'yield_results' not in st.session_state:
            st.warning("⚠️ Please calculate energy yield first")
        else:
            base_yield = st.session_state['yield_results']['total_energy_kwh']

            st.write("### Uncertainty Factors")

            col1, col2 = st.columns(2)

            with col1:
                irr_unc = st.slider("Irradiance Uncertainty (%):", 0, 10, 5) / 100
                temp_unc = st.slider("Temperature Uncertainty (%):", 0, 5, 2) / 100
                soil_unc = st.slider("Soiling Uncertainty (%):", 0, 10, 3) / 100

            with col2:
                deg_unc = st.slider("Degradation Uncertainty (%):", 0, 5, 2) / 100
                equip_unc = st.slider("Equipment Uncertainty (%):", 0, 5, 3) / 100
                avail_unc = st.slider("Availability Uncertainty (%):", 0, 5, 2) / 100

            if st.button("📊 Calculate P50/P90", key="calc_p50"):
                uncertainty_factors = {
                    'irradiance': irr_unc,
                    'temperature': temp_unc,
                    'soiling': soil_unc,
                    'degradation': deg_unc,
                    'equipment': equip_unc,
                    'availability': avail_unc
                }

                p_values = analyzer.calculate_p50_p90(base_yield, uncertainty_factors)

                st.success("✅ P-value Analysis Complete")

                # Display P-values
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("P99 (Conservative)", f"{p_values['P99']:,.0f} kWh")

                with col2:
                    st.metric("P90 (Likely)", f"{p_values['P90']:,.0f} kWh")

                with col3:
                    st.metric("P50 (Expected)", f"{p_values['P50']:,.0f} kWh")

                with col4:
                    st.metric("P10 (Optimistic)", f"{p_values['P10']:,.0f} kWh")

                # Visualization
                fig = go.Figure()

                p_levels = ['P99', 'P90', 'P50', 'P10']
                p_energies = [p_values[p] for p in p_levels]
                colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

                fig.add_trace(go.Bar(
                    x=p_levels,
                    y=p_energies,
                    marker_color=colors,
                    text=[f"{e:,.0f} kWh" for e in p_energies],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Energy Yield Probability Distribution",
                    xaxis_title="Exceedance Probability",
                    yaxis_title="Annual Energy (kWh)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Uncertainty breakdown
                st.subheader("Uncertainty Factor Breakdown")

                factor_names = list(uncertainty_factors.keys())
                factor_values = [v * 100 for v in uncertainty_factors.values()]

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=factor_names,
                    y=factor_values,
                    marker_color='#9B59B6'
                ))

                fig.update_layout(
                    title="Individual Uncertainty Contributions",
                    xaxis_title="Factor",
                    yaxis_title="Uncertainty (%)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(f"📈 Total Combined Uncertainty: {p_values['uncertainty'] * 100:.2f}%")

    with tab4:
        st.subheader("Long-Term Degradation Modeling")

        col1, col2 = st.columns(2)

        with col1:
            initial_power = st.number_input("Initial System Power (kW):", min_value=1, max_value=10000, value=100, step=10, key="deg_power")
            degradation_rate = st.slider("Annual Degradation Rate (%):", 0.0, 2.0, 0.5, 0.1) / 100

        with col2:
            lifetime_years = st.slider("Analysis Period (years):", 10, 40, 25, 5)

        if st.button("📉 Model Degradation", key="model_deg"):
            results = analyzer.model_degradation(initial_power, degradation_rate, lifetime_years)

            # Power degradation chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results['years'],
                y=results['linear_power'],
                name='Linear Degradation',
                line=dict(color='#3498DB', width=2, dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=results['years'],
                y=results['power_with_lid'],
                name='With LID',
                line=dict(color='#E74C3C', width=3)
            ))

            # Add warranty line (typically 80% at 25 years)
            warranty_power = initial_power * 0.80
            fig.add_hline(
                y=warranty_power,
                line_dash="dot",
                line_color="green",
                annotation_text="Warranty (80%)"
            )

            fig.update_layout(
                title="System Power Degradation Over Time",
                xaxis_title="Year",
                yaxis_title="System Power (kW)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cumulative energy
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results['years'],
                y=results['cumulative_energy'],
                fill='tozeroy',
                line=dict(color='#2ECC71', width=3),
                name='Cumulative Energy'
            ))

            fig.update_layout(
                title="Cumulative Energy Production Over Lifetime",
                xaxis_title="Year",
                yaxis_title="Cumulative Energy (kWh)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Lifetime Energy", f"{results['cumulative_energy'][-1]:,.0f} kWh")

            with col2:
                final_power = results['power_with_lid'][-1]
                power_retained = (final_power / initial_power) * 100
                st.metric(f"Power After {lifetime_years}Y", f"{power_retained:.1f}%",
                         delta=f"-{100 - power_retained:.1f}%")

            with col3:
                avg_annual = results['cumulative_energy'][-1] / lifetime_years
                st.metric("Avg Annual Energy", f"{avg_annual:,.0f} kWh")

    with tab5:
        st.subheader("Energy Production Forecasting")

        if 'yield_results' not in st.session_state:
            st.warning("⚠️ Please calculate energy yield first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                forecast_days = st.slider("Forecast Horizon (days):", 7, 90, 30)

            with col2:
                forecast_method = st.selectbox("Method:", ["statistical", "ml_ensemble", "prophet"])

            if st.button("🔮 Generate Forecast", key="forecast"):
                with st.spinner("Generating energy production forecast..."):
                    forecast = analyzer.forecast_energy_production(
                        st.session_state['yield_results']['weather_df'],
                        forecast_days,
                        forecast_method
                    )

                st.success(f"✅ {forecast_days}-day forecast generated")

                # Plot forecast
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=forecast['forecast_dates'],
                    y=forecast['predicted_energy'],
                    name='Forecast',
                    line=dict(color='#2ECC71', width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=forecast['forecast_dates'],
                    y=forecast['upper_bound'],
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=forecast['forecast_dates'],
                    y=forecast['lower_bound'],
                    name='95% Confidence',
                    fill='tonexty',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    line=dict(width=0)
                ))

                fig.update_layout(
                    title=f"{forecast_days}-Day Energy Production Forecast ({forecast_method})",
                    xaxis_title="Date",
                    yaxis_title="Daily Energy (kWh)",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Forecast summary
                total_forecast = sum(forecast['predicted_energy'])
                avg_daily = total_forecast / forecast_days

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Forecast", f"{total_forecast:,.0f} kWh")

                with col2:
                    st.metric("Daily Average", f"{avg_daily:.1f} kWh")

                with col3:
                    st.metric("Confidence Level", f"{forecast['confidence_level'] * 100:.0f}%")

    st.divider()
    st.info("💡 **Weather Data & Energy Yield Assessment** - Branch B06 | Complete Energy Analysis Suite")
