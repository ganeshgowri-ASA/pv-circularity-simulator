"""
Energy Forecasting Module
=========================

Short-term and long-term energy production forecasting.
Uses weather forecasts and machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the energy forecasting module.

    Args:
        session: Session manager instance

    Features:
        - Intraday forecasting (15-min to hourly)
        - Day-ahead forecasting
        - Week-ahead forecasting
        - Seasonal forecasting
        - Weather forecast integration
        - Machine learning models
        - Forecast accuracy tracking
        - Confidence intervals
    """
    st.header("🔮 Energy Forecasting")

    st.info("""
    Generate short-term and long-term energy production forecasts using
    weather data and advanced forecasting models.
    """)

    # Forecast type selection
    st.subheader("⚙️ Forecast Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["Intraday (Next 24h)", "Day-ahead", "Week-ahead", "Month-ahead", "Seasonal"]
        )

    with col2:
        forecast_resolution = st.selectbox(
            "Time Resolution",
            ["15-minute", "Hourly", "Daily"]
        )

    with col3:
        forecast_model = st.selectbox(
            "Forecasting Model",
            ["Persistence", "Statistical (ARIMA)", "Machine Learning", "Ensemble"]
        )

    # Weather data source
    st.markdown("---")
    st.subheader("🌤️ Weather Data Source")

    col1, col2 = st.columns(2)

    with col1:
        weather_source = st.selectbox(
            "Weather Forecast Provider",
            ["NOAA GFS", "ECMWF", "Weather API", "Custom Data"]
        )

        update_frequency = st.selectbox(
            "Update Frequency",
            ["Real-time", "Hourly", "Every 3 hours", "Daily"]
        )

    with col2:
        include_parameters = st.multiselect(
            "Forecast Parameters",
            ["GHI", "DNI", "DHI", "Temperature", "Wind Speed", "Cloud Cover", "Humidity"],
            default=["GHI", "Temperature", "Cloud Cover"]
        )

    # Model settings
    with st.expander("🔧 Advanced Model Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Data**")
            training_period = st.slider("Training Period (months)", 1, 36, 12)
            include_seasonal = st.checkbox("Include Seasonal Patterns", True)
            include_weather = st.checkbox("Include Weather Forecasts", True)

        with col2:
            st.markdown("**Optimization**")
            optimization_metric = st.selectbox("Optimize For", ["RMSE", "MAE", "MAPE"])
            confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)

    # Run forecast
    st.markdown("---")
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating energy forecast..."):
            import time
            time.sleep(2)

            # Generate dummy forecast data
            if "Intraday" in forecast_horizon:
                periods = 24
                freq = 'H'
                unit = 'kW'
                scale = 1
            elif "Day-ahead" in forecast_horizon:
                periods = 24
                freq = 'H'
                unit = 'kW'
                scale = 1
            elif "Week-ahead" in forecast_horizon:
                periods = 7
                freq = 'D'
                unit = 'MWh'
                scale = 15
            else:
                periods = 30
                freq = 'D'
                unit = 'MWh'
                scale = 15

            timestamps = pd.date_range(start=datetime.now(), periods=periods, freq=freq)

            # Generate forecast with confidence intervals
            if freq == 'H':
                # Hourly with solar curve
                base_curve = []
                for ts in timestamps:
                    hour = ts.hour
                    if hour < 6 or hour > 19:
                        base_curve.append(0)
                    else:
                        # Solar curve approximation
                        sun_factor = np.sin((hour - 6) * np.pi / 13)
                        base_curve.append(900 * sun_factor * scale)

                forecast_values = [v + np.random.uniform(-50, 50) if v > 0 else 0 for v in base_curve]
            else:
                # Daily with seasonal variation
                base_values = []
                for i in range(periods):
                    base = 15 * scale  # Base production
                    seasonal = 1 + 0.2 * np.sin(i * 2 * np.pi / 365)
                    base_values.append(base * seasonal)

                forecast_values = [v + np.random.uniform(-v*0.1, v*0.1) for v in base_values]

            # Confidence intervals
            lower_bound = [v * 0.85 for v in forecast_values]
            upper_bound = [v * 1.15 for v in forecast_values]

            forecast_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Forecast': forecast_values,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })

            st.success("Forecast generated successfully!")

            # Forecast summary
            st.markdown("---")
            st.subheader("📊 Forecast Summary")

            col1, col2, col3, col4 = st.columns(4)

            total_forecast = forecast_df['Forecast'].sum()
            max_power = forecast_df['Forecast'].max()
            min_power = forecast_df['Forecast'].min()
            avg_power = forecast_df['Forecast'].mean()

            with col1:
                st.metric(f"Total Energy", f"{total_forecast:.1f} {unit}h" if freq == 'H' else f"{total_forecast:.1f} {unit}")

            with col2:
                st.metric(f"Peak Power", f"{max_power:.1f} {unit}")

            with col3:
                st.metric(f"Average", f"{avg_power:.1f} {unit}")

            with col4:
                capacity_factor = (avg_power / 1000) * 100 if freq == 'H' else (avg_power / 15) * 100
                st.metric("Capacity Factor", f"{capacity_factor:.1f}%")

            # Forecast visualization
            st.markdown("---")
            st.subheader("📈 Forecast Visualization")

            # Create chart data
            chart_data = forecast_df.set_index('Timestamp')[['Forecast', 'Lower Bound', 'Upper Bound']]

            st.line_chart(chart_data)

            # Detailed forecast table
            st.markdown("---")
            st.subheader("📋 Detailed Forecast")

            display_df = forecast_df.copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M' if freq == 'H' else '%Y-%m-%d')
            display_df['Forecast'] = display_df['Forecast'].round(1)
            display_df['Lower Bound'] = display_df['Lower Bound'].round(1)
            display_df['Upper Bound'] = display_df['Upper Bound'].round(1)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Forecast accuracy (if historical data available)
            st.markdown("---")
            st.subheader("🎯 Forecast Accuracy Metrics")

            col1, col2, col3, col4 = st.columns(4)

            # Dummy accuracy metrics
            with col1:
                rmse = np.random.uniform(50, 100)
                st.metric("RMSE", f"{rmse:.1f} {unit}")

            with col2:
                mae = np.random.uniform(30, 80)
                st.metric("MAE", f"{mae:.1f} {unit}")

            with col3:
                mape = np.random.uniform(5, 15)
                st.metric("MAPE", f"{mape:.1f}%")

            with col4:
                r2 = np.random.uniform(0.85, 0.95)
                st.metric("R² Score", f"{r2:.3f}")

            # Historical accuracy
            with st.expander("📊 Historical Forecast Accuracy"):
                st.info("Forecast accuracy trends over the past 30 days")

                # Generate dummy accuracy trend
                days = 30
                dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
                mape_trend = [np.random.uniform(8, 12) for _ in range(days)]

                accuracy_df = pd.DataFrame({
                    'Date': dates,
                    'MAPE (%)': mape_trend
                })

                st.line_chart(accuracy_df.set_index('Date'))

            # Weather forecast comparison
            st.markdown("---")
            st.subheader("🌤️ Weather Forecast")

            weather_df = pd.DataFrame({
                'Time': display_df['Timestamp'][:12],  # First 12 periods
                'GHI (W/m²)': [np.random.uniform(200, 1000) if i >= 6 and i <= 18 else 0
                              for i in range(12)],
                'Temp (°C)': [np.random.uniform(20, 35) for _ in range(12)],
                'Cloud (%)': [np.random.uniform(0, 40) for _ in range(12)]
            })

            st.dataframe(weather_df, use_container_width=True, hide_index=True)

            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Export Forecast"):
                    st.info("Export functionality coming soon")

            with col2:
                if st.button("📧 Schedule Email Report"):
                    st.info("Email scheduling coming soon")

            with col3:
                if st.button("🔄 Update Forecast"):
                    st.rerun()

            # Save forecast
            session.set('forecast_data', {
                'horizon': forecast_horizon,
                'model': forecast_model,
                'total_forecast': total_forecast,
                'forecast_df': forecast_df.to_dict()
            })
