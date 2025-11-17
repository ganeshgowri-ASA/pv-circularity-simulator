"""
Streamlit Dashboard for Weather API Integration.

This module provides an interactive dashboard for viewing live weather data,
configuring API providers, and monitoring data quality metrics.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pv_simulator.config import get_settings
from pv_simulator.models.weather import GeoLocation, WeatherProvider
from pv_simulator.weather.fetcher import RealTimeWeatherFetcher
from pv_simulator.weather.integrator import WeatherAPIIntegrator
from pv_simulator.weather.validator import WeatherDataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PV Circularity Simulator - Weather Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "fetcher" not in st.session_state:
        st.session_state.fetcher = RealTimeWeatherFetcher()

    if "integrator" not in st.session_state:
        st.session_state.integrator = WeatherAPIIntegrator()

    if "validator" not in st.session_state:
        st.session_state.validator = WeatherDataValidator()

    if "location" not in st.session_state:
        st.session_state.location = {"latitude": 40.7128, "longitude": -74.0060}

    if "current_data" not in st.session_state:
        st.session_state.current_data = None


def sidebar_config() -> dict:
    """
    Render sidebar configuration.

    Returns:
        Dictionary with user selections
    """
    st.sidebar.title("⚙️ Configuration")

    # Location input
    st.sidebar.subheader("Location")
    latitude = st.sidebar.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=st.session_state.location["latitude"],
        step=0.1,
        format="%.4f",
    )
    longitude = st.sidebar.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=st.session_state.location["longitude"],
        step=0.1,
        format="%.4f",
    )

    # Provider selection
    st.sidebar.subheader("Weather Provider")
    available_providers = st.session_state.integrator.get_available_providers()
    provider_options = ["Auto-select"] + [p.value for p in available_providers]

    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        options=provider_options,
        index=0,
    )

    # Forecast settings
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
    )

    # Historical settings
    st.sidebar.subheader("Historical Data")
    historical_days = st.sidebar.slider(
        "Historical Days",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "provider": selected_provider if selected_provider != "Auto-select" else None,
        "forecast_days": forecast_days,
        "historical_days": historical_days,
    }


def display_current_weather(config: dict) -> None:
    """
    Display current weather conditions.

    Args:
        config: Configuration dictionary
    """
    st.header("🌤️ Current Weather Conditions")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Fetch Current Weather", type="primary"):
            try:
                with st.spinner("Fetching current weather..."):
                    provider = (
                        WeatherProvider(config["provider"])
                        if config["provider"]
                        else None
                    )

                    current = st.session_state.fetcher.current_conditions(
                        latitude=config["latitude"],
                        longitude=config["longitude"],
                        provider=provider,
                    )

                    st.session_state.current_data = current
                    st.session_state.location = {
                        "latitude": config["latitude"],
                        "longitude": config["longitude"],
                    }
                    st.success("Weather data fetched successfully!")

            except Exception as e:
                st.error(f"Failed to fetch weather data: {e}")
                logger.error(f"Error fetching weather: {e}")

    with col2:
        if st.session_state.current_data:
            st.metric(
                "Data Source",
                st.session_state.current_data.data.provider.value,
            )

    # Display current conditions
    if st.session_state.current_data:
        data = st.session_state.current_data.data

        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if data.temperature is not None:
                st.metric("Temperature", f"{data.temperature:.1f}°C")
            if data.feels_like is not None:
                st.metric("Feels Like", f"{data.feels_like:.1f}°C")

        with col2:
            if data.humidity is not None:
                st.metric("Humidity", f"{data.humidity:.0f}%")
            if data.pressure is not None:
                st.metric("Pressure", f"{data.pressure:.0f} hPa")

        with col3:
            if data.wind_speed is not None:
                st.metric("Wind Speed", f"{data.wind_speed:.1f} m/s")
            if data.wind_direction is not None:
                st.metric("Wind Direction", f"{data.wind_direction:.0f}°")

        with col4:
            if data.ghi is not None:
                st.metric("Solar GHI", f"{data.ghi:.0f} W/m²")
            if data.cloud_cover is not None:
                st.metric("Cloud Cover", f"{data.cloud_cover:.0f}%")

        # Additional details
        with st.expander("📊 Detailed Information"):
            details_col1, details_col2 = st.columns(2)

            with details_col1:
                st.write("**Environmental Data:**")
                if data.visibility is not None:
                    st.write(f"- Visibility: {data.visibility:.0f} m")
                if data.precipitation is not None:
                    st.write(f"- Precipitation: {data.precipitation:.1f} mm")
                if data.dew_point is not None:
                    st.write(f"- Dew Point: {data.dew_point:.1f}°C")
                if data.uv_index is not None:
                    st.write(f"- UV Index: {data.uv_index:.1f}")

            with details_col2:
                st.write("**Solar Data:**")
                if data.ghi is not None:
                    st.write(f"- GHI: {data.ghi:.0f} W/m²")
                if data.dni is not None:
                    st.write(f"- DNI: {data.dni:.0f} W/m²")
                if data.dhi is not None:
                    st.write(f"- DHI: {data.dhi:.0f} W/m²")
                if data.solar_elevation is not None:
                    st.write(f"- Solar Elevation: {data.solar_elevation:.1f}°")

            st.write(f"**Timestamp:** {data.timestamp}")
            st.write(f"**Location:** {data.location}")


def display_forecast(config: dict) -> None:
    """
    Display weather forecast.

    Args:
        config: Configuration dictionary
    """
    st.header("📅 Weather Forecast")

    if st.button("Fetch Forecast", type="primary"):
        try:
            with st.spinner(f"Fetching {config['forecast_days']}-day forecast..."):
                provider = (
                    WeatherProvider(config["provider"]) if config["provider"] else None
                )

                forecast = st.session_state.fetcher.forecast_data(
                    latitude=config["latitude"],
                    longitude=config["longitude"],
                    days=config["forecast_days"],
                    provider=provider,
                )

                # Convert forecast to DataFrame for plotting
                df = pd.DataFrame([
                    {
                        "timestamp": point.timestamp,
                        "temperature": point.temperature,
                        "humidity": point.humidity,
                        "wind_speed": point.wind_speed,
                        "ghi": point.ghi,
                        "precipitation": point.precipitation,
                    }
                    for point in forecast.forecast_data
                ])

                # Temperature forecast plot
                fig_temp = px.line(
                    df,
                    x="timestamp",
                    y="temperature",
                    title="Temperature Forecast",
                    labels={"temperature": "Temperature (°C)", "timestamp": "Time"},
                )
                st.plotly_chart(fig_temp, use_container_width=True)

                # Solar irradiance plot
                if df["ghi"].notna().any():
                    fig_solar = px.line(
                        df,
                        x="timestamp",
                        y="ghi",
                        title="Solar Irradiance Forecast (GHI)",
                        labels={"ghi": "GHI (W/m²)", "timestamp": "Time"},
                    )
                    st.plotly_chart(fig_solar, use_container_width=True)

                # Multi-variable plot
                col1, col2 = st.columns(2)

                with col1:
                    fig_humidity = px.line(
                        df,
                        x="timestamp",
                        y="humidity",
                        title="Humidity Forecast",
                        labels={"humidity": "Humidity (%)", "timestamp": "Time"},
                    )
                    st.plotly_chart(fig_humidity, use_container_width=True)

                with col2:
                    fig_wind = px.line(
                        df,
                        x="timestamp",
                        y="wind_speed",
                        title="Wind Speed Forecast",
                        labels={"wind_speed": "Wind Speed (m/s)", "timestamp": "Time"},
                    )
                    st.plotly_chart(fig_wind, use_container_width=True)

                st.success(f"Fetched {len(forecast.forecast_data)} forecast data points")

        except Exception as e:
            st.error(f"Failed to fetch forecast: {e}")
            logger.error(f"Error fetching forecast: {e}")


def display_providers_status() -> None:
    """Display status of all weather providers."""
    st.header("🔌 API Providers Status")

    status = st.session_state.fetcher.get_providers_status()

    cols = st.columns(3)

    for idx, (provider_name, info) in enumerate(status.items()):
        with cols[idx % 3]:
            is_available = info.get("available", False)
            status_emoji = "✅" if is_available else "❌"

            st.subheader(f"{status_emoji} {provider_name}")

            if is_available:
                st.success("Available")
            else:
                st.error("Unavailable")

            # Rate limit info
            rate_limit = info.get("rate_limit", {})
            if "requests_per_minute" in rate_limit:
                st.write(
                    f"**Rate Limit:** {rate_limit['requests_per_minute']} req/min"
                )

            if "current_tokens" in rate_limit:
                st.write(
                    f"**Available Tokens:** {rate_limit['current_tokens']:.1f} / "
                    f"{rate_limit['max_tokens']:.0f}"
                )


def display_data_quality() -> None:
    """Display data quality metrics."""
    st.header("📈 Data Quality Metrics")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Cache Status")
        cache_info = st.session_state.fetcher.cache_manager()

        st.write(f"**Cache Type:** {cache_info.get('cache_type', 'N/A')}")
        st.write(f"**Default TTL:** {cache_info.get('default_ttl', 0)} seconds")
        st.write(f"**Backend:** {cache_info.get('backend', 'N/A')}")

        if st.button("Clear Cache"):
            st.session_state.fetcher.clear_cache()
            st.success("Cache cleared successfully!")

    with col2:
        st.subheader("System Settings")
        settings = get_settings()

        st.write(f"**Temperature Unit:** {settings.temperature_unit}")
        st.write(f"**Wind Speed Unit:** {settings.wind_speed_unit}")
        st.write(f"**Irradiance Unit:** {settings.irradiance_unit}")
        st.write(f"**Outlier Detection:** {'Enabled' if settings.outlier_detection_enabled else 'Disabled'}")
        st.write(f"**Gap Filling:** {'Enabled' if settings.gap_filling_enabled else 'Disabled'}")


def main() -> None:
    """Main dashboard application."""
    # Initialize session state
    initialize_session_state()

    # Header
    st.title("☀️ PV Circularity Simulator - Weather Dashboard")
    st.markdown(
        "Real-time weather data integration for photovoltaic system simulation"
    )

    # Sidebar configuration
    config = sidebar_config()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Current Weather",
        "Forecast",
        "Provider Status",
        "Data Quality",
    ])

    with tab1:
        display_current_weather(config)

    with tab2:
        display_forecast(config)

    with tab3:
        display_providers_status()

    with tab4:
        display_data_quality()

    # Footer
    st.markdown("---")
    st.markdown(
        "PV Circularity Simulator v0.1.0 | "
        "Weather API Integration | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
