"""
Weather Data UI Component for Streamlit.

Interactive component for TMY data visualization, location selection,
historical trends, and data quality indicators.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pv_simulator.api.nsrdb_client import NSRDBClient
from pv_simulator.api.pvgis_client import PVGISClient
from pv_simulator.config.settings import settings
from pv_simulator.models.weather import GlobalLocation, TMYData
from pv_simulator.services.global_coverage import GlobalWeatherCoverage
from pv_simulator.services.historical_analyzer import HistoricalWeatherAnalyzer
from pv_simulator.services.tmy_generator import TMYGenerator
from pv_simulator.services.tmy_manager import TMYDataManager
from pv_simulator.services.weather_database import WeatherDatabaseBuilder

logger = logging.getLogger(__name__)


class WeatherDataUI:
    """
    Streamlit UI component for weather data visualization and management.

    Features:
    - Interactive location selector with map
    - TMY data visualization (irradiance, temperature, wind)
    - Historical trends charts
    - Data quality indicators
    - Download TMY files in multiple formats
    """

    def __init__(self) -> None:
        """Initialize Weather Data UI component."""
        # Initialize services
        self.tmy_manager = TMYDataManager()
        self.weather_db = WeatherDatabaseBuilder()
        self.coverage = GlobalWeatherCoverage()
        self.analyzer = HistoricalWeatherAnalyzer()
        self.generator = TMYGenerator()

        # Initialize session state
        if "selected_location" not in st.session_state:
            st.session_state.selected_location = None
        if "tmy_data" not in st.session_state:
            st.session_state.tmy_data = None

    def render(self) -> None:
        """Render the complete weather data UI."""
        st.title("🌤️ TMY Weather Database & Analysis")

        st.markdown(
            """
            Comprehensive weather data management system with TMY (Typical Meteorological Year)
            data, global coverage, and advanced analytics.
            """
        )

        # Sidebar for navigation
        page = st.sidebar.radio(
            "Navigation",
            [
                "Location Selection",
                "TMY Data Viewer",
                "Historical Analysis",
                "Generate TMY",
                "Data Upload",
            ],
        )

        # Render selected page
        if page == "Location Selection":
            self.render_location_selection()
        elif page == "TMY Data Viewer":
            self.render_tmy_viewer()
        elif page == "Historical Analysis":
            self.render_historical_analysis()
        elif page == "Generate TMY":
            self.render_tmy_generator()
        elif page == "Data Upload":
            self.render_data_upload()

    def render_location_selection(self) -> None:
        """Render interactive location selector."""
        st.header("📍 Location Selection")

        # Location input methods
        input_method = st.radio(
            "Select location method:",
            ["Coordinates", "Search by Name", "Select from Map"],
        )

        location = None

        if input_method == "Coordinates":
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=39.7392,
                    step=0.1,
                )
            with col2:
                longitude = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=-104.9903,
                    step=0.1,
                )

            # Find nearest location
            nearest = self.coverage.nearest_station_finder(latitude, longitude, num_stations=1)

            if nearest:
                location = nearest[0][0]
                distance = nearest[0][1]
                st.info(f"📍 Nearest station: {location.name} ({distance:.1f} km away)")
            else:
                # Create custom location
                location = GlobalLocation(
                    name=f"Custom_{latitude:.2f}_{longitude:.2f}",
                    country="Unknown",
                    latitude=latitude,
                    longitude=longitude,
                    elevation=0.0,
                    timezone="UTC",
                )

        elif input_method == "Search by Name":
            search_term = st.text_input("Enter location name:", "Denver")
            country_filter = st.text_input("Country (optional):", "")

            if st.button("Search"):
                results = self.coverage.search_by_name(
                    search_term, country_filter if country_filter else None
                )

                if results:
                    st.success(f"Found {len(results)} locations")

                    # Display results
                    for loc in results[:5]:  # Show top 5
                        if st.button(
                            f"{loc.name}, {loc.country} ({loc.latitude:.2f}, {loc.longitude:.2f})"
                        ):
                            location = loc
                            st.session_state.selected_location = location
                else:
                    st.warning("No locations found")

        elif input_method == "Select from Map":
            st.info("Map selection - using default location for now")
            # In a full implementation, would use st.map() with clickable markers
            locations = self.coverage.worldwide_location_database()

            if locations:
                location_names = [f"{loc.name}, {loc.country}" for loc in locations]
                selected_name = st.selectbox("Select location:", location_names)

                selected_idx = location_names.index(selected_name)
                location = locations[selected_idx]

        # Save selected location
        if location:
            st.session_state.selected_location = location

            # Display location details
            st.success(f"✅ Selected: {location.name}, {location.country}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latitude", f"{location.latitude:.4f}°")
            with col2:
                st.metric("Longitude", f"{location.longitude:.4f}°")
            with col3:
                st.metric("Elevation", f"{location.elevation:.0f} m")

            # Data source selection
            st.subheader("Select Data Source")

            data_source = st.selectbox(
                "Choose weather data source:",
                ["NREL NSRDB (USA)", "PVGIS (Europe/Africa/Asia)", "Upload File"],
            )

            if st.button("Fetch TMY Data"):
                with st.spinner("Fetching TMY data..."):
                    try:
                        if data_source == "NREL NSRDB (USA)":
                            tmy_data = self.weather_db.nrel_nsrdb_integration(
                                latitude=location.latitude,
                                longitude=location.longitude,
                            )
                        elif data_source == "PVGIS (Europe/Africa/Asia)":
                            tmy_data = self.weather_db.pvgis_data_fetcher(
                                latitude=location.latitude,
                                longitude=location.longitude,
                            )
                        else:
                            st.error("Please use Data Upload page for file upload")
                            return

                        st.session_state.tmy_data = tmy_data
                        st.success(
                            f"✅ Successfully fetched {len(tmy_data.hourly_data)} "
                            f"data points!"
                        )

                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")
                        logger.error(f"Error fetching TMY data: {e}")

    def render_tmy_viewer(self) -> None:
        """Render TMY data visualization."""
        st.header("📊 TMY Data Viewer")

        if st.session_state.tmy_data is None:
            st.warning("⚠️ No TMY data loaded. Please select a location and fetch data first.")
            return

        tmy_data: TMYData = st.session_state.tmy_data

        # Display metadata
        st.subheader("Dataset Information")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Location", tmy_data.location.name)
        with col2:
            st.metric("Data Source", tmy_data.data_source.value)
        with col3:
            st.metric("Data Points", len(tmy_data.hourly_data))
        with col4:
            st.metric("Quality", tmy_data.data_quality.value.title())

        # Data quality indicator
        completeness = tmy_data.completeness_percentage
        st.progress(completeness / 100.0)
        st.caption(f"Data Completeness: {completeness:.1f}%")

        # Convert to DataFrame for visualization
        df = pd.DataFrame([point.model_dump() for point in tmy_data.hourly_data])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Time range selector
        st.subheader("Time Range Selection")
        time_range = st.selectbox(
            "Select time range:",
            ["Full Year", "January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
        )

        if time_range != "Full Year":
            month_num = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ].index(time_range) + 1
            df = df[df["timestamp"].dt.month == month_num]

        # Irradiance visualization
        st.subheader("☀️ Solar Irradiance")

        fig_irradiance = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Solar Irradiance Components"],
        )

        fig_irradiance.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["irradiance_ghi"],
                name="GHI",
                line=dict(color="orange"),
            )
        )

        fig_irradiance.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["irradiance_dni"],
                name="DNI",
                line=dict(color="red"),
            )
        )

        fig_irradiance.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["irradiance_dhi"],
                name="DHI",
                line=dict(color="blue"),
            )
        )

        fig_irradiance.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Irradiance (W/m²)",
            hovermode="x unified",
        )

        st.plotly_chart(fig_irradiance, use_container_width=True)

        # Temperature and wind visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌡️ Temperature")
            fig_temp = px.line(
                df,
                x="timestamp",
                y="temperature",
                title="Ambient Temperature",
            )
            fig_temp.update_layout(
                yaxis_title="Temperature (°C)",
                xaxis_title="Time",
                height=300,
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            st.subheader("💨 Wind Speed")
            fig_wind = px.line(
                df,
                x="timestamp",
                y="wind_speed",
                title="Wind Speed",
            )
            fig_wind.update_layout(
                yaxis_title="Wind Speed (m/s)",
                xaxis_title="Time",
                height=300,
            )
            st.plotly_chart(fig_wind, use_container_width=True)

        # Statistics summary
        st.subheader("📈 Statistics Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Annual GHI",
                f"{tmy_data.get_annual_irradiation():.0f} kWh/m²",
            )
            st.metric(
                "Max GHI",
                f"{df['irradiance_ghi'].max():.0f} W/m²",
            )

        with col2:
            st.metric(
                "Average Temperature",
                f"{tmy_data.get_average_temperature():.1f} °C",
            )
            st.metric(
                "Temperature Range",
                f"{df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C",
            )

        with col3:
            st.metric(
                "Average Wind Speed",
                f"{df['wind_speed'].mean():.1f} m/s",
            )
            st.metric(
                "Max Wind Speed",
                f"{df['wind_speed'].max():.1f} m/s",
            )

        # Download section
        st.subheader("💾 Download TMY Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{tmy_data.location.name}_TMY.csv",
                mime="text/csv",
            )

        with col2:
            # JSON download
            json_data = tmy_data.model_dump_json(indent=2)

            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"{tmy_data.location.name}_TMY.json",
                mime="application/json",
            )

        with col3:
            st.info("TMY3 and EPW formats coming soon")

    def render_historical_analysis(self) -> None:
        """Render historical weather analysis."""
        st.header("📈 Historical Weather Analysis")

        st.info("Historical analysis requires multi-year data. Upload multiple years of data to enable this feature.")

        # Placeholder for historical analysis
        st.markdown("""
        ### Available Analysis:
        - **Multi-year Statistics**: Mean, P90, P50, P10 values
        - **Extreme Weather Events**: Heatwaves, high winds, low irradiance
        - **Climate Change Trends**: Temperature and irradiance trends
        - **Seasonal Variability**: Season-by-season comparison
        - **Inter-annual Variability**: Year-to-year variation
        """)

    def render_tmy_generator(self) -> None:
        """Render TMY generation interface."""
        st.header("⚙️ Generate Synthetic TMY")

        st.markdown("""
        Generate a synthetic Typical Meteorological Year (TMY) from multi-year historical data
        using the Sandia method or custom algorithms.
        """)

        st.info("TMY generation requires at least 3 years of historical data.")

        # Method selection
        method = st.selectbox(
            "Generation Method:",
            ["Sandia TMY Method", "Median Method", "Average Method"],
        )

        st.markdown(f"""
        **{method}**:
        - Sandia: Selects most typical months using weighted FS statistics
        - Median: Selects months closest to median values
        - Average: Simple averaging of all years
        """)

    def render_data_upload(self) -> None:
        """Render data upload interface."""
        st.header("📤 Upload Weather Data")

        st.markdown("""
        Upload your own weather data files in supported formats:
        - TMY2, TMY3 (NREL format)
        - EPW (EnergyPlus Weather)
        - CSV (generic format)
        - Meteonorm files
        - Local weather station data
        """)

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "tm2", "tm3", "epw", "json"],
        )

        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")

            # Detect format
            file_ext = Path(uploaded_file.name).suffix.lower()

            format_map = {
                ".csv": "CSV",
                ".tm2": "TMY2",
                ".tm3": "TMY3",
                ".epw": "EPW",
                ".json": "JSON",
            }

            detected_format = format_map.get(file_ext, "Unknown")
            st.info(f"Detected format: {detected_format}")

            if st.button("Parse and Load"):
                with st.spinner("Parsing file..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = Path(f"/tmp/{uploaded_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Parse file
                        tmy_data = self.tmy_manager.load_tmy_data(temp_path)

                        st.session_state.tmy_data = tmy_data
                        st.success(
                            f"✅ Successfully loaded {len(tmy_data.hourly_data)} data points!"
                        )

                        # Clean up
                        temp_path.unlink()

                    except Exception as e:
                        st.error(f"Failed to parse file: {e}")
                        logger.error(f"Error parsing uploaded file: {e}")


def main() -> None:
    """Main function to run the Weather Data UI."""
    st.set_page_config(
        page_title="TMY Weather Database",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize and render UI
    ui = WeatherDataUI()
    ui.render()


if __name__ == "__main__":
    main()
