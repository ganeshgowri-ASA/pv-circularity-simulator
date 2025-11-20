"""
Main Streamlit application for PV Circularity Simulator.

This is the entry point for the web-based user interface.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st

from ui.components.weather_component import WeatherDataUI


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="PV Circularity Simulator",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("PV Circularity Simulator")
    st.sidebar.markdown("---")

    app_mode = st.sidebar.selectbox(
        "Choose Module",
        [
            "TMY Weather Database",
            "Cell Design",
            "Module Engineering",
            "System Planning",
            "Performance Monitoring",
            "Circular Economy",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Version**: 0.1.0

        **About**: End-to-end PV lifecycle simulation platform
        with comprehensive weather data management.
        """
    )

    # Render selected module
    if app_mode == "TMY Weather Database":
        ui = WeatherDataUI()
        ui.render()
    elif app_mode == "Cell Design":
        st.title("🔬 Cell Design")
        st.info("Cell design module coming soon...")
    elif app_mode == "Module Engineering":
        st.title("⚡ Module Engineering")
        st.info("Module engineering module coming soon...")
    elif app_mode == "System Planning":
        st.title("🏗️ System Planning")
        st.info("System planning module coming soon...")
    elif app_mode == "Performance Monitoring":
        st.title("📊 Performance Monitoring")
        st.info("Performance monitoring module coming soon...")
    elif app_mode == "Circular Economy":
        st.title("♻️ Circular Economy")
        st.info("Circular economy module coming soon...")


if __name__ == "__main__":
    main()
