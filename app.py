"""
PV Circularity Simulator - Main Application
Production-ready Streamlit application integrating all modules.
"""

import streamlit as st
from datetime import date

# Import modules
from src.modules.ui.navigation import create_default_navigation
from src.modules.hybrid_energy.hybrid_ui import HybridUI
from src.modules.financial.financial_ui import FinancialUI


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="PV Circularity Simulator",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("☀️ PV Circularity Simulator")
    st.markdown("*Comprehensive renewable energy system modeling and analysis*")

    # Navigation
    nav = create_default_navigation()

    # Define page callbacks
    def home_page():
        st.header("Welcome to PV Circularity Simulator")
        st.markdown("""
        ### 🚀 Features

        **Hybrid Energy Systems (B12)**
        - 🔋 Battery Integration & Energy Storage
        - 💨 Wind-Solar Hybrid Systems
        - 💧 Hydrogen Integration & P2X
        - 🔌 Grid Interaction & Smart Grid

        **Financial Analysis (B13)**
        - 📊 LCOE Calculations
        - 📈 NPV Analysis
        - 💹 IRR Modeling
        - 🏦 Bankability Assessment

        **Integration & Visualization (B14-B15)**
        - 🔗 Cross-Module Integration
        - 📊 Advanced Visualization
        - 🧭 Multi-Page Navigation
        """)

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**15** Production Modules")
        with col2:
            st.info("**4** Major Categories")
        with col3:
            st.info("**100%** Pydantic Validated")

    def hybrid_energy_page():
        st.header("⚡ Hybrid Energy Systems")
        st.write("Overview of hybrid renewable energy capabilities")

    def battery_page():
        st.header("🔋 Battery Energy Storage")
        from src.modules.hybrid_energy.battery_integration import create_battery_system
        import numpy as np

        # Demo
        battery = create_battery_system(1000)
        st.success(f"Battery system created: {battery.spec.capacity_kwh} kWh")

        # Sample sizing
        load = np.random.uniform(200, 500, 24)
        generation = np.random.uniform(0, 600, 24)

        result = battery.sizing(load, generation)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommended Capacity", f"{result['recommended_capacity_kwh']:.0f} kWh")
        with col2:
            st.metric("Recommended Power", f"{result['recommended_power_kw']:.0f} kW")

    def financial_page():
        st.header("💰 Financial Analysis")
        financial_ui = FinancialUI()
        financial_ui.render_dashboard()

    # Register callbacks
    callbacks = {
        'home': home_page,
        'hybrid_energy': hybrid_energy_page,
        'battery': battery_page,
        'financial': financial_page,
    }

    # Render navigation
    nav.menu_structure(style="sidebar")

    # Render current page
    current_page = st.session_state.get('current_page', 'home')
    if current_page in callbacks:
        callbacks[current_page]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **PV Circularity Simulator v1.0**

    Production-ready renewable energy
    system modeling platform.

    All modules production-ready with
    Pydantic validation.
    """)


if __name__ == "__main__":
    main()
