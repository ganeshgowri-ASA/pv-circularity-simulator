"""Main entry point for PV Circularity Simulator Streamlit application.

This module sets up the navigation system and registers all application pages.
Run with: streamlit run src/main.py
"""

import streamlit as st

# Import navigation system
from navigation import NavigationManager, AccessLevel

# Import page modules
from pages import (
    home,
    cell_design,
    module_engineering,
    system_planning,
    performance_monitoring,
    circularity,
)


# Configure Streamlit page
st.set_page_config(
    page_title="PV Circularity Simulator",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_navigation() -> NavigationManager:
    """Initialize and configure the navigation manager.

    Returns:
        Configured NavigationManager instance.
    """
    # Create navigation manager
    nav = NavigationManager(default_page="home")

    # Register pages using the decorator pattern
    nav.register_page(
        nav.page_registry(
            name="home",
            title="Home",
            icon="🏠",
            description="Main dashboard and overview",
            access_level=AccessLevel.PUBLIC,
            show_in_sidebar=True,
            order=0,
        )(home.render)
    )

    # Actually, let's use the decorator more directly
    # Re-initialize with direct registration
    return nav


def main():
    """Main application entry point."""

    # Initialize navigation manager
    nav = NavigationManager(default_page="home")

    # Register all pages
    # Home page
    @nav.page_registry(
        name="home",
        title="Home",
        icon="🏠",
        description="Main dashboard and overview of PV Circularity Simulator",
        access_level=AccessLevel.PUBLIC,
        order=0,
        keywords=["home", "dashboard", "overview"],
    )
    def home_page():
        home.render()

    # Cell Design page
    @nav.page_registry(
        name="cell_design",
        title="Cell Design",
        icon="🔬",
        description="Photovoltaic cell design and SCAPS integration",
        parent="home",
        order=1,
        keywords=["cell", "design", "scaps", "optimization"],
    )
    def cell_design_page():
        cell_design.render()

    # Module Engineering page
    @nav.page_registry(
        name="module_engineering",
        title="Module Engineering",
        icon="⚡",
        description="Solar module design and CTM loss analysis",
        parent="home",
        order=2,
        keywords=["module", "engineering", "ctm", "losses"],
    )
    def module_engineering_page():
        module_engineering.render()

    # System Planning page
    @nav.page_registry(
        name="system_planning",
        title="System Planning",
        icon="🏗️",
        description="System design, layout, and energy forecasting",
        parent="home",
        order=3,
        keywords=["system", "planning", "forecast", "sizing"],
    )
    def system_planning_page():
        system_planning.render()

    # Performance Monitoring page
    @nav.page_registry(
        name="performance_monitoring",
        title="Performance Monitoring",
        icon="📊",
        description="Real-time monitoring and reliability testing",
        parent="home",
        order=4,
        keywords=["performance", "monitoring", "reliability", "degradation"],
    )
    def performance_monitoring_page():
        performance_monitoring.render()

    # Circularity (3R) page
    @nav.page_registry(
        name="circularity",
        title="Circularity (3R)",
        icon="♻️",
        description="Circular economy modeling: Reduce, Reuse, Recycle",
        parent="home",
        order=5,
        keywords=["circularity", "circular economy", "3r", "recycling", "sustainability"],
    )
    def circularity_page():
        circularity.render()

    # Render sidebar navigation
    with st.sidebar:
        nav.render_sidebar_navigation(
            show_icons=True,
            group_by_parent=True,
            show_search=True,
        )

        # Add some sidebar info
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("- [Documentation](#)")
        st.markdown("- [GitHub Repository](#)")
        st.markdown("- [API Reference](#)")

        st.markdown("---")
        st.caption("PV Circularity Simulator v0.1.0")

    # Display breadcrumbs
    nav.breadcrumbs(separator=" → ", show_icons=True, clickable=False)

    st.markdown("---")

    # Handle routing and render current page
    nav.route_handler()

    # Optional: Display current route info in expander (for debugging)
    with st.expander("🔍 Debug: Navigation Info", expanded=False):
        st.write("**Current Page:**", nav.get_current_page())
        st.write("**Current Params:**", nav.get_current_params())
        st.write("**Previous Page:**", nav.get_previous_page())

        # Show deep link
        current_page = nav.get_current_page()
        params = nav.get_current_params()
        if params:
            from urllib.parse import urlencode
            query_string = urlencode({"page": current_page, **params})
            st.code(f"Deep link: ?{query_string}")
        else:
            st.code(f"Deep link: ?page={current_page}")


if __name__ == "__main__":
    main()
