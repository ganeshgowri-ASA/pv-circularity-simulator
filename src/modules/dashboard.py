"""
Dashboard Module - Overview and key metrics
"""

import streamlit as st


def render():
    """Render the Dashboard module"""
    st.header("📊 Dashboard")
    st.markdown("---")

    st.markdown("""
    ### Welcome to PV Circularity Simulator

    End-to-end PV lifecycle simulation platform covering:
    - Cell design → Module engineering → System planning
    - Performance monitoring → Circularity (3R)
    - CTM loss analysis, SCAPS integration, reliability testing
    - Energy forecasting and circular economy modeling
    """)

    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Projects", "0", "0")

    with col2:
        st.metric("Active Simulations", "0", "0")

    with col3:
        st.metric("Energy Forecast", "0 MWh", "0%")

    with col4:
        st.metric("Circularity Score", "0%", "0%")

    st.markdown("---")

    # Quick access
    st.subheader("Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔬 Start Cell Design", use_container_width=True):
            st.session_state.current_module = "Cell Design"
            st.rerun()

    with col2:
        if st.button("📐 Module Design", use_container_width=True):
            st.session_state.current_module = "Module Design"
            st.rerun()

    with col3:
        if st.button("🔄 Circularity Analysis", use_container_width=True):
            st.session_state.current_module = "Circularity"
            st.rerun()

    # Recent activity
    st.markdown("---")
    st.subheader("Recent Activity")
    st.info("No recent activity to display. Start by creating a new project.")
