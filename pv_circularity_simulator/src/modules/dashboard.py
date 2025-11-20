"""
Dashboard Module
================

Main dashboard for project overview and navigation.
Displays project status, key metrics, and workflow progress.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the dashboard module.

    Args:
        session: Session manager instance

    Displays:
        - Project overview
        - Module completion status
        - Key performance indicators
        - Quick access to all modules
    """
    st.header("📊 Dashboard")

    # Project information
    st.subheader("Project Information")
    project_name = session.get('project_name', '')

    if not project_name:
        st.info("👋 Welcome! Please create a new project to get started.")
        new_project_name = st.text_input("Project Name")
        if st.button("Create Project"):
            if new_project_name:
                session.set('project_name', new_project_name)
                st.success(f"Project '{new_project_name}' created!")
                st.rerun()
            else:
                st.error("Please enter a project name")
    else:
        st.success(f"Current Project: **{project_name}**")

        # Module status overview
        st.subheader("Workflow Progress")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Material Selection", "Pending", "")
            st.metric("Module Design", "Pending", "")
            st.metric("CTM Analysis", "Pending", "")
            st.metric("System Design", "Pending", "")

        with col2:
            st.metric("EYA Simulation", "Pending", "")
            st.metric("Performance Monitor", "Pending", "")
            st.metric("Fault Diagnostics", "Pending", "")
            st.metric("HYA Simulation", "Pending", "")

        with col3:
            st.metric("Energy Forecasting", "Pending", "")
            st.metric("Revamp & Repower", "Pending", "")
            st.metric("Circularity (3R)", "Pending", "")

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Export Project"):
                st.info("Export functionality coming soon")

        with col2:
            if st.button("📊 Generate Report"):
                st.info("Report generation coming soon")

        with col3:
            if st.button("🗑️ Clear Project"):
                if st.button("Confirm Clear"):
                    session.clear()
                    st.success("Project cleared!")
                    st.rerun()
