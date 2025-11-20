"""
PV Circularity Simulator - Main Application Entry Point

A comprehensive photovoltaic lifecycle simulation platform with advanced
planning, resource management, and portfolio tracking capabilities.

This application provides:
- Interactive project creation wizard
- Timeline and milestone planning
- Resource allocation dashboard
- Contract template management
- Portfolio management and analytics
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.state_manager import StateManager
from src.ui.planning import (
    project_wizard,
    timeline_planner,
    resource_allocation_dashboard,
    contract_templates
)


# Page configuration
st.set_page_config(
    page_title="PV Circularity Simulator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 1rem 2rem;
        }

        /* Sidebar styling */
        .css-1d391kg {
            padding: 2rem 1rem;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
        }

        /* Headers */
        h1 {
            color: #1E88E5;
            padding-bottom: 1rem;
        }

        h2 {
            color: #424242;
            padding-top: 1rem;
        }

        /* Form styling */
        .stForm {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 0.3rem;
            font-weight: 500;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Success/Info/Warning boxes */
        .stSuccess, .stInfo, .stWarning {
            padding: 1rem;
            border-radius: 0.5rem;
        }

        /* Divider */
        hr {
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the application sidebar with navigation and info."""
    with st.sidebar:
        st.title("☀️ PV Circularity Simulator")
        st.markdown("---")

        st.markdown("### Navigation")

        # Page selection
        page = st.radio(
            "Select Page",
            options=[
                "🏠 Home",
                "📊 Portfolio Dashboard",
                "🧙 Project Wizard",
                "📅 Timeline Planner",
                "📦 Resource Allocation",
                "📄 Contract Management"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### Quick Stats")

        projects = StateManager.get_all_projects()
        st.metric("Total Projects", len(projects))

        resources = list(st.session_state.resources.values())
        st.metric("Total Resources", len(resources))

        contracts = list(st.session_state.contracts.values())
        st.metric("Active Contracts", len(contracts))

        st.markdown("---")

        # Info
        st.markdown("### About")
        st.info(
            "PV Circularity Simulator provides comprehensive planning "
            "and management tools for photovoltaic projects."
        )

        st.markdown("**Version:** 1.0.0")
        st.markdown("**Build:** Production")

        return page


def render_home_page():
    """Render the home/landing page."""
    st.title("☀️ PV Circularity Simulator")
    st.markdown("### End-to-End Photovoltaic Lifecycle Management Platform")

    st.markdown("""
    Welcome to the PV Circularity Simulator - your comprehensive solution for managing
    photovoltaic projects from design through end-of-life circularity.
    """)

    st.divider()

    # Feature highlights
    st.markdown("## 🎯 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧙 Project Wizard")
        st.markdown("""
        Create new PV projects with guided step-by-step setup:
        - Basic project information
        - Technical specifications
        - Timeline and budget planning
        - Initial resource estimation
        """)

        st.markdown("### 📅 Timeline Planner")
        st.markdown("""
        Comprehensive project timeline management:
        - Gantt chart visualization
        - Milestone tracking
        - Phase planning
        - Dependency management
        """)

        st.markdown("### 📦 Resource Allocation")
        st.markdown("""
        Intelligent resource planning and tracking:
        - Inventory management
        - Cost analysis
        - Supplier tracking
        - Allocation optimization
        """)

    with col2:
        st.markdown("### 📄 Contract Management")
        st.markdown("""
        Complete contract lifecycle management:
        - Template library
        - File upload support
        - Status tracking
        - Payment schedules
        """)

        st.markdown("### 📊 Portfolio Dashboard")
        st.markdown("""
        High-level portfolio insights:
        - Multi-project overview
        - Budget tracking
        - Performance metrics
        - ROI analysis
        """)

        st.markdown("### ♻️ Circularity Integration")
        st.markdown("""
        End-of-life planning for sustainability:
        - 3R assessment (Reuse/Recycle/Recover)
        - Environmental impact analysis
        - Material tracking
        - Compliance monitoring
        """)

    st.divider()

    # Quick start
    st.markdown("## 🚀 Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Create Project")
        st.markdown("Use the **Project Wizard** to set up your first PV project.")
        if st.button("Launch Wizard", use_container_width=True):
            st.session_state.selected_page = "🧙 Project Wizard"
            st.rerun()

    with col2:
        st.markdown("### 2️⃣ Plan Timeline")
        st.markdown("Define milestones and phases in the **Timeline Planner**.")

    with col3:
        st.markdown("### 3️⃣ Allocate Resources")
        st.markdown("Add and track resources in the **Resource Dashboard**.")

    st.divider()

    # Recent activity
    st.markdown("## 📈 Recent Activity")

    projects = StateManager.get_all_projects()

    if projects:
        # Sort by updated date
        recent_projects = sorted(
            projects,
            key=lambda p: p.get("updated_date", ""),
            reverse=True
        )[:5]

        st.markdown("### Recently Updated Projects")

        for project in recent_projects:
            with st.expander(f"📁 {project['name']} - {project['status']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Owner:** {project['owner']}")
                    st.markdown(f"**Location:** {project['location']}")

                with col2:
                    st.markdown(f"**Capacity:** {project['capacity_kwp']} kWp")
                    st.markdown(f"**Budget:** ${project['budget']:,.2f}")

                with col3:
                    st.markdown(f"**Updated:** {project['updated_date'][:10]}")
    else:
        st.info("No projects yet. Create your first project to get started!")


def render_portfolio_dashboard():
    """Render the portfolio overview dashboard."""
    st.title("📊 Portfolio Dashboard")
    st.markdown("Comprehensive overview of all projects and portfolio metrics.")

    projects = StateManager.get_all_projects()

    if not projects:
        st.warning("No projects in portfolio. Create a project to get started.")
        return

    # Summary metrics
    st.markdown("### Portfolio Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Projects", len(projects))

    with col2:
        total_capacity = sum(p.get("capacity_kwp", 0) for p in projects)
        st.metric("Total Capacity", f"{total_capacity:,.1f} kWp")

    with col3:
        total_budget = sum(p.get("budget", 0) for p in projects)
        st.metric("Total Budget", f"${total_budget:,.0f}")

    with col4:
        # Count active projects
        active_count = sum(
            1 for p in projects
            if "Implementation" in p.get("status", "") or "Monitoring" in p.get("status", "")
        )
        st.metric("Active Projects", active_count)

    st.divider()

    # Projects by status
    st.markdown("### Projects by Status")

    import plotly.express as px
    import pandas as pd

    status_counts = {}
    for project in projects:
        status = project.get("status", "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    fig = px.pie(
        values=list(status_counts.values()),
        names=list(status_counts.keys()),
        title="Project Status Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Project list
    st.markdown("### All Projects")

    # Create dataframe
    df_data = []
    for project in projects:
        df_data.append({
            "Name": project["name"],
            "Owner": project["owner"],
            "Status": project["status"],
            "Capacity (kWp)": project.get("capacity_kwp", 0),
            "Budget": f"${project.get('budget', 0):,.0f}",
            "Location": project["location"]
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main application entry point."""
    # Initialize state
    StateManager.initialize()

    # Apply custom styling
    apply_custom_css()

    # Render sidebar and get selected page
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = None

    selected_page = render_sidebar()

    # Override if page was changed programmatically
    if st.session_state.selected_page:
        selected_page = st.session_state.selected_page
        st.session_state.selected_page = None

    # Render selected page
    try:
        if selected_page == "🏠 Home":
            render_home_page()

        elif selected_page == "📊 Portfolio Dashboard":
            render_portfolio_dashboard()

        elif selected_page == "🧙 Project Wizard":
            project_wizard()

        elif selected_page == "📅 Timeline Planner":
            timeline_planner()

        elif selected_page == "📦 Resource Allocation":
            resource_allocation_dashboard()

        elif selected_page == "📄 Contract Management":
            contract_templates()

        else:
            render_home_page()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "PV Circularity Simulator v1.0.0 | Production Build | "
        "© 2025 All Rights Reserved"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
