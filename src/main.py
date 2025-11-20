"""
PV Circularity Simulator - Main Application
End-to-end PV lifecycle simulation platform
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.session_manager import SessionManager
from modules import (
    dashboard,
    materials_selection,
    cell_design,
    module_design,
    ctm_loss,
    iec_testing,
    system_design,
    eya,
    performance_monitoring,
    fault_diagnostics,
    energy_forecasting,
    revamp_repower,
    circularity,
    hybrid_systems,
    financial_modeling
)

# Version info
__version__ = "1.0.0"
__author__ = "PV Circularity Team"


def apply_custom_css():
    """Apply custom CSS styling to the application"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #1976D2;
        --accent-color: #FFA000;
        --background-color: #FAFAFA;
        --text-color: #212121;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    [data-testid="stSidebar"] .sidebar-content {
        padding: 1rem;
    }

    /* Module navigation buttons */
    .module-nav-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.75rem 1rem;
        text-align: left;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: white;
        transition: all 0.3s ease;
    }

    .module-nav-button:hover {
        background-color: #e8f5e9;
        border-color: #2E7D32;
        transform: translateX(5px);
    }

    /* Cards and containers */
    .info-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2E7D32;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background-color: #f5f5f5;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #757575;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }

    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 8px;
        padding: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header"""
    st.markdown("""
    <div class="main-header">
        <h1>☀️ PV Circularity Simulator</h1>
        <p>End-to-end PV Lifecycle Simulation Platform</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(session_manager):
    """Render sidebar with navigation and project management"""

    with st.sidebar:
        # App logo/title
        st.markdown("### ☀️ PV Circularity")
        st.markdown(f"**Version {__version__}**")
        st.markdown("---")

        # Project Management Section
        st.markdown("### 📁 Project")

        project_name = st.text_input(
            "Project Name",
            value=st.session_state.get('project_name', 'Untitled Project'),
            key="project_name_input"
        )

        if project_name != st.session_state.project_name:
            st.session_state.project_name = project_name

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🆕 New", use_container_width=True, help="Create new project"):
                session_manager.create_new_project("Untitled Project")
                st.success("New project created!")
                st.rerun()

        with col2:
            if st.button("💾 Save", use_container_width=True, help="Save current project"):
                # Default save location
                save_path = f"./projects/{st.session_state.project_name.replace(' ', '_')}.json"
                if session_manager.save_project(save_path):
                    st.success("Project saved!")
                else:
                    st.error("Failed to save project")

        # File uploader for loading projects
        uploaded_file = st.file_uploader(
            "Load Project",
            type=['json'],
            help="Upload a saved project file",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            import tempfile
            import os

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load the project
            if session_manager.load_project(tmp_path):
                st.success(f"Loaded: {st.session_state.project_name}")
                os.unlink(tmp_path)
                st.rerun()
            else:
                st.error("Failed to load project")
                os.unlink(tmp_path)

        st.markdown("---")

        # Module Navigation
        st.markdown("### 📑 Modules")

        # Define all modules with icons
        modules = [
            ("📊 Dashboard", "Dashboard"),
            ("🧪 Materials Selection", "Materials Selection"),
            ("🔬 Cell Design", "Cell Design"),
            ("📐 Module Design", "Module Design"),
            ("⚡ CTM Loss", "CTM Loss"),
            ("🧪 IEC Testing", "IEC Testing"),
            ("🏗️ System Design", "System Design"),
            ("☀️ EYA", "EYA"),
            ("📈 Performance Monitoring", "Performance Monitoring"),
            ("🔍 Fault Diagnostics", "Fault Diagnostics"),
            ("🔮 Energy Forecasting", "Energy Forecasting"),
            ("🔄 Revamp/Repower", "Revamp/Repower"),
            ("♻️ Circularity", "Circularity"),
            ("🔋 Hybrid Systems", "Hybrid Systems"),
            ("💰 Financial Modeling", "Financial Modeling"),
        ]

        # Render module navigation buttons
        current_module = session_manager.get_current_module()

        for icon_name, module_name in modules:
            # Highlight current module
            if module_name == current_module:
                st.markdown(f"**▶ {icon_name}**")
            else:
                if st.button(icon_name, use_container_width=True, key=f"nav_{module_name}"):
                    session_manager.set_current_module(module_name)
                    st.rerun()

        st.markdown("---")

        # Settings Toggle
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.get('show_settings', False)
            st.rerun()

        # Help Toggle
        if st.button("❓ Help", use_container_width=True):
            st.session_state.show_help = not st.session_state.get('show_help', False)
            st.rerun()

        st.markdown("---")

        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Modules", "15")
        st.metric("Project Age", f"{(st.session_state.get('project_created', '2024-01-01')[:10])}")


def render_settings_panel(session_manager):
    """Render settings panel"""
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["General", "Display", "Advanced"])

    with tab1:
        st.subheader("General Settings")

        col1, col2 = st.columns(2)

        with col1:
            units = st.selectbox(
                "Units System",
                ["Metric", "Imperial"],
                index=0 if session_manager.get_setting('units') == 'Metric' else 1
            )
            session_manager.update_setting('units', units)

            currency = st.selectbox(
                "Currency",
                ["USD", "EUR", "GBP", "JPY", "CNY", "INR"],
                index=["USD", "EUR", "GBP", "JPY", "CNY", "INR"].index(
                    session_manager.get_setting('currency', 'USD')
                )
            )
            session_manager.update_setting('currency', currency)

        with col2:
            language = st.selectbox(
                "Language",
                ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
                index=0
            )
            session_manager.update_setting('language', language)

            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"],
                index=0
            )
            session_manager.update_setting('date_format', date_format)

    with tab2:
        st.subheader("Display Settings")

        col1, col2 = st.columns(2)

        with col1:
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "Auto"],
                index=0
            )
            session_manager.update_setting('theme', theme)

            decimal_places = st.slider(
                "Decimal Places",
                min_value=0,
                max_value=6,
                value=session_manager.get_setting('decimal_places', 2)
            )
            session_manager.update_setting('decimal_places', decimal_places)

        with col2:
            st.checkbox("Show Module Descriptions", value=True)
            st.checkbox("Show Tooltips", value=True)
            st.checkbox("Enable Animations", value=True)

    with tab3:
        st.subheader("Advanced Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Auto-save Project", value=False)
            st.number_input("Auto-save Interval (minutes)", min_value=1, max_value=60, value=5)

        with col2:
            st.checkbox("Enable Debug Mode", value=False)
            st.checkbox("Cache Calculations", value=True)

        st.markdown("---")
        st.markdown("#### Data Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export Settings", use_container_width=True):
                st.info("Settings export not yet implemented")

        with col2:
            if st.button("Import Settings", use_container_width=True):
                st.info("Settings import not yet implemented")

        with col3:
            if st.button("Reset to Defaults", use_container_width=True):
                session_manager.initialize_session_state()
                st.success("Settings reset to defaults!")
                st.rerun()

    st.markdown("---")
    if st.button("✓ Close Settings", use_container_width=True):
        st.session_state.show_settings = False
        st.rerun()


def render_help_panel():
    """Render help panel"""
    st.markdown("## ❓ Help & Documentation")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Quick Start", "Modules Guide", "Resources", "About"])

    with tab1:
        st.markdown("""
        ### Quick Start Guide

        #### Getting Started
        1. **Create a New Project**: Click the "🆕 New" button in the sidebar
        2. **Navigate Modules**: Use the sidebar to access different simulation modules
        3. **Enter Data**: Fill in the required parameters for each module
        4. **View Results**: Each module provides visualizations and analysis
        5. **Save Your Work**: Click "💾 Save" to save your project

        #### Module Workflow
        The modules are organized to follow a typical PV system lifecycle:

        1. **Design Phase**: Materials → Cell Design → Module Design
        2. **Analysis Phase**: CTM Loss → IEC Testing → System Design
        3. **Planning Phase**: EYA → Financial Modeling
        4. **Operations Phase**: Performance Monitoring → Fault Diagnostics
        5. **Optimization**: Energy Forecasting → Revamp/Repower
        6. **Sustainability**: Circularity → Hybrid Systems

        #### Tips
        - Use the Dashboard for an overview of your project
        - Save your project regularly
        - Explore the Settings to customize the application
        - Check module-specific help for detailed guidance
        """)

    with tab2:
        st.markdown("""
        ### Module Descriptions

        #### Design & Engineering
        - **Materials Selection**: Choose and configure PV materials
        - **Cell Design**: Design solar cells with SCAPS integration
        - **Module Design**: Configure module layout and specifications
        - **CTM Loss**: Analyze cell-to-module power losses

        #### Testing & Validation
        - **IEC Testing**: Simulate IEC compliance testing
        - **System Design**: Design complete PV systems

        #### Performance & Analysis
        - **EYA**: Energy yield assessment and P50/P90 analysis
        - **Performance Monitoring**: Real-time system monitoring
        - **Fault Diagnostics**: AI-powered fault detection

        #### Forecasting & Planning
        - **Energy Forecasting**: ML-based production forecasting
        - **Revamp/Repower**: System upgrade analysis

        #### Sustainability & Economics
        - **Circularity**: 3R analysis (Reduce, Reuse, Recycle)
        - **Hybrid Systems**: PV + storage configurations
        - **Financial Modeling**: Comprehensive financial analysis
        """)

    with tab3:
        st.markdown("""
        ### Resources & Links

        #### Documentation
        - [User Manual](#) - Complete user guide
        - [API Documentation](#) - Developer documentation
        - [Video Tutorials](#) - Step-by-step tutorials

        #### Standards & References
        - [IEC 61215](https://webstore.iec.ch/) - PV module design qualification
        - [IEC 61730](https://webstore.iec.ch/) - PV module safety qualification
        - [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) - Solar radiation data
        - [NREL](https://www.nrel.gov/) - Renewable energy research

        #### Tools & Calculators
        - [SCAPS](https://scaps.elis.ugent.be/) - Solar cell simulation
        - [PVWatts](https://pvwatts.nrel.gov/) - Energy production estimator
        - [SAM](https://sam.nrel.gov/) - System Advisor Model

        #### Community & Support
        - [GitHub Repository](#) - Source code and issues
        - [Community Forum](#) - User discussions
        - [Contact Support](#) - Technical support

        #### Publications
        - Latest research on PV circularity
        - Lifecycle assessment methodologies
        - Best practices for PV systems
        """)

    with tab4:
        st.markdown(f"""
        ### About PV Circularity Simulator

        **Version**: {__version__}

        **Description**: End-to-end PV lifecycle simulation platform covering cell design,
        module engineering, system planning, performance monitoring, and circularity (3R).

        #### Features
        - ✅ 15 comprehensive simulation modules
        - ✅ SCAPS integration for cell design
        - ✅ IEC standards compliance testing
        - ✅ AI-powered fault diagnostics
        - ✅ ML-based energy forecasting
        - ✅ Circular economy analysis (3R)
        - ✅ Hybrid system design
        - ✅ Advanced financial modeling

        #### Technology Stack
        - **Framework**: Streamlit
        - **Language**: Python 3.8+
        - **ML Libraries**: scikit-learn, TensorFlow
        - **Data Processing**: pandas, numpy
        - **Visualization**: plotly, matplotlib

        #### Development Team
        {__author__}

        #### License
        This software is proprietary. All rights reserved.

        #### Acknowledgments
        - SCAPS development team
        - IEC standards committee
        - PV research community

        ---

        © 2024 PV Circularity Simulator. All rights reserved.
        """)

    st.markdown("---")
    if st.button("✓ Close Help", use_container_width=True):
        st.session_state.show_help = False
        st.rerun()


def render_module(module_name):
    """Render the selected module with error handling"""
    try:
        # Map module names to module objects
        module_map = {
            "Dashboard": dashboard,
            "Materials Selection": materials_selection,
            "Cell Design": cell_design,
            "Module Design": module_design,
            "CTM Loss": ctm_loss,
            "IEC Testing": iec_testing,
            "System Design": system_design,
            "EYA": eya,
            "Performance Monitoring": performance_monitoring,
            "Fault Diagnostics": fault_diagnostics,
            "Energy Forecasting": energy_forecasting,
            "Revamp/Repower": revamp_repower,
            "Circularity": circularity,
            "Hybrid Systems": hybrid_systems,
            "Financial Modeling": financial_modeling,
        }

        # Get and render the module
        module = module_map.get(module_name)
        if module:
            module.render()
        else:
            st.error(f"Module '{module_name}' not found!")

    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        st.exception(e)

        if st.button("🔄 Reload Module"):
            st.rerun()


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p>PV Circularity Simulator v{__version__} | © 2024 {__author__}</p>
        <p>End-to-end PV lifecycle simulation platform</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="PV Circularity Simulator",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/pv-circularity-simulator/help',
            'Report a bug': 'https://github.com/pv-circularity-simulator/issues',
            'About': f"PV Circularity Simulator v{__version__} - End-to-end PV lifecycle simulation platform"
        }
    )

    # Apply custom CSS
    apply_custom_css()

    # Initialize session manager
    session_manager = SessionManager()

    # Render sidebar
    render_sidebar(session_manager)

    # Main content area
    render_header()

    # Check if settings or help panel should be shown
    if st.session_state.get('show_settings', False):
        render_settings_panel(session_manager)
    elif st.session_state.get('show_help', False):
        render_help_panel()
    else:
        # Render the current module
        current_module = session_manager.get_current_module()

        # Loading state
        with st.spinner(f"Loading {current_module}..."):
            render_module(current_module)

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
