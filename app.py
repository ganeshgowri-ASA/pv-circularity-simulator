"""Energy Yield Analysis Dashboard - Main Streamlit Application.

This is the main entry point for the multi-page Streamlit dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PV Energy Yield Analysis Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">☀️ PV Energy Yield Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=PV+Simulator", use_container_width=True)
    st.markdown("---")
    st.markdown("## Navigation")
    st.markdown("""
    Welcome to the **Energy Yield Analysis Dashboard**!

    Use the navigation above to explore:
    - 🏠 **Home**: Project overview and quick stats
    - 📊 **Energy Analysis**: Detailed energy production analysis
    - 📉 **Performance**: Performance ratio and metrics
    - 🔻 **Losses**: System loss breakdown
    - 💰 **Financial**: Financial analysis and metrics
    - 📋 **Reports**: Generate comprehensive reports
    - 📈 **Visualizations**: Interactive charts and graphs
    """)

    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **PV Circularity Simulator**
    Version 0.1.0

    End-to-end PV lifecycle simulation platform with comprehensive Energy Yield Analysis.
    """)

# Main content
st.markdown('<h2 class="section-header">Welcome to Energy Yield Analysis</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🔋 Module",
        value="B06",
        delta="Energy Yield Analysis",
    )

with col2:
    st.metric(
        label="📅 Version",
        value="0.1.0",
        delta="Production Ready",
    )

with col3:
    st.metric(
        label="⚡ Status",
        value="Active",
        delta="Fully Operational",
    )

with col4:
    st.metric(
        label="🎯 Accuracy",
        value="High",
        delta="Industry Standard",
    )

st.markdown("---")

# Overview sections
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<h3 class="section-header">Key Features</h3>', unsafe_allow_html=True)
    st.markdown("""
    ✅ **Comprehensive Analysis**
    - Annual and monthly energy production forecasting
    - Performance ratio calculations
    - Detailed loss analysis
    - Financial metrics (LCOE, NPV, IRR)

    ✅ **Advanced Analytics**
    - Sensitivity analysis
    - Probabilistic analysis (P50/P90/P99)
    - Weather correlation analysis
    - Degradation projections

    ✅ **Professional Reporting**
    - PDF report generation
    - Excel export with multiple sheets
    - Interactive visualizations
    - Customizable parameters
    """)

with col_right:
    st.markdown('<h3 class="section-header">Quick Start Guide</h3>', unsafe_allow_html=True)
    st.markdown("""
    **Step 1: Configure Project** 📝
    Navigate to the sidebar and select your analysis page.

    **Step 2: Input Parameters** ⚙️
    Enter your project details, system configuration, and location data.

    **Step 3: Run Analysis** 🚀
    Click "Calculate" to generate comprehensive energy yield analysis.

    **Step 4: Review Results** 📊
    Explore interactive charts, tables, and performance metrics.

    **Step 5: Generate Reports** 📄
    Export professional PDF and Excel reports for stakeholders.
    """)

st.markdown("---")

# System modules overview
st.markdown('<h3 class="section-header">System Modules</h3>', unsafe_allow_html=True)

modules_col1, modules_col2, modules_col3 = st.columns(3)

with modules_col1:
    with st.expander("🔮 B05: Energy Forecasting"):
        st.markdown("""
        **Energy Forecasting Module**

        - Weather data processing
        - POA irradiance calculations
        - Cell temperature modeling
        - DC/AC power forecasting
        - Hourly energy production
        - Uncertainty quantification
        """)

with modules_col2:
    with st.expander("📊 B06: Energy Yield Analysis"):
        st.markdown("""
        **Energy Yield Analysis Module**

        - Performance metrics calculation
        - Detailed loss analysis
        - Financial analysis (LCOE, NPV, IRR)
        - Sensitivity analysis
        - Probabilistic analysis (P-values)
        - Monthly/annual aggregations
        """)

with modules_col3:
    with st.expander("📈 Interactive Dashboard"):
        st.markdown("""
        **Dashboard Features**

        - Multi-page Streamlit interface
        - Plotly interactive charts
        - Altair declarative visualizations
        - PDF report generation
        - Excel export capabilities
        - Real-time calculations
        """)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    © {datetime.now().year} PV Circularity Simulator | Powered by Streamlit, Plotly, and pvlib
    </div>
    """,
    unsafe_allow_html=True,
)

# Instructions
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    ### Getting Started

    1. **Select a Page**: Use the sidebar navigation to access different analysis modules
    2. **Configure Parameters**: Enter your project and system configuration details
    3. **Run Analysis**: Click the "Calculate" or "Analyze" button to generate results
    4. **Explore Results**: Review charts, tables, and metrics
    5. **Export Reports**: Download PDF or Excel reports for documentation

    ### Navigation Guide

    - **Home**: Overview and welcome page (current page)
    - **Energy Analysis**: Annual and monthly energy production analysis
    - **Performance**: Performance ratio and system efficiency metrics
    - **Losses**: Detailed breakdown of system losses
    - **Financial**: Economic analysis including LCOE, NPV, and IRR
    - **Reports**: Generate and download comprehensive reports
    - **Visualizations**: Interactive charts and correlation analysis

    ### Tips

    - Use realistic values for accurate results
    - Check units carefully when entering data
    - Hover over charts for detailed information
    - Download reports for offline review
    - Adjust parameters to perform sensitivity analysis
    """)
