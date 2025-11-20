"""
Main Streamlit application for PV Circularity Simulator.

This application provides an interactive interface for:
- Cell design and module engineering
- Thermal modeling and temperature prediction
- Performance monitoring
- Circular economy analysis
"""

import streamlit as st
from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

st.set_page_config(
    page_title="PV Circularity Simulator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">☀️ PV Circularity Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Welcome to the PV Circularity Simulator</h3>
    <p>
    An end-to-end photovoltaic lifecycle simulation platform covering everything from
    cell design through circular economy modeling.
    </p>
</div>
""", unsafe_allow_html=True)

# Main features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔬 Cell & Module Design
    - Cell-to-Module (CTM) loss analysis
    - Module engineering optimization
    - Material selection and design
    """)

with col2:
    st.markdown("""
    ### 🌡️ Thermal Modeling
    - Multiple temperature models (Sandia, PVsyst, Faiman)
    - NOCT-based predictions
    - Heat transfer analysis
    - B03 NOCT database integration
    """)

with col3:
    st.markdown("""
    ### ♻️ Circularity Analysis
    - Lifecycle assessment
    - Performance monitoring
    - End-of-life planning
    - Sustainability metrics
    """)

st.divider()

# Quick start guide
st.subheader("🚀 Quick Start")

st.markdown("""
1. **Navigate** using the sidebar to access different modules
2. **Thermal Modeling** - Start here to explore temperature prediction and cooling analysis
3. **Input your data** or use B03 verified module database
4. **Analyze results** with interactive charts and dashboards
5. **Export** your findings for further analysis
""")

st.divider()

# System information
with st.expander("ℹ️ System Information"):
    st.write("**Version:** 0.1.0")
    st.write("**Framework:** Streamlit + pvlib")
    st.write("**Features:**")
    st.write("- Advanced thermal modeling with multiple physics-based models")
    st.write("- B03 NOCT database with 20+ verified modules")
    st.write("- Real-time temperature predictions")
    st.write("- Comprehensive heat transfer calculations")
    st.write("- Interactive visualization and analysis tools")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    PV Circularity Simulator | Built with Streamlit & pvlib | MIT License
</div>
""", unsafe_allow_html=True)
