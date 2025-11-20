"""
PV Circularity Simulator - Main Application
============================================

Main entry point for the PV Circularity Simulator Streamlit application.

Usage:
    streamlit run app.py

Author: PV Circularity Simulator Team
Version: 1.0.0
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from modules.dashboard import render_dashboard

# Configure Streamlit page
st.set_page_config(
    page_title="PV Circularity Simulator",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main application
if __name__ == "__main__":
    render_dashboard()
