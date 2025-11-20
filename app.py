#!/usr/bin/env python3
"""
Main entry point for PV Circularity Simulator Financial Dashboard.

Run this file to launch the Streamlit financial analysis dashboard.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from financial.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()
