"""
Main application entry point for PV Circularity Simulator.

This script launches the Streamlit dashboard for weather data visualization
and system monitoring.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the dashboard
from pv_simulator.ui.dashboard import main

if __name__ == "__main__":
    main()
