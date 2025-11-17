"""
Hybrid Energy System UI - Main Application

This is the main entry point for the Hybrid Energy System UI application.
Run this file with Streamlit to launch the interactive dashboard.

Usage:
    streamlit run app.py

Author: PV Circularity Simulator Team
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.hybrid_system_ui import HybridSystemUI


def main():
    """
    Main application entry point.

    Initializes and renders the Hybrid Energy System UI.
    """
    # Create and render the UI
    ui = HybridSystemUI()
    ui.render()


if __name__ == "__main__":
    main()
