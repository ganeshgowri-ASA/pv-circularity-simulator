"""
Main Streamlit application for PV Circularity Simulator.

This is the entry point for the web-based PV circularity and CTM testing interface.
"""

import streamlit as st

from pv_circularity_simulator.ui.pages.iec_ctm_testing import CTMTestUI


def main() -> None:
    """
    Main application entry point.

    Launches the PV Circularity Simulator web interface with CTM testing capabilities.
    """
    # Initialize and run CTM Test UI
    ui = CTMTestUI()
    ui.run()


if __name__ == "__main__":
    main()
