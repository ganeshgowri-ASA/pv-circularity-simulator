"""
PV Circularity Simulator - Main Application
===========================================

Main entry point for the Streamlit application.
"""

import streamlit as st
from core.session_manager import SessionManager
from core.config import PAGE_CONFIG, APP_NAME, APP_DESCRIPTION, MODULES


def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(**PAGE_CONFIG)

    # Initialize session manager
    session = SessionManager()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    # Module selection
    module_options = {
        f"{config['icon']} {config['name']}": module_id
        for module_id, config in MODULES.items()
    }

    selected_display = st.sidebar.radio(
        "Select Module",
        options=list(module_options.keys()),
        index=0
    )
    selected_module = module_options[selected_display]

    # Main content
    st.title(APP_NAME)
    st.markdown(APP_DESCRIPTION)
    st.markdown("---")

    # Load selected module
    if selected_module == 'dashboard':
        from modules import dashboard
        dashboard.render(session)
    elif selected_module == 'material_selection':
        from modules import material_selection
        material_selection.render(session)
    elif selected_module == 'module_design':
        from modules import module_design
        module_design.render(session)
    elif selected_module == 'ctm_loss_analysis':
        from modules import ctm_loss_analysis
        ctm_loss_analysis.render(session)
    elif selected_module == 'system_design':
        from modules import system_design
        system_design.render(session)
    elif selected_module == 'eya_simulation':
        from modules import eya_simulation
        eya_simulation.render(session)
    elif selected_module == 'performance_monitoring':
        from modules import performance_monitoring
        performance_monitoring.render(session)
    elif selected_module == 'fault_diagnostics':
        from modules import fault_diagnostics
        fault_diagnostics.render(session)
    elif selected_module == 'hya_simulation':
        from modules import hya_simulation
        hya_simulation.render(session)
    elif selected_module == 'energy_forecasting':
        from modules import energy_forecasting
        energy_forecasting.render(session)
    elif selected_module == 'revamp_repower':
        from modules import revamp_repower
        revamp_repower.render(session)
    elif selected_module == 'circularity_3r':
        from modules import circularity_3r
        circularity_3r.render(session)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"Version {PAGE_CONFIG.get('page_icon')} 0.1.0")


if __name__ == "__main__":
    main()
