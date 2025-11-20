"""
Session Manager
===============

Manages user sessions and state persistence across modules.
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime


class SessionManager:
    """
    Manages session state for the PV Circularity Simulator.

    Handles data flow between modules and maintains simulation state
    throughout the user's workflow.
    """

    def __init__(self):
        """Initialize the session manager."""
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize default session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_module = 'dashboard'
            st.session_state.project_name = ''
            st.session_state.created_at = datetime.now()
            st.session_state.material_data = {}
            st.session_state.module_design_data = {}
            st.session_state.ctm_losses = {}
            st.session_state.system_design_data = {}
            st.session_state.eya_results = {}
            st.session_state.performance_data = {}
            st.session_state.fault_data = {}
            st.session_state.hya_results = {}
            st.session_state.forecast_data = {}
            st.session_state.revamp_data = {}
            st.session_state.circularity_data = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from session state.

        Args:
            key: Session state key
            default: Default value if key not found

        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in session state.

        Args:
            key: Session state key
            value: Value to store
        """
        st.session_state[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update multiple session state values.

        Args:
            data: Dictionary of key-value pairs to update
        """
        for key, value in data.items():
            st.session_state[key] = value

    def clear(self) -> None:
        """Clear all session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_session_state()

    def export_state(self) -> Dict[str, Any]:
        """
        Export current session state.

        Returns:
            Dictionary of session state data
        """
        return dict(st.session_state)

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import session state from dictionary.

        Args:
            state: Dictionary of session state data
        """
        self.update(state)
