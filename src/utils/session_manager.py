"""
Session Manager for PV Circularity Simulator
Handles project state, settings, and data persistence
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class SessionManager:
    """Manages application session state and project data"""

    def __init__(self):
        """Initialize session manager with default settings"""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""

        # Project metadata
        if 'project_name' not in st.session_state:
            st.session_state.project_name = "Untitled Project"

        if 'project_created' not in st.session_state:
            st.session_state.project_created = datetime.now().isoformat()

        if 'project_modified' not in st.session_state:
            st.session_state.project_modified = datetime.now().isoformat()

        # Navigation
        if 'current_module' not in st.session_state:
            st.session_state.current_module = "Dashboard"

        # Settings
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'units': 'Metric',
                'currency': 'USD',
                'language': 'English',
                'theme': 'Light',
                'decimal_places': 2,
                'date_format': 'YYYY-MM-DD'
            }

        # Module data storage
        if 'module_data' not in st.session_state:
            st.session_state.module_data = {}

        # Project file path
        if 'project_file_path' not in st.session_state:
            st.session_state.project_file_path = None

        # User preferences
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False

        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False

    def create_new_project(self, project_name: str = "Untitled Project"):
        """Create a new project with default settings"""
        st.session_state.project_name = project_name
        st.session_state.project_created = datetime.now().isoformat()
        st.session_state.project_modified = datetime.now().isoformat()
        st.session_state.module_data = {}
        st.session_state.project_file_path = None
        st.session_state.current_module = "Dashboard"

    def save_project(self, file_path: str) -> bool:
        """
        Save current project to JSON file

        Args:
            file_path: Path to save the project file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            project_data = {
                'project_name': st.session_state.project_name,
                'project_created': st.session_state.project_created,
                'project_modified': datetime.now().isoformat(),
                'settings': st.session_state.settings,
                'module_data': st.session_state.module_data,
                'version': '1.0.0'
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)

            st.session_state.project_file_path = file_path
            st.session_state.project_modified = datetime.now().isoformat()

            return True
        except Exception as e:
            st.error(f"Error saving project: {str(e)}")
            return False

    def load_project(self, file_path: str) -> bool:
        """
        Load project from JSON file

        Args:
            file_path: Path to the project file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            st.session_state.project_name = project_data.get('project_name', 'Untitled Project')
            st.session_state.project_created = project_data.get('project_created')
            st.session_state.project_modified = project_data.get('project_modified')
            st.session_state.settings = project_data.get('settings', st.session_state.settings)
            st.session_state.module_data = project_data.get('module_data', {})
            st.session_state.project_file_path = file_path

            return True
        except Exception as e:
            st.error(f"Error loading project: {str(e)}")
            return False

    def update_setting(self, key: str, value: Any):
        """Update a specific setting"""
        if 'settings' not in st.session_state:
            st.session_state.settings = {}
        st.session_state.settings[key] = value
        st.session_state.project_modified = datetime.now().isoformat()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        return st.session_state.settings.get(key, default)

    def save_module_data(self, module_name: str, data: Dict[str, Any]):
        """Save data for a specific module"""
        if 'module_data' not in st.session_state:
            st.session_state.module_data = {}
        st.session_state.module_data[module_name] = data
        st.session_state.project_modified = datetime.now().isoformat()

    def get_module_data(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific module"""
        return st.session_state.module_data.get(module_name)

    def set_current_module(self, module_name: str):
        """Set the current active module"""
        st.session_state.current_module = module_name

    def get_current_module(self) -> str:
        """Get the current active module"""
        return st.session_state.current_module

    def export_to_dict(self) -> Dict[str, Any]:
        """Export current session to dictionary"""
        return {
            'project_name': st.session_state.project_name,
            'project_created': st.session_state.project_created,
            'project_modified': st.session_state.project_modified,
            'settings': st.session_state.settings,
            'module_data': st.session_state.module_data,
            'current_module': st.session_state.current_module
        }
