"""
State management for Streamlit application.

This module provides centralized state management for the PV Circularity Simulator,
handling session state, data persistence, and application state transitions.
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

from src.data.models import (
    Project, Resource, Contract, Portfolio, Timeline,
    ProjectStatus, ResourceType, ContractType, ContractStatus
)


class StateManager:
    """
    Centralized state management for the Streamlit application.

    Handles initialization, persistence, and management of application state
    including projects, resources, contracts, and portfolios.
    """

    DATA_DIR = Path("data")
    PROJECTS_FILE = DATA_DIR / "projects.json"
    RESOURCES_FILE = DATA_DIR / "resources.json"
    CONTRACTS_FILE = DATA_DIR / "contracts.json"
    PORTFOLIOS_FILE = DATA_DIR / "portfolios.json"
    TIMELINES_FILE = DATA_DIR / "timelines.json"

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize Streamlit session state with default values.

        Sets up all necessary session state variables for the application,
        including data stores, UI state, and user preferences.
        """
        # Create data directory if it doesn't exist
        cls.DATA_DIR.mkdir(exist_ok=True)

        # Initialize data stores
        if "projects" not in st.session_state:
            st.session_state.projects = cls._load_data(cls.PROJECTS_FILE, {})

        if "resources" not in st.session_state:
            st.session_state.resources = cls._load_data(cls.RESOURCES_FILE, {})

        if "contracts" not in st.session_state:
            st.session_state.contracts = cls._load_data(cls.CONTRACTS_FILE, {})

        if "portfolios" not in st.session_state:
            st.session_state.portfolios = cls._load_data(cls.PORTFOLIOS_FILE, {})

        if "timelines" not in st.session_state:
            st.session_state.timelines = cls._load_data(cls.TIMELINES_FILE, {})

        # Initialize UI state
        if "current_project_id" not in st.session_state:
            st.session_state.current_project_id = None

        if "current_portfolio_id" not in st.session_state:
            st.session_state.current_portfolio_id = None

        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Overview"

        # Initialize filters
        if "filter_status" not in st.session_state:
            st.session_state.filter_status = []

        if "filter_date_range" not in st.session_state:
            st.session_state.filter_date_range = None

        # Initialize wizard state
        if "wizard_step" not in st.session_state:
            st.session_state.wizard_step = 0

        if "wizard_data" not in st.session_state:
            st.session_state.wizard_data = {}

    @staticmethod
    def _load_data(file_path: Path, default: Any) -> Any:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file
            default: Default value if file doesn't exist

        Returns:
            Loaded data or default value
        """
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")
                return default
        return default

    @classmethod
    def save_all(cls) -> None:
        """Save all session state data to persistent storage."""
        try:
            cls._save_data(cls.PROJECTS_FILE, st.session_state.projects)
            cls._save_data(cls.RESOURCES_FILE, st.session_state.resources)
            cls._save_data(cls.CONTRACTS_FILE, st.session_state.contracts)
            cls._save_data(cls.PORTFOLIOS_FILE, st.session_state.portfolios)
            cls._save_data(cls.TIMELINES_FILE, st.session_state.timelines)
        except Exception as e:
            st.error(f"Failed to save data: {e}")

    @staticmethod
    def _save_data(file_path: Path, data: Any) -> None:
        """
        Save data to JSON file.

        Args:
            file_path: Path to JSON file
            data: Data to save
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    # Project Management Methods

    @staticmethod
    def add_project(project: Project) -> None:
        """
        Add a new project to the state.

        Args:
            project: Project instance to add
        """
        st.session_state.projects[project.id] = project.to_dict()
        StateManager.save_all()

    @staticmethod
    def update_project(project_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing project.

        Args:
            project_id: Project identifier
            updates: Dictionary of fields to update
        """
        if project_id in st.session_state.projects:
            st.session_state.projects[project_id].update(updates)
            StateManager.save_all()

    @staticmethod
    def delete_project(project_id: str) -> None:
        """
        Delete a project and its associated resources.

        Args:
            project_id: Project identifier
        """
        if project_id in st.session_state.projects:
            del st.session_state.projects[project_id]

            # Delete associated resources
            st.session_state.resources = {
                k: v for k, v in st.session_state.resources.items()
                if v.get("project_id") != project_id
            }

            # Delete associated contracts
            st.session_state.contracts = {
                k: v for k, v in st.session_state.contracts.items()
                if v.get("project_id") != project_id
            }

            StateManager.save_all()

    @staticmethod
    def get_project(project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project dictionary or None if not found
        """
        return st.session_state.projects.get(project_id)

    @staticmethod
    def get_all_projects() -> List[Dict[str, Any]]:
        """
        Retrieve all projects.

        Returns:
            List of project dictionaries
        """
        return list(st.session_state.projects.values())

    # Resource Management Methods

    @staticmethod
    def add_resource(resource: Resource) -> None:
        """
        Add a new resource to the state.

        Args:
            resource: Resource instance to add
        """
        st.session_state.resources[resource.id] = resource.to_dict()
        StateManager.save_all()

    @staticmethod
    def update_resource(resource_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing resource.

        Args:
            resource_id: Resource identifier
            updates: Dictionary of fields to update
        """
        if resource_id in st.session_state.resources:
            st.session_state.resources[resource_id].update(updates)
            StateManager.save_all()

    @staticmethod
    def delete_resource(resource_id: str) -> None:
        """
        Delete a resource.

        Args:
            resource_id: Resource identifier
        """
        if resource_id in st.session_state.resources:
            del st.session_state.resources[resource_id]
            StateManager.save_all()

    @staticmethod
    def get_project_resources(project_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all resources for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of resource dictionaries
        """
        return [
            r for r in st.session_state.resources.values()
            if r.get("project_id") == project_id
        ]

    # Contract Management Methods

    @staticmethod
    def add_contract(contract: Contract) -> None:
        """
        Add a new contract to the state.

        Args:
            contract: Contract instance to add
        """
        st.session_state.contracts[contract.id] = contract.to_dict()
        StateManager.save_all()

    @staticmethod
    def update_contract(contract_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing contract.

        Args:
            contract_id: Contract identifier
            updates: Dictionary of fields to update
        """
        if contract_id in st.session_state.contracts:
            st.session_state.contracts[contract_id].update(updates)
            StateManager.save_all()

    @staticmethod
    def delete_contract(contract_id: str) -> None:
        """
        Delete a contract.

        Args:
            contract_id: Contract identifier
        """
        if contract_id in st.session_state.contracts:
            del st.session_state.contracts[contract_id]
            StateManager.save_all()

    @staticmethod
    def get_project_contracts(project_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all contracts for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of contract dictionaries
        """
        return [
            c for c in st.session_state.contracts.values()
            if c.get("project_id") == project_id
        ]

    # Portfolio Management Methods

    @staticmethod
    def add_portfolio(portfolio: Portfolio) -> None:
        """
        Add a new portfolio to the state.

        Args:
            portfolio: Portfolio instance to add
        """
        st.session_state.portfolios[portfolio.id] = portfolio.to_dict()
        StateManager.save_all()

    @staticmethod
    def update_portfolio(portfolio_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing portfolio.

        Args:
            portfolio_id: Portfolio identifier
            updates: Dictionary of fields to update
        """
        if portfolio_id in st.session_state.portfolios:
            st.session_state.portfolios[portfolio_id].update(updates)
            StateManager.save_all()

    # Timeline Management Methods

    @staticmethod
    def add_timeline(timeline: Timeline) -> None:
        """
        Add a new timeline to the state.

        Args:
            timeline: Timeline instance to add
        """
        st.session_state.timelines[timeline.id] = timeline.to_dict()
        StateManager.save_all()

    @staticmethod
    def update_timeline(timeline_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing timeline.

        Args:
            timeline_id: Timeline identifier
            updates: Dictionary of fields to update
        """
        if timeline_id in st.session_state.timelines:
            st.session_state.timelines[timeline_id].update(updates)
            StateManager.save_all()

    @staticmethod
    def get_project_timeline(project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve timeline for a project.

        Args:
            project_id: Project identifier

        Returns:
            Timeline dictionary or None if not found
        """
        for timeline in st.session_state.timelines.values():
            if timeline.get("project_id") == project_id:
                return timeline
        return None
