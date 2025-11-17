"""
Session Manager for PV Circularity Simulator.

This module handles project management, session state, activity logging,
and progress tracking across all 11 simulation modules.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, field, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Module Enumeration
# ============================================================================

class SimulationModule(str, Enum):
    """All simulation modules in the PV Circularity Simulator."""
    MATERIAL_SELECTION = "material_selection"
    CELL_DESIGN = "cell_design"
    MODULE_ENGINEERING = "module_engineering"
    CUTTING_PATTERN = "cutting_pattern"
    SYSTEM_DESIGN = "system_design"
    PERFORMANCE_SIMULATION = "performance_simulation"
    FINANCIAL_ANALYSIS = "financial_analysis"
    RELIABILITY_TESTING = "reliability_testing"
    CIRCULARITY_ASSESSMENT = "circularity_assessment"
    SCAPS_INTEGRATION = "scaps_integration"
    ENERGY_FORECASTING = "energy_forecasting"


# ============================================================================
# Type Definitions
# ============================================================================

class ActivityEntry(TypedDict):
    """Type definition for activity log entry."""
    timestamp: str
    module: str
    action: str
    details: Optional[str]
    user: Optional[str]


@dataclass
class ModuleCompletionStatus:
    """Completion status for a simulation module."""
    module_name: str
    completed: bool = False
    completion_percentage: float = 0.0
    last_updated: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'module_name': self.module_name,
            'completed': self.completed,
            'completion_percentage': self.completion_percentage,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'data': self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleCompletionStatus':
        """Create from dictionary."""
        last_updated = data.get('last_updated')
        if last_updated and isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)

        return cls(
            module_name=data['module_name'],
            completed=data.get('completed', False),
            completion_percentage=data.get('completion_percentage', 0.0),
            last_updated=last_updated,
            data=data.get('data', {})
        )


@dataclass
class ProjectMetadata:
    """Metadata for a simulation project."""
    project_name: str
    project_id: str
    created_date: datetime
    last_modified: datetime
    description: str = ""
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'project_name': self.project_name,
            'project_id': self.project_id,
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectMetadata':
        """Create from dictionary."""
        return cls(
            project_name=data['project_name'],
            project_id=data['project_id'],
            created_date=datetime.fromisoformat(data['created_date']),
            last_modified=datetime.fromisoformat(data['last_modified']),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author'),
            tags=data.get('tags', [])
        )


@dataclass
class SessionState:
    """Complete session state for the simulator."""
    metadata: ProjectMetadata
    module_status: Dict[str, ModuleCompletionStatus]
    activity_log: List[ActivityEntry] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata.to_dict(),
            'module_status': {
                k: v.to_dict() for k, v in self.module_status.items()
            },
            'activity_log': self.activity_log,
            'custom_data': self.custom_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary."""
        return cls(
            metadata=ProjectMetadata.from_dict(data['metadata']),
            module_status={
                k: ModuleCompletionStatus.from_dict(v)
                for k, v in data['module_status'].items()
            },
            activity_log=data.get('activity_log', []),
            custom_data=data.get('custom_data', {})
        )


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """
    Manages simulation sessions, project state, and activity logging.

    Handles project creation, loading, saving, and tracking progress
    across all 11 simulation modules.
    """

    def __init__(self, projects_directory: Optional[Path] = None):
        """
        Initialize the SessionManager.

        Args:
            projects_directory: Directory to store project files.
                               Defaults to ./projects
        """
        self.projects_directory = projects_directory or Path("./projects")
        self.projects_directory.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[SessionState] = None
        self.current_project_path: Optional[Path] = None

        logger.info(f"SessionManager initialized with directory: {self.projects_directory}")

    def initialize_session_state(
        self,
        project_name: str,
        project_id: str,
        description: str = "",
        author: Optional[str] = None
    ) -> SessionState:
        """
        Initialize a new session state with all 11 module completion flags.

        Args:
            project_name: Name of the project
            project_id: Unique project identifier
            description: Project description
            author: Project author name

        Returns:
            Initialized SessionState object
        """
        # Create metadata
        now = datetime.now()
        metadata = ProjectMetadata(
            project_name=project_name,
            project_id=project_id,
            created_date=now,
            last_modified=now,
            description=description,
            author=author
        )

        # Initialize all 11 module completion statuses
        module_status: Dict[str, ModuleCompletionStatus] = {}
        for module in SimulationModule:
            module_status[module.value] = ModuleCompletionStatus(
                module_name=module.value,
                completed=False,
                completion_percentage=0.0,
                last_updated=None
            )

        # Create session state
        session_state = SessionState(
            metadata=metadata,
            module_status=module_status,
            activity_log=[],
            custom_data={}
        )

        logger.info(f"Initialized session state for project: {project_name} (ID: {project_id})")

        return session_state

    def create_new_project(
        self,
        project_name: str,
        description: str = "",
        author: Optional[str] = None
    ) -> SessionState:
        """
        Create a new project with initialized session state.

        Args:
            project_name: Name of the project
            description: Project description
            author: Project author name

        Returns:
            New SessionState object

        Raises:
            ValueError: If project name already exists
        """
        # Generate project ID from name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{project_name.lower().replace(' ', '_')}_{timestamp}"

        # Check if project already exists
        project_path = self.projects_directory / f"{project_id}.json"
        if project_path.exists():
            raise ValueError(f"Project already exists: {project_id}")

        # Initialize session state
        session_state = self.initialize_session_state(
            project_name=project_name,
            project_id=project_id,
            description=description,
            author=author
        )

        # Set as current session
        self.current_session = session_state
        self.current_project_path = project_path

        # Log activity
        self.log_activity(
            module="system",
            action="create_project",
            details=f"Created new project: {project_name}"
        )

        # Save project
        self.save_project()

        logger.info(f"Created new project: {project_name} at {project_path}")

        return session_state

    def save_project(self, project_path: Optional[Path] = None) -> Path:
        """
        Save the current project to disk.

        Args:
            project_path: Path to save the project. If None, uses current project path.

        Returns:
            Path where project was saved

        Raises:
            ValueError: If no current session exists
        """
        if self.current_session is None:
            raise ValueError("No current session to save")

        # Determine save path
        if project_path is None:
            if self.current_project_path is None:
                # Generate default path
                project_id = self.current_session.metadata.project_id
                project_path = self.projects_directory / f"{project_id}.json"
            else:
                project_path = self.current_project_path

        # Update last modified timestamp
        self.current_session.metadata.last_modified = datetime.now()

        # Convert to dictionary and save as JSON
        project_data = self.current_session.to_dict()

        try:
            with open(project_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            self.current_project_path = project_path
            logger.info(f"Saved project to: {project_path}")

            return project_path

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise

    def load_project(self, project_path: Path) -> SessionState:
        """
        Load a project from disk.

        Args:
            project_path: Path to the project file

        Returns:
            Loaded SessionState object

        Raises:
            FileNotFoundError: If project file doesn't exist
            ValueError: If project file is invalid
        """
        if not project_path.exists():
            raise FileNotFoundError(f"Project file not found: {project_path}")

        try:
            with open(project_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            # Reconstruct session state
            session_state = SessionState.from_dict(project_data)

            # Set as current session
            self.current_session = session_state
            self.current_project_path = project_path

            logger.info(f"Loaded project: {session_state.metadata.project_name} from {project_path}")

            # Log activity
            self.log_activity(
                module="system",
                action="load_project",
                details=f"Loaded project: {session_state.metadata.project_name}"
            )

            return session_state

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in project file: {e}")
            raise ValueError(f"Invalid project file format: {e}")
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all available projects.

        Returns:
            List of project summaries with metadata
        """
        projects = []

        for project_file in self.projects_directory.glob("*.json"):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)

                metadata = project_data.get('metadata', {})
                module_status = project_data.get('module_status', {})

                # Calculate overall completion
                total_modules = len(module_status)
                completed_modules = sum(
                    1 for status in module_status.values()
                    if status.get('completed', False)
                )
                overall_percentage = (completed_modules / total_modules * 100) if total_modules > 0 else 0

                projects.append({
                    'project_name': metadata.get('project_name', 'Unknown'),
                    'project_id': metadata.get('project_id', ''),
                    'created_date': metadata.get('created_date', ''),
                    'last_modified': metadata.get('last_modified', ''),
                    'description': metadata.get('description', ''),
                    'author': metadata.get('author'),
                    'completion_percentage': overall_percentage,
                    'completed_modules': completed_modules,
                    'total_modules': total_modules,
                    'file_path': str(project_file)
                })

            except Exception as e:
                logger.warning(f"Failed to read project file {project_file}: {e}")
                continue

        # Sort by last modified date (newest first)
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)

        logger.info(f"Listed {len(projects)} projects")

        return projects

    def log_activity(
        self,
        module: str,
        action: str,
        details: Optional[str] = None,
        user: Optional[str] = None
    ) -> None:
        """
        Log an activity in the current session.

        Args:
            module: Module name where activity occurred
            action: Action performed
            details: Additional details about the activity
            user: User who performed the action
        """
        if self.current_session is None:
            logger.warning("No current session - activity not logged")
            return

        activity_entry: ActivityEntry = {
            'timestamp': datetime.now().isoformat(),
            'module': module,
            'action': action,
            'details': details,
            'user': user
        }

        self.current_session.activity_log.append(activity_entry)

        logger.debug(f"Logged activity: {module} - {action}")

    def get_completion_percentage(self, module: Optional[str] = None) -> float:
        """
        Get completion percentage for a specific module or overall project.

        Args:
            module: Module name (from SimulationModule enum).
                   If None, returns overall project completion.

        Returns:
            Completion percentage (0-100)

        Raises:
            ValueError: If no current session or invalid module name
        """
        if self.current_session is None:
            raise ValueError("No current session")

        if module is None:
            # Calculate overall completion
            total_percentage = sum(
                status.completion_percentage
                for status in self.current_session.module_status.values()
            )
            overall_percentage = total_percentage / len(self.current_session.module_status)
            return overall_percentage

        else:
            # Get specific module completion
            if module not in self.current_session.module_status:
                raise ValueError(f"Invalid module name: {module}")

            return self.current_session.module_status[module].completion_percentage

    def update_module_status(
        self,
        module: str,
        completion_percentage: Optional[float] = None,
        completed: Optional[bool] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the status of a simulation module.

        Args:
            module: Module name (from SimulationModule enum)
            completion_percentage: New completion percentage (0-100)
            completed: Whether module is completed
            data: Additional module-specific data

        Raises:
            ValueError: If no current session or invalid module
        """
        if self.current_session is None:
            raise ValueError("No current session")

        if module not in self.current_session.module_status:
            raise ValueError(f"Invalid module name: {module}")

        status = self.current_session.module_status[module]

        # Update fields
        if completion_percentage is not None:
            if not 0 <= completion_percentage <= 100:
                raise ValueError("Completion percentage must be between 0 and 100")
            status.completion_percentage = completion_percentage

        if completed is not None:
            status.completed = completed
            # If marked complete, set percentage to 100
            if completed:
                status.completion_percentage = 100.0

        if data is not None:
            status.data.update(data)

        status.last_updated = datetime.now()

        logger.info(
            f"Updated module {module}: "
            f"{status.completion_percentage:.1f}% complete, "
            f"completed={status.completed}"
        )

        # Log activity
        self.log_activity(
            module=module,
            action="update_status",
            details=f"Completion: {status.completion_percentage:.1f}%"
        )

    def get_module_status(self, module: str) -> ModuleCompletionStatus:
        """
        Get the status of a specific module.

        Args:
            module: Module name (from SimulationModule enum)

        Returns:
            ModuleCompletionStatus object

        Raises:
            ValueError: If no current session or invalid module
        """
        if self.current_session is None:
            raise ValueError("No current session")

        if module not in self.current_session.module_status:
            raise ValueError(f"Invalid module name: {module}")

        return self.current_session.module_status[module]

    def get_all_module_statuses(self) -> Dict[str, ModuleCompletionStatus]:
        """
        Get status of all modules.

        Returns:
            Dictionary mapping module names to their status

        Raises:
            ValueError: If no current session
        """
        if self.current_session is None:
            raise ValueError("No current session")

        return self.current_session.module_status.copy()

    def get_activity_log(
        self,
        module: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ActivityEntry]:
        """
        Get activity log entries.

        Args:
            module: Filter by module name (None for all)
            limit: Maximum number of entries to return (None for all)

        Returns:
            List of activity log entries (newest first)

        Raises:
            ValueError: If no current session
        """
        if self.current_session is None:
            raise ValueError("No current session")

        log = self.current_session.activity_log.copy()

        # Filter by module if specified
        if module is not None:
            log = [entry for entry in log if entry['module'] == module]

        # Reverse to get newest first
        log.reverse()

        # Apply limit if specified
        if limit is not None and limit > 0:
            log = log[:limit]

        return log

    def export_session_summary(self) -> Dict[str, Any]:
        """
        Export a summary of the current session.

        Returns:
            Dictionary with session summary

        Raises:
            ValueError: If no current session
        """
        if self.current_session is None:
            raise ValueError("No current session")

        summary = {
            'project_name': self.current_session.metadata.project_name,
            'project_id': self.current_session.metadata.project_id,
            'created_date': self.current_session.metadata.created_date.isoformat(),
            'last_modified': self.current_session.metadata.last_modified.isoformat(),
            'description': self.current_session.metadata.description,
            'author': self.current_session.metadata.author,
            'overall_completion': self.get_completion_percentage(),
            'modules': {},
            'total_activities': len(self.current_session.activity_log)
        }

        # Add module summaries
        for module_name, status in self.current_session.module_status.items():
            summary['modules'][module_name] = {
                'completed': status.completed,
                'completion_percentage': status.completion_percentage,
                'last_updated': status.last_updated.isoformat() if status.last_updated else None
            }

        return summary

    def close_session(self, save: bool = True) -> None:
        """
        Close the current session.

        Args:
            save: Whether to save the session before closing
        """
        if self.current_session is None:
            logger.warning("No current session to close")
            return

        project_name = self.current_session.metadata.project_name

        if save:
            self.save_project()

        self.current_session = None
        self.current_project_path = None

        logger.info(f"Closed session for project: {project_name}")
