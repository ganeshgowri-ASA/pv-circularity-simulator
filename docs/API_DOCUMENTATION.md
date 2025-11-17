# PV Circularity Simulator - API Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Data Models](#data-models)
3. [State Manager API](#state-manager-api)
4. [Planning UI Functions](#planning-ui-functions)
5. [File Structure](#file-structure)
6. [Extension Guide](#extension-guide)

## Architecture Overview

### Technology Stack

- **Framework**: Streamlit 1.28+
- **Visualization**: Plotly 5.18+
- **Data Processing**: Pandas 2.0+
- **Data Validation**: Pydantic 2.0+
- **File Handling**: PyPDF2, python-docx, openpyxl

### Application Structure

```
┌─────────────────────────────────────────┐
│         Streamlit Frontend              │
│  (app.py + src/ui/planning.py)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      State Manager (Session State)      │
│    (src/core/state_manager.py)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Data Models & Persistence        │
│      (src/data/models.py + JSON)        │
└─────────────────────────────────────────┘
```

### Design Patterns

- **Model-View Pattern**: Clean separation between data models and UI
- **State Management**: Centralized state with StateManager class
- **Data Persistence**: JSON-based storage with auto-save
- **Functional UI**: Pure functions for UI components

## Data Models

### Project

```python
@dataclass
class Project:
    """Represents a PV system installation/simulation project."""
    id: str
    name: str
    description: str
    status: ProjectStatus  # Enum: DESIGN, ENGINEERING, PLANNING, etc.
    capacity_kwp: float
    location: str
    created_date: datetime
    updated_date: datetime
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    owner: str
    budget: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

**Enums:**
```python
class ProjectStatus(Enum):
    DESIGN = "Design"
    ENGINEERING = "Engineering"
    PLANNING = "Planning"
    IMPLEMENTATION = "Implementation"
    MONITORING = "Monitoring"
    EOL = "End of Life"
```

### Resource

```python
@dataclass
class Resource:
    """Represents materials, components, labor, and financial resources."""
    id: str
    project_id: str
    name: str
    resource_type: ResourceType  # Enum
    quantity: float
    unit: str
    unit_cost: float
    total_cost: float
    supplier: str
    availability_start: Optional[datetime]
    availability_end: Optional[datetime]
    allocated: bool
    constraints: List[str]
    metadata: Dict[str, Any]

    def calculate_total_cost(self) -> float:
        """Calculate and update total cost."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

**Enums:**
```python
class ResourceType(Enum):
    MODULE = "PV Module"
    INVERTER = "Inverter"
    CABLE = "Cable"
    MOUNTING = "Mounting System"
    LABOR = "Labor"
    CAPITAL = "Capital"
    EQUIPMENT = "Equipment"
    OTHER = "Other"
```

### Contract

```python
@dataclass
class Contract:
    """Represents supply agreements, labor contracts, and service agreements."""
    id: str
    project_id: str
    contract_type: ContractType  # Enum
    vendor: str
    title: str
    description: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    value: float
    currency: str
    terms: str
    status: ContractStatus  # Enum
    payment_schedule: List[Dict[str, Any]]
    deliverables: List[str]
    template_file: Optional[str]
    signed_file: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

**Enums:**
```python
class ContractType(Enum):
    SUPPLY = "Supply Agreement"
    LABOR = "Labor Contract"
    SERVICE = "Service Agreement"
    MAINTENANCE = "Maintenance Contract"
    CONSULTING = "Consulting Agreement"

class ContractStatus(Enum):
    DRAFT = "Draft"
    PENDING = "Pending Approval"
    ACTIVE = "Active"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"
```

### Portfolio

```python
@dataclass
class Portfolio:
    """Collection of related PV projects for portfolio management."""
    id: str
    name: str
    description: str
    owner: str
    project_ids: List[str]
    created_date: datetime
    total_capacity_kwp: float
    total_budget: float
    roi_target: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

### Timeline

```python
@dataclass
class Timeline:
    """Project timeline and milestone tracking."""
    id: str
    project_id: str
    milestones: List[Dict[str, Any]]
    phases: List[Dict[str, Any]]
    critical_path: List[str]
    dependencies: Dict[str, List[str]]

    def add_milestone(self, name: str, date: datetime, description: str = "") -> None:
        """Add a milestone to the timeline."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

## State Manager API

### StateManager Class

Central state management for the application.

#### Initialization

```python
class StateManager:
    """Centralized state management for Streamlit application."""

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

        Sets up all necessary session state variables including:
        - Data stores (projects, resources, contracts, portfolios, timelines)
        - UI state (current_project_id, selected_tab, etc.)
        - Filters (filter_status, filter_date_range)
        - Wizard state (wizard_step, wizard_data)
        """
```

**Usage:**
```python
from src.core.state_manager import StateManager

# Initialize at app startup
StateManager.initialize()
```

#### Project Management

```python
@staticmethod
def add_project(project: Project) -> None:
    """
    Add a new project to the state.

    Args:
        project: Project instance to add
    """

@staticmethod
def update_project(project_id: str, updates: Dict[str, Any]) -> None:
    """
    Update an existing project.

    Args:
        project_id: Project identifier
        updates: Dictionary of fields to update
    """

@staticmethod
def delete_project(project_id: str) -> None:
    """
    Delete a project and its associated resources.

    Args:
        project_id: Project identifier
    """

@staticmethod
def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a project by ID.

    Args:
        project_id: Project identifier

    Returns:
        Project dictionary or None if not found
    """

@staticmethod
def get_all_projects() -> List[Dict[str, Any]]:
    """
    Retrieve all projects.

    Returns:
        List of project dictionaries
    """
```

**Example:**
```python
from src.data.models import Project, ProjectStatus

# Create new project
project = Project(
    name="Solar Farm Alpha",
    description="100 kWp rooftop installation",
    status=ProjectStatus.DESIGN,
    capacity_kwp=100.0,
    location="Phoenix, AZ",
    owner="John Doe",
    budget=150000.0
)

# Add to state
StateManager.add_project(project)

# Retrieve project
retrieved = StateManager.get_project(project.id)

# Update project
StateManager.update_project(project.id, {
    "status": ProjectStatus.ENGINEERING.value,
    "budget": 160000.0
})

# Get all projects
all_projects = StateManager.get_all_projects()
```

#### Resource Management

```python
@staticmethod
def add_resource(resource: Resource) -> None:
    """Add a new resource to the state."""

@staticmethod
def update_resource(resource_id: str, updates: Dict[str, Any]) -> None:
    """Update an existing resource."""

@staticmethod
def delete_resource(resource_id: str) -> None:
    """Delete a resource."""

@staticmethod
def get_project_resources(project_id: str) -> List[Dict[str, Any]]:
    """Retrieve all resources for a project."""
```

**Example:**
```python
from src.data.models import Resource, ResourceType

# Create resource
resource = Resource(
    project_id=project.id,
    name="Monocrystalline PV Module 400W",
    resource_type=ResourceType.MODULE,
    quantity=250,
    unit="pieces",
    unit_cost=200.0,
    supplier="SolarTech Inc.",
    allocated=True
)

resource.calculate_total_cost()  # Updates total_cost
StateManager.add_resource(resource)

# Get project resources
resources = StateManager.get_project_resources(project.id)
```

#### Contract Management

```python
@staticmethod
def add_contract(contract: Contract) -> None:
    """Add a new contract to the state."""

@staticmethod
def update_contract(contract_id: str, updates: Dict[str, Any]) -> None:
    """Update an existing contract."""

@staticmethod
def delete_contract(contract_id: str) -> None:
    """Delete a contract."""

@staticmethod
def get_project_contracts(project_id: str) -> List[Dict[str, Any]]:
    """Retrieve all contracts for a project."""
```

**Example:**
```python
from src.data.models import Contract, ContractType, ContractStatus

# Create contract
contract = Contract(
    project_id=project.id,
    title="PV Module Supply Agreement",
    vendor="SolarTech Inc.",
    contract_type=ContractType.SUPPLY,
    value=50000.0,
    currency="USD",
    status=ContractStatus.ACTIVE,
    deliverables=["250 PV Modules", "Installation Support"]
)

StateManager.add_contract(contract)

# Get project contracts
contracts = StateManager.get_project_contracts(project.id)
```

#### Data Persistence

```python
@classmethod
def save_all(cls) -> None:
    """Save all session state data to persistent storage."""

@staticmethod
def _load_data(file_path: Path, default: Any) -> Any:
    """Load data from JSON file."""

@staticmethod
def _save_data(file_path: Path, data: Any) -> None:
    """Save data to JSON file."""
```

**Usage:**
```python
# Save is automatic on add/update/delete, but can be called manually
StateManager.save_all()
```

## Planning UI Functions

### project_wizard()

```python
def project_wizard() -> None:
    """
    Interactive multi-step project creation wizard.

    Provides a guided workflow for creating new PV projects with:
    - Basic project information (name, description, owner)
    - Technical specifications (capacity, location)
    - Budget and timeline planning
    - Initial resource estimation

    Side Effects:
        - Creates new Project instance in session state
        - Initializes associated Timeline
        - Saves data to persistent storage
        - Updates wizard state in session_state

    UI Components:
        - Multi-step form with progress indicator
        - Input validation
        - Navigation buttons (Next, Back, Cancel)
        - Summary review page

    Session State Variables:
        - wizard_step: Current step (0-3)
        - wizard_data: Accumulated form data
        - current_project_id: ID of created project
    """
```

**Usage:**
```python
from src.ui.planning import project_wizard

# In Streamlit app
project_wizard()
```

### timeline_planner()

```python
def timeline_planner(project_id: Optional[str] = None) -> None:
    """
    Interactive timeline and milestone planning interface.

    Provides comprehensive project timeline management with:
    - Gantt chart visualization of project phases
    - Interactive milestone creation with date pickers
    - Phase duration planning and tracking
    - Critical path identification
    - Dependency management between tasks

    Args:
        project_id: Optional project identifier. If None, prompts for selection.

    Side Effects:
        - Creates/updates Timeline instances in session state
        - Renders interactive Plotly Gantt charts
        - Saves timeline data to persistent storage

    UI Components:
        - Project selector dropdown
        - Tabbed interface (Gantt Chart, Milestones, Phases)
        - Date pickers for milestone/phase dates
        - Expandable milestone/phase cards
        - Completion tracking checkboxes

    Visualizations:
        - Plotly timeline chart (Gantt)
        - Color-coded by type (Project, Phase, Milestone)
    """
```

**Usage:**
```python
from src.ui.planning import timeline_planner

# For specific project
timeline_planner(project_id="project-123")

# With project selection
timeline_planner()
```

### resource_allocation_dashboard()

```python
def resource_allocation_dashboard(project_id: Optional[str] = None) -> None:
    """
    Interactive resource allocation and management dashboard.

    Provides comprehensive resource planning and tracking with:
    - Resource inventory management (modules, inverters, labor, etc.)
    - Allocation status tracking and visualization
    - Cost analysis and budget tracking
    - Supplier management
    - Availability timeline visualization
    - Resource utilization metrics

    Args:
        project_id: Optional project identifier. If None, shows all resources.

    Side Effects:
        - Creates/updates Resource instances in session state
        - Renders resource allocation charts and tables
        - Saves resource data to persistent storage
        - Updates project cost calculations

    UI Components:
        - Metric cards (total, allocated, cost, budget usage)
        - Tabbed interface (List, Add, Analytics)
        - Interactive dataframe
        - Resource detail expanders
        - Add resource form

    Visualizations:
        - Pie chart: Cost breakdown by type
        - Bar chart: Allocation status
        - Bar chart: Top suppliers by cost
    """
```

**Usage:**
```python
from src.ui.planning import resource_allocation_dashboard

# For specific project
resource_allocation_dashboard(project_id="project-123")

# View all resources
resource_allocation_dashboard()
```

### contract_templates()

```python
def contract_templates(project_id: Optional[str] = None) -> None:
    """
    Contract template management with file upload functionality.

    Provides comprehensive contract lifecycle management with:
    - Contract template library (upload/download)
    - Contract creation from templates
    - File upload for signed contracts
    - Contract status tracking
    - Payment schedule management
    - Deliverable tracking
    - Vendor/contractor database

    Args:
        project_id: Optional project identifier. If None, shows all contracts.

    Side Effects:
        - Creates/updates Contract instances in session state
        - Saves uploaded files to uploads/ directory
        - Renders contract management interface
        - Saves contract data to persistent storage

    UI Components:
        - Metric cards (total, active, value, pending)
        - Tabbed interface (List, Create, Upload, Library)
        - Contract detail expanders
        - File upload widgets
        - Payment schedule builder

    File Handling:
        - Supported formats: PDF, DOCX, TXT
        - Storage: uploads/contracts/
        - Template storage: uploads/contracts/templates/
    """
```

**Usage:**
```python
from src.ui.planning import contract_templates

# For specific project
contract_templates(project_id="project-123")

# View all contracts
contract_templates()
```

## File Structure

### Project Layout

```
pv-circularity-simulator/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
├── .streamlit/
│   └── config.toml                # Streamlit config
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── models.py              # Data models
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── state_manager.py       # State management
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── planning.py            # Planning functions
│   │   ├── components/            # Reusable components
│   │   └── pages/                 # Additional pages
│   │
│   └── utils/
│       └── __init__.py            # Utility functions
│
├── data/                          # Data persistence (created at runtime)
│   ├── projects.json
│   ├── resources.json
│   ├── contracts.json
│   ├── portfolios.json
│   └── timelines.json
│
├── uploads/                       # File uploads (created at runtime)
│   └── contracts/
│       ├── templates/
│       └── signed/
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_state_manager.py
│   └── test_ui.py
│
└── docs/                          # Documentation
    ├── USER_GUIDE.md
    ├── API_DOCUMENTATION.md
    └── ARCHITECTURE.md
```

### Import Paths

```python
# Data models
from src.data.models import (
    Project, Resource, Contract, Portfolio, Timeline,
    ProjectStatus, ResourceType, ContractType, ContractStatus
)

# State management
from src.core.state_manager import StateManager

# Planning UI
from src.ui.planning import (
    project_wizard,
    timeline_planner,
    resource_allocation_dashboard,
    contract_templates
)
```

## Extension Guide

### Adding a New Data Model

1. **Define the model** in `src/data/models.py`:

```python
@dataclass
class NewModel:
    """Model description."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    # ... other fields

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            # ... other fields
        }
```

2. **Add StateManager methods** in `src/core/state_manager.py`:

```python
@staticmethod
def add_new_model(model: NewModel) -> None:
    """Add new model to state."""
    st.session_state.new_models[model.id] = model.to_dict()
    StateManager.save_all()

@staticmethod
def get_new_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve model by ID."""
    return st.session_state.new_models.get(model_id)
```

3. **Create UI function** in `src/ui/` or new module:

```python
def new_model_manager() -> None:
    """UI for managing new model."""
    st.header("New Model Manager")
    # ... implement UI
```

### Adding a New Page

1. **Create page file** in `src/ui/pages/`:

```python
# src/ui/pages/07_NewPage.py
import streamlit as st
from src.core.state_manager import StateManager

def render():
    """Render the new page."""
    st.title("New Page")
    # ... implement page
```

2. **Add to navigation** in `app.py`:

```python
# In render_sidebar()
page = st.radio(
    "Select Page",
    options=[
        # ... existing pages
        "🆕 New Page"
    ]
)

# In main()
elif selected_page == "🆕 New Page":
    from src.ui.pages import NewPage
    NewPage.render()
```

### Adding Custom Visualizations

Example: Add a custom chart type

```python
import plotly.graph_objects as go

def create_custom_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create a custom visualization.

    Args:
        data: DataFrame with required columns

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='lines+markers',
        name='Series 1'
    ))

    # Update layout
    fig.update_layout(
        title="Custom Chart",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        hovermode='x unified'
    )

    return fig

# Usage in UI
st.plotly_chart(create_custom_chart(df), use_container_width=True)
```

### Adding File Export

Example: Export project data to Excel

```python
import pandas as pd
from io import BytesIO

def export_project_to_excel(project_id: str) -> BytesIO:
    """
    Export project data to Excel file.

    Args:
        project_id: Project identifier

    Returns:
        BytesIO buffer with Excel file
    """
    project = StateManager.get_project(project_id)
    resources = StateManager.get_project_resources(project_id)
    contracts = StateManager.get_project_contracts(project_id)

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Project sheet
        pd.DataFrame([project]).to_excel(writer, sheet_name='Project', index=False)

        # Resources sheet
        pd.DataFrame(resources).to_excel(writer, sheet_name='Resources', index=False)

        # Contracts sheet
        pd.DataFrame(contracts).to_excel(writer, sheet_name='Contracts', index=False)

    output.seek(0)
    return output

# Usage in UI
if st.button("Export to Excel"):
    excel_file = export_project_to_excel(project_id)
    st.download_button(
        label="Download Excel",
        data=excel_file,
        file_name=f"project_{project_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
```

### Testing

Example test structure:

```python
# tests/test_models.py
import pytest
from src.data.models import Project, ProjectStatus

def test_project_creation():
    """Test project creation."""
    project = Project(
        name="Test Project",
        status=ProjectStatus.DESIGN,
        capacity_kwp=100.0,
        location="Test Location",
        owner="Test Owner",
        budget=100000.0
    )

    assert project.name == "Test Project"
    assert project.status == ProjectStatus.DESIGN
    assert project.capacity_kwp == 100.0

def test_project_to_dict():
    """Test project serialization."""
    project = Project(name="Test", owner="Owner", location="Location", budget=0)
    data = project.to_dict()

    assert isinstance(data, dict)
    assert data["name"] == "Test"
    assert "id" in data
```

## Best Practices

### State Management

1. **Always use StateManager** for data operations
2. **Call save_all()** after batch updates
3. **Initialize state** at app startup
4. **Use session_state** for UI state only

### UI Development

1. **Use forms** for multi-field inputs
2. **Validate inputs** before processing
3. **Provide feedback** with success/error messages
4. **Use expanders** for detailed information
5. **Cache expensive operations** with `@st.cache_data`

### Data Persistence

1. **Validate before saving** to JSON
2. **Handle file paths** consistently
3. **Clean up old files** periodically
4. **Backup data** regularly
5. **Use atomic writes** for critical data

### Error Handling

```python
try:
    # Operation
    StateManager.add_project(project)
    st.success("Project added successfully!")
except Exception as e:
    st.error(f"Failed to add project: {str(e)}")
    logger.exception("Project addition failed")
```

---

**For more information**, see the [User Guide](USER_GUIDE.md) or the [README](../README.md).
