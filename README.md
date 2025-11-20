# ☀️ PV Circularity Simulator

End-to-end photovoltaic lifecycle simulation platform: **Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R)**.

Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling with comprehensive planning and portfolio management capabilities.

## 🎯 Features

### 🧙 **Planning UI & Portfolio Management** (NEW)

This release introduces a completely revamped planning interface with four core components:

#### 1. **Project Wizard** (`project_wizard()`)
Interactive multi-step project creation wizard with:
- Guided 4-step workflow (Basic Info → Technical Details → Timeline & Budget → Review)
- Streamlit forms with validation
- Technical specifications input (capacity, module type, inverter, mounting)
- Budget and timeline planning
- Milestone initialization
- Production-ready with full docstrings

#### 2. **Timeline Planner** (`timeline_planner()`)
Comprehensive timeline and milestone management:
- Interactive Gantt chart visualization using Plotly
- Date pickers for milestone creation
- Project phase management
- Milestone tracking with completion status
- Phase duration planning
- Visual timeline representation

#### 3. **Resource Allocation Dashboard** (`resource_allocation_dashboard()`)
Intelligent resource planning and tracking:
- Resource inventory management (modules, inverters, labor, capital)
- Interactive data editor for resource allocation
- Cost analysis and budget tracking
- Supplier management
- Allocation status visualization
- Resource utilization metrics with Plotly charts
- Availability timeline management

#### 4. **Contract Templates** (`contract_templates()`)
Complete contract lifecycle management:
- Contract template library with file upload (PDF, DOCX, TXT)
- Contract creation from templates
- File upload for signed contracts
- Contract status tracking (Draft, Pending, Active, Completed, Cancelled)
- Payment schedule management
- Deliverable tracking
- Vendor/contractor database

### 📊 Portfolio Dashboard
- Multi-project overview
- Budget tracking across portfolio
- Performance metrics
- Status distribution visualization
- Quick stats and recent activity

## 🏗️ Architecture

```
pv-circularity-simulator/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Production dependencies
├──
├── src/
│   ├── data/
│   │   └── models.py              # Core data models (Project, Resource, Contract, Portfolio, Timeline)
│   │
│   ├── core/
│   │   └── state_manager.py       # Centralized state management with persistence
│   │
│   └── ui/
│       ├── planning.py            # Planning UI components (4 core functions)
│       ├── components/            # Reusable UI components
│       └── pages/                 # Additional page modules
│
├── .streamlit/
│   └── config.toml                # Streamlit configuration
│
├── data/                          # Data persistence directory (auto-created)
├── uploads/                       # File upload storage (auto-created)
├── tests/                         # Test suite
└── docs/                          # Documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First Steps

1. **Create a Project**: Navigate to "🧙 Project Wizard" to create your first PV project
2. **Plan Timeline**: Use "📅 Timeline Planner" to add milestones and phases
3. **Allocate Resources**: Go to "📦 Resource Allocation" to add and track resources
4. **Manage Contracts**: Visit "📄 Contract Management" to upload templates and create contracts

## 📖 Documentation

### Core Functions

#### `project_wizard()`
```python
def project_wizard() -> None:
    """
    Interactive multi-step project creation wizard.

    Provides a guided workflow for creating new PV projects with:
    - Basic project information (name, description, owner)
    - Technical specifications (capacity, location)
    - Budget and timeline planning
    - Initial resource estimation
    """
```

**Usage:**
- Navigate to Project Wizard page in the sidebar
- Follow the 4-step guided process
- Review and create project
- Project is saved to persistent storage

#### `timeline_planner(project_id: Optional[str] = None)`
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
    """
```

**Usage:**
- Select a project from the dropdown
- View Gantt chart visualization
- Add milestones with target dates
- Create project phases with start/end dates
- Track milestone completion

#### `resource_allocation_dashboard(project_id: Optional[str] = None)`
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
    """
```

**Usage:**
- Filter by project or view all resources
- View resource inventory and metrics
- Add new resources with detailed specifications
- Track allocation status
- View analytics and cost breakdowns

#### `contract_templates(project_id: Optional[str] = None)`
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
    """
```

**Usage:**
- Browse contract template library
- Upload new templates (PDF, DOCX, TXT)
- Create contracts from templates
- Upload signed contract files
- Track contract status and deliverables
- Manage payment schedules

### Data Models

#### Project
Core project entity with:
- Basic info (name, description, owner)
- Technical specs (capacity, location)
- Timeline (start/end dates)
- Budget tracking
- Status lifecycle

#### Resource
Resource management with:
- Type classification (Module, Inverter, Cable, Labor, Capital, etc.)
- Quantity and cost tracking
- Supplier information
- Availability windows
- Allocation status
- Constraints tracking

#### Contract
Contract lifecycle with:
- Type classification (Supply, Labor, Service, Maintenance, Consulting)
- Vendor management
- Value and currency tracking
- Status tracking (Draft, Pending, Active, Completed, Cancelled)
- Payment schedules
- Deliverables list
- File attachments

#### Timeline
Project timeline with:
- Milestones (name, date, description, completion status)
- Phases (name, start/end dates, description)
- Critical path tracking
- Dependency management

#### Portfolio
Portfolio aggregation with:
- Project collection
- Total capacity tracking
- Total budget aggregation
- ROI targets

### State Management

The application uses `StateManager` for centralized state management:

- **Persistent Storage**: Data saved to JSON files in `data/` directory
- **Session State**: Streamlit session state for UI state
- **CRUD Operations**: Full create, read, update, delete for all entities
- **Relationships**: Automatic handling of project-resource-contract relationships

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
```

### Data Persistence

Data is automatically saved to:
- `data/projects.json` - Project data
- `data/resources.json` - Resource data
- `data/contracts.json` - Contract data
- `data/portfolios.json` - Portfolio data
- `data/timelines.json` - Timeline data

File uploads are saved to:
- `uploads/contracts/` - Signed contract files
- `uploads/contracts/templates/` - Contract templates

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📋 Requirements

### Core Dependencies
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `plotly>=5.18.0` - Interactive visualizations
- `python-dateutil>=2.8.2` - Date utilities
- `pydantic>=2.0.0` - Data validation

### File Handling
- `openpyxl>=3.1.0` - Excel support
- `python-docx>=1.0.0` - Word document support
- `PyPDF2>=3.0.0` - PDF support

See `requirements.txt` for complete list.

## 🎨 UI Components

### Interactive Forms
- Multi-step wizards with progress indicators
- Form validation and error handling
- Date pickers for timeline planning
- File upload widgets
- Number inputs with validation

### Visualizations
- Gantt charts (Plotly timeline)
- Pie charts (cost distribution)
- Bar charts (allocation status, supplier costs)
- Metric cards (KPIs)
- Progress indicators

### Data Tables
- Interactive dataframes
- Sortable and filterable tables
- Expandable row details
- Inline editing capabilities

## 🔐 Production Ready

### Features
- ✅ Full docstrings on all functions and classes
- ✅ Type hints throughout codebase
- ✅ Error handling and validation
- ✅ Data persistence
- ✅ File upload security
- ✅ XSRF protection enabled
- ✅ Comprehensive logging
- ✅ Modular architecture
- ✅ Clean separation of concerns
- ✅ Reusable components

### Best Practices
- **Data Validation**: Pydantic models for data validation
- **State Management**: Centralized state with StateManager
- **Error Handling**: Try-except blocks with user-friendly messages
- **Code Organization**: Clean module structure
- **Documentation**: Comprehensive docstrings and comments
- **Type Safety**: Type hints for better IDE support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

See LICENSE file for details.

## 🎓 Version History

### v1.0.0 (Current)
- ✨ Complete Planning UI revamp
- ✨ Project Wizard with 4-step guided setup
- ✨ Timeline Planner with Gantt visualization
- ✨ Resource Allocation Dashboard
- ✨ Contract Template Management
- ✨ Portfolio Dashboard
- ✨ Full data persistence
- ✨ Production-ready with complete documentation

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for sustainable solar energy management**
