# Core Infrastructure Documentation

## Overview

This document describes the core infrastructure components for the PV Circularity Simulator, including the SessionManager and comprehensive Pydantic data models.

## Components

### 1. Session Manager (`src/core/session_manager.py`)

The SessionManager class provides complete project lifecycle management:

#### Key Features

- **Project Management**: Create, save, load, and list projects
- **11 Module Tracking**: Tracks completion status for all simulation modules:
  1. Material Selection
  2. Cell Design
  3. Module Engineering
  4. Cutting Pattern
  5. System Design
  6. Performance Simulation
  7. Financial Analysis
  8. Reliability Testing
  9. Circularity Assessment
  10. SCAPS Integration
  11. Energy Forecasting

- **Activity Logging**: Comprehensive activity tracking with timestamps
- **Progress Tracking**: Real-time completion percentage calculation
- **State Persistence**: JSON-based project file storage

#### Usage Example

```python
from src.core import SessionManager, SimulationModule

# Initialize manager
manager = SessionManager()

# Create new project
manager.create_new_project(
    project_name="My PV System",
    description="Design of 100kW system",
    author="John Doe"
)

# Update module status
manager.update_module_status(
    module=SimulationModule.CELL_DESIGN.value,
    completion_percentage=100,
    completed=True,
    data={'cell_efficiency': 0.225}
)

# Get overall progress
progress = manager.get_completion_percentage()
print(f"Project {progress:.1f}% complete")

# Save project
manager.save_project()
```

### 2. Pydantic Data Models (`src/core/data_models.py`)

Comprehensive data models with full validation:

#### Material Models

- **MaterialProperties**: Physical/chemical properties with validation
- **Material**: Complete material specification with cost/environmental data
- **CircularityMetrics**: 3R metrics (Recyclability, Reusability, Repairability)

#### Cell Models

- **TemperatureCoefficients**: Temperature-dependent performance
- **Cell**: Complete cell specification with electrical parameters
- Validates electrical consistency (P=V×I, Vmp<Voc, etc.)

#### Module Models

- **CuttingPattern**: Cell cutting configurations (half-cut, etc.)
- **ModuleLayout**: Physical cell arrangement with bypass diodes
- **Module**: Complete module specification
- Validates CTM (Cell-to-Module) ratio and power consistency

#### System Models

- **Location**: Geographic data with timezone
- **PVSystem**: Complete system configuration
- Validates DC/AC ratio, capacity calculations

#### Performance & Financial Models

- **PerformanceData**: Real-time monitoring data with validation
- **FinancialModel**: Complete financial analysis
  - LCOE calculation
  - NPV calculation
  - Payback period
  - Includes degradation and escalation

## Model Features

### Comprehensive Validation

All models include:
- Field constraints (min/max values, ranges)
- Cross-field validation
- Consistency checks
- Domain-specific rules

### Example: Cell Validation

```python
from src.core import create_default_monocrystalline_cell

cell = create_default_monocrystalline_cell()

# Automatic validation ensures:
# - Vmp < Voc
# - Imp < Isc
# - P = Vmp × Imp
# - Fill Factor = P / (Voc × Isc)
# - All values in valid ranges
```

### Production-Ready Features

- **Type Safety**: Full type hints throughout
- **Documentation**: Comprehensive docstrings
- **Validators**: Field and model-level validation
- **Calculations**: Automatic derived properties
- **Serialization**: JSON-compatible via Pydantic

## Testing

Comprehensive test suite with 55 tests covering:

- All data models
- All validators
- SessionManager functionality
- Integration workflows
- Edge cases and error handling

Run tests:
```bash
pytest tests/test_core.py -v
```

## File Structure

```
pv-circularity-simulator/
├── src/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── data_models.py      # All Pydantic models
│       └── session_manager.py  # SessionManager class
├── tests/
│   ├── __init__.py
│   └── test_core.py           # Comprehensive tests
├── requirements.txt
└── example_usage.py           # Complete example
```

## Dependencies

- **pydantic >= 2.0.0**: Data validation and serialization
- **pytest >= 7.0.0**: Testing framework

## Data Model Hierarchy

```
Material Models
├── MaterialProperties (physical/chemical)
├── Material (bill of materials)
└── CircularityMetrics (3R metrics)

Cell Models
├── TemperatureCoefficients
└── Cell (electrical parameters)

Module Models
├── CuttingPattern (half-cut, etc.)
├── ModuleLayout (physical arrangement)
└── Module (complete module spec)
    ├── uses Cell
    ├── uses ModuleLayout
    └── uses List[Material]

System Models
├── Location (geographic)
└── PVSystem (complete system)
    ├── uses Location
    └── uses List[Module]

Analysis Models
├── PerformanceData (monitoring)
└── FinancialModel (economics)
    ├── LCOE calculation
    ├── NPV calculation
    └── Payback period
```

## Session State Structure

```python
SessionState
├── metadata: ProjectMetadata
│   ├── project_name
│   ├── project_id
│   ├── created_date
│   ├── last_modified
│   ├── description
│   ├── version
│   └── author
├── module_status: Dict[str, ModuleCompletionStatus]
│   ├── [11 modules with completion tracking]
│   └── each with: completed, percentage, last_updated, data
├── activity_log: List[ActivityEntry]
│   └── timestamped action history
└── custom_data: Dict[str, Any]
    └── user-defined project data
```

## Key Design Decisions

1. **Pydantic V2**: Using latest Pydantic for best performance and features
2. **Comprehensive Validation**: All models validate consistency
3. **Type Safety**: Full type hints for IDE support
4. **JSON Storage**: Human-readable project files
5. **Modular Design**: Each model is independent and composable
6. **Production Ready**: No placeholders, all features complete

## Next Steps

This core infrastructure enables:
- CTM loss analysis module
- SCAPS integration module
- Reliability testing module
- Performance simulation module
- Circularity assessment module
- Energy forecasting module

All these modules can build on the validated data models and session management.
