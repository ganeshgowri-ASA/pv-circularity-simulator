# PV Circularity Simulator - Architecture Documentation

## System Overview

The PV Circularity Simulator is built on a modular, session-based architecture using Streamlit as the web framework. The system follows a clean separation of concerns with distinct layers for presentation, business logic, data management, and utilities.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│              (Streamlit UI - modules/*.py)                   │
├─────────────────────────────────────────────────────────────┤
│                    Business Logic Layer                      │
│              (Core - core/*.py, utils/*.py)                  │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│            (Session State, JSON Files, Models)               │
├─────────────────────────────────────────────────────────────┤
│                   External Services Layer                    │
│         (Weather APIs, SCADA Systems, File I/O)             │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
pv_circularity_simulator/
├── docs/                          # Documentation
│   ├── MASTER_PROMPT.md          # Complete specification
│   ├── ARCHITECTURE.md           # This file
│   └── MODULE_SPECS.md           # Detailed module specs
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   │
│   ├── modules/                  # Feature modules (UI layer)
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   ├── material_selection.py
│   │   ├── module_design.py
│   │   ├── ctm_loss_analysis.py
│   │   ├── system_design.py
│   │   ├── eya_simulation.py
│   │   ├── performance_monitoring.py
│   │   ├── fault_diagnostics.py
│   │   ├── hya_simulation.py
│   │   ├── energy_forecasting.py
│   │   ├── revamp_repower.py
│   │   └── circularity_3r.py
│   │
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── session_manager.py   # State management
│   │   ├── data_models.py       # Pydantic models
│   │   └── config.py            # Configuration constants
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── charts.py             # Plotly/Matplotlib charts
│       ├── calculations.py       # PV calculations
│       └── validators.py         # Input validation
│
├── data/                         # Data files
│   ├── materials_db.json         # Material database
│   ├── cell_types.json           # Cell technology specs
│   └── standards.json            # Industry standards
│
├── tests/                        # Unit tests
│   └── __init__.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Core Components

### 1. Application Entry Point (`main.py`)

**Responsibilities**:
- Initialize Streamlit page configuration
- Create session manager instance
- Handle module navigation
- Load and render selected module

**Key Functions**:
```python
def main():
    # Page config
    st.set_page_config(**PAGE_CONFIG)

    # Session manager
    session = SessionManager()

    # Navigation
    selected_module = render_sidebar()

    # Load module
    load_module(selected_module, session)
```

### 2. Session Manager (`core/session_manager.py`)

**Responsibilities**:
- Manage Streamlit session state
- Persist data across modules
- Provide get/set interface
- Export/import session data

**Architecture Pattern**: Singleton-like access to session state

**Key Methods**:
```python
class SessionManager:
    def get(key: str, default: Any) -> Any
    def set(key: str, value: Any) -> None
    def update(data: Dict[str, Any]) -> None
    def clear() -> None
    def export_state() -> Dict
    def import_state(state: Dict) -> None
```

**State Structure**:
```python
{
    'project_name': str,
    'created_at': datetime,
    'material_data': dict,
    'module_design_data': dict,
    'ctm_losses': dict,
    'system_design_data': dict,
    'eya_results': dict,
    'performance_data': dict,
    'fault_data': dict,
    'hya_results': dict,
    'forecast_data': dict,
    'revamp_data': dict,
    'circularity_data': dict
}
```

### 3. Data Models (`core/data_models.py`)

**Responsibilities**:
- Define data structures using Pydantic
- Validate input data
- Provide type safety
- Enable serialization/deserialization

**Key Models**:
- `Material`: Material specifications
- `ModuleDesign`: PV module design
- `CTMLosses`: K-factor loss model
- `SystemDesign`: Complete system configuration
- `ProjectData`: Aggregated project data

**Example**:
```python
class ModuleDesign(BaseModel):
    name: str
    cell_type: CellType
    num_cells: int
    cell_efficiency: float = Field(gt=0, le=30)
    module_power: float
    voltage_voc: float
    # ... additional fields

    @validator('cell_efficiency')
    def validate_efficiency(cls, v):
        if v > 30:
            raise ValueError('Exceeds tech limits')
        return v
```

### 4. Configuration (`core/config.py`)

**Responsibilities**:
- Store application constants
- Define module metadata
- Maintain default values
- Provide configuration access

**Key Configurations**:
- `MODULES`: Module definitions and icons
- `CTM_LOSS_FACTORS`: K-factor specifications
- `PAGE_CONFIG`: Streamlit page settings
- `CHART_THEME`: Visualization styling
- `CIRCULARITY_METRICS`: Sustainability targets

## Module Architecture

Each feature module follows a consistent pattern:

### Standard Module Structure

```python
"""
Module Name
===========

Brief description of module purpose.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the module interface.

    Args:
        session: Session manager instance

    Features:
        - Feature 1
        - Feature 2
        - ...
    """
    # 1. Header
    st.header("Module Name")

    # 2. Load existing data
    existing_data = session.get('module_data', {})

    # 3. Input controls
    # ... Streamlit widgets

    # 4. Calculations/Processing
    # ... Business logic

    # 5. Display results
    # ... Charts, tables, metrics

    # 6. Save data
    if st.button("Save"):
        session.set('module_data', new_data)
```

### Module Communication

Modules communicate via session state:

```python
# Module A saves data
session.set('module_a_output', {
    'parameter1': value1,
    'parameter2': value2
})

# Module B reads data
module_a_data = session.get('module_a_output', {})
if module_a_data:
    # Use data from Module A
    process(module_a_data)
```

## Utility Layer

### Charts (`utils/charts.py`)

**Responsibilities**:
- Create Plotly visualizations
- Generate Matplotlib plots
- Provide consistent styling
- Support common chart types

**Key Functions**:
```python
def create_line_chart(data, x, y, title) -> go.Figure
def create_bar_chart(data, x, y, title) -> go.Figure
def create_pie_chart(labels, values, title) -> go.Figure
def create_heatmap(data, title) -> go.Figure
def create_waterfall_chart(categories, values) -> go.Figure
def create_sankey_diagram(source, target, value) -> go.Figure
def create_gauge_chart(value, max_value) -> go.Figure
def create_time_series_forecast(historical, forecast) -> go.Figure
```

### Calculations (`utils/calculations.py`)

**Responsibilities**:
- PV-specific calculations
- Performance metrics
- Financial analysis
- Statistical computations

**Key Functions**:
```python
def calculate_module_power(efficiency, cells, area, ctm) -> float
def calculate_ctm_ratio(k_factors: dict) -> float
def calculate_performance_ratio(actual, expected) -> float
def calculate_capacity_factor(energy, capacity, hours) -> float
def calculate_degradation_rate(pr_values, dates) -> tuple
def calculate_temperature_corrected_power(power, temp) -> float
def calculate_lcoe(capex, opex, energy, lifetime) -> float
def calculate_npv(capex, revenue, opex, lifetime) -> float
```

### Validators (`utils/validators.py`)

**Responsibilities**:
- Input validation
- Data quality checks
- Type verification
- Range validation

**Key Functions**:
```python
def validate_latitude(lat) -> Tuple[bool, str]
def validate_longitude(lon) -> Tuple[bool, str]
def validate_efficiency(eff) -> Tuple[bool, str]
def validate_module_design(design) -> Tuple[bool, List[str]]
def validate_system_design(design) -> Tuple[bool, List[str]]
def validate_financial_inputs(capex, opex, rate) -> Tuple[bool, List[str]]
```

## Data Flow Patterns

### Pattern 1: Sequential Dependency

```
Material Selection → Module Design → CTM Analysis → System Design
         ↓                ↓               ↓              ↓
   session_state    session_state   session_state  session_state
```

Each module reads from the previous and writes its output.

### Pattern 2: Parallel Processing

```
                    System Design
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
  Performance      Forecasting      Circularity
   Monitoring
```

Multiple modules can independently read the same upstream data.

### Pattern 3: Aggregation

```
Material + Module + System + ... → Circularity (3R)
                                        ↓
                              Comprehensive Analysis
```

Circularity module aggregates data from all other modules.

## State Management Strategy

### Initialization

```python
def _initialize_session_state(self):
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        # Set default values
        st.session_state.current_module = 'dashboard'
        st.session_state.project_name = ''
        # Initialize module-specific states
        for module in MODULES:
            st.session_state[f'{module}_data'] = {}
```

### State Persistence

- **In-memory**: Streamlit session state (volatile)
- **Export/Import**: JSON serialization for persistence
- **Future**: Database backend for multi-user support

## Error Handling

### Validation Strategy

```python
# 1. Input validation
is_valid, errors = validate_module_design(design_data)
if not is_valid:
    for error in errors:
        st.error(error)
    return

# 2. Try-except for calculations
try:
    result = calculate_something(input_data)
except Exception as e:
    st.error(f"Calculation error: {str(e)}")
    return

# 3. Data availability checks
if not session.get('required_data'):
    st.warning("Please complete previous module first")
    return
```

### User Feedback

- **Success**: `st.success()` for completed operations
- **Warnings**: `st.warning()` for non-critical issues
- **Errors**: `st.error()` for failures
- **Info**: `st.info()` for guidance

## Performance Optimization

### Streamlit Caching

```python
@st.cache_data
def load_material_database():
    with open(MATERIALS_DB_PATH) as f:
        return json.load(f)

@st.cache_data
def calculate_annual_production(system_data, weather_data):
    # Expensive calculation
    return results
```

### Lazy Loading

- Load data only when module is accessed
- Cache frequently used calculations
- Minimize session state footprint

## Testing Strategy

### Unit Tests

```python
# tests/test_calculations.py
def test_ctm_ratio_calculation():
    k_factors = {'k1': 0.98, 'k2': 0.97, ...}
    ratio = calculate_ctm_ratio(k_factors)
    assert 0.88 <= ratio <= 0.97

def test_performance_ratio():
    pr = calculate_performance_ratio(
        actual=9500,
        expected=10000
    )
    assert pr == 95.0
```

### Integration Tests

```python
# tests/test_module_flow.py
def test_material_to_module_flow():
    session = SessionManager()

    # Material selection
    material_selection.render(session)
    assert 'material_data' in session.export_state()

    # Module design
    module_design.render(session)
    assert 'module_design_data' in session.export_state()
```

## Security Considerations

### Input Sanitization

- Validate all user inputs
- Sanitize file names
- Check file types before processing
- Limit file sizes

### Data Privacy

- No sensitive data stored in session
- Export files are user-controlled
- No external data transmission (in current version)

## Deployment Architecture

### Local Deployment

```bash
streamlit run pv_circularity_simulator/src/main.py
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pv_circularity_simulator /app
WORKDIR /app/src
CMD ["streamlit", "run", "main.py"]
```

### Cloud Deployment (Future)

- Streamlit Cloud
- AWS/Azure/GCP
- Multi-user authentication
- Database backend for persistence

## Extension Points

### Adding New Modules

1. Create module file in `src/modules/`
2. Implement `render(session)` function
3. Add module to `MODULES` in `config.py`
4. Update `main.py` navigation
5. Document in `MODULE_SPECS.md`

### Adding New Calculations

1. Implement function in `utils/calculations.py`
2. Add unit tests
3. Document parameters and return values
4. Update relevant modules

### Custom Data Sources

1. Implement data loader in `utils/`
2. Add configuration in `config.py`
3. Integrate into relevant modules
4. Validate data format

## Monitoring and Logging

### Current Implementation

- Streamlit built-in logging
- User-facing messages via st.success/error/warning

### Future Enhancements

- Structured logging (loguru, logging module)
- Performance metrics tracking
- Usage analytics
- Error reporting

---

**Version**: 1.0
**Last Updated**: 2024
**Next Review**: Q2 2024
