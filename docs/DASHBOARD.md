# Dashboard & Project Management UI

## Overview

The Dashboard module (`src/modules/dashboard.py`) provides a comprehensive web-based interface for managing and monitoring PV Circularity Simulator projects. Built with Streamlit, it offers real-time project tracking, simulation execution, and performance visualization.

## Features

### 1. Main Dashboard Interface

The `render_dashboard()` function provides:

- **Project Branding Header**: Professional gradient-styled header with project title
- **4-Column Metrics Display**:
  - Project Name
  - System Capacity (kWp)
  - Number of Modules
  - Overall Completion Percentage

### 2. Module Completion Tracking

Displays status for all 11 simulation modules:

1. 🔬 Cell Design & SCAPS Integration
2. 🔗 CTM Loss Analysis
3. ⚙️ Module Engineering
4. 🧪 Reliability Testing
5. 🏗️ System Planning
6. ☀️ Energy Yield Assessment (EYA)
7. 📈 Performance Monitoring
8. 📉 Degradation Modeling
9. ♻️ Circularity (3R) Assessment
10. 💰 Economic Analysis
11. 📄 Reporting

Each module shows:
- ✅ Completion indicator (completed)
- ⏳ Pending indicator (not started)
- Module description

### 3. Quick Actions

Three main action buttons:

- **🚀 Run Full Simulation**: Execute all 11 modules sequentially with progress tracking
- **📄 Generate Report**: Create comprehensive PDF report of simulation results
- **📊 Export Data**: Export all project data to Excel format

### 4. Recent Activity Log

Displays the last 5 activities with:
- Timestamp
- Activity description
- Activity type (success, info, warning, error)

### 5. Key Performance Metrics

Displayed when Energy Yield Assessment (EYA) is complete:

- **☀️ Annual Energy**: Total energy production (kWh/year)
- **📈 Performance Ratio**: System efficiency (%)
- **💰 LCOE**: Levelized Cost of Energy ($/kWh)
- **♻️ Circularity Score**: 3R assessment score (%)

## Helper Functions

### `calculate_completion()`

Computes overall project completion percentage based on module completion flags.

**Returns:**
- `Tuple[float, Dict[str, bool]]`: Completion percentage and module status dictionary

**Example:**
```python
completion_pct, modules_status = calculate_completion()
print(f"Project is {completion_pct:.1f}% complete")
```

### `run_full_simulation()`

Executes all simulation modules in sequence with real-time progress tracking.

**Returns:**
- `bool`: True if successful, False otherwise

**Features:**
- Progress bar visualization
- Status updates for each module
- Automatic activity logging
- Session state updates

### `generate_comprehensive_report()`

Generates a comprehensive PDF report (currently text format) of simulation results.

**Returns:**
- `Optional[BytesIO]`: PDF buffer for download, or None if failed

**Includes:**
- Project summary
- Module completion status
- Timestamp
- Performance metrics (if available)

### `export_all_data()`

Exports all simulation data to Excel format with multiple sheets.

**Returns:**
- `Optional[BytesIO]`: Excel buffer for download, or None if failed

**Sheets:**
1. **Module Status**: Completion status for all modules
2. **Project Summary**: Key project metrics and metadata

### `display_recent_activity(max_items=5)`

Displays recent activity log entries with timestamps and status icons.

**Parameters:**
- `max_items` (int): Maximum number of activities to display (default: 5)

### `log_activity(message, activity_type='info')`

Logs an activity to the session state for tracking.

**Parameters:**
- `message` (str): Activity description
- `activity_type` (str): Type - 'info', 'success', 'warning', or 'error'

**Example:**
```python
log_activity("Simulation completed", "success")
log_activity("Warning: Low performance ratio", "warning")
```

## Session State Variables

The dashboard uses the following session state variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_name` | str | "PV Circularity Project" | Project name |
| `system_capacity` | float | 1000.0 | System capacity in kWp |
| `num_modules` | int | 4500 | Number of PV modules |
| `completion_flags` | dict | All False | Module completion status |
| `activity_log` | list | [] | Activity history |
| `annual_energy` | float | 1500000 | Annual energy (kWh) |
| `performance_ratio` | float | 84.5 | Performance ratio (%) |
| `lcoe` | float | 0.045 | LCOE ($/kWh) |
| `circularity_score` | float | 72.3 | Circularity score (%) |

## Usage

### Basic Usage

```python
import streamlit as st
from modules.dashboard import render_dashboard

# Configure Streamlit
st.set_page_config(page_title="PV Simulator", layout="wide")

# Render dashboard
render_dashboard()
```

### Programmatic Module Completion

```python
import streamlit as st
from modules.dashboard import log_activity

# Mark a module as complete
if 'completion_flags' not in st.session_state:
    st.session_state.completion_flags = {}

st.session_state.completion_flags['cell_design'] = True
log_activity("Completed Cell Design module", "success")
```

### Custom Activity Logging

```python
from modules.dashboard import log_activity

# Log different types of activities
log_activity("Simulation started", "info")
log_activity("Module completed successfully", "success")
log_activity("Performance below threshold", "warning")
log_activity("Critical error in calculation", "error")
```

## Custom Styling

The dashboard includes extensive CSS customization:

- **Gradient Headers**: Professional purple gradient for main header
- **Metric Cards**: Clean white cards with colored left borders
- **Module Grid**: Responsive grid layout with hover effects
- **Activity Log**: Styled activity cards with timestamps
- **Progress Bars**: Custom gradient progress indicators

## Error Handling

All functions include comprehensive error handling:

```python
try:
    # Function logic
    result = perform_operation()
    return result
except Exception as e:
    st.error(f"Error message: {str(e)}")
    return None  # or appropriate default
```

## Performance Considerations

- Activity log limited to 100 entries to prevent memory issues
- Efficient session state management
- Progress tracking with optimized sleep intervals
- Lazy loading of metrics (only when modules complete)

## Future Enhancements

1. **Real PDF Generation**: Integrate ReportLab for actual PDF reports
2. **Advanced Visualizations**: Interactive charts with Plotly
3. **Module Details**: Expandable sections for each module
4. **User Settings**: Customizable dashboard layouts
5. **Data Persistence**: Save/load project states
6. **Multi-Project Support**: Switch between multiple projects
7. **Export Formats**: Additional export formats (JSON, CSV)

## Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
openpyxl>=3.1.0
```

## Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

## API Reference

### Main Function

#### `render_dashboard()`

Main entry point for rendering the complete dashboard interface.

**Usage:**
```python
from modules.dashboard import render_dashboard
render_dashboard()
```

### Utility Functions

All helper functions are documented with full docstrings in the source code. Import them as needed:

```python
from modules.dashboard import (
    calculate_completion,
    run_full_simulation,
    generate_comprehensive_report,
    export_all_data,
    display_recent_activity,
    log_activity
)
```

## Support

For issues or questions, please refer to the main project documentation or contact the development team.

---

**Version:** 1.0.0
**Last Updated:** 2024
**Author:** PV Circularity Simulator Team
