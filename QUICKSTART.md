# Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pv-circularity-simulator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

### Option 1: Using the Main App

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

### Option 2: Direct Module Import

Create a custom app:

```python
import streamlit as st
from src.modules.dashboard import render_dashboard

st.set_page_config(
    page_title="My PV Project",
    page_icon="🔆",
    layout="wide"
)

render_dashboard()
```

## Dashboard Features

### 1. View Project Overview
- See your project name, system capacity, and overall completion status
- Track progress across all 11 simulation modules

### 2. Run Full Simulation
- Click "🚀 Run Full Simulation" to execute all modules sequentially
- Watch real-time progress with the progress bar
- View completion status updates

### 3. Generate Reports
- Click "📄 Generate Report" to create a comprehensive project report
- Download the generated report file

### 4. Export Data
- Click "📊 Export Data" to export all simulation data to Excel
- Download the Excel file with multiple sheets

### 5. Monitor Activity
- View recent activities in the "Recent Activity" section
- Track all simulation steps and completions

### 6. View Performance Metrics
- After completing Energy Yield Assessment (EYA), view:
  - Annual Energy Production
  - Performance Ratio
  - Levelized Cost of Energy (LCOE)
  - Circularity Score

## Customization

### Update Project Settings

```python
import streamlit as st

# Set custom project values
st.session_state.project_name = "Solar Farm Alpha"
st.session_state.system_capacity = 5000.0  # kWp
st.session_state.num_modules = 15000
```

### Mark Modules as Complete

```python
import streamlit as st
from src.modules.dashboard import log_activity

# Complete a specific module
st.session_state.completion_flags['cell_design'] = True
log_activity("Completed Cell Design module", "success")
```

### Log Custom Activities

```python
from src.modules.dashboard import log_activity

log_activity("Started system optimization", "info")
log_activity("Optimization complete", "success")
log_activity("Performance warning detected", "warning")
```

## Modules Overview

The simulator tracks these 11 modules:

1. **Cell Design & SCAPS** - Solar cell design and SCAPS-1D integration
2. **CTM Loss Analysis** - Cell-to-Module loss characterization
3. **Module Engineering** - PV module design and optimization
4. **Reliability Testing** - Environmental and durability testing
5. **System Planning** - PV system configuration and sizing
6. **Energy Yield (EYA)** - Energy production forecasting
7. **Performance Monitoring** - Real-time performance analytics
8. **Degradation Modeling** - Long-term degradation analysis
9. **Circularity (3R)** - Reduce, Reuse, Recycle assessment
10. **Economic Analysis** - LCOE and financial modeling
11. **Reporting** - Documentation and report generation

## Troubleshooting

### Port Already in Use
If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

### Import Errors
Make sure you're in the project root directory and have installed all dependencies:
```bash
pip install -r requirements.txt
```

### Session State Issues
Clear the Streamlit cache:
- Press 'C' in the terminal running Streamlit
- Or add `st.cache_data.clear()` in your code

## Next Steps

- Explore the detailed documentation in `docs/DASHBOARD.md`
- Integrate with other simulation modules
- Customize the UI styling in `src/modules/dashboard.py`

## Support

For detailed API documentation, see `docs/DASHBOARD.md`.

---

Happy simulating! 🔆
