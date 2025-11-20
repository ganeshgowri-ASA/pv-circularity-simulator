# PV Circularity Simulator - Energy Yield Analysis Dashboard

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R).

This release focuses on **BATCH5-B06-S05: Energy Yield Analysis (EYA) UI Dashboard** with production-ready capabilities for comprehensive energy yield analysis.

## Features

### ✨ Core Capabilities

#### B05 Energy Forecasting Module
- Weather data processing and synthetic generation
- POA (Plane of Array) irradiance calculations
- Cell temperature modeling
- DC/AC power forecasting with inverter clipping
- Hourly energy production simulation
- Uncertainty quantification

#### B06 Energy Yield Analysis Module
- Performance ratio (PR) calculations
- Detailed loss analysis (13+ loss categories)
- Financial metrics (LCOE, NPV, IRR, payback)
- Sensitivity analysis for key parameters
- Probabilistic analysis (P50, P90, P99)
- Degradation impact modeling

### 📊 Dashboard Components

#### EYADashboard
- `project_overview()`: Project and system configuration
- `annual_energy_output()`: Energy production analysis
- `performance_ratio()`: PR and efficiency metrics
- `losses_waterfall()`: Comprehensive loss breakdown
- `financial_metrics()`: Complete economic analysis

#### ComprehensiveReports
- `eya_pdf_generator()`: Professional PDF reports (reportlab)
- `excel_export()`: Multi-sheet Excel workbooks (openpyxl)
- `sensitivity_analysis_tables()`: Parameter sensitivity
- `p50_p90_p99_analysis()`: Monte Carlo probabilistic analysis

#### InteractiveVisualizations
- `monthly_production_charts()`: Plotly multi-panel charts
- `loss_breakdown_sankey()`: Energy flow diagrams
- `weather_correlation_plots()`: Environmental impact
- Custom Altair chart builder

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

### Using the API

```python
from datetime import datetime
from src.models.eya_models import ProjectInfo, SystemConfiguration, ModuleType
from src.ui.dashboard import EYADashboard

# Configure project
project_info = ProjectInfo(
    project_name="Solar PV Project",
    location="San Francisco, CA",
    latitude=37.7749,
    longitude=-122.4194,
    commissioning_date=datetime(2024, 1, 1),
)

system_config = SystemConfiguration(
    capacity_dc=1000.0,
    capacity_ac=850.0,
    module_type=ModuleType.MONO_SI,
    module_efficiency=0.20,
    module_count=5000,
    tilt_angle=30.0,
    azimuth_angle=180.0,
)

# Initialize dashboard
dashboard = EYADashboard(project_info, system_config)

# Get results
energy_data = dashboard.annual_energy_output()
print(f"Annual Energy: {energy_data['Annual Totals']['AC Energy']}")
```

## Dashboard Pages

1. **🏠 Home**: Overview and navigation
2. **📊 Energy Analysis**: Production forecasts and monthly breakdowns
3. **📉 Performance**: PR analysis and efficiency metrics
4. **🔻 Losses**: Detailed loss waterfall and mitigation strategies
5. **💰 Financial**: LCOE, NPV, IRR, cash flow projections
6. **📋 Reports**: PDF/Excel generation, sensitivity & probabilistic analysis
7. **📈 Visualizations**: Interactive charts and custom builders

## Technology Stack

- **Framework**: Streamlit multi-page
- **Charts**: Plotly, Altair
- **PDF**: reportlab
- **Excel**: openpyxl
- **Validation**: Pydantic v2
- **PV Modeling**: pvlib
- **Data**: pandas, numpy, scipy

## Project Structure

```
pv-circularity-simulator/
├── app.py                          # Main Streamlit app
├── pages/                          # Dashboard pages
│   ├── 1_📊_Energy_Analysis.py
│   ├── 2_📉_Performance.py
│   ├── 3_🔻_Losses.py
│   ├── 4_💰_Financial.py
│   ├── 5_📋_Reports.py
│   └── 6_📈_Visualizations.py
├── src/
│   ├── models/eya_models.py        # Pydantic data models
│   ├── modules/
│   │   ├── B05_energy_forecasting/
│   │   └── B06_energy_yield_analysis/
│   └── ui/
│       ├── dashboard.py            # Main controller
│       ├── reports.py              # PDF/Excel generation
│       └── visualizations.py       # Interactive charts
├── tests/                          # Unit tests
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Documentation

Full docstrings throughout (Google style). Key models:

- **ProjectInfo**: Location and metadata
- **SystemConfiguration**: PV system parameters
- **WeatherData**: Irradiance and weather
- **EnergyOutput**: Production results
- **PerformanceMetrics**: PR and yields
- **LossBreakdown**: System losses
- **FinancialMetrics**: Economic analysis
- **ProbabilisticAnalysis**: P-values

## License

MIT License - see LICENSE file

## Version

**0.1.0** - Production-ready
**Module**: BATCH5-B06-S05
**Status**: ✅ Complete
