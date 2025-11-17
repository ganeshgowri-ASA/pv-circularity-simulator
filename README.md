# PV Circularity Simulator

Comprehensive PV system lifecycle and circularity simulator with hybrid energy systems, financial analysis, and integration capabilities.

## 🚀 Features

### B12 - Hybrid Energy Systems (5 Modules)
- **B12-S01**: Battery Integration & Energy Storage
  - Battery sizing, charge/discharge modeling
  - Arbitrage optimization
  - Degradation analysis

- **B12-S02**: Wind-Solar Hybrid Systems
  - Wind resource analysis
  - Hybrid capacity optimization
  - Temporal complementarity analysis

- **B12-S03**: Hydrogen Integration & P2X
  - Electrolyzer sizing
  - H2 storage modeling
  - Fuel cell integration

- **B12-S04**: Grid Interaction & Smart Grid
  - Grid services (frequency regulation, voltage support)
  - Demand response
  - Power quality analysis

- **B12-S05**: Hybrid Systems UI
  - System topology visualization
  - Optimization dashboard
  - Dispatch strategies

### B13 - Financial Analysis (5 Modules)
- **B13-S01**: LCOE Calculations
  - Levelized cost of energy
  - Sensitivity analysis
  - Scenario comparison

- **B13-S02**: NPV Analysis
  - Cash flow projections
  - Net present value
  - Payback period analysis

- **B13-S03**: IRR Modeling
  - Internal rate of return
  - Modified IRR (MIRR)
  - Hurdle rate comparison

- **B13-S04**: Bankability Assessment
  - Risk assessment
  - Debt service coverage ratio (DSCR)
  - Credit rating

- **B13-S05**: Financial Dashboard
  - Financial summary metrics
  - Cash flow waterfall charts
  - Sensitivity tornado charts

### B14 - Core Infrastructure (3 Modules)
- **B14-S02**: Data Models & Utilities
  - Comprehensive Pydantic models
  - Validators and utilities

- **B14-S03**: Integration Layer
  - Cross-module data flow
  - API endpoints
  - Data synchronization

- **B14-S04**: Utilities & Helpers
  - Unit conversions
  - Financial utilities
  - Statistical functions
  - Data export helpers

### B15 - UI & Visualization (2 Modules)
- **B15-S03**: Navigation & Routing
  - Multi-page routing
  - Menu structure
  - Breadcrumb navigation

- **B15-S04**: Data Visualization Library
  - Chart templates (line, bar, scatter, pie, heatmap, Sankey)
  - Interactive plots
  - Export capabilities

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## 🎯 Usage

### Run Streamlit App

```bash
streamlit run app.py
```

## 🏗️ Project Structure

```
pv-circularity-simulator/
├── src/
│   └── modules/
│       ├── hybrid_energy/
│       │   ├── battery_integration.py
│       │   ├── wind_hybrid.py
│       │   ├── hydrogen_system.py
│       │   ├── grid_connector.py
│       │   └── hybrid_ui.py
│       ├── financial/
│       │   ├── lcoe_calculator.py
│       │   ├── npv_analyzer.py
│       │   ├── irr_calculator.py
│       │   ├── bankability_analyzer.py
│       │   └── financial_ui.py
│       ├── core/
│       │   ├── data_models.py
│       │   ├── utilities.py
│       │   └── integration_layer.py
│       └── ui/
│           ├── navigation.py
│           └── visualization.py
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Technology Stack

- **Python 3.9+**
- **Pydantic 2.0+**: Data validation and modeling
- **NumPy & Pandas**: Numerical computing and data analysis
- **Streamlit**: Interactive web applications
- **Plotly**: Interactive visualizations
- **SciPy**: Scientific computing

## 📊 Module Coverage

| Category | Modules | Status |
|----------|---------|--------|
| Hybrid Energy | 5 | ✅ Complete |
| Financial | 5 | ✅ Complete |
| Core Infrastructure | 3 | ✅ Complete |
| UI & Visualization | 2 | ✅ Complete |
| **Total** | **15** | **✅ 100%** |
