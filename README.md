# PV Circularity Simulator

An end-to-end PV (photovoltaic) lifecycle simulation platform with comprehensive circularity assessment dashboard.

**End-to-end PV lifecycle simulation**: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🌟 Features

### Circularity Assessment Dashboard

The **CircularityDashboardUI** provides a production-ready Streamlit interface for analyzing PV circularity metrics:

- **📊 Material Flow Visualizer**: Interactive Sankey diagrams showing material movement through manufacturing, operation, and end-of-life stages
- **♻️ 3R Strategies Analysis**: Comprehensive tracking of Reuse, Repair, and Recycling metrics
- **📈 Impact Scorecards**: Environmental and economic impact assessment with baseline vs circular comparisons
- **📋 Policy Compliance Tracker**: Multi-jurisdiction regulatory compliance monitoring with real-time status updates

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Using Sample Data

The application includes a sample data generator for testing:

```python
from examples.sample_data_generator import generate_sample_circularity_data

# Generate sample data
metrics = generate_sample_circularity_data()

# Use with dashboard
from pv_circularity_simulator.dashboards import CircularityDashboardUI
dashboard = CircularityDashboardUI(metrics=metrics)
```

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity_simulator/
│       ├── core/                      # Core data models
│       │   ├── data_models.py        # Circularity metrics models
│       │   └── __init__.py
│       ├── dashboards/                # Dashboard UI components
│       │   ├── circularity_dashboard.py
│       │   ├── components/            # Reusable UI components
│       │   └── __init__.py
│       ├── data/                      # Data processing
│       ├── utils/                     # Utilities
│       └── config/                    # Configuration
├── examples/
│   └── sample_data_generator.py      # Sample data for testing
├── tests/                             # Unit tests
├── .streamlit/
│   └── config.toml                   # Streamlit configuration
├── app.py                            # Main application entry point
├── requirements.txt                  # Python dependencies
└── README.md
```

## 📊 Dashboard Components

### 1. Material Flow Visualizer

Visualizes material flows through the PV lifecycle:
- Interactive Sankey diagrams
- Material-specific filtering
- Stage-by-stage efficiency analysis
- Mass balance calculations
- Loss identification

### 2. Reuse, Repair, Recycling Tabs

Comprehensive 3R strategy metrics:

**Reuse:**
- Collection and reuse rates
- Quality grade distribution
- Cost savings and CO₂ avoidance
- Residual capacity analysis

**Repair:**
- Assessment and success rates
- Common failure modes
- Performance recovery metrics
- Economic analysis

**Recycling:**
- Material recovery rates by type
- Process efficiency metrics
- Resource consumption tracking
- Economic value analysis

### 3. Impact Scorecards

Environmental and economic impact assessment:
- Baseline vs circular comparison
- Multi-category tracking (carbon, water, waste, energy)
- Target progress monitoring
- Data quality indicators

### 4. Policy Compliance Tracker

Regulatory compliance monitoring:
- Multi-jurisdiction tracking (EU, US, China, Japan)
- Collection and recovery rate gauges
- Deadline tracking and alerts
- Penalty assessment

## 🔧 API Usage

### Creating a Dashboard

```python
from pv_circularity_simulator.dashboards import CircularityDashboardUI
from pv_circularity_simulator.core import CircularityMetrics

# Initialize with your metrics
metrics = CircularityMetrics(
    assessment_id="ASSESS-001",
    circularity_index=75.5
)

# Create dashboard
dashboard = CircularityDashboardUI(
    metrics=metrics,
    title="My Custom Dashboard",
    cache_enabled=True
)

# Render (when using Streamlit)
dashboard.render()
```

### Working with Data Models

```python
from pv_circularity_simulator.core.data_models import (
    MaterialFlow,
    ReuseMetrics,
    RecyclingMetrics,
    PolicyCompliance,
    ImpactScorecard,
    MaterialType,
    ProcessStage
)

# Create material flow
flow = MaterialFlow(
    material_type=MaterialType.SILICON,
    stage=ProcessStage.MANUFACTURING,
    input_mass_kg=10000,
    output_mass_kg=9500,
    loss_mass_kg=500,
    location="Germany"
)

# Create reuse metrics
reuse = ReuseMetrics(
    total_modules_collected=1000,
    modules_reused=750,
    avg_residual_capacity_pct=85.0,
    cost_savings_usd=125000
)
```

## 📦 Dependencies

- **Streamlit** (>=1.28.0): Dashboard framework
- **Pandas** (>=2.0.0): Data manipulation
- **Plotly** (>=5.17.0): Interactive visualizations
- **NumPy** (>=1.24.0): Numerical computing
- **Pydantic** (>=2.0.0): Data validation

See `requirements.txt` for complete list.

## 🧪 Testing

Run the sample data generator:
```bash
python examples/sample_data_generator.py
```

Run tests (when implemented):
```bash
pytest tests/
```

## 📚 Documentation

Key classes and methods:

### CircularityDashboardUI

Main dashboard class with four core visualization methods:

- `material_flow_visualizer()`: Visualize material flows with Sankey diagrams
- `reuse_repair_recycling_tabs()`: Display 3R strategy metrics in tabs
- `impact_scorecards()`: Show environmental/economic impact assessments
- `policy_compliance_tracker()`: Track regulatory compliance status

### Data Models

- `CircularityMetrics`: Comprehensive circularity assessment container
- `MaterialFlow`: Material flow tracking through lifecycle stages
- `ReuseMetrics`: Module reuse strategy metrics
- `RepairMetrics`: Module repair operation metrics
- `RecyclingMetrics`: Material recycling efficiency metrics
- `PolicyCompliance`: Regulatory compliance tracking
- `ImpactScorecard`: Impact assessment by category

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with full docstrings
4. Add tests if applicable
5. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python)
- [PV Circularity Best Practices](https://github.com)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Version:** 0.1.0
**Status:** Production-ready
**Last Updated:** 2025-01-17
