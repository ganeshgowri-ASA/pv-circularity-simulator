# PV Circularity Simulator

A comprehensive web application for simulating the complete lifecycle of photovoltaic (PV) systems, from design through operation to end-of-life circularity strategies.

## Overview

The PV Circularity Simulator bridges the gap between traditional PV system design tools and circular economy principles, enabling stakeholders to make data-driven decisions that optimize both performance and sustainability.

### Key Features

- **Design Optimization**: Data-driven material selection and module design with circularity considerations
- **Performance Simulation**: Accurate energy yield and performance predictions (EYA/HYA)
- **Operational Monitoring**: Real-time performance tracking and fault diagnostics
- **Circular Economy**: Integrated reduce-reuse-recycle (3R) analysis
- **Financial Analysis**: Comprehensive LCOE, NPV, IRR, and payback calculations

## Modules

### 1. 📊 Dashboard
Central hub for project management and workflow navigation.

### 2. 🔬 Material Selection
Select and compare PV materials based on performance, cost, and recyclability.

### 3. ⚡ Module Design
Design PV module specifications including cell configuration and electrical parameters.

### 4. 📉 CTM Loss Analysis
Analyze Cell-to-Module losses using detailed k-factor model (k1-k15, k21-k24).

### 5. 🏗️ System Design
Design complete PV system configuration including site, array, and inverter.

### 6. ☀️ EYA Simulation
Energy Yield Assessment for pre-construction production estimation.

### 7. 📈 Performance Monitoring
Real-time operational performance tracking and analysis.

### 8. 🔍 Fault Diagnostics
Automated fault detection and classification using multiple diagnostic methods.

### 9. 📅 HYA Simulation
Historical Yield Analysis for post-construction performance validation.

### 10. 🔮 Energy Forecasting
Short-term and long-term energy production forecasting.

### 11. 🔄 Revamp & Repower
System upgrade planning and ROI analysis.

### 12. ♻️ Circularity (3R)
Comprehensive circular economy analysis: Reduce, Reuse, Recycle.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run pv_circularity_simulator/src/main.py
```

4. Open your browser to `http://localhost:8501`

## Project Structure

```
pv_circularity_simulator/
├── docs/                          # Documentation
│   ├── MASTER_PROMPT.md          # Complete specification
│   ├── ARCHITECTURE.md           # System architecture
│   └── MODULE_SPECS.md           # Module specifications
│
├── src/                          # Source code
│   ├── main.py                   # Application entry point
│   ├── modules/                  # Feature modules
│   ├── core/                     # Core functionality
│   └── utils/                    # Utility functions
│
├── data/                         # Data files
│   ├── materials_db.json         # Material database
│   ├── cell_types.json           # Cell technology specs
│   └── standards.json            # Industry standards
│
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### Creating a New Project

1. Launch the application
2. Navigate to the Dashboard
3. Enter a project name and click "Create Project"
4. Follow the workflow through the modules

### Module Workflow

**Design Phase:**
1. Material Selection → Choose materials for module components
2. Module Design → Configure cell type and electrical parameters
3. CTM Loss Analysis → Analyze cell-to-module losses
4. System Design → Configure complete system layout

**Simulation Phase:**
5. EYA Simulation → Estimate pre-construction energy yield

**Operational Phase:**
6. Performance Monitoring → Track real-time performance
7. Fault Diagnostics → Identify and diagnose system faults
8. HYA Simulation → Validate actual vs. expected performance
9. Energy Forecasting → Predict future production

**Lifecycle Phase:**
10. Revamp & Repower → Plan system upgrades
11. Circularity (3R) → Analyze sustainability and end-of-life strategies

## Key Technologies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **PVlib**: PV system modeling
- **Pydantic**: Data validation
- **SciPy**: Scientific computing

## CTM Loss Factors

The simulator uses a comprehensive 19-factor model for Cell-to-Module loss analysis:

**Optical (k1-k3):** Reflection, Shading, Absorption
**Electrical (k4-k6):** Resistive, Mismatch, Junction Box
**Thermal (k7-k8):** Temperature, Hotspot
**Assembly (k9-k10):** Encapsulation, Lamination
**Degradation (k11-k15):** LID, PID, Mechanical, Cell, Interconnect
**Environmental (k21-k24):** Humidity, UV, Thermal Cycling, Corrosion

Total CTM Ratio = k1 × k2 × ... × k24 (typically 0.88-0.97)

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[MASTER_PROMPT.md](pv_circularity_simulator/docs/MASTER_PROMPT.md)**: Complete vision, scope, and specifications
- **[ARCHITECTURE.md](pv_circularity_simulator/docs/ARCHITECTURE.md)**: System architecture and design patterns
- **[MODULE_SPECS.md](pv_circularity_simulator/docs/MODULE_SPECS.md)**: Detailed module specifications

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 guidelines. Use type hints where applicable.

### Contributing

1. Create a feature branch from `main`
2. Follow the naming convention: `feature/module-name`
3. Implement module following the standard structure
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

## Roadmap

### Version 1.0 (Current)
- ✅ Core design modules
- ✅ Basic simulation capabilities
- ✅ Circularity analysis
- ✅ Comprehensive documentation

### Version 2.0 (Planned)
- Machine learning for fault classification
- Automated optimization algorithms
- Multi-site portfolio management
- Battery storage integration
- API for third-party integrations

### Version 3.0 (Future)
- Real-time data streaming
- Cloud deployment
- Mobile application
- Advanced predictive analytics
- Blockchain for circularity tracking

## Standards and Compliance

The simulator aligns with industry standards:

- **IEC 61215**: PV Module Design Qualification
- **IEC 61853**: PV Module Performance Testing
- **IEC 62804**: PID Testing Methods
- **IEEE 1547**: Grid Interconnection
- **WEEE Directive**: E-waste Management
- **RoHS Directive**: Hazardous Substances Restriction

## License

[License information to be added]

## Authors

PV Circularity Simulator Development Team

## Acknowledgments

- PVlib community for open-source PV modeling tools
- IEC and IEEE for industry standards
- Streamlit for the excellent web framework

## Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the development team.

## Citation

If you use this tool in your research, please cite:

```
[Citation format to be added]
```

---

**Version**: 0.1.0
**Status**: Initial Release
**Last Updated**: 2024
