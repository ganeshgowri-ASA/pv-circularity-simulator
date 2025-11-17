# PV Circularity Simulator

**End-to-end PV lifecycle simulation platform** covering the complete solar panel journey: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R: Reduce, Reuse, Recycle).

## Features

- **CTM Loss Analysis**: Cell-to-Module conversion efficiency analysis
- **SCAPS Integration**: Solar Cell Capacitance Simulator integration
- **Reliability Testing**: Comprehensive reliability and durability testing
- **Energy Forecasting**: Advanced energy production forecasting
- **Circular Economy Modeling**: Complete lifecycle circular economy analysis
- **🆕 B08 Diagnostics & Maintenance**: Advanced defect detection, fault reporting, and maintenance management

## B08: Diagnostics & Maintenance Management

The B08 module provides production-ready diagnostics and maintenance management for PV installations:

### B08-S04: Fault Reports & Maintenance Recommendations

- **FaultReportGenerator**: Automated fault report generation with defect categorization, severity assessment, and repair cost estimation
- **MaintenanceScheduler**: Preventive maintenance planning, corrective action tracking, and spare parts management
- **WorkOrderManagement**: Complete work order lifecycle with technician assignment, task tracking, and completion verification

### B08-S05: Diagnostics UI & Defect Management Dashboard

- **DefectDatabase**: Comprehensive defect data management with historical tracking, pattern recognition (ML-based), and fleet-wide analysis
- **DiagnosticsUI**: Interactive Streamlit dashboards with defect galleries, severity heatmaps, repair tracking, and cost analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[dev]"
```

### Running the Demo

```bash
# Run the B08 diagnostics and maintenance demo
python docs/examples/b08_demo.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/pv_circularity --cov-report=html

# Run specific module tests
pytest tests/unit/test_b08/
```

## Project Structure

```
pv-circularity-simulator/
├── src/pv_circularity/
│   ├── models/                    # Pydantic data models
│   ├── b08_diagnostics/           # B08 diagnostics module
│   │   ├── b08_s04/              # Fault reports & maintenance
│   │   └── b08_s05/              # Diagnostics UI & database
│   ├── core/                      # Core utilities
│   └── utils/                     # Helper utilities
├── tests/                         # Comprehensive test suite
├── docs/                          # Documentation
│   ├── examples/                  # Example scripts
│   └── B08_DIAGNOSTICS_MAINTENANCE.md
├── config/                        # Configuration files
├── pyproject.toml                 # Project metadata
└── requirements.txt               # Dependencies
```

## Usage Examples

### Fault Report Generation

```python
from pv_circularity.b08_diagnostics.b08_s04 import FaultReportGenerator
from pv_circularity.models import Defect, DefectType, DefectSeverity

# Initialize generator
generator = FaultReportGenerator()

# Generate automated fault report
report = generator.automated_report_generation(
    site_id="SITE-001",
    defects=detected_defects
)

print(f"Total defects: {report.total_defects}")
print(f"Critical defects: {report.critical_defects}")
print(f"Estimated repair cost: ${report.estimated_total_cost:,.2f}")
```

### Maintenance Scheduling

```python
from pv_circularity.b08_diagnostics.b08_s04 import MaintenanceScheduler

# Initialize scheduler
scheduler = MaintenanceScheduler()

# Generate maintenance plan for the year
schedules = scheduler.preventive_maintenance_planning(
    site_id="SITE-001",
    planning_horizon_days=365,
    panel_count=500
)

print(f"Generated {len(schedules)} maintenance schedules")
```

### Defect Analytics

```python
from pv_circularity.b08_diagnostics.b08_s05 import DefectDatabase

# Initialize database
db = DefectDatabase()

# Add and analyze defects
for defect in defects:
    db.add_defect(defect)

# Pattern recognition
patterns = db.pattern_recognition()

# Fleet analysis
analysis = db.fleet_wide_analysis(
    fleet_id="FLEET-001",
    site_ids=["SITE-001", "SITE-002"]
)

print(f"Fleet health score: {analysis.fleet_health_score:.1f}/100")
```

## Documentation

- **[B08 Diagnostics & Maintenance Guide](./docs/B08_DIAGNOSTICS_MAINTENANCE.md)**: Comprehensive guide for the B08 module
- **[API Documentation](./docs/api/)**: Detailed API reference
- **[Examples](./docs/examples/)**: Example scripts and demos

## Technology Stack

- **Python 3.10+**
- **Pydantic 2.5+**: Data validation and serialization
- **Streamlit 1.29+**: Interactive dashboards
- **Plotly 5.18+**: Interactive visualizations
- **Pandas & NumPy**: Data analysis
- **scikit-learn**: Machine learning for pattern recognition
- **pytest**: Testing framework

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or feature requests:
- **GitHub Issues**: [Create an issue](https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues)
- **Documentation**: [Read the docs](./docs/)

## Roadmap

- [x] B08-S04: Fault Reports & Maintenance Recommendations
- [x] B08-S05: Diagnostics UI & Defect Management Dashboard
- [ ] Integration with external IoT sensors
- [ ] Advanced ML models for defect prediction
- [ ] Mobile application for field technicians
- [ ] Cloud-based fleet management
- [ ] API endpoints for third-party integrations

## Acknowledgments

Built with ❤️ for the solar energy industry to improve PV system reliability and efficiency through advanced diagnostics and maintenance management.
