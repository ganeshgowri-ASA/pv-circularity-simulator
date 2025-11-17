# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes **IEC 63202 CTM testing**, CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### IEC 63202 Cell-to-Module (CTM) Testing ⚡ **NEW**

Comprehensive CTM testing and power loss validation system per IEC 63202 standard:

- **IEC63202CTMTester**: Complete test procedure implementation
  - Reference cell measurement under STC (1000 W/m², 25°C, AM1.5)
  - Module flash testing with spectral and spatial corrections
  - CTM power ratio calculation with uncertainty analysis
  - Automated compliance validation
  - IEC 63202 certificate generation

- **B03 CTM Loss Model**: 24-factor (k1-k24) comprehensive loss analysis
  - Cell-level losses (k1-k5): binning, degradation, breakage, measurement
  - Interconnection losses (k6-k10): ribbon, solder, busbar, mismatch, shading
  - Encapsulation losses (k11-k15): glass, encapsulant, backsheet, lamination
  - Assembly losses (k16-k20): junction box, frame, edge effects, thermal, QC
  - Measurement losses (k21-k24): spectral, spatial, uncertainty, temperature

- **CTMPowerLossAnalyzer**: Detailed loss component analysis
  - Optical losses: reflection, absorption, grid shading
  - Electrical losses: series resistance, cell mismatch
  - Thermal losses: assembly temperature effects
  - Spatial non-uniformity analysis
  - Spectral mismatch correction (IEC 60904-7)

- **ReferenceDeviceCalibration**: Traceability management
  - Calibration against primary standards (WPVS)
  - Temperature and spectral corrections
  - Flash simulator uniformity validation
  - Full SI traceability documentation

- **CTMTestReport**: Professional reporting
  - Interactive HTML reports with Plotly visualizations
  - Excel exports with detailed data tables
  - PDF certificates for compliance
  - Loss waterfall charts and IV curve comparisons

- **Streamlit Web Interface**: User-friendly testing platform
  - Interactive test configuration
  - Real-time CTM calculation
  - Loss breakdown visualization
  - Compliance dashboard
  - Multi-format report export

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic CTM Testing Example

```python
from datetime import datetime
from pv_circularity_simulator.core.iec63202 import (
    IEC63202CTMTester,
    CTMTestConfig,
    CellProperties,
    ModuleConfiguration,
    CellTechnology,
)

# Configure test
cell_props = CellProperties(
    technology=CellTechnology.PERC,
    area=244.3,
    efficiency=22.8,
    voc=0.682,
    isc=8.52,
    pmax=5.22,
    # ... other parameters
)

module_config = ModuleConfiguration(
    num_cells_series=60,
    num_strings_parallel=1,
)

test_config = CTMTestConfig(
    test_id="CTM-2025-001",
    laboratory="PV Testing Lab",
    operator="Your Name",
    cell_properties=cell_props,
    module_config=module_config,
    # ... reference device and flash simulator config
)

# Run CTM test
tester = IEC63202CTMTester(config=test_config)
result = tester.ctm_power_ratio_test(
    cell_measurements=cell_iv_curves,
    module_measurements=module_iv_curves,
)

print(f"CTM Ratio: {result.ctm_ratio:.2f}%")
print(f"Compliance: {'PASS' if result.compliance_status else 'FAIL'}")

# Generate certificate
certificate = tester.generate_ctm_certificate()
```

### B03 CTM Loss Model Example

```python
from pv_circularity_simulator.core.ctm.b03_ctm_loss_model import (
    B03CTMLossModel,
    B03CTMConfiguration,
)

# Create model
model = B03CTMLossModel()

# Analyze premium quality scenario
config = B03CTMConfiguration.from_scenario("premium_quality")
result = model.calculate_ctm_losses(config)

print(f"CTM Ratio: {result.total_ctm_ratio_percent:.2f}%")
print(f"Total Loss: {result.total_loss_percent:.2f}%")

# Get loss breakdown
breakdown = result.get_loss_breakdown()
for category, loss in breakdown.items():
    print(f"{category}: {loss:.3f}%")

# Sensitivity analysis
sensitivity = model.sensitivity_analysis(
    base_configuration=config,
    factor_to_vary="k10_interconnect_shading",
)
```

### Launch Streamlit Web Interface

```bash
streamlit run src/pv_circularity_simulator/ui/app.py
```

Then navigate to http://localhost:8501 in your browser.

## Documentation

### CTM Testing Standards Compliance

- **IEC 63202**: Cell-to-module power ratio testing
- **IEC 60904-2**: Reference device calibration
- **IEC 60904-7**: Spectral mismatch correction
- **IEC 60904-9**: Flash simulator classification
- **IEC 60891**: Temperature and irradiance corrections
- **GUM**: Guide to Uncertainty in Measurement

### Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity_simulator/
│       ├── core/
│       │   ├── iec63202/          # IEC 63202 CTM testing
│       │   │   ├── models.py      # Pydantic data models
│       │   │   ├── tester.py      # IEC63202CTMTester
│       │   │   ├── loss_analyzer.py  # CTMPowerLossAnalyzer
│       │   │   ├── calibration.py    # ReferenceDeviceCalibration
│       │   │   └── report.py         # CTMTestReport
│       │   ├── ctm/               # CTM loss models
│       │   │   └── b03_ctm_loss_model.py  # B03 k1-k24 model
│       │   └── utils/
│       │       └── constants.py   # Physical constants, k factors
│       └── ui/
│           ├── app.py             # Main Streamlit app
│           └── pages/
│               └── iec_ctm_testing.py  # CTM testing UI
├── tests/
│   ├── unit/                      # Unit tests
│   └── conftest.py                # Pytest fixtures
├── examples/
│   └── example_ctm_testing.py     # Usage examples
├── pyproject.toml                 # Project configuration
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_circularity_simulator --cov-report=html

# Run specific test suite
pytest tests/unit/test_iec63202_tester.py

# Run tests in parallel
pytest -n auto
```

## Examples

See `examples/example_ctm_testing.py` for comprehensive examples including:

1. **Basic CTM Test**: Complete workflow from configuration to certification
2. **B03 Loss Model**: Quality scenario comparison and sensitivity analysis
3. **Advanced Loss Analysis**: Detailed optical, electrical, thermal analysis
4. **Report Generation**: Multi-format export (HTML, Excel, PDF)

Run examples:
```bash
python examples/example_ctm_testing.py
```

## Key Technologies

- **Pydantic**: Data validation and settings management
- **NumPy/SciPy**: Scientific computing and numerical analysis
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **ReportLab**: PDF report generation

## CTM Loss Factors (k1-k24)

The B03 model includes 24 individual loss factors organized into 5 categories:

### Cell-Level (k1-k5)
- k1: Cell binning tolerance
- k2: Storage degradation
- k3: Cell breakage/microcracks
- k4: Measurement uncertainty
- k5: Temperature variation

### Interconnection (k6-k10)
- k6: Ribbon resistance
- k7: Solder joint quality
- k8: Busbar resistance
- k9: Cell mismatch
- k10: Interconnect shading

### Encapsulation (k11-k15)
- k11: Glass transmission
- k12: Encapsulant transmission
- k13: Encapsulant absorption
- k14: Backsheet reflectance
- k15: Lamination defects

### Assembly (k16-k20)
- k16: Junction box/diode losses
- k17: Frame shading
- k18: Module edge effects
- k19: Thermal stress
- k20: Quality control

### Measurement (k21-k24)
- k21: Flash simulator spectrum
- k22: Spatial uniformity
- k23: Measurement uncertainty
- k24: Temperature variation

Typical CTM ratios by quality:
- **Premium Quality**: 98-100% (tight binning, MBB, excellent QC)
- **Standard Quality**: 96-98% (normal manufacturing)
- **Economy Quality**: 94-96% (cost-optimized)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator: IEC 63202 CTM Testing and Power Loss Validation},
  author = {PV Circularity Team},
  year = {2025},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [Project maintainers]

## Acknowledgments

- IEC TC82 for PV testing standards
- NREL for reference calibration methodology
- PV research community for CTM loss modeling insights
