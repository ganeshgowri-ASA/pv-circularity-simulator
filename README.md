# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R).

Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### ✅ CTM Loss Modeling Engine (IMPLEMENTED)

Comprehensive **Cell-to-Module (CTM) Loss Analysis** implementing the Fraunhofer ISE SmartCalc methodology with all **k1-k24 factors**:

- **Optical Losses/Gains (k1-k7)**: Glass reflection, encapsulant effects, shading, absorption
- **Coupling Effects (k8-k11)**: Cell gaps, mismatch, LID/LETID degradation
- **Electrical Losses (k12-k15)**: Resistive losses, interconnections, manufacturing damage
- **Environmental Factors (k21-k24)**: Temperature, irradiance, spectral response, AOI

**Supported Module Architectures**:
- Standard (60/72 cell)
- Half-cut cells
- Quarter-cut cells
- Shingled cells
- IBC (Interdigitated Back Contact)
- Bifacial modules

**Capabilities**:
- Complete CTM power analysis
- Loss/gain waterfall visualization
- Sensitivity analysis
- Environmental modeling
- Production-ready with full type hints and validation

📖 **[Full CTM Documentation](docs/CTM_LOSS_MODEL.md)**

### Planned Modules

- **SCAPS Integration**: Cell-level semiconductor modeling
- **Reliability Testing**: Accelerated aging, degradation models
- **Energy Forecasting**: System-level performance prediction
- **Circular Economy**: 3R analysis (Reduce, Reuse, Recycle)

## Quick Start

### Installation

```bash
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator
pip install -r requirements.txt
```

### CTM Loss Model Example

```python
from src.modules.ctm_loss_model import CTMLossModel, CellParameters, ModuleParameters

# Define cell parameters (5.25W PERC cell)
cell = CellParameters(
    power_stc=5.25,
    voltage_mpp=0.650,
    current_mpp=8.08,
    voltage_oc=0.720,
    current_sc=8.60,
    efficiency=22.8,
    width=166,
    height=166,
)

# Define module (60-cell standard)
module = ModuleParameters(cells_in_series=60)

# Analyze CTM losses
model = CTMLossModel(cell, module)
module_power = model.calculate_module_power()
ctm_ratio = model.get_ctm_ratio()

print(f"Module Power: {module_power:.2f} W")
print(f"CTM Ratio: {ctm_ratio:.4f} ({(ctm_ratio-1)*100:+.2f}%)")

# Generate waterfall visualization
fig = model.generate_loss_waterfall()
fig.savefig('ctm_analysis.png')

# Print detailed report
print(model.generate_report())
```

### Run Demonstrations

```bash
# Comprehensive CTM demonstrations
python examples/ctm_demo.py
```

This generates:
- Standard module analysis with waterfall chart
- Half-cut vs. standard comparison
- Bifacial module rear gain analysis
- Shingled module advantages
- Multi-parameter sensitivity analysis
- Environmental effects (temperature, irradiance, AOI)
- Architecture comparison across all module types

### Run Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_ctm_loss_model.py -v
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       └── ctm_loss_model.py      # CTM Loss Modeling Engine (k1-k24)
├── tests/
│   ├── __init__.py
│   └── test_ctm_loss_model.py     # Comprehensive CTM tests
├── examples/
│   └── ctm_demo.py                # CTM demonstrations
├── docs/
│   └── CTM_LOSS_MODEL.md          # Detailed CTM documentation
├── requirements.txt
├── LICENSE
└── README.md
```

## CTM Loss Model Highlights

### All k-Factors Implemented

| Category | Factors | Description |
|----------|---------|-------------|
| **Optical** | k1-k7 | Glass reflection gain, encapsulant effects, shading, absorption, bifacial |
| **Coupling** | k8-k11 | Cell gaps, internal/module mismatch, LID/LETID |
| **Electrical** | k12-k15 | Resistive losses, interconnection, manufacturing damage |
| **Environmental** | k21-k24 | Temperature, low irradiance, spectral response, AOI |

### Advanced Features

- **Pydantic Models**: Full parameter validation
- **Type Safety**: Complete type hints throughout
- **Visualization**: Matplotlib and Plotly waterfall charts
- **Sensitivity Analysis**: Single and multi-parameter analysis
- **Module Architectures**: Support for 6 different types
- **Production Ready**: Comprehensive testing and documentation

### Validation

Validated against:
- Fraunhofer ISE SmartCalc methodology
- Cell-to-Module.com reference data
- Industry standard CTM ratios (96-99% for monofacial, 110-130% for bifacial)

## CTM Example Results

**Standard 60-Cell PERC Module** (5.25W cells):
- Total Cell Power: 315.0 W
- Module Power: ~303-306 W
- **CTM Ratio: 96-97%** ✓
- Main losses: Glass absorption, cell gaps, resistive losses, LID

**Half-Cut 120-Cell Module**:
- **CTM Ratio: 97-98%** ✓
- Improvement: +0.5-1% vs. standard (reduced I²R losses)

**Bifacial Module** (75% bifaciality, 20% rear irradiance):
- **CTM Ratio: 115-120%** ✓
- Gain: +15-20% from rear side generation

## Dependencies

- **numpy**: Numerical calculations
- **pydantic**: Data validation
- **matplotlib**: Visualization
- **plotly**: Interactive charts
- **scipy**: Scientific computing
- **pandas**: Data handling
- **pytest**: Testing

## Development Status

- ✅ **CTM Loss Modeling Engine**: Complete with k1-k24 factors
- 🔄 **SCAPS Integration**: Planned
- 🔄 **Reliability Testing**: Planned
- 🔄 **Energy Forecasting**: Planned
- 🔄 **Circular Economy Models**: Planned

## Contributing

Contributions welcome! Areas of interest:
- Additional module architectures (e.g., multi-busbar, tandem cells)
- Integration with field data for validation
- Machine learning for parameter optimization
- Real-time performance monitoring integration

## Citation

If you use this simulator in research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator: End-to-End PV Lifecycle Simulation},
  author = {PV Circularity Simulator Development Team},
  year = {2024},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## License

See LICENSE file for details.

## References

1. Fraunhofer ISE, "SmartCalc CTM", https://www.ise.fraunhofer.de
2. Cell-to-Module.com, CTM Calculator and Database
3. IEC 61853 series: PV module performance testing standards
4. Photovoltaic Module Power Rating per IEC 61853-1

## Contact

For questions, issues, or contributions:
- GitHub Issues: [Create an issue](https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues)
- Documentation: [docs/](docs/)

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: 2024
