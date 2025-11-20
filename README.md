# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Recycling Economics & Material Recovery

Comprehensive economic modeling for PV panel recycling with:

- **Material Extraction Costs**: Full cost modeling including collection, processing, energy, labor, and overhead
- **Recovery Rates**: Technology-specific material recovery rates with quality grading
- **Revenue Calculation**: Market-based revenue modeling with quality discounts and transportation costs
- **Environmental Credits**: LCA-integrated environmental benefits quantification (CO2, energy, water)
- **Economic Viability Analysis**: ROI, breakeven analysis, and carbon price sensitivity

**Technologies Supported**:
- Mechanical recycling (crushing, separation)
- Thermal recycling (pyrolysis, delamination)
- Chemical recycling (leaching, extraction)
- Hybrid processes
- Advanced technologies (electrochemical, supercritical)

**Key Materials**:
- High-value: Silicon, Silver, Copper, Indium, Gallium, Tellurium
- Structural: Aluminum, Glass
- Polymers: EVA, backsheet materials

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"
```

## Quick Start

```python
from pv_circularity_simulator.recycling import (
    RecyclingEconomics,
    MaterialComposition,
    MaterialExtractionCosts,
    RecoveryRates,
    PVMaterialType,
    RecyclingTechnology,
)

# Define panel composition
composition = [
    MaterialComposition(
        material_type=PVMaterialType.SILICON,
        mass_kg=5.0,
        purity_percent=99.0,
        market_value_per_kg=15.0,
    ),
]

# Define recycling costs
costs = MaterialExtractionCosts(
    technology=RecyclingTechnology.HYBRID,
    collection_cost_per_panel=5.0,
    preprocessing_cost_per_kg=2.0,
    processing_cost_per_kg=3.0,
    purification_cost_per_kg=1.5,
    labor_cost_per_hour=25.0,
    processing_time_hours=0.5,
    energy_cost_per_kwh=0.12,
    energy_consumption_kwh_per_kg=2.0,
    disposal_cost_per_kg=0.5,
    equipment_depreciation_per_panel=2.0,
)

# Define recovery rates
recovery = RecoveryRates(
    technology=RecyclingTechnology.HYBRID,
    material_recovery_rates={
        PVMaterialType.SILICON: 95.0,
    },
    technology_efficiency=0.85,
)

# Create economics model
economics = RecyclingEconomics(
    panel_composition=composition,
    extraction_costs=costs,
    recovery_rates_model=recovery,
    panel_mass_kg=20.0,
)

# Analyze economics
net_value = economics.net_economic_value()
print(f"Net value: ${net_value['net_value']:.2f}")
print(f"ROI: {net_value['roi_percent']:.1f}%")
```

## Documentation

- [Recycling Economics Module](docs/RECYCLING_ECONOMICS.md) - Comprehensive guide
- [API Reference](docs/API_REFERENCE.md) - Detailed API documentation
- [Examples](examples/) - Usage examples and tutorials

## Examples

Run the comprehensive example:

```bash
python examples/recycling_economics_example.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific module tests
pytest tests/recycling/
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity_simulator/
│       ├── recycling/
│       │   ├── __init__.py
│       │   └── economics.py
│       └── __init__.py
├── tests/
│   └── recycling/
│       └── test_economics.py
├── examples/
│   └── recycling_economics_example.py
├── docs/
│   └── RECYCLING_ECONOMICS.md
├── pyproject.toml
└── README.md
```

## Technology Stack

- **Python 3.9+**
- **Pydantic 2.0+**: Data validation and settings management
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **pytest**: Testing framework

## Roadmap

- [x] Recycling Economics & Material Recovery
- [ ] Cell Design & Manufacturing
- [ ] Module Engineering
- [ ] System Planning & Performance
- [ ] Reliability Testing
- [ ] Energy Forecasting
- [ ] Full Circular Economy Modeling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. IRENA (2016). "End-of-Life Management: Solar Photovoltaic Panels"
2. Latunussa et al. (2016). "Life Cycle Assessment of an innovative recycling process for crystalline silicon photovoltaic panels"
3. Deng et al. (2019). "Economic analysis and environmental assessment of PV panel recycling"
4. IEA-PVPS Task 12 (2018). "Review of Failures of Photovoltaic Modules"

## Contact

For questions or feedback, please open an issue on GitHub.
