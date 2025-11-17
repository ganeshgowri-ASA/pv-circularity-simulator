# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, circular economy modeling, and **hydrogen system integration**.

## Features

### 🔋 Hydrogen System Integration & Power-to-X (New!)

Production-ready hydrogen system modeling for renewable energy integration:

- **Electrolyzer Modeling**: PEM, Alkaline, SOEC, AEM technologies with dynamic efficiency, degradation, and LCOH analysis
- **H2 Storage Design**: Compressed gas, liquid H2, metal hydride, LOHC, underground storage with SOC dynamics
- **Fuel Cell Integration**: PEMFC, SOFC, MCFC with CHP capabilities and performance analysis
- **Power-to-X Pathways**: Methanol, methane, ammonia, Fischer-Tropsch fuels with techno-economic analysis

See [Hydrogen Module Documentation](src/pv_circularity_simulator/hydrogen/README.md) for detailed information.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install the package
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,analysis]"
```

### Basic Example - Hydrogen System

```python
from pv_circularity_simulator.hydrogen import (
    HydrogenIntegrator,
    ElectrolyzerConfig,
    ElectrolyzerType,
)

# Initialize integrator
integrator = HydrogenIntegrator(
    discount_rate=0.05,
    electricity_price_kwh=0.04,
)

# Configure PEM electrolyzer
config = ElectrolyzerConfig(
    electrolyzer_type=ElectrolyzerType.PEM,
    rated_power_kw=1000.0,
    efficiency=0.68,
)

# Simulate with renewable power profile
power_profile = [800.0] * 8760  # 1 year hourly data
results = integrator.electrolyzer_modeling(
    config=config,
    power_input_profile=power_profile,
    timestep_hours=1.0,
)

print(f"Annual H2: {results.annual_h2_production_kg:.0f} kg")
print(f"LCOH: ${results.levelized_cost_h2:.2f}/kg")
```

## Project Structure

```
pv-circularity-simulator/
├── src/pv_circularity_simulator/
│   └── hydrogen/                    # Hydrogen system integration module
│       ├── __init__.py
│       ├── models.py               # Pydantic models for all components
│       ├── integrator.py           # Core HydrogenIntegrator class
│       └── README.md               # Detailed module documentation
├── examples/
│   └── hydrogen_integration_example.py  # Comprehensive examples
├── tests/
│   └── test_hydrogen_integration.py     # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Examples

Run comprehensive examples:

```bash
python examples/hydrogen_integration_example.py
```

This demonstrates:
1. Electrolyzer modeling with variable renewable power
2. Storage system design with charge/discharge cycles
3. Fuel cell CHP integration
4. Power-to-Methanol pathway analysis

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=pv_circularity_simulator --cov-report=html
```

## Documentation

- [Hydrogen System Integration](src/pv_circularity_simulator/hydrogen/README.md) - Comprehensive hydrogen module docs
- API documentation available in docstrings (Google style)

## Technology Stack

- **Python 3.9+**: Modern Python with type hints
- **Pydantic v2**: Data validation and settings management
- **NumPy**: Numerical computations
- **Pytest**: Testing framework

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Roadmap

- [x] Hydrogen system integration (BATCH8-B12-S03)
  - [x] Electrolyzer modeling with multiple technologies
  - [x] H2 storage design and optimization
  - [x] Fuel cell integration with CHP
  - [x] Power-to-X pathway analysis
- [ ] PV module lifecycle modeling
- [ ] CTM loss analysis
- [ ] Circular economy metrics
- [ ] Integration with SCAPS

## Contributing

We welcome contributions! Please ensure:
- Full type hints and docstrings
- Pydantic validation for all models
- Unit tests with pytest
- Code formatted with black
- Passes ruff linting

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator},
  author = {PV Circularity Team},
  year = {2025},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.
