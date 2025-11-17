# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Wind Hybrid System Integration

The platform includes comprehensive wind-PV hybrid system integration capabilities through the `WindHybridIntegrator` class:

- **Wind Resource Assessment**: Weibull distribution analysis, wind power density calculation, turbulence characterization
- **Turbine Modeling**: Detailed performance prediction with power curve interpolation and loss analysis
- **Hybrid Optimization**: Multi-objective optimization for wind-PV-storage systems
- **Wind-PV Coordination**: Real-time dispatch coordination with grid support services

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- pydantic >= 2.0
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10

## Quick Start

```python
from pv_simulator import (
    WindHybridIntegrator,
    HybridSystemConfig,
    WindResourceData,
    TurbineSpecifications,
    PVSystemConfig,
)

# Configure hybrid system
config = HybridSystemConfig(
    system_id="hybrid_001",
    site_name="Example Site",
    pv_capacity_mw=10.0,
    wind_capacity_mw=15.0,
    num_turbines=5,
    pv_system=pv_config,
    turbine_specs=turbine_specs,
    grid_connection_capacity_mw=20.0
)

# Initialize integrator
integrator = WindHybridIntegrator(config)
integrator.initialize()

# Assess wind resources
assessment = integrator.wind_resource_assessment(wind_data)

# Model turbine performance
performance = integrator.turbine_modeling(wind_data)

# Optimize hybrid configuration
optimization = integrator.hybrid_optimization(
    wind_data=wind_data,
    objective="maximize_energy"
)

# Coordinate wind and PV generation
coordination = integrator.wind_pv_coordination(
    wind_generation_mw=wind_gen,
    pv_generation_mw=pv_gen
)
```

See `examples/wind_hybrid_example.py` for a complete example.

## Documentation

- **Architecture**: See `docs/architecture.md`
- **API Reference**: Auto-generated from docstrings
- **Examples**: See `examples/` directory

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_simulator --cov-report=html

# Run specific test file
pytest tests/test_wind_hybrid_integrator.py
```

## Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/           # Main package
│   ├── core/                   # Core models and base classes
│   ├── integrators/            # System integrators (Wind, PV, Hybrid)
│   ├── modules/                # Component modules
│   ├── forecasting/            # Energy forecasting
│   ├── simulation/             # Simulation engine
│   └── utils/                  # Utilities
├── tests/                      # Test suite
├── examples/                   # Example scripts
├── docs/                       # Documentation
└── pyproject.toml             # Package configuration
```

## Development

### Code Quality

This project uses:
- **Pydantic v2** for data validation
- **Type hints** throughout
- **Comprehensive docstrings** (Google style)
- **pytest** for testing
- **ruff** for linting
- **mypy** for type checking

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator},
  author = {PV Circularity Team},
  year = {2024},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Contact

For questions and support, please open an issue on GitHub.
