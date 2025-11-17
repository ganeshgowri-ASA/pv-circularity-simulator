# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Overview

The PV Circularity Simulator is a comprehensive platform for analyzing photovoltaic systems throughout their entire lifecycle, with a focus on circular economy principles (Reduce, Reuse, Recycle). The platform provides engineering-grade analysis tools for system design, performance monitoring, and end-of-life decision making.

## Features

### ✅ RepowerAnalyzer (Production-Ready)

Comprehensive repower analysis and feasibility study tool for existing PV systems:

- **Capacity Upgrade Analysis**: Evaluate maximum capacity increase potential with constraint identification (space, structural, electrical)
- **Component Replacement Planning**: Prioritized, time-phased replacement planning based on health status and risk
- **Technical Feasibility Assessment**: Multi-dimensional feasibility scoring across structural, electrical, spatial, regulatory, and integration dimensions
- **Economic Viability Analysis**: Complete financial modeling with NPV, IRR, ROI, LCOE, sensitivity analysis, and break-even conditions

### 🚧 Coming Soon

- Cell design optimization
- Module engineering tools
- System planning and layout
- Performance monitoring and degradation analysis
- Circularity modeling and optimization
- CTM (Cell-to-Module) loss analysis
- SCAPS integration for device simulation
- Reliability testing frameworks
- Energy forecasting models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Quick Start

### RepowerAnalyzer Example

```python
from pv_simulator import RepowerAnalyzer
from pv_simulator.core.models import PVSystem, RepowerScenario
from pv_simulator.core.enums import RepowerStrategy

# Create your PV system model
system = PVSystem(
    system_id="MY-SYSTEM-001",
    dc_capacity=100.0,  # kW
    # ... other parameters
)

# Initialize the analyzer
analyzer = RepowerAnalyzer()

# 1. Analyze capacity upgrade potential
capacity_analysis = analyzer.capacity_upgrade_analysis(system)
print(f"Max additional capacity: {capacity_analysis.max_additional_capacity:.2f} kW")
print(f"Recommended upgrade: {capacity_analysis.recommended_upgrade:.2f} kW")
print(f"Limiting factor: {capacity_analysis.limiting_factor}")

# 2. Plan component replacements
replacement_plan = analyzer.component_replacement_planning(system)
print(f"Immediate replacements: {len(replacement_plan.immediate_replacements)}")
print(f"Total cost: ${replacement_plan.total_replacement_cost:,.2f}")

# 3. Check technical feasibility
target_capacity = system.dc_capacity + capacity_analysis.recommended_upgrade
feasibility = analyzer.technical_feasibility_check(
    system,
    target_capacity=target_capacity
)
print(f"Technically feasible: {feasibility.is_feasible}")
print(f"Feasibility score: {feasibility.feasibility_score:.1f}/100")

# 4. Analyze economic viability
scenarios = [
    RepowerScenario(
        scenario_id="SCENARIO-1",
        strategy=RepowerStrategy.MODULE_ONLY,
        # ... scenario parameters
    ),
    # Add more scenarios...
]

economics = analyzer.economic_viability_analysis(
    system,
    repower_scenarios=scenarios,
    electricity_rate=0.12,
    incentives={"Federal ITC": 10000.0}
)

print(f"Economically viable: {economics.is_viable}")
print(f"Best scenario NPV: ${economics.best_scenario.economic_metrics.npv:,.2f}")
print(f"Payback period: {economics.best_scenario.economic_metrics.payback_period:.1f} years")
```

See `examples/repower_analysis_example.py` for a complete workflow demonstration.

## Documentation

### Core Concepts

The simulator is built around several core concepts:

- **PVSystem**: Complete system model including location, modules, components, and performance data
- **ComponentHealth**: Health status and performance metrics for individual system components
- **RepowerScenario**: Complete repower plan with technical specs and cost breakdown
- **Economic Metrics**: Standard financial metrics (NPV, IRR, ROI, LCOE, payback period)

### Key Models

All models use Pydantic for validation and type safety:

- `PVSystem`: Complete system specification
- `PVModule`: Module specifications and performance
- `Location`: Geographic and environmental data
- `ComponentHealth`: Component status and health
- `RepowerScenario`: Repower plan and strategy
- `EconomicMetrics`: Financial performance metrics
- `TechnicalFeasibilityResult`: Feasibility assessment results

### Analyzers

- **RepowerAnalyzer**: Repower feasibility analysis
  - `capacity_upgrade_analysis()`: Evaluate upgrade potential
  - `component_replacement_planning()`: Plan replacements
  - `technical_feasibility_check()`: Assess feasibility
  - `economic_viability_analysis()`: Analyze economics

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/test_repower_analyzer.py -v
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_simulator/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── enums.py          # Enumerations
│       │   └── models.py         # Pydantic models
│       └── analyzers/
│           ├── __init__.py
│           └── repower_analyzer.py
├── tests/
│   ├── __init__.py
│   └── test_repower_analyzer.py
├── examples/
│   ├── README.md
│   └── repower_analysis_example.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Requirements

- Python >= 3.9
- pydantic >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

See `pyproject.toml` for complete dependency list.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is part of ongoing research in photovoltaic system circularity and lifecycle optimization.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title={PV Circularity Simulator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pv-circularity-simulator}
}
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
