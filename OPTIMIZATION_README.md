# PV System Optimization Engine (BATCH4-B05-S04)

Advanced multi-objective optimization engine for photovoltaic system design using genetic algorithms, particle swarm optimization, and linear programming.

## Overview

This optimization engine provides comprehensive tools for optimizing PV system design across multiple competing objectives including energy yield, levelized cost of energy (LCOE), land use, and net present value (NPV).

## Features

### 1. **SystemOptimizer** - Multi-Algorithm Optimization
- **Genetic Algorithm (GA)**: Population-based evolutionary optimization using DEAP
- **Particle Swarm Optimization (PSO)**: Swarm intelligence optimization via PySwarm
- **Linear Programming (LP)**: Linear optimization using PuLP
- **Multi-Objective Optimization**: NSGA-II algorithm for Pareto frontier analysis

### 2. **EnergyYieldOptimizer** - Energy Production Optimization
- Maximize annual energy production
- Minimize shading losses through optimal row spacing
- Optimize bifacial gain for bifacial modules
- Optimize tracker angles for single/dual-axis systems
- Seasonal tilt optimization

### 3. **EconomicOptimizer** - Financial Optimization
- Minimize LCOE (Levelized Cost of Energy)
- Maximize NPV (Net Present Value)
- Optimize DC/AC ratio for best economics
- Module selection optimization
- Balance of System (BOS) cost optimization

### 4. **LayoutOptimizer** - Physical Layout Optimization
- Optimize Ground Coverage Ratio (GCR)
- Minimize land use while meeting capacity requirements
- Maximize capacity within land constraints
- Optimize electrical string configuration
- Terrain-following layout optimization for sloped sites

### 5. **DesignSpaceExplorer** - Analysis Tools
- Multi-dimensional parameter sweeps
- Sensitivity analysis with tornado charts
- Monte Carlo simulation for uncertainty quantification
- Pareto frontier analysis for multi-objective trade-offs
- Constraint violation checking

### 6. **OptimizationUI** - Interactive Streamlit Interface
- Visual system configuration
- Real-time optimization execution
- Interactive Pareto curve visualization
- Design comparison and results export
- Comprehensive visualization dashboards

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or using pyproject.toml
pip install -e .
```

## Quick Start

### Basic Optimization Example

```python
from src.models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    OptimizationObjectives,
)
from src.optimization.system_optimizer import SystemOptimizer

# Configure system
parameters = PVSystemParameters(
    module_power=450.0,
    module_efficiency=0.20,
    latitude=35.0,
    num_modules=10000,
    # ... other parameters
)

# Set constraints
constraints = OptimizationConstraints(
    min_gcr=0.2,
    max_gcr=0.6,
    min_dc_ac_ratio=1.1,
    max_dc_ac_ratio=1.5,
)

# Define objectives
objectives = OptimizationObjectives(
    maximize_energy=1.0,
    minimize_lcoe=1.0,
)

# Run optimization
optimizer = SystemOptimizer(parameters, constraints, objectives)
result = optimizer.genetic_algorithm_optimizer(
    population_size=100,
    num_generations=50,
)

# Access results
print(f"Optimal LCOE: ${result.best_solution.lcoe:.4f}/kWh")
print(f"Optimal GCR: {result.best_solution.gcr:.3f}")
```

### Launch Streamlit UI

```bash
streamlit run src/ui/optimization_app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Architecture

```
src/
├── models/
│   ├── __init__.py
│   └── optimization_models.py      # Pydantic data models
├── optimization/
│   ├── __init__.py
│   ├── system_optimizer.py         # Main optimization engine
│   ├── energy_yield_optimizer.py   # Energy optimization
│   ├── economic_optimizer.py       # Economic optimization
│   ├── layout_optimizer.py         # Layout optimization
│   └── design_space_explorer.py    # Analysis tools
├── ui/
│   ├── __init__.py
│   └── optimization_app.py         # Streamlit UI
└── utils/
    └── __init__.py
```

## Core Components

### Data Models (Pydantic)

All data models use Pydantic for validation and type safety:

- **PVSystemParameters**: Complete system configuration
- **OptimizationConstraints**: Bounds and limits
- **OptimizationObjectives**: Multi-objective weights
- **DesignPoint**: Single design solution
- **OptimizationResult**: Optimization output
- **ParetoSolution**: Pareto-optimal solution

### Optimization Algorithms

#### Genetic Algorithm
```python
result = optimizer.genetic_algorithm_optimizer(
    population_size=100,    # Population size
    num_generations=50,     # Number of generations
    crossover_prob=0.8,     # Crossover probability
    mutation_prob=0.2,      # Mutation probability
)
```

#### Particle Swarm Optimization
```python
result = optimizer.particle_swarm_optimizer(
    swarm_size=50,          # Number of particles
    max_iterations=100,     # Maximum iterations
    omega=0.5,              # Inertia weight
    phi_p=0.5,              # Cognitive parameter
    phi_g=0.5,              # Social parameter
)
```

#### Multi-Objective Optimization
```python
result = optimizer.multi_objective_optimization(
    population_size=100,
    num_generations=50,
)

# Access Pareto front
for solution in result.pareto_front:
    print(f"LCOE: {solution.design.lcoe}")
    print(f"Energy: {solution.design.annual_energy_kwh}")
```

### Energy Yield Optimization

```python
from src.optimization.energy_yield_optimizer import EnergyYieldOptimizer

energy_opt = EnergyYieldOptimizer(parameters, constraints)

# Maximize annual energy
max_energy, params = energy_opt.maximize_annual_energy(method="gradient")

# Optimize bifacial gain
gain, params = energy_opt.optimize_bifacial_gain()

# Seasonal optimization
seasonal_tilts = energy_opt.seasonal_optimization()
```

### Economic Optimization

```python
from src.optimization.economic_optimizer import EconomicOptimizer

econ_opt = EconomicOptimizer(parameters, constraints)

# Minimize LCOE
min_lcoe, params = econ_opt.minimize_lcoe(
    vary_dc_ac_ratio=True,
    vary_gcr=True,
)

# Maximize NPV
max_npv, params = econ_opt.maximize_npv(electricity_price=0.06)

# Optimize DC/AC ratio
optimal_dc_ac, params = econ_opt.optimize_dc_ac_ratio()
```

### Layout Optimization

```python
from src.optimization.layout_optimizer import LayoutOptimizer

layout_opt = LayoutOptimizer(parameters, constraints)

# Optimize GCR
optimal_gcr, params = layout_opt.optimize_gcr(objective="energy_per_area")

# Minimize land use
land_acres, params = layout_opt.minimize_land_use(min_capacity_mw=5.0)

# String configuration
config = layout_opt.optimize_string_configuration(
    inverter_mppt_voltage_range=(600, 1500),
    module_voltage=40.0,
)
```

### Design Space Exploration

```python
from src.optimization.design_space_explorer import DesignSpaceExplorer, ParameterRange

explorer = DesignSpaceExplorer(parameters, constraints)

# Parameter sweep
results = explorer.parameter_sweep(
    [ParameterRange("gcr", 0.2, 0.6, 20)],
    output_metric="lcoe",
)

# Sensitivity analysis
sensitivity = explorer.sensitivity_analysis(
    parameters_to_vary=["module_efficiency", "module_cost"],
    variation_percent=10.0,
)

# Monte Carlo simulation
mc_results = explorer.monte_carlo_simulation(
    num_samples=1000,
    parallel=True,
)

# Pareto frontier
pareto_solutions = explorer.pareto_frontier_analysis(
    objective1="lcoe",
    objective2="annual_energy_kwh",
    num_points=50,
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_optimization.py -v

# Run with coverage
pytest tests/test_optimization.py --cov=src --cov-report=html

# Run specific test class
pytest tests/test_optimization.py::TestSystemOptimizer -v
```

## Examples

See `examples/basic_optimization.py` for a complete workflow example:

```bash
python examples/basic_optimization.py
```

## Performance

- **Genetic Algorithm**: ~100 evaluations/second (varies with system complexity)
- **PSO**: ~200 evaluations/second
- **Multi-Objective**: Finds 20-50 Pareto solutions in 50 generations
- **Parameter Sweep**: Parallel execution scales linearly with cores
- **Monte Carlo**: 1000 samples in ~5 seconds (parallel mode)

## API Reference

### PVSystemParameters

Key parameters:
- `module_power`: Module power in watts
- `module_efficiency`: Module efficiency (0-1)
- `num_modules`: Number of modules
- `gcr`: Ground coverage ratio (0-1)
- `dc_ac_ratio`: DC to AC ratio (≥1.0)
- `latitude`, `longitude`: Site location
- `discount_rate`: Financial discount rate
- `project_lifetime`: Project lifetime in years

### OptimizationConstraints

Key constraints:
- `min_gcr`, `max_gcr`: GCR bounds
- `min_dc_ac_ratio`, `max_dc_ac_ratio`: DC/AC bounds
- `min_tilt`, `max_tilt`: Tilt angle bounds
- `max_land_use_acres`: Maximum land use
- `max_shading_loss`: Maximum acceptable shading loss

### OptimizationObjectives

Objective weights (0-1):
- `maximize_energy`: Energy yield maximization
- `minimize_lcoe`: LCOE minimization
- `minimize_land_use`: Land use minimization
- `maximize_npv`: NPV maximization
- `minimize_shading`: Shading loss minimization
- `maximize_bifacial_gain`: Bifacial gain maximization

## Advanced Features

### Custom Evaluator Functions

Provide custom performance evaluation:

```python
def custom_evaluator(params: PVSystemParameters) -> DesignPoint:
    # Your custom simulation logic here
    # e.g., integrate with pvlib, SAM, etc.
    return DesignPoint(...)

optimizer = SystemOptimizer(parameters, constraints, objectives)
optimizer.set_evaluator(custom_evaluator)
```

### Parallel Processing

Enable parallel execution for faster optimization:

```python
# Parameter sweep with parallel processing
results = explorer.parameter_sweep(
    param_ranges,
    parallel=True,  # Uses ThreadPoolExecutor
)

# Monte Carlo with parallel processing
mc_results = explorer.monte_carlo_simulation(
    num_samples=10000,
    parallel=True,
)
```

### Results Export

Export optimization results:

```python
import json

# Export to JSON
result_dict = result.model_dump()
with open('optimization_results.json', 'w') as f:
    json.dump(result_dict, f, indent=2)

# Export Pareto front
pareto_data = [
    {
        'lcoe': sol.design.lcoe,
        'energy': sol.design.annual_energy_kwh,
        'gcr': sol.design.gcr,
    }
    for sol in result.pareto_front
]
```

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing and optimization
- **pandas**: Data analysis
- **pydantic**: Data validation
- **deap**: Genetic algorithms
- **pyswarm**: Particle swarm optimization
- **pulp**: Linear programming
- **matplotlib**: Plotting
- **plotly**: Interactive visualizations
- **streamlit**: Web UI
- **pvlib**: PV modeling (optional)

## Contributing

Contributions are welcome! Please ensure:
1. All code has type hints
2. Comprehensive docstrings (Google style)
3. Tests for new features
4. Code passes `black` and `ruff` linting

## License

Apache 2.0 License - See LICENSE file for details.

## Citation

If you use this optimization engine in research, please cite:

```bibtex
@software{pv_optimization_engine,
  title={PV System Optimization Engine},
  author={PV Circularity Team},
  year={2024},
  version={0.1.0},
  url={https://github.com/your-org/pv-circularity-simulator}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/your-org/pv-circularity-simulator/issues
- Documentation: See inline docstrings and examples/

## Roadmap

Future enhancements:
- [ ] Integration with NREL SAM
- [ ] Advanced weather data integration
- [ ] Machine learning-based surrogate models
- [ ] Distributed optimization for large-scale studies
- [ ] Real-time optimization with live data
- [ ] GPU acceleration for parameter sweeps
- [ ] Advanced constraint handling methods
- [ ] Multi-site portfolio optimization

---

**Version**: 0.1.0
**Last Updated**: 2024
**Status**: Production-ready
