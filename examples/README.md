# RepowerAnalyzer Examples

This directory contains example scripts demonstrating how to use the RepowerAnalyzer for PV system repower feasibility analysis.

## Running the Examples

### Prerequisites

First, install the package and its dependencies:

```bash
# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Repower Analysis Example

The main example demonstrates a complete repower analysis workflow:

```bash
python examples/repower_analysis_example.py
```

This example shows:

1. **Capacity Upgrade Analysis**: Determine maximum possible capacity increase
2. **Component Replacement Planning**: Prioritize component replacements
3. **Technical Feasibility Assessment**: Evaluate technical constraints
4. **Economic Viability Analysis**: Compare multiple repower scenarios

## Example Output

The example will produce a comprehensive analysis report including:

- Current system status and performance
- Upgrade potential and limiting factors
- Component replacement timeline and costs
- Technical feasibility scores across multiple dimensions
- Economic metrics (NPV, IRR, ROI, LCOE, payback period)
- Sensitivity analysis for key variables
- Break-even conditions
- Financing options
- Final recommendations

## Customizing the Analysis

You can modify the example to analyze your own system by:

1. Updating the system parameters in `create_example_system()`
2. Adjusting the analyzer configuration in `RepowerAnalyzerConfig`
3. Creating custom repower scenarios with different strategies
4. Modifying incentive structures and electricity rates

## Quick Start

Here's a minimal example to get started:

```python
from pv_simulator import RepowerAnalyzer
from pv_simulator.core.models import PVSystem, Location, PVModule

# Create your system
system = PVSystem(...)

# Initialize analyzer
analyzer = RepowerAnalyzer()

# Analyze capacity upgrade potential
capacity = analyzer.capacity_upgrade_analysis(system)
print(f"Max upgrade: {capacity.max_additional_capacity} kW")

# Plan component replacements
replacements = analyzer.component_replacement_planning(system)
print(f"Total replacement cost: ${replacements.total_replacement_cost}")

# Check technical feasibility
feasibility = analyzer.technical_feasibility_check(
    system,
    target_capacity=120.0
)
print(f"Feasible: {feasibility.is_feasible}")

# Analyze economics
scenarios = [...]  # Define your scenarios
economics = analyzer.economic_viability_analysis(
    system,
    repower_scenarios=scenarios
)
print(f"Best NPV: ${economics.best_scenario.economic_metrics.npv}")
```

## Additional Resources

- See the test suite in `tests/test_repower_analyzer.py` for more usage examples
- Refer to the API documentation for detailed method descriptions
- Check the Pydantic models in `src/pv_simulator/core/models.py` for all available fields
