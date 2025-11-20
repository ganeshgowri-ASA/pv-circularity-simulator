# PV Circularity Simulator

**End-to-end PV lifecycle simulation platform with comprehensive circular economy modeling**

A production-ready Python package for analyzing the circular economy aspects of photovoltaic (PV) modules, including material recovery, reuse assessment, repair optimization, recycling economics, life cycle assessment, and interactive dashboards.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The PV Circularity Simulator implements the complete 3R framework (Reduce, Reuse, Recycle) for photovoltaic modules, enabling comprehensive analysis of circular economy opportunities in the solar industry.

### Key Features

- **Material Recovery Analysis**: Calculate recovery rates and costs for metals, glass, and silicon
- **Reuse Assessment**: Evaluate modules for second-life applications with technical and economic analysis
- **Repair Optimization**: Prioritize defects and optimize repair vs. replace decisions
- **Recycling Economics**: Analyze material value, ROI, and environmental credits
- **Life Cycle Assessment**: Comprehensive carbon footprint and environmental impact analysis
- **Interactive Dashboards**: Streamlit-based visualizations with material flow diagrams and 3R metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from src.circularity import MaterialRecoveryCalculator, ReuseAnalyzer, LCAAnalyzer
from src.circularity.material_recovery import ModuleComposition

# Define a PV module
module = ModuleComposition(
    glass=15.0, aluminum=2.5, silicon=0.5,
    silver=0.005, copper=0.2, eva_polymer=1.0,
    backsheet=0.5, junction_box=0.3
)

# Analyze material recovery
calculator = MaterialRecoveryCalculator()
results = calculator.full_recovery_analysis(
    composition=module,
    num_modules=1000,
    transport_distance_km=200.0
)

print(f"Recovery rate: {results['overall_recovery_rate']*100:.1f}%")
print(f"Total cost: ${results['total_cost_usd']:,.2f}")
```

## Components

### B11-S01: Material Recovery & Recycling Economics

Calculate material recovery rates and recycling costs for end-of-life PV modules.

```python
from src.circularity import MaterialRecoveryCalculator

calculator = MaterialRecoveryCalculator()

# Metal recovery
metal_recovery = calculator.metal_recovery(module, recovery_method="combined")

# Glass recovery
glass_recovery = calculator.glass_recovery(module, processing_quality="standard")

# Silicon recovery
silicon_recovery = calculator.silicon_recovery(module, recovery_technique="thermal_chemical")

# Recycling costs
costs = calculator.recycling_costs(module, num_modules=100, transport_distance_km=200.0)
```

**Key Methods:**
- `metal_recovery()`: Analyze metal recovery (aluminum, silver, copper)
- `glass_recovery()`: Calculate glass cullet recovery and quality
- `silicon_recovery()`: Determine silicon recovery by grade
- `recycling_costs()`: Detailed cost breakdown for recycling operations

### B11-S02: Reuse Assessment & Second-Life Applications

Evaluate PV modules for reuse potential and second-life market opportunities.

```python
from src.circularity import ReuseAnalyzer
from src.circularity.reuse_analyzer import ModuleTestResults, ModuleCondition

analyzer = ReuseAnalyzer()

# Module testing
test_results = ModuleTestResults(
    visual_inspection_passed=True,
    electrical_test_passed=True,
    insulation_test_passed=True,
    current_power_w=340,
    rated_power_w=400,
    voltage_v=32.5,
    current_a=10.5,
    fill_factor=0.78,
    insulation_resistance_mohm=50.0,
    defects=[],
    condition=ModuleCondition.GOOD
)

# Assess eligibility
eligibility = analyzer.module_testing(test_results, age_years=8)

# Find second-life markets
markets = analyzer.second_life_markets(
    capacity_retention=0.85,
    available_quantity_kw=340.0,
    module_specs={"voltage": 32.5, "current": 10.5}
)
```

**Key Methods:**
- `module_testing()`: Evaluate reuse eligibility based on test results
- `residual_capacity()`: Analyze remaining capacity and lifetime
- `second_life_markets()`: Identify market opportunities for reused modules

### B11-S03: Repair Optimization & Maintenance Strategies

Optimize repair decisions and plan preventive maintenance schedules.

```python
from src.circularity import RepairOptimizer
from src.circularity.repair_optimizer import Defect, DefectSeverity

optimizer = RepairOptimizer()

# Define defects
defects = [
    Defect(
        defect_id="D001",
        defect_type="junction_box_failure",
        severity=DefectSeverity.HIGH,
        location="Module A1-15",
        power_loss_w=25.0,
        safety_risk=True
    )
]

# Prioritize defects
prioritized = optimizer.defect_prioritization(defects, system_size_kw=100.0, age_years=10.0)

# Repair vs replace decision
decision = optimizer.repair_vs_replace(
    module_age_years=10,
    current_power_w=340,
    rated_power_w=400,
    repair_costs=repair_costs,
    defects=defects
)

# Maintenance schedule
schedule = optimizer.preventive_maintenance(
    system_size_kw=100.0,
    system_age_years=5.0,
    climate_zone="temperate"
)
```

**Key Methods:**
- `defect_prioritization()`: Rank defects by urgency and impact
- `repair_vs_replace()`: Economic analysis of repair vs. replacement
- `preventive_maintenance()`: Generate maintenance schedules

### B11-S04: Recycling Economics & Material Value

Analyze the economics of recycling operations with material pricing and ROI calculations.

```python
from src.circularity import RecyclingEconomics

economics = RecyclingEconomics()

# Material pricing
revenue = economics.material_pricing(recovered_materials)

# ROI analysis
roi = economics.recycling_roi(
    num_modules=10000,
    avg_module_weight_kg=20.0,
    recycling_cost_per_module=15.0,
    recovered_materials=recovered_materials
)

# Environmental credits
env_credits = economics.environmental_credits(
    num_modules=10000,
    avg_module_weight_kg=20.0,
    region="EU"
)
```

**Key Methods:**
- `material_pricing()`: Calculate revenue from recovered materials
- `recycling_roi()`: Comprehensive ROI analysis
- `environmental_credits()`: Value environmental benefits

### B11-S05: Environmental Impact & LCA Analysis

Comprehensive life cycle assessment with carbon footprint and environmental indicators.

```python
from src.circularity import LCAAnalyzer

lca = LCAAnalyzer()

# Carbon footprint
carbon = lca.carbon_footprint(
    module_power_w=400,
    module_weight_kg=20.0,
    manufacturing_location="China",
    recycling_at_eol=True
)

# Energy payback
energy = lca.energy_payback(
    module_power_w=400,
    module_weight_kg=20.0,
    annual_irradiation_kwh_per_m2=1800,
    module_area_m2=2.0
)

# Environmental indicators
indicators = lca.environmental_indicators(
    module_power_w=400,
    module_weight_kg=20.0
)
```

**Key Methods:**
- `carbon_footprint()`: Detailed lifecycle CO2eq emissions
- `energy_payback()`: Energy payback time and EROI
- `environmental_indicators()`: Comprehensive impact assessment

### B11-S06: Circularity UI & 3R Dashboard

Interactive visualizations and dashboards for circular economy analysis.

```python
from src.circularity import CircularityUI

ui = CircularityUI()

# Material flow Sankey diagram
flow_fig = ui.material_flow_diagrams(
    input_materials={"glass": 15.0, "aluminum": 2.5},
    recovered_materials={"glass": 14.0, "aluminum": 2.4},
    waste_materials={"mixed": 1.2}
)

# 3R metrics dashboard
metrics_fig = ui.three_r_metrics(metrics, show_details=True)

# Circular economy score
ce_score = ui.circular_economy_score(
    material_circularity_index=0.75,
    recovery_rate=0.85,
    reuse_rate=0.20,
    lifetime_extension_factor=1.3,
    carbon_footprint_kg=1500,
    roi_percent=12.5
)
```

**Key Methods:**
- `material_flow_diagrams()`: Create Sankey diagrams for material flows
- `3R_metrics()`: Visualize Reduce, Reuse, Recycle metrics
- `circular_economy_score()`: Calculate comprehensive circularity score

## Examples

See the `examples/` directory for detailed usage examples:

```bash
# Run basic usage examples
python examples/basic_usage.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_material_recovery.py
```

## Architecture

```
pv-circularity-simulator/
├── src/circularity/
│   ├── __init__.py
│   ├── material_recovery.py      # B11-S01: Material Recovery
│   ├── reuse_analyzer.py          # B11-S02: Reuse Assessment
│   ├── repair_optimizer.py        # B11-S03: Repair Optimization
│   ├── recycling_economics.py     # B11-S04: Recycling Economics
│   ├── lca_analyzer.py            # B11-S05: LCA Analysis
│   └── circularity_ui.py          # B11-S06: UI & Dashboard
├── tests/
│   ├── test_material_recovery.py
│   └── test_integration.py
├── examples/
│   └── basic_usage.py
└── pyproject.toml
```

## Technology Stack

- **Core**: Python 3.9+
- **Data Validation**: Pydantic 2.0+
- **Numerical Computing**: NumPy, SciPy, Pandas
- **Visualization**: Plotly, Matplotlib
- **Dashboard**: Streamlit
- **Testing**: pytest, pytest-cov

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator: Circular Economy Analysis for Photovoltaic Modules},
  author = {PV Circularity Team},
  year = {2024},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Acknowledgments

- Built with production-ready standards for circular economy analysis
- Implements best practices from circular economy frameworks (Ellen MacArthur Foundation)
- Based on industry standards for PV recycling and life cycle assessment

## Contact

For questions or support, please open an issue on GitHub.
