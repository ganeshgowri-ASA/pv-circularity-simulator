# Hydrogen System Integration & Power-to-X Module

Production-ready hydrogen system modeling for renewable energy integration with comprehensive electrolyzer, storage, fuel cell, and power-to-X analysis capabilities.

## Features

### 1. **Electrolyzer Modeling** (`electrolyzer_modeling`)
- Multiple technologies: PEM, Alkaline, SOEC, AEM
- Dynamic part-load efficiency curves
- Temperature effects on performance
- Stack degradation modeling
- Economic analysis (LCOH - Levelized Cost of Hydrogen)
- Start-stop cycle tracking

### 2. **H2 Storage Design** (`h2_storage_design`)
- Storage types: Compressed gas, Liquid H2, Metal hydride, LOHC, Underground
- State of charge (SOC) dynamics
- Charging/discharging efficiency modeling
- Self-discharge losses
- Cycle counting and analysis
- Storage economics (LCOS)

### 3. **Fuel Cell Integration** (`fuel_cell_integration`)
- Technologies: PEMFC, SOFC, MCFC, AFC, PAFC
- Part-load efficiency characteristics
- Combined Heat and Power (CHP) operation
- Stack degradation modeling
- H2 consumption tracking
- Economic analysis (LCOE - Levelized Cost of Electricity)

### 4. **Power-to-X Analysis** (`power_to_x_analysis`)
- Pathways:
  - Power-to-H2 (direct hydrogen)
  - Power-to-Methane (Sabatier reaction)
  - Power-to-Methanol (synthesis)
  - Power-to-Ammonia (Haber-Bosch)
  - Power-to-Liquid (Fischer-Tropsch)
  - Power-to-SNG (synthetic natural gas)
- Mass and energy balance calculations
- CO2 and N2 requirement tracking
- Economic analysis (LCOP - Levelized Cost of Product)
- Environmental impact (carbon intensity)
- Avoided emissions calculation

## Installation

```bash
# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with analysis tools
pip install -e ".[analysis]"
```

## Quick Start

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
    project_lifetime_years=25,
)

# Configure electrolyzer
config = ElectrolyzerConfig(
    electrolyzer_type=ElectrolyzerType.PEM,
    rated_power_kw=1000.0,
    efficiency=0.68,
)

# Simulate operation with power profile
power_profile = [800.0] * 8760  # 1 year hourly data
results = integrator.electrolyzer_modeling(
    config=config,
    power_input_profile=power_profile,
    timestep_hours=1.0,
)

print(f"Annual H2 production: {results.annual_h2_production_kg:.2f} kg")
print(f"LCOH: ${results.levelized_cost_h2:.2f}/kg")
```

## Examples

See `/examples/hydrogen_integration_example.py` for comprehensive examples covering:
- Variable renewable power electrolyzer operation
- Storage system design with charge/discharge cycles
- Fuel cell CHP integration
- Power-to-Methanol pathway analysis

Run examples:
```bash
python examples/hydrogen_integration_example.py
```

## Architecture

### Pydantic Models (`models.py`)
All configurations and results use Pydantic v2 for:
- Type safety and validation
- Computed properties (e.g., `h2_production_rate_kg_h`)
- Clear documentation
- Easy serialization/deserialization

### Core Integrator (`integrator.py`)
The `HydrogenIntegrator` class provides four main methods:

1. **`electrolyzer_modeling()`**
   - Inputs: Configuration, power profile, timestep, optional temperature
   - Outputs: H2 production, efficiency, degradation, LCOH
   - Accounts for: Part-load curves, degradation, temperature effects

2. **`h2_storage_design()`**
   - Inputs: Configuration, charge/discharge profiles, timestep, initial SOC
   - Outputs: SOC dynamics, throughput, losses, cycling, LCOS
   - Accounts for: Efficiency losses, self-discharge, rate limits

3. **`fuel_cell_integration()`**
   - Inputs: Configuration, power demand, optional H2 supply, heat demand
   - Outputs: Electricity, heat, H2 consumption, degradation, LCOE
   - Accounts for: Part-load efficiency, CHP operation, H2 availability

4. **`power_to_x_analysis()`**
   - Inputs: Configuration, power profile, CO2 availability
   - Outputs: Product output, economics, environmental metrics
   - Accounts for: Full pathway efficiency, stoichiometry, economics

## Technical Specifications

### Electrolyzer Types
| Type | Typical Efficiency | Operating Temp | Pressure | Maturity |
|------|-------------------|----------------|----------|----------|
| PEM | 65-70% | 50-80°C | 30-80 bar | Commercial |
| Alkaline | 60-70% | 60-80°C | 1-30 bar | Mature |
| SOEC | 75-85% | 650-850°C | 1-25 bar | Demonstration |
| AEM | 60-68% | 40-60°C | 1-30 bar | Emerging |

### Storage Types
| Type | Energy Density | Efficiency | Maturity | Use Case |
|------|---------------|------------|----------|----------|
| Compressed Gas | Low | 90-95% | Mature | Short-medium term |
| Liquid H2 | High | 70-80% | Commercial | Large scale |
| Metal Hydride | Medium | 85-90% | Niche | Stationary |
| LOHC | High | 75-85% | Demonstration | Transport |
| Underground | Very High | 85-95% | Emerging | Seasonal |

### Fuel Cell Types
| Type | Efficiency | Operating Temp | Applications |
|------|-----------|----------------|--------------|
| PEMFC | 50-60% | 60-80°C | Transport, stationary |
| SOFC | 55-65% | 650-1000°C | CHP, distributed |
| MCFC | 50-60% | 600-650°C | Stationary power |
| AFC | 60-70% | 60-90°C | Space, niche |
| PAFC | 40-50% | 150-200°C | Stationary CHP |

## Validation & Testing

The module includes comprehensive validation:
- Pydantic field validators for all inputs
- Physical constraint checks (efficiency bounds, etc.)
- Mass and energy balance verification
- Economic calculation validation

Run tests:
```bash
pytest tests/
```

## Performance Considerations

- **Timestep Selection**: Hourly (1.0) is standard, but sub-hourly (0.25, 0.5) captures fast dynamics
- **Profile Length**: Minimum 8760 hours (1 year) recommended for annual metrics
- **Memory**: O(n) where n = profile length; handles multi-year simulations efficiently
- **Computation**: Fast; typical 1-year simulation < 1 second

## Economic Calculations

All levelized cost calculations use standard financial formulas:

**LCOH (Levelized Cost of Hydrogen):**
```
LCOH = (Annual Capital Cost + Annual OPEX + Annual Electricity + Stack Replacement) / Annual H2 Production
```

**Annuity Factor:**
```
AF = r(1+r)^n / ((1+r)^n - 1)
```
where r = discount rate, n = project lifetime

## Environmental Metrics

**Carbon Intensity:**
```
CI = (Electricity Emissions - CO2 Utilized) / Product Output
```

**Avoided Emissions:**
```
AE = (Fossil Baseline - Green Product) × Product Output
```

## API Reference

Full API documentation available in docstrings. Key classes:

- `HydrogenIntegrator`: Main integration class
- `ElectrolyzerConfig`: Electrolyzer parameters
- `ElectrolyzerResults`: Electrolyzer outputs
- `StorageConfig`: Storage parameters
- `StorageResults`: Storage outputs
- `FuelCellConfig`: Fuel cell parameters
- `FuelCellResults`: Fuel cell outputs
- `PowerToXConfig`: P2X pathway parameters
- `PowerToXResults`: P2X outputs

## Contributing

This module follows production-ready best practices:
- Full type hints (Python 3.9+)
- Comprehensive docstrings (Google style)
- Pydantic validation
- Unit tests with pytest
- Code formatting with black
- Linting with ruff

## License

MIT License - see LICENSE file

## References

- Hydrogen production: IEA Hydrogen Reports
- Storage technologies: DOE Hydrogen Storage Program
- Fuel cells: US DOE Fuel Cell Technologies Office
- Power-to-X: IRENA Innovation Landscape Reports
- Economics: NREL H2@Scale studies

## Support

For issues, feature requests, or questions, please open an issue on the repository.
