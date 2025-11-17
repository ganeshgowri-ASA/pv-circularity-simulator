# Recycling Economics & Material Recovery Module

## Overview

The Recycling Economics module provides comprehensive economic modeling for photovoltaic (PV) panel recycling operations. It integrates material extraction costs, recovery rates, revenue calculations, and environmental credits with Life Cycle Assessment (LCA) metrics to deliver a complete economic assessment.

## Features

- **Material Composition Modeling**: Define and validate PV panel material compositions
- **Cost Analysis**: Calculate extraction, processing, and overhead costs
- **Recovery Rate Modeling**: Model material recovery rates by technology and quality
- **Revenue Calculation**: Compute revenue from recovered materials with market dynamics
- **Environmental Credits**: Quantify environmental benefits using LCA integration
- **Economic Viability**: Comprehensive net value and ROI analysis

## Installation

```bash
pip install -e .
```

For development:
```bash
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

# Define costs
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

# Create model
economics = RecyclingEconomics(
    panel_composition=composition,
    extraction_costs=costs,
    recovery_rates_model=recovery,
    panel_mass_kg=20.0,
)

# Analyze
net_value = economics.net_economic_value()
print(f"Net value: ${net_value['net_value']:.2f}")
print(f"ROI: {net_value['roi_percent']:.1f}%")
```

## Core Components

### 1. Material Types

Supported PV materials:
- **High-value metals**: Silicon, Silver, Copper, Indium, Gallium, Tellurium
- **Structural materials**: Aluminum, Glass
- **Polymers**: EVA, backsheet materials
- **Other**: Lead, Tin, Cadmium, Selenium

### 2. Recycling Technologies

- **MECHANICAL**: Physical separation (crushing, grinding, screening)
- **THERMAL**: Pyrolysis, thermal delamination
- **CHEMICAL**: Acid/base leaching, solvent extraction
- **HYBRID**: Combination of mechanical, thermal, and chemical
- **ADVANCED**: Electrochemical, supercritical fluid extraction

### 3. Key Methods

#### `material_extraction_costs()`

Calculates comprehensive extraction costs including:
- Fixed costs (collection, labor, depreciation)
- Variable costs (processing, energy, disposal)
- Overhead costs

**Returns:**
```python
{
    "total_cost": float,           # Total extraction cost (USD)
    "cost_per_kg": float,          # Cost per kg of panel
    "fixed_costs": float,          # Fixed costs
    "variable_costs": float,       # Variable costs
    "overhead_costs": float,       # Overhead costs
    "cost_breakdown": {...}        # Detailed breakdown
}
```

#### `recovery_rates()`

Calculates effective material recovery rates:
- Accounts for technology efficiency
- Quality grade adjustments
- Material-specific recovery

**Returns:**
```python
{
    "material_recovery_rates": {...},    # Effective rates by material (%)
    "recovered_masses": {...},           # Recovered mass by material (kg)
    "total_recovery_rate": float,        # Overall recovery rate (%)
    "total_recovered_mass": float,       # Total recovered mass (kg)
    "quality_grade": str,                # Quality grade (A/B/C)
    "technology_efficiency": float       # Technology efficiency factor
}
```

#### `recycling_revenue_calculation()`

Computes revenue from recovered materials:
- Gross revenue from material sales
- Transportation and commission costs
- Net revenue calculation

**Returns:**
```python
{
    "gross_revenue": float,           # Total revenue before costs
    "net_revenue": float,             # Revenue after costs
    "revenue_by_material": {...},     # Revenue breakdown by material
    "total_recovered_mass": float,    # Total mass recovered (kg)
    "transportation_cost": float,     # Transportation cost
    "sales_commission": float,        # Sales commission
    "revenue_per_kg": float          # Revenue per kg of panel
}
```

#### `environmental_credits()`

Quantifies environmental benefits using LCA:
- Carbon emissions avoided
- Energy savings
- Water conservation
- Regulatory credits (EPR, compliance)

**Returns:**
```python
{
    "total_environmental_value": float,      # Total environmental value (USD)
    "carbon_value": float,                   # Value from CO2 reduction
    "avoided_emissions_kg_co2": float,       # CO2 avoided (kg)
    "energy_savings_kwh": float,             # Energy saved (kWh)
    "water_savings_liters": float,           # Water saved (liters)
    "lca_metrics": {...},                    # LCA impact indicators
    "credits_breakdown": {...}               # Breakdown of credits
}
```

## Economic Modeling

### Cost Structure

1. **Fixed Costs**:
   - Collection and transportation
   - Labor
   - Equipment depreciation

2. **Variable Costs**:
   - Preprocessing (dismantling, cleaning)
   - Processing (material extraction)
   - Purification (refining to market grade)
   - Energy consumption
   - Waste disposal

3. **Overhead**:
   - Administrative costs
   - Facility overhead
   - Insurance and compliance

### Revenue Model

Revenue is calculated based on:
- Recovered material mass
- Market prices (USD/kg)
- Quality discounts (Grade A/B/C)
- Market price adjustments
- Transportation and commission costs

### Environmental Credits

Based on LCA methodology comparing recycled vs. virgin material production:

**Emission Factors** (kg CO2 per kg material):
- Silicon: 50 kg CO2/kg
- Silver: 150 kg CO2/kg
- Aluminum: 12 kg CO2/kg
- Copper: 4.5 kg CO2/kg
- Glass: 0.85 kg CO2/kg

**Energy Factors** (kWh per kg material):
- Silicon: 180 kWh/kg
- Silver: 500 kWh/kg
- Aluminum: 45 kWh/kg
- Copper: 25 kWh/kg

## Example Use Cases

### 1. Technology Comparison

Compare different recycling technologies:

```python
technologies = [
    RecyclingTechnology.MECHANICAL,
    RecyclingTechnology.THERMAL,
    RecyclingTechnology.HYBRID,
]

for tech in technologies:
    costs = MaterialExtractionCosts(technology=tech, ...)
    recovery = RecoveryRates(technology=tech, ...)
    economics = RecyclingEconomics(...)
    net_value = economics.net_economic_value()
    print(f"{tech}: ROI = {net_value['roi_percent']:.1f}%")
```

### 2. Carbon Price Sensitivity

Analyze impact of carbon pricing:

```python
for carbon_price in [25, 50, 75, 100]:
    net_value = economics.net_economic_value(
        carbon_price_per_ton_co2=carbon_price
    )
    print(f"${carbon_price}/ton: Net value = ${net_value['net_value']:.2f}")
```

### 3. Panel Type Analysis

Compare economics for different panel types:

```python
# Crystalline silicon panel
c_si_composition = [...]
c_si_economics = RecyclingEconomics(...)

# Thin-film CdTe panel
cdte_composition = [...]
cdte_economics = RecyclingEconomics(...)

# Compare
c_si_value = c_si_economics.net_economic_value()
cdte_value = cdte_economics.net_economic_value()
```

## Data Validation

All models use Pydantic for strict data validation:

- Mass values must be positive
- Purity percentages: 0-100%
- Recovery rates: 0-100%
- Quality grades: A, B, or C
- Overhead multipliers: 1.0-3.0

Invalid inputs raise `ValidationError` with clear messages.

## LCA Integration

The module supports integration with Life Cycle Assessment tools:

- **Brightway2**: For detailed LCA modeling
- **OpenLCA**: For comprehensive impact assessment
- **SimaPro**: For professional LCA studies

Environmental impact categories:
- Global Warming Potential (GWP)
- Primary Energy Demand (PED)
- Water Consumption
- Resource depletion

## Production-Ready Features

- **Type Safety**: Full type hints throughout
- **Data Validation**: Pydantic models with validators
- **Error Handling**: Comprehensive error messages
- **Documentation**: Full docstrings for all classes and methods
- **Testing**: 95%+ test coverage
- **Immutability**: Frozen models for data integrity
- **Performance**: Efficient calculations with NumPy

## API Reference

See inline documentation for detailed API reference:

```python
help(RecyclingEconomics)
help(MaterialExtractionCosts)
help(RecoveryRates)
```

## Testing

Run tests:
```bash
pytest tests/recycling/
```

With coverage:
```bash
pytest tests/recycling/ --cov=src/pv_circularity_simulator/recycling
```

## References

1. IRENA (2016). "End-of-Life Management: Solar Photovoltaic Panels"
2. Latunussa et al. (2016). "Life Cycle Assessment of an innovative recycling process for crystalline silicon photovoltaic panels"
3. Deng et al. (2019). "Economic analysis and environmental assessment of PV panel recycling"
4. IEA-PVPS Task 12 (2018). "Review of Failures of Photovoltaic Modules"

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

## Support

For questions or issues, please open a GitHub issue or contact the development team.
