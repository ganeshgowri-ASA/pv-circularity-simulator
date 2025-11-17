# BOM Generator & Cost Calculator

Production-ready Bill of Materials (BOM) generator and cost calculator for photovoltaic module manufacturing.

## Features

### Core Functionality
- **BOM Generation**: Automatically generate comprehensive BOMs from module design specifications
- **Cost Calculation**: Calculate material costs with waste factors and volume-based pricing tiers
- **Multi-Currency Support**: Handle multiple currencies with automatic conversion
- **Supplier Comparison**: Compare pricing across multiple suppliers
- **Cost Optimization**: Identify cost-saving opportunities through alternative materials
- **Budget Analysis**: Track budget vs actual costs with variance reporting
- **Export Capabilities**: Export BOMs to CSV, Excel, and JSON formats

### Cost Components
- Material costs per unit with quantity-based pricing tiers
- Waste factors (configurable per material, default 5-10%)
- Manufacturing overhead (configurable, default 15%)
- Transportation costs per unit
- Currency conversion support

### BOM Structure

The generator creates comprehensive BOMs including:

#### Cell Components
- Silicon wafer
- Emitter layer (phosphorus)
- BSF (Back Surface Field) layer (aluminum)
- Anti-reflective coating (silicon nitride)
- Front metallization (silver)
- Back metallization (aluminum)

#### Module Components
- Front glass (tempered, 3.2mm)
- Encapsulant (EVA/POE, top and bottom layers)
- Backsheet (weatherproof)
- Frame (anodized aluminum)
- Junction box (with bypass diodes)

#### Interconnect Materials
- Tabbing ribbon (tinned copper)
- Bus ribbon
- Solder (SnPb or lead-free)
- Soldering flux

#### Adhesives and Sealants
- Edge sealant (silicone)
- Junction box adhesive

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.modules.bom_generator import BOMGenerator
from src.modules.models import Material, ModuleDesign, ComponentCategory, Currency
from decimal import Decimal

# 1. Create materials database
materials = [
    Material(
        id="MAT-WAFER-001",
        name="Silicon Wafer",
        category=ComponentCategory.CELL,
        supplier="SolarTech Inc.",
        unit="pieces",
        base_price=Decimal("5.50"),
        currency=Currency.USD,
        waste_factor=0.05,
    ),
    # ... add more materials
]

# 2. Initialize generator
generator = BOMGenerator(
    materials=materials,
    manufacturing_overhead_rate=0.15
)

# 3. Define module design
module_design = ModuleDesign(
    module_id="MOD-400W",
    module_type="mono-Si",
    power_rating=400.0,
    efficiency=20.5,
    dimensions={"length": 1640, "width": 990, "thickness": 35},
    num_cells=60,
    cell_size=156.75,
)

# 4. Generate BOM
bom = generator.generate_bom(module_design)

# 5. Calculate costs
result = generator.calculate_material_costs(bom)
total_cost = generator.calculate_module_cost(result)

print(f"Total module cost: ${total_cost:,.2f}")

# 6. Export BOM
generator.export_bom(result['bom_with_costs'], format="excel")
```

## Advanced Usage

### Volume-Based Pricing Tiers

```python
from src.modules.models import MaterialWithTiers, PricingTier

material = MaterialWithTiers(
    id="MAT-GLASS-001",
    name="Front Glass",
    category=ComponentCategory.MODULE,
    supplier="GlassCo",
    unit="m2",
    base_price=Decimal("18.00"),
    pricing_tiers=[
        PricingTier(
            min_quantity=Decimal("0"),
            max_quantity=Decimal("100"),
            unit_price=Decimal("18.00"),
            discount_percentage=0
        ),
        PricingTier(
            min_quantity=Decimal("100"),
            max_quantity=Decimal("500"),
            unit_price=Decimal("16.20"),
            discount_percentage=10
        ),
        PricingTier(
            min_quantity=Decimal("500"),
            max_quantity=None,  # Unlimited
            unit_price=Decimal("14.40"),
            discount_percentage=20
        ),
    ]
)
```

### Multi-Currency Support

```python
from src.modules.models import CurrencyExchangeRate

# Add exchange rates
generator.add_exchange_rate(
    CurrencyExchangeRate(
        from_currency=Currency.USD,
        to_currency=Currency.EUR,
        rate=Decimal("0.92")
    )
)

# Convert amounts
eur_amount = generator.convert_currency(
    Decimal("100.00"),
    Currency.USD,
    Currency.EUR
)
```

### Supplier Comparison

```python
# Compare suppliers for a material
comparison = generator.compare_suppliers("Silicon Wafer")

print(f"Recommended supplier: {comparison.recommended_supplier}")
print(f"Potential savings: ${comparison.potential_savings}")
for supplier, price in comparison.suppliers.items():
    print(f"  {supplier}: ${price}")
```

### BOM Optimization

```python
# Optimize BOM to reduce costs
optimized_bom = generator.optimize_bom_cost(bom)

# Compare original vs optimized
original_cost = generator.calculate_module_cost(
    generator.calculate_material_costs(bom)
)
optimized_cost = generator.calculate_module_cost(
    generator.calculate_material_costs(optimized_bom)
)

savings = original_cost - optimized_cost
print(f"Cost savings: ${savings:,.2f}")
```

### Budget Analysis

```python
# Analyze budget variance
analysis = generator.analyze_budget(
    budgeted_cost=150.00,
    actual_cost=total_cost
)

print(f"Variance: ${analysis.variance} ({analysis.variance_percentage:.1f}%)")
print(f"Over budget: {analysis.over_budget}")
```

## API Reference

### BOMGenerator Class

#### Methods

- **`__init__(materials, manufacturing_overhead_rate=0.15, default_currency=Currency.USD)`**
  - Initialize the BOM generator with materials database

- **`generate_bom(module_design: ModuleDesign) -> pd.DataFrame`**
  - Generate BOM from module design specification
  - Returns DataFrame with material_id, component_name, category, quantity, unit

- **`calculate_material_costs(bom: pd.DataFrame, materials_db: List[Material]) -> Dict`**
  - Calculate material costs with waste factors and pricing tiers
  - Returns dict with bom_with_costs, cost_breakdown, missing_materials

- **`calculate_module_cost(bom_with_costs: Dict) -> float`**
  - Calculate total module cost
  - Returns total cost as float

- **`export_bom(bom: pd.DataFrame, format: str, output_path: str) -> str`**
  - Export BOM to CSV, Excel, or JSON
  - Returns path to exported file

- **`optimize_bom_cost(bom: pd.DataFrame) -> pd.DataFrame`**
  - Optimize BOM by selecting best suppliers and materials
  - Returns optimized BOM DataFrame

- **`compare_suppliers(material_name: str) -> SupplierComparison`**
  - Compare suppliers for a specific material
  - Returns SupplierComparison object

- **`analyze_budget(budgeted_cost: float, actual_cost: float) -> BudgetAnalysis`**
  - Analyze budget vs actual costs
  - Returns BudgetAnalysis object

- **`get_cost_breakdown_by_category(bom_with_costs: pd.DataFrame) -> Dict[str, float]`**
  - Get cost breakdown by component category
  - Returns dict mapping category to total cost

## Data Models

### Material
- id: Unique material identifier
- name: Material name
- category: Component category (CELL, MODULE, INTERCONNECT, ADHESIVE, PACKAGING)
- supplier: Supplier name
- unit: Unit of measurement
- base_price: Base price per unit (Decimal)
- currency: Currency code
- waste_factor: Material waste factor (0-1)
- transportation_cost_per_unit: Transportation cost per unit

### ModuleDesign
- module_id: Module identifier
- module_type: Module type (mono-Si, poly-Si, etc.)
- power_rating: Power rating in Watts
- efficiency: Module efficiency percentage
- dimensions: Dict with length, width, thickness in mm
- num_cells: Number of cells in module
- cell_size: Cell size in mm
- frame_type, glass_type, backsheet_type, etc.

### CostBreakdown
- cell_costs, module_costs, interconnect_costs, adhesive_costs, packaging_costs
- material_subtotal: Subtotal of all materials
- waste_costs: Costs from waste
- transportation_costs: Transportation costs
- manufacturing_overhead: Manufacturing overhead
- total_cost: Grand total cost
- currency: Currency code

## Error Handling

The module provides specific exceptions for different error cases:

- **`MaterialNotFoundError`**: Raised when required material is not in database
- **`InvalidModuleDesignError`**: Raised when module design is invalid
- **`ExportError`**: Raised when BOM export fails
- **`BOMGeneratorError`**: Base exception for all BOM generator errors

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_bom_generator.py -v

# Run with coverage
pytest tests/test_bom_generator.py --cov=src/modules --cov-report=html

# Run specific test class
pytest tests/test_bom_generator.py::TestBOMGeneration -v
```

Test coverage: **94%**

## Examples

See `examples/bom_generator_example.py` for a complete working example.

## License

MIT License - See LICENSE file for details
