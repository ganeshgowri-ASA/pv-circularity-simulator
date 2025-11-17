# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Incentives & Tax Credit Modeling (v0.1.0)

Production-ready tax incentive modeling for solar PV systems with comprehensive support for:

- **Investment Tax Credit (ITC)**: Federal tax credit calculations with bonus adders
  - Base ITC (typically 30% for solar)
  - Domestic content bonus (+10%)
  - Energy community bonus (+10%)
  - Basis reductions for grants and subsidies

- **Production Tax Credit (PTC)**: Multi-year production credit modeling
  - Per-kWh credit calculations over 10-year period
  - Inflation adjustments
  - Production degradation modeling
  - Bonus multipliers (up to 5x)
  - Net Present Value (NPV) analysis

- **MACRS Depreciation**: Complete depreciation schedule calculations
  - MACRS 5-year schedule (standard for solar)
  - MACRS 7-year schedule
  - ITC basis adjustment (50% reduction)
  - Bonus depreciation (up to 100%)
  - Straight-line and declining balance methods

- **Tax Equity Partnership Modeling**: Partnership flip structure analysis
  - Pre-flip and post-flip allocation modeling
  - IRR calculations for investor and sponsor
  - NPV analysis with customizable discount rates
  - Annual cash flow and tax benefit allocation
  - Flip year determination based on target returns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic ITC Calculation

```python
from datetime import date
from pv_simulator import IncentiveModeler
from pv_simulator.models import SystemConfiguration, ITCConfiguration

# Define your solar system
system = SystemConfiguration(
    system_size_kw=100.0,
    installation_cost_total=250_000.0,
    installation_date=date(2024, 1, 15),
    location_state="CA",
    expected_annual_production_kwh=150_000.0,
)

# Configure ITC calculation
itc_config = ITCConfiguration(
    system_config=system,
    itc_rate=0.30,  # 30% ITC
    apply_bonus=True,
    meets_domestic_content=True,  # +10% bonus
)

# Calculate ITC
modeler = IncentiveModeler()
result = modeler.itc_calculation(itc_config)

print(f"Total ITC Credit: ${result.total_itc_amount:,.2f}")
print(f"Effective Rate: {result.effective_rate:.1%}")
```

### PTC Analysis

```python
from pv_simulator.models import PTCConfiguration

# Configure PTC
ptc_config = PTCConfiguration(
    system_config=system,
    ptc_rate_per_kwh=0.0275,
    credit_period_years=10,
    inflation_adjustment=True,
    production_degradation_rate=0.005,
)

# Calculate PTC over 10 years
result = modeler.ptc_computation(ptc_config)

print(f"Total PTC (Nominal): ${result.total_ptc_lifetime:,.2f}")
print(f"NPV (6% discount): ${result.present_value_ptc:,.2f}")
```

### Depreciation Schedule

```python
from pv_simulator.models import DepreciationMethod

# Calculate depreciation (with ITC basis adjustment)
asset_basis = 250_000.0 - (0.5 * 75_000.0)  # Cost - 50% of ITC

result = modeler.depreciation_schedule(
    asset_basis=asset_basis,
    method=DepreciationMethod.MACRS_5,
    bonus_depreciation_rate=0.80,  # 80% bonus
)

print(f"Year 1 Depreciation: ${result.annual_depreciation[0]:,.2f}")
print(f"Total Depreciation: ${result.total_depreciation:,.2f}")
```

### Complete Tax Equity Analysis

```python
from pv_simulator.models import TaxEquityConfiguration

# Configure tax equity partnership
te_config = TaxEquityConfiguration(
    system_config=system,
    investor_equity_percentage=0.99,  # 99% pre-flip
    target_flip_irr=0.08,  # 8% target IRR
    post_flip_investor_percentage=0.05,  # 5% post-flip
)

# Model partnership flip
result = modeler.tax_equity_modeling(
    config=te_config,
    itc_amount=75_000.0,
    depreciation_schedule=[...],  # From depreciation calculation
)

print(f"Flip Year: {result.flip_year}")
print(f"Investor IRR: {result.investor_irr:.2%}")
print(f"Sponsor IRR: {result.sponsor_irr:.2%}")
```

## Documentation

- **Examples**: See `examples/` directory for comprehensive usage examples
- **API Documentation**: All classes and methods include detailed docstrings
- **Tests**: See `tests/` directory for extensive test coverage

## Running Examples

```bash
# Basic ITC calculation
python examples/basic_itc_calculation.py

# ITC with bonus credits
python examples/itc_with_bonuses.py

# PTC analysis
python examples/ptc_analysis.py

# Depreciation schedules
python examples/depreciation_example.py

# Complete tax equity analysis
python examples/complete_tax_equity_example.py
```

## Testing

Run the test suite with pytest:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/unit/test_incentive_modeler.py -v
```

## Technology Stack

- **Python 3.10+**: Modern Python features and type hints
- **Pydantic v2**: Data validation and settings management
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing utilities

## Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/
│   ├── models/              # Pydantic data models
│   │   ├── base.py         # Base model classes
│   │   └── incentives.py   # Tax incentive models
│   ├── simulators/          # Core simulation modules
│   │   └── incentive_modeler.py  # Tax incentive calculations
│   └── utils/               # Utility functions
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Usage examples
├── docs/                   # Documentation
└── pyproject.toml         # Project configuration
```

## API Reference

### IncentiveModeler

Main class for tax incentive calculations.

#### Methods

- `itc_calculation(config: ITCConfiguration) -> ITCResult`
  - Calculate Investment Tax Credit with bonus adders

- `ptc_computation(config: PTCConfiguration, discount_rate: float) -> PTCResult`
  - Compute Production Tax Credit over credit period

- `depreciation_schedule(asset_basis: float, method: DepreciationMethod, bonus_rate: float) -> DepreciationScheduleResult`
  - Generate MACRS or other depreciation schedules

- `tax_equity_modeling(config: TaxEquityConfiguration, ...) -> TaxEquityResult`
  - Model partnership flip tax equity structures

### Data Models

All input and output models are Pydantic-based with:
- Full type validation
- Comprehensive field descriptions
- Default values where applicable
- Custom validators for business logic

See `src/pv_simulator/models/incentives.py` for complete model definitions.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code is formatted: `black .`
3. Imports are sorted: `isort .`
4. Type hints are present: `mypy src/`
5. Linting passes: `ruff check .`

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- IRS Publication 946 for MACRS depreciation schedules
- Solar Energy Industries Association (SEIA) for tax policy guidance
- Industry standard partnership flip structures

## Roadmap

Future development priorities:

- [ ] State-level incentive programs
- [ ] REAP (Rural Energy for America Program) modeling
- [ ] Advanced partnership structures (sale-leaseback, inverted lease)
- [ ] Multi-year tax appetite modeling
- [ ] Monte Carlo sensitivity analysis
- [ ] Integration with energy production forecasting
- [ ] Debt financing and DSCR calculations

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Refer to the examples in `examples/`
- Check the comprehensive test suite in `tests/`
