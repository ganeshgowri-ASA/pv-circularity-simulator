# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### ROI Calculator & Investment Analysis

Production-ready financial modeling calculator with comprehensive investment analysis capabilities:

- **ROI Calculation**: Return on investment with tax considerations
- **NPV Analysis**: Net Present Value with configurable discount rates
- **IRR Calculation**: Internal Rate of Return using numerical optimization
- **Payback Period**: Simple and discounted payback period calculations
- **Sensitivity Analysis**: Multi-parameter sensitivity testing
- **Cash Flow Modeling**: Detailed yearly cash flow projections with inflation
- **Pydantic Validation**: Robust input validation and type safety
- **Full Test Coverage**: 49 comprehensive tests with 90% code coverage

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

### Basic ROI Calculation

```python
from src.pv_simulator.calculators.roi_calculator import ROICalculator
from src.pv_simulator.core.models import InvestmentInput

# Define investment parameters
investment = InvestmentInput(
    initial_investment=100000,
    annual_revenue=25000,
    annual_costs=5000,
    discount_rate=0.10,
    project_lifetime=25
)

# Calculate financial metrics
calculator = ROICalculator()
result = calculator.calculate(investment)

# Access results
print(f"ROI: {result.roi_percentage:.2f}%")
print(f"NPV: ${result.net_present_value:,.2f}")
print(f"IRR: {result.internal_rate_of_return:.2f}%")
print(f"Payback Period: {result.payback_period_years:.1f} years")
```

### Sensitivity Analysis

```python
from src.pv_simulator.core.models import SensitivityInput
from src.pv_simulator.core.enums import SensitivityParameter

# Define sensitivity parameters
sensitivity = SensitivityInput(
    parameter=SensitivityParameter.DISCOUNT_RATE,
    base_value=0.10,
    variation_range=[-20, -10, 0, 10, 20]
)

# Run sensitivity analysis
results = calculator.sensitivity_analysis(investment, [sensitivity])

# Analyze impact
for sens_result in results:
    print(f"NPV Range: ${sens_result.npv_range[0]:,.2f} to ${sens_result.npv_range[1]:,.2f}")
    print(f"Volatility: ${sens_result.npv_volatility:,.2f}")
```

### Complex Investment Scenario

```python
# Investment with tax, salvage value, and inflation
investment = InvestmentInput(
    initial_investment=500000,
    annual_revenue=120000,
    annual_costs=30000,
    discount_rate=0.08,
    project_lifetime=30,
    currency=CurrencyType.EUR,
    tax_rate=0.21,           # 21% corporate tax
    salvage_value=50000,     # End-of-life value
    inflation_rate=0.02      # 2% annual inflation
)

result = calculator.calculate(investment)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/pv_simulator --cov-report=term-missing

# Run specific test file
pytest tests/calculators/test_roi_calculator.py -v
```

## Examples

Run the comprehensive demo script:

```bash
PYTHONPATH=. python examples/roi_calculator_demo.py
```

This demonstrates:
- Basic ROI calculation
- Complex investment analysis
- Sensitivity analysis
- Cash flow projections

## Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/
│   ├── calculators/
│   │   ├── base.py              # Abstract base calculator
│   │   └── roi_calculator.py    # ROI calculator implementation
│   ├── core/
│   │   ├── models.py            # Pydantic data models
│   │   └── enums.py             # Enumerations
│   ├── analytics/               # Circularity & performance analytics
│   ├── integrations/            # External integrations (SCAPS, etc.)
│   └── exceptions.py            # Custom exceptions
├── tests/
│   └── calculators/
│       └── test_roi_calculator.py  # Comprehensive test suite
├── examples/
│   └── roi_calculator_demo.py      # Usage examples
└── docs/                           # Documentation

```

## API Reference

### ROICalculator

Main calculator class for investment analysis.

**Methods:**

- `calculate(inputs: InvestmentInput) -> ROIResult`: Comprehensive analysis
- `roi_calculation(inputs: InvestmentInput) -> float`: Calculate ROI percentage
- `payback_period(cash_flows: List[CashFlow], discounted: bool) -> Optional[float]`: Calculate payback period
- `irr_calculation(cash_flows: List[CashFlow]) -> Optional[float]`: Calculate IRR
- `sensitivity_analysis(base_inputs: InvestmentInput, sensitivity_inputs: List[SensitivityInput]) -> List[SensitivityAnalysisResult]`: Multi-parameter sensitivity analysis

### Pydantic Models

**InvestmentInput**: Input parameters for investment analysis
- `initial_investment`: Capital investment required
- `annual_revenue`: Expected annual revenue
- `annual_costs`: Annual operating costs
- `discount_rate`: Discount rate for NPV (0-1)
- `project_lifetime`: Project duration in years
- `tax_rate`: Corporate tax rate (0-1)
- `salvage_value`: End-of-life asset value
- `inflation_rate`: Annual inflation rate (0-1)
- `currency`: Currency type

**ROIResult**: Comprehensive analysis results
- `roi_percentage`: Return on Investment (%)
- `net_present_value`: NPV
- `internal_rate_of_return`: IRR (%)
- `payback_period_years`: Simple payback period
- `discounted_payback_period_years`: Discounted payback period
- `total_revenue`: Total revenue over lifetime
- `total_costs`: Total costs over lifetime
- `net_profit`: Total profit
- `profitability_index`: NPV / initial investment
- `annual_roi`: Average annual ROI
- `cash_flows`: Detailed yearly cash flows

**SensitivityInput**: Sensitivity analysis configuration
- `parameter`: Parameter to vary (enum)
- `base_value`: Base parameter value
- `variation_range`: Percentage variations
- `variation_values`: Explicit test values

**SensitivityAnalysisResult**: Sensitivity analysis results
- `parameter`: Parameter analyzed
- `base_case`: Base case results
- `results`: Results for each variation
- `roi_range`: ROI range [min, max]
- `npv_range`: NPV range [min, max]
- `elasticity`: Elasticity coefficient

## Technical Details

### Financial Calculations

- **ROI**: `((Total Net Revenue - Initial Investment) / Initial Investment) × 100`
- **NPV**: Sum of discounted cash flows: `Σ(CF_t / (1 + r)^t)`
- **IRR**: Discount rate where NPV = 0 (solved numerically)
- **Payback Period**: Time to recover initial investment (with interpolation)

### Numerical Methods

- IRR calculation uses scipy's `brentq` (Brent's method) and `newton` (Newton's method)
- Convergence tolerance: 1e-6 (configurable)
- Maximum iterations: 100 (configurable)

### Validation

- Pydantic v2 models with comprehensive field validation
- Business logic validation in calculator
- Warning system for unusual input combinations
- Type safety throughout

## Test Coverage

```
Name                                             Coverage
---------------------------------------------------------
src/pv_simulator/calculators/roi_calculator.py   91%
src/pv_simulator/core/models.py                  95%
src/pv_simulator/core/enums.py                   100%
src/pv_simulator/exceptions.py                   80%
---------------------------------------------------------
TOTAL                                            90%
```

49 tests covering:
- Input validation (8 tests)
- ROI calculations (4 tests)
- Payback period (5 tests)
- IRR calculations (5 tests)
- Comprehensive analysis (5 tests)
- Sensitivity analysis (5 tests)
- Cash flow generation (4 tests)
- Edge cases (5 tests)
- Configuration (4 tests)
- Parametrized scenarios (4 tests)

## License

MIT License - see LICENSE file for details
