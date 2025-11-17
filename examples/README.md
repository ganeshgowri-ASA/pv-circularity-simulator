# IncentiveModeler Examples

This directory contains comprehensive examples demonstrating the use of the IncentiveModeler for solar tax incentive calculations.

## Examples

### 1. Basic ITC Calculation
**File:** `basic_itc_calculation.py`

Demonstrates basic Investment Tax Credit (ITC) calculation for a commercial solar installation with the standard 30% federal ITC.

**Run:**
```bash
python examples/basic_itc_calculation.py
```

### 2. ITC with Bonus Credits
**File:** `itc_with_bonuses.py`

Shows how to calculate ITC with bonus credits for:
- Domestic content requirements (+10%)
- Energy community designation (+10%)
- Combined bonuses (up to 50% total ITC)

**Run:**
```bash
python examples/itc_with_bonuses.py
```

### 3. Production Tax Credit (PTC) Analysis
**File:** `ptc_analysis.py`

Comprehensive PTC analysis including:
- 10-year credit period calculations
- Production degradation modeling
- Inflation adjustment
- Bonus multipliers (5x for certain projects)
- Net Present Value calculation

**Run:**
```bash
python examples/ptc_analysis.py
```

### 4. MACRS Depreciation
**File:** `depreciation_example.py`

Demonstrates depreciation schedule calculations:
- MACRS 5-year and 7-year schedules
- ITC basis adjustment (50% reduction)
- Bonus depreciation (up to 100%)
- Tax shield NPV analysis

**Run:**
```bash
python examples/depreciation_example.py
```

### 5. Complete Tax Equity Partnership Flip
**File:** `complete_tax_equity_example.py`

Full end-to-end tax equity analysis for a 5MW commercial project:
1. ITC calculation with domestic content bonus
2. MACRS depreciation with bonus depreciation
3. Partnership flip structure modeling
4. IRR and NPV calculations
5. Annual cash flow allocations
6. Pre-flip and post-flip analysis

**Run:**
```bash
python examples/complete_tax_equity_example.py
```

## Running All Examples

To run all examples in sequence:

```bash
python examples/basic_itc_calculation.py
python examples/itc_with_bonuses.py
python examples/ptc_analysis.py
python examples/depreciation_example.py
python examples/complete_tax_equity_example.py
```

## Requirements

All examples require the `pv-circularity-simulator` package to be installed:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Understanding the Examples

### Tax Credits
- **ITC (Investment Tax Credit)**: One-time credit based on installation cost
- **PTC (Production Tax Credit)**: Annual credits based on energy production

### Depreciation
- **MACRS 5-Year**: Standard for solar installations
- **ITC Basis Adjustment**: Depreciable basis reduced by 50% of ITC
- **Bonus Depreciation**: Accelerated first-year depreciation

### Tax Equity
- **Partnership Flip**: Common solar financing structure
- **Pre-Flip**: Investor receives ~99% of tax benefits
- **Flip Year**: When investor reaches target IRR
- **Post-Flip**: Developer receives ~95% of benefits

## Additional Resources

- [IRS Form 3468](https://www.irs.gov/forms-pubs/about-form-3468) - Investment Credit
- [IRS Publication 946](https://www.irs.gov/publications/p946) - Depreciation
- [SEIA Solar Tax Manual](https://www.seia.org/) - Solar Energy Industries Association
