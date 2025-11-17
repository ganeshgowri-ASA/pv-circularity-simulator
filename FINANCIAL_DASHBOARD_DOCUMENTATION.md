# Financial Analysis Dashboard & Reporting - Documentation

## Overview

The Financial Analysis Dashboard is a comprehensive, production-ready system for analyzing the financial viability of photovoltaic (PV) systems with integrated circular economy considerations. This module provides state-of-the-art financial modeling, visualization, and reporting capabilities specifically designed for the PV industry.

## Features

### Core Components

1. **FinancialDashboardUI** - Main dashboard interface
   - `lcoe_calculator_display()` - LCOE calculation interface
   - `cashflow_visualization()` - Interactive cash flow charts
   - `sensitivity_analysis_ui()` - Multi-dimensional sensitivity analysis
   - `financial_reports_generator()` - Multi-format report generation

2. **LCOECalculator** - Levelized Cost of Energy calculations
   - Complete LCOE analysis with circularity integration
   - Cost breakdown by category
   - Real vs nominal LCOE
   - Scenario comparison capabilities

3. **SensitivityAnalyzer** - Risk and sensitivity analysis
   - One-way sensitivity analysis
   - Two-way (2D) sensitivity heatmaps
   - Tornado diagrams
   - Monte Carlo simulation

4. **FinancialChartBuilder** - Professional visualizations
   - Cash flow waterfalls
   - Cost breakdown charts
   - Sensitivity plots
   - Interactive Plotly dashboards

5. **FinancialReportGenerator** - Report generation
   - PDF executive summaries
   - Detailed Excel workbooks
   - Interactive HTML reports
   - CSV data export

## Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **numpy & pandas** - Numerical computing and data analysis
- **plotly** - Interactive visualizations
- **streamlit** - Dashboard UI framework
- **reportlab** - PDF report generation
- **openpyxl** - Excel file handling
- **pvlib** - PV system modeling
- **scipy** - Scientific computing

## Quick Start

### Launch Dashboard

```bash
# Method 1: Using streamlit directly
streamlit run app.py

# Method 2: Using Python
python app.py
```

### Basic Usage Example

```python
from financial.models import CostStructure, RevenueStream, CircularityMetrics
from financial.calculators import LCOECalculator

# Define cost structure
costs = CostStructure(
    initial_capex=200000.0,
    equipment_cost=150000.0,
    installation_cost=30000.0,
    soft_costs=20000.0,
    annual_opex=2000.0,
    maintenance_cost=1500.0,
    insurance_cost=500.0,
)

# Define revenue stream
revenue = RevenueStream(
    annual_energy_production=150000.0,  # kWh/year
    energy_price=0.12,  # $/kWh
    degradation_rate=0.005,  # 0.5% per year
)

# Define circularity metrics
circularity = CircularityMetrics(
    material_recovery_rate=0.90,
    system_weight=1000.0,
    refurbishment_potential=0.30,
)

# Calculate LCOE
calculator = LCOECalculator(
    cost_structure=costs,
    revenue_stream=revenue,
    circularity_metrics=circularity,
    lifetime_years=25,
    discount_rate=0.06,
)

result = calculator.calculate_lcoe()

print(f"LCOE: ${result.lcoe:.4f}/kWh")
print(f"Circularity Benefit: ${result.circularity_benefit:.4f}/kWh")
print(f"NPV: ${result.calculate_npv():,.2f}")
```

## Detailed Component Documentation

### 1. LCOE Calculator (`lcoe_calculator_display()`)

The LCOE calculator provides comprehensive levelized cost of energy analysis.

#### Key Features:

- **Complete Cost Modeling**
  - Capital expenditure (CAPEX): Equipment, installation, soft costs
  - Operating expenditure (OPEX): Maintenance, insurance, land lease
  - End-of-life costs: Decommissioning, disposal/recovery

- **Revenue Modeling**
  - Energy production with degradation
  - Dynamic energy pricing with escalation
  - Feed-in tariffs
  - Tax credits and subsidies

- **Circularity Integration**
  - End-of-life material recovery value
  - Refurbishment potential
  - Recycling economics
  - Avoided disposal costs

#### Formulas:

**Basic LCOE:**
```
LCOE = Σ(Ct / (1+r)^t) / Σ(Et / (1+r)^t)
```

Where:
- `Ct` = Total costs in year t
- `Et` = Energy produced in year t
- `r` = Discount rate
- `t` = Year

**With Circularity:**
```
LCOE_circ = (Initial_Costs + Σ(OPEX_t) - EOL_Recovery) / Σ(Energy_t)
```

#### Usage:

```python
result = calculator.calculate_lcoe(include_circularity=True)

# Access results
print(f"LCOE: ${result.lcoe:.4f}/kWh")
print(f"Real LCOE: ${result.lcoe_real:.4f}/kWh")
print(f"Total Lifetime Cost: ${result.total_lifetime_cost:,.2f}")
print(f"Circularity Benefit: {result.circularity_benefit:.4f}/kWh")

# Get cost breakdown
for category, cost in result.cost_breakdown.items():
    print(f"{category}: ${cost:,.2f}")
```

### 2. Cash Flow Visualization (`cashflow_visualization()`)

Interactive visualization of project cash flows over the system lifetime.

#### Visualizations:

1. **Cash Flow Waterfall**
   - Annual revenue, costs, and net cash flow
   - Cumulative cash flow overlay
   - Payback period identification

2. **Revenue vs Costs Trends**
   - Stacked area chart
   - Revenue growth with degradation and escalation
   - Cost trends over time

3. **Cumulative Analysis**
   - Break-even point visualization
   - Total value creation over lifetime

#### Key Metrics Displayed:

- **NPV (Net Present Value)** - Total value in today's dollars
- **IRR (Internal Rate of Return)** - Effective annual return
- **Payback Period** - Time to recover initial investment
- **ROI (Return on Investment)** - Percentage return

#### Usage:

```python
from financial.models import CashFlowModel

cash_flow_model = CashFlowModel(
    cost_structure=costs,
    revenue_stream=revenue,
    circularity_metrics=circularity,
    lifetime_years=25,
    discount_rate=0.06,
)

# Generate cash flow series
df = cash_flow_model.generate_cash_flow_series()

# Calculate metrics
npv = cash_flow_model.calculate_npv()
irr = cash_flow_model.calculate_irr()
payback = cash_flow_model.calculate_payback_period()
roi = cash_flow_model.calculate_roi()

print(f"NPV: ${npv:,.2f}")
print(f"IRR: {irr*100:.2f}%")
print(f"Payback: {payback:.1f} years")
print(f"ROI: {roi:.1f}%")
```

### 3. Sensitivity Analysis (`sensitivity_analysis_ui()`)

Comprehensive risk analysis and parameter sensitivity assessment.

#### Analysis Types:

**a) One-Way Sensitivity**
- Vary single parameter across a range
- Observe impact on output metric (LCOE, NPV, IRR)
- Calculate elasticity

```python
from financial.calculators import SensitivityAnalyzer, SensitivityMetric
from financial.models import SensitivityParameter

analyzer = SensitivityAnalyzer(
    base_cost_structure=costs,
    base_revenue_stream=revenue,
    base_circularity_metrics=circularity,
)

# Define parameter to analyze
param = SensitivityParameter(
    name='equipment_cost',
    base_value=150000.0,
    min_value=100000.0,
    max_value=200000.0,
    step=10000.0,
)

# Run analysis
result = analyzer.one_way_sensitivity(param, SensitivityMetric.LCOE)

print(f"Elasticity: {result.elasticity:.3f}")
```

**b) Tornado Diagram**
- Compare impact of multiple parameters simultaneously
- Sorted by impact magnitude
- Quick identification of key drivers

```python
parameters = [param1, param2, param3, ...]
tornado_data = analyzer.tornado_analysis(
    parameters,
    SensitivityMetric.LCOE,
    variation_percent=20.0
)
```

**c) Two-Way Sensitivity**
- Analyze interaction between two parameters
- Heatmap visualization
- Identify optimal parameter combinations

```python
result_df = analyzer.two_way_sensitivity(
    parameter1=param1,
    parameter2=param2,
    metric=SensitivityMetric.NPV
)
```

**d) Monte Carlo Simulation**
- Probabilistic risk analysis
- Account for parameter uncertainty
- Generate probability distributions of outcomes

```python
from scipy import stats

distributions = {
    'equipment_cost': (stats.norm, {'loc': 150000, 'scale': 15000}),
    'energy_price': (stats.norm, {'loc': 0.12, 'scale': 0.02}),
}

results_df = analyzer.monte_carlo_simulation(
    distributions,
    SensitivityMetric.LCOE,
    n_simulations=10000
)

# Analyze results
p10 = results_df['lcoe'].quantile(0.10)
p50 = results_df['lcoe'].quantile(0.50)
p90 = results_df['lcoe'].quantile(0.90)

print(f"P10: ${p10:.4f}/kWh")
print(f"P50: ${p50:.4f}/kWh")
print(f"P90: ${p90:.4f}/kWh")
```

### 4. Financial Reports Generator (`financial_reports_generator()`)

Professional report generation in multiple formats.

#### Report Types:

**a) PDF Executive Summary**
- Key financial metrics
- Cost breakdown tables
- Circularity impact analysis
- Embedded visualizations
- Professional formatting

```python
from financial.reporting import FinancialReportGenerator

report_gen = FinancialReportGenerator(
    project_name="Solar Farm Project Alpha",
    company_name="Green Energy Corp",
    analyst_name="Jane Smith"
)

pdf_path = report_gen.generate_executive_summary_pdf(
    cash_flow_model=cash_flow_model,
    lcoe_result=lcoe_result,
    output_path="reports/executive_summary.pdf",
    include_charts=True
)
```

**b) Detailed Excel Report**
- Multiple worksheets:
  - Executive Summary
  - Cash Flow Analysis
  - Cost Breakdown
  - Circularity Metrics
  - Sensitivity Analysis
- Formatted with charts and conditional formatting
- Downloadable and shareable

```python
excel_path = report_gen.generate_detailed_excel_report(
    cash_flow_model=cash_flow_model,
    lcoe_result=lcoe_result,
    output_path="reports/detailed_analysis.xlsx",
    include_sensitivity=True
)
```

**c) Interactive HTML Report**
- Web-based report with embedded Plotly charts
- Fully interactive visualizations
- Responsive design
- No software dependencies for viewing

```python
html_path = report_gen.generate_html_report(
    cash_flow_model=cash_flow_model,
    lcoe_result=lcoe_result,
    output_path="reports/interactive_report.html"
)
```

**d) CSV Data Export**
- Raw data export for further analysis
- Multiple CSV files in organized structure
- Compatible with any spreadsheet software

```python
csv_files = report_gen.export_data_to_csv(
    cash_flow_model=cash_flow_model,
    output_dir="reports/data"
)
```

## Circular Economy Integration

### 3R Approach (Reduce, Reuse, Recycle)

The financial models explicitly account for circular economy benefits:

#### 1. Material Recovery
- End-of-life material value recovery
- Recycling revenue vs disposal costs
- Avoided environmental costs

#### 2. Refurbishment
- Component reuse potential
- Value retention after refurbishment
- Extended effective lifetime

#### 3. System Weight & Composition
- Material types and quantities
- Recovery rates by material
- Market values for recovered materials

### Circularity Score

The system calculates a comprehensive circularity score (0-100) based on:
- Material recovery rate (40%)
- Refurbishment potential (30%)
- Economic viability of recycling (30%)

```python
score = circularity_metrics.get_circularity_score()
# Returns: 0-100, where 100 = fully circular system
```

### End-of-Life Value Calculation

```python
eol_value = circularity_metrics.get_eol_recovery_value(
    original_system_value=150000.0
)

# Components:
# 1. Refurbishment value = Original × Refurb% × ValueRetention%
# 2. Recycling value = Weight × Recovery% × (Revenue - Cost)
# 3. Avoided costs = Weight × Recovery% × AvoidedDisposal
# Total = Sum of above
```

## Technical Architecture

### Module Structure

```
src/financial/
├── models/
│   ├── __init__.py
│   └── financial_models.py          # Data models
├── calculators/
│   ├── __init__.py
│   ├── lcoe_calculator.py           # LCOE calculations
│   └── sensitivity_analysis.py      # Sensitivity analysis
├── visualization/
│   ├── __init__.py
│   └── charts.py                    # Plotly visualizations
├── reporting/
│   ├── __init__.py
│   └── report_generator.py          # Report generation
└── dashboard/
    ├── __init__.py
    └── financial_dashboard_ui.py    # Main UI component
```

### Data Flow

```
Input Parameters
    ↓
Financial Models (CostStructure, RevenueStream, CircularityMetrics)
    ↓
Calculators (LCOE, NPV, IRR, Sensitivity)
    ↓
Visualizations (Plotly Charts)
    ↓
Reports (PDF, Excel, HTML)
    ↓
User Interface (Streamlit Dashboard)
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/financial --cov-report=html

# Run specific test file
pytest tests/financial/test_lcoe_calculator.py -v
```

### Test Coverage

The test suite includes:
- ✅ LCOE calculation accuracy
- ✅ Cost breakdown validation
- ✅ Circularity impact verification
- ✅ Edge case handling
- ✅ Parameter validation
- ✅ Sensitivity analysis correctness

## Best Practices

### 1. Input Validation

Always validate input parameters:

```python
# Good practice
if discount_rate < 0 or discount_rate > 0.3:
    raise ValueError("Discount rate should be between 0 and 30%")

if lifetime_years < 1 or lifetime_years > 50:
    raise ValueError("Lifetime should be between 1 and 50 years")
```

### 2. Error Handling

Use try-except blocks for file operations and calculations:

```python
try:
    result = calculator.calculate_lcoe()
except Exception as e:
    logger.error(f"LCOE calculation failed: {e}")
    # Provide fallback or user-friendly error
```

### 3. Performance Optimization

For large-scale sensitivity analysis:

```python
# Use vectorized operations
param_values = np.linspace(min_val, max_val, 100)

# Parallelize if needed
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(calculate_metric, param_values)
```

### 4. Documentation

All functions include comprehensive docstrings:

```python
def calculate_lcoe(self, include_circularity: bool = True) -> LCOEResult:
    """
    Calculate Levelized Cost of Energy.

    Args:
        include_circularity: Whether to include circular economy benefits

    Returns:
        LCOEResult with comprehensive LCOE analysis

    Example:
        >>> result = calculator.calculate_lcoe()
        >>> print(f"LCOE: ${result.lcoe:.4f}/kWh")
    """
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**2. Plotly Charts Not Displaying**
```bash
# Solution: Install kaleido for image export
pip install kaleido
```

**3. PDF Generation Fails**
```bash
# Solution: Install reportlab properly
pip install --upgrade reportlab pillow
```

**4. Streamlit Port Already in Use**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

## Performance Considerations

### Memory Usage

- Cash flow models: ~1-10 MB depending on lifetime
- Monte Carlo simulations: ~100-500 MB for 10,000+ runs
- Report generation: ~5-50 MB depending on charts

### Computation Time

- LCOE calculation: < 0.1 seconds
- One-way sensitivity (100 points): < 1 second
- Tornado diagram (5 parameters): < 5 seconds
- Monte Carlo (10,000 simulations): 5-30 seconds
- PDF report generation: 2-10 seconds

### Optimization Tips

1. **Cache Results**: Use Streamlit's `@st.cache_data` for expensive calculations
2. **Reduce Simulations**: Start with 1,000 Monte Carlo runs, increase if needed
3. **Limit Chart Points**: Use reasonable step sizes in sensitivity analysis
4. **Batch Processing**: Generate multiple reports in batch mode

## API Reference

### Core Classes

#### CostStructure
```python
CostStructure(
    initial_capex: float,
    equipment_cost: float,
    installation_cost: float,
    soft_costs: float,
    annual_opex: float,
    maintenance_cost: float,
    insurance_cost: float = 0.0,
    land_lease_cost: float = 0.0,
    decommissioning_cost: float = 0.0,
    disposal_cost: float = 0.0,
    replacement_costs: Dict[int, float] = None
)
```

#### RevenueStream
```python
RevenueStream(
    annual_energy_production: float,
    energy_price: float,
    feed_in_tariff: float = 0.0,
    tariff_duration: int = 0,
    tax_credits: float = 0.0,
    subsidies: float = 0.0,
    rec_value: float = 0.0,
    degradation_rate: float = 0.005,
    escalation_rate: float = 0.02
)
```

#### CircularityMetrics
```python
CircularityMetrics(
    material_recovery_rate: float = 0.90,
    recovered_material_value: float = 15.0,
    system_weight: float = 1000.0,
    refurbishment_potential: float = 0.30,
    refurbishment_value: float = 0.40,
    recycling_cost: float = 5.0,
    recycling_revenue: float = 12.0,
    avoided_disposal_cost: float = 8.0
)
```

## Contributing

This is a production-ready module with comprehensive documentation.
For enhancements or bug fixes:

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## License

This module is part of the PV Circularity Simulator project.
Licensed under MIT License.

## Support

For questions or issues:
- Review this documentation
- Check the example code in tests/
- Run the interactive dashboard for hands-on exploration

## Version History

### v1.0.0 (Current)
- ✅ Complete LCOE calculator with circularity integration
- ✅ Comprehensive cash flow modeling
- ✅ Multi-dimensional sensitivity analysis
- ✅ Professional report generation (PDF, Excel, HTML, CSV)
- ✅ Interactive Streamlit dashboard
- ✅ Production-ready with full docstrings
- ✅ Comprehensive test coverage
- ✅ Plotly-based visualizations

## Conclusion

The Financial Analysis Dashboard provides a complete, production-ready solution for PV system financial analysis. With its integration of circular economy considerations, comprehensive sensitivity analysis, and professional reporting capabilities, it serves as a powerful tool for investment decisions, project evaluation, and stakeholder communication.
