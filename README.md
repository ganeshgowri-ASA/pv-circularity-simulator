# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Bankability Assessment & Risk Analysis

Production-ready financial analysis module providing comprehensive bankability assessment for PV projects:

- **Credit Rating Analysis**: Standardized credit assessment (AAA to D) based on financial strength, debt capacity, liquidity, and profitability
- **Risk Assessment**: Multi-dimensional risk analysis covering technical, financial, market, and regulatory risks
- **Debt Service Coverage**: DSCR calculation with multi-year projections accounting for degradation and inflation
- **Bankability Scoring**: Integrated scoring system with actionable recommendations

#### Quick Start

```python
from pv_circularity_simulator.financial import BankabilityAssessor, FinancialMetrics, ProjectContext, ProjectStage

# Define financial metrics
metrics = FinancialMetrics(
    total_project_cost=10_000_000,
    equity_contribution=3_000_000,
    debt_amount=7_000_000,
    annual_revenue=2_000_000,
    annual_operating_cost=400_000,
    annual_debt_service=800_000,
    project_lifespan_years=25,
    discount_rate=0.08
)

# Define project context
context = ProjectContext(
    project_name="Solar Park Alpha",
    project_stage=ProjectStage.DEVELOPMENT,
    location="Arizona, USA",
    capacity_mw=10.0,
    technology_type="Monocrystalline Silicon",
    ppa_term_years=20,
    ppa_rate_usd_per_kwh=0.08
)

# Create assessor
assessor = BankabilityAssessor(financial_metrics=metrics, project_context=context)

# Perform assessments
credit_rating = assessor.credit_rating()
risk_assessment = assessor.risk_assessment()
dscr_analysis = assessor.debt_service_coverage()
bankability = assessor.project_bankability_score()

# View results
print(f"Credit Rating: {credit_rating.rating.value} ({credit_rating.rating_score:.1f}/100)")
print(f"Risk Level: {risk_assessment.overall_risk_level.value}")
print(f"DSCR: {dscr_analysis.dscr:.2f}x")
print(f"Bankability Score: {bankability.overall_score:.1f}/100")
print(f"Is Bankable: {bankability.is_bankable}")
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_circularity_simulator --cov-report=html

# Run specific test module
pytest tests/unit/test_financial.py -v
```

## Requirements

- Python >= 3.9
- pydantic >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
