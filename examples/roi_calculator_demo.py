"""
ROI Calculator Demo

This script demonstrates the usage of the ROICalculator for comprehensive
investment analysis including ROI, NPV, IRR, payback period, and sensitivity analysis.
"""

from src.pv_simulator.calculators.roi_calculator import ROICalculator
from src.pv_simulator.core.models import InvestmentInput, SensitivityInput
from src.pv_simulator.core.enums import CurrencyType, SensitivityParameter


def basic_roi_example():
    """Demonstrate basic ROI calculation."""
    print("=" * 70)
    print("BASIC ROI CALCULATION EXAMPLE")
    print("=" * 70)

    # Create investment scenario
    investment = InvestmentInput(
        initial_investment=100000,
        annual_revenue=25000,
        annual_costs=5000,
        discount_rate=0.10,
        project_lifetime=25,
        currency=CurrencyType.USD,
    )

    # Calculate ROI and financial metrics
    calculator = ROICalculator()
    result = calculator.calculate(investment)

    # Display results
    print(f"\nInvestment Analysis Results:")
    print(f"  Initial Investment:        ${result.initial_investment:,.2f}")
    print(f"  Project Lifetime:          {investment.project_lifetime} years")
    print(f"  Discount Rate:             {result.discount_rate:.1%}")
    print(f"\nFinancial Metrics:")
    print(f"  ROI:                       {result.roi_percentage:.2f}%")
    print(f"  Net Present Value (NPV):   ${result.net_present_value:,.2f}")
    print(f"  Internal Rate of Return:   {result.internal_rate_of_return:.2f}%")
    print(f"  Payback Period:            {result.payback_period_years:.1f} years")
    print(f"  Discounted Payback:        {result.discounted_payback_period_years:.1f} years"
          if result.discounted_payback_period_years else "  Discounted Payback:        Never recovered")
    print(f"\nProfitability Analysis:")
    print(f"  Total Revenue:             ${result.total_revenue:,.2f}")
    print(f"  Total Costs:               ${result.total_costs:,.2f}")
    print(f"  Net Profit:                ${result.net_profit:,.2f}")
    print(f"  Profitability Index:       {result.profitability_index:.2f}")
    print(f"  Annual ROI:                {result.annual_roi:.2f}%")
    print(f"\nInvestment Quality:")
    print(f"  Is Profitable?             {'Yes' if result.is_profitable else 'No'}")
    print(f"  Meets Hurdle Rate?         {'Yes' if result.meets_hurdle_rate else 'No'}")


def complex_investment_example():
    """Demonstrate complex investment with tax, salvage value, and inflation."""
    print("\n\n" + "=" * 70)
    print("COMPLEX INVESTMENT ANALYSIS")
    print("=" * 70)

    # Create complex investment scenario
    investment = InvestmentInput(
        initial_investment=500000,
        annual_revenue=120000,
        annual_costs=30000,
        discount_rate=0.08,
        project_lifetime=30,
        currency=CurrencyType.EUR,
        tax_rate=0.21,
        salvage_value=50000,
        inflation_rate=0.02,
    )

    calculator = ROICalculator()
    result = calculator.calculate(investment)

    print(f"\nInvestment Parameters:")
    print(f"  Initial Investment:        €{result.initial_investment:,.2f}")
    print(f"  Annual Revenue (base):     €{investment.annual_revenue:,.2f}")
    print(f"  Annual Costs (base):       €{investment.annual_costs:,.2f}")
    print(f"  Tax Rate:                  {investment.tax_rate:.1%}")
    print(f"  Salvage Value:             €{investment.salvage_value:,.2f}")
    print(f"  Inflation Rate:            {investment.inflation_rate:.1%}")
    print(f"\nResults:")
    print(f"  ROI:                       {result.roi_percentage:.2f}%")
    print(f"  NPV:                       €{result.net_present_value:,.2f}")
    print(f"  IRR:                       {result.internal_rate_of_return:.2f}%")
    print(f"  Payback Period:            {result.payback_period_years:.1f} years")


def sensitivity_analysis_example():
    """Demonstrate sensitivity analysis."""
    print("\n\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS EXAMPLE")
    print("=" * 70)

    # Base investment
    base_investment = InvestmentInput(
        initial_investment=100000,
        annual_revenue=25000,
        annual_costs=5000,
        discount_rate=0.10,
        project_lifetime=25,
    )

    # Define sensitivity parameters
    discount_rate_sensitivity = SensitivityInput(
        parameter=SensitivityParameter.DISCOUNT_RATE,
        base_value=0.10,
        variation_range=[-20, -10, 0, 10, 20],
    )

    revenue_sensitivity = SensitivityInput(
        parameter=SensitivityParameter.ANNUAL_REVENUE,
        base_value=25000,
        variation_range=[-30, -15, 0, 15, 30],
    )

    # Run sensitivity analysis
    calculator = ROICalculator()
    results = calculator.sensitivity_analysis(
        base_investment,
        [discount_rate_sensitivity, revenue_sensitivity]
    )

    # Display results for each parameter
    for sens_result in results:
        print(f"\n{'─' * 70}")
        print(f"Sensitivity to: {sens_result.parameter.value.upper()}")
        print(f"{'─' * 70}")
        print(f"\nBase Case:")
        print(f"  NPV: ${sens_result.base_case.net_present_value:,.2f}")
        print(f"  ROI: {sens_result.base_case.roi_percentage:.2f}%")
        print(f"\nSensitivity Range:")
        print(f"  NPV Range:  ${sens_result.npv_range[0]:,.2f} to ${sens_result.npv_range[1]:,.2f}")
        print(f"  ROI Range:  {sens_result.roi_range[0]:.2f}% to {sens_result.roi_range[1]:.2f}%")
        print(f"\nVolatility:")
        print(f"  NPV Volatility: ${sens_result.npv_volatility:,.2f}")
        print(f"  ROI Volatility: {sens_result.roi_volatility:.2f}%")
        print(f"\nMost Sensitive To:")
        print(f"  {sens_result.most_sensitive_to}")

        if sens_result.elasticity is not None:
            print(f"\nElasticity: {sens_result.elasticity:.2f}")

        print(f"\nDetailed Results:")
        print(f"  {'Parameter Value':<20} {'NPV':>15} {'ROI':>10} {'IRR':>10}")
        print(f"  {'-'*60}")
        for r in sens_result.results:
            irr_str = f"{r.internal_rate_of_return:.2f}%" if r.internal_rate_of_return else "N/A"
            print(f"  {r.parameter_value:<20.6f} ${r.net_present_value:>13,.2f} "
                  f"{r.roi_percentage:>9.2f}% {irr_str:>10}")


def cash_flow_example():
    """Demonstrate detailed cash flow analysis."""
    print("\n\n" + "=" * 70)
    print("CASH FLOW ANALYSIS EXAMPLE")
    print("=" * 70)

    investment = InvestmentInput(
        initial_investment=100000,
        annual_revenue=25000,
        annual_costs=5000,
        discount_rate=0.10,
        project_lifetime=10,
        inflation_rate=0.02,
    )

    calculator = ROICalculator()
    result = calculator.calculate(investment)

    print(f"\nDetailed Cash Flow Analysis (First 10 years):")
    print(f"{'Year':<6} {'Inflow':>12} {'Outflow':>12} {'Net Flow':>12} "
          f"{'Cumulative':>14} {'Discounted':>14}")
    print("─" * 76)

    for cf in result.cash_flows[:11]:  # Show first 11 (year 0-10)
        print(f"{cf.year:<6} ${cf.inflow:>10,.2f} ${cf.outflow:>10,.2f} "
              f"${cf.net_flow:>10,.2f} ${cf.cumulative_flow:>12,.2f} "
              f"${cf.discounted_flow:>12,.2f}")


if __name__ == "__main__":
    # Run all examples
    basic_roi_example()
    complex_investment_example()
    sensitivity_analysis_example()
    cash_flow_example()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
