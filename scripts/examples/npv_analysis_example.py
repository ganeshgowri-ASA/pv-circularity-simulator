"""Example usage of NPVAnalyzer for PV project financial analysis.

This script demonstrates how to use the NPVAnalyzer to evaluate
the financial viability of a photovoltaic (PV) installation project.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pv_simulator.financial import (
    NPVAnalyzer,
    CashFlowInput,
    CashFlowProjection,
    DiscountRateConfig,
)


def example_1_simple_npv_irr() -> None:
    """Example 1: Simple NPV and IRR calculation."""
    print("=" * 80)
    print("Example 1: Simple NPV and IRR Calculation")
    print("=" * 80)

    # Create analyzer
    analyzer = NPVAnalyzer()

    # Define cash flows for a residential PV installation
    cash_flow_input = CashFlowInput(
        initial_investment=150000.0,  # $150k installation cost
        cash_flows=[
            25000.0,  # Year 1: Energy savings + sales
            26000.0,  # Year 2
            27000.0,  # Year 3
            28000.0,  # Year 4
            29000.0,  # Year 5
            30000.0,  # Year 6
            31000.0,  # Year 7
            32000.0,  # Year 8
            33000.0,  # Year 9
            34000.0,  # Year 10
        ],
        discount_rate=0.08,  # 8% discount rate
        project_name="Residential PV Installation"
    )

    # Perform analysis
    result = analyzer.analyze(cash_flow_input)

    # Display results
    print(f"\nProject: {result.project_name}")
    print(f"Initial Investment: ${cash_flow_input.initial_investment:,.2f}")
    print(f"Discount Rate: {result.discount_rate:.1%}")
    print(f"\nFinancial Metrics:")
    print(f"  NPV:                        ${result.npv:,.2f}")
    print(f"  IRR:                        {result.irr:.2%}" if result.irr else "  IRR:                        N/A")
    print(f"  Profitability Index:        {result.profitability_index:.2f}")
    print(f"  Payback Period:             {result.payback_period:.2f} years" if result.payback_period else "  Payback Period:             N/A")
    print(f"  Discounted Payback Period:  {result.discounted_payback_period:.2f} years" if result.discounted_payback_period else "  Discounted Payback Period:  N/A")
    print(f"  Total Cash Flows:           ${result.total_cash_flows:,.2f}")

    # Investment decision
    print(f"\nInvestment Decision:")
    if result.npv > 0:
        print("  ✓ ACCEPT - Positive NPV indicates the project adds value")
    else:
        print("  ✗ REJECT - Negative NPV indicates the project destroys value")

    if result.irr and result.irr > cash_flow_input.discount_rate:
        print(f"  ✓ IRR ({result.irr:.2%}) exceeds the discount rate ({cash_flow_input.discount_rate:.1%})")
    else:
        print(f"  ✗ IRR does not exceed the discount rate")

    print()


def example_2_sensitivity_analysis() -> None:
    """Example 2: Discount rate sensitivity analysis."""
    print("=" * 80)
    print("Example 2: Discount Rate Sensitivity Analysis")
    print("=" * 80)

    analyzer = NPVAnalyzer()

    # Same cash flows as Example 1
    cash_flow_input = CashFlowInput(
        initial_investment=150000.0,
        cash_flows=[25000.0 + i * 1000.0 for i in range(10)],
        discount_rate=0.08,
        project_name="PV Installation - Sensitivity Analysis"
    )

    # Configure sensitivity analysis
    rate_config = DiscountRateConfig(
        base_rate=0.08,
        min_rate=0.04,
        max_rate=0.16,
        step_size=0.02
    )

    # Perform sensitivity analysis
    sensitivity_result = analyzer.discount_rate_sensitivity(cash_flow_input, rate_config)

    # Display results
    print(f"\nSensitivity Analysis: {sensitivity_result.parameter_name}")
    print(f"Base Rate: {sensitivity_result.base_value:.1%}")
    print(f"Base NPV: ${sensitivity_result.base_npv:,.2f}")
    print(f"\nNPV Range:")
    print(f"  Minimum NPV: ${sensitivity_result.min_npv:,.2f}")
    print(f"  Maximum NPV: ${sensitivity_result.max_npv:,.2f}")
    print(f"  NPV Spread:  ${sensitivity_result.max_npv - sensitivity_result.min_npv:,.2f}")

    print(f"\nDetailed Sensitivity Results:")
    print(f"{'Discount Rate':>15} | {'NPV':>15} | {'Decision':>10}")
    print("-" * 45)
    for dp in sensitivity_result.data_points:
        decision = "ACCEPT" if dp.npv > 0 else "REJECT"
        print(f"{dp.parameter_value:>14.1%} | ${dp.npv:>13,.2f} | {decision:>10}")

    print()


def example_3_cash_flow_modeling() -> None:
    """Example 3: Comprehensive cash flow modeling."""
    print("=" * 80)
    print("Example 3: Comprehensive Cash Flow Modeling")
    print("=" * 80)

    analyzer = NPVAnalyzer()

    # Create detailed cash flow projection
    projection = CashFlowProjection(
        periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        revenues=[
            80000.0,   # Year 1: Energy sales
            82000.0,   # Year 2: 2.5% annual increase
            84050.0,   # Year 3
            86151.0,   # Year 4
            88305.0,   # Year 5
            90512.0,   # Year 6
            92774.0,   # Year 7
            95093.0,   # Year 8
            97470.0,   # Year 9
            99907.0,   # Year 10
        ],
        operating_costs=[
            15000.0,   # Year 1: Maintenance, insurance
            15450.0,   # Year 2: 3% annual increase
            15914.0,   # Year 3
            16391.0,   # Year 4
            16883.0,   # Year 5
            17389.0,   # Year 6
            17911.0,   # Year 7
            18448.0,   # Year 8
            19002.0,   # Year 9
            19572.0,   # Year 10
        ],
        capital_expenditures=[
            10000.0,   # Year 1: Initial monitoring system
            0.0,       # Year 2
            0.0,       # Year 3
            5000.0,    # Year 4: Minor equipment replacement
            0.0,       # Year 5
            0.0,       # Year 6
            8000.0,    # Year 7: Inverter replacement
            0.0,       # Year 8
            0.0,       # Year 9
            0.0,       # Year 10
        ],
        initial_investment=500000.0,  # $500k total installation
        terminal_value=200000.0       # Estimated salvage value
    )

    # Perform analysis
    result = analyzer.cash_flow_modeling(
        projection,
        discount_rate=0.08,
        project_name="Commercial PV Installation"
    )

    # Display projection details
    print(f"\nProject: {result.project_name}")
    print(f"Initial Investment: ${projection.initial_investment:,.2f}")
    print(f"Terminal Value: ${projection.terminal_value:,.2f}")
    print(f"Discount Rate: {result.discount_rate:.1%}")

    print(f"\nYearly Cash Flow Breakdown:")
    print(f"{'Year':>5} | {'Revenue':>12} | {'Op Costs':>12} | {'CapEx':>12} | {'Net CF':>12}")
    print("-" * 70)
    for i, period in enumerate(projection.periods):
        print(
            f"{period:>5} | "
            f"${projection.revenues[i]:>10,.0f} | "
            f"${projection.operating_costs[i]:>10,.0f} | "
            f"${projection.capital_expenditures[i]:>10,.0f} | "
            f"${projection.net_cash_flows[i]:>10,.0f}"
        )

    # Display financial metrics
    print(f"\nFinancial Analysis Results:")
    print(f"  NPV:                        ${result.npv:,.2f}")
    print(f"  IRR:                        {result.irr:.2%}" if result.irr else "  IRR:                        N/A")
    print(f"  Profitability Index:        {result.profitability_index:.2f}")
    print(f"  Payback Period:             {result.payback_period:.2f} years" if result.payback_period else "  Payback Period:             N/A")
    print(f"  Discounted Payback Period:  {result.discounted_payback_period:.2f} years" if result.discounted_payback_period else "  Discounted Payback Period:  N/A")

    # Investment recommendation
    print(f"\nInvestment Recommendation:")
    if result.npv > 0 and result.profitability_index > 1.0:
        print(f"  ✓✓ STRONGLY RECOMMEND")
        print(f"     - Positive NPV of ${result.npv:,.2f}")
        print(f"     - Profitability Index of {result.profitability_index:.2f} (> 1.0)")
        if result.irr:
            print(f"     - IRR of {result.irr:.2%} exceeds discount rate")
    elif result.npv > 0:
        print(f"  ✓ RECOMMEND with caution")
        print(f"     - Positive NPV but consider other factors")
    else:
        print(f"  ✗ DO NOT RECOMMEND")
        print(f"     - Negative NPV of ${result.npv:,.2f}")

    print()


def example_4_direct_calculations() -> None:
    """Example 4: Using individual calculation methods."""
    print("=" * 80)
    print("Example 4: Direct NPV and IRR Calculations")
    print("=" * 80)

    analyzer = NPVAnalyzer()

    # Simple project data
    initial_investment = 100000.0
    cash_flows = [30000.0, 35000.0, 40000.0, 45000.0]

    # Calculate NPV directly
    npv = analyzer.npv_calculation(
        cash_flows=cash_flows,
        discount_rate=0.10,
        initial_investment=initial_investment
    )

    # Calculate IRR directly
    irr = analyzer.irr_computation(
        cash_flows=cash_flows,
        initial_investment=initial_investment
    )

    print(f"\nDirect Calculation Results:")
    print(f"Initial Investment: ${initial_investment:,.2f}")
    print(f"Cash Flows: {[f'${cf:,.0f}' for cf in cash_flows]}")
    print(f"\nCalculated Metrics:")
    print(f"  NPV @ 10%: ${npv:,.2f}")
    print(f"  IRR:       {irr:.2%}")

    # Test different discount rates
    print(f"\nNPV at Different Discount Rates:")
    for rate in [0.05, 0.10, 0.15, 0.20, 0.25]:
        npv_at_rate = analyzer.npv_calculation(
            cash_flows=cash_flows,
            discount_rate=rate,
            initial_investment=initial_investment
        )
        print(f"  {rate:>5.1%}: ${npv_at_rate:>12,.2f}")

    print()


def main() -> None:
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NPV & IRR Analysis Engine Examples" + " " * 24 + "║")
    print("║" + " " * 15 + "PV Circularity Simulator - Financial Module" + " " * 19 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run examples
    example_1_simple_npv_irr()
    example_2_sensitivity_analysis()
    example_3_cash_flow_modeling()
    example_4_direct_calculations()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
