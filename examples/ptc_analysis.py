"""Example: Production Tax Credit (PTC) Analysis.

This example demonstrates how to calculate the Production Tax Credit (PTC)
over a 10-year credit period with production degradation and inflation adjustment.
"""

from datetime import date

from pv_simulator import IncentiveModeler
from pv_simulator.models import PTCConfiguration, SystemConfiguration


def main() -> None:
    """Run PTC analysis example."""
    print("=" * 80)
    print("Production Tax Credit (PTC) Analysis")
    print("=" * 80)

    # Define a large commercial solar system
    system = SystemConfiguration(
        system_size_kw=5000.0,
        installation_cost_total=11_000_000.0,
        installation_date=date(2024, 1, 1),
        location_state="AZ",
        expected_annual_production_kwh=9_500_000.0,
        system_lifetime_years=30,
    )

    print(f"\nSystem Details:")
    print(f"  Size: {system.system_size_kw:,.0f} kW")
    print(f"  Expected Year 1 Production: {system.expected_annual_production_kwh:,.0f} kWh")
    print(f"  Location: {system.location_state}")

    modeler = IncentiveModeler()

    # Configure PTC with standard parameters
    ptc_config = PTCConfiguration(
        system_config=system,
        ptc_rate_per_kwh=0.0275,  # $0.0275 per kWh
        credit_period_years=10,
        inflation_adjustment=True,
        inflation_rate=0.025,
        production_degradation_rate=0.005,
        apply_bonus=False,
    )

    # Calculate PTC
    result = modeler.ptc_computation(ptc_config, discount_rate=0.06)

    # Display results
    print(f"\n{'-' * 80}")
    print("PTC Configuration:")
    print(f"{'-' * 80}")
    print(f"  Base PTC Rate: ${ptc_config.ptc_rate_per_kwh:.4f} per kWh")
    print(f"  Credit Period: {ptc_config.credit_period_years} years")
    print(f"  Inflation Adjustment: {ptc_config.inflation_adjustment}")
    print(f"  Annual Inflation Rate: {ptc_config.inflation_rate:.1%}")
    print(f"  Production Degradation: {ptc_config.production_degradation_rate:.1%} per year")

    print(f"\n{'-' * 80}")
    print("Annual PTC Credits by Year:")
    print(f"{'-' * 80}")
    print(f"{'Year':<8} {'Production (kWh)':>18} {'PTC Rate':>12} {'Annual Credit':>18}")
    print(f"{'-' * 80}")

    for year in range(result.credit_period_years):
        production = result.annual_production[year]
        credit = result.annual_credits[year]
        effective_rate = credit / production if production > 0 else 0

        print(
            f"{year + 1:<8} {production:>18,.0f} "
            f"${effective_rate:>11.4f} ${credit:>17,.2f}"
        )

    print(f"{'-' * 80}")
    print(f"{'Total':>27} ${result.total_ptc_lifetime:>17,.2f}")

    print(f"\n{'-' * 80}")
    print("Financial Summary:")
    print(f"{'-' * 80}")
    print(f"  Total PTC (Nominal): ${result.total_ptc_lifetime:,.2f}")
    print(f"  Present Value (6% discount): ${result.present_value_ptc:,.2f}")
    print(f"  First Year Credit: ${result.first_year_credit:,.2f}")
    print(f"  Last Year Credit: ${result.last_year_credit:,.2f}")
    print(f"  Average Annual Credit: ${result.total_ptc_lifetime / result.credit_period_years:,.2f}")

    print(f"\n{'-' * 80}")
    print("Production Analysis:")
    print(f"{'-' * 80}")
    total_production = sum(result.annual_production)
    avg_production = total_production / len(result.annual_production)
    production_decline = (
        (result.annual_production[0] - result.annual_production[-1])
        / result.annual_production[0]
    )

    print(f"  Total 10-Year Production: {total_production:,.0f} kWh")
    print(f"  Average Annual Production: {avg_production:,.0f} kWh")
    print(f"  Total Production Decline: {production_decline:.1%}")
    print(f"  Effective PTC Rate: ${result.total_ptc_lifetime / total_production:.4f} per kWh")

    # Compare with bonus multiplier scenario
    print(f"\n{'-' * 80}")
    print("Comparison: PTC with 5x Bonus Multiplier")
    print(f"{'-' * 80}")

    ptc_config_bonus = PTCConfiguration(
        system_config=system,
        ptc_rate_per_kwh=0.0275,
        credit_period_years=10,
        inflation_adjustment=True,
        inflation_rate=0.025,
        production_degradation_rate=0.005,
        apply_bonus=True,
        bonus_multiplier=5.0,
    )

    result_bonus = modeler.ptc_computation(ptc_config_bonus, discount_rate=0.06)

    print(f"  Standard PTC (Nominal): ${result.total_ptc_lifetime:,.2f}")
    print(f"  PTC with 5x Bonus (Nominal): ${result_bonus.total_ptc_lifetime:,.2f}")
    print(f"  Additional Credit: ${result_bonus.total_ptc_lifetime - result.total_ptc_lifetime:,.2f}")
    print(f"\n  Standard PTC (NPV): ${result.present_value_ptc:,.2f}")
    print(f"  PTC with 5x Bonus (NPV): ${result_bonus.present_value_ptc:,.2f}")
    print(f"  Additional NPV: ${result_bonus.present_value_ptc - result.present_value_ptc:,.2f}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
