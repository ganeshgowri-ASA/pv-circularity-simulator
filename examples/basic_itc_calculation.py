"""Example: Basic Investment Tax Credit (ITC) Calculation.

This example demonstrates how to calculate the Investment Tax Credit (ITC)
for a commercial solar installation.
"""

from datetime import date

from pv_simulator import IncentiveModeler
from pv_simulator.models import ITCConfiguration, SystemConfiguration


def main() -> None:
    """Run basic ITC calculation example."""
    print("=" * 70)
    print("Basic Investment Tax Credit (ITC) Calculation")
    print("=" * 70)

    # Define a commercial solar system
    system = SystemConfiguration(
        system_size_kw=500.0,
        installation_cost_total=1_250_000.0,
        installation_date=date(2024, 6, 15),
        location_state="CA",
        expected_annual_production_kwh=750_000.0,
        system_lifetime_years=25,
        module_efficiency=0.21,
        inverter_efficiency=0.97,
    )

    print(f"\nSystem Details:")
    print(f"  Size: {system.system_size_kw:,.0f} kW")
    print(f"  Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  Location: {system.location_state}")
    print(f"  Annual Production: {system.expected_annual_production_kwh:,.0f} kWh")

    # Configure ITC calculation (standard 30% ITC)
    itc_config = ITCConfiguration(
        system_config=system,
        itc_rate=0.30,
        apply_bonus=False,
    )

    # Create modeler and calculate ITC
    modeler = IncentiveModeler()
    result = modeler.itc_calculation(itc_config)

    # Display results
    print(f"\n{'-' * 70}")
    print("ITC Calculation Results:")
    print(f"{'-' * 70}")
    print(f"  Eligible Basis: ${result.eligible_basis:,.2f}")
    print(f"  ITC Rate: {result.effective_rate:.1%}")
    print(f"  Base ITC Amount: ${result.base_itc:,.2f}")
    print(f"  Bonus ITC Amount: ${result.bonus_itc:,.2f}")
    print(f"  Total ITC Credit: ${result.total_itc_amount:,.2f}")
    print(f"  Recapture Period: {result.recapture_period_years} years")
    print(f"\n  Cost per Watt: ${result.calculation_details['cost_per_watt']:.2f}/W")
    print(f"  Effective ITC Reduction: ${result.total_itc_amount / system.system_size_kw:,.2f}/kW")

    print(f"\n{'-' * 70}")
    print("Financial Impact:")
    print(f"{'-' * 70}")
    net_cost = system.installation_cost_total - result.total_itc_amount
    print(f"  Gross Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  ITC Credit: -${result.total_itc_amount:,.2f}")
    print(f"  Net Cost After ITC: ${net_cost:,.2f}")
    print(f"  Effective Cost Reduction: {(result.total_itc_amount / system.installation_cost_total):.1%}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
