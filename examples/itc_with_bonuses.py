"""Example: ITC Calculation with Bonus Credits.

This example demonstrates how to calculate ITC with bonus credits for
domestic content and energy community adders.
"""

from datetime import date

from pv_simulator import IncentiveModeler
from pv_simulator.models import ITCConfiguration, SystemConfiguration


def main() -> None:
    """Run ITC calculation with bonus credits example."""
    print("=" * 70)
    print("ITC Calculation with Bonus Credits")
    print("=" * 70)

    # Define a commercial solar system
    system = SystemConfiguration(
        system_size_kw=2000.0,
        installation_cost_total=4_500_000.0,
        installation_date=date(2024, 9, 1),
        location_state="TX",
        expected_annual_production_kwh=3_000_000.0,
        system_lifetime_years=30,
    )

    print(f"\nSystem Details:")
    print(f"  Size: {system.system_size_kw:,.0f} kW")
    print(f"  Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  Location: {system.location_state}")

    modeler = IncentiveModeler()

    # Scenario 1: Base ITC only (30%)
    print(f"\n{'-' * 70}")
    print("Scenario 1: Base ITC Only")
    print(f"{'-' * 70}")

    config_base = ITCConfiguration(
        system_config=system,
        itc_rate=0.30,
        apply_bonus=False,
    )

    result_base = modeler.itc_calculation(config_base)
    print(f"  ITC Rate: {result_base.effective_rate:.1%}")
    print(f"  Total ITC Credit: ${result_base.total_itc_amount:,.2f}")

    # Scenario 2: ITC with Domestic Content Bonus
    print(f"\n{'-' * 70}")
    print("Scenario 2: ITC + Domestic Content Bonus")
    print(f"{'-' * 70}")

    config_domestic = ITCConfiguration(
        system_config=system,
        itc_rate=0.30,
        apply_bonus=True,
        bonus_rate=0.10,
        meets_domestic_content=True,
        is_energy_community=False,
    )

    result_domestic = modeler.itc_calculation(config_domestic)
    print(f"  Base ITC Rate: 30%")
    print(f"  Domestic Content Bonus: +10%")
    print(f"  Total ITC Rate: {result_domestic.effective_rate:.1%}")
    print(f"  Base ITC: ${result_domestic.base_itc:,.2f}")
    print(f"  Bonus ITC: ${result_domestic.bonus_itc:,.2f}")
    print(f"  Total ITC Credit: ${result_domestic.total_itc_amount:,.2f}")

    # Scenario 3: ITC with Both Bonuses (Domestic Content + Energy Community)
    print(f"\n{'-' * 70}")
    print("Scenario 3: ITC + Domestic Content + Energy Community Bonuses")
    print(f"{'-' * 70}")

    config_both = ITCConfiguration(
        system_config=system,
        itc_rate=0.30,
        apply_bonus=True,
        bonus_rate=0.10,
        meets_domestic_content=True,
        is_energy_community=True,
    )

    result_both = modeler.itc_calculation(config_both)
    print(f"  Base ITC Rate: 30%")
    print(f"  Domestic Content Bonus: +10%")
    print(f"  Energy Community Bonus: +10%")
    print(f"  Total ITC Rate: {result_both.effective_rate:.1%}")
    print(f"  Base ITC: ${result_both.base_itc:,.2f}")
    print(f"  Bonus ITC: ${result_both.bonus_itc:,.2f}")
    print(f"  Total ITC Credit: ${result_both.total_itc_amount:,.2f}")

    # Comparison
    print(f"\n{'-' * 70}")
    print("Bonus Credit Impact Comparison")
    print(f"{'-' * 70}")
    print(f"  Base ITC (30%): ${result_base.total_itc_amount:,.2f}")
    print(f"  With Domestic Content (40%): ${result_domestic.total_itc_amount:,.2f}")
    print(f"    Additional Credit: ${result_domestic.total_itc_amount - result_base.total_itc_amount:,.2f}")
    print(f"  With Both Bonuses (50%): ${result_both.total_itc_amount:,.2f}")
    print(f"    Additional Credit: ${result_both.total_itc_amount - result_base.total_itc_amount:,.2f}")

    print(f"\n{'-' * 70}")
    print("Net Installation Cost After ITC")
    print(f"{'-' * 70}")
    print(f"  Base (30% ITC): ${system.installation_cost_total - result_base.total_itc_amount:,.2f}")
    print(f"  With Domestic (40% ITC): ${system.installation_cost_total - result_domestic.total_itc_amount:,.2f}")
    print(f"  With Both (50% ITC): ${system.installation_cost_total - result_both.total_itc_amount:,.2f}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
