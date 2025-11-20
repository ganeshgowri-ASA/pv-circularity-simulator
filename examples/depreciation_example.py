"""Example: MACRS Depreciation Schedule Analysis.

This example demonstrates how to calculate depreciation schedules using
MACRS 5-year method with bonus depreciation.
"""

from pv_simulator import IncentiveModeler
from pv_simulator.models import DepreciationMethod


def main() -> None:
    """Run depreciation schedule analysis example."""
    print("=" * 80)
    print("MACRS Depreciation Schedule Analysis")
    print("=" * 80)

    modeler = IncentiveModeler()

    # Scenario: $1M solar installation with 30% ITC
    installation_cost = 1_000_000.0
    itc_amount = installation_cost * 0.30

    # Asset basis for depreciation = cost - 50% of ITC (ITC basis adjustment)
    asset_basis = installation_cost - (0.5 * itc_amount)

    print(f"\nInstallation Cost: ${installation_cost:,.2f}")
    print(f"ITC (30%): ${itc_amount:,.2f}")
    print(f"ITC Basis Adjustment (50% of ITC): ${0.5 * itc_amount:,.2f}")
    print(f"Depreciable Basis: ${asset_basis:,.2f}")

    # Scenario 1: MACRS 5-year without bonus depreciation
    print(f"\n{'-' * 80}")
    print("Scenario 1: MACRS 5-Year (No Bonus Depreciation)")
    print(f"{'-' * 80}")

    result_no_bonus = modeler.depreciation_schedule(
        asset_basis=asset_basis,
        method=DepreciationMethod.MACRS_5,
        bonus_depreciation_rate=0.0,
    )

    print(f"\nMethod: {result_no_bonus.method.value}")
    print(f"Convention: {result_no_bonus.macrs_convention}")
    print(f"Schedule Years: {result_no_bonus.schedule_years}")

    print(f"\n{'Year':<8} {'Depreciation':>18} {'Cumulative':>18} {'Remaining Basis':>18}")
    print(f"{'-' * 80}")

    for year in range(result_no_bonus.schedule_years):
        depr = result_no_bonus.annual_depreciation[year]
        cumul = result_no_bonus.cumulative_depreciation[year]
        remain = result_no_bonus.remaining_basis[year]
        print(f"{year + 1:<8} ${depr:>17,.2f} ${cumul:>17,.2f} ${remain:>17,.2f}")

    print(f"{'-' * 80}")
    print(f"{'Total':<8} ${result_no_bonus.total_depreciation:>17,.2f}")

    # Scenario 2: MACRS 5-year with 60% bonus depreciation
    print(f"\n{'-' * 80}")
    print("Scenario 2: MACRS 5-Year with 60% Bonus Depreciation")
    print(f"{'-' * 80}")

    result_with_bonus = modeler.depreciation_schedule(
        asset_basis=asset_basis,
        method=DepreciationMethod.MACRS_5,
        bonus_depreciation_rate=0.60,
    )

    print(f"\nBonus Depreciation Rate: {result_with_bonus.bonus_depreciation_rate:.0%}")
    print(f"Bonus Depreciation Amount: ${result_with_bonus.bonus_depreciation_amount:,.2f}")
    print(f"Remaining Basis After Bonus: ${asset_basis - result_with_bonus.bonus_depreciation_amount:,.2f}")

    print(f"\n{'Year':<8} {'Depreciation':>18} {'Cumulative':>18} {'Remaining Basis':>18}")
    print(f"{'-' * 80}")

    for year in range(result_with_bonus.schedule_years):
        depr = result_with_bonus.annual_depreciation[year]
        cumul = result_with_bonus.cumulative_depreciation[year]
        remain = result_with_bonus.remaining_basis[year]

        year_label = f"{year + 1}"
        if year == 0:
            year_label += "*"  # Mark first year with bonus

        print(f"{year_label:<8} ${depr:>17,.2f} ${cumul:>17,.2f} ${remain:>17,.2f}")

    print(f"{'-' * 80}")
    print(f"{'Total':<8} ${result_with_bonus.total_depreciation:>17,.2f}")
    print(f"\n* Year 1 includes ${result_with_bonus.bonus_depreciation_amount:,.2f} bonus depreciation")

    # Tax benefit comparison (assuming 40% tax rate)
    tax_rate = 0.40

    print(f"\n{'-' * 80}")
    print("Tax Benefit Comparison (40% Tax Rate):")
    print(f"{'-' * 80}")

    # Without bonus depreciation
    tax_benefit_year1_no_bonus = result_no_bonus.annual_depreciation[0] * tax_rate
    tax_benefit_year1_with_bonus = result_with_bonus.annual_depreciation[0] * tax_rate

    print(f"\nYear 1 Depreciation:")
    print(f"  Without Bonus: ${result_no_bonus.annual_depreciation[0]:,.2f}")
    print(f"    Tax Shield: ${tax_benefit_year1_no_bonus:,.2f}")
    print(f"\n  With 60% Bonus: ${result_with_bonus.annual_depreciation[0]:,.2f}")
    print(f"    Tax Shield: ${tax_benefit_year1_with_bonus:,.2f}")
    print(f"\n  Additional Year 1 Tax Benefit: ${tax_benefit_year1_with_bonus - tax_benefit_year1_no_bonus:,.2f}")

    # Total tax shields
    total_tax_shield_no_bonus = result_no_bonus.total_depreciation * tax_rate
    total_tax_shield_with_bonus = result_with_bonus.total_depreciation * tax_rate

    print(f"\nTotal Depreciation Tax Shield:")
    print(f"  Without Bonus: ${total_tax_shield_no_bonus:,.2f}")
    print(f"  With 60% Bonus: ${total_tax_shield_with_bonus:,.2f}")
    print(f"  (Note: Total is same, but timing differs)")

    # NPV comparison (assuming 6% discount rate)
    discount_rate = 0.06
    npv_no_bonus = sum(
        (depr * tax_rate) / ((1 + discount_rate) ** year)
        for year, depr in enumerate(result_no_bonus.annual_depreciation)
    )
    npv_with_bonus = sum(
        (depr * tax_rate) / ((1 + discount_rate) ** year)
        for year, depr in enumerate(result_with_bonus.annual_depreciation)
    )

    print(f"\nNet Present Value of Tax Shields (6% discount rate):")
    print(f"  Without Bonus: ${npv_no_bonus:,.2f}")
    print(f"  With 60% Bonus: ${npv_with_bonus:,.2f}")
    print(f"  NPV Advantage from Bonus: ${npv_with_bonus - npv_no_bonus:,.2f}")

    # Comparison with MACRS 7-year
    print(f"\n{'-' * 80}")
    print("Comparison: MACRS 5-Year vs MACRS 7-Year")
    print(f"{'-' * 80}")

    result_macrs7 = modeler.depreciation_schedule(
        asset_basis=asset_basis,
        method=DepreciationMethod.MACRS_7,
        bonus_depreciation_rate=0.0,
    )

    print(f"\nMACRS 5-Year:")
    print(f"  Schedule Years: {result_no_bonus.schedule_years}")
    print(f"  Year 1 Depreciation: ${result_no_bonus.annual_depreciation[0]:,.2f}")
    print(f"  Year 1 as % of Basis: {result_no_bonus.annual_depreciation[0] / asset_basis:.1%}")

    print(f"\nMACRS 7-Year:")
    print(f"  Schedule Years: {result_macrs7.schedule_years}")
    print(f"  Year 1 Depreciation: ${result_macrs7.annual_depreciation[0]:,.2f}")
    print(f"  Year 1 as % of Basis: {result_macrs7.annual_depreciation[0] / asset_basis:.1%}")

    print(f"\nRecommendation: MACRS 5-year is standard for solar installations")
    print(f"and provides faster tax benefits.")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
