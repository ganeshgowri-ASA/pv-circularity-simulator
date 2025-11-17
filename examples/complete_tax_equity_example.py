"""Example: Complete Tax Equity Partnership Flip Analysis.

This example demonstrates a complete tax equity analysis including:
- ITC calculation
- MACRS depreciation with bonus depreciation
- Partnership flip structure modeling
- IRR and NPV calculations for investor and sponsor
"""

from datetime import date

from pv_simulator import IncentiveModeler
from pv_simulator.models import (
    DepreciationMethod,
    ITCConfiguration,
    SystemConfiguration,
    TaxEquityConfiguration,
)


def main() -> None:
    """Run complete tax equity partnership flip analysis."""
    print("=" * 90)
    print("Complete Tax Equity Partnership Flip Analysis")
    print("=" * 90)

    # Define a 5MW commercial solar installation
    system = SystemConfiguration(
        system_size_kw=5000.0,
        installation_cost_total=11_000_000.0,
        installation_date=date(2024, 3, 1),
        location_state="NC",
        expected_annual_production_kwh=8_500_000.0,
        system_lifetime_years=25,
        module_efficiency=0.21,
        inverter_efficiency=0.98,
    )

    print(f"\nProject Details:")
    print(f"  System Size: {system.system_size_kw:,.0f} kW ({system.system_size_kw / 1000:.1f} MW)")
    print(f"  Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  Cost per Watt: ${system.installation_cost_total / (system.system_size_kw * 1000):.2f}/W")
    print(f"  Location: {system.location_state}")
    print(f"  Expected Annual Production: {system.expected_annual_production_kwh:,.0f} kWh")
    print(f"  Capacity Factor: {(system.expected_annual_production_kwh / (system.system_size_kw * 8760)):.1%}")

    modeler = IncentiveModeler()

    # Step 1: Calculate ITC with domestic content bonus
    print(f"\n{'-' * 90}")
    print("STEP 1: Investment Tax Credit (ITC) Calculation")
    print(f"{'-' * 90}")

    itc_config = ITCConfiguration(
        system_config=system,
        itc_rate=0.30,
        apply_bonus=True,
        bonus_rate=0.10,
        meets_domestic_content=True,
        is_energy_community=False,
    )

    itc_result = modeler.itc_calculation(itc_config)

    print(f"\nITC Configuration:")
    print(f"  Base ITC Rate: 30%")
    print(f"  Domestic Content Bonus: +10%")
    print(f"  Total ITC Rate: {itc_result.effective_rate:.0%}")

    print(f"\nITC Results:")
    print(f"  Eligible Basis: ${itc_result.eligible_basis:,.2f}")
    print(f"  Base ITC (30%): ${itc_result.base_itc:,.2f}")
    print(f"  Domestic Content Bonus (10%): ${itc_result.bonus_itc:,.2f}")
    print(f"  Total ITC Credit: ${itc_result.total_itc_amount:,.2f}")

    # Step 2: Calculate depreciation with ITC basis adjustment and bonus depreciation
    print(f"\n{'-' * 90}")
    print("STEP 2: MACRS Depreciation Schedule")
    print(f"{'-' * 90}")

    # ITC basis adjustment: reduce depreciable basis by 50% of ITC
    depreciable_basis = system.installation_cost_total - (0.5 * itc_result.total_itc_amount)

    print(f"\nDepreciation Basis Calculation:")
    print(f"  Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  Less: 50% ITC Adjustment: -${0.5 * itc_result.total_itc_amount:,.2f}")
    print(f"  Depreciable Basis: ${depreciable_basis:,.2f}")

    depr_result = modeler.depreciation_schedule(
        asset_basis=depreciable_basis,
        method=DepreciationMethod.MACRS_5,
        bonus_depreciation_rate=0.80,  # 80% bonus depreciation
    )

    print(f"\nDepreciation Schedule (MACRS 5-Year with 80% Bonus):")
    print(f"  Bonus Depreciation (Year 1): ${depr_result.bonus_depreciation_amount:,.2f}")
    print(f"\n  {'Year':<6} {'Annual Depreciation':>22} {'Cumulative':>18}")
    print(f"  {'-' * 46}")

    for year in range(min(6, len(depr_result.annual_depreciation))):
        depr = depr_result.annual_depreciation[year]
        cumul = depr_result.cumulative_depreciation[year]
        year_label = f"{year + 1}" + ("*" if year == 0 else "")
        print(f"  {year_label:<6} ${depr:>21,.2f} ${cumul:>17,.2f}")

    print(f"  {'-' * 46}")
    print(f"  {'Total':<6} ${depr_result.total_depreciation:>21,.2f}")

    # Step 3: Tax Equity Partnership Flip Structure
    print(f"\n{'-' * 90}")
    print("STEP 3: Tax Equity Partnership Flip Modeling")
    print(f"{'-' * 90}")

    te_config = TaxEquityConfiguration(
        system_config=system,
        investor_equity_percentage=0.99,
        target_flip_irr=0.08,
        post_flip_investor_percentage=0.05,
        tax_rate=0.40,
        project_lifetime_years=25,
        discount_rate=0.06,
        include_itc=True,
        include_depreciation=True,
    )

    print(f"\nPartnership Structure:")
    print(f"  Tax Equity Investor:")
    print(f"    Pre-Flip Ownership: {te_config.investor_equity_percentage:.0%}")
    print(f"    Post-Flip Ownership: {te_config.post_flip_investor_percentage:.0%}")
    print(f"    Target IRR: {te_config.target_flip_irr:.1%}")
    print(f"  Sponsor/Developer:")
    print(f"    Pre-Flip Ownership: {1 - te_config.investor_equity_percentage:.0%}")
    print(f"    Post-Flip Ownership: {1 - te_config.post_flip_investor_percentage:.0%}")
    print(f"\n  Tax Rate: {te_config.tax_rate:.0%}")
    print(f"  Project Lifetime: {te_config.project_lifetime_years} years")

    te_result = modeler.tax_equity_modeling(
        config=te_config,
        itc_amount=itc_result.total_itc_amount,
        depreciation_schedule=depr_result.annual_depreciation,
    )

    print(f"\n{'-' * 90}")
    print("Partnership Flip Results:")
    print(f"{'-' * 90}")

    print(f"\nFlip Timing:")
    print(f"  Partnership Flip Year: Year {te_result.flip_year}")
    print(f"  Pre-Flip Period: {te_result.pre_flip_years} years")
    print(f"  Post-Flip Period: {te_result.post_flip_years} years")

    print(f"\nInvestor Returns:")
    print(f"  Internal Rate of Return (IRR): {te_result.investor_irr:.2%}")
    print(f"  Net Present Value (NPV): ${te_result.investor_npv:,.2f}")
    print(f"  Total Tax Benefits: ${te_result.total_investor_benefit:,.2f}")

    print(f"\nSponsor/Developer Returns:")
    print(f"  Internal Rate of Return (IRR): {te_result.sponsor_irr:.2%}")
    print(f"  Net Present Value (NPV): ${te_result.sponsor_npv:,.2f}")
    print(f"  Total Tax Benefits: ${te_result.total_sponsor_benefit:,.2f}")

    print(f"\nTotal Partnership:")
    print(f"  Combined Tax Benefits: ${te_result.total_tax_benefits:,.2f}")
    print(f"  Combined NPV: ${te_result.investor_npv + te_result.sponsor_npv:,.2f}")

    # Detailed annual breakdown (first 10 years)
    print(f"\n{'-' * 90}")
    print("Annual Cash Flow and Tax Benefit Allocation (First 10 Years):")
    print(f"{'-' * 90}")
    print(
        f"\n{'Year':<6} {'Investor Tax':>15} {'Sponsor Tax':>15} "
        f"{'Investor Cash':>16} {'Sponsor Cash':>15}"
    )
    print(f"{'-' * 90}")

    for year in range(min(10, len(te_result.annual_tax_benefits_investor))):
        inv_tax = te_result.annual_tax_benefits_investor[year]
        spon_tax = te_result.annual_tax_benefits_sponsor[year]
        inv_cash = te_result.annual_cash_flows_investor[year]
        spon_cash = te_result.annual_cash_flows_sponsor[year]

        year_label = f"{year + 1}"
        if year == te_result.flip_year:
            year_label += " <FLIP>"

        print(
            f"{year_label:<6} ${inv_tax:>14,.0f} ${spon_tax:>14,.0f} "
            f"${inv_cash:>15,.0f} ${spon_cash:>14,.0f}"
        )

    # Summary of tax benefits
    print(f"\n{'-' * 90}")
    print("Tax Benefit Summary:")
    print(f"{'-' * 90}")

    # Tax benefits breakdown
    itc_to_investor = itc_result.total_itc_amount * te_config.investor_equity_percentage
    itc_to_sponsor = itc_result.total_itc_amount * (1 - te_config.investor_equity_percentage)

    depr_tax_shield = depr_result.total_depreciation * te_config.tax_rate

    print(f"\nInvestment Tax Credit (Year 0):")
    print(f"  Total ITC: ${itc_result.total_itc_amount:,.2f}")
    print(f"    To Investor ({te_config.investor_equity_percentage:.0%}): ${itc_to_investor:,.2f}")
    print(f"    To Sponsor ({1 - te_config.investor_equity_percentage:.0%}): ${itc_to_sponsor:,.2f}")

    print(f"\nDepreciation Tax Shield:")
    print(f"  Total Depreciation: ${depr_result.total_depreciation:,.2f}")
    print(f"  Tax Shield (@ {te_config.tax_rate:.0%}): ${depr_tax_shield:,.2f}")
    print(f"  Allocated based on partnership %s through flip period")

    print(f"\nTotal Project Economics:")
    print(f"  Installation Cost: ${system.installation_cost_total:,.2f}")
    print(f"  ITC Credit: ${itc_result.total_itc_amount:,.2f}")
    print(f"  Depreciation Tax Shield: ${depr_tax_shield:,.2f}")
    print(f"  Total Tax Benefits: ${itc_result.total_itc_amount + depr_tax_shield:,.2f}")
    print(f"  Net Project Cost: ${system.installation_cost_total - itc_result.total_itc_amount - depr_tax_shield:,.2f}")
    print(
        f"  Effective Tax Benefit: {(itc_result.total_itc_amount + depr_tax_shield) / system.installation_cost_total:.1%}"
    )

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    main()
