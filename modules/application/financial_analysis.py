"""
Financial Analysis & Bankability Module (Branch B13).

Features:
- LCOE calculation (levelized cost of energy)
- NPV analysis (net present value)
- IRR calculation (internal rate of return)
- Payback period (simple and discounted)
- Cash flow projections (25-year)
- Tax benefits and incentives (ITC, depreciation - MACRS)
- Sensitivity analysis (electricity price, discount rate, degradation)
- Debt financing and DSCR
- P50/P90 bankability metrics
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from utils.constants import FINANCIAL_DEFAULTS, MACRS_SCHEDULES
from utils.validators import FinancialAnalysis
from utils.helpers import (
    calculate_lcoe,
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    format_number
)


class FinancialAnalyzer:
    """Comprehensive financial analysis and bankability assessment."""

    def __init__(self):
        """Initialize financial analyzer with default parameters."""
        self.defaults = FINANCIAL_DEFAULTS
        self.macrs_schedules = MACRS_SCHEDULES

    def calculate_project_capex(
        self,
        system_capacity_kw: float,
        module_cost_per_wp: float = 0.45,
        inverter_cost_per_kw: float = 150,
        bos_cost_per_kw: float = 300,
        soft_costs_per_kw: float = 200,
        contingency_pct: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate total project capital expenditure.

        Args:
            system_capacity_kw: System capacity (kW)
            module_cost_per_wp: Module cost ($/Wp)
            inverter_cost_per_kw: Inverter cost ($/kW)
            bos_cost_per_kw: Balance of system cost ($/kW)
            soft_costs_per_kw: Soft costs ($/kW)
            contingency_pct: Contingency percentage

        Returns:
            Dictionary with cost breakdown
        """
        module_cost = system_capacity_kw * 1000 * module_cost_per_wp
        inverter_cost = system_capacity_kw * inverter_cost_per_kw
        bos_cost = system_capacity_kw * bos_cost_per_kw
        soft_costs = system_capacity_kw * soft_costs_per_kw

        subtotal = module_cost + inverter_cost + bos_cost + soft_costs
        contingency = subtotal * contingency_pct
        total_capex = subtotal + contingency

        return {
            'module_cost': module_cost,
            'inverter_cost': inverter_cost,
            'bos_cost': bos_cost,
            'soft_costs': soft_costs,
            'contingency': contingency,
            'total_capex': total_capex,
            'cost_per_watt': total_capex / (system_capacity_kw * 1000)
        }

    def calculate_annual_opex(
        self,
        system_capacity_kw: float,
        o_and_m_per_kw: float = 15,
        insurance_rate: float = 0.005,
        property_tax_rate: float = 0.01,
        total_capex: float = 0
    ) -> Dict[str, float]:
        """
        Calculate annual operating expenditure.

        Args:
            system_capacity_kw: System capacity (kW)
            o_and_m_per_kw: O&M cost ($/kW/year)
            insurance_rate: Insurance rate (fraction of CAPEX)
            property_tax_rate: Property tax rate (fraction of CAPEX)
            total_capex: Total capital expenditure ($)

        Returns:
            Dictionary with OPEX breakdown
        """
        o_and_m = system_capacity_kw * o_and_m_per_kw
        insurance = total_capex * insurance_rate
        property_tax = total_capex * property_tax_rate

        total_opex = o_and_m + insurance + property_tax

        return {
            'o_and_m': o_and_m,
            'insurance': insurance,
            'property_tax': property_tax,
            'total_opex': total_opex
        }

    def calculate_lcoe_detailed(
        self,
        total_capex: float,
        annual_energy_kwh: float,
        annual_opex: float,
        discount_rate: float = 0.08,
        lifetime_years: int = 25,
        degradation_rate: float = 0.005,
        federal_itc: float = 0.30
    ) -> Dict[str, float]:
        """
        Calculate detailed LCOE with tax benefits.

        Args:
            total_capex: Total capital expenditure ($)
            annual_energy_kwh: First-year energy production (kWh)
            annual_opex: Annual operating expenditure ($)
            discount_rate: Discount rate (fraction)
            lifetime_years: Project lifetime (years)
            degradation_rate: Annual degradation rate (fraction)
            federal_itc: Federal ITC rate (fraction)

        Returns:
            Dictionary with LCOE components
        """
        # Net CAPEX after ITC
        net_capex = total_capex * (1 - federal_itc)

        # Calculate standard LCOE
        lcoe = calculate_lcoe(
            net_capex,
            annual_energy_kwh,
            annual_opex,
            discount_rate,
            lifetime_years,
            degradation_rate
        )

        # Calculate LCOE without ITC for comparison
        lcoe_no_itc = calculate_lcoe(
            total_capex,
            annual_energy_kwh,
            annual_opex,
            discount_rate,
            lifetime_years,
            degradation_rate
        )

        # Calculate lifetime energy production
        total_energy = sum(
            annual_energy_kwh * ((1 - degradation_rate) ** year)
            for year in range(lifetime_years)
        )

        return {
            'lcoe': lcoe,
            'lcoe_no_itc': lcoe_no_itc,
            'itc_benefit': lcoe_no_itc - lcoe,
            'net_capex': net_capex,
            'total_energy_lifetime': total_energy,
            'avg_annual_energy': total_energy / lifetime_years
        }

    def calculate_cash_flows(
        self,
        annual_energy_kwh: float,
        electricity_price: float,
        annual_opex: float,
        lifetime_years: int = 25,
        degradation_rate: float = 0.005,
        electricity_escalation: float = 0.025
    ) -> pd.DataFrame:
        """
        Calculate annual cash flows over project lifetime.

        Args:
            annual_energy_kwh: First-year energy production (kWh)
            electricity_price: Initial electricity price ($/kWh)
            annual_opex: Annual operating expenditure ($)
            lifetime_years: Project lifetime (years)
            degradation_rate: Annual degradation rate (fraction)
            electricity_escalation: Annual electricity price escalation (fraction)

        Returns:
            DataFrame with annual cash flows
        """
        years = np.arange(1, lifetime_years + 1)

        # Energy production with degradation
        energy_production = annual_energy_kwh * ((1 - degradation_rate) ** (years - 1))

        # Electricity price with escalation
        electricity_prices = electricity_price * ((1 + electricity_escalation) ** (years - 1))

        # Revenue
        revenue = energy_production * electricity_prices

        # Operating expenses
        opex = np.full(lifetime_years, annual_opex)

        # Cash flow
        cash_flow = revenue - opex

        # Cumulative cash flow
        cumulative_cash_flow = np.cumsum(cash_flow)

        df = pd.DataFrame({
            'year': years,
            'energy_kwh': energy_production,
            'electricity_price': electricity_prices,
            'revenue': revenue,
            'opex': opex,
            'cash_flow': cash_flow,
            'cumulative_cash_flow': cumulative_cash_flow
        })

        return df

    def calculate_npv_irr(
        self,
        total_capex: float,
        cash_flows: List[float],
        discount_rate: float = 0.08,
        federal_itc: float = 0.30
    ) -> Dict[str, float]:
        """
        Calculate NPV and IRR with tax benefits.

        Args:
            total_capex: Total capital expenditure ($)
            cash_flows: Annual cash flows ($)
            discount_rate: Discount rate (fraction)
            federal_itc: Federal ITC rate (fraction)

        Returns:
            Dictionary with NPV and IRR
        """
        # Net CAPEX after ITC (received in Year 1)
        net_capex = total_capex * (1 - federal_itc)
        itc_benefit = total_capex * federal_itc

        # Adjust first year cash flow for ITC
        adjusted_cash_flows = cash_flows.copy()
        adjusted_cash_flows[0] += itc_benefit

        # Calculate NPV
        npv = calculate_npv(adjusted_cash_flows, discount_rate, net_capex)

        # Calculate IRR
        irr = calculate_irr(adjusted_cash_flows, net_capex)

        # Calculate payback periods
        simple_payback = calculate_payback_period(net_capex, adjusted_cash_flows)

        # Discounted payback
        discounted_flows = [
            cf / ((1 + discount_rate) ** (i + 1))
            for i, cf in enumerate(adjusted_cash_flows)
        ]
        discounted_payback = calculate_payback_period(net_capex, discounted_flows)

        return {
            'npv': npv,
            'irr': irr if irr is not None else 0,
            'simple_payback': simple_payback if simple_payback is not None else float('inf'),
            'discounted_payback': discounted_payback if discounted_payback is not None else float('inf'),
            'net_capex': net_capex,
            'itc_benefit': itc_benefit
        }

    def calculate_macrs_depreciation(
        self,
        depreciable_basis: float,
        schedule: str = "MACRS_5"
    ) -> pd.DataFrame:
        """
        Calculate MACRS depreciation schedule.

        Args:
            depreciable_basis: Depreciable basis ($)
            schedule: MACRS schedule (MACRS_5 or MACRS_7)

        Returns:
            DataFrame with depreciation schedule
        """
        schedule_rates = self.macrs_schedules.get(schedule, self.macrs_schedules["MACRS_5"])

        years = np.arange(1, len(schedule_rates) + 1)
        annual_depreciation = depreciable_basis * np.array(schedule_rates)
        cumulative_depreciation = np.cumsum(annual_depreciation)
        book_value = depreciable_basis - cumulative_depreciation

        df = pd.DataFrame({
            'year': years,
            'depreciation_rate': schedule_rates,
            'annual_depreciation': annual_depreciation,
            'cumulative_depreciation': cumulative_depreciation,
            'book_value': book_value
        })

        return df

    def calculate_tax_benefits(
        self,
        total_capex: float,
        annual_revenue: List[float],
        annual_opex: float,
        federal_itc: float = 0.30,
        corporate_tax_rate: float = 0.21,
        macrs_schedule: str = "MACRS_5"
    ) -> pd.DataFrame:
        """
        Calculate comprehensive tax benefits including ITC and depreciation.

        Args:
            total_capex: Total capital expenditure ($)
            annual_revenue: Annual revenue stream ($)
            annual_opex: Annual operating expenditure ($)
            federal_itc: Federal ITC rate (fraction)
            corporate_tax_rate: Corporate tax rate (fraction)
            macrs_schedule: MACRS depreciation schedule

        Returns:
            DataFrame with tax benefit details
        """
        # ITC (received in Year 1)
        itc_benefit = total_capex * federal_itc

        # Depreciable basis (85% of total CAPEX with ITC)
        depreciable_basis = total_capex * 0.85

        # MACRS depreciation
        depreciation_df = self.calculate_macrs_depreciation(depreciable_basis, macrs_schedule)

        # Extend depreciation to match revenue years
        years = len(annual_revenue)
        if len(depreciation_df) < years:
            padding = pd.DataFrame({
                'year': range(len(depreciation_df) + 1, years + 1),
                'depreciation_rate': [0] * (years - len(depreciation_df)),
                'annual_depreciation': [0] * (years - len(depreciation_df)),
                'cumulative_depreciation': [depreciation_df['cumulative_depreciation'].iloc[-1]] * (years - len(depreciation_df)),
                'book_value': [0] * (years - len(depreciation_df))
            })
            depreciation_df = pd.concat([depreciation_df, padding], ignore_index=True)

        # Calculate taxable income and tax shield
        revenue_array = np.array(annual_revenue)
        opex_array = np.full(years, annual_opex)
        depreciation_array = depreciation_df['annual_depreciation'].values[:years]

        taxable_income = revenue_array - opex_array - depreciation_array
        tax_liability = np.maximum(0, taxable_income * corporate_tax_rate)
        depreciation_tax_shield = depreciation_array * corporate_tax_rate

        # ITC in year 1
        itc_array = np.zeros(years)
        itc_array[0] = itc_benefit

        total_tax_benefit = itc_array + depreciation_tax_shield

        df = pd.DataFrame({
            'year': np.arange(1, years + 1),
            'revenue': revenue_array,
            'opex': opex_array,
            'depreciation': depreciation_array,
            'taxable_income': taxable_income,
            'tax_liability': tax_liability,
            'depreciation_tax_shield': depreciation_tax_shield,
            'itc_benefit': itc_array,
            'total_tax_benefit': total_tax_benefit
        })

        return df

    def calculate_debt_financing(
        self,
        total_capex: float,
        debt_fraction: float = 0.70,
        interest_rate: float = 0.05,
        loan_term_years: int = 18
    ) -> Dict[str, any]:
        """
        Calculate debt financing parameters.

        Args:
            total_capex: Total capital expenditure ($)
            debt_fraction: Debt as fraction of total CAPEX
            interest_rate: Annual interest rate (fraction)
            loan_term_years: Loan term (years)

        Returns:
            Dictionary with debt financing details
        """
        loan_amount = total_capex * debt_fraction
        equity_amount = total_capex * (1 - debt_fraction)

        # Annual payment (amortizing loan)
        if interest_rate > 0:
            annual_payment = loan_amount * (
                interest_rate * (1 + interest_rate) ** loan_term_years
            ) / ((1 + interest_rate) ** loan_term_years - 1)
        else:
            annual_payment = loan_amount / loan_term_years

        # Build amortization schedule
        years = []
        beginning_balance = []
        interest_payment = []
        principal_payment = []
        ending_balance = []

        balance = loan_amount
        for year in range(1, loan_term_years + 1):
            years.append(year)
            beginning_balance.append(balance)

            interest = balance * interest_rate
            principal = annual_payment - interest

            interest_payment.append(interest)
            principal_payment.append(principal)

            balance -= principal
            ending_balance.append(max(0, balance))

        amortization_df = pd.DataFrame({
            'year': years,
            'beginning_balance': beginning_balance,
            'annual_payment': annual_payment,
            'interest_payment': interest_payment,
            'principal_payment': principal_payment,
            'ending_balance': ending_balance
        })

        return {
            'loan_amount': loan_amount,
            'equity_amount': equity_amount,
            'annual_payment': annual_payment,
            'total_interest': sum(interest_payment),
            'debt_fraction': debt_fraction,
            'amortization_schedule': amortization_df
        }

    def calculate_dscr(
        self,
        cash_flows: List[float],
        debt_service: List[float]
    ) -> List[float]:
        """
        Calculate Debt Service Coverage Ratio (DSCR).

        Args:
            cash_flows: Annual cash flows ($)
            debt_service: Annual debt service payments ($)

        Returns:
            List of annual DSCR values
        """
        dscr_values = []
        for cf, ds in zip(cash_flows, debt_service):
            if ds > 0:
                dscr_values.append(cf / ds)
            else:
                dscr_values.append(float('inf'))

        return dscr_values

    def sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        sensitivity_vars: Dict[str, Tuple[float, float, int]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform sensitivity analysis on key financial parameters.

        Args:
            base_params: Base case parameters
            sensitivity_vars: Variables to test with (min, max, steps)

        Returns:
            Dictionary of sensitivity results
        """
        results = {}

        for var_name, (min_val, max_val, steps) in sensitivity_vars.items():
            var_values = np.linspace(min_val, max_val, steps)
            npv_values = []
            irr_values = []
            lcoe_values = []

            for val in var_values:
                # Create modified parameters
                params = base_params.copy()
                params[var_name] = val

                # Recalculate metrics
                cash_flow_df = self.calculate_cash_flows(
                    params['annual_energy'],
                    params.get('electricity_price', 0.12),
                    params['annual_opex'],
                    params.get('lifetime', 25),
                    params.get('degradation_rate', 0.005),
                    params.get('electricity_escalation', 0.025)
                )

                npv_irr = self.calculate_npv_irr(
                    params['total_capex'],
                    cash_flow_df['cash_flow'].tolist(),
                    params.get('discount_rate', 0.08),
                    params.get('federal_itc', 0.30)
                )

                lcoe_result = self.calculate_lcoe_detailed(
                    params['total_capex'],
                    params['annual_energy'],
                    params['annual_opex'],
                    params.get('discount_rate', 0.08),
                    params.get('lifetime', 25),
                    params.get('degradation_rate', 0.005),
                    params.get('federal_itc', 0.30)
                )

                npv_values.append(npv_irr['npv'])
                irr_values.append(npv_irr['irr'])
                lcoe_values.append(lcoe_result['lcoe'])

            results[var_name] = pd.DataFrame({
                'value': var_values,
                'npv': npv_values,
                'irr': irr_values,
                'lcoe': lcoe_values
            })

        return results

    def calculate_p50_p90_financial(
        self,
        base_npv: float,
        base_irr: float,
        uncertainty_pct: float = 0.10
    ) -> Dict[str, float]:
        """
        Calculate P50/P90 financial metrics for bankability.

        Args:
            base_npv: Base case NPV ($)
            base_irr: Base case IRR (fraction)
            uncertainty_pct: Total uncertainty (fraction)

        Returns:
            Dictionary with P-value metrics
        """
        # Assuming normal distribution
        p50_npv = base_npv
        p90_npv = base_npv * (1 - 1.28 * uncertainty_pct)
        p10_npv = base_npv * (1 + 1.28 * uncertainty_pct)

        p50_irr = base_irr
        p90_irr = base_irr * (1 - 1.28 * uncertainty_pct)
        p10_irr = base_irr * (1 + 1.28 * uncertainty_pct)

        return {
            'P50_NPV': p50_npv,
            'P90_NPV': p90_npv,
            'P10_NPV': p10_npv,
            'P50_IRR': p50_irr,
            'P90_IRR': p90_irr,
            'P10_IRR': p10_irr,
            'uncertainty': uncertainty_pct
        }


def render_financial_analysis():
    """Render financial analysis interface in Streamlit."""
    st.header("💰 Financial Analysis & Bankability")
    st.markdown("Comprehensive financial modeling including LCOE, NPV, IRR, tax benefits, and debt financing.")

    analyzer = FinancialAnalyzer()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💵 CAPEX & OPEX",
        "📊 LCOE & Cash Flow",
        "📈 NPV & IRR",
        "🏦 Tax Benefits",
        "💳 Debt Financing",
        "🎯 Sensitivity & P90"
    ])

    with tab1:
        st.subheader("Capital and Operating Expenditure")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**System Configuration**")
            system_capacity = st.number_input("System Capacity (kW):", min_value=10, max_value=100000, value=1000, step=100)

            st.write("**CAPEX Components**")
            module_cost = st.slider("Module Cost ($/Wp):", 0.30, 0.80, 0.45, 0.01)
            inverter_cost = st.slider("Inverter Cost ($/kW):", 50, 300, 150, 10)
            bos_cost = st.slider("BOS Cost ($/kW):", 100, 500, 300, 50)
            soft_cost = st.slider("Soft Costs ($/kW):", 50, 400, 200, 25)
            contingency = st.slider("Contingency (%):", 0, 15, 5) / 100

        with col2:
            st.write("**OPEX Components**")
            o_and_m = st.slider("O&M ($/kW/year):", 5, 30, 15)
            insurance_rate = st.slider("Insurance Rate (%):", 0.0, 2.0, 0.5, 0.1) / 100
            property_tax_rate = st.slider("Property Tax Rate (%):", 0.0, 3.0, 1.0, 0.1) / 100

        if st.button("💵 Calculate CAPEX & OPEX", key="calc_capex"):
            # Calculate CAPEX
            capex_breakdown = analyzer.calculate_project_capex(
                system_capacity,
                module_cost,
                inverter_cost,
                bos_cost,
                soft_cost,
                contingency
            )

            # Calculate OPEX
            opex_breakdown = analyzer.calculate_annual_opex(
                system_capacity,
                o_and_m,
                insurance_rate,
                property_tax_rate,
                capex_breakdown['total_capex']
            )

            st.session_state['capex'] = capex_breakdown
            st.session_state['opex'] = opex_breakdown
            st.session_state['system_capacity'] = system_capacity

            st.success("✅ Financial Parameters Calculated")

            # Display CAPEX breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("CAPEX Breakdown")

                capex_data = {
                    'Component': ['Modules', 'Inverters', 'BOS', 'Soft Costs', 'Contingency'],
                    'Cost ($)': [
                        capex_breakdown['module_cost'],
                        capex_breakdown['inverter_cost'],
                        capex_breakdown['bos_cost'],
                        capex_breakdown['soft_costs'],
                        capex_breakdown['contingency']
                    ]
                }
                st.dataframe(pd.DataFrame(capex_data), use_container_width=True)

                st.metric("Total CAPEX", f"${format_number(capex_breakdown['total_capex'])}")
                st.metric("Cost per Watt", f"${capex_breakdown['cost_per_watt']:.3f}/W")

            with col2:
                st.subheader("Annual OPEX Breakdown")

                opex_data = {
                    'Component': ['O&M', 'Insurance', 'Property Tax'],
                    'Cost ($)': [
                        opex_breakdown['o_and_m'],
                        opex_breakdown['insurance'],
                        opex_breakdown['property_tax']
                    ]
                }
                st.dataframe(pd.DataFrame(opex_data), use_container_width=True)

                st.metric("Total Annual OPEX", f"${format_number(opex_breakdown['total_opex'])}")

            # Pie charts
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                subplot_titles=('CAPEX Distribution', 'OPEX Distribution')
            )

            fig.add_trace(
                go.Pie(
                    labels=capex_data['Component'],
                    values=capex_data['Cost ($)'],
                    textinfo='label+percent',
                    hoverinfo='label+value+percent'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Pie(
                    labels=opex_data['Component'],
                    values=opex_data['Cost ($)'],
                    textinfo='label+percent',
                    hoverinfo='label+value+percent'
                ),
                row=1, col=2
            )

            fig.update_layout(height=400, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("LCOE & Cash Flow Analysis")

        if 'capex' not in st.session_state:
            st.warning("⚠️ Please calculate CAPEX & OPEX first in the previous tab")
        else:
            col1, col2 = st.columns(2)

            with col1:
                annual_energy = st.number_input("Annual Energy Production (kWh):", min_value=100000, max_value=10000000, value=1500000, step=50000)
                electricity_price = st.slider("Electricity Price ($/kWh):", 0.05, 0.30, 0.12, 0.01)
                electricity_escalation = st.slider("Electricity Escalation (%/year):", 0.0, 5.0, 2.5, 0.1) / 100

            with col2:
                discount_rate = st.slider("Discount Rate (%):", 3.0, 15.0, 8.0, 0.5) / 100
                lifetime = st.slider("Project Lifetime (years):", 15, 35, 25)
                degradation_rate = st.slider("Degradation Rate (%/year):", 0.0, 2.0, 0.5, 0.1) / 100
                federal_itc = st.slider("Federal ITC (%):", 0, 50, 30) / 100

            if st.button("📊 Calculate LCOE & Cash Flow", key="calc_lcoe"):
                # Calculate LCOE
                lcoe_result = analyzer.calculate_lcoe_detailed(
                    st.session_state['capex']['total_capex'],
                    annual_energy,
                    st.session_state['opex']['total_opex'],
                    discount_rate,
                    lifetime,
                    degradation_rate,
                    federal_itc
                )

                # Calculate cash flows
                cash_flow_df = analyzer.calculate_cash_flows(
                    annual_energy,
                    electricity_price,
                    st.session_state['opex']['total_opex'],
                    lifetime,
                    degradation_rate,
                    electricity_escalation
                )

                st.session_state['lcoe_result'] = lcoe_result
                st.session_state['cash_flow_df'] = cash_flow_df
                st.session_state['annual_energy'] = annual_energy
                st.session_state['electricity_price'] = electricity_price
                st.session_state['discount_rate'] = discount_rate
                st.session_state['federal_itc'] = federal_itc
                st.session_state['lifetime'] = lifetime

                st.success("✅ LCOE and Cash Flow Calculated")

                # Display LCOE metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("LCOE (with ITC)", f"${lcoe_result['lcoe']:.4f}/kWh")

                with col2:
                    st.metric("LCOE (without ITC)", f"${lcoe_result['lcoe_no_itc']:.4f}/kWh")

                with col3:
                    st.metric("ITC Benefit", f"${lcoe_result['itc_benefit']:.4f}/kWh")

                with col4:
                    st.metric("Lifetime Energy", f"{format_number(lcoe_result['total_energy_lifetime'])} kWh")

                # Cash flow visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Annual Cash Flow', 'Cumulative Cash Flow'),
                    vertical_spacing=0.15
                )

                # Annual cash flow
                fig.add_trace(
                    go.Bar(
                        x=cash_flow_df['year'],
                        y=cash_flow_df['cash_flow'],
                        name='Annual Cash Flow',
                        marker_color='#2ECC71'
                    ),
                    row=1, col=1
                )

                # Cumulative cash flow
                fig.add_trace(
                    go.Scatter(
                        x=cash_flow_df['year'],
                        y=cash_flow_df['cumulative_cash_flow'],
                        name='Cumulative Cash Flow',
                        fill='tozeroy',
                        line=dict(color='#3498DB', width=3)
                    ),
                    row=2, col=1
                )

                # Add breakeven line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red",
                    row=2, col=1
                )

                fig.update_xaxes(title_text="Year", row=1, col=1)
                fig.update_xaxes(title_text="Year", row=2, col=1)
                fig.update_yaxes(title_text="Cash Flow ($)", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative ($)", row=2, col=1)

                fig.update_layout(height=600, showlegend=True, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Cash flow table
                st.subheader("Detailed Cash Flow Projection")
                st.dataframe(cash_flow_df.head(10), use_container_width=True)

    with tab3:
        st.subheader("NPV & IRR Analysis")

        if 'cash_flow_df' not in st.session_state:
            st.warning("⚠️ Please calculate cash flows first")
        else:
            if st.button("📈 Calculate NPV & IRR", key="calc_npv"):
                npv_irr_result = analyzer.calculate_npv_irr(
                    st.session_state['capex']['total_capex'],
                    st.session_state['cash_flow_df']['cash_flow'].tolist(),
                    st.session_state['discount_rate'],
                    st.session_state['federal_itc']
                )

                st.session_state['npv_irr'] = npv_irr_result

                st.success("✅ NPV and IRR Calculated")

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("NPV", f"${format_number(npv_irr_result['npv'])}")

                with col2:
                    st.metric("IRR", f"{npv_irr_result['irr'] * 100:.2f}%")

                with col3:
                    payback = npv_irr_result['simple_payback']
                    st.metric("Simple Payback", f"{payback:.1f} years" if payback != float('inf') else "N/A")

                with col4:
                    disc_payback = npv_irr_result['discounted_payback']
                    st.metric("Discounted Payback", f"{disc_payback:.1f} years" if disc_payback != float('inf') else "N/A")

                # NPV over time
                st.subheader("NPV Development Over Time")

                years = st.session_state['cash_flow_df']['year'].values
                cash_flows = st.session_state['cash_flow_df']['cash_flow'].values
                discount_rate = st.session_state['discount_rate']

                # Calculate NPV at each year
                npv_over_time = []
                for i in range(len(cash_flows)):
                    flows = cash_flows[:i+1]
                    npv_at_year = calculate_npv(flows.tolist(), discount_rate, npv_irr_result['net_capex'])
                    npv_over_time.append(npv_at_year)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=years,
                    y=npv_over_time,
                    mode='lines+markers',
                    line=dict(color='#2ECC71', width=3),
                    fill='tozeroy',
                    name='NPV'
                ))

                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Breakeven"
                )

                fig.update_layout(
                    title="NPV Development Over Project Lifetime",
                    xaxis_title="Year",
                    yaxis_title="NPV ($)",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary
                st.info(f"""
                **Investment Summary:**
                - Net CAPEX (after ITC): ${format_number(npv_irr_result['net_capex'])}
                - ITC Benefit: ${format_number(npv_irr_result['itc_benefit'])}
                - Project NPV: ${format_number(npv_irr_result['npv'])}
                - Project IRR: {npv_irr_result['irr'] * 100:.2f}%
                - Payback Period: {npv_irr_result['simple_payback']:.1f} years
                """)

    with tab4:
        st.subheader("Tax Benefits & Incentives")

        if 'cash_flow_df' not in st.session_state:
            st.warning("⚠️ Please calculate cash flows first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                federal_itc_tax = st.slider("Federal ITC (%):_tax", 0, 50, 30, key="itc_tax") / 100
                corporate_tax = st.slider("Corporate Tax Rate (%):", 15, 35, 21) / 100

            with col2:
                macrs_schedule = st.selectbox("MACRS Schedule:", ["MACRS_5", "MACRS_7"])
                state_incentive = st.number_input("State Incentives ($):", min_value=0, max_value=10000000, value=0, step=10000)

            if st.button("🏦 Calculate Tax Benefits", key="calc_tax"):
                revenue_list = (st.session_state['cash_flow_df']['revenue']).tolist()

                tax_benefits_df = analyzer.calculate_tax_benefits(
                    st.session_state['capex']['total_capex'],
                    revenue_list,
                    st.session_state['opex']['total_opex'],
                    federal_itc_tax,
                    corporate_tax,
                    macrs_schedule
                )

                st.session_state['tax_benefits_df'] = tax_benefits_df

                st.success("✅ Tax Benefits Calculated")

                # Summary metrics
                total_itc = tax_benefits_df['itc_benefit'].sum()
                total_depreciation_shield = tax_benefits_df['depreciation_tax_shield'].sum()
                total_tax_benefits = tax_benefits_df['total_tax_benefit'].sum()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total ITC Benefit", f"${format_number(total_itc)}")

                with col2:
                    st.metric("Depreciation Tax Shield", f"${format_number(total_depreciation_shield)}")

                with col3:
                    st.metric("Total Tax Benefits", f"${format_number(total_tax_benefits + state_incentive)}")

                # Visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Annual Tax Benefits', 'Cumulative Tax Benefits'),
                    vertical_spacing=0.15
                )

                # Annual benefits
                fig.add_trace(
                    go.Bar(
                        x=tax_benefits_df['year'],
                        y=tax_benefits_df['depreciation_tax_shield'],
                        name='Depreciation Tax Shield',
                        marker_color='#3498DB'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(
                        x=tax_benefits_df['year'],
                        y=tax_benefits_df['itc_benefit'],
                        name='ITC Benefit',
                        marker_color='#2ECC71'
                    ),
                    row=1, col=1
                )

                # Cumulative benefits
                cumulative_benefits = tax_benefits_df['total_tax_benefit'].cumsum()
                fig.add_trace(
                    go.Scatter(
                        x=tax_benefits_df['year'],
                        y=cumulative_benefits,
                        name='Cumulative Benefits',
                        fill='tozeroy',
                        line=dict(color='#9B59B6', width=3)
                    ),
                    row=2, col=1
                )

                fig.update_xaxes(title_text="Year", row=1, col=1)
                fig.update_xaxes(title_text="Year", row=2, col=1)
                fig.update_yaxes(title_text="Annual Benefit ($)", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative ($)", row=2, col=1)

                fig.update_layout(height=600, barmode='stack', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # MACRS depreciation schedule
                st.subheader("MACRS Depreciation Schedule")
                depreciable_basis = st.session_state['capex']['total_capex'] * 0.85
                macrs_df = analyzer.calculate_macrs_depreciation(depreciable_basis, macrs_schedule)
                st.dataframe(macrs_df, use_container_width=True)

    with tab5:
        st.subheader("Debt Financing & DSCR")

        if 'cash_flow_df' not in st.session_state:
            st.warning("⚠️ Please calculate cash flows first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                debt_fraction = st.slider("Debt Fraction (%):", 0, 90, 70) / 100
                interest_rate = st.slider("Interest Rate (%):", 2.0, 10.0, 5.0, 0.25) / 100

            with col2:
                loan_term = st.slider("Loan Term (years):", 5, 25, 18)

            if st.button("💳 Calculate Debt Financing", key="calc_debt"):
                debt_result = analyzer.calculate_debt_financing(
                    st.session_state['capex']['total_capex'],
                    debt_fraction,
                    interest_rate,
                    loan_term
                )

                st.session_state['debt_result'] = debt_result

                st.success("✅ Debt Financing Calculated")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Loan Amount", f"${format_number(debt_result['loan_amount'])}")

                with col2:
                    st.metric("Equity Required", f"${format_number(debt_result['equity_amount'])}")

                with col3:
                    st.metric("Annual Payment", f"${format_number(debt_result['annual_payment'])}")

                with col4:
                    st.metric("Total Interest", f"${format_number(debt_result['total_interest'])}")

                # Amortization visualization
                amort_df = debt_result['amortization_schedule']

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=amort_df['year'],
                    y=amort_df['interest_payment'],
                    name='Interest Payment',
                    marker_color='#E74C3C'
                ))

                fig.add_trace(go.Bar(
                    x=amort_df['year'],
                    y=amort_df['principal_payment'],
                    name='Principal Payment',
                    marker_color='#3498DB'
                ))

                fig.add_trace(go.Scatter(
                    x=amort_df['year'],
                    y=amort_df['ending_balance'],
                    name='Remaining Balance',
                    yaxis='y2',
                    line=dict(color='#2ECC71', width=3)
                ))

                fig.update_layout(
                    title="Loan Amortization Schedule",
                    xaxis_title="Year",
                    yaxis_title="Annual Payment ($)",
                    yaxis2=dict(
                        title="Remaining Balance ($)",
                        overlaying='y',
                        side='right'
                    ),
                    barmode='stack',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # DSCR calculation
                st.subheader("Debt Service Coverage Ratio (DSCR)")

                # Extend debt service to full lifetime
                debt_service_full = [debt_result['annual_payment']] * loan_term
                remaining_years = len(st.session_state['cash_flow_df']) - loan_term
                if remaining_years > 0:
                    debt_service_full.extend([0] * remaining_years)

                dscr_values = analyzer.calculate_dscr(
                    st.session_state['cash_flow_df']['cash_flow'].tolist(),
                    debt_service_full
                )

                # Create DSCR dataframe
                dscr_df = pd.DataFrame({
                    'year': st.session_state['cash_flow_df']['year'],
                    'cash_flow': st.session_state['cash_flow_df']['cash_flow'],
                    'debt_service': debt_service_full,
                    'dscr': dscr_values
                })

                # DSCR visualization
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=dscr_df['year'],
                    y=dscr_df['dscr'],
                    mode='lines+markers',
                    line=dict(color='#2ECC71', width=3),
                    name='DSCR'
                ))

                # Add minimum DSCR threshold
                fig.add_hline(
                    y=1.2,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Min DSCR (1.2x)"
                )

                fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Breakeven (1.0x)"
                )

                fig.update_layout(
                    title="Debt Service Coverage Ratio Over Time",
                    xaxis_title="Year",
                    yaxis_title="DSCR",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # DSCR table (first 10 years)
                st.dataframe(dscr_df.head(10), use_container_width=True)

                # Min DSCR during loan term
                min_dscr = min(dscr_values[:loan_term])
                avg_dscr = np.mean(dscr_values[:loan_term])

                if min_dscr >= 1.2:
                    st.success(f"✅ Bankable: Minimum DSCR = {min_dscr:.2f}x (Average: {avg_dscr:.2f}x)")
                elif min_dscr >= 1.0:
                    st.warning(f"⚠️ Marginal: Minimum DSCR = {min_dscr:.2f}x (Average: {avg_dscr:.2f}x)")
                else:
                    st.error(f"❌ Not Bankable: Minimum DSCR = {min_dscr:.2f}x (Average: {avg_dscr:.2f}x)")

    with tab6:
        st.subheader("Sensitivity Analysis & P90 Metrics")

        if 'npv_irr' not in st.session_state:
            st.warning("⚠️ Please calculate NPV & IRR first")
        else:
            st.write("### Sensitivity Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                elec_price_range = st.slider(
                    "Electricity Price Range (%):",
                    -50, 50, (-20, 20),
                    help="Variation from base case"
                )

            with col2:
                discount_range = st.slider(
                    "Discount Rate Range (%):",
                    -50, 50, (-20, 20),
                    help="Variation from base case"
                )

            with col3:
                degradation_range = st.slider(
                    "Degradation Rate Range (%):",
                    -50, 50, (-20, 20),
                    help="Variation from base case"
                )

            if st.button("🎯 Run Sensitivity Analysis", key="sensitivity"):
                with st.spinner("Running sensitivity analysis..."):
                    base_params = {
                        'total_capex': st.session_state['capex']['total_capex'],
                        'annual_energy': st.session_state['annual_energy'],
                        'annual_opex': st.session_state['opex']['total_opex'],
                        'electricity_price': st.session_state['electricity_price'],
                        'discount_rate': st.session_state['discount_rate'],
                        'degradation_rate': 0.005,
                        'lifetime': st.session_state['lifetime'],
                        'federal_itc': st.session_state['federal_itc'],
                        'electricity_escalation': 0.025
                    }

                    base_elec_price = base_params['electricity_price']
                    base_discount = base_params['discount_rate']
                    base_degradation = base_params['degradation_rate']

                    sensitivity_vars = {
                        'electricity_price': (
                            base_elec_price * (1 + elec_price_range[0]/100),
                            base_elec_price * (1 + elec_price_range[1]/100),
                            20
                        ),
                        'discount_rate': (
                            base_discount * (1 + discount_range[0]/100),
                            base_discount * (1 + discount_range[1]/100),
                            20
                        ),
                        'degradation_rate': (
                            base_degradation * (1 + degradation_range[0]/100),
                            base_degradation * (1 + degradation_range[1]/100),
                            20
                        )
                    }

                    sensitivity_results = analyzer.sensitivity_analysis(base_params, sensitivity_vars)

                    st.session_state['sensitivity_results'] = sensitivity_results

                st.success("✅ Sensitivity Analysis Complete")

                # Tornado diagram for NPV
                st.subheader("Tornado Diagram - NPV Impact")

                base_npv = st.session_state['npv_irr']['npv']

                tornado_data = []
                for var_name, df in sensitivity_results.items():
                    min_npv = df['npv'].min()
                    max_npv = df['npv'].max()

                    downside = min_npv - base_npv
                    upside = max_npv - base_npv

                    tornado_data.append({
                        'variable': var_name.replace('_', ' ').title(),
                        'downside': downside,
                        'upside': upside,
                        'range': abs(upside - downside)
                    })

                tornado_df = pd.DataFrame(tornado_data)
                tornado_df = tornado_df.sort_values('range', ascending=True)

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    y=tornado_df['variable'],
                    x=tornado_df['downside'],
                    name='Downside',
                    orientation='h',
                    marker_color='#E74C3C'
                ))

                fig.add_trace(go.Bar(
                    y=tornado_df['variable'],
                    x=tornado_df['upside'],
                    name='Upside',
                    orientation='h',
                    marker_color='#2ECC71'
                ))

                fig.update_layout(
                    title="NPV Sensitivity (Tornado Diagram)",
                    xaxis_title="NPV Change from Base Case ($)",
                    barmode='overlay',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Spider chart for all metrics
                st.subheader("Spider Chart - Multi-Metric Sensitivity")

                fig = go.Figure()

                for var_name, df in sensitivity_results.items():
                    # Normalize to percentage change from base
                    npv_pct = ((df['npv'] / base_npv) - 1) * 100

                    fig.add_trace(go.Scatter(
                        x=df['value'],
                        y=npv_pct,
                        mode='lines+markers',
                        name=var_name.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))

                fig.add_hline(y=0, line_dash="dash", line_color="gray")

                fig.update_layout(
                    title="NPV Sensitivity - All Variables",
                    xaxis_title="Variable Value",
                    yaxis_title="NPV Change (%)",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            # P50/P90 analysis
            st.divider()
            st.write("### P50/P90 Bankability Metrics")

            uncertainty_pct = st.slider("Total Uncertainty (%):", 5, 20, 10) / 100

            if st.button("📊 Calculate P90 Metrics", key="p90"):
                p_values = analyzer.calculate_p50_p90_financial(
                    st.session_state['npv_irr']['npv'],
                    st.session_state['npv_irr']['irr'],
                    uncertainty_pct
                )

                st.session_state['p_values'] = p_values

                st.success("✅ P-Value Analysis Complete")

                # Display P-values
                st.subheader("NPV P-Values")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("P90 (Conservative)", f"${format_number(p_values['P90_NPV'])}")

                with col2:
                    st.metric("P50 (Expected)", f"${format_number(p_values['P50_NPV'])}")

                with col3:
                    st.metric("P10 (Optimistic)", f"${format_number(p_values['P10_NPV'])}")

                st.subheader("IRR P-Values")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("P90 (Conservative)", f"{p_values['P90_IRR'] * 100:.2f}%")

                with col2:
                    st.metric("P50 (Expected)", f"{p_values['P50_IRR'] * 100:.2f}%")

                with col3:
                    st.metric("P10 (Optimistic)", f"{p_values['P10_IRR'] * 100:.2f}%")

                # Visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('NPV P-Values', 'IRR P-Values')
                )

                p_levels = ['P90', 'P50', 'P10']
                npv_values = [p_values['P90_NPV'], p_values['P50_NPV'], p_values['P10_NPV']]
                irr_values = [p_values['P90_IRR'] * 100, p_values['P50_IRR'] * 100, p_values['P10_IRR'] * 100]
                colors = ['#E74C3C', '#F39C12', '#2ECC71']

                fig.add_trace(
                    go.Bar(
                        x=p_levels,
                        y=npv_values,
                        marker_color=colors,
                        text=[f"${format_number(v)}" for v in npv_values],
                        textposition='auto',
                        showlegend=False
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(
                        x=p_levels,
                        y=irr_values,
                        marker_color=colors,
                        text=[f"{v:.2f}%" for v in irr_values],
                        textposition='auto',
                        showlegend=False
                    ),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Exceedance Probability", row=1, col=1)
                fig.update_xaxes(title_text="Exceedance Probability", row=1, col=2)
                fig.update_yaxes(title_text="NPV ($)", row=1, col=1)
                fig.update_yaxes(title_text="IRR (%)", row=1, col=2)

                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Bankability assessment
                if p_values['P90_NPV'] > 0 and p_values['P90_IRR'] > 0.06:
                    st.success("✅ **Bankable Project**: P90 NPV > 0 and P90 IRR > 6%")
                elif p_values['P90_NPV'] > 0:
                    st.warning("⚠️ **Marginal**: P90 NPV > 0 but P90 IRR < 6%")
                else:
                    st.error("❌ **Not Bankable**: P90 NPV < 0")

    st.divider()
    st.info("💡 **Financial Analysis & Bankability** - Branch B13 | Complete Financial Modeling Suite")
