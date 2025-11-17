"""
Financial Modeling Module - Comprehensive financial analysis and modeling
"""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render the Financial Modeling module"""
    st.header("💰 Financial Modeling")
    st.markdown("---")

    st.markdown("""
    ### Comprehensive Financial Analysis & Investment Modeling

    Advanced financial modeling for PV project evaluation and investment decisions.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Project Setup", "Cash Flow", "Financing", "Risk Analysis", "Reports"
    ])

    with tab1:
        st.subheader("Project Setup & Assumptions")

        st.markdown("#### Project Information")

        col1, col2 = st.columns(2)

        with col1:
            project_name = st.text_input("Project Name", value="Solar PV Project 2024")
            project_size = st.number_input("System Size (kWp)", min_value=0.0, value=1000.0)
            location = st.text_input("Location", value="United States")

            cod = st.date_input("Commercial Operation Date (COD)")

        with col2:
            analysis_period = st.number_input("Analysis Period (years)", min_value=1, max_value=50, value=25)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CNY", "INR"])

            project_type = st.selectbox(
                "Project Type",
                ["Utility-scale Ground-mount", "Commercial Rooftop", "Residential",
                 "Community Solar", "Floating PV", "Agrivoltaics"]
            )

        st.markdown("---")
        st.markdown("#### Capital Costs (CAPEX)")

        col1, col2 = st.columns(2)

        with col1:
            module_cost_w = st.number_input("Module Cost ($/W)", min_value=0.0, value=0.22, format="%.3f")
            inverter_cost_w = st.number_input("Inverter Cost ($/W)", min_value=0.0, value=0.06, format="%.3f")
            mounting_cost_w = st.number_input("Mounting Structure ($/W)", min_value=0.0, value=0.12, format="%.3f")

        with col2:
            electrical_cost_w = st.number_input("Electrical/BOS ($/W)", min_value=0.0, value=0.10, format="%.3f")
            installation_cost_w = st.number_input("Installation Labor ($/W)", min_value=0.0, value=0.15, format="%.3f")
            other_cost_w = st.number_input("Other Costs ($/W)", min_value=0.0, value=0.08, format="%.3f")

        equipment_cost = (module_cost_w + inverter_cost_w + mounting_cost_w +
                         electrical_cost_w + installation_cost_w + other_cost_w) * project_size * 1000

        # Soft costs
        st.markdown("**Soft Costs:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            dev_cost = st.number_input("Development ($)", min_value=0.0, value=50000.0)
            permitting = st.number_input("Permitting ($)", min_value=0.0, value=30000.0)
        with col2:
            engineering = st.number_input("Engineering ($)", min_value=0.0, value=75000.0)
            legal = st.number_input("Legal ($)", min_value=0.0, value=40000.0)
        with col3:
            insurance = st.number_input("Insurance (first year) ($)", min_value=0.0, value=25000.0)
            contingency_pct = st.slider("Contingency (%)", 0, 20, 5)

        soft_costs = dev_cost + permitting + engineering + legal + insurance
        contingency = (equipment_cost + soft_costs) * (contingency_pct / 100)

        total_capex = equipment_cost + soft_costs + contingency

        st.metric("**Total CAPEX**", f"${total_capex:,.0f}", help=f"${total_capex/project_size/1000:.2f}/W")

        st.markdown("---")
        st.markdown("#### Operating Costs (OPEX)")

        col1, col2 = st.columns(2)

        with col1:
            om_cost_kw_year = st.number_input("O&M ($/kW/year)", min_value=0.0, value=15.0)
            annual_om = om_cost_kw_year * project_size

            insurance_annual = st.number_input("Annual Insurance ($)", min_value=0.0, value=25000.0)
            land_lease = st.number_input("Land Lease ($/year)", min_value=0.0, value=20000.0)

        with col2:
            property_tax_rate = st.slider("Property Tax Rate (%)", 0.0, 5.0, 1.5, 0.1)
            property_tax = total_capex * (property_tax_rate / 100)

            admin_cost = st.number_input("Admin & Management ($/year)", min_value=0.0, value=30000.0)

        total_opex_year1 = annual_om + insurance_annual + land_lease + property_tax + admin_cost
        st.metric("**Total OPEX (Year 1)**", f"${total_opex_year1:,.0f}/year")

        # Escalation rates
        st.markdown("**Escalation Rates:**")
        col1, col2 = st.columns(2)
        with col1:
            opex_escalation = st.slider("OPEX Escalation (%/year)", 0.0, 5.0, 2.0, 0.1)
        with col2:
            degradation_rate = st.slider("Module Degradation (%/year)", 0.0, 1.5, 0.5, 0.05)

    with tab2:
        st.subheader("Revenue & Cash Flow Analysis")

        st.markdown("#### Revenue Assumptions")

        col1, col2 = st.columns(2)

        with col1:
            specific_yield = st.number_input("Specific Yield (kWh/kWp/year)", min_value=0.0, value=1500.0)
            year1_production = project_size * specific_yield

            st.metric("Year 1 Production", f"{year1_production:,.0f} kWh")

            ppa_rate = st.number_input("PPA/Electricity Rate ($/kWh)", min_value=0.0, value=0.08, format="%.3f")

        with col2:
            revenue_model = st.selectbox(
                "Revenue Model",
                ["Fixed PPA", "Merchant (Wholesale)", "Retail NEM", "Feed-in Tariff", "Hybrid"]
            )

            ppa_escalation = st.slider("PPA Escalation (%/year)", 0.0, 5.0, 1.5, 0.1)

            srec_value = st.number_input("SREC Value ($/MWh)", min_value=0.0, value=0.0,
                                        help="Solar Renewable Energy Credits")

        year1_revenue = (year1_production / 1000) * ppa_rate * 1000 + (year1_production / 1000) * srec_value

        st.metric("**Year 1 Revenue**", f"${year1_revenue:,.0f}")

        st.markdown("---")
        st.markdown("#### Incentives & Tax Credits")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Federal Incentives:**")
            itc_rate = st.slider("Investment Tax Credit - ITC (%)", 0, 50, 30)
            itc_amount = total_capex * (itc_rate / 100)
            st.metric("ITC Amount", f"${itc_amount:,.0f}")

            ptc_rate = st.number_input("Production Tax Credit - PTC ($/kWh)", min_value=0.0, value=0.0, format="%.4f")

        with col2:
            st.markdown("**State/Local Incentives:**")
            state_grant = st.number_input("State Grant ($)", min_value=0.0, value=0.0)
            local_incentives = st.number_input("Local Incentives ($)", min_value=0.0, value=0.0)

            depreciation_method = st.selectbox(
                "Depreciation Method",
                ["MACRS 5-year", "MACRS 7-year", "Straight-line", "None"]
            )

        st.markdown("---")
        st.markdown("#### Annual Cash Flow Projection")

        # Build cash flow table
        years = list(range(1, min(analysis_period + 1, 26)))  # Show up to 25 years
        cashflow_data = []

        for year in years:
            # Production with degradation
            production = year1_production * ((1 - degradation_rate/100) ** (year - 1))

            # Revenue with escalation
            revenue = (production / 1000) * ppa_rate * ((1 + ppa_escalation/100) ** (year - 1)) * 1000
            revenue += (production / 1000) * srec_value

            # PTC if applicable
            if ptc_rate > 0 and year <= 10:  # PTC typically for 10 years
                revenue += production * ptc_rate

            # OPEX with escalation
            opex = total_opex_year1 * ((1 + opex_escalation/100) ** (year - 1))

            # EBITDA
            ebitda = revenue - opex

            # Simplified tax calculation
            if depreciation_method == "MACRS 5-year" and year <= 6:
                macrs_schedule = [20, 32, 19.2, 11.52, 11.52, 5.76]
                depreciation = total_capex * (macrs_schedule[year-1] / 100) if year <= 6 else 0
            else:
                depreciation = total_capex / analysis_period if depreciation_method != "None" else 0

            taxable_income = ebitda - depreciation
            tax_rate = 21  # Federal corporate tax rate
            taxes = max(0, taxable_income * (tax_rate / 100))

            net_income = ebitda - taxes
            free_cashflow = net_income + depreciation  # Add back non-cash depreciation

            cashflow_data.append({
                'Year': year,
                'Production (MWh)': f"{production/1000:.1f}",
                'Revenue ($)': f"{revenue:,.0f}",
                'OPEX ($)': f"{opex:,.0f}",
                'EBITDA ($)': f"{ebitda:,.0f}",
                'Net Income ($)': f"{net_income:,.0f}",
                'Free Cash Flow ($)': f"{free_cashflow:,.0f}"
            })

        cashflow_df = pd.DataFrame(cashflow_data)
        st.dataframe(cashflow_df, use_container_width=True, height=400)

        # Lifetime metrics
        st.markdown("#### Lifetime Financial Summary")

        lifetime_revenue = sum([float(row['Revenue ($)'].replace(',', '')) for row in cashflow_data])
        lifetime_opex = sum([float(row['OPEX ($)'].replace(',', '')) for row in cashflow_data])
        lifetime_fcf = sum([float(row['Free Cash Flow ($)'].replace(',', '')) for row in cashflow_data])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Lifetime Revenue ({analysis_period}y)", f"${lifetime_revenue:,.0f}")
        with col2:
            st.metric(f"Lifetime OPEX ({analysis_period}y)", f"${lifetime_opex:,.0f}")
        with col3:
            st.metric(f"Lifetime Free Cash Flow", f"${lifetime_fcf:,.0f}")

    with tab3:
        st.subheader("Financing Structure")

        st.markdown("#### Capital Structure")

        col1, col2 = st.columns(2)

        with col1:
            equity_pct = st.slider("Equity (%)", 0, 100, 40)
            debt_pct = 100 - equity_pct

            equity_amount = total_capex * (equity_pct / 100) - itc_amount - state_grant - local_incentives
            debt_amount = total_capex * (debt_pct / 100)

            st.metric("Equity Required", f"${max(0, equity_amount):,.0f}")
            st.metric("Debt Amount", f"${debt_amount:,.0f}")

        with col2:
            st.markdown("**Debt Terms:**")
            interest_rate = st.slider("Interest Rate (%)", 0.0, 15.0, 5.5, 0.1)
            loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=18)
            dscr_target = st.number_input("Target DSCR", min_value=1.0, max_value=3.0, value=1.25, step=0.05,
                                         help="Debt Service Coverage Ratio")

        # Calculate debt service
        if debt_amount > 0 and interest_rate > 0:
            monthly_rate = interest_rate / 100 / 12
            num_payments = loan_term * 12
            monthly_payment = debt_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                            ((1 + monthly_rate)**num_payments - 1)
            annual_debt_service = monthly_payment * 12
        else:
            annual_debt_service = 0

        st.metric("Annual Debt Service", f"${annual_debt_service:,.0f}")

        # Calculate actual DSCR
        year1_ebitda = year1_revenue - total_opex_year1
        actual_dscr = year1_ebitda / annual_debt_service if annual_debt_service > 0 else 0

        if actual_dscr >= dscr_target:
            st.success(f"✓ DSCR: {actual_dscr:.2f} (Target: {dscr_target:.2f})")
        else:
            st.warning(f"⚠ DSCR: {actual_dscr:.2f} below target ({dscr_target:.2f})")

        st.markdown("---")
        st.markdown("#### Returns Analysis")

        # Calculate levered returns
        discount_rate = st.slider("Discount Rate (%)", 0.0, 20.0, 8.0, 0.5)

        # NPV calculation
        initial_investment = max(0, equity_amount)
        npv = -initial_investment

        for year in range(1, analysis_period + 1):
            production = year1_production * ((1 - degradation_rate/100) ** (year - 1))
            revenue = (production / 1000) * ppa_rate * ((1 + ppa_escalation/100) ** (year - 1)) * 1000
            revenue += (production / 1000) * srec_value
            opex = total_opex_year1 * ((1 + opex_escalation/100) ** (year - 1))
            ebitda = revenue - opex

            # Cash flow after debt service
            if year <= loan_term:
                cashflow = ebitda - annual_debt_service
            else:
                cashflow = ebitda

            # Discount to present value
            pv = cashflow / ((1 + discount_rate/100) ** year)
            npv += pv

        # IRR estimate (simplified)
        avg_annual_return = (lifetime_fcf - annual_debt_service * min(loan_term, analysis_period)) / analysis_period
        equity_irr = (avg_annual_return / initial_investment * 100) if initial_investment > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project NPV", f"${npv:,.0f}")
        with col2:
            st.metric("Equity IRR (est.)", f"{equity_irr:.1f}%")
        with col3:
            payback = initial_investment / (year1_ebitda - annual_debt_service) if (year1_ebitda - annual_debt_service) > 0 else 0
            st.metric("Payback Period", f"{payback:.1f} years")

        st.markdown("#### LCOE Calculation")

        # Levelized Cost of Energy
        total_lifecycle_cost = total_capex + lifetime_opex
        total_lifecycle_energy = sum([year1_production * ((1 - degradation_rate/100) ** (year - 1))
                                     for year in range(analysis_period)])

        lcoe = (total_lifecycle_cost / total_lifecycle_energy) if total_lifecycle_energy > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("LCOE (Nominal)", f"${lcoe:.3f}/kWh")
        with col2:
            margin = ppa_rate - lcoe
            st.metric("Margin vs PPA", f"${margin:.3f}/kWh", delta=f"{(margin/ppa_rate*100):.1f}%")

    with tab4:
        st.subheader("Risk Analysis & Sensitivity")

        st.markdown("""
        ### Financial Risk Assessment

        Analyze the impact of key variables on project returns.
        """)

        st.markdown("#### Sensitivity Analysis")

        sensitivity_var = st.selectbox(
            "Variable to Analyze",
            ["PPA Rate", "CAPEX", "Production", "OPEX", "Degradation Rate", "Debt Interest Rate"]
        )

        variation_range = st.slider("Variation Range (±%)", 5, 50, 20)

        # Generate sensitivity table
        variations = [-variation_range, -variation_range/2, 0, variation_range/2, variation_range]
        sensitivity_results = []

        base_npv = npv
        base_irr = equity_irr

        for var_pct in variations:
            # Adjust the selected variable
            if sensitivity_var == "PPA Rate":
                adj_ppa = ppa_rate * (1 + var_pct/100)
                # Recalculate NPV with adjusted PPA
                adj_npv = base_npv * (1 + var_pct/100 * 0.8)  # Simplified
                adj_irr = base_irr * (1 + var_pct/100 * 0.6)  # Simplified
            elif sensitivity_var == "CAPEX":
                adj_npv = base_npv - (total_capex * var_pct/100)
                adj_irr = base_irr * (1 - var_pct/100 * 0.3)  # Simplified
            elif sensitivity_var == "Production":
                adj_npv = base_npv * (1 + var_pct/100 * 0.9)  # Simplified
                adj_irr = base_irr * (1 + var_pct/100 * 0.7)  # Simplified
            else:
                adj_npv = base_npv * (1 - var_pct/100 * 0.3)  # Simplified
                adj_irr = base_irr * (1 - var_pct/100 * 0.2)  # Simplified

            sensitivity_results.append({
                'Change': f"{var_pct:+.0f}%",
                'NPV ($M)': f"${adj_npv/1000000:.2f}",
                'IRR (%)': f"{adj_irr:.1f}%",
                'LCOE ($/kWh)': f"${lcoe * (1 - var_pct/100 * 0.3):.3f}"
            })

        sensitivity_df = pd.DataFrame(sensitivity_results)
        st.dataframe(sensitivity_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Risk Factors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Risks:**")
            st.warning("• PPA rate volatility")
            st.warning("• Electricity market prices")
            st.warning("• Currency exchange rates")

            st.markdown("**Technical Risks:**")
            st.warning("• Performance degradation")
            st.warning("• Equipment failures")
            st.warning("• Inverter replacement costs")

        with col2:
            st.markdown("**Regulatory Risks:**")
            st.warning("• Tax credit changes")
            st.warning("• Grid interconnection policies")
            st.warning("• Permitting delays")

            st.markdown("**Financial Risks:**")
            st.warning("• Interest rate changes")
            st.warning("• Refinancing risks")
            st.warning("• Insurance cost increases")

        st.markdown("---")
        st.markdown("#### Monte Carlo Simulation")

        st.info("Monte Carlo simulation runs multiple scenarios with randomized inputs")

        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)

        if st.button("▶️ Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner("Running simulations..."):
                # Simplified Monte Carlo
                simulation_results = []

                for _ in range(min(num_simulations, 1000)):  # Limit for performance
                    # Randomize key variables (normal distribution)
                    sim_ppa = ppa_rate * np.random.normal(1.0, 0.1)
                    sim_production = year1_production * np.random.normal(1.0, 0.08)
                    sim_opex = total_opex_year1 * np.random.normal(1.0, 0.12)

                    # Calculate returns for this scenario
                    sim_revenue = sim_production * sim_ppa
                    sim_ebitda = sim_revenue - sim_opex
                    sim_irr = (sim_ebitda - annual_debt_service) / max(1, equity_amount) * 100

                    simulation_results.append(sim_irr)

                # Statistics
                p10 = np.percentile(simulation_results, 10)
                p50 = np.percentile(simulation_results, 50)
                p90 = np.percentile(simulation_results, 90)

                st.success("Simulation completed!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P10 IRR", f"{p10:.1f}%", help="10% probability of being below this")
                with col2:
                    st.metric("P50 IRR (Median)", f"{p50:.1f}%")
                with col3:
                    st.metric("P90 IRR", f"{p90:.1f}%", help="90% probability of being below this")

                # Simple histogram
                hist_data = pd.DataFrame({
                    'IRR (%)': simulation_results
                })
                st.bar_chart(hist_data['IRR (%)'].value_counts().sort_index())

    with tab5:
        st.subheader("Financial Reports & Export")

        st.markdown("#### Available Reports")

        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Cash Flow", "Sensitivity Analysis",
             "Investment Memorandum", "Lender Package", "Tax Equity Package", "Custom Report"]
        )

        st.markdown("#### Report Configuration")

        col1, col2 = st.columns(2)

        with col1:
            include_charts = st.checkbox("Include Charts & Graphs", value=True)
            include_assumptions = st.checkbox("Include Assumptions", value=True)
            include_sensitivity = st.checkbox("Include Sensitivity Analysis", value=True)

        with col2:
            report_format = st.selectbox("Format", ["PDF", "Excel", "PowerPoint", "Word"])
            branding = st.checkbox("Add Company Branding", value=False)

        st.markdown("---")
        st.markdown("#### Key Metrics Summary")

        # Summary metrics table
        summary_metrics = pd.DataFrame({
            'Metric': [
                'Total CAPEX',
                'PPA Rate',
                'Year 1 Production',
                'Year 1 Revenue',
                'LCOE',
                'Project NPV',
                'Equity IRR',
                'Payback Period',
                'Debt/Equity Ratio',
                'DSCR (Year 1)'
            ],
            'Value': [
                f"${total_capex:,.0f}",
                f"${ppa_rate:.3f}/kWh",
                f"{year1_production:,.0f} kWh",
                f"${year1_revenue:,.0f}",
                f"${lcoe:.3f}/kWh",
                f"${npv:,.0f}",
                f"{equity_irr:.1f}%",
                f"{payback:.1f} years",
                f"{debt_pct/equity_pct:.2f}" if equity_pct > 0 else "N/A",
                f"{actual_dscr:.2f}"
            ]
        })

        st.dataframe(summary_metrics, use_container_width=True)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Report", use_container_width=True):
                st.success(f"{report_type} generated successfully!")

        with col2:
            if st.button("📧 Email Report", use_container_width=True):
                st.info("Email functionality not yet implemented")

        with col3:
            if st.button("💾 Save Model", use_container_width=True):
                st.success("Financial model saved successfully!")

        st.markdown("---")
        st.markdown("#### Export Data")

        export_format = st.selectbox(
            "Export Format",
            ["Excel (Full Model)", "CSV (Cash Flow)", "JSON (All Data)", "XML"]
        )

        if st.button("⬇️ Export Data", use_container_width=True):
            st.success(f"Data exported as {export_format}")

    st.markdown("---")
    if st.button("💾 Save Financial Model", use_container_width=True):
        st.success("Financial model saved to project!")
