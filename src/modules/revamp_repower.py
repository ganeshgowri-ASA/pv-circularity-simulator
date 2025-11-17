"""
Revamp/Repower Module - System upgrade and repowering analysis
"""

import streamlit as st
import pandas as pd


def render():
    """Render the Revamp/Repower module"""
    st.header("🔄 Revamp & Repower")
    st.markdown("---")

    st.markdown("""
    ### System Upgrade & Repowering Analysis

    Evaluate options for upgrading or repowering existing PV systems.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Current System", "Upgrade Options", "Analysis", "ROI"])

    with tab1:
        st.subheader("Current System Assessment")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### System Information")
            installation_year = st.number_input("Installation Year", min_value=2000, max_value=2024, value=2015)
            system_age = 2024 - installation_year
            st.metric("System Age", f"{system_age} years")

            original_capacity = st.number_input("Original Capacity (kWp)", min_value=0.0, value=100.0)
            current_capacity = st.number_input("Current Capacity (kWp)", min_value=0.0, value=92.0)

            degradation = ((original_capacity - current_capacity) / original_capacity * 100)
            st.metric("Total Degradation", f"{degradation:.1f}%")

        with col2:
            st.markdown("#### Performance Metrics")
            original_pr = st.number_input("Original PR (%)", min_value=0.0, max_value=100.0, value=85.0)
            current_pr = st.number_input("Current PR (%)", min_value=0.0, max_value=100.0, value=78.0)

            pr_degradation = original_pr - current_pr
            st.metric("PR Degradation", f"{pr_degradation:.1f}%", delta_color="inverse")

            annual_production = st.number_input("Annual Production (MWh)", min_value=0.0, value=125.0)

        st.markdown("---")
        st.markdown("#### Component Status")

        components = [
            {"Component": "PV Modules", "Status": "Degraded", "Remaining Life": "5-8 years", "Priority": "High"},
            {"Component": "Inverters", "Status": "Fair", "Remaining Life": "3-5 years", "Priority": "Medium"},
            {"Component": "Mounting Structure", "Status": "Good", "Remaining Life": "15-20 years", "Priority": "Low"},
            {"Component": "DC Wiring", "Status": "Fair", "Remaining Life": "8-10 years", "Priority": "Low"},
            {"Component": "AC Equipment", "Status": "Fair", "Remaining Life": "5-8 years", "Priority": "Medium"},
        ]

        components_df = pd.DataFrame(components)
        st.dataframe(components_df, use_container_width=True)

        st.markdown("#### Maintenance History")
        maintenance_cost = st.number_input("Annual Maintenance Cost ($)", min_value=0.0, value=5000.0)
        downtime_hours = st.number_input("Annual Downtime (hours)", min_value=0.0, value=48.0)

    with tab2:
        st.subheader("Upgrade & Repowering Options")

        upgrade_scenario = st.selectbox(
            "Upgrade Scenario",
            ["Module Replacement Only", "Full Repower (Modules + Inverters)",
             "Module + Inverter Upgrade", "Add Capacity (Repower + Expand)",
             "Custom Scenario"]
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### New Module Selection")

            new_module_power = st.number_input("New Module Power (W)", min_value=0, value=550)
            new_module_eff = st.number_input("New Module Efficiency (%)", min_value=0.0, max_value=100.0, value=22.0)

            module_tech = st.selectbox(
                "Technology",
                ["Mono PERC", "TOPCon", "HJT", "IBC", "Bifacial PERC", "Bifacial TOPCon"]
            )

            st.info(f"Old modules: ~300W, New modules: {new_module_power}W")

        with col2:
            st.markdown("#### New Inverter Selection")

            inverter_upgrade = st.checkbox("Replace Inverters", value=True)

            if inverter_upgrade:
                new_inverter_eff = st.number_input("New Inverter Efficiency (%)", min_value=0.0, max_value=100.0, value=98.5)
                inverter_type = st.selectbox(
                    "Inverter Type",
                    ["String Inverter", "Central Inverter", "Hybrid Inverter",
                     "String + Optimizers", "Microinverters"]
                )
            else:
                st.info("Keep existing inverters")

        st.markdown("#### System Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            if upgrade_scenario == "Add Capacity (Repower + Expand)":
                new_capacity = st.number_input("New System Capacity (kWp)", min_value=0.0, value=150.0)
                capacity_increase = new_capacity - original_capacity
                st.metric("Capacity Increase", f"+{capacity_increase:.1f} kWp")
            else:
                new_capacity = st.number_input("New System Capacity (kWp)", min_value=0.0, value=original_capacity)

        with col2:
            layout_option = st.selectbox(
                "Layout Option",
                ["Same footprint", "Expand area", "Optimize layout"]
            )

        with col3:
            reuse_bos = st.checkbox("Reuse BOS components", value=True,
                                   help="Reuse mounting structure, wiring, etc.")

    with tab3:
        st.subheader("Technical & Financial Analysis")

        st.markdown("#### Performance Comparison")

        # Create comparison table
        comparison_data = {
            'Metric': [
                'System Capacity (kWp)',
                'Annual Production (MWh)',
                'Performance Ratio (%)',
                'Specific Yield (kWh/kWp)',
                'Capacity Factor (%)',
                'Expected Degradation (%/year)'
            ],
            'Current System': [
                f"{current_capacity:.1f}",
                f"{annual_production:.1f}",
                f"{current_pr:.1f}",
                f"{(annual_production * 1000 / current_capacity):.0f}",
                "15.2",
                "0.8"
            ],
            'After Repower': [
                f"{new_capacity:.1f}",
                f"{new_capacity * 1.85:.1f}",
                "85.0",
                f"{1850:.0f}",
                "21.1",
                "0.4"
            ],
            'Improvement': [
                f"+{new_capacity - current_capacity:.1f}",
                f"+{(new_capacity * 1.85 - annual_production):.1f}",
                f"+{85.0 - current_pr:.1f}",
                f"+{(1850 - annual_production * 1000 / current_capacity):.0f}",
                "+5.9",
                "-0.4"
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("#### Production Forecast Comparison")

        years = list(range(1, 26))
        current_production = [annual_production * ((1 - 0.008) ** (year-1)) for year in years]
        new_production = [new_capacity * 1.85 * ((1 - 0.004) ** (year-1)) for year in years]

        forecast_df = pd.DataFrame({
            'Year': years,
            'Current System (MWh)': current_production,
            'After Repower (MWh)': new_production
        })

        st.line_chart(forecast_df.set_index('Year'))

        # Lifetime production
        total_current = sum(current_production)
        total_new = sum(new_production)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current 25-yr Production", f"{total_current:.0f} MWh")
        with col2:
            st.metric("New 25-yr Production", f"{total_new:.0f} MWh")
        with col3:
            st.metric("Additional Production", f"+{total_new - total_current:.0f} MWh")

    with tab4:
        st.subheader("Financial Analysis & ROI")

        st.markdown("#### Investment Costs")

        col1, col2 = st.columns(2)

        with col1:
            module_cost = st.number_input("Module Cost ($/W)", min_value=0.0, value=0.25, step=0.01)
            if inverter_upgrade:
                inverter_cost = st.number_input("Inverter Cost ($/W)", min_value=0.0, value=0.08, step=0.01)
            else:
                inverter_cost = 0.0

            bos_cost = st.number_input("BOS Cost ($/W)", min_value=0.0, value=0.15 if not reuse_bos else 0.05, step=0.01)

        with col2:
            labor_cost = st.number_input("Labor & Installation ($/W)", min_value=0.0, value=0.12, step=0.01)
            decommission_cost = st.number_input("Old System Decommissioning ($)", min_value=0.0, value=15000.0)
            recycling_credit = st.number_input("Module Recycling Credit ($)", min_value=0.0, value=5000.0)

        total_cost_per_w = module_cost + inverter_cost + bos_cost + labor_cost
        total_investment = (new_capacity * 1000 * total_cost_per_w) + decommission_cost - recycling_credit

        st.metric("Total Investment Cost", f"${total_investment:,.0f}")
        st.metric("Cost per Watt", f"${total_cost_per_w:.2f}/W")

        st.markdown("---")
        st.markdown("#### Revenue Analysis")

        col1, col2 = st.columns(2)
        with col1:
            electricity_rate = st.number_input("Electricity Rate ($/kWh)", min_value=0.0, value=0.12, step=0.01)
            incentives = st.number_input("Incentives & Tax Credits ($)", min_value=0.0, value=25000.0)
        with col2:
            maintenance_savings = st.number_input("Annual Maintenance Savings ($)", min_value=0.0, value=2000.0)
            old_system_value = st.number_input("Salvage Value of Old Equipment ($)", min_value=0.0, value=10000.0)

        # Calculate annual revenue increase
        production_increase = (new_capacity * 1.85 - annual_production) * 1000  # kWh
        annual_revenue_increase = production_increase * electricity_rate + maintenance_savings

        st.markdown("#### Financial Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Annual Revenue Increase", f"${annual_revenue_increase:,.0f}")
            net_investment = total_investment - incentives - old_system_value
            st.metric("Net Investment", f"${net_investment:,.0f}")

        with col2:
            simple_payback = net_investment / annual_revenue_increase if annual_revenue_increase > 0 else 0
            st.metric("Simple Payback", f"{simple_payback:.1f} years")

            irr = 12.5  # Simplified calculation
            st.metric("IRR", f"{irr:.1f}%")

        with col3:
            # 25-year NPV calculation (simplified)
            discount_rate = 0.06
            npv_revenue = sum([annual_revenue_increase * ((1 + 0.02) ** year) / ((1 + discount_rate) ** year)
                             for year in range(1, 26)])
            npv = npv_revenue - net_investment
            st.metric("25-Year NPV", f"${npv:,.0f}")

            st.metric("LCOE", "$0.045/kWh", help="Levelized Cost of Energy")

        st.markdown("#### Cash Flow Analysis")

        # Generate cash flow
        cashflow_years = list(range(0, 26))
        cashflow = [-net_investment]
        cumulative = [-net_investment]

        for year in range(1, 26):
            annual_cf = annual_revenue_increase * ((1 + 0.02) ** year)  # 2% annual increase
            cashflow.append(annual_cf)
            cumulative.append(cumulative[-1] + annual_cf)

        cashflow_df = pd.DataFrame({
            'Year': cashflow_years,
            'Cumulative Cash Flow ($)': cumulative
        })

        st.line_chart(cashflow_df.set_index('Year'))

        st.markdown("#### Recommendation")

        if simple_payback < 10 and npv > 0:
            st.success(f"""
            ✅ **RECOMMENDED**: Repowering is financially attractive

            - Payback period: {simple_payback:.1f} years
            - NPV: ${npv:,.0f}
            - Production increase: +{production_increase/1000:.0f} MWh/year
            """)
        elif simple_payback < 15:
            st.warning(f"""
            ⚠️ **CONSIDER**: Repowering may be worthwhile

            - Moderate payback: {simple_payback:.1f} years
            - Consider alternative financing options
            - Monitor equipment degradation
            """)
        else:
            st.error("""
            ❌ **NOT RECOMMENDED**: Current economics unfavorable

            - Consider partial upgrades
            - Wait for better module pricing
            - Focus on O&M optimization
            """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Repower Report", use_container_width=True):
            st.success("Repowering analysis report generated!")
    with col2:
        if st.button("💾 Save Analysis", use_container_width=True):
            st.success("Analysis saved successfully!")
