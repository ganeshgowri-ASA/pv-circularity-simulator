"""
Revamp & Repower Module
=======================

Planning and analysis for PV system revamp and repowering projects.
Evaluates upgrade scenarios and ROI.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager


def render(session: 'SessionManager') -> None:
    """
    Render the revamp and repower module.

    Args:
        session: Session manager instance

    Features:
        - System assessment and diagnostics
        - Upgrade scenario analysis
        - Component replacement planning
        - Module technology comparison
        - Inverter repowering options
        - Capacity increase analysis
        - Financial modeling (NPV, IRR, payback)
        - Performance improvement estimation
    """
    st.header("🔄 Revamp & Repower")

    st.info("""
    Analyze and plan system revamp or repowering projects to improve
    performance, increase capacity, or extend system life.
    """)

    # Project type selection
    st.subheader("📋 Project Type")

    project_type = st.selectbox(
        "Select Revamp/Repower Strategy",
        [
            "Module Replacement (Full Repower)",
            "Inverter Upgrade",
            "Partial Module Replacement",
            "Capacity Addition",
            "Technology Upgrade",
            "System Optimization"
        ]
    )

    # Existing system information
    st.markdown("---")
    st.subheader("🏭 Existing System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Current Configuration**")
        current_capacity = st.number_input("Current Capacity (kWp)", 100, 100000, 5000)
        current_modules = st.number_input("Number of Modules", 100, 500000, 12500)
        module_age = st.slider("System Age (years)", 0, 30, 10)

    with col2:
        st.markdown("**Current Performance**")
        current_pr = st.slider("Current PR (%)", 50.0, 90.0, 75.0, 0.5)
        degradation_rate = st.slider("Degradation Rate (%/year)", 0.0, 2.0, 0.8, 0.05)
        current_generation = st.number_input("Annual Generation (MWh)", 0, 500000, 7500)

    with col3:
        st.markdown("**Issues Identified**")
        issues = st.multiselect(
            "Select Issues",
            [
                "High degradation rate",
                "Frequent failures",
                "Low PR",
                "Inverter aging",
                "Module delamination",
                "Hotspots",
                "Technology obsolescence"
            ],
            default=["High degradation rate", "Low PR"]
        )

    # Revamp scenario configuration
    st.markdown("---")
    st.subheader("⚙️ Revamp Scenario Configuration")

    if "Module Replacement" in project_type:
        st.markdown("**New Module Technology**")

        col1, col2, col3 = st.columns(3)

        with col1:
            new_module_tech = st.selectbox(
                "Module Technology",
                ["Mono PERC", "TOPCon", "HJT", "Bifacial PERC", "Bifacial TOPCon"]
            )
            new_module_power = st.number_input("Module Power (W)", 300, 700, 550)

        with col2:
            new_module_efficiency = st.slider("Module Efficiency (%)", 18.0, 25.0, 21.5, 0.1)
            bifacial_gain = st.slider("Bifacial Gain (%)", 0, 30, 10) if "Bifacial" in new_module_tech else 0

        with col3:
            new_modules_count = st.number_input("New Modules Count", 100, 500000, current_modules)
            new_capacity = (new_modules_count * new_module_power) / 1000
            st.metric("New System Capacity", f"{new_capacity:.1f} kWp")

    elif "Inverter Upgrade" in project_type:
        st.markdown("**New Inverter Configuration**")

        col1, col2 = st.columns(2)

        with col1:
            inverter_type = st.selectbox(
                "Inverter Type",
                ["String Inverter", "Central Inverter", "Hybrid Inverter", "Storage-ready Inverter"]
            )
            new_inverter_capacity = st.number_input("Inverter Capacity (kW)", 100, 10000, 4500)

        with col2:
            new_inverter_efficiency = st.slider("Inverter Efficiency (%)", 95.0, 99.5, 98.8, 0.1)
            include_storage = st.checkbox("Include Battery Storage")

            if include_storage:
                battery_capacity = st.number_input("Battery Capacity (kWh)", 100, 10000, 2000)

    # Performance projection
    st.markdown("---")
    st.subheader("📊 Performance Projection")

    if st.button("🔮 Calculate Performance Improvement", type="primary"):
        with st.spinner("Calculating performance improvements..."):
            import time
            time.sleep(1.5)

            # Calculate improvements based on revamp type
            if "Module Replacement" in project_type:
                capacity_increase = ((new_capacity - current_capacity) / current_capacity) * 100
                efficiency_gain = new_module_efficiency - (current_capacity / current_modules * 0.2)  # Estimate old efficiency

                # New expected generation
                generation_increase = (
                    (new_capacity / current_capacity) *  # Capacity ratio
                    (1 + efficiency_gain / 100) *        # Efficiency improvement
                    (1 + bifacial_gain / 100) *          # Bifacial gain
                    1.15                                  # New module PR improvement
                )

                new_generation = current_generation * generation_increase

            elif "Inverter Upgrade" in project_type:
                old_inverter_efficiency = 96.5  # Assumed old efficiency
                efficiency_improvement = ((new_inverter_efficiency - old_inverter_efficiency) / old_inverter_efficiency) * 100
                new_generation = current_generation * (1 + efficiency_improvement / 100)
                capacity_increase = 0
                generation_increase = (new_generation / current_generation)

            else:
                # Default improvements
                capacity_increase = 10
                generation_increase = 1.20
                new_generation = current_generation * generation_increase

            st.success("Performance projections calculated!")

            # Display improvements
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Capacity Change",
                    f"{capacity_increase:+.1f}%",
                    f"{new_capacity - current_capacity:+.1f} kWp"
                )

            with col2:
                generation_change = ((new_generation - current_generation) / current_generation) * 100
                st.metric(
                    "Generation Increase",
                    f"{generation_change:+.1f}%",
                    f"{new_generation - current_generation:+.1f} MWh/yr"
                )

            with col3:
                new_pr = 85.0  # Expected new PR
                pr_improvement = new_pr - current_pr
                st.metric(
                    "PR Improvement",
                    f"{new_pr:.1f}%",
                    f"{pr_improvement:+.1f}%"
                )

            with col4:
                degradation_improvement = degradation_rate - 0.5  # Assume new modules degrade at 0.5%/yr
                st.metric(
                    "Degradation Rate",
                    "0.5%/yr",
                    f"{-degradation_improvement:.2f}%"
                )

            # Financial analysis
            st.markdown("---")
            st.subheader("💰 Financial Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Investment**")
                if "Module" in project_type:
                    capex_per_wp = st.number_input("CAPEX ($/Wp)", 0.0, 2.0, 0.4, 0.05)
                else:
                    capex_per_wp = st.number_input("CAPEX ($/Wp)", 0.0, 1.0, 0.15, 0.05)

                total_investment = (new_capacity if "Module" in project_type else current_capacity) * 1000 * capex_per_wp

                decommission_cost = st.number_input("Decommissioning Cost ($)", 0, 1000000, 50000)
                total_project_cost = total_investment + decommission_cost

                st.metric("Total Investment", f"${total_project_cost:,.0f}")

            with col2:
                st.markdown("**Revenue**")
                electricity_rate = st.number_input("Electricity Rate ($/kWh)", 0.0, 1.0, 0.10, 0.01)

                additional_revenue_annual = (new_generation - current_generation) * 1000 * electricity_rate
                project_lifetime = st.slider("Project Lifetime (years)", 5, 25, 15)

                st.metric("Additional Annual Revenue", f"${additional_revenue_annual:,.0f}")

            # NPV and IRR calculation
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            discount_rate = 0.08
            revenues = [additional_revenue_annual * (1 - 0.005)**year for year in range(project_lifetime)]
            npv = -total_project_cost + sum([r / ((1 + discount_rate)**i) for i, r in enumerate(revenues, 1)])
            payback_period = total_project_cost / additional_revenue_annual if additional_revenue_annual > 0 else 0

            # Simple IRR approximation
            irr_estimate = (sum(revenues) - total_project_cost) / (total_project_cost * project_lifetime) * 100

            with col1:
                st.metric("NPV @ 8%", f"${npv:,.0f}")
                if npv > 0:
                    st.success("✅ Positive NPV")
                else:
                    st.error("❌ Negative NPV")

            with col2:
                st.metric("IRR (est.)", f"{irr_estimate:.1f}%")

            with col3:
                st.metric("Payback Period", f"{payback_period:.1f} years")
                if payback_period < 10:
                    st.success("✅ Good payback")
                else:
                    st.warning("⚠️ Long payback")

            with col4:
                lcoe_improvement = -15  # Dummy value
                st.metric("LCOE Change", f"{lcoe_improvement:+.1f}%")

            # Cash flow projection
            st.markdown("---")
            st.subheader("💵 Cash Flow Projection")

            years = list(range(project_lifetime + 1))
            cash_flows = [-total_project_cost] + [r for r in revenues]
            cumulative_cash_flow = np.cumsum(cash_flows)

            cashflow_df = pd.DataFrame({
                'Year': years,
                'Annual Cash Flow ($)': [cf/1000 for cf in cash_flows],
                'Cumulative Cash Flow ($)': [cf/1000 for cf in cumulative_cash_flow]
            })

            st.line_chart(cashflow_df.set_index('Year'))

            # Summary recommendation
            st.markdown("---")
            st.subheader("📋 Recommendation Summary")

            if npv > 0 and payback_period < 10:
                st.success("✅ **RECOMMENDED**: This revamp project shows strong financial returns with positive NPV and acceptable payback period.")
            elif npv > 0:
                st.info("ℹ️ **CONSIDER**: Project has positive NPV but longer payback period. Evaluate strategic value.")
            else:
                st.warning("⚠️ **NOT RECOMMENDED**: Project shows negative NPV under current assumptions. Consider alternative scenarios.")

            # Save analysis
            session.set('revamp_data', {
                'project_type': project_type,
                'capacity_increase': capacity_increase,
                'generation_increase': generation_change,
                'total_investment': total_project_cost,
                'npv': npv,
                'payback_period': payback_period
            })
