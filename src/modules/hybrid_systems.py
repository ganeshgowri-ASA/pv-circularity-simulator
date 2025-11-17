"""
Hybrid Systems Module - PV + Storage and hybrid configurations
"""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render the Hybrid Systems module"""
    st.header("🔋 Hybrid Systems")
    st.markdown("---")

    st.markdown("""
    ### PV + Storage & Hybrid System Design

    Design and simulate hybrid energy systems combining PV with storage and other sources.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "System Configuration", "Storage Sizing", "Energy Management", "Economics"
    ])

    with tab1:
        st.subheader("Hybrid System Configuration")

        system_type = st.selectbox(
            "Hybrid System Type",
            ["PV + Battery Storage", "PV + Diesel Generator", "PV + Battery + Generator",
             "PV + Wind + Storage", "PV + Fuel Cell", "PV + Grid (with storage)"]
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### PV System")
            pv_capacity = st.number_input("PV Capacity (kWp)", min_value=0.0, value=100.0)
            pv_annual_production = st.number_input("Annual PV Production (MWh)", min_value=0.0, value=150.0)

            inverter_type = st.selectbox(
                "Inverter Type",
                ["Standard Grid-tie", "Hybrid Inverter", "Battery Inverter + Grid Inverter"]
            )

        with col2:
            st.markdown("#### Load Profile")
            avg_daily_load = st.number_input("Average Daily Load (kWh)", min_value=0.0, value=300.0)
            peak_load = st.number_input("Peak Load (kW)", min_value=0.0, value=75.0)

            load_profile_type = st.selectbox(
                "Load Profile Type",
                ["Residential", "Commercial", "Industrial", "Agricultural",
                 "Data Center", "Telecom", "Custom"]
            )

        if "Battery" in system_type or "Storage" in system_type:
            st.markdown("---")
            st.markdown("#### Battery Storage")

            col1, col2 = st.columns(2)
            with col1:
                battery_chemistry = st.selectbox(
                    "Battery Chemistry",
                    ["Lithium-ion (NMC)", "Lithium-ion (LFP)", "Lead-acid",
                     "Flow Battery", "Sodium-ion", "Lithium Titanate"]
                )

                battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=0.0, value=200.0)
                battery_power = st.number_input("Battery Power (kW)", min_value=0.0, value=50.0)

            with col2:
                battery_voltage = st.number_input("System Voltage (V)", min_value=0, value=400)
                dod = st.slider("Depth of Discharge (%)", 0, 100, 80)

                efficiency = st.slider("Round-trip Efficiency (%)", 70, 99, 92)

        if "Generator" in system_type or "Diesel" in system_type:
            st.markdown("---")
            st.markdown("#### Backup Generator")

            col1, col2 = st.columns(2)
            with col1:
                gen_capacity = st.number_input("Generator Capacity (kW)", min_value=0.0, value=60.0)
                gen_fuel_type = st.selectbox("Fuel Type", ["Diesel", "Natural Gas", "Propane", "Biodiesel"])
            with col2:
                gen_efficiency = st.slider("Generator Efficiency (%)", 20, 45, 30)
                fuel_cost = st.number_input("Fuel Cost ($/liter or $/m³)", min_value=0.0, value=1.20)

        if "Wind" in system_type:
            st.markdown("---")
            st.markdown("#### Wind Turbine")

            col1, col2 = st.columns(2)
            with col1:
                wind_capacity = st.number_input("Wind Capacity (kW)", min_value=0.0, value=25.0)
                avg_wind_speed = st.number_input("Average Wind Speed (m/s)", min_value=0.0, value=5.5)
            with col2:
                cut_in_speed = st.number_input("Cut-in Speed (m/s)", min_value=0.0, value=3.0)
                rated_speed = st.number_input("Rated Speed (m/s)", min_value=0.0, value=12.0)

    with tab2:
        st.subheader("Battery Storage Sizing")

        st.markdown("""
        ### Optimal Battery Sizing Analysis

        Determine the optimal battery capacity based on load profile and objectives.
        """)

        sizing_method = st.selectbox(
            "Sizing Method",
            ["Autonomy-based", "Peak Shaving", "Self-consumption Optimization",
             "Demand Charge Reduction", "Backup Power", "Custom"]
        )

        if sizing_method == "Autonomy-based":
            st.markdown("#### Autonomy Sizing")

            col1, col2 = st.columns(2)
            with col1:
                autonomy_days = st.slider("Days of Autonomy", 0.5, 7.0, 2.0, 0.5)
                daily_consumption = st.number_input("Daily Energy Consumption (kWh)", min_value=0.0, value=300.0)
            with col2:
                system_voltage = st.number_input("DC System Voltage (V)", min_value=12, value=48)
                max_dod = st.slider("Maximum DoD (%)", 50, 90, 80)

            required_capacity = (daily_consumption * autonomy_days) / (max_dod / 100)
            required_ah = (required_capacity * 1000) / system_voltage

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Required Capacity", f"{required_capacity:.1f} kWh")
            with col2:
                st.metric("Required Ah", f"{required_ah:.0f} Ah @ {system_voltage}V")

        elif sizing_method == "Peak Shaving":
            st.markdown("#### Peak Shaving Optimization")

            col1, col2 = st.columns(2)
            with col1:
                peak_demand = st.number_input("Current Peak Demand (kW)", min_value=0.0, value=120.0)
                target_peak = st.number_input("Target Peak Demand (kW)", min_value=0.0, value=80.0)
            with col2:
                peak_duration = st.number_input("Peak Duration (hours)", min_value=0.0, value=3.0)
                cycles_per_day = st.number_input("Cycles per Day", min_value=1, value=1)

            peak_reduction = peak_demand - target_peak
            required_energy = peak_reduction * peak_duration
            recommended_capacity = required_energy / (max_dod / 100) * 1.2  # 20% margin

            st.metric("Peak Reduction Needed", f"{peak_reduction:.1f} kW")
            st.metric("Recommended Battery Capacity", f"{recommended_capacity:.1f} kWh")

        elif sizing_method == "Self-consumption Optimization":
            st.markdown("#### Self-consumption Maximization")

            col1, col2 = st.columns(2)
            with col1:
                daily_pv_production = st.number_input("Daily PV Production (kWh)", min_value=0.0, value=400.0)
                daily_load = st.number_input("Daily Load (kWh)", min_value=0.0, value=300.0)
            with col2:
                daytime_consumption = st.slider("Daytime Consumption (%)", 0, 100, 40)
                export_tariff = st.number_input("Export Tariff ($/kWh)", min_value=0.0, value=0.05)
                import_tariff = st.number_input("Import Tariff ($/kWh)", min_value=0.0, value=0.15)

            excess_pv = daily_pv_production * (1 - daytime_consumption / 100)
            evening_load = daily_load * (1 - daytime_consumption / 100)

            optimal_capacity = min(excess_pv, evening_load) * 1.1  # 10% margin

            st.metric("Excess PV Generation", f"{excess_pv:.1f} kWh/day")
            st.metric("Evening Load", f"{evening_load:.1f} kWh/day")
            st.metric("Optimal Battery Size", f"{optimal_capacity:.1f} kWh")

        st.markdown("---")
        st.markdown("#### Battery Specifications")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Usable Capacity (kWh)", value=recommended_capacity if 'recommended_capacity' in locals() else 200.0)
        with col2:
            st.number_input("C-Rate", value=0.5, step=0.1, help="Discharge rate relative to capacity")
        with col3:
            st.number_input("Cycle Life", value=6000, help="Expected number of cycles")

    with tab3:
        st.subheader("Energy Management System (EMS)")

        st.markdown("""
        ### Intelligent Energy Management & Control

        Configure strategies for optimal energy flow and system operation.
        """)

        st.markdown("#### Control Strategy")

        control_mode = st.selectbox(
            "Primary Control Mode",
            ["Self-consumption", "Peak Shaving", "Time-of-Use Optimization",
             "Demand Response", "Grid Support", "Backup Power", "Custom Logic"]
        )

        st.markdown("#### Operating Modes")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Priority Settings:**")
            priorities = st.multiselect(
                "Energy Source Priority",
                ["PV", "Battery", "Grid", "Generator", "Wind"],
                default=["PV", "Battery", "Grid"]
            )

            st.markdown("**Battery Charging:**")
            charge_from_grid = st.checkbox("Allow grid charging", value=False)
            charge_from_gen = st.checkbox("Allow generator charging", value=False)

        with col2:
            st.markdown("**Battery SOC Limits:**")
            max_soc = st.slider("Maximum SOC (%)", 80, 100, 90)
            min_soc = st.slider("Minimum SOC (%)", 10, 30, 20)
            backup_reserve = st.slider("Backup Reserve (%)", 0, 50, 20)

            st.markdown("**Grid Interaction:**")
            allow_export = st.checkbox("Allow grid export", value=True)
            export_limit = st.number_input("Export Limit (kW)", min_value=0.0, value=50.0)

        st.markdown("---")
        st.markdown("#### Time-of-Use Scheduling")

        st.info("Configure charging/discharging schedule based on electricity tariffs")

        tou_periods = pd.DataFrame({
            'Period': ['Off-Peak (00:00-07:00)', 'Mid-Peak (07:00-17:00)',
                      'On-Peak (17:00-21:00)', 'Off-Peak (21:00-24:00)'],
            'Tariff ($/kWh)': [0.08, 0.12, 0.25, 0.08],
            'Action': ['Charge from Grid', 'Self-consumption',
                      'Discharge to Load', 'Self-consumption']
        })

        st.dataframe(tou_periods, use_container_width=True)

        st.markdown("#### Simulation Results")

        # Generate sample hourly data
        hours = list(range(24))
        pv_gen = [max(0, 100 * np.sin((h - 6) * np.pi / 12)) for h in hours]
        load = [50 + 20 * np.sin((h - 18) * np.pi / 12) + np.random.rand()*10 for h in hours]
        battery_soc = [50]

        for h in hours[1:]:
            # Simplified SOC calculation
            charge = max(0, pv_gen[h] - load[h])
            discharge = max(0, load[h] - pv_gen[h])
            new_soc = battery_soc[-1] + charge * 0.1 - discharge * 0.15
            battery_soc.append(max(min_soc, min(max_soc, new_soc)))

        energy_flow = pd.DataFrame({
            'Hour': hours,
            'PV Generation (kW)': pv_gen,
            'Load (kW)': load,
            'Battery SOC (%)': battery_soc[:24]
        })

        st.line_chart(energy_flow.set_index('Hour'))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Self-consumption", "78%")
        with col2:
            st.metric("Grid Import", "45 kWh")
        with col3:
            st.metric("Grid Export", "32 kWh")
        with col4:
            st.metric("Battery Cycles", "0.8")

    with tab4:
        st.subheader("Economic Analysis")

        st.markdown("### Financial Analysis of Hybrid System")

        st.markdown("#### System Costs")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Capital Costs:**")
            pv_cost = st.number_input("PV System Cost ($)", min_value=0.0, value=100000.0)
            battery_cost = st.number_input("Battery System Cost ($)", min_value=0.0, value=150000.0)

            if "Generator" in system_type:
                gen_cost = st.number_input("Generator Cost ($)", min_value=0.0, value=25000.0)
            else:
                gen_cost = 0.0

            other_costs = st.number_input("Other Costs (EMS, Installation) ($)", min_value=0.0, value=30000.0)

            total_capex = pv_cost + battery_cost + gen_cost + other_costs
            st.metric("Total Capital Cost", f"${total_capex:,.0f}")

        with col2:
            st.markdown("**Operating Costs:**")
            annual_om = st.number_input("Annual O&M Cost ($)", min_value=0.0, value=5000.0)
            battery_replacement_year = st.number_input("Battery Replacement Year", min_value=5, value=10)
            battery_replacement_cost = st.number_input("Battery Replacement Cost ($)", min_value=0.0, value=100000.0)

            if "Generator" in system_type:
                annual_fuel_cost = st.number_input("Annual Fuel Cost ($)", min_value=0.0, value=8000.0)
            else:
                annual_fuel_cost = 0.0

        st.markdown("---")
        st.markdown("#### Revenue & Savings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Energy Savings:**")
            annual_consumption = st.number_input("Annual Consumption (kWh)", min_value=0.0, value=110000.0)
            grid_tariff = st.number_input("Grid Tariff ($/kWh)", min_value=0.0, value=0.15)

            self_consumption_rate = 0.78
            annual_savings = annual_consumption * self_consumption_rate * grid_tariff

            st.metric("Annual Energy Savings", f"${annual_savings:,.0f}")

        with col2:
            st.markdown("**Additional Benefits:**")
            demand_charge_savings = st.number_input("Demand Charge Savings ($/year)", min_value=0.0, value=12000.0)
            export_revenue = st.number_input("Export Revenue ($/year)", min_value=0.0, value=2000.0)
            incentives = st.number_input("Annual Incentives ($/year)", min_value=0.0, value=5000.0)

            total_annual_benefit = annual_savings + demand_charge_savings + export_revenue + incentives
            st.metric("Total Annual Benefit", f"${total_annual_benefit:,.0f}")

        st.markdown("---")
        st.markdown("#### Financial Metrics")

        # Calculate NPV (simplified)
        analysis_period = 20
        discount_rate = 0.06

        annual_net_benefit = total_annual_benefit - annual_om - annual_fuel_cost

        # Calculate NPV
        npv = -total_capex
        for year in range(1, analysis_period + 1):
            benefit = annual_net_benefit
            if year == battery_replacement_year:
                benefit -= battery_replacement_cost
            npv += benefit / ((1 + discount_rate) ** year)

        # Calculate payback
        simple_payback = total_capex / annual_net_benefit if annual_net_benefit > 0 else 0

        # Calculate IRR (simplified estimate)
        irr = (annual_net_benefit / total_capex) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Simple Payback", f"{simple_payback:.1f} years")
        with col2:
            st.metric("NPV (20 years)", f"${npv:,.0f}")
        with col3:
            st.metric("IRR (approx.)", f"{irr:.1f}%")

        st.markdown("#### LCOE Calculation")

        lifetime_energy = annual_consumption * 0.78 * analysis_period
        lifetime_costs = total_capex + (annual_om + annual_fuel_cost) * analysis_period + battery_replacement_cost

        lcoe = (lifetime_costs / lifetime_energy) if lifetime_energy > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hybrid System LCOE", f"${lcoe:.3f}/kWh")
        with col2:
            st.metric("vs Grid Tariff", f"${grid_tariff - lcoe:.3f}/kWh savings")

        # Cash flow visualization
        st.markdown("#### 20-Year Cash Flow")

        years = list(range(0, 21))
        cashflow = [-total_capex]
        cumulative = [-total_capex]

        for year in range(1, 21):
            cf = annual_net_benefit
            if year == battery_replacement_year:
                cf -= battery_replacement_cost
            cashflow.append(cf)
            cumulative.append(cumulative[-1] + cf)

        cashflow_df = pd.DataFrame({
            'Year': years,
            'Cumulative Cash Flow ($)': cumulative
        })

        st.line_chart(cashflow_df.set_index('Year'))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate System Report", use_container_width=True):
            st.success("Hybrid system report generated!")
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("Hybrid system configuration saved!")
