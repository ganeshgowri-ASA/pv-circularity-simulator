"""
Hybrid Energy System Design Module (Branch B12).

Features:
- PV + Battery storage (lithium-ion, lead-acid, flow batteries)
- PV + Wind hybrid systems
- PV + Hydrogen (electrolyzer, H2 storage, fuel cell)
- Energy management strategies (self-consumption, grid support)
- Self-sufficiency and self-consumption calculations
- Component sizing optimization
- Economic analysis (LCOE, ROI, grid independence)
- Energy flow modeling and visualization
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.constants import (
    BATTERY_TYPES,
    INVERTER_TYPES,
    FINANCIAL_DEFAULTS
)
from utils.validators import HybridSystemDesign
from utils.helpers import calculate_npv, calculate_irr


class HybridSystemDesigner:
    """Comprehensive hybrid energy system design and optimization."""

    def __init__(self):
        """Initialize hybrid system designer."""
        self.battery_types = BATTERY_TYPES
        self.hydrogen_efficiency = {
            'electrolyzer': 0.70,
            'storage': 0.95,
            'fuel_cell': 0.55
        }

    def size_battery_system(
        self,
        pv_capacity: float,
        daily_load: float,
        autonomy_days: float = 1.0,
        battery_type: str = 'lithium_ion',
        depth_of_discharge: float = 0.80
    ) -> Dict[str, any]:
        """
        Size battery storage system.

        Args:
            pv_capacity: PV system capacity (kW)
            daily_load: Daily energy consumption (kWh)
            autonomy_days: Days of autonomy required
            battery_type: Battery technology
            depth_of_discharge: Maximum depth of discharge (fraction)

        Returns:
            Battery sizing results
        """
        battery_specs = self.battery_types.get(battery_type, self.battery_types['lithium_ion'])

        # Required energy capacity
        required_capacity = (daily_load * autonomy_days) / depth_of_discharge

        # Battery power sizing (based on PV capacity)
        battery_power = pv_capacity * 0.8  # 0.8 C-rate typical

        # Calculate number of cycles per year
        daily_cycles = 1.0
        annual_cycles = daily_cycles * 365

        # Battery lifetime
        cycle_life = battery_specs['cycle_life']
        battery_lifetime_years = min(25, cycle_life / annual_cycles)

        # System costs
        battery_cost = required_capacity * battery_specs['cost_per_kwh']
        inverter_cost = battery_power * 200  # $/kW for hybrid inverter
        bos_cost = (battery_cost + inverter_cost) * 0.20
        total_cost = battery_cost + inverter_cost + bos_cost

        # Operating characteristics
        roundtrip_efficiency = battery_specs['efficiency'] ** 2
        annual_throughput = daily_load * 365
        lifetime_throughput = annual_throughput * battery_lifetime_years * roundtrip_efficiency

        # Replacement schedule
        replacements_needed = int(25 / battery_lifetime_years)
        total_lifetime_cost = total_cost * (1 + replacements_needed * 0.7)  # 70% cost for replacements

        return {
            'battery_type': battery_type,
            'battery_capacity_kwh': required_capacity,
            'battery_power_kw': battery_power,
            'battery_voltage': 400,  # Typical high-voltage battery
            'depth_of_discharge': depth_of_discharge,
            'roundtrip_efficiency': roundtrip_efficiency,
            'cycle_life': cycle_life,
            'battery_lifetime_years': battery_lifetime_years,
            'annual_cycles': annual_cycles,
            'battery_cost': battery_cost,
            'inverter_cost': inverter_cost,
            'bos_cost': bos_cost,
            'total_initial_cost': total_cost,
            'replacements_needed': replacements_needed,
            'total_lifetime_cost': total_lifetime_cost,
            'annual_throughput_kwh': annual_throughput,
            'lifetime_throughput_kwh': lifetime_throughput,
            'cost_per_kwh_throughput': total_lifetime_cost / lifetime_throughput if lifetime_throughput > 0 else 0
        }

    def size_wind_system(
        self,
        pv_capacity: float,
        wind_fraction: float = 0.30,
        avg_wind_speed: float = 6.5,
        hub_height: float = 80
    ) -> Dict[str, any]:
        """
        Size wind turbine for hybrid system.

        Args:
            pv_capacity: PV system capacity (kW)
            wind_fraction: Wind as fraction of total capacity
            avg_wind_speed: Average wind speed (m/s)
            hub_height: Hub height (m)

        Returns:
            Wind system sizing results
        """
        # Wind capacity based on desired fraction
        wind_capacity = (pv_capacity * wind_fraction) / (1 - wind_fraction)

        # Wind power calculation (simplified)
        air_density = 1.225  # kg/m³
        rotor_diameter = (wind_capacity * 1000 / (0.5 * air_density * np.pi * (avg_wind_speed ** 3) * 0.35)) ** 0.5
        swept_area = np.pi * (rotor_diameter / 2) ** 2

        # Capacity factor estimation
        if avg_wind_speed < 5:
            capacity_factor = 0.15
        elif avg_wind_speed < 6:
            capacity_factor = 0.25
        elif avg_wind_speed < 7:
            capacity_factor = 0.35
        else:
            capacity_factor = 0.45

        # Annual energy production
        annual_energy = wind_capacity * capacity_factor * 8760

        # Costs
        turbine_cost = wind_capacity * 1500  # $/kW
        installation_cost = turbine_cost * 0.25
        foundation_cost = wind_capacity * 200
        total_cost = turbine_cost + installation_cost + foundation_cost

        # O&M costs
        annual_om_cost = wind_capacity * 30  # $/kW/year

        return {
            'wind_capacity_kw': wind_capacity,
            'rotor_diameter_m': rotor_diameter,
            'swept_area_m2': swept_area,
            'hub_height_m': hub_height,
            'avg_wind_speed_ms': avg_wind_speed,
            'capacity_factor': capacity_factor,
            'annual_energy_kwh': annual_energy,
            'turbine_cost': turbine_cost,
            'installation_cost': installation_cost,
            'foundation_cost': foundation_cost,
            'total_cost': total_cost,
            'annual_om_cost': annual_om_cost,
            'cost_per_kw': total_cost / wind_capacity,
            'levelized_cost': (total_cost + annual_om_cost * 20) / (annual_energy * 20)
        }

    def size_hydrogen_system(
        self,
        pv_capacity: float,
        daily_excess_energy: float,
        storage_days: int = 7
    ) -> Dict[str, any]:
        """
        Size hydrogen energy storage system.

        Args:
            pv_capacity: PV system capacity (kW)
            daily_excess_energy: Average daily excess PV energy (kWh)
            storage_days: Days of hydrogen storage

        Returns:
            Hydrogen system sizing results
        """
        # Electrolyzer sizing
        electrolyzer_efficiency = self.hydrogen_efficiency['electrolyzer']
        electrolyzer_capacity = pv_capacity * 0.6  # 60% of PV capacity

        # Hydrogen production
        daily_h2_production = (daily_excess_energy * electrolyzer_efficiency) / 33.3  # kg H2 (33.3 kWh/kg LHV)
        annual_h2_production = daily_h2_production * 365

        # Storage sizing
        h2_storage_capacity = daily_h2_production * storage_days
        storage_pressure = 350  # bar (typical for stationary storage)
        storage_volume = h2_storage_capacity / (0.042 * storage_pressure / 100)  # m³

        # Fuel cell sizing
        fuel_cell_efficiency = self.hydrogen_efficiency['fuel_cell']
        fuel_cell_capacity = pv_capacity * 0.4  # 40% of PV capacity

        # Energy flows
        h2_energy_content = h2_storage_capacity * 33.3  # kWh (LHV)
        recoverable_energy = h2_energy_content * fuel_cell_efficiency

        # Costs
        electrolyzer_cost = electrolyzer_capacity * 1200  # $/kW
        h2_storage_cost = h2_storage_capacity * 500  # $/kg storage
        fuel_cell_cost = fuel_cell_capacity * 2000  # $/kW
        bos_cost = (electrolyzer_cost + h2_storage_cost + fuel_cell_cost) * 0.25

        total_cost = electrolyzer_cost + h2_storage_cost + fuel_cell_cost + bos_cost

        # Overall efficiency
        roundtrip_efficiency = electrolyzer_efficiency * self.hydrogen_efficiency['storage'] * fuel_cell_efficiency

        # O&M costs
        annual_om_cost = total_cost * 0.03  # 3% of capital cost

        return {
            'electrolyzer_capacity_kw': electrolyzer_capacity,
            'electrolyzer_efficiency': electrolyzer_efficiency,
            'daily_h2_production_kg': daily_h2_production,
            'annual_h2_production_kg': annual_h2_production,
            'h2_storage_capacity_kg': h2_storage_capacity,
            'storage_pressure_bar': storage_pressure,
            'storage_volume_m3': storage_volume,
            'fuel_cell_capacity_kw': fuel_cell_capacity,
            'fuel_cell_efficiency': fuel_cell_efficiency,
            'h2_energy_content_kwh': h2_energy_content,
            'recoverable_energy_kwh': recoverable_energy,
            'roundtrip_efficiency': roundtrip_efficiency,
            'electrolyzer_cost': electrolyzer_cost,
            'storage_cost': h2_storage_cost,
            'fuel_cell_cost': fuel_cell_cost,
            'bos_cost': bos_cost,
            'total_cost': total_cost,
            'annual_om_cost': annual_om_cost,
            'cost_per_kg_h2': total_cost / (annual_h2_production * 20) if annual_h2_production > 0 else 0
        }

    def calculate_energy_flows(
        self,
        pv_generation: np.ndarray,
        wind_generation: np.ndarray,
        load_demand: np.ndarray,
        battery_capacity: float,
        battery_efficiency: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calculate energy flows for hybrid system.

        Args:
            pv_generation: PV generation profile (kW)
            wind_generation: Wind generation profile (kW)
            load_demand: Load demand profile (kW)
            battery_capacity: Battery capacity (kWh)
            battery_efficiency: Battery roundtrip efficiency

        Returns:
            Energy flow results
        """
        timesteps = len(pv_generation)
        dt = 1.0  # hour

        # Initialize arrays
        total_generation = pv_generation + wind_generation
        battery_soc = np.zeros(timesteps)
        battery_charge = np.zeros(timesteps)
        battery_discharge = np.zeros(timesteps)
        grid_export = np.zeros(timesteps)
        grid_import = np.zeros(timesteps)
        load_served = np.zeros(timesteps)
        curtailment = np.zeros(timesteps)

        # Initial SOC
        soc = battery_capacity * 0.5  # Start at 50%

        for t in range(timesteps):
            generation = total_generation[t]
            demand = load_demand[t]
            net_energy = generation - demand

            if net_energy > 0:
                # Excess generation
                # Try to charge battery
                available_capacity = battery_capacity - soc
                charge_power = min(net_energy, available_capacity / dt, battery_capacity * 0.5)  # Max 0.5C charge rate

                battery_charge[t] = charge_power
                soc += charge_power * dt * battery_efficiency

                # Remaining excess to grid or curtail
                remaining = net_energy - charge_power
                grid_export[t] = remaining

                load_served[t] = demand

            else:
                # Deficit
                deficit = -net_energy

                # Try to discharge battery
                available_energy = soc
                discharge_power = min(deficit, available_energy / dt, battery_capacity * 0.5)  # Max 0.5C discharge rate

                battery_discharge[t] = discharge_power
                soc -= discharge_power * dt / battery_efficiency

                # Remaining deficit from grid
                remaining_deficit = deficit - discharge_power
                grid_import[t] = remaining_deficit

                load_served[t] = demand

            # Ensure SOC within limits
            soc = max(0, min(battery_capacity, soc))
            battery_soc[t] = soc

        return {
            'pv_generation': pv_generation,
            'wind_generation': wind_generation,
            'total_generation': total_generation,
            'load_demand': load_demand,
            'battery_soc': battery_soc,
            'battery_charge': battery_charge,
            'battery_discharge': battery_discharge,
            'grid_export': grid_export,
            'grid_import': grid_import,
            'load_served': load_served,
            'curtailment': curtailment
        }

    def calculate_self_sufficiency(
        self,
        energy_flows: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate self-sufficiency and self-consumption metrics.

        Args:
            energy_flows: Energy flow results

        Returns:
            Performance metrics
        """
        # Total energy calculations
        total_generation = energy_flows['total_generation'].sum()
        total_load = energy_flows['load_demand'].sum()
        grid_import = energy_flows['grid_import'].sum()
        grid_export = energy_flows['grid_export'].sum()
        battery_charge = energy_flows['battery_charge'].sum()
        battery_discharge = energy_flows['battery_discharge'].sum()

        # Self-sufficiency: fraction of load served by local generation
        energy_from_local = total_load - grid_import
        self_sufficiency = energy_from_local / total_load if total_load > 0 else 0

        # Self-consumption: fraction of local generation consumed locally
        energy_consumed_locally = total_generation - grid_export
        self_consumption = energy_consumed_locally / total_generation if total_generation > 0 else 0

        # Grid independence
        grid_independence = 1 - (grid_import / total_load) if total_load > 0 else 0

        # Battery utilization
        battery_utilization = battery_discharge / (total_generation * 0.3) if total_generation > 0 else 0

        # Energy balance check
        energy_balance = total_generation + grid_import - total_load - grid_export - (battery_charge - battery_discharge)

        return {
            'self_sufficiency': self_sufficiency,
            'self_consumption': self_consumption,
            'grid_independence': grid_independence,
            'battery_utilization': min(1.0, battery_utilization),
            'total_generation_kwh': total_generation,
            'total_load_kwh': total_load,
            'grid_import_kwh': grid_import,
            'grid_export_kwh': grid_export,
            'energy_balance': energy_balance
        }

    def calculate_economics(
        self,
        system_cost: float,
        annual_generation: float,
        annual_load: float,
        grid_import: float,
        grid_export: float,
        electricity_price: float = 0.12,
        feed_in_tariff: float = 0.06,
        project_lifetime: int = 25,
        discount_rate: float = 0.08
    ) -> Dict[str, float]:
        """
        Calculate hybrid system economics.

        Args:
            system_cost: Total system cost ($)
            annual_generation: Annual energy generation (kWh)
            annual_load: Annual load (kWh)
            grid_import: Annual grid import (kWh)
            grid_export: Annual grid export (kWh)
            electricity_price: Retail electricity price ($/kWh)
            feed_in_tariff: Feed-in tariff ($/kWh)
            project_lifetime: Project lifetime (years)
            discount_rate: Discount rate (fraction)

        Returns:
            Economic metrics
        """
        # Annual savings
        grid_cost_avoided = (annual_load - grid_import) * electricity_price
        feed_in_revenue = grid_export * feed_in_tariff
        annual_savings = grid_cost_avoided + feed_in_revenue

        # O&M costs
        annual_om = system_cost * 0.02  # 2% of capital cost

        # Net annual benefit
        annual_net_benefit = annual_savings - annual_om

        # Simple payback
        simple_payback = system_cost / annual_net_benefit if annual_net_benefit > 0 else 999

        # NPV calculation
        cash_flows = [-system_cost]
        for year in range(1, project_lifetime + 1):
            # Escalate electricity price
            escalated_savings = annual_savings * ((1.025) ** year)
            escalated_om = annual_om * ((1.03) ** year)
            net_cf = escalated_savings - escalated_om
            cash_flows.append(net_cf)

        npv = calculate_npv(cash_flows, discount_rate)
        irr = calculate_irr(cash_flows)

        # LCOE
        total_energy = annual_generation * project_lifetime
        total_cost = system_cost + annual_om * project_lifetime
        lcoe = total_cost / total_energy if total_energy > 0 else 0

        # ROI
        total_savings = sum([annual_savings * ((1.025) ** year) for year in range(1, project_lifetime + 1)])
        total_om_cost = sum([annual_om * ((1.03) ** year) for year in range(1, project_lifetime + 1)])
        roi = ((total_savings - total_om_cost - system_cost) / system_cost) * 100

        return {
            'system_cost': system_cost,
            'annual_savings': annual_savings,
            'annual_om_cost': annual_om,
            'annual_net_benefit': annual_net_benefit,
            'simple_payback_years': simple_payback,
            'npv': npv,
            'irr': irr,
            'lcoe': lcoe,
            'roi_pct': roi,
            'total_lifetime_savings': total_savings - total_om_cost
        }


def render_hybrid_systems():
    """Render hybrid energy system design interface in Streamlit."""
    st.header("⚡ Hybrid Energy System Design")
    st.markdown("Design optimized hybrid systems: PV + Battery + Wind + Hydrogen.")

    designer = HybridSystemDesigner()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔋 PV + Battery",
        "💨 PV + Wind",
        "💧 PV + Hydrogen",
        "📊 Energy Management",
        "💰 Economic Analysis"
    ])

    with tab1:
        st.subheader("PV + Battery Storage System")

        col1, col2 = st.columns(2)

        with col1:
            pv_capacity = st.number_input("PV System Capacity (kW):", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
            daily_load = st.number_input("Daily Energy Consumption (kWh):", min_value=1.0, max_value=100000.0, value=400.0, step=10.0)

        with col2:
            battery_type = st.selectbox("Battery Technology:", list(designer.battery_types.keys()))
            autonomy_days = st.slider("Days of Autonomy:", 0.5, 5.0, 1.0, 0.5)
            dod = st.slider("Depth of Discharge:", 0.5, 0.95, 0.80, 0.05)

        if st.button("🔋 Size Battery System", key="size_battery"):
            with st.spinner("Sizing battery system..."):
                battery_system = designer.size_battery_system(
                    pv_capacity,
                    daily_load,
                    autonomy_days,
                    battery_type,
                    dod
                )

            st.session_state['battery_system'] = battery_system
            st.session_state['pv_capacity'] = pv_capacity

            st.success(f"✅ Battery system sized: {battery_system['battery_capacity_kwh']:.1f} kWh / {battery_system['battery_power_kw']:.1f} kW")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Battery Capacity", f"{battery_system['battery_capacity_kwh']:.1f} kWh")
                st.metric("Battery Power", f"{battery_system['battery_power_kw']:.1f} kW")

            with col2:
                st.metric("Roundtrip Efficiency", f"{battery_system['roundtrip_efficiency']*100:.1f}%")
                st.metric("Cycle Life", f"{battery_system['cycle_life']:,} cycles")

            with col3:
                st.metric("Battery Lifetime", f"{battery_system['battery_lifetime_years']:.1f} years")
                st.metric("Annual Cycles", f"{battery_system['annual_cycles']:.0f}")

            with col4:
                st.metric("Total Cost", f"${battery_system['total_initial_cost']:,.0f}")
                st.metric("Cost/kWh", f"${battery_system['battery_cost']/battery_system['battery_capacity_kwh']:.0f}")

            # Cost breakdown
            st.subheader("Cost Breakdown")

            fig = go.Figure()

            costs = {
                'Battery': battery_system['battery_cost'],
                'Inverter': battery_system['inverter_cost'],
                'BOS': battery_system['bos_cost']
            }

            fig.add_trace(go.Pie(
                labels=list(costs.keys()),
                values=list(costs.values()),
                marker=dict(colors=['#3498DB', '#2ECC71', '#F39C12'])
            ))

            fig.update_layout(
                title="System Cost Distribution",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Lifetime analysis
            st.subheader("Lifetime Cost Analysis")

            years = list(range(26))
            cumulative_cost = [0]
            current_cost = battery_system['total_initial_cost']

            for year in years[1:]:
                if year % battery_system['battery_lifetime_years'] == 0 and year < 25:
                    current_cost += battery_system['battery_cost'] * 0.7  # Replacement at 70% cost
                cumulative_cost.append(current_cost)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_cost,
                mode='lines+markers',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=8),
                name='Cumulative Cost'
            ))

            fig.update_layout(
                title="25-Year Cumulative Cost with Replacements",
                xaxis_title="Year",
                yaxis_title="Cumulative Cost ($)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics
            st.subheader("Performance Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Lifetime Throughput", f"{battery_system['lifetime_throughput_kwh']:,.0f} kWh")

            with col2:
                st.metric("Replacements Needed", f"{battery_system['replacements_needed']}")

            with col3:
                st.metric("Cost per kWh Throughput", f"${battery_system['cost_per_kwh_throughput']:.3f}")

    with tab2:
        st.subheader("PV + Wind Hybrid System")

        col1, col2 = st.columns(2)

        with col1:
            pv_cap_wind = st.number_input("PV Capacity (kW):", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="pv_wind")
            wind_fraction = st.slider("Wind Fraction of Total Capacity:", 0.1, 0.7, 0.30, 0.05)

        with col2:
            avg_wind_speed = st.slider("Average Wind Speed (m/s):", 3.0, 12.0, 6.5, 0.5)
            hub_height = st.number_input("Hub Height (m):", min_value=20, max_value=150, value=80, step=10)

        if st.button("💨 Size Wind System", key="size_wind"):
            with st.spinner("Sizing wind system..."):
                wind_system = designer.size_wind_system(
                    pv_cap_wind,
                    wind_fraction,
                    avg_wind_speed,
                    hub_height
                )

            st.session_state['wind_system'] = wind_system

            st.success(f"✅ Wind system sized: {wind_system['wind_capacity_kw']:.1f} kW")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Wind Capacity", f"{wind_system['wind_capacity_kw']:.1f} kW")
                st.metric("Total Capacity", f"{pv_cap_wind + wind_system['wind_capacity_kw']:.1f} kW")

            with col2:
                st.metric("Rotor Diameter", f"{wind_system['rotor_diameter_m']:.1f} m")
                st.metric("Swept Area", f"{wind_system['swept_area_m2']:.0f} m²")

            with col3:
                st.metric("Capacity Factor", f"{wind_system['capacity_factor']*100:.1f}%")
                st.metric("Annual Energy", f"{wind_system['annual_energy_kwh']:,.0f} kWh")

            with col4:
                st.metric("Total Cost", f"${wind_system['total_cost']:,.0f}")
                st.metric("Annual O&M", f"${wind_system['annual_om_cost']:,.0f}")

            # Cost breakdown
            st.subheader("Wind System Cost Breakdown")

            fig = go.Figure()

            costs = {
                'Turbine': wind_system['turbine_cost'],
                'Installation': wind_system['installation_cost'],
                'Foundation': wind_system['foundation_cost']
            }

            fig.add_trace(go.Bar(
                x=list(costs.keys()),
                y=list(costs.values()),
                marker_color=['#3498DB', '#2ECC71', '#F39C12'],
                text=[f"${v:,.0f}" for v in costs.values()],
                textposition='auto'
            ))

            fig.update_layout(
                title="Wind System Cost Components",
                xaxis_title="Component",
                yaxis_title="Cost ($)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Complementarity analysis
            st.subheader("PV-Wind Complementarity")

            # Generate sample hourly profiles
            hours = np.arange(24)
            pv_profile = np.array([0 if h < 6 or h > 18 else np.sin(np.pi * (h - 6) / 12) for h in hours])
            wind_profile = 0.3 + 0.2 * np.random.random(24)  # More consistent

            pv_power = pv_profile * pv_cap_wind
            wind_power = wind_profile * wind_system['wind_capacity_kw']
            total_power = pv_power + wind_power

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hours,
                y=pv_power,
                mode='lines',
                name='PV',
                line=dict(color='#F39C12', width=3),
                fill='tozeroy'
            ))

            fig.add_trace(go.Scatter(
                x=hours,
                y=wind_power,
                mode='lines',
                name='Wind',
                line=dict(color='#3498DB', width=3),
                fill='tozeroy'
            ))

            fig.add_trace(go.Scatter(
                x=hours,
                y=total_power,
                mode='lines',
                name='Total',
                line=dict(color='#2ECC71', width=3, dash='dash')
            ))

            fig.update_layout(
                title="Typical Daily Generation Profile",
                xaxis_title="Hour of Day",
                yaxis_title="Power (kW)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("PV + Hydrogen Energy Storage")

        col1, col2 = st.columns(2)

        with col1:
            pv_cap_h2 = st.number_input("PV Capacity (kW):", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="pv_h2")
            daily_excess = st.number_input("Daily Excess Energy (kWh):", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)

        with col2:
            storage_days = st.slider("Hydrogen Storage Days:", 1, 30, 7)

        if st.button("💧 Size Hydrogen System", key="size_h2"):
            with st.spinner("Sizing hydrogen system..."):
                h2_system = designer.size_hydrogen_system(
                    pv_cap_h2,
                    daily_excess,
                    storage_days
                )

            st.session_state['h2_system'] = h2_system

            st.success(f"✅ Hydrogen system sized: {h2_system['h2_storage_capacity_kg']:.1f} kg H2 storage")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Electrolyzer", f"{h2_system['electrolyzer_capacity_kw']:.1f} kW")
                st.metric("Electrolyzer Eff.", f"{h2_system['electrolyzer_efficiency']*100:.0f}%")

            with col2:
                st.metric("H2 Storage", f"{h2_system['h2_storage_capacity_kg']:.1f} kg")
                st.metric("Storage Volume", f"{h2_system['storage_volume_m3']:.1f} m³")

            with col3:
                st.metric("Fuel Cell", f"{h2_system['fuel_cell_capacity_kw']:.1f} kW")
                st.metric("Fuel Cell Eff.", f"{h2_system['fuel_cell_efficiency']*100:.0f}%")

            with col4:
                st.metric("Roundtrip Eff.", f"{h2_system['roundtrip_efficiency']*100:.1f}%")
                st.metric("Total Cost", f"${h2_system['total_cost']:,.0f}")

            # Energy flow diagram
            st.subheader("Hydrogen Energy Flow")

            fig = go.Figure()

            stages = ['PV Energy', 'Electrolyzer', 'H2 Storage', 'Fuel Cell', 'Output']
            energy = [
                daily_excess,
                daily_excess * h2_system['electrolyzer_efficiency'],
                daily_excess * h2_system['electrolyzer_efficiency'] * 0.95,
                daily_excess * h2_system['roundtrip_efficiency'],
                daily_excess * h2_system['roundtrip_efficiency']
            ]

            losses = [0]
            for i in range(len(energy) - 1):
                losses.append(energy[i] - energy[i + 1])

            fig.add_trace(go.Waterfall(
                x=stages,
                y=[energy[0]] + losses[1:],
                text=[f"{e:.1f} kWh" for e in [energy[0]] + losses[1:]],
                textposition='outside',
                connector={'line': {'color': 'rgb(63, 63, 63)'}},
                decreasing={'marker': {'color': '#E74C3C'}},
                increasing={'marker': {'color': '#2ECC71'}},
                totals={'marker': {'color': '#3498DB'}}
            ))

            fig.update_layout(
                title="Daily Energy Flow (kWh)",
                yaxis_title="Energy (kWh)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cost breakdown
            st.subheader("System Cost Breakdown")

            fig = go.Figure()

            costs = {
                'Electrolyzer': h2_system['electrolyzer_cost'],
                'H2 Storage': h2_system['storage_cost'],
                'Fuel Cell': h2_system['fuel_cell_cost'],
                'BOS': h2_system['bos_cost']
            }

            fig.add_trace(go.Pie(
                labels=list(costs.keys()),
                values=list(costs.values()),
                marker=dict(colors=['#3498DB', '#2ECC71', '#F39C12', '#9B59B6'])
            ))

            fig.update_layout(
                title="Hydrogen System Cost Distribution",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Annual production
            st.subheader("Hydrogen Production")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Daily Production", f"{h2_system['daily_h2_production_kg']:.2f} kg/day")

            with col2:
                st.metric("Annual Production", f"{h2_system['annual_h2_production_kg']:.0f} kg/year")

            with col3:
                st.metric("Cost per kg H2", f"${h2_system['cost_per_kg_h2']:.2f}")

    with tab4:
        st.subheader("Energy Management & Self-Sufficiency")

        st.write("### System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            pv_cap_em = st.number_input("PV Capacity (kW):", min_value=1.0, max_value=1000.0, value=50.0, step=5.0, key="pv_em")
            wind_cap_em = st.number_input("Wind Capacity (kW):", min_value=0.0, max_value=1000.0, value=20.0, step=5.0)

        with col2:
            battery_cap_em = st.number_input("Battery Capacity (kWh):", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
            avg_load = st.number_input("Average Load (kW):", min_value=1.0, max_value=500.0, value=30.0, step=5.0)

        if st.button("⚡ Simulate Energy Flows", key="simulate_flows"):
            with st.spinner("Simulating energy management..."):
                # Generate 24-hour profiles
                hours = np.arange(24)

                # PV generation
                pv_gen = np.array([0 if h < 6 or h > 18 else pv_cap_em * np.sin(np.pi * (h - 6) / 12) for h in hours])

                # Wind generation (more variable)
                wind_gen = wind_cap_em * (0.3 + 0.4 * np.random.random(24))

                # Load profile (typical residential/commercial)
                load_profile = avg_load * np.array([
                    0.6, 0.5, 0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 0.9, 0.8,
                    0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9,
                    0.8, 0.7, 0.7, 0.6
                ])

                # Calculate energy flows
                energy_flows = designer.calculate_energy_flows(
                    pv_gen,
                    wind_gen,
                    load_profile,
                    battery_cap_em,
                    0.95
                )

                metrics = designer.calculate_self_sufficiency(energy_flows)

                st.session_state['energy_flows'] = energy_flows
                st.session_state['metrics'] = metrics

            st.success("✅ Energy flow simulation completed")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Self-Sufficiency", f"{metrics['self_sufficiency']*100:.1f}%")

            with col2:
                st.metric("Self-Consumption", f"{metrics['self_consumption']*100:.1f}%")

            with col3:
                st.metric("Grid Independence", f"{metrics['grid_independence']*100:.1f}%")

            with col4:
                st.metric("Battery Utilization", f"{metrics['battery_utilization']*100:.1f}%")

            # Energy flow visualization
            st.subheader("24-Hour Energy Flow")

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Generation & Load', 'Battery State of Charge', 'Grid Interaction', 'Energy Balance')
            )

            # Generation and load
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['pv_generation'],
                          name='PV', line=dict(color='#F39C12')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['wind_generation'],
                          name='Wind', line=dict(color='#3498DB')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['load_demand'],
                          name='Load', line=dict(color='#E74C3C', dash='dash')),
                row=1, col=1
            )

            # Battery SOC
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['battery_soc'],
                          name='SOC', line=dict(color='#2ECC71', width=3)),
                row=1, col=2
            )

            # Grid interaction
            fig.add_trace(
                go.Bar(x=hours, y=energy_flows['grid_export'],
                      name='Grid Export', marker_color='#2ECC71'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=hours, y=-energy_flows['grid_import'],
                      name='Grid Import', marker_color='#E74C3C'),
                row=2, col=1
            )

            # Energy balance
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['battery_charge'],
                          name='Charge', line=dict(color='#2ECC71')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=hours, y=energy_flows['battery_discharge'],
                          name='Discharge', line=dict(color='#E74C3C')),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Hour", row=1, col=1)
            fig.update_xaxes(title_text="Hour", row=1, col=2)
            fig.update_xaxes(title_text="Hour", row=2, col=1)
            fig.update_xaxes(title_text="Hour", row=2, col=2)

            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Energy (kWh)", row=1, col=2)
            fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

            fig.update_layout(height=700, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # Energy summary
            st.subheader("Energy Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Generation", f"{metrics['total_generation_kwh']:.1f} kWh")
                st.metric("PV Generation", f"{energy_flows['pv_generation'].sum():.1f} kWh")

            with col2:
                st.metric("Total Load", f"{metrics['total_load_kwh']:.1f} kWh")
                st.metric("Wind Generation", f"{energy_flows['wind_generation'].sum():.1f} kWh")

            with col3:
                st.metric("Grid Import", f"{metrics['grid_import_kwh']:.1f} kWh")
                st.metric("Grid Export", f"{metrics['grid_export_kwh']:.1f} kWh")

    with tab5:
        st.subheader("Economic Analysis")

        if 'metrics' not in st.session_state:
            st.warning("⚠️ Please simulate energy flows first in the Energy Management tab")
        else:
            col1, col2 = st.columns(2)

            with col1:
                system_cost = st.number_input("Total System Cost ($):", min_value=1000, max_value=10000000, value=200000, step=10000)
                electricity_price = st.number_input("Electricity Price ($/kWh):", min_value=0.01, max_value=1.0, value=0.12, step=0.01)

            with col2:
                feed_in_tariff = st.number_input("Feed-in Tariff ($/kWh):", min_value=0.0, max_value=0.5, value=0.06, step=0.01)
                discount_rate = st.slider("Discount Rate (%):", 1, 20, 8) / 100

            if st.button("💰 Calculate Economics", key="calc_econ"):
                metrics = st.session_state['metrics']

                # Annualize metrics (24 hours * 365 days)
                annual_generation = metrics['total_generation_kwh'] * 365
                annual_load = metrics['total_load_kwh'] * 365
                annual_grid_import = metrics['grid_import_kwh'] * 365
                annual_grid_export = metrics['grid_export_kwh'] * 365

                economics = designer.calculate_economics(
                    system_cost,
                    annual_generation,
                    annual_load,
                    annual_grid_import,
                    annual_grid_export,
                    electricity_price,
                    feed_in_tariff,
                    25,
                    discount_rate
                )

                st.session_state['economics'] = economics

                st.success("✅ Economic analysis completed")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("NPV", f"${economics['npv']:,.0f}")
                    st.metric("IRR", f"{economics['irr']*100:.2f}%" if economics['irr'] > 0 else "N/A")

                with col2:
                    st.metric("Simple Payback", f"{economics['simple_payback_years']:.1f} years")
                    st.metric("ROI", f"{economics['roi_pct']:.1f}%")

                with col3:
                    st.metric("LCOE", f"${economics['lcoe']:.4f}/kWh")
                    st.metric("Annual Savings", f"${economics['annual_savings']:,.0f}")

                with col4:
                    st.metric("Annual O&M", f"${economics['annual_om_cost']:,.0f}")
                    st.metric("Net Benefit", f"${economics['annual_net_benefit']:,.0f}/year")

                # Cash flow analysis
                st.subheader("25-Year Cash Flow Projection")

                years = list(range(26))
                cash_flow = [-system_cost]

                for year in range(1, 26):
                    annual_savings = economics['annual_savings'] * ((1.025) ** year)
                    annual_om = economics['annual_om_cost'] * ((1.03) ** year)
                    cash_flow.append(annual_savings - annual_om)

                cumulative_cf = np.cumsum(cash_flow)

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Annual Cash Flow', 'Cumulative Cash Flow')
                )

                fig.add_trace(
                    go.Bar(x=years, y=cash_flow,
                          marker_color=['#E74C3C' if cf < 0 else '#2ECC71' for cf in cash_flow]),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=years, y=cumulative_cf,
                              line=dict(color='#3498DB', width=3)),
                    row=1, col=2
                )

                fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

                fig.update_xaxes(title_text="Year", row=1, col=1)
                fig.update_xaxes(title_text="Year", row=1, col=2)
                fig.update_yaxes(title_text="Cash Flow ($)", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative ($)", row=1, col=2)

                fig.update_layout(height=400, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Sensitivity analysis
                st.subheader("Sensitivity Analysis - NPV vs Electricity Price")

                prices = np.linspace(electricity_price * 0.7, electricity_price * 1.3, 20)
                npvs = []

                for price in prices:
                    temp_econ = designer.calculate_economics(
                        system_cost,
                        annual_generation,
                        annual_load,
                        annual_grid_import,
                        annual_grid_export,
                        price,
                        feed_in_tariff,
                        25,
                        discount_rate
                    )
                    npvs.append(temp_econ['npv'])

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=prices,
                    y=npvs,
                    mode='lines',
                    line=dict(color='#3498DB', width=3),
                    fill='tozeroy'
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.add_vline(x=electricity_price, line_dash="dot", line_color="green",
                            annotation_text="Base Case")

                fig.update_layout(
                    title="NPV Sensitivity to Electricity Price",
                    xaxis_title="Electricity Price ($/kWh)",
                    yaxis_title="NPV ($)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Comparison with grid-only
                st.subheader("Comparison: Hybrid System vs Grid-Only")

                grid_only_cost = annual_load * electricity_price * 25

                comparison_data = {
                    'Scenario': ['Grid Only', 'Hybrid System'],
                    '25-Year Cost': [grid_only_cost, system_cost + economics['annual_om_cost'] * 25],
                    '25-Year Revenue': [0, economics['total_lifetime_savings']],
                    'Net Position': [-grid_only_cost, economics['npv']]
                }

                fig = go.Figure()

                x = comparison_data['Scenario']

                fig.add_trace(go.Bar(
                    x=x,
                    y=comparison_data['25-Year Cost'],
                    name='Total Cost',
                    marker_color='#E74C3C'
                ))

                fig.add_trace(go.Bar(
                    x=x,
                    y=comparison_data['25-Year Revenue'],
                    name='Savings',
                    marker_color='#2ECC71'
                ))

                fig.update_layout(
                    title="25-Year Economic Comparison",
                    yaxis_title="Value ($)",
                    barmode='group',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.info("💡 **Hybrid Energy System Design** - Branch B12 | PV + Battery + Wind + Hydrogen Integration")
