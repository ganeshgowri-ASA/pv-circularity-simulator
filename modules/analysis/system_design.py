"""
System Design & Optimization Module (Branch B05).

Features:
- System capacity sizing
- Inverter selection and sizing (string, central, micro, hybrid)
- String configuration optimization
- DC/AC ratio optimization
- Mounting system selection (fixed, single-axis, dual-axis, rooftop)
- Array layout and spacing
- Tilt and azimuth optimization
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constants import INVERTER_TYPES, MOUNTING_TYPES
from utils.validators import SystemConfiguration, ModuleSpecification
from utils.helpers import create_heatmap, create_comparison_bar_chart


class SystemDesignOptimizer:
    """PV system design and optimization."""

    def __init__(self):
        """Initialize system design optimizer."""
        self.inverter_types = INVERTER_TYPES
        self.mounting_types = MOUNTING_TYPES

    def size_system_capacity(
        self,
        load_profile: np.ndarray,
        target_coverage: float = 0.80,
        panel_power: float = 400
    ) -> Dict[str, any]:
        """
        Size system capacity based on load requirements.

        Args:
            load_profile: Hourly load profile (kW)
            target_coverage: Target load coverage fraction
            panel_power: Panel rated power (W)

        Returns:
            System sizing recommendations
        """
        # Calculate energy requirements
        daily_energy = np.sum(load_profile) * 24 / len(load_profile)  # kWh/day
        peak_load = np.max(load_profile)
        avg_load = np.mean(load_profile)

        # Size DC capacity (assuming 4.5 peak sun hours)
        peak_sun_hours = 4.5
        required_dc = (daily_energy * target_coverage) / peak_sun_hours  # kW

        # Calculate number of panels
        num_panels = int(np.ceil(required_dc * 1000 / panel_power))

        # Adjust for actual panel count
        actual_dc = (num_panels * panel_power) / 1000  # kW

        # Size AC capacity
        dc_ac_ratio = 1.25
        required_ac = actual_dc / dc_ac_ratio

        sizing = {
            "load_analysis": {
                "daily_energy": daily_energy,
                "peak_load": peak_load,
                "avg_load": avg_load,
                "target_coverage": target_coverage
            },
            "system_sizing": {
                "dc_capacity": actual_dc,
                "ac_capacity": required_ac,
                "dc_ac_ratio": dc_ac_ratio,
                "num_panels": num_panels,
                "panel_power": panel_power
            },
            "energy_production": {
                "estimated_daily": actual_dc * peak_sun_hours,
                "estimated_annual": actual_dc * peak_sun_hours * 365,
                "coverage_ratio": (actual_dc * peak_sun_hours) / daily_energy
            }
        }

        return sizing

    def select_inverter(
        self,
        dc_capacity: float,
        ac_capacity: float,
        system_type: str = "grid_tied",
        budget_constraint: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Select optimal inverter configuration.

        Args:
            dc_capacity: DC system capacity (kW)
            ac_capacity: Desired AC capacity (kW)
            system_type: System type (grid_tied, off_grid, hybrid)
            budget_constraint: Optional budget limit ($)

        Returns:
            Inverter selection recommendations
        """
        recommendations = []

        for inv_type, inv_specs in self.inverter_types.items():
            # Skip hybrid if not needed
            if system_type != "hybrid" and inv_type == "hybrid":
                continue

            # Check if inverter type can handle the capacity
            min_power, max_power = inv_specs["power_range"]

            # Calculate number of inverters needed
            if inv_type == "micro":
                # Microinverters: one per panel or per 2 panels
                num_inverters = int(np.ceil(dc_capacity / max_power))
            else:
                # String/central/hybrid inverters
                num_inverters = int(np.ceil(ac_capacity / max_power))
                if num_inverters == 0:
                    num_inverters = 1

            # Calculate total capacity and cost
            total_inv_ac = num_inverters * max_power if inv_type == "micro" else ac_capacity
            total_cost = total_inv_ac * inv_specs["cost_per_kw"]

            # Check budget constraint
            if budget_constraint and total_cost > budget_constraint:
                continue

            # Calculate efficiency-weighted score
            efficiency = inv_specs["efficiency"]
            cost_per_kw = inv_specs["cost_per_kw"]

            # Score: balance efficiency and cost
            score = (efficiency * 100) - (cost_per_kw / 10)

            recommendations.append({
                "type": inv_type,
                "name": inv_specs["name"],
                "num_inverters": num_inverters,
                "total_ac_capacity": total_inv_ac,
                "efficiency": efficiency,
                "cost_per_kw": cost_per_kw,
                "total_cost": total_cost,
                "mppt_channels": inv_specs["mppt_channels"],
                "score": score
            })

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return {
            "dc_capacity": dc_capacity,
            "ac_capacity": ac_capacity,
            "recommendations": recommendations,
            "optimal": recommendations[0] if recommendations else None
        }

    def optimize_string_configuration(
        self,
        num_modules: int,
        module_voc: float,
        module_vmp: float,
        module_isc: float,
        module_imp: float,
        inverter_voltage_range: Tuple[float, float],
        inverter_max_current: float = 20.0
    ) -> Dict[str, any]:
        """
        Optimize string configuration.

        Args:
            num_modules: Total number of modules
            module_voc: Module open-circuit voltage (V)
            module_vmp: Module maximum power voltage (V)
            module_isc: Module short-circuit current (A)
            module_imp: Module maximum power current (A)
            inverter_voltage_range: Inverter voltage range (min, max) in V
            inverter_max_current: Inverter maximum current (A)

        Returns:
            Optimal string configuration
        """
        v_min, v_max = inverter_voltage_range

        # Temperature corrections
        temp_coeff_voc = -0.003  # per °C
        temp_coeff_vmp = -0.004  # per °C
        min_temp = -10  # °C
        max_temp = 70  # °C

        # Calculate voltage at temperature extremes
        voc_max_temp = module_voc * (1 + temp_coeff_voc * (min_temp - 25))
        voc_min_temp = module_voc * (1 + temp_coeff_voc * (max_temp - 25))
        vmp_operating = module_vmp * (1 + temp_coeff_vmp * (max_temp - 25))

        # Find valid module counts per string
        valid_configs = []

        for modules_per_string in range(1, 51):
            string_voc_max = modules_per_string * voc_max_temp
            string_voc_min = modules_per_string * voc_min_temp
            string_vmp = modules_per_string * vmp_operating

            # Check voltage constraints
            if string_voc_max > v_max:
                break  # Too many modules

            if string_vmp >= v_min and string_voc_max <= v_max:
                # Check current constraints
                if module_imp <= inverter_max_current:
                    num_strings = int(np.ceil(num_modules / modules_per_string))
                    total_modules_used = num_strings * modules_per_string
                    string_power = modules_per_string * module_vmp * module_imp

                    valid_configs.append({
                        "modules_per_string": modules_per_string,
                        "num_strings": num_strings,
                        "total_modules": total_modules_used,
                        "string_voc_max": string_voc_max,
                        "string_vmp": string_vmp,
                        "string_current": module_imp,
                        "string_power": string_power,
                        "utilization": (num_modules / total_modules_used) * 100
                    })

        if not valid_configs:
            return {"error": "No valid configuration found"}

        # Select optimal: maximize utilization, minimize number of strings
        valid_configs.sort(key=lambda x: (x["utilization"], -x["num_strings"]), reverse=True)

        return {
            "optimal_config": valid_configs[0],
            "alternative_configs": valid_configs[1:5] if len(valid_configs) > 1 else [],
            "all_configs": valid_configs
        }

    def optimize_dc_ac_ratio(
        self,
        location_latitude: float,
        inverter_efficiency: float = 0.98,
        clipping_tolerance: float = 0.01
    ) -> Dict[str, any]:
        """
        Optimize DC/AC ratio based on location and clipping tolerance.

        Args:
            location_latitude: Installation latitude
            inverter_efficiency: Inverter efficiency
            clipping_tolerance: Acceptable energy clipping (fraction)

        Returns:
            Optimal DC/AC ratio recommendations
        """
        # Estimate irradiance profile based on latitude
        # Higher latitudes have more variable solar resource
        base_ratio = 1.20

        if abs(location_latitude) > 45:
            # High latitude: more variability, can use higher ratio
            optimal_ratio = 1.30
        elif abs(location_latitude) > 30:
            # Mid latitude
            optimal_ratio = 1.25
        else:
            # Low latitude: more consistent, lower ratio
            optimal_ratio = 1.20

        # Adjust for clipping tolerance
        if clipping_tolerance > 0.02:
            optimal_ratio += 0.05
        elif clipping_tolerance < 0.005:
            optimal_ratio -= 0.05

        # Calculate expected clipping and benefits
        clipping_loss = max(0, (optimal_ratio - 1.15) * 0.5)  # Simplified model
        capacity_benefit = (optimal_ratio - 1.0) * 100  # % increase in DC

        return {
            "optimal_dc_ac_ratio": round(optimal_ratio, 2),
            "range": (round(optimal_ratio - 0.05, 2), round(optimal_ratio + 0.05, 2)),
            "expected_clipping": round(clipping_loss, 3),
            "capacity_benefit": round(capacity_benefit, 1),
            "recommendation": f"Install {capacity_benefit:.0f}% more DC capacity than AC rating",
            "energy_gain": round((optimal_ratio - 1.0) * inverter_efficiency * 100, 1)
        }

    def select_mounting_system(
        self,
        installation_type: str,
        available_area: float,
        latitude: float,
        budget_per_kw: float,
        ground_coverage_ratio: float = 0.35
    ) -> Dict[str, any]:
        """
        Select optimal mounting system.

        Args:
            installation_type: Installation type (ground, rooftop, carport)
            available_area: Available area (m²)
            latitude: Installation latitude
            budget_per_kw: Available budget ($/kW)
            ground_coverage_ratio: GCR for ground-mount (0-1)

        Returns:
            Mounting system recommendations
        """
        recommendations = []

        for mount_type, mount_specs in self.mounting_types.items():
            # Filter by installation type
            if installation_type == "rooftop" and mount_type != "rooftop":
                continue
            if installation_type == "ground" and mount_type == "rooftop":
                continue

            # Check budget
            if mount_specs["cost_per_kw"] > budget_per_kw:
                continue

            # Calculate performance and cost
            performance_boost = mount_specs["performance_boost"]
            maintenance_factor = mount_specs["maintenance_factor"]
            cost = mount_specs["cost_per_kw"]

            # Calculate value score
            value_score = (performance_boost / cost) * 1000

            recommendations.append({
                "type": mount_type,
                "name": mount_specs["name"],
                "cost_per_kw": cost,
                "performance_boost": performance_boost,
                "maintenance_factor": maintenance_factor,
                "value_score": value_score,
                "annual_energy_gain": (performance_boost - 1.0) * 100
            })

        # Sort by value score
        recommendations.sort(key=lambda x: x["value_score"], reverse=True)

        return {
            "installation_type": installation_type,
            "latitude": latitude,
            "recommendations": recommendations,
            "optimal": recommendations[0] if recommendations else None
        }

    def optimize_tilt_azimuth(
        self,
        latitude: float,
        azimuth_constraint: Optional[Tuple[float, float]] = None
    ) -> Dict[str, any]:
        """
        Optimize tilt and azimuth angles.

        Args:
            latitude: Installation latitude
            azimuth_constraint: Optional azimuth constraints (min, max)

        Returns:
            Optimal tilt and azimuth recommendations
        """
        # Optimal tilt approximation
        optimal_tilt = abs(latitude)

        # Adjust for seasonal optimization
        summer_tilt = abs(latitude) - 15
        winter_tilt = abs(latitude) + 15

        # Optimal azimuth (facing equator)
        if latitude >= 0:
            optimal_azimuth = 180  # South for northern hemisphere
        else:
            optimal_azimuth = 0  # North for southern hemisphere

        # Apply constraints
        if azimuth_constraint:
            az_min, az_max = azimuth_constraint
            if optimal_azimuth < az_min or optimal_azimuth > az_max:
                optimal_azimuth = (az_min + az_max) / 2

        # Calculate expected energy production relative to optimal
        tilt_angles = np.arange(0, 91, 5)
        relative_production = []

        for tilt in tilt_angles:
            # Simplified annual energy model
            tilt_factor = np.cos(np.radians(abs(tilt - optimal_tilt)))
            relative_production.append(tilt_factor * 100)

        return {
            "optimal_tilt": round(optimal_tilt, 1),
            "optimal_azimuth": optimal_azimuth,
            "seasonal_tilts": {
                "summer_optimized": round(summer_tilt, 1),
                "winter_optimized": round(winter_tilt, 1)
            },
            "sensitivity_analysis": {
                "tilt_angles": tilt_angles.tolist(),
                "relative_production": relative_production
            },
            "recommendation": f"Install at {optimal_tilt:.0f}° tilt, {optimal_azimuth:.0f}° azimuth"
        }

    def calculate_array_spacing(
        self,
        module_length: float,
        tilt_angle: float,
        latitude: float,
        ground_coverage_ratio: float = 0.35
    ) -> Dict[str, any]:
        """
        Calculate optimal array spacing to minimize shading.

        Args:
            module_length: Module length (m)
            tilt_angle: Module tilt angle (degrees)
            latitude: Installation latitude
            ground_coverage_ratio: Desired GCR

        Returns:
            Array spacing recommendations
        """
        # Calculate row spacing for winter solstice (worst case)
        winter_solstice_altitude = 90 - abs(latitude) - 23.5

        # Shadow length
        shadow_length = module_length * np.sin(np.radians(tilt_angle)) / np.tan(np.radians(winter_solstice_altitude))

        # Row spacing to avoid shading
        min_spacing = shadow_length * 1.1  # 10% buffer

        # Calculate based on GCR
        gcr_spacing = module_length * np.cos(np.radians(tilt_angle)) / ground_coverage_ratio

        # Use the larger of the two
        optimal_spacing = max(min_spacing, gcr_spacing)

        # Calculate actual GCR
        actual_gcr = module_length * np.cos(np.radians(tilt_angle)) / optimal_spacing

        return {
            "module_length": module_length,
            "tilt_angle": tilt_angle,
            "optimal_row_spacing": round(optimal_spacing, 2),
            "minimum_spacing": round(min_spacing, 2),
            "gcr_based_spacing": round(gcr_spacing, 2),
            "actual_gcr": round(actual_gcr, 3),
            "target_gcr": ground_coverage_ratio,
            "shadow_length_winter": round(shadow_length, 2),
            "land_use_efficiency": round(actual_gcr * 100, 1)
        }


def render_system_design():
    """Render system design interface in Streamlit."""
    st.header("⚡ System Design & Optimization")
    st.markdown("Comprehensive PV system design, sizing, and configuration optimization.")

    optimizer = SystemDesignOptimizer()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Capacity Sizing",
        "🔌 Inverter Selection",
        "🔗 String Configuration",
        "🏗️ Mounting System",
        "📐 Tilt & Layout"
    ])

    with tab1:
        st.subheader("System Capacity Sizing")
        st.markdown("Size your PV system based on load requirements.")

        col1, col2 = st.columns(2)

        with col1:
            daily_energy = st.number_input("Daily Energy Requirement (kWh):", min_value=1, max_value=10000, value=50, step=5)
            peak_load = st.number_input("Peak Load (kW):", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)

        with col2:
            target_coverage = st.slider("Target Load Coverage (%):", 50, 150, 80, 5) / 100
            panel_power = st.number_input("Panel Power (W):", min_value=100, max_value=700, value=400, step=50)

        if st.button("🔍 Calculate System Size", key="calc_size"):
            # Create simple load profile
            load_profile = np.random.uniform(peak_load * 0.3, peak_load, 24)
            load_profile = load_profile * (daily_energy / (np.sum(load_profile)))

            results = optimizer.size_system_capacity(load_profile, target_coverage, panel_power)

            st.success("✅ System Sizing Complete")

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("DC Capacity", f"{results['system_sizing']['dc_capacity']:.1f} kW")

            with col2:
                st.metric("AC Capacity", f"{results['system_sizing']['ac_capacity']:.1f} kW")

            with col3:
                st.metric("Number of Panels", f"{results['system_sizing']['num_panels']:,}")

            with col4:
                st.metric("DC/AC Ratio", f"{results['system_sizing']['dc_ac_ratio']:.2f}")

            # Energy production
            st.subheader("Energy Production Estimates")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Daily Production", f"{results['energy_production']['estimated_daily']:.1f} kWh")

            with col2:
                st.metric("Annual Production", f"{results['energy_production']['estimated_annual']:,.0f} kWh")

            with col3:
                coverage = results['energy_production']['coverage_ratio'] * 100
                st.metric("Load Coverage", f"{coverage:.1f}%",
                         delta=f"{coverage - target_coverage * 100:+.1f}%")

            # Visualize load vs production
            fig = go.Figure()

            hours = np.arange(24)
            production_profile = np.zeros(24)
            # Simplified production curve (bell curve)
            production_profile = results['system_sizing']['dc_capacity'] * np.exp(-((hours - 12)**2) / 18)

            fig.add_trace(go.Scatter(
                x=hours, y=load_profile,
                name="Load Demand",
                fill='tozeroy',
                line=dict(color='#E74C3C', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=hours, y=production_profile,
                name="PV Production",
                fill='tozeroy',
                line=dict(color='#2ECC71', width=2)
            ))

            fig.update_layout(
                title="Typical Daily Load vs PV Production",
                xaxis_title="Hour of Day",
                yaxis_title="Power (kW)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Inverter Selection & Sizing")

        col1, col2 = st.columns(2)

        with col1:
            dc_capacity_inv = st.number_input("DC Capacity (kW):", min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key="dc_inv")
            ac_capacity_inv = st.number_input("AC Capacity (kW):", min_value=1.0, max_value=5000.0, value=80.0, step=10.0, key="ac_inv")

        with col2:
            system_type = st.selectbox("System Type:", ["grid_tied", "off_grid", "hybrid"])
            budget_constraint_inv = st.number_input("Budget Constraint ($, optional):", min_value=0, value=0, step=1000)

        if st.button("🔌 Find Optimal Inverter", key="select_inv"):
            budget = budget_constraint_inv if budget_constraint_inv > 0 else None

            results = optimizer.select_inverter(dc_capacity_inv, ac_capacity_inv, system_type, budget)

            if results["optimal"]:
                st.success(f"✅ Optimal Choice: {results['optimal']['name']}")

                # Display optimal configuration
                opt = results["optimal"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Inverter Type", opt['name'])

                with col2:
                    st.metric("Number Required", opt['num_inverters'])

                with col3:
                    st.metric("Efficiency", f"{opt['efficiency'] * 100:.1f}%")

                with col4:
                    st.metric("Total Cost", f"${opt['total_cost']:,.0f}")

                # Comparison chart
                st.subheader("Inverter Type Comparison")

                inv_names = [r['name'] for r in results['recommendations']]
                costs = [r['total_cost'] for r in results['recommendations']]
                efficiencies = [r['efficiency'] * 100 for r in results['recommendations']]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Total Cost", "Efficiency")
                )

                fig.add_trace(
                    go.Bar(x=inv_names, y=costs, name="Cost", marker_color='#3498DB'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(x=inv_names, y=efficiencies, name="Efficiency", marker_color='#2ECC71'),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Inverter Type", row=1, col=1)
                fig.update_xaxes(title_text="Inverter Type", row=1, col=2)
                fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
                fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)

                fig.update_layout(height=400, showlegend=False, template='plotly_white')

                st.plotly_chart(fig, use_container_width=True)

                # Detailed comparison table
                st.subheader("Detailed Comparison")
                df_inv = pd.DataFrame(results['recommendations'])
                df_inv = df_inv[['name', 'num_inverters', 'total_ac_capacity', 'efficiency', 'total_cost', 'score']]
                df_inv['efficiency'] = df_inv['efficiency'] * 100
                df_inv.columns = ['Inverter Type', 'Quantity', 'Total AC (kW)', 'Efficiency (%)', 'Total Cost ($)', 'Score']
                st.dataframe(df_inv, use_container_width=True)

    with tab3:
        st.subheader("String Configuration Optimization")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Total Number of Modules:", min_value=1, max_value=10000, value=250, step=10)
            module_voc = st.number_input("Module Voc (V):", min_value=10.0, max_value=100.0, value=49.5, step=0.1)
            module_vmp = st.number_input("Module Vmp (V):", min_value=10.0, max_value=100.0, value=41.7, step=0.1)

        with col2:
            module_isc = st.number_input("Module Isc (A):", min_value=1.0, max_value=20.0, value=11.5, step=0.1)
            module_imp = st.number_input("Module Imp (A):", min_value=1.0, max_value=20.0, value=10.8, step=0.1)
            inv_v_max = st.number_input("Inverter Max Voltage (V):", min_value=100, max_value=1500, value=1000, step=50)

        if st.button("🔗 Optimize String Configuration", key="opt_string"):
            inv_v_min = 150  # Typical minimum

            results = optimizer.optimize_string_configuration(
                num_modules, module_voc, module_vmp, module_isc, module_imp,
                (inv_v_min, inv_v_max)
            )

            if "error" not in results:
                opt = results["optimal_config"]

                st.success(f"✅ Optimal: {opt['modules_per_string']} modules/string × {opt['num_strings']} strings")

                # Display configuration
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Modules per String", opt['modules_per_string'])

                with col2:
                    st.metric("Number of Strings", opt['num_strings'])

                with col3:
                    st.metric("String Voltage", f"{opt['string_vmp']:.1f} V")

                with col4:
                    st.metric("Utilization", f"{opt['utilization']:.1f}%")

                # Alternative configurations
                if results["alternative_configs"]:
                    st.subheader("Alternative Configurations")

                    configs_data = []
                    for config in [opt] + results["alternative_configs"]:
                        configs_data.append({
                            "Modules/String": config['modules_per_string'],
                            "Num Strings": config['num_strings'],
                            "Total Modules": config['total_modules'],
                            "String Voltage (V)": f"{config['string_vmp']:.1f}",
                            "Utilization (%)": f"{config['utilization']:.1f}"
                        })

                    df_configs = pd.DataFrame(configs_data)
                    st.dataframe(df_configs, use_container_width=True)

    with tab4:
        st.subheader("Mounting System Selection")

        col1, col2 = st.columns(2)

        with col1:
            installation_type = st.selectbox("Installation Type:", ["ground", "rooftop", "carport"])
            available_area = st.number_input("Available Area (m²):", min_value=10, max_value=1000000, value=1000, step=100)

        with col2:
            latitude_mount = st.slider("Latitude:", -90, 90, 35)
            budget_per_kw = st.number_input("Budget ($/kW):", min_value=50, max_value=1000, value=300, step=50)

        if st.button("🏗️ Select Mounting System", key="select_mount"):
            results = optimizer.select_mounting_system(
                installation_type, available_area, latitude_mount, budget_per_kw
            )

            if results["optimal"]:
                opt = results["optimal"]

                st.success(f"✅ Recommended: {opt['name']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Cost", f"${opt['cost_per_kw']:.0f}/kW")

                with col2:
                    st.metric("Energy Gain", f"+{opt['annual_energy_gain']:.1f}%")

                with col3:
                    st.metric("Maintenance Factor", f"{opt['maintenance_factor']:.1f}x")

                # Comparison
                mount_names = [r['name'] for r in results['recommendations']]
                energy_gains = [r['annual_energy_gain'] for r in results['recommendations']]
                costs = [r['cost_per_kw'] for r in results['recommendations']]

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=mount_names,
                    y=energy_gains,
                    name="Energy Gain",
                    marker_color='#2ECC71',
                    yaxis='y',
                    text=[f"+{eg:.1f}%" for eg in energy_gains],
                    textposition='auto'
                ))

                fig.add_trace(go.Scatter(
                    x=mount_names,
                    y=costs,
                    name="Cost",
                    marker_color='#E74C3C',
                    yaxis='y2',
                    mode='lines+markers'
                ))

                fig.update_layout(
                    title="Mounting System Performance vs Cost",
                    xaxis_title="Mounting Type",
                    yaxis=dict(title="Energy Gain (%)", side='left'),
                    yaxis2=dict(title="Cost ($/kW)", side='right', overlaying='y'),
                    height=400,
                    template='plotly_white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Tilt, Azimuth & Array Layout Optimization")

        col1, col2 = st.columns(2)

        with col1:
            latitude_tilt = st.slider("Installation Latitude:", -90, 90, 35, key="lat_tilt")
            module_length = st.number_input("Module Length (m):", min_value=0.5, max_value=3.0, value=2.0, step=0.1)

        with col2:
            azimuth_constrained = st.checkbox("Azimuth Constrained?")
            gcr = st.slider("Ground Coverage Ratio:", 0.1, 0.6, 0.35, 0.05)

        az_constraint = None
        if azimuth_constrained:
            az_min = st.slider("Min Azimuth:", 0, 360, 150)
            az_max = st.slider("Max Azimuth:", 0, 360, 210)
            az_constraint = (az_min, az_max)

        if st.button("📐 Optimize Layout", key="opt_layout"):
            # Tilt and azimuth
            tilt_results = optimizer.optimize_tilt_azimuth(latitude_tilt, az_constraint)

            st.success(f"✅ {tilt_results['recommendation']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Optimal Tilt", f"{tilt_results['optimal_tilt']:.1f}°")

            with col2:
                st.metric("Optimal Azimuth", f"{tilt_results['optimal_azimuth']:.0f}°")

            with col3:
                st.metric("Summer Tilt", f"{tilt_results['seasonal_tilts']['summer_optimized']:.1f}°")

            # Tilt sensitivity
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=tilt_results['sensitivity_analysis']['tilt_angles'],
                y=tilt_results['sensitivity_analysis']['relative_production'],
                mode='lines+markers',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=6)
            ))

            fig.add_vline(
                x=tilt_results['optimal_tilt'],
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal"
            )

            fig.update_layout(
                title="Energy Production Sensitivity to Tilt Angle",
                xaxis_title="Tilt Angle (degrees)",
                yaxis_title="Relative Annual Production (%)",
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Array spacing
            spacing_results = optimizer.calculate_array_spacing(
                module_length, tilt_results['optimal_tilt'], latitude_tilt, gcr
            )

            st.subheader("Array Spacing Analysis")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Row Spacing", f"{spacing_results['optimal_row_spacing']:.2f} m")

            with col2:
                st.metric("Actual GCR", f"{spacing_results['actual_gcr']:.2f}")

            with col3:
                st.metric("Shadow Length", f"{spacing_results['shadow_length_winter']:.2f} m")

            with col4:
                st.metric("Land Efficiency", f"{spacing_results['land_use_efficiency']:.1f}%")

            # Visualize array layout
            fig = go.Figure()

            # Draw multiple rows
            num_rows = 5
            for i in range(num_rows):
                y_pos = i * spacing_results['optimal_row_spacing']

                # Module
                fig.add_shape(
                    type="rect",
                    x0=0, y0=y_pos,
                    x1=module_length * np.cos(np.radians(tilt_results['optimal_tilt'])),
                    y1=y_pos + 0.2,
                    fillcolor="#3498DB",
                    line=dict(color="#2C3E50", width=2)
                )

                # Shadow
                shadow_end = spacing_results['shadow_length_winter']
                fig.add_shape(
                    type="rect",
                    x0=0, y0=y_pos,
                    x1=shadow_end,
                    y1=y_pos - 0.1,
                    fillcolor="rgba(0,0,0,0.2)",
                    line=dict(width=0)
                )

            fig.update_layout(
                title=f"Array Layout (Top View) - GCR: {spacing_results['actual_gcr']:.2f}",
                xaxis_title="Distance (m)",
                yaxis_title="Distance (m)",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            fig.update_xaxes(range=[0, spacing_results['optimal_row_spacing'] * 1.2])
            fig.update_yaxes(range=[-1, num_rows * spacing_results['optimal_row_spacing']])

            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.info("💡 **System Design & Optimization** - Branch B05 | Complete Design Suite")
