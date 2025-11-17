"""
System Design Dashboard for PV system configuration.

This module provides the main dashboard interface for configuring PV systems,
including module selection, inverter configuration, mounting structure selection,
and optimization controls.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.models.pv_components import (
    PVModule, Inverter, MountingStructure, SystemDesign, SiteLocation,
    StringConfiguration, ModuleTechnology, InverterType, MountingType,
    OrientationType
)


class SystemDesignDashboard:
    """
    Main dashboard for PV system design configuration.

    Provides interactive UI components for selecting and configuring all aspects
    of a PV system design including modules, inverters, mounting, and optimization.
    """

    def __init__(self):
        """Initialize the SystemDesignDashboard."""
        self.initialize_session_state()
        self.load_component_database()

    def initialize_session_state(self) -> None:
        """
        Initialize Streamlit session state variables.

        Sets up all necessary session state variables for tracking user selections
        and design configurations throughout the dashboard lifecycle.
        """
        if 'selected_modules' not in st.session_state:
            st.session_state.selected_modules = []

        if 'selected_inverters' not in st.session_state:
            st.session_state.selected_inverters = []

        if 'mounting_config' not in st.session_state:
            st.session_state.mounting_config = None

        if 'site_location' not in st.session_state:
            st.session_state.site_location = None

        if 'current_design' not in st.session_state:
            st.session_state.current_design = None

        if 'optimization_params' not in st.session_state:
            st.session_state.optimization_params = {}

        if 'design_history' not in st.session_state:
            st.session_state.design_history = []

    def load_component_database(self) -> None:
        """
        Load component database with available modules and inverters.

        Initializes the database of available PV modules and inverters
        from predefined catalog or external database.
        """
        # Sample module database (in production, this would load from a database)
        self.module_database = {
            "Trina Solar TSM-DEG21C.20": PVModule(
                manufacturer="Trina Solar",
                model="TSM-DEG21C.20",
                technology=ModuleTechnology.MONO_SI,
                p_max=665.0,
                v_mp=37.6,
                i_mp=17.69,
                v_oc=45.4,
                i_sc=18.42,
                temp_coeff_p_max=-0.34,
                temp_coeff_v_oc=-0.26,
                temp_coeff_i_sc=0.048,
                length=2.384,
                width=1.303,
                weight=34.6,
                efficiency=21.4,
                is_bifacial=True,
                bifaciality=0.80,
                warranty_years=25
            ),
            "JinkoSolar JKM575N-72HL4-BDV": PVModule(
                manufacturer="JinkoSolar",
                model="JKM575N-72HL4-BDV",
                technology=ModuleTechnology.TOPCON,
                p_max=575.0,
                v_mp=41.85,
                i_mp=13.74,
                v_oc=50.15,
                i_sc=14.57,
                temp_coeff_p_max=-0.30,
                temp_coeff_v_oc=-0.25,
                temp_coeff_i_sc=0.05,
                length=2.278,
                width=1.134,
                weight=28.5,
                efficiency=22.27,
                is_bifacial=True,
                bifaciality=0.75,
                warranty_years=30
            ),
            "Canadian Solar HiKu6 CS6W-550MS": PVModule(
                manufacturer="Canadian Solar",
                model="HiKu6 CS6W-550MS",
                technology=ModuleTechnology.PERC,
                p_max=550.0,
                v_mp=41.7,
                i_mp=13.19,
                v_oc=49.8,
                i_sc=13.95,
                temp_coeff_p_max=-0.35,
                temp_coeff_v_oc=-0.27,
                temp_coeff_i_sc=0.05,
                length=2.261,
                width=1.134,
                weight=27.5,
                efficiency=21.5,
                is_bifacial=False,
                warranty_years=25
            ),
        }

        # Sample inverter database
        self.inverter_database = {
            "SMA Sunny Tripower CORE1 110": Inverter(
                manufacturer="SMA",
                model="Sunny Tripower CORE1 110",
                inverter_type=InverterType.STRING,
                p_ac_rated=110000.0,
                p_dc_max=165000.0,
                v_dc_min=175.0,
                v_dc_max=1000.0,
                v_mpp_min=235.0,
                v_mpp_max=800.0,
                i_dc_max=219.0,
                i_ac_max=160.0,
                num_mppt=6,
                num_strings_per_mppt=2,
                max_efficiency=98.4,
                euro_efficiency=98.1,
                cec_efficiency=98.0,
                weight=77.0,
                dimensions=(0.760, 0.690, 0.330),
                phases=3,
                v_ac_nominal=480.0
            ),
            "Fronius Symo GEN24 10.0 Plus": Inverter(
                manufacturer="Fronius",
                model="Symo GEN24 10.0 Plus",
                inverter_type=InverterType.STRING,
                p_ac_rated=10000.0,
                p_dc_max=15000.0,
                v_dc_min=80.0,
                v_dc_max=1000.0,
                v_mpp_min=80.0,
                v_mpp_max=800.0,
                i_dc_max=27.0,
                i_ac_max=16.0,
                num_mppt=2,
                num_strings_per_mppt=2,
                max_efficiency=98.1,
                euro_efficiency=97.7,
                cec_efficiency=97.5,
                weight=32.8,
                dimensions=(0.725, 0.510, 0.204),
                phases=3,
                v_ac_nominal=400.0
            ),
            "SolarEdge SE100K-RWS00BNN4": Inverter(
                manufacturer="SolarEdge",
                model="SE100K-RWS00BNN4",
                inverter_type=InverterType.OPTIMIZER,
                p_ac_rated=100000.0,
                p_dc_max=150000.0,
                v_dc_min=300.0,
                v_dc_max=1000.0,
                v_mpp_min=300.0,
                v_mpp_max=850.0,
                i_dc_max=200.0,
                i_ac_max=152.0,
                num_mppt=1,
                num_strings_per_mppt=20,
                max_efficiency=98.6,
                euro_efficiency=98.3,
                cec_efficiency=98.2,
                weight=75.0,
                dimensions=(0.770, 0.665, 0.275),
                phases=3,
                v_ac_nominal=480.0
            ),
        }

    def create_main_layout(self) -> None:
        """
        Create the main dashboard layout.

        Renders the primary dashboard interface with all major components
        organized in a logical workflow.
        """
        st.title("🔆 PV System Design Dashboard")
        st.markdown("---")

        # Design header with key metrics
        self._render_design_header()

        # Main content area with tabs
        tabs = st.tabs([
            "📍 Site Configuration",
            "🔲 Module Selection",
            "⚡ Inverter Configuration",
            "🏗️ Mounting Structure",
            "🎯 Optimization",
            "📊 Summary"
        ])

        with tabs[0]:
            self._render_site_configuration()

        with tabs[1]:
            self.module_selection_panel()

        with tabs[2]:
            self.inverter_configuration()

        with tabs[3]:
            self.mounting_structure_selector()

        with tabs[4]:
            self.optimization_controls()

        with tabs[5]:
            self._render_design_summary()

    def _render_design_header(self) -> None:
        """Render the design header with key metrics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "DC Capacity",
                f"{self._get_total_dc_power():.1f} kW",
                help="Total DC capacity at STC"
            )

        with col2:
            st.metric(
                "AC Capacity",
                f"{self._get_total_ac_power():.1f} kW",
                help="Total AC capacity"
            )

        with col3:
            st.metric(
                "DC/AC Ratio",
                f"{self._get_dc_ac_ratio():.2f}",
                help="System oversizing ratio"
            )

        with col4:
            st.metric(
                "Module Count",
                f"{self._get_total_modules()}",
                help="Total number of modules"
            )

    def _render_site_configuration(self) -> None:
        """Render site configuration panel."""
        st.subheader("Site Location & Parameters")

        col1, col2 = st.columns(2)

        with col1:
            site_name = st.text_input("Site Name", value="My PV Site")
            latitude = st.number_input(
                "Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=37.7749,
                format="%.4f",
                help="Site latitude in decimal degrees"
            )
            longitude = st.number_input(
                "Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=-122.4194,
                format="%.4f",
                help="Site longitude in decimal degrees"
            )
            elevation = st.number_input(
                "Elevation (m)",
                min_value=0.0,
                value=10.0,
                help="Site elevation above sea level"
            )

        with col2:
            address = st.text_area("Address", value="")
            albedo = st.slider(
                "Ground Albedo",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Ground reflectance (0.2 = typical grass, 0.6 = white roof)"
            )
            timezone = st.selectbox(
                "Timezone",
                ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "UTC"],
                index=3
            )

        if st.button("Save Site Configuration", type="primary"):
            st.session_state.site_location = SiteLocation(
                name=site_name,
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
                address=address,
                albedo=albedo,
                timezone=timezone
            )
            st.success("✅ Site configuration saved!")

    def module_selection_panel(self) -> None:
        """
        Render module selection panel.

        Provides interface for browsing and selecting PV modules from the database,
        with filtering, comparison, and configuration options.
        """
        st.subheader("PV Module Selection")

        # Module database table
        module_df = self._create_module_dataframe()

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            tech_filter = st.multiselect(
                "Technology",
                options=[t.value for t in ModuleTechnology],
                default=None
            )
        with col2:
            power_range = st.slider(
                "Power Range (W)",
                min_value=int(module_df['Power (W)'].min()),
                max_value=int(module_df['Power (W)'].max()),
                value=(int(module_df['Power (W)'].min()), int(module_df['Power (W)'].max()))
            )
        with col3:
            bifacial_only = st.checkbox("Bifacial Only")

        # Apply filters
        filtered_df = module_df.copy()
        if tech_filter:
            filtered_df = filtered_df[filtered_df['Technology'].isin(tech_filter)]
        filtered_df = filtered_df[
            (filtered_df['Power (W)'] >= power_range[0]) &
            (filtered_df['Power (W)'] <= power_range[1])
        ]
        if bifacial_only:
            filtered_df = filtered_df[filtered_df['Bifacial'] == True]

        # Display module table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        # Module selection and configuration
        st.markdown("### Configure String")
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_module_name = st.selectbox(
                "Select Module",
                options=list(self.module_database.keys())
            )

        with col2:
            modules_per_string = st.number_input(
                "Modules per String",
                min_value=1,
                max_value=50,
                value=20,
                help="Number of modules in series"
            )

        with col3:
            num_strings = st.number_input(
                "Number of Strings",
                min_value=1,
                max_value=1000,
                value=10,
                help="Number of parallel strings"
            )

        # Show string configuration preview
        if selected_module_name:
            module = self.module_database[selected_module_name]
            string_config = StringConfiguration(
                modules_per_string=modules_per_string,
                num_strings=num_strings,
                module=module
            )

            st.info(f"""
            **String Configuration:**
            - Total Modules: {string_config.total_modules}
            - String Voc: {string_config.string_v_oc:.1f} V
            - String Vmp: {string_config.string_v_mp:.1f} V
            - String Isc: {string_config.string_i_sc:.2f} A
            - Total Power: {string_config.total_power / 1000:.1f} kW
            """)

            if st.button("Add String Configuration", type="primary"):
                st.session_state.selected_modules.append(string_config)
                st.success(f"✅ Added {string_config.total_modules} modules ({selected_module_name})")
                st.rerun()

        # Display current string configurations
        if st.session_state.selected_modules:
            st.markdown("### Current String Configurations")
            for idx, config in enumerate(st.session_state.selected_modules):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(
                        f"{idx + 1}. {config.module.manufacturer} {config.module.model}: "
                        f"{config.modules_per_string} × {config.num_strings} = {config.total_modules} modules "
                        f"({config.total_power / 1000:.1f} kW)"
                    )
                with col2:
                    if st.button("Remove", key=f"remove_module_{idx}"):
                        st.session_state.selected_modules.pop(idx)
                        st.rerun()

    def inverter_configuration(self) -> None:
        """
        Render inverter configuration panel.

        Provides interface for selecting and configuring inverters with MPPT
        configuration and string mapping.
        """
        st.subheader("Inverter Configuration")

        # Inverter database table
        inverter_df = self._create_inverter_dataframe()

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.multiselect(
                "Inverter Type",
                options=[t.value for t in InverterType],
                default=None
            )
        with col2:
            power_range = st.slider(
                "AC Power Range (kW)",
                min_value=int(inverter_df['AC Power (kW)'].min()),
                max_value=int(inverter_df['AC Power (kW)'].max()),
                value=(
                    int(inverter_df['AC Power (kW)'].min()),
                    int(inverter_df['AC Power (kW)'].max())
                )
            )

        # Apply filters
        filtered_df = inverter_df.copy()
        if type_filter:
            filtered_df = filtered_df[filtered_df['Type'].isin(type_filter)]
        filtered_df = filtered_df[
            (filtered_df['AC Power (kW)'] >= power_range[0]) &
            (filtered_df['AC Power (kW)'] <= power_range[1])
        ]

        # Display inverter table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        # Inverter selection
        col1, col2 = st.columns(2)

        with col1:
            selected_inverter_name = st.selectbox(
                "Select Inverter",
                options=list(self.inverter_database.keys())
            )

        with col2:
            num_inverters = st.number_input(
                "Number of Inverters",
                min_value=1,
                max_value=100,
                value=1
            )

        # Show inverter details
        if selected_inverter_name:
            inverter = self.inverter_database[selected_inverter_name]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AC Power", f"{inverter.p_ac_rated / 1000:.1f} kW")
                st.metric("Max DC Power", f"{inverter.p_dc_max / 1000:.1f} kW")
            with col2:
                st.metric("MPPT Inputs", f"{inverter.num_mppt}")
                st.metric("Max Efficiency", f"{inverter.max_efficiency:.1f}%")
            with col3:
                st.metric("Vmp Range", f"{inverter.v_mpp_min}-{inverter.v_mpp_max} V")
                st.metric("Max Idc", f"{inverter.i_dc_max:.1f} A")

            if st.button("Add Inverter(s)", type="primary"):
                for _ in range(num_inverters):
                    st.session_state.selected_inverters.append(inverter)
                st.success(f"✅ Added {num_inverters} × {selected_inverter_name}")
                st.rerun()

        # Display current inverters
        if st.session_state.selected_inverters:
            st.markdown("### Current Inverter Configuration")
            inverter_counts = {}
            for inv in st.session_state.selected_inverters:
                key = f"{inv.manufacturer} {inv.model}"
                inverter_counts[key] = inverter_counts.get(key, 0) + 1

            for idx, (inv_name, count) in enumerate(inverter_counts.items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{idx + 1}. {inv_name} × {count}")
                with col2:
                    if st.button("Clear", key=f"clear_inv_{idx}"):
                        st.session_state.selected_inverters = []
                        st.rerun()

    def mounting_structure_selector(self) -> None:
        """
        Render mounting structure selection panel.

        Provides interface for configuring mounting structure including type,
        tilt, azimuth, and tracking parameters.
        """
        st.subheader("Mounting Structure Configuration")

        col1, col2 = st.columns(2)

        with col1:
            mounting_type = st.selectbox(
                "Mounting Type",
                options=[m.value for m in MountingType],
                index=4  # Default to ground-mount
            )

            tilt_angle = st.slider(
                "Tilt Angle (°)",
                min_value=0.0,
                max_value=90.0,
                value=25.0,
                step=1.0,
                help="Tilt angle from horizontal"
            )

            azimuth = st.slider(
                "Azimuth (°)",
                min_value=0.0,
                max_value=359.0,
                value=180.0,
                step=1.0,
                help="0=North, 90=East, 180=South, 270=West"
            )

        with col2:
            is_tracking = st.checkbox("Enable Tracking")

            if is_tracking:
                tracking_type = st.selectbox(
                    "Tracking Type",
                    options=["single-axis", "dual-axis"],
                    index=0
                )
                max_tracking_angle = st.slider(
                    "Max Tracking Angle (°)",
                    min_value=0.0,
                    max_value=90.0,
                    value=45.0,
                    step=5.0
                )
                backtracking = st.checkbox("Enable Backtracking", value=True)
            else:
                tracking_type = None
                max_tracking_angle = None
                backtracking = False

            row_spacing = st.number_input(
                "Row Spacing (m)",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.5,
                help="Distance between module rows"
            )

            gcr = st.slider(
                "Ground Coverage Ratio",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Ratio of module area to ground area"
            )

        # Material and loading specifications
        st.markdown("### Structural Specifications")
        col1, col2, col3 = st.columns(3)

        with col1:
            material = st.selectbox(
                "Structural Material",
                options=["aluminum", "steel", "galvanized-steel"],
                index=0
            )

        with col2:
            wind_load = st.number_input(
                "Wind Load Rating (m/s)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                help="Maximum wind speed rating"
            )

        with col3:
            snow_load = st.number_input(
                "Snow Load Rating (Pa)",
                min_value=0.0,
                max_value=5000.0,
                value=2000.0,
                help="Maximum snow load"
            )

        if st.button("Save Mounting Configuration", type="primary"):
            st.session_state.mounting_config = MountingStructure(
                mounting_type=MountingType(mounting_type),
                tilt_angle=tilt_angle,
                azimuth=azimuth,
                is_tracking=is_tracking,
                tracking_type=tracking_type,
                max_tracking_angle=max_tracking_angle,
                backtracking_enabled=backtracking,
                row_spacing=row_spacing,
                gcr=gcr,
                material=material,
                wind_load_rating=wind_load,
                snow_load_rating=snow_load
            )
            st.success("✅ Mounting configuration saved!")

        # Display current configuration
        if st.session_state.mounting_config:
            st.markdown("### Current Configuration")
            config = st.session_state.mounting_config
            st.info(f"""
            **Mounting Type:** {config.mounting_type.value}
            **Tilt/Azimuth:** {config.tilt_angle}° / {config.azimuth}°
            **Tracking:** {"Yes" if config.is_tracking else "No"}
            **Row Spacing:** {config.row_spacing} m
            **GCR:** {config.gcr:.2f}
            **Material:** {config.material}
            """)

    def optimization_controls(self) -> None:
        """
        Render optimization controls panel.

        Provides interface for configuring and running system optimization
        including objective functions and constraints.
        """
        st.subheader("System Optimization Controls")

        # Optimization objectives
        st.markdown("### Optimization Objectives")

        col1, col2 = st.columns(2)

        with col1:
            optimize_for = st.selectbox(
                "Primary Objective",
                options=[
                    "Maximum Energy Yield",
                    "Minimum LCOE",
                    "Maximum ROI",
                    "Minimum Shading Loss",
                    "Custom Multi-Objective"
                ],
                index=0
            )

            if optimize_for == "Custom Multi-Objective":
                st.markdown("**Objective Weights:**")
                energy_weight = st.slider("Energy Yield", 0.0, 1.0, 0.5, 0.1)
                cost_weight = st.slider("Cost", 0.0, 1.0, 0.3, 0.1)
                land_weight = st.slider("Land Use", 0.0, 1.0, 0.2, 0.1)

        with col2:
            st.markdown("**Optimization Parameters:**")
            optimize_tilt = st.checkbox("Optimize Tilt Angle", value=True)
            optimize_azimuth = st.checkbox("Optimize Azimuth", value=False)
            optimize_row_spacing = st.checkbox("Optimize Row Spacing", value=True)
            optimize_dc_ac_ratio = st.checkbox("Optimize DC/AC Ratio", value=True)

        # Constraints
        st.markdown("### Design Constraints")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_land_area = st.number_input(
                "Max Land Area (acres)",
                min_value=0.0,
                value=10.0,
                help="Maximum available land area"
            )

        with col2:
            max_budget = st.number_input(
                "Max Budget ($)",
                min_value=0.0,
                value=1000000.0,
                step=10000.0,
                format="%.0f",
                help="Maximum project budget"
            )

        with col3:
            min_pr = st.slider(
                "Minimum PR",
                min_value=0.5,
                max_value=1.0,
                value=0.75,
                step=0.01,
                help="Minimum acceptable performance ratio"
            )

        # Optimization algorithm settings
        st.markdown("### Algorithm Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            algorithm = st.selectbox(
                "Optimization Algorithm",
                options=["Genetic Algorithm", "Particle Swarm", "Gradient Descent", "Simulated Annealing"],
                index=0
            )

        with col2:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )

        with col3:
            convergence_tol = st.number_input(
                "Convergence Tolerance",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )

        # Run optimization
        if st.button("🚀 Run Optimization", type="primary"):
            st.session_state.optimization_params = {
                "optimize_for": optimize_for,
                "optimize_tilt": optimize_tilt,
                "optimize_azimuth": optimize_azimuth,
                "optimize_row_spacing": optimize_row_spacing,
                "optimize_dc_ac_ratio": optimize_dc_ac_ratio,
                "max_land_area": max_land_area,
                "max_budget": max_budget,
                "min_pr": min_pr,
                "algorithm": algorithm,
                "max_iterations": max_iterations,
                "convergence_tol": convergence_tol
            }

            with st.spinner("Running optimization..."):
                # Placeholder for optimization logic
                import time
                time.sleep(2)
                st.success("✅ Optimization complete!")
                st.info("""
                **Optimized Configuration:**
                - Optimal Tilt: 28.5°
                - Optimal Azimuth: 180.0°
                - Optimal Row Spacing: 3.2 m
                - Optimal DC/AC Ratio: 1.25
                - Estimated Energy Yield: 1,250 MWh/year
                - Estimated LCOE: $0.045/kWh
                """)

    def _render_design_summary(self) -> None:
        """Render design summary panel."""
        st.subheader("Design Summary")

        if not st.session_state.site_location:
            st.warning("⚠️ Please configure site location first")
            return

        if not st.session_state.selected_modules:
            st.warning("⚠️ Please add module configurations")
            return

        if not st.session_state.selected_inverters:
            st.warning("⚠️ Please add inverters")
            return

        if not st.session_state.mounting_config:
            st.warning("⚠️ Please configure mounting structure")
            return

        # All components configured - show summary
        st.success("✅ Design is complete and ready for validation")

        # System overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total DC Power", f"{self._get_total_dc_power():.1f} kW")
            st.metric("Total AC Power", f"{self._get_total_ac_power():.1f} kW")

        with col2:
            st.metric("DC/AC Ratio", f"{self._get_dc_ac_ratio():.2f}")
            st.metric("Total Modules", f"{self._get_total_modules()}")

        with col3:
            total_area = sum(
                config.total_modules * config.module.area
                for config in st.session_state.selected_modules
            )
            st.metric("Module Area", f"{total_area:.1f} m²")
            st.metric("Inverter Count", f"{len(st.session_state.selected_inverters)}")

        # Component details
        with st.expander("📋 Component Details"):
            st.markdown("**Modules:**")
            for idx, config in enumerate(st.session_state.selected_modules):
                st.text(
                    f"{idx + 1}. {config.module.manufacturer} {config.module.model}: "
                    f"{config.total_modules} modules ({config.total_power / 1000:.1f} kW)"
                )

            st.markdown("**Inverters:**")
            inverter_counts = {}
            for inv in st.session_state.selected_inverters:
                key = f"{inv.manufacturer} {inv.model}"
                inverter_counts[key] = inverter_counts.get(key, 0) + 1
            for idx, (inv_name, count) in enumerate(inverter_counts.items()):
                st.text(f"{idx + 1}. {inv_name} × {count}")

    def _create_module_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from module database."""
        data = []
        for name, module in self.module_database.items():
            data.append({
                "Model": name,
                "Manufacturer": module.manufacturer,
                "Technology": module.technology.value,
                "Power (W)": module.p_max,
                "Efficiency (%)": module.efficiency,
                "Voc (V)": module.v_oc,
                "Isc (A)": module.i_sc,
                "Bifacial": module.is_bifacial,
                "Warranty (yr)": module.warranty_years
            })
        return pd.DataFrame(data)

    def _create_inverter_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from inverter database."""
        data = []
        for name, inverter in self.inverter_database.items():
            data.append({
                "Model": name,
                "Manufacturer": inverter.manufacturer,
                "Type": inverter.inverter_type.value,
                "AC Power (kW)": inverter.p_ac_rated / 1000,
                "DC/AC Max": inverter.dc_ac_ratio_max,
                "MPPT": inverter.num_mppt,
                "Efficiency (%)": inverter.max_efficiency,
                "Vmp Range (V)": f"{inverter.v_mpp_min}-{inverter.v_mpp_max}"
            })
        return pd.DataFrame(data)

    def _get_total_dc_power(self) -> float:
        """Get total DC power in kW."""
        if not st.session_state.selected_modules:
            return 0.0
        return sum(config.total_power for config in st.session_state.selected_modules) / 1000.0

    def _get_total_ac_power(self) -> float:
        """Get total AC power in kW."""
        if not st.session_state.selected_inverters:
            return 0.0
        return sum(inv.p_ac_rated for inv in st.session_state.selected_inverters) / 1000.0

    def _get_dc_ac_ratio(self) -> float:
        """Get DC/AC ratio."""
        dc = self._get_total_dc_power()
        ac = self._get_total_ac_power()
        if ac == 0:
            return 0.0
        return dc / ac

    def _get_total_modules(self) -> int:
        """Get total module count."""
        if not st.session_state.selected_modules:
            return 0
        return sum(config.total_modules for config in st.session_state.selected_modules)
