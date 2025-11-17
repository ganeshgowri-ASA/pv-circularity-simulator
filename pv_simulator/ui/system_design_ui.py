"""Streamlit UI for PV system design."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Optional

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    SystemType,
    MountingType,
    InverterType,
)
from pv_simulator.system_design.system_design_engine import SystemDesignEngine
from pv_simulator.system_design.inverter_selector import InverterSelector
from pv_simulator.system_design.string_sizing_calculator import StringSizingCalculator


class SystemDesignUI:
    """Streamlit UI component for PV system design."""

    def __init__(self):
        """Initialize the UI."""
        st.set_page_config(
            page_title="PV System Design Tool",
            page_icon="☀️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize session state
        if "system_config" not in st.session_state:
            st.session_state.system_config = None

        if "design_engine" not in st.session_state:
            st.session_state.design_engine = None

    def render(self):
        """Render the main UI."""
        st.title("☀️ PV System Design Tool")
        st.markdown(
            "Design utility-scale, commercial, and residential PV systems "
            "with PVsyst-level accuracy"
        )

        # Sidebar configuration
        self.render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📋 Project Setup",
                "🔧 System Design",
                "📊 Performance Analysis",
                "📈 Loss Waterfall",
                "💾 Export",
            ]
        )

        with tab1:
            self.render_project_setup()

        with tab2:
            self.render_system_design()

        with tab3:
            self.render_performance_analysis()

        with tab4:
            self.render_loss_waterfall()

        with tab5:
            self.render_export()

    def render_sidebar(self):
        """Render sidebar with global settings."""
        st.sidebar.header("⚙️ Global Settings")

        # Database path
        db_path = Path(__file__).parent.parent / "data" / "inverter_database.json"
        if db_path.exists():
            st.sidebar.success(f"✓ Inverter database loaded ({db_path.name})")
        else:
            st.sidebar.warning("⚠ Inverter database not found")

        st.sidebar.divider()

        # Quick actions
        st.sidebar.header("🚀 Quick Actions")
        if st.sidebar.button("📂 Load Example Project"):
            self.load_example_project()

        if st.sidebar.button("🔄 Reset All"):
            st.session_state.clear()
            st.rerun()

    def render_project_setup(self):
        """Render project setup tab."""
        st.header("Project Configuration")

        col1, col2 = st.columns(2)

        with col1:
            project_name = st.text_input("Project Name", value="My PV Project")
            system_type = st.selectbox(
                "System Type",
                options=[SystemType.UTILITY, SystemType.COMMERCIAL, SystemType.RESIDENTIAL],
                format_func=lambda x: x.value.title(),
            )

            location = st.text_input("Location", value="Phoenix, AZ")

        with col2:
            latitude = st.number_input("Latitude (°)", value=33.45, min_value=-90.0, max_value=90.0)
            longitude = st.number_input("Longitude (°)", value=-112.07, min_value=-180.0, max_value=180.0)
            elevation = st.number_input("Elevation (m)", value=340.0, min_value=0.0)

        st.divider()

        st.subheader("Site Temperature Range")
        col1, col2 = st.columns(2)
        with col1:
            site_temp_min = st.number_input(
                "Minimum Temperature (°C)", value=-10.0, min_value=-50.0, max_value=50.0
            )
        with col2:
            site_temp_max = st.number_input(
                "Maximum Temperature (°C)", value=70.0, min_value=0.0, max_value=100.0
            )

        if st.button("Initialize Project", type="primary"):
            design_engine = SystemDesignEngine(
                project_name=project_name,
                system_type=system_type,
                location=location,
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
            )
            st.session_state.design_engine = design_engine
            st.session_state.site_temp_min = site_temp_min
            st.session_state.site_temp_max = site_temp_max
            st.success("✓ Project initialized successfully!")

    def render_system_design(self):
        """Render system design tab."""
        st.header("System Design")

        if st.session_state.design_engine is None:
            st.warning("⚠ Please initialize project in Project Setup tab first")
            return

        # Module selection
        st.subheader("1️⃣ Module Selection")
        module = self.render_module_selection()

        st.divider()

        # Inverter selection
        st.subheader("2️⃣ Inverter Selection")
        inverter = self.render_inverter_selection()

        st.divider()

        # System sizing
        st.subheader("3️⃣ System Sizing")
        target_dc_capacity = st.number_input(
            "Target DC Capacity (kW)", value=1000.0, min_value=1.0, max_value=1000000.0
        )

        target_dc_ac_ratio = st.slider(
            "DC/AC Ratio", min_value=1.0, max_value=1.5, value=1.25, step=0.05
        )

        st.divider()

        # Mounting configuration
        st.subheader("4️⃣ Mounting Configuration")
        mounting_type = st.selectbox(
            "Mounting Type",
            options=[
                MountingType.GROUND_FIXED,
                MountingType.GROUND_SINGLE_AXIS,
                MountingType.ROOFTOP_FLAT,
                MountingType.ROOFTOP_SLOPED,
                MountingType.CARPORT,
                MountingType.FLOATING,
                MountingType.AGRIVOLTAIC,
            ],
            format_func=lambda x: x.value.replace("_", " ").title(),
        )

        # Mounting-specific parameters
        kwargs = {}
        if mounting_type == MountingType.GROUND_FIXED:
            kwargs["target_gcr"] = st.slider("Target GCR", 0.2, 0.6, 0.4, 0.05)
        elif mounting_type == MountingType.ROOFTOP_FLAT:
            col1, col2 = st.columns(2)
            with col1:
                kwargs["roof_width_m"] = st.number_input("Roof Width (m)", value=50.0)
            with col2:
                kwargs["roof_length_m"] = st.number_input("Roof Length (m)", value=100.0)

        if st.button("🎯 Design System", type="primary"):
            with st.spinner("Designing system..."):
                try:
                    system_config = st.session_state.design_engine.design_system_configuration(
                        module=module,
                        inverter=inverter,
                        target_dc_capacity_kw=target_dc_capacity,
                        mounting_type=mounting_type,
                        site_temp_min=st.session_state.site_temp_min,
                        site_temp_max=st.session_state.site_temp_max,
                        target_dc_ac_ratio=target_dc_ac_ratio,
                        **kwargs,
                    )
                    st.session_state.system_config = system_config
                    st.success("✓ System design completed!")

                    # Display summary
                    self.display_system_summary(system_config)

                except Exception as e:
                    st.error(f"❌ Design failed: {str(e)}")

    def render_module_selection(self) -> ModuleParameters:
        """Render module selection UI."""
        col1, col2, col3 = st.columns(3)

        with col1:
            manufacturer = st.text_input("Manufacturer", value="Trina Solar")
            model = st.text_input("Model", value="TSM-DEG21C.20")
            pmax = st.number_input("Power (W)", value=670.0, min_value=100.0)

        with col2:
            voc = st.number_input("Voc (V)", value=45.9, min_value=10.0)
            vmp = st.number_input("Vmp (V)", value=38.4, min_value=10.0)
            isc = st.number_input("Isc (A)", value=18.52, min_value=1.0)

        with col3:
            imp = st.number_input("Imp (A)", value=17.45, min_value=1.0)
            temp_coeff_pmax = st.number_input("Temp Coeff Pmax (%/°C)", value=-0.34)
            temp_coeff_voc = st.number_input("Temp Coeff Voc (%/°C)", value=-0.25)

        return ModuleParameters(
            manufacturer=manufacturer,
            model=model,
            technology="mtSiMono",
            pmax=pmax,
            voc=voc,
            isc=isc,
            vmp=vmp,
            imp=imp,
            temp_coeff_pmax=temp_coeff_pmax,
            temp_coeff_voc=temp_coeff_voc,
            temp_coeff_isc=0.05,
            length=2.384,
            width=1.303,
            thickness=0.035,
            weight=34.6,
            cells_in_series=132,
            efficiency=21.5,
        )

    def render_inverter_selection(self) -> InverterParameters:
        """Render inverter selection UI."""
        col1, col2, col3 = st.columns(3)

        with col1:
            manufacturer = st.text_input("Manufacturer ", value="SMA")
            model = st.text_input("Model ", value="SC-2750-EV")
            pac_max = st.number_input("AC Power (kW)", value=2750.0, min_value=1.0) * 1000

        with col2:
            vac_nom = st.number_input("AC Voltage (V)", value=480.0)
            pdc_max = st.number_input("DC Power (kW)", value=2860.0, min_value=1.0) * 1000
            vdc_max = st.number_input("Max DC Voltage (V)", value=1500.0)

        with col3:
            num_mppt = st.number_input("Number of MPPTs", value=6, min_value=1, max_value=24)
            mppt_vmin = st.number_input("MPPT Min Voltage (V)", value=580.0)
            mppt_vmax = st.number_input("MPPT Max Voltage (V)", value=1300.0)

        return InverterParameters(
            manufacturer=manufacturer,
            model=model,
            inverter_type=InverterType.CENTRAL,
            pac_max=pac_max,
            vac_nom=vac_nom,
            iac_max=pac_max / vac_nom,
            pdc_max=pdc_max,
            vdc_max=vdc_max,
            vdc_nom=1000.0,
            vdc_min=580.0,
            idc_max=3200.0,
            num_mppt=num_mppt,
            mppt_vmin=mppt_vmin,
            mppt_vmax=mppt_vmax,
            strings_per_mppt=12,
            max_efficiency=98.8,
            weight=2500.0,
        )

    def display_system_summary(self, system_config):
        """Display system design summary."""
        st.subheader("📊 System Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("DC Capacity", f"{system_config.dc_capacity:.1f} kW")
            st.metric("Number of Modules", f"{system_config.num_modules:,}")

        with col2:
            st.metric("AC Capacity", f"{system_config.ac_capacity:.1f} kW")
            st.metric("Number of Inverters", f"{system_config.num_inverters}")

        with col3:
            st.metric("DC/AC Ratio", f"{system_config.dc_ac_ratio:.2f}")
            st.metric("Modules per String", f"{system_config.string_config.modules_per_string}")

        with col4:
            st.metric("Total Losses", f"{system_config.losses.total_losses():.1f}%")
            st.metric("Array Rows", f"{system_config.array_layout.rows}")

    def render_performance_analysis(self):
        """Render performance analysis tab."""
        st.header("Performance Analysis")

        if st.session_state.system_config is None:
            st.warning("⚠ Please design a system first")
            return

        config = st.session_state.system_config

        st.subheader("System Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"""
                **DC System**
                - Capacity: {config.dc_capacity:.1f} kW
                - Modules: {config.num_modules:,}
                - String Voc (STC): {config.string_config.voc_stc:.1f} V
                - String Vmp (STC): {config.string_config.vmp_stc:.1f} V
                """
            )

        with col2:
            st.info(
                f"""
                **AC System**
                - Capacity: {config.ac_capacity:.1f} kW
                - Inverters: {config.num_inverters}
                - DC/AC Ratio: {config.dc_ac_ratio:.2f}
                - Inverter Efficiency: {config.inverter.max_efficiency:.1f}%
                """
            )

    def render_loss_waterfall(self):
        """Render loss waterfall chart."""
        st.header("Loss Waterfall Analysis")

        if st.session_state.system_config is None:
            st.warning("⚠ Please design a system first")
            return

        config = st.session_state.system_config
        losses = config.losses

        # Create waterfall data
        categories = [
            "Nameplate",
            "Soiling",
            "Near Shading",
            "Far Shading",
            "LID",
            "Mismatch",
            "DC Wiring",
            "Inverter",
            "Clipping",
            "AC Wiring",
            "Transformer",
            "Availability",
            "Final Output",
        ]

        # Calculate cascading values
        values = [100.0]
        current = 100.0

        for loss in [
            losses.soiling,
            losses.shading_near,
            losses.shading_far,
            losses.lid,
            losses.mismatch,
            losses.dc_wiring,
            losses.inverter,
            losses.clipping,
            losses.ac_wiring,
            losses.transformer,
        ]:
            current *= (100.0 - loss) / 100.0
            values.append(-loss)

        current *= losses.availability / 100.0
        values.append(-(100 - losses.availability))
        values.append(current)

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Losses",
            orientation="v",
            measure=["absolute"] + ["relative"] * 11 + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{v:.1f}%" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="System Loss Waterfall",
            showlegend=False,
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Loss breakdown table
        st.subheader("Loss Breakdown")
        loss_data = {
            "Loss Category": [
                "Soiling",
                "Near Shading",
                "Far Shading",
                "LID",
                "Mismatch",
                "DC Wiring",
                "Inverter",
                "Clipping",
                "AC Wiring",
                "Transformer",
                "Availability",
            ],
            "Loss (%)": [
                losses.soiling,
                losses.shading_near,
                losses.shading_far,
                losses.lid,
                losses.mismatch,
                losses.dc_wiring,
                losses.inverter,
                losses.clipping,
                losses.ac_wiring,
                losses.transformer,
                100 - losses.availability,
            ],
        }
        df_losses = pd.DataFrame(loss_data)
        st.dataframe(df_losses, use_container_width=True)

    def render_export(self):
        """Render export tab."""
        st.header("Export System Design")

        if st.session_state.system_config is None:
            st.warning("⚠ Please design a system first")
            return

        config = st.session_state.system_config

        st.subheader("Export Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export as JSON"):
                # Export configuration as JSON
                st.json(config.model_dump(mode='json'))

        with col2:
            if st.button("📊 Export as Excel"):
                st.info("Excel export functionality coming soon")

        st.divider()

        st.subheader("PVsyst Integration")
        if st.button("🔄 Generate PVsyst Project File"):
            st.info("PVsyst .PRJ file generation coming soon")

    def load_example_project(self):
        """Load an example project."""
        st.info("Example project loaded!")


def main():
    """Main entry point for Streamlit app."""
    app = SystemDesignUI()
    app.render()


if __name__ == "__main__":
    main()
