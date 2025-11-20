"""Streamlit UI for PV Mounting Structure Design."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pv_simulator.mounting_structure.models import (
    SiteParameters,
    ModuleDimensions,
    GroundMountConfig,
    RooftopMountConfig,
    CarportConfig,
    FloatingPVConfig,
    AgrivoltaicConfig,
    BIPVConfig,
    MountingType,
    ModuleOrientation,
    RackingConfiguration,
    FoundationType,
    ExposureCategory,
    SeismicDesignCategory,
    SoilType,
    MaterialType,
)
from pv_simulator.mounting_structure.ground_mount import GroundMountDesign
from pv_simulator.mounting_structure.rooftop_mount import RooftopMountDesign
from pv_simulator.mounting_structure.carport_canopy import CarportCanopyDesign
from pv_simulator.mounting_structure.floating_pv import FloatingPVDesign
from pv_simulator.mounting_structure.agrivoltaic import AgrivoltaicDesign
from pv_simulator.mounting_structure.bipv import BIPVDesign


class MountingStructureUI:
    """Streamlit UI for mounting structure design and analysis."""

    def __init__(self):
        """Initialize the UI."""
        st.set_page_config(
            page_title="PV Mounting Structure Design",
            page_icon="🔧",
            layout="wide",
        )

    def run(self):
        """Run the Streamlit application."""
        st.title("🔧 PV Mounting Structure Design & Engineering")
        st.markdown("""
        Comprehensive mounting structure design for all PV mounting types with ASCE 7 structural calculations.
        """)

        # Sidebar for mounting type selection
        with st.sidebar:
            st.header("Configuration")

            mounting_type = st.selectbox(
                "Mounting Type",
                options=[
                    "Ground Mount - Fixed Tilt",
                    "Ground Mount - Single-Axis Tracker",
                    "Ground Mount - Dual-Axis Tracker",
                    "Rooftop - Flat",
                    "Rooftop - Pitched",
                    "Carport",
                    "Floating PV",
                    "Agrivoltaic",
                    "BIPV - Facade",
                    "BIPV - Skylight",
                ],
            )

        # Site parameters (common to all)
        site_params = self._get_site_parameters()

        # Module dimensions (common to all)
        module_dims = self._get_module_dimensions()

        # Design based on selected mounting type
        if "Ground Mount" in mounting_type:
            if "Fixed Tilt" in mounting_type:
                self._ground_mount_fixed_tilt(site_params, module_dims)
            elif "Single-Axis" in mounting_type:
                self._ground_mount_single_axis(site_params, module_dims)
            else:  # Dual-Axis
                self._ground_mount_dual_axis(site_params, module_dims)

        elif "Rooftop" in mounting_type:
            if "Flat" in mounting_type:
                self._rooftop_flat(site_params, module_dims)
            else:  # Pitched
                self._rooftop_pitched(site_params, module_dims)

        elif "Carport" in mounting_type:
            self._carport_design(site_params, module_dims)

        elif "Floating" in mounting_type:
            self._floating_pv(site_params, module_dims)

        elif "Agrivoltaic" in mounting_type:
            self._agrivoltaic(site_params, module_dims)

        else:  # BIPV
            if "Facade" in mounting_type:
                self._bipv_facade(site_params, module_dims)
            else:  # Skylight
                self._bipv_skylight(site_params, module_dims)

    def _get_site_parameters(self) -> SiteParameters:
        """Get site parameters from user input."""
        with st.expander("📍 Site Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                latitude = st.number_input("Latitude (°)", value=40.0, min_value=-90.0, max_value=90.0)
                longitude = st.number_input("Longitude (°)", value=-105.0, min_value=-180.0, max_value=180.0)
                elevation = st.number_input("Elevation (m)", value=1000.0, min_value=0.0)

            with col2:
                wind_speed = st.number_input("Design Wind Speed (m/s)", value=35.0, min_value=0.0)
                exposure = st.selectbox("Exposure Category", options=["B", "C", "D"])
                ground_snow_load = st.number_input("Ground Snow Load (kN/m²)", value=1.0, min_value=0.0)

            with col3:
                seismic = st.selectbox("Seismic Design Category", options=["A", "B", "C", "D", "E", "F"])
                soil_type = st.selectbox("Soil Type", options=["clay", "sand", "silt", "gravel", "rock", "mixed"])
                frost_depth = st.number_input("Frost Depth (m)", value=0.8, min_value=0.0)

        return SiteParameters(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            wind_speed=wind_speed,
            exposure_category=ExposureCategory(exposure),
            ground_snow_load=ground_snow_load,
            seismic_category=SeismicDesignCategory(seismic),
            soil_type=SoilType(soil_type),
            frost_depth=frost_depth,
        )

    def _get_module_dimensions(self) -> ModuleDimensions:
        """Get module dimensions from user input."""
        with st.expander("📐 Module Dimensions", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                length = st.number_input("Module Length (m)", value=2.0, min_value=0.1)
                width = st.number_input("Module Width (m)", value=1.0, min_value=0.1)
                thickness = st.number_input("Module Thickness (m)", value=0.04, min_value=0.01)

            with col2:
                weight = st.number_input("Module Weight (kg)", value=25.0, min_value=1.0)
                frame_width = st.number_input("Frame Width (m)", value=0.035, min_value=0.01)
                glass_thickness = st.number_input("Glass Thickness (m)", value=0.0032, min_value=0.001)

        return ModuleDimensions(
            length=length,
            width=width,
            thickness=thickness,
            weight=weight,
            frame_width=frame_width,
            glass_thickness=glass_thickness,
        )

    def _ground_mount_fixed_tilt(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design fixed-tilt ground mount system."""
        st.header("Ground Mount - Fixed Tilt Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=1000, min_value=1)
            tilt_angle = st.slider("Tilt Angle (°)", 0, 60, int(abs(site_params.latitude)))
            azimuth = st.number_input("Azimuth (°)", value=180.0, min_value=0.0, max_value=360.0)

        with col2:
            orientation = st.selectbox("Module Orientation", options=["portrait", "landscape"])
            racking_config = st.selectbox("Racking Configuration", options=["1P", "2P", "3P", "4P"])
            foundation_type = st.selectbox("Foundation Type", options=["driven_pile", "helical_pile", "ballasted"])
            post_spacing = st.number_input("Post Spacing (m)", value=3.0, min_value=1.0)

        # Calculate button
        if st.button("🔍 Design Fixed-Tilt System", type="primary"):
            with st.spinner("Calculating structural design..."):
                config = GroundMountConfig(
                    mounting_type=MountingType.GROUND_FIXED_TILT,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    azimuth=azimuth,
                    orientation=ModuleOrientation(orientation),
                    racking_config=RackingConfiguration(racking_config),
                    foundation_type=FoundationType(foundation_type),
                    post_spacing=post_spacing,
                )

                designer = GroundMountDesign(config)
                result = designer.fixed_tilt_structure()
                row_spacing = designer.calculate_row_spacing()

                self._display_results(result, row_spacing)

    def _ground_mount_single_axis(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design single-axis tracker system."""
        st.header("Ground Mount - Single-Axis Tracker Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=5000, min_value=1)
            max_tracking_angle = st.slider("Max Tracking Angle (°)", 30, 70, 60)
            backtracking = st.checkbox("Enable Backtracking", value=True)

        with col2:
            foundation_type = st.selectbox("Foundation Type", options=["driven_pile", "helical_pile"])
            post_spacing = st.number_input("Post Spacing (m)", value=15.0, min_value=5.0)

        if st.button("🔍 Design Single-Axis Tracker", type="primary"):
            with st.spinner("Calculating tracker design..."):
                config = GroundMountConfig(
                    mounting_type=MountingType.GROUND_SINGLE_AXIS,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=0,  # Horizontal at rest
                    orientation=ModuleOrientation.PORTRAIT,
                    racking_config=RackingConfiguration.ONE_PORTRAIT,
                    foundation_type=FoundationType(foundation_type),
                    post_spacing=post_spacing,
                    max_tracking_angle=max_tracking_angle,
                    backtracking_enabled=backtracking,
                )

                designer = GroundMountDesign(config)
                result = designer.single_axis_tracker()

                self._display_results(result, {"backtracking": backtracking})

    def _ground_mount_dual_axis(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design dual-axis tracker system."""
        st.header("Ground Mount - Dual-Axis Tracker Design")

        num_modules = st.number_input("Number of Modules", value=1000, min_value=1)

        if st.button("🔍 Design Dual-Axis Tracker", type="primary"):
            with st.spinner("Calculating dual-axis tracker design..."):
                config = GroundMountConfig(
                    mounting_type=MountingType.GROUND_DUAL_AXIS,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=0,
                    orientation=ModuleOrientation.PORTRAIT,
                    racking_config=RackingConfiguration.ONE_PORTRAIT,
                    foundation_type=FoundationType.CONCRETE_PAD,
                )

                designer = GroundMountDesign(config)
                result = designer.dual_axis_tracker()

                self._display_results(result, {})

    def _rooftop_flat(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design flat roof system."""
        st.header("Rooftop - Flat Roof Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=500, min_value=1)
            tilt_angle = st.slider("Tilt Angle (°)", 5, 20, 10)
            ballasted = st.checkbox("Ballasted System", value=True)

        with col2:
            roof_dead_capacity = st.number_input("Roof Dead Load Capacity (kN/m²)", value=2.0, min_value=0.1)
            roof_live_capacity = st.number_input("Roof Live Load Capacity (kN/m²)", value=1.5, min_value=0.1)

        if st.button("🔍 Design Flat Roof System", type="primary"):
            with st.spinner("Calculating rooftop design..."):
                config = RooftopMountConfig(
                    mounting_type=MountingType.ROOFTOP_FLAT,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    roof_type="flat",
                    roof_material="membrane",
                    attachment_type="ballasted" if ballasted else "attached",
                    roof_dead_load_capacity=roof_dead_capacity,
                    roof_live_load_capacity=roof_live_capacity,
                )

                designer = RooftopMountDesign(config)
                result = designer.flat_roof_design(ballasted=ballasted)

                self._display_results(result, {})

    def _rooftop_pitched(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design pitched roof system."""
        st.header("Rooftop - Pitched Roof Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=200, min_value=1)
            roof_pitch = st.slider("Roof Pitch (°)", 10, 45, 25)

        with col2:
            roof_material = st.selectbox("Roof Material", options=["asphalt_shingle", "metal", "tile"])
            roof_dead_capacity = st.number_input("Roof Dead Load Capacity (kN/m²)", value=1.5, min_value=0.1)
            roof_live_capacity = st.number_input("Roof Live Load Capacity (kN/m²)", value=1.0, min_value=0.1)

        if st.button("🔍 Design Pitched Roof System", type="primary"):
            with st.spinner("Calculating pitched roof design..."):
                config = RooftopMountConfig(
                    mounting_type=MountingType.ROOFTOP_PITCHED,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=0,  # Flush mount
                    roof_type="pitched",
                    roof_pitch=roof_pitch,
                    roof_material=roof_material,
                    attachment_type="L-foot",
                    roof_dead_load_capacity=roof_dead_capacity,
                    roof_live_load_capacity=roof_live_capacity,
                )

                designer = RooftopMountDesign(config)
                result = designer.pitched_roof_design()

                self._display_results(result, {})

    def _carport_design(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design carport system."""
        st.header("Carport/Canopy Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=300, min_value=1)
            carport_type = st.selectbox("Carport Type", options=["single_cantilever", "double_cantilever", "four_post"])
            span_length = st.number_input("Span Length (m)", value=6.0, min_value=3.0)

        with col2:
            clearance_height = st.number_input("Clearance Height (m)", value=2.5, min_value=2.0)
            column_spacing = st.number_input("Column Spacing (m)", value=5.0, min_value=3.0)
            tilt_angle = st.slider("Tilt Angle (°)", 0, 15, 5)

        if st.button("🔍 Design Carport System", type="primary"):
            with st.spinner("Calculating carport design..."):
                config = CarportConfig(
                    mounting_type=MountingType.CARPORT,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    carport_type=carport_type,
                    span_length=span_length,
                    clearance_height=clearance_height,
                    column_spacing=column_spacing,
                )

                designer = CarportCanopyDesign(config)

                if carport_type == "single_cantilever":
                    result = designer.single_cantilever_carport()
                elif carport_type == "double_cantilever":
                    result = designer.double_cantilever_carport()
                else:
                    result = designer.four_post_canopy()

                self._display_results(result, {})

    def _floating_pv(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design floating PV system."""
        st.header("Floating PV Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=2000, min_value=1)
            water_depth = st.number_input("Water Depth (m)", value=5.0, min_value=1.0)
            max_wave_height = st.number_input("Max Wave Height (m)", value=0.5, min_value=0.0)

        with col2:
            coverage_ratio = st.slider("Water Coverage Ratio", 0.05, 0.40, 0.15)
            tilt_angle = st.slider("Tilt Angle (°)", 5, 15, 10)

        if st.button("🔍 Design Floating PV System", type="primary"):
            with st.spinner("Calculating floating PV design..."):
                config = FloatingPVConfig(
                    mounting_type=MountingType.FLOATING,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    water_body_type="reservoir",
                    water_depth=water_depth,
                    max_wave_height=max_wave_height,
                    water_level_variation=1.0,
                    coverage_ratio=coverage_ratio,
                    anchoring_type="pile",
                )

                designer = FloatingPVDesign(config)
                result = designer.pontoon_layout()

                self._display_results(result, {})

    def _agrivoltaic(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design agrivoltaic system."""
        st.header("Agrivoltaic Design")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=1500, min_value=1)
            crop_type = st.selectbox("Crop Type", options=["wheat", "corn", "soybeans", "vegetables", "vineyard", "grazing"])
            clearance_height = st.number_input("Clearance Height (m)", value=4.0, min_value=2.0)

        with col2:
            row_spacing = st.number_input("Row Spacing (m)", value=12.0, min_value=5.0)
            tilt_angle = st.slider("Tilt Angle (°)", 10, 35, 20)
            bifacial = st.checkbox("Bifacial Modules", value=True)

        if st.button("🔍 Design Agrivoltaic System", type="primary"):
            with st.spinner("Calculating agrivoltaic design..."):
                config = AgrivoltaicConfig(
                    mounting_type=MountingType.AGRIVOLTAIC,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    clearance_height=clearance_height,
                    crop_type=crop_type,
                    row_spacing_for_crops=row_spacing,
                    bifacial_modules=bifacial,
                )

                designer = AgrivoltaicDesign(config)
                result = designer.high_clearance_structure()

                self._display_results(result, {})

    def _bipv_facade(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design BIPV facade system."""
        st.header("BIPV - Facade Integration")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=500, min_value=1)
            building_height = st.number_input("Building Height (m)", value=30.0, min_value=5.0)
            vertical = st.checkbox("Vertical Installation", value=True)

        with col2:
            structural_glazing = st.checkbox("Structural Glazing", value=True)
            thermal_break = st.checkbox("Thermal Break", value=True)

        if st.button("🔍 Design BIPV Facade", type="primary"):
            with st.spinner("Calculating BIPV facade design..."):
                config = BIPVConfig(
                    mounting_type=MountingType.BIPV_FACADE,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=0 if vertical else 15,
                    integration_type="facade",
                    building_height=building_height,
                    vertical_installation=vertical,
                    structural_glazing=structural_glazing,
                    thermal_break=thermal_break,
                    junction_box_location="back",
                    conduit_routing="behind_facade",
                )

                designer = BIPVDesign(config)
                result = designer.facade_integration()

                self._display_results(result, {})

    def _bipv_skylight(self, site_params: SiteParameters, module_dims: ModuleDimensions):
        """Design BIPV skylight system."""
        st.header("BIPV - Skylight/Canopy")

        col1, col2 = st.columns(2)

        with col1:
            num_modules = st.number_input("Number of Modules", value=100, min_value=1)
            tilt_angle = st.slider("Tilt Angle (°)", 0, 15, 5)
            translucent = st.checkbox("Translucent Modules", value=True)

        with col2:
            building_height = st.number_input("Building Height (m)", value=10.0, min_value=3.0)

        if st.button("🔍 Design BIPV Skylight", type="primary"):
            with st.spinner("Calculating BIPV skylight design..."):
                config = BIPVConfig(
                    mounting_type=MountingType.BIPV_SKYLIGHT,
                    site_parameters=site_params,
                    module_dimensions=module_dims,
                    num_modules=num_modules,
                    tilt_angle=tilt_angle,
                    integration_type="skylight",
                    building_height=building_height,
                    translucent_modules=translucent,
                    structural_glazing=True,
                    junction_box_location="edge",
                    conduit_routing="above_ceiling",
                )

                designer = BIPVDesign(config)
                result = designer.skylight_canopy()

                self._display_results(result, {})

    def _display_results(self, result: Any, additional_info: Dict[str, Any] = None):
        """Display design results."""
        st.success("✅ Design Complete!")

        # Tabs for different result sections
        tabs = st.tabs(["📊 Load Analysis", "🏗️ Structural Design", "💰 Bill of Materials", "📈 Visualizations"])

        with tabs[0]:
            self._display_load_analysis(result.load_analysis)

        with tabs[1]:
            self._display_structural_design(result)

        with tabs[2]:
            self._display_bom(result.bill_of_materials)

        with tabs[3]:
            self._display_visualizations(result)

    def _display_load_analysis(self, load_analysis: Any):
        """Display load analysis results."""
        st.subheader("Load Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dead Load", f"{load_analysis.dead_load:.2f} kN/m²")
            st.metric("Live Load", f"{load_analysis.live_load:.2f} kN/m²")

        with col2:
            st.metric("Wind Uplift", f"{load_analysis.wind_load_uplift:.2f} kN/m²")
            st.metric("Wind Downward", f"{load_analysis.wind_load_downward:.2f} kN/m²")

        with col3:
            st.metric("Snow Load", f"{load_analysis.snow_load:.2f} kN/m²")
            st.metric("Total Load Combination", f"{load_analysis.total_load_combination:.2f} kN/m²")

    def _display_structural_design(self, result: Any):
        """Display structural design results."""
        st.subheader("Structural Design")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Foundation:**")
            st.write(f"Type: {result.foundation_design.foundation_type.value}")
            st.write(f"Depth: {result.foundation_design.depth:.2f} m")
            st.write(f"Capacity: {result.foundation_design.capacity:.2f} kN")
            st.write(f"Quantity: {result.foundation_design.quantity}")

        with col2:
            st.write("**Deflection:**")
            st.write(f"Max Deflection: {result.max_deflection*1000:.1f} mm")
            st.write(f"Deflection Limit: {result.deflection_limit*1000:.1f} mm")
            st.write(f"Status: {'✅ PASS' if result.max_deflection <= result.deflection_limit else '❌ FAIL'}")

        st.write("**Structural Members:**")
        members_df = pd.DataFrame([
            {
                "Type": m.member_type,
                "Material": m.material.value,
                "Profile": m.profile,
                "Length (m)": m.length,
                "Quantity": m.quantity,
                "Utilization": f"{m.utilization:.1%}",
            }
            for m in result.structural_members
        ])
        st.dataframe(members_df, use_container_width=True)

        st.write("**Compliance Notes:**")
        for note in result.compliance_notes:
            st.info(note)

    def _display_bom(self, bom: list):
        """Display bill of materials."""
        st.subheader("Bill of Materials")

        bom_df = pd.DataFrame([
            {
                "Item": item.item_number,
                "Description": item.description,
                "Material": item.material.value,
                "Quantity": item.quantity,
                "Unit": item.unit,
                "Unit Cost ($)": f"${item.unit_cost:.2f}" if item.unit_cost else "-",
                "Total Cost ($)": f"${item.total_cost:.2f}" if item.total_cost else "-",
                "Weight (kg)": f"{item.total_weight:.1f}" if item.total_weight else "-",
            }
            for item in bom
        ])

        st.dataframe(bom_df, use_container_width=True)

        # Summary
        total_cost = sum(item.total_cost or 0 for item in bom)
        total_weight = sum(item.total_weight or 0 for item in bom)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col2:
            st.metric("Total Weight", f"{total_weight:,.0f} kg")

    def _display_visualizations(self, result: Any):
        """Display visualizations."""
        st.subheader("Design Visualizations")

        # Load distribution pie chart
        fig_loads = go.Figure(data=[go.Pie(
            labels=["Dead Load", "Live Load", "Wind Load", "Snow Load"],
            values=[
                result.load_analysis.dead_load,
                result.load_analysis.live_load,
                abs(result.load_analysis.wind_load_uplift),
                result.load_analysis.snow_load,
            ],
            hole=0.3,
        )])
        fig_loads.update_layout(title="Load Distribution")
        st.plotly_chart(fig_loads, use_container_width=True)

        # Cost breakdown
        cost_data = pd.DataFrame([
            {"Category": item.description[:20], "Cost": item.total_cost or 0}
            for item in result.bill_of_materials
        ]).groupby("Category").sum().reset_index()

        fig_cost = px.bar(cost_data, x="Category", y="Cost", title="Cost Breakdown by Component")
        st.plotly_chart(fig_cost, use_container_width=True)


def main():
    """Main entry point."""
    app = MountingStructureUI()
    app.run()


if __name__ == "__main__":
    main()
