"""
Streamlit UI for Helioscope Shade Analysis.

This module provides an interactive web interface for 3D shade analysis,
visualization, and reporting.
"""

import logging
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure

from .electrical_shading import ElectricalShadingModel
from .helioscope_model import HelioscapeModel
from .horizon_profiler import HorizonProfiler
from .irradiance import IrradianceOnSurface
from .layout_optimizer import SystemLayoutOptimizer
from .models import (
    ArrayGeometry,
    HorizonProfile,
    IrradianceComponents,
    Location,
    ModuleElectricalParams,
    ShadeAnalysisConfig,
    SiteModel,
    TrackerType,
    TranspositionModel,
    AOIModel,
)
from .shade_analysis import ShadeAnalysisEngine
from .sun_position import SunPositionCalculator

logger = logging.getLogger(__name__)


class ShadeAnalysisUI:
    """
    Streamlit-based interactive UI for shade analysis.

    Provides visualization, configuration, and reporting tools for
    comprehensive shade analysis.
    """

    def __init__(self):
        """Initialize the Shade Analysis UI."""
        self.setup_page()

    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Helioscope Shade Analysis",
            page_icon="☀️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        """Run the main Streamlit application."""
        st.title("☀️ Helioscope 3D Shade Analysis")
        st.markdown("Comprehensive PV system shade analysis and optimization")

        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            analysis_mode = st.selectbox(
                "Analysis Mode",
                [
                    "Quick Analysis",
                    "Full Shade Analysis",
                    "Layout Optimization",
                    "Electrical Modeling",
                    "3D Visualization"
                ]
            )

        # Main content based on selected mode
        if analysis_mode == "Quick Analysis":
            self.quick_analysis_page()
        elif analysis_mode == "Full Shade Analysis":
            self.full_shade_analysis_page()
        elif analysis_mode == "Layout Optimization":
            self.layout_optimization_page()
        elif analysis_mode == "Electrical Modeling":
            self.electrical_modeling_page()
        elif analysis_mode == "3D Visualization":
            self.visualization_3d_page()

    def quick_analysis_page(self):
        """Quick shade analysis interface."""
        st.header("Quick Shade Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Site Configuration")
            latitude = st.number_input("Latitude (°)", -90.0, 90.0, 37.7749, 0.0001)
            longitude = st.number_input("Longitude (°)", -180.0, 180.0, -122.4194, 0.0001)
            elevation = st.number_input("Elevation (m)", 0.0, 5000.0, 0.0)

        with col2:
            st.subheader("Array Configuration")
            tilt = st.slider("Tilt Angle (°)", 0, 60, 20)
            azimuth = st.slider("Azimuth (°)", 0, 360, 180)
            gcr = st.slider("Ground Coverage Ratio", 0.1, 0.8, 0.4, 0.01)

        if st.button("Run Quick Analysis"):
            with st.spinner("Running shade analysis..."):
                # Create models
                location = Location(
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation
                )

                site_model = SiteModel(location=location, albedo=0.2)

                array_geometry = ArrayGeometry(
                    tilt=tilt,
                    azimuth=azimuth,
                    gcr=gcr,
                    module_width=1.0,
                    module_height=2.0,
                    modules_per_string=20,
                    row_spacing=3.0
                )

                # Generate sample irradiance data
                irradiance_data = self._generate_sample_irradiance(location)

                # Run analysis
                config = ShadeAnalysisConfig(
                    start_date=datetime.now(),
                    end_date=datetime.now() + timedelta(days=365)
                )

                shade_engine = ShadeAnalysisEngine(site_model, array_geometry, config)
                result = shade_engine.run_full_analysis(irradiance_data, total_rows=10)

                # Display results
                st.success("Analysis complete!")
                self._display_quick_results(result)

    def full_shade_analysis_page(self):
        """Full shade analysis interface."""
        st.header("Full Shade Analysis")

        # Tabs for different configuration sections
        tabs = st.tabs(["Site", "Array", "Analysis Config", "Results"])

        with tabs[0]:
            self._site_configuration_form()

        with tabs[1]:
            self._array_configuration_form()

        with tabs[2]:
            self._analysis_configuration_form()

        with tabs[3]:
            if st.button("Run Full Analysis"):
                self._run_full_analysis()

    def layout_optimization_page(self):
        """Layout optimization interface."""
        st.header("Layout Optimization")

        optimization_type = st.selectbox(
            "Optimization Type",
            [
                "Row Spacing",
                "Tilt Angle",
                "Azimuth",
                "Tracker Parameters",
                "Multi-Parameter"
            ]
        )

        st.subheader("Optimization Parameters")

        col1, col2 = st.columns(2)

        with col1:
            if optimization_type == "Row Spacing":
                min_spacing = st.number_input("Min Spacing (m)", 1.0, 10.0, 2.0)
                max_spacing = st.number_input("Max Spacing (m)", 2.0, 20.0, 10.0)
            elif optimization_type == "Tilt Angle":
                min_tilt = st.number_input("Min Tilt (°)", 0.0, 45.0, 0.0)
                max_tilt = st.number_input("Max Tilt (°)", 10.0, 60.0, 45.0)

        with col2:
            num_rows = st.number_input("Number of Rows", 1, 100, 10)
            objective = st.selectbox(
                "Objective",
                ["Maximize Energy", "Maximize Land Use", "Minimize Shading"]
            )

        if st.button("Run Optimization"):
            with st.spinner("Optimizing layout..."):
                self._run_layout_optimization(optimization_type)

    def electrical_modeling_page(self):
        """Electrical shading modeling interface."""
        st.header("Electrical Shading Model")

        st.subheader("Module Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            cells_in_series = st.number_input("Cells in Series", 60, 144, 72)
            bypass_diodes = st.number_input("Bypass Diodes", 1, 6, 3)

        with col2:
            v_oc = st.number_input("Voc (V)", 30.0, 60.0, 48.0)
            i_sc = st.number_input("Isc (A)", 5.0, 15.0, 10.0)

        with col3:
            p_max = st.number_input("Pmax (W)", 200.0, 600.0, 400.0)

        st.subheader("Shading Scenario")

        shading_pattern = st.selectbox(
            "Shading Pattern",
            ["Uniform", "Partial Bottom", "Partial Top", "Random", "Custom"]
        )

        if shading_pattern == "Custom":
            shaded_cells_input = st.text_input(
                "Shaded Cells (comma-separated indices)",
                "0,1,2,3,4"
            )
            shaded_cells = [int(x.strip()) for x in shaded_cells_input.split(",")]
        else:
            num_shaded = st.slider("Number of Shaded Cells", 0, cells_in_series, 10)
            shaded_cells = self._generate_shading_pattern(
                shading_pattern, num_shaded, cells_in_series
            )

        if st.button("Simulate Electrical Effects"):
            self._simulate_electrical_shading(
                cells_in_series, bypass_diodes, v_oc, i_sc, p_max, shaded_cells
            )

    def visualization_3d_page(self):
        """3D visualization interface."""
        st.header("3D System Visualization")

        viz_type = st.selectbox(
            "Visualization Type",
            [
                "Sun Path",
                "System Layout",
                "Shade Animation",
                "Horizon Profile"
            ]
        )

        if viz_type == "Sun Path":
            self._visualize_sun_path()
        elif viz_type == "System Layout":
            self._visualize_system_layout()
        elif viz_type == "Shade Animation":
            self._visualize_shade_animation()
        elif viz_type == "Horizon Profile":
            self._visualize_horizon_profile()

    # Helper methods for UI components

    def _site_configuration_form(self):
        """Display site configuration form."""
        st.subheader("Site Configuration")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input("Latitude", -90.0, 90.0, 37.7749)
            longitude = st.number_input("Longitude", -180.0, 180.0, -122.4194)
            elevation = st.number_input("Elevation (m)", 0.0, 5000.0, 0.0)

        with col2:
            albedo = st.slider("Ground Albedo", 0.0, 1.0, 0.2, 0.01)
            timezone = st.text_input("Timezone", "America/Los_Angeles")

        # Store in session state
        st.session_state.site_config = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "albedo": albedo,
            "timezone": timezone
        }

    def _array_configuration_form(self):
        """Display array configuration form."""
        st.subheader("Array Configuration")

        col1, col2 = st.columns(2)

        with col1:
            tilt = st.slider("Tilt Angle (°)", 0, 60, 20)
            azimuth = st.slider("Azimuth (°)", 0, 360, 180)
            gcr = st.slider("Ground Coverage Ratio", 0.1, 0.8, 0.4, 0.01)

        with col2:
            tracker_type = st.selectbox(
                "Tracker Type",
                ["Fixed Tilt", "Single Axis", "Dual Axis"]
            )
            row_spacing = st.number_input("Row Spacing (m)", 1.0, 20.0, 5.0)

        st.session_state.array_config = {
            "tilt": tilt,
            "azimuth": azimuth,
            "gcr": gcr,
            "tracker_type": tracker_type,
            "row_spacing": row_spacing
        }

    def _analysis_configuration_form(self):
        """Display analysis configuration form."""
        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start Date", datetime.now())
            end_date = st.date_input("End Date", datetime.now() + timedelta(days=365))

        with col2:
            timestep = st.selectbox("Timestep", ["15 min", "30 min", "1 hour"], index=2)
            transposition_model = st.selectbox(
                "Transposition Model",
                ["Perez", "Hay-Davies", "Isotropic"]
            )

        enable_near_shading = st.checkbox("Enable Near Shading", value=True)
        enable_far_shading = st.checkbox("Enable Far Shading", value=True)
        enable_electrical = st.checkbox("Enable Electrical Model", value=True)

        st.session_state.analysis_config = {
            "start_date": start_date,
            "end_date": end_date,
            "timestep": timestep,
            "transposition_model": transposition_model,
            "enable_near_shading": enable_near_shading,
            "enable_far_shading": enable_far_shading,
            "enable_electrical": enable_electrical
        }

    def _display_quick_results(self, result):
        """Display quick analysis results."""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Annual Shading Loss",
                f"{result.annual_shading_loss:.2%}"
            )

        with col2:
            st.metric(
                "Worst Month Loss",
                f"{max(result.monthly_losses.values()):.2%}"
            )

        with col3:
            st.metric(
                "Affected Modules",
                f"{len(result.worst_shaded_modules)}"
            )

        # Monthly loss chart
        st.subheader("Monthly Shading Losses")
        fig = self._plot_monthly_losses(result.monthly_losses)
        st.pyplot(fig)

    def _plot_monthly_losses(self, monthly_losses: Dict[int, float]) -> Figure:
        """Create monthly losses bar chart."""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = list(monthly_losses.keys())
        losses = [monthly_losses[m] * 100 for m in months]

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax.bar(month_names, losses, color='steelblue')
        ax.set_xlabel("Month")
        ax.set_ylabel("Shading Loss (%)")
        ax.set_title("Monthly Shading Losses")
        ax.grid(axis='y', alpha=0.3)

        return fig

    def _generate_sample_irradiance(self, location: Location) -> pd.DataFrame:
        """Generate sample irradiance data for testing."""
        # Generate hourly data for one year
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=8760, freq='h')

        # Simplified irradiance model
        data = []
        for dt in dates:
            # Very simplified - real implementation would use weather data
            hour = dt.hour
            if 6 <= hour <= 18:
                ghi = 800 * np.sin((hour - 6) * np.pi / 12)
                dni = ghi * 0.8
                dhi = ghi * 0.2
            else:
                ghi = dni = dhi = 0.0

            data.append({
                'timestamp': dt,
                'ghi': ghi,
                'dni': dni,
                'dhi': dhi
            })

        return pd.DataFrame(data)

    def _generate_shading_pattern(
        self,
        pattern: str,
        num_shaded: int,
        total_cells: int
    ) -> List[int]:
        """Generate shading pattern for electrical modeling."""
        if pattern == "Uniform":
            return list(range(0, num_shaded))
        elif pattern == "Partial Bottom":
            return list(range(0, num_shaded))
        elif pattern == "Partial Top":
            return list(range(total_cells - num_shaded, total_cells))
        elif pattern == "Random":
            return list(np.random.choice(total_cells, num_shaded, replace=False))
        else:
            return []

    def _simulate_electrical_shading(
        self,
        cells_in_series: int,
        bypass_diodes: int,
        v_oc: float,
        i_sc: float,
        p_max: float,
        shaded_cells: List[int]
    ):
        """Run electrical shading simulation and display results."""
        module_params = ModuleElectricalParams(
            cells_in_series=cells_in_series,
            cell_rows=6,
            cell_columns=cells_in_series // 6,
            bypass_diodes=bypass_diodes,
            cells_per_diode=cells_in_series // bypass_diodes,
            v_oc=v_oc,
            i_sc=i_sc,
            v_mp=v_oc * 0.85,
            i_mp=i_sc * 0.95,
            p_max=p_max
        )

        electrical_model = ElectricalShadingModel(module_params)

        result = electrical_model.bypass_diode_simulation(shaded_cells)

        st.success("Simulation complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Power Loss", f"{result.power_loss:.2%}")

        with col2:
            st.metric("Active Bypass Diodes", len(result.active_bypass_diodes))

        with col3:
            st.metric("Hotspot Risk", "Yes" if result.hotspot_risk else "No")

        # I-V curve
        st.subheader("I-V Curve Under Shading")
        voltage, current = electrical_model.module_iv_under_shade(
            shaded_cells,
            [0 if i in shaded_cells else 1000 for i in range(cells_in_series)]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(voltage, current, 'b-', linewidth=2)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title("Module I-V Curve Under Partial Shading")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    def _visualize_sun_path(self):
        """Visualize 3D sun path."""
        st.subheader("Sun Path Visualization")

        latitude = st.slider("Latitude", -90.0, 90.0, 37.7749)
        longitude = st.slider("Longitude", -180.0, 180.0, -122.4194)
        date = st.date_input("Date", datetime(2024, 6, 21))  # Summer solstice

        location = Location(latitude=latitude, longitude=longitude)
        sun_calc = SunPositionCalculator(location)

        sun_path = sun_calc.sun_path_3d(datetime.combine(date, datetime.min.time()))

        # Create 3D plot
        fig = go.Figure()

        # Extract coordinates
        azimuths = [p.azimuth for p in sun_path if p.is_daylight]
        elevations = [p.elevation for p in sun_path if p.is_daylight]

        # Convert to 3D coordinates
        radius = 100
        x = [radius * np.cos(np.radians(el)) * np.sin(np.radians(az)) for az, el in zip(azimuths, elevations)]
        y = [radius * np.cos(np.radians(el)) * np.cos(np.radians(az)) for az, el in zip(azimuths, elevations)]
        z = [radius * np.sin(np.radians(el)) for el in elevations]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(color='orange', width=3),
            name='Sun Path'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='East-West',
                yaxis_title='North-South',
                zaxis_title='Elevation',
                aspectmode='cube'
            ),
            title='3D Sun Path'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _visualize_system_layout(self):
        """Visualize 3D system layout."""
        st.subheader("3D System Layout")
        st.info("3D system layout visualization would render here")

    def _visualize_shade_animation(self):
        """Create shade animation."""
        st.subheader("Shade Animation")
        st.info("Shade animation for selected date would render here")

    def _visualize_horizon_profile(self):
        """Visualize horizon profile."""
        st.subheader("Horizon Profile")

        # Sample horizon data
        azimuths = list(range(0, 361, 10))
        elevations = [5 * np.sin(np.radians(az * 2)) for az in azimuths]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=elevations,
            theta=azimuths,
            mode='lines',
            line=dict(color='darkgreen', width=2),
            fill='toself',
            name='Horizon'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, max(elevations) * 1.2]),
                angularaxis=dict(direction='clockwise')
            ),
            title='Horizon Profile (Polar View)'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _run_full_analysis(self):
        """Execute full shade analysis."""
        st.info("Running full shade analysis... (Implementation would run comprehensive analysis)")

    def _run_layout_optimization(self, optimization_type: str):
        """Execute layout optimization."""
        st.success(f"Running {optimization_type} optimization...")
        st.info("Optimization results would be displayed here")


def main():
    """Main entry point for Streamlit app."""
    ui = ShadeAnalysisUI()
    ui.run()


if __name__ == "__main__":
    main()
