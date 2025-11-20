"""Module Design UI for PV Circularity Simulator.

This module provides a comprehensive Streamlit interface for designing and analyzing
PV modules, including:
- Interactive module designer
- CTM (Cell-to-Module) loss visualization
- Module configuration tools (BOM, cost, weight)
- Performance prediction
- PAN file generation
- Module comparison
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import our models and calculators
from ..models.cell import CellDesign, CellTemplate
from ..models.module import ModuleConfiguration, ModuleLayout, ModuleDesign
from ..templates.cell_templates import get_cell_templates, get_template_names
from ..ctm.calculator import CTMCalculator, CTMFactors
from ..bom.calculator import BOMCalculator
from ..bom.costs import CostCalculator
from ..bom.weights import WeightCalculator
from ..performance.predictor import PerformancePredictor
from ..performance.degradation import DegradationModel
from ..pan.generator import PANGenerator


# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
"""


def render_custom_css() -> None:
    """Render custom CSS for the application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def create_layout_config(num_cells: int) -> ModuleLayout:
    """Create default layout configuration based on number of cells.

    Args:
        num_cells: Number of cells in module

    Returns:
        ModuleLayout with default configuration
    """
    layout_configs = {
        60: {"rows": 6, "cols": 10, "strings": 3, "cells_per_string": 20, "diodes": 3, "cells_per_diode": 20},
        72: {"rows": 6, "cols": 12, "strings": 3, "cells_per_string": 24, "diodes": 3, "cells_per_diode": 24},
        120: {"rows": 6, "cols": 20, "strings": 3, "cells_per_string": 40, "diodes": 3, "cells_per_diode": 40},
        132: {"rows": 6, "cols": 22, "strings": 3, "cells_per_string": 44, "diodes": 3, "cells_per_diode": 44},
        144: {"rows": 6, "cols": 24, "strings": 3, "cells_per_string": 48, "diodes": 3, "cells_per_diode": 48},
    }

    config = layout_configs.get(num_cells, layout_configs[72])

    return ModuleLayout(
        num_cells=num_cells,
        rows=config["rows"],
        columns=config["cols"],
        num_strings=config["strings"],
        cells_per_string=config["cells_per_string"],
        bypass_diodes=config["diodes"],
        cells_per_diode=config["cells_per_diode"],
    )


def calculate_module_dimensions(
    cell_length_mm: float, cell_width_mm: float, rows: int, cols: int, cell_config: str
) -> Tuple[float, float]:
    """Calculate module dimensions based on cell layout.

    Args:
        cell_length_mm: Cell length in mm
        cell_width_mm: Cell width in mm
        rows: Number of rows
        cols: Number of columns
        cell_config: Cell configuration (full-cell, half-cut, etc.)

    Returns:
        Tuple of (length_mm, width_mm)
    """
    # Add gaps between cells (5mm) and frame (20mm on each side)
    gap_mm = 5
    frame_mm = 20

    if cell_config == "half-cut":
        # Half-cut cells are arranged vertically
        effective_cell_height = cell_length_mm / 2
        length_mm = (effective_cell_height * rows) + (gap_mm * (rows - 1)) + (2 * frame_mm)
        width_mm = (cell_width_mm * cols) + (gap_mm * (cols - 1)) + (2 * frame_mm)
    elif cell_config == "quarter-cut":
        # Quarter-cut cells
        effective_cell_height = cell_length_mm / 2
        effective_cell_width = cell_width_mm / 2
        length_mm = (effective_cell_height * rows) + (gap_mm * (rows - 1)) + (2 * frame_mm)
        width_mm = (effective_cell_width * cols) + (gap_mm * (cols - 1)) + (2 * frame_mm)
    else:
        # Full cell or shingled
        length_mm = (cell_length_mm * rows) + (gap_mm * (rows - 1)) + (2 * frame_mm)
        width_mm = (cell_width_mm * cols) + (gap_mm * (cols - 1)) + (2 * frame_mm)

    return length_mm, width_mm


def render_module_designer() -> Optional[ModuleConfiguration]:
    """Render the interactive module designer section.

    Returns:
        ModuleConfiguration if design is complete, None otherwise
    """
    st.markdown('<div class="section-header">1. Interactive Module Designer</div>', unsafe_allow_html=True)

    # Cell Selection
    st.subheader("Cell Selection")
    cell_templates = get_cell_templates()
    template_names = get_template_names()

    selected_template_name = st.selectbox(
        "Select Cell Template",
        options=template_names,
        help="Choose from pre-defined cell templates with different technologies and sizes",
    )

    selected_template = cell_templates[selected_template_name]

    # Display cell specifications
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cell Technology", selected_template.technology)
    with col2:
        st.metric("Cell Size", f"{selected_template.length_mm}×{selected_template.width_mm} mm")
    with col3:
        st.metric("Cell Efficiency", f"{selected_template.efficiency_pct:.2f}%")
    with col4:
        st.metric("Cell Power", f"{selected_template.pmax_w:.2f} W")

    # Layout Configuration
    st.subheader("Layout Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_cells = st.selectbox(
            "Number of Cells",
            options=[60, 72, 120, 132, 144],
            index=1,
            help="Total number of cells in the module",
        )

    with col2:
        cell_configuration = st.selectbox(
            "Cell Configuration",
            options=["full-cell", "half-cut", "quarter-cut", "shingled"],
            index=1,
            help="Cell cutting and interconnection method",
        )

    with col3:
        bifacial_enabled = selected_template.bifacial
        if bifacial_enabled:
            st.success(f"Bifacial (Factor: {selected_template.bifaciality_factor:.0%})")
        else:
            st.info("Monofacial")

    # Create layout
    layout = create_layout_config(num_cells)

    # String Configuration
    st.subheader("String & Bypass Diode Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_strings = st.number_input(
            "Parallel Strings", min_value=1, max_value=6, value=layout.num_strings, help="Number of parallel strings"
        )

    with col2:
        cells_per_string = st.number_input(
            "Cells per String (Series)",
            min_value=1,
            max_value=200,
            value=layout.cells_per_string,
            help="Cells in series per string",
        )

    with col3:
        bypass_diodes = st.number_input(
            "Bypass Diodes", min_value=1, max_value=10, value=layout.bypass_diodes, help="Number of bypass diodes"
        )

    with col4:
        cells_per_diode = st.number_input(
            "Cells per Diode",
            min_value=1,
            max_value=100,
            value=layout.cells_per_diode,
            help="Cells protected by each diode",
        )

    # Update layout
    layout = ModuleLayout(
        num_cells=num_cells,
        rows=layout.rows,
        columns=layout.columns,
        num_strings=num_strings,
        cells_per_string=cells_per_string,
        bypass_diodes=bypass_diodes,
        cells_per_diode=cells_per_diode,
    )

    # Module Dimensions
    st.subheader("Module Dimensions")

    # Calculate suggested dimensions
    suggested_length, suggested_width = calculate_module_dimensions(
        selected_template.length_mm,
        selected_template.width_mm,
        layout.rows,
        layout.columns,
        cell_configuration,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        module_length = st.number_input(
            "Length (mm)",
            min_value=500.0,
            max_value=3000.0,
            value=float(suggested_length),
            step=10.0,
            help="Module length in millimeters",
        )

    with col2:
        module_width = st.number_input(
            "Width (mm)",
            min_value=500.0,
            max_value=2000.0,
            value=float(suggested_width),
            step=10.0,
            help="Module width in millimeters",
        )

    with col3:
        module_thickness = st.number_input(
            "Thickness (mm)",
            min_value=20.0,
            max_value=50.0,
            value=35.0,
            step=1.0,
            help="Module thickness in millimeters",
        )

    # Frame and Encapsulation
    st.subheader("Frame & Encapsulation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        frame_type = st.selectbox("Frame Type", options=["aluminum", "frameless", "composite"], index=0)

    with col2:
        glass_front = st.number_input("Front Glass (mm)", min_value=2.0, max_value=5.0, value=3.2, step=0.1)

    with col3:
        if bifacial_enabled:
            glass_back = st.number_input("Back Glass (mm)", min_value=1.5, max_value=4.0, value=2.0, step=0.1)
            backsheet_type = None
        else:
            glass_back = None
            backsheet_type = st.selectbox("Backsheet", options=["PET", "TPT", "PPE"], index=0)

    with col4:
        encapsulant_type = st.selectbox("Encapsulant", options=["EVA", "POE", "TPO"], index=0)

    # Electrical Components
    st.subheader("Electrical Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        junction_box = st.selectbox("Junction Box", options=["standard", "smart", "potted"], index=0)

    with col2:
        cable_length = st.number_input("Cable Length (mm)", min_value=500, max_value=2000, value=1200, step=100)

    with col3:
        connector_type = st.selectbox("Connector Type", options=["MC4", "MC4-EVO2", "Amphenol H4"], index=0)

    # Create cell design
    cell_design = CellDesign(template=selected_template, quantity=num_cells, configuration=cell_configuration)

    # Create module configuration
    module_config = ModuleConfiguration(
        name=f"{selected_template.technology}_{num_cells}cell_{int(cell_design.total_pmax_w)}W",
        cell_design=cell_design,
        layout=layout,
        length_mm=module_length,
        width_mm=module_width,
        thickness_mm=module_thickness,
        frame_type=frame_type,
        glass_front_mm=glass_front,
        glass_back_mm=glass_back,
        backsheet_type=backsheet_type,
        encapsulant_type=encapsulant_type,
        junction_box_type=junction_box,
        cable_length_mm=cable_length,
        connector_type=connector_type,
    )

    # Real-time Power Rating Estimation
    st.subheader("Real-time Power Estimation")

    # Calculate CTM with default factors
    ctm_calc = CTMCalculator()
    ctm_result = ctm_calc.calculate(selected_template.pmax_w, num_cells, cell_configuration)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Cell Power Total</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ctm_result.cell_pmax_total_w:.1f} W</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Module Power (STC)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ctm_result.module_pmax_w:.1f} W</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">CTM Ratio</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ctm_result.ctm_ratio:.3f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        module_efficiency = (ctm_result.module_pmax_w / (module_config.area_m2 * 1000)) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Module Efficiency</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{module_efficiency:.2f}%</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return module_config


def render_ctm_visualization(module_config: ModuleConfiguration) -> CTMFactors:
    """Render CTM loss visualization and k-factor editor.

    Args:
        module_config: Module configuration

    Returns:
        CTMFactors with user-adjusted values
    """
    st.markdown('<div class="section-header">2. CTM Loss Visualization</div>', unsafe_allow_html=True)

    # Create tabs for different CTM analysis views
    tab1, tab2, tab3 = st.tabs(["K-Factor Editor", "Loss Waterfall", "Sensitivity Analysis"])

    # Initialize with default factors
    if "ctm_factors" not in st.session_state:
        st.session_state.ctm_factors = CTMFactors()

    ctm_factors = st.session_state.ctm_factors

    with tab1:
        st.subheader("Interactive K-Factor Editor")
        st.markdown(
            '<div class="info-box">Adjust k-factors to model different loss mechanisms. '
            "Values closer to 1.0 indicate lower losses.</div>",
            unsafe_allow_html=True,
        )

        # Optical Losses (k1-k8)
        st.markdown("**Optical Losses (k1-k8)**")
        col1, col2 = st.columns(2)

        with col1:
            k1 = st.slider("k1: Reflection Loss", 0.90, 1.00, float(ctm_factors.k1_reflection_loss), 0.001)
            k2 = st.slider("k2: Glass Transmission", 0.90, 1.00, float(ctm_factors.k2_glass_transmission), 0.001)
            k3 = st.slider(
                "k3: Encapsulant Transmission", 0.95, 1.00, float(ctm_factors.k3_encapsulant_transmission), 0.001
            )
            k4 = st.slider("k4: Soiling Factor", 0.90, 1.00, float(ctm_factors.k4_soiling_factor), 0.001)

        with col2:
            k5 = st.slider("k5: Spectral Mismatch", 0.95, 1.00, float(ctm_factors.k5_spectral_mismatch), 0.001)
            k6 = st.slider("k6: Angular Losses", 0.90, 1.00, float(ctm_factors.k6_angular_losses), 0.001)
            k7 = st.slider("k7: Inactive Area", 0.85, 1.00, float(ctm_factors.k7_inactive_area), 0.001)
            k8 = st.slider("k8: Optical Coupling", 0.98, 1.00, float(ctm_factors.k8_optical_coupling), 0.001)

        # Electrical Losses (k9-k16)
        st.markdown("**Electrical Losses (k9-k16)**")
        col1, col2 = st.columns(2)

        with col1:
            k9 = st.slider(
                "k9: Interconnection Loss", 0.95, 1.00, float(ctm_factors.k9_interconnection_loss), 0.001
            )
            k10 = st.slider("k10: Series Resistance", 0.95, 1.00, float(ctm_factors.k10_series_resistance), 0.001)
            k11 = st.slider("k11: Shunt Resistance", 0.99, 1.00, float(ctm_factors.k11_shunt_resistance), 0.001)
            k12 = st.slider("k12: Cell Mismatch", 0.95, 1.00, float(ctm_factors.k12_cell_mismatch), 0.001)

        with col2:
            k13 = st.slider("k13: Diode Losses", 0.99, 1.00, float(ctm_factors.k13_diode_losses), 0.001)
            k14 = st.slider("k14: Junction Box", 0.99, 1.00, float(ctm_factors.k14_junction_box), 0.001)
            k15 = st.slider("k15: Cable Resistance", 0.99, 1.00, float(ctm_factors.k15_cable_resistance), 0.001)
            k16 = st.slider("k16: Contact Resistance", 0.99, 1.00, float(ctm_factors.k16_contact_resistance), 0.001)

        # Thermal Losses (k17-k20)
        st.markdown("**Thermal Losses (k17-k20)**")
        col1, col2 = st.columns(2)

        with col1:
            k17 = st.slider("k17: Thermal Mismatch", 0.95, 1.00, float(ctm_factors.k17_thermal_mismatch), 0.001)
            k18 = st.slider("k18: Heat Dissipation", 0.95, 1.00, float(ctm_factors.k18_heat_dissipation), 0.001)

        with col2:
            k19 = st.slider("k19: NOCT Effect", 0.90, 1.00, float(ctm_factors.k19_noct_effect), 0.001)
            k20 = st.slider("k20: Hot Spot Risk", 0.98, 1.00, float(ctm_factors.k20_hot_spot_risk), 0.001)

        # Manufacturing Losses (k21-k24)
        st.markdown("**Manufacturing & Quality Losses (k21-k24)**")
        col1, col2 = st.columns(2)

        with col1:
            k21 = st.slider(
                "k21: Manufacturing Tolerance", 0.95, 1.00, float(ctm_factors.k21_manufacturing_tolerance), 0.001
            )
            k22 = st.slider("k22: Lamination Quality", 0.98, 1.00, float(ctm_factors.k22_lamination_quality), 0.001)

        with col2:
            k23 = st.slider("k23: Edge Deletion", 0.99, 1.00, float(ctm_factors.k23_edge_deletion), 0.001)
            k24 = st.slider(
                "k24: Measurement Uncertainty", 0.97, 1.00, float(ctm_factors.k24_measurement_uncertainty), 0.001
            )

        # Update factors
        ctm_factors = CTMFactors(
            k1_reflection_loss=k1,
            k2_glass_transmission=k2,
            k3_encapsulant_transmission=k3,
            k4_soiling_factor=k4,
            k5_spectral_mismatch=k5,
            k6_angular_losses=k6,
            k7_inactive_area=k7,
            k8_optical_coupling=k8,
            k9_interconnection_loss=k9,
            k10_series_resistance=k10,
            k11_shunt_resistance=k11,
            k12_cell_mismatch=k12,
            k13_diode_losses=k13,
            k14_junction_box=k14,
            k15_cable_resistance=k15,
            k16_contact_resistance=k16,
            k17_thermal_mismatch=k17,
            k18_heat_dissipation=k18,
            k19_noct_effect=k19,
            k20_hot_spot_risk=k20,
            k21_manufacturing_tolerance=k21,
            k22_lamination_quality=k22,
            k23_edge_deletion=k23,
            k24_measurement_uncertainty=k24,
        )

        st.session_state.ctm_factors = ctm_factors

        # Display overall CTM ratio
        st.markdown("**Overall CTM Ratio**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Optical", f"{ctm_factors.optical_loss_total:.4f}", f"-{(1-ctm_factors.optical_loss_total)*100:.2f}%")
        with col2:
            st.metric(
                "Electrical", f"{ctm_factors.electrical_loss_total:.4f}", f"-{(1-ctm_factors.electrical_loss_total)*100:.2f}%"
            )
        with col3:
            st.metric("Thermal", f"{ctm_factors.thermal_loss_total:.4f}", f"-{(1-ctm_factors.thermal_loss_total)*100:.2f}%")
        with col4:
            st.metric("CTM Total", f"{ctm_factors.ctm_ratio:.4f}", f"-{(1-ctm_factors.ctm_ratio)*100:.2f}%")

    with tab2:
        st.subheader("Loss Waterfall Chart")

        # Get waterfall data
        waterfall_data = ctm_factors.get_loss_waterfall()

        # Create waterfall chart
        categories = [item["category"] for item in waterfall_data]
        values = [item["value"] * 100 for item in waterfall_data]  # Convert to percentage

        # Create figure
        fig = go.Figure()

        # Add bars for waterfall
        for i in range(len(categories)):
            if i == 0 or i == len(categories) - 1:
                # First and last are totals
                fig.add_trace(
                    go.Bar(
                        x=[categories[i]],
                        y=[values[i]],
                        name=categories[i],
                        marker_color="lightblue" if i == 0 else "lightgreen",
                        text=[f"{values[i]:.2f}%"],
                        textposition="outside",
                    )
                )
            else:
                # Intermediate losses
                loss_pct = (values[i - 1] - values[i])
                fig.add_trace(
                    go.Bar(
                        x=[categories[i]],
                        y=[values[i]],
                        name=categories[i],
                        marker_color="coral",
                        text=[f"{values[i]:.2f}%<br>(-{loss_pct:.2f}%)"],
                        textposition="outside",
                    )
                )

        fig.update_layout(
            title="Cell-to-Module Power Loss Waterfall",
            xaxis_title="Loss Category",
            yaxis_title="Power Retention (%)",
            height=600,
            showlegend=False,
            xaxis_tickangle=-45,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display loss summary table
        st.markdown("**Loss Summary**")
        loss_df = pd.DataFrame(waterfall_data)
        loss_df["loss_pct"] = loss_df["loss_pct"].round(3)
        loss_df["value"] = (loss_df["value"] * 100).round(2)
        loss_df.columns = ["Category", "Power Retention (%)", "Loss (%)"]
        st.dataframe(loss_df, use_container_width=True)

    with tab3:
        st.subheader("Sensitivity Analysis")

        st.markdown(
            '<div class="info-box">Analyze how changes in individual k-factors affect module power output.</div>',
            unsafe_allow_html=True,
        )

        # Select factor for sensitivity analysis
        all_factors = list(ctm_factors.get_all_factors().keys())
        selected_factor = st.selectbox("Select K-Factor for Analysis", options=all_factors)

        # Range and steps
        col1, col2 = st.columns(2)
        with col1:
            range_pct = st.slider("Variation Range (±%)", 1, 20, 10)
        with col2:
            steps = st.slider("Analysis Steps", 10, 50, 20)

        # Perform sensitivity analysis
        ctm_calc = CTMCalculator(ctm_factors)
        cell_pmax = module_config.cell_design.template.pmax_w
        num_cells = module_config.layout.num_cells

        sensitivity_result = ctm_calc.sensitivity_analysis(cell_pmax, num_cells, selected_factor, range_pct, steps)

        # Plot sensitivity
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sensitivity_result["factor_values"],
                y=sensitivity_result["module_pmax_values"],
                mode="lines+markers",
                name="Module Power",
                line=dict(color="blue", width=3),
                marker=dict(size=6),
            )
        )

        # Add current value marker
        current_value = getattr(ctm_factors, selected_factor)
        current_calc = CTMCalculator(ctm_factors)
        current_result = current_calc.calculate(cell_pmax, num_cells)

        fig.add_trace(
            go.Scatter(
                x=[current_value],
                y=[current_result.module_pmax_w],
                mode="markers",
                name="Current Value",
                marker=dict(size=15, color="red", symbol="star"),
            )
        )

        fig.update_layout(
            title=f"Sensitivity Analysis: {selected_factor}",
            xaxis_title=f"{selected_factor} Value",
            yaxis_title="Module Power (W)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display statistics
        power_range = sensitivity_result["module_pmax_values"].max() - sensitivity_result["module_pmax_values"].min()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Min Power", f"{sensitivity_result['module_pmax_values'].min():.2f} W")
        with col2:
            st.metric("Max Power", f"{sensitivity_result['module_pmax_values'].max():.2f} W")
        with col3:
            st.metric("Power Range", f"{power_range:.2f} W")

    return ctm_factors


def render_module_configuration_tools(module_config: ModuleConfiguration, module_power_w: float) -> None:
    """Render module configuration tools (BOM, cost, weight).

    Args:
        module_config: Module configuration
        module_power_w: Module power in watts
    """
    st.markdown('<div class="section-header">3. Module Configuration Tools</div>', unsafe_allow_html=True)

    # Calculate BOM
    bom_calc = BOMCalculator()
    bom_result = bom_calc.calculate(module_config)

    # Calculate costs
    cost_calc = CostCalculator(labor_cost_multiplier=0.3, overhead_multiplier=0.2, margin_pct=20.0)
    cost_result = cost_calc.calculate(bom_result, module_power_w)

    # Calculate weights
    weight_calc = WeightCalculator()
    weight_result = weight_calc.calculate(bom_result, module_config.area_m2, module_power_w)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["BOM Generator", "Cost Calculator", "Weight Calculator"])

    with tab1:
        st.subheader("Bill of Materials (BOM)")

        # BOM summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Components", len(bom_result.items))
        with col2:
            st.metric("Total Weight", f"{bom_result.total_weight_kg:.2f} kg")
        with col3:
            st.metric("Material Cost", f"${bom_result.total_cost_usd:.2f}")
        with col4:
            st.metric("Avg Recyclability", f"{bom_result.average_recyclability_pct:.1f}%")

        # BOM table
        st.markdown("**Component Breakdown**")
        bom_data = []
        for item in bom_result.items:
            bom_data.append(
                {
                    "Component": item.component,
                    "Material": item.material,
                    "Quantity": f"{item.quantity:.2f}",
                    "Unit": item.unit,
                    "Weight (kg)": f"{item.weight_kg:.3f}",
                    "Cost (USD)": f"${item.cost_usd:.2f}",
                    "Recyclability (%)": f"{item.recyclability_pct:.1f}%",
                }
            )

        bom_df = pd.DataFrame(bom_data)
        st.dataframe(bom_df, use_container_width=True)

        # Download BOM as CSV
        csv = bom_df.to_csv(index=False)
        st.download_button(
            label="Download BOM as CSV",
            data=csv,
            file_name=f"{module_config.name}_BOM.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Cost Calculator")

        # Cost breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cost Breakdown**")
            st.metric("Material Cost", f"${cost_result.material_cost_usd:.2f}")
            st.metric("Labor Cost", f"${cost_result.labor_cost_usd:.2f}")
            st.metric("Overhead Cost", f"${cost_result.overhead_cost_usd:.2f}")
            st.metric("Total Mfg Cost", f"${cost_result.total_manufacturing_cost_usd:.2f}")

        with col2:
            st.markdown("**Pricing**")
            st.metric("Profit Margin", f"{cost_result.margin_pct:.1f}%")
            st.metric("Selling Price", f"${cost_result.selling_price_usd:.2f}")
            st.metric("Cost per Watt", f"${cost_result.cost_per_watt_usd:.3f}/W")

            # Calculate LCOE contribution
            lcoe = cost_calc.calculate_lcoe_contribution(cost_result.selling_price_usd, module_power_w)
            st.metric("LCOE Contribution", f"${lcoe['lcoe_module_usd_kwh']:.4f}/kWh")

        # Cost breakdown chart
        cost_breakdown_data = {
            "Category": ["Material", "Labor", "Overhead", "Profit"],
            "Amount": [
                cost_result.material_cost_usd,
                cost_result.labor_cost_usd,
                cost_result.overhead_cost_usd,
                cost_result.selling_price_usd - cost_result.total_manufacturing_cost_usd,
            ],
        }

        fig = px.pie(
            cost_breakdown_data,
            values="Amount",
            names="Category",
            title="Cost Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Weight Calculator")

        # Weight metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Weight", f"{weight_result.total_weight_kg:.2f} kg")
        with col2:
            st.metric("Weight per m²", f"{weight_result.weight_per_m2_kg:.2f} kg/m²")
        with col3:
            st.metric("Weight per Watt", f"{weight_result.weight_per_watt_kg * 1000:.2f} g/W")

        # Component weight breakdown
        st.markdown("**Weight by Component**")
        component_weight_df = pd.DataFrame(
            list(weight_result.component_weights.items()), columns=["Component", "Weight (kg)"]
        )
        component_weight_df["Weight (kg)"] = component_weight_df["Weight (kg)"].round(3)
        component_weight_df["Percentage"] = (
            (component_weight_df["Weight (kg)"] / weight_result.total_weight_kg) * 100
        ).round(1)

        fig = px.bar(
            component_weight_df,
            x="Component",
            y="Weight (kg)",
            title="Weight Distribution by Component",
            text="Weight (kg)",
            color="Percentage",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.2f} kg", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(component_weight_df, use_container_width=True)


def render_performance_prediction(module_config: ModuleConfiguration, ctm_factors: CTMFactors) -> ModuleDesign:
    """Render performance prediction section.

    Args:
        module_config: Module configuration
        ctm_factors: CTM factors

    Returns:
        ModuleDesign with performance specifications
    """
    st.markdown('<div class="section-header">4. Performance Prediction</div>', unsafe_allow_html=True)

    # Calculate module power with CTM factors
    ctm_calc = CTMCalculator(ctm_factors)
    ctm_result = ctm_calc.calculate(
        module_config.cell_design.template.pmax_w,
        module_config.layout.num_cells,
        module_config.cell_design.configuration,
    )

    # Calculate voltage and current at module level
    cells_series = module_config.layout.cells_per_string
    cells_parallel = module_config.layout.num_strings

    # STC parameters (scaled from cell to module)
    voc_stc = module_config.cell_design.template.voc_v * cells_series
    isc_stc = module_config.cell_design.template.isc_a * cells_parallel
    vmp_stc = module_config.cell_design.template.vmp_v * cells_series
    imp_stc = module_config.cell_design.template.imp_a * cells_parallel
    pmax_stc = ctm_result.module_pmax_w

    efficiency_stc = (pmax_stc / (module_config.area_m2 * 1000)) * 100

    # Create module design
    module_design = ModuleDesign(
        configuration=module_config,
        pmax_stc_w=pmax_stc,
        voc_stc_v=voc_stc,
        isc_stc_a=isc_stc,
        vmp_stc_v=vmp_stc,
        imp_stc_a=imp_stc,
        efficiency_stc_pct=efficiency_stc,
        pmax_noct_w=pmax_stc * 0.90,  # Approximate NOCT power
        noct_temp_c=45.0,
        temp_coeff_pmax_pct=module_config.cell_design.template.temp_coeff_pmax,
        temp_coeff_voc_pct=module_config.cell_design.template.temp_coeff_voc,
        temp_coeff_isc_pct=module_config.cell_design.template.temp_coeff_isc,
        bifacial_gain_pct=15.0 if module_config.is_bifacial else None,
        initial_degradation_pct=2.0,
        annual_degradation_pct=0.5,
        ctm_ratio=ctm_factors.ctm_ratio,
    )

    # Performance predictor
    predictor = PerformancePredictor(module_design)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["STC & NOCT Performance", "Temperature Effects", "Bifacial Gain", "Degradation Projection"]
    )

    with tab1:
        st.subheader("Standard Test Conditions (STC) Performance")
        st.caption("STC: 1000 W/m², 25°C, AM1.5")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Pmax", f"{pmax_stc:.2f} W")
        with col2:
            st.metric("Voc", f"{voc_stc:.2f} V")
        with col3:
            st.metric("Isc", f"{isc_stc:.2f} A")
        with col4:
            st.metric("Efficiency", f"{efficiency_stc:.2f}%")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Vmp", f"{vmp_stc:.2f} V")
        with col2:
            st.metric("Imp", f"{imp_stc:.2f} A")
        with col3:
            fill_factor = (pmax_stc / (voc_stc * isc_stc))
            st.metric("Fill Factor", f"{fill_factor:.3f}")
        with col4:
            st.metric("Area", f"{module_config.area_m2:.3f} m²")

        st.markdown("---")

        st.subheader("NOCT Performance")
        st.caption("NOCT: 800 W/m², 20°C ambient, 1 m/s wind, 45°C cell temp")

        noct_result = predictor.predict_noct()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Pmax", f"{noct_result['pmax_w']:.2f} W")
        with col2:
            st.metric("Voc", f"{noct_result['voc_v']:.2f} V")
        with col3:
            st.metric("Isc", f"{noct_result['isc_a']:.2f} A")
        with col4:
            st.metric("Efficiency", f"{noct_result['efficiency_pct']:.2f}%")

    with tab2:
        st.subheader("Temperature Coefficient Effects")

        # Temperature coefficients
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Temp Coeff (Pmax)", f"{module_design.temp_coeff_pmax_pct:.3f} %/°C")
        with col2:
            st.metric("Temp Coeff (Voc)", f"{module_design.temp_coeff_voc_pct:.3f} %/°C")
        with col3:
            st.metric("Temp Coeff (Isc)", f"{module_design.temp_coeff_isc_pct:.3f} %/°C")

        # Power at different temperatures
        st.markdown("**Power Output at Different Cell Temperatures**")

        temps = np.arange(0, 85, 5)
        powers = []

        for temp in temps:
            result = predictor.predict_at_temperature(temp)
            powers.append(result["pmax_w"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=temps, y=powers, mode="lines+markers", name="Module Power", line=dict(width=3)))

        fig.add_hline(y=pmax_stc, line_dash="dash", line_color="red", annotation_text="STC (25°C)")

        fig.update_layout(
            title="Module Power vs. Cell Temperature",
            xaxis_title="Cell Temperature (°C)",
            yaxis_title="Power (W)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Specific temperature points
        col1, col2, col3 = st.columns(3)

        with col1:
            result_25 = predictor.predict_at_temperature(25)
            st.metric("Power @ 25°C", f"{result_25['pmax_w']:.2f} W", "STC")

        with col2:
            result_50 = predictor.predict_at_temperature(50)
            delta_50 = result_50["pmax_w"] - pmax_stc
            st.metric("Power @ 50°C", f"{result_50['pmax_w']:.2f} W", f"{delta_50:.2f} W")

        with col3:
            result_75 = predictor.predict_at_temperature(75)
            delta_75 = result_75["pmax_w"] - pmax_stc
            st.metric("Power @ 75°C", f"{result_75['pmax_w']:.2f} W", f"{delta_75:.2f} W")

    with tab3:
        st.subheader("Bifacial Gain Estimation")

        if module_config.is_bifacial:
            # Bifacial parameters
            col1, col2 = st.columns(2)

            with col1:
                albedo = st.slider("Ground Albedo", 0.0, 1.0, 0.25, 0.05, help="Reflectivity of ground surface")
            with col2:
                rear_irradiance_factor = st.slider(
                    "Rear Irradiance Factor",
                    0.0,
                    0.5,
                    0.3,
                    0.05,
                    help="Rear irradiance as fraction of front",
                )

            # Calculate bifacial gain
            bifacial_result = predictor.predict_bifacial_gain(albedo, rear_irradiance_factor)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Front Power", f"{pmax_stc:.2f} W")
            with col2:
                st.metric("Bifacial Gain", f"{bifacial_result['bifacial_gain_w']:.2f} W")
            with col3:
                st.metric(
                    "Total Power",
                    f"{bifacial_result['total_power_w']:.2f} W",
                    f"+{bifacial_result['gain_pct']:.1f}%",
                )

            # Bifacial gain vs albedo
            st.markdown("**Bifacial Gain vs. Ground Albedo**")

            albedos = np.arange(0, 1.05, 0.05)
            gains = []
            total_powers = []

            for alb in albedos:
                result = predictor.predict_bifacial_gain(alb, rear_irradiance_factor)
                gains.append(result["bifacial_gain_w"])
                total_powers.append(result["total_power_w"])

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=albedos,
                    y=total_powers,
                    mode="lines+markers",
                    name="Total Power",
                    line=dict(width=3, color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=albedos, y=gains, mode="lines+markers", name="Bifacial Gain", line=dict(width=3, color="green")
                )
            )

            fig.update_layout(
                title="Bifacial Performance vs. Albedo",
                xaxis_title="Ground Albedo",
                yaxis_title="Power (W)",
                height=500,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("This module is monofacial. No bifacial gain available.")

    with tab4:
        st.subheader("Degradation Projections (25 Years)")

        # Degradation model
        degradation_model = DegradationModel(
            initial_power_w=pmax_stc,
            initial_degradation_pct=module_design.initial_degradation_pct,
            annual_degradation_pct=module_design.annual_degradation_pct,
            degradation_mode="linear",
        )

        degradation_result = degradation_model.calculate(25)

        # Degradation chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=degradation_result.years,
                y=degradation_result.absolute_power_w,
                mode="lines+markers",
                name="Module Power",
                line=dict(width=3, color="blue"),
                fill="tozeroy",
            )
        )

        # Add warranty line (typical: 84.8% after 25 years)
        warranty_power = pmax_stc * 0.848
        fig.add_hline(
            y=warranty_power, line_dash="dash", line_color="red", annotation_text="Typical Warranty (84.8%)"
        )

        fig.update_layout(
            title="Power Degradation over 25 Years",
            xaxis_title="Years",
            yaxis_title="Power (W)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key milestones
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            year_1_power = degradation_result.absolute_power_w[1]
            st.metric("Year 1", f"{year_1_power:.2f} W", f"{degradation_result.power_retention_pct[1]:.2f}%")

        with col2:
            year_10_power = degradation_result.absolute_power_w[10]
            st.metric("Year 10", f"{year_10_power:.2f} W", f"{degradation_result.power_retention_pct[10]:.2f}%")

        with col3:
            year_25_power = degradation_result.absolute_power_w[25]
            st.metric("Year 25", f"{year_25_power:.2f} W", f"{degradation_result.power_retention_pct[25]:.2f}%")

        with col4:
            total_degradation = degradation_result.cumulative_degradation_pct[25]
            st.metric("Total Degradation", f"{total_degradation:.2f}%", delta=None)

        # Warranty compliance check
        warranty_check = degradation_model.calculate_warranty_compliance(25)

        if warranty_check["all_compliant"]:
            st.markdown(
                '<div class="success-box">Module meets all warranty requirements over 25 years.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">Module may not meet all warranty requirements. Check details below.</div>',
                unsafe_allow_html=True,
            )

    return module_design


def render_pan_file_generator(module_design: ModuleDesign) -> None:
    """Render PAN file generator section.

    Args:
        module_design: Module design specification
    """
    st.markdown('<div class="section-header">5. PAN File Generator</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">Generate PVsyst-compatible PAN files for system simulation.</div>',
        unsafe_allow_html=True,
    )

    # PAN file parameters
    col1, col2 = st.columns(2)

    with col1:
        module_name = st.text_input("Module Name", value=module_design.configuration.name)
        manufacturer = st.text_input("Manufacturer", value="Custom Manufacturer")

    with col2:
        model_number = st.text_input("Model Number", value=f"MOD-{int(module_design.pmax_stc_w)}")
        date = st.date_input("Date")

    # Generate PAN file
    if st.button("Generate PAN File", type="primary"):
        pan_gen = PANGenerator(module_design)
        pan_content = pan_gen.generate(
            module_name=module_name,
            manufacturer=manufacturer,
            model_number=model_number,
            date=date.strftime("%Y-%m-%d"),
        )

        # Validate PAN file
        validation = pan_gen.validate(pan_content.content)

        # Display validation results
        if validation["is_valid"]:
            st.markdown('<div class="success-box">PAN file is valid!</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warning-box">PAN file has validation errors. See below.</div>',
                unsafe_allow_html=True,
            )

            if validation["errors"]:
                st.error("Errors:")
                for error in validation["errors"]:
                    st.write(f"- {error}")

        if validation["warnings"]:
            st.warning("Warnings:")
            for warning in validation["warnings"]:
                st.write(f"- {warning}")

        # Preview PAN file
        st.subheader("PAN File Preview")
        st.code(pan_content.content, language="text")

        # Download button
        st.download_button(
            label="Download PAN File",
            data=pan_content.content,
            file_name=pan_content.filename,
            mime="text/plain",
        )


def render_module_comparison() -> None:
    """Render module comparison tool."""
    st.markdown('<div class="section-header">6. Module Comparison Tool</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">Compare up to 4 module configurations side-by-side. '
        "Save modules in previous sections to compare here.</div>",
        unsafe_allow_html=True,
    )

    # Initialize session state for saved modules
    if "saved_modules" not in st.session_state:
        st.session_state.saved_modules = []

    # Display saved modules count
    st.info(f"Saved modules: {len(st.session_state.saved_modules)}/4")

    # Comparison table (placeholder)
    if len(st.session_state.saved_modules) >= 2:
        st.markdown("**Module Comparison**")

        comparison_data = []
        for module in st.session_state.saved_modules:
            comparison_data.append(
                {
                    "Module": module.get("name", "N/A"),
                    "Power (W)": module.get("power", "N/A"),
                    "Efficiency (%)": module.get("efficiency", "N/A"),
                    "Area (m²)": module.get("area", "N/A"),
                    "Cost ($/W)": module.get("cost_per_watt", "N/A"),
                    "Weight (kg)": module.get("weight", "N/A"),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Radar chart
        if len(st.session_state.saved_modules) >= 2:
            st.markdown("**Technology Comparison Radar Chart**")
            # Placeholder for radar chart
            st.info("Radar chart will be displayed here comparing key metrics across modules.")

    else:
        st.info("Save at least 2 modules to enable comparison.")


def render_3d_visualization(module_config: ModuleConfiguration) -> None:
    """Render 3D visualization (optional).

    Args:
        module_config: Module configuration
    """
    st.markdown('<div class="section-header">7. 3D Visualization (Optional)</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">3D visualization of module cross-section and cell layout.</div>',
        unsafe_allow_html=True,
    )

    # Placeholder for 3D visualization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Module Cross-Section**")
        st.info("3D cross-section view showing glass, cells, encapsulant, and backsheet layers.")

    with col2:
        st.markdown("**Cell Layout**")
        st.info(f"Grid layout: {module_config.layout.rows} rows × {module_config.layout.columns} columns")

        # Simple 2D representation
        fig = go.Figure()

        # Draw cells as rectangles
        for row in range(module_config.layout.rows):
            for col in range(module_config.layout.columns):
                fig.add_shape(
                    type="rect",
                    x0=col,
                    y0=row,
                    x1=col + 0.9,
                    y1=row + 0.9,
                    line=dict(color="blue"),
                    fillcolor="lightblue",
                )

        fig.update_layout(
            title="Cell Layout (2D View)",
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=400,
            showlegend=False,
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Module Design - PV Circularity Simulator",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Render custom CSS
    render_custom_css()

    # Header
    st.markdown(
        '<div class="main-header">☀️ Module Design & Interactive Configuration</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="info-box">
    <strong>PV Circularity Simulator - Module Design Tool</strong><br>
    Design and analyze PV modules with comprehensive CTM loss modeling, performance prediction,
    cost/weight analysis, and PVsyst PAN file generation.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("**Sections:**")
        st.markdown("1. Interactive Module Designer")
        st.markdown("2. CTM Loss Visualization")
        st.markdown("3. Module Configuration Tools")
        st.markdown("4. Performance Prediction")
        st.markdown("5. PAN File Generator")
        st.markdown("6. Module Comparison")
        st.markdown("7. 3D Visualization")

        st.markdown("---")

        # Save/Load functionality
        st.header("Save/Load")

        if st.button("Save Current Design"):
            st.success("Design saved! (Feature in development)")

        if st.button("Load Saved Design"):
            st.info("Load design (Feature in development)")

        # Reset button
        if st.button("Reset All", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # Main content
    try:
        # 1. Module Designer
        module_config = render_module_designer()

        if module_config:
            st.markdown("---")

            # 2. CTM Visualization
            ctm_factors = render_ctm_visualization(module_config)

            st.markdown("---")

            # Calculate module power with CTM
            ctm_calc = CTMCalculator(ctm_factors)
            ctm_result = ctm_calc.calculate(
                module_config.cell_design.template.pmax_w,
                module_config.layout.num_cells,
                module_config.cell_design.configuration,
            )
            module_power_w = ctm_result.module_pmax_w

            # 3. Module Configuration Tools
            render_module_configuration_tools(module_config, module_power_w)

            st.markdown("---")

            # 4. Performance Prediction
            module_design = render_performance_prediction(module_config, ctm_factors)

            st.markdown("---")

            # 5. PAN File Generator
            render_pan_file_generator(module_design)

            st.markdown("---")

            # 6. Module Comparison
            render_module_comparison()

            st.markdown("---")

            # 7. 3D Visualization
            render_3d_visualization(module_config)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
    <strong>PV Circularity Simulator</strong> | Module Design Tool v0.1.0<br>
    Production-ready Streamlit UI for comprehensive PV module design and analysis
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
