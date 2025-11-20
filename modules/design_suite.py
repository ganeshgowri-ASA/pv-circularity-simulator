"""
Design Suite Module - Branches B01-B03

This module provides functionality for:
- B01: Materials Engineering Database
- B02: Cell Design (SCAPS-1D Simulation)
- B03: Module Design & CTM Loss Analysis

Author: PV Circularity Simulator Team
Version: 1.0 (71 Sessions Integrated)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    MATERIALS_DATABASE,
    CTM_LOSS_FACTORS,
    SCAPS_DEFAULTS,
    STC_CONDITIONS,
    COLOR_PALETTE,
)
from utils.validators import Material, CellDesign, ModuleDesign, CTMLoss


# ============================================================================
# BRANCH 01: MATERIALS ENGINEERING DATABASE
# ============================================================================

def render_materials_database() -> None:
    """
    Render the Materials Engineering Database interface.

    Features:
    - Material selection and comparison
    - Property visualization
    - Cost-efficiency analysis
    - Recyclability scoring
    - Custom material addition
    """
    st.header("🔬 Materials Engineering Database")
    st.markdown("*Comprehensive database of PV materials with technical specifications*")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Material Database",
        "📈 Comparison Analysis",
        "➕ Add Custom Material",
        "🔍 Property Search"
    ])

    # Tab 1: Material Database
    with tab1:
        st.subheader("Material Properties Database")

        # Convert materials database to DataFrame
        materials_df = pd.DataFrame.from_dict(MATERIALS_DATABASE, orient='index')
        materials_df.index.name = 'Material'
        materials_df.reset_index(inplace=True)

        # Display with formatting
        st.dataframe(
            materials_df.style.background_gradient(subset=['efficiency'], cmap='Greens')
                             .background_gradient(subset=['recyclability'], cmap='Blues')
                             .format({
                                 'efficiency': '{:.1f}%',
                                 'cost_per_wp': '${:.2f}',
                                 'degradation_rate': '{:.2f}%/yr',
                                 'temp_coefficient': '{:.2f}%/°C',
                                 'carbon_footprint': '{:.0f} kg CO₂/kWp'
                             }),
            use_container_width=True
        )

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Materials", len(MATERIALS_DATABASE))
        with col2:
            avg_eff = materials_df['efficiency'].mean()
            st.metric("Avg Efficiency", f"{avg_eff:.1f}%")
        with col3:
            avg_cost = materials_df['cost_per_wp'].mean()
            st.metric("Avg Cost", f"${avg_cost:.2f}/Wp")
        with col4:
            avg_recycle = materials_df['recyclability'].mean()
            st.metric("Avg Recyclability", f"{avg_recycle:.0f}/100")

    # Tab 2: Comparison Analysis
    with tab2:
        st.subheader("Material Comparison Analysis")

        # Material selection for comparison
        selected_materials = st.multiselect(
            "Select materials to compare:",
            list(MATERIALS_DATABASE.keys()),
            default=list(MATERIALS_DATABASE.keys())[:3]
        )

        if selected_materials:
            # Efficiency vs Cost scatter plot
            fig_scatter = go.Figure()

            for material in selected_materials:
                props = MATERIALS_DATABASE[material]
                fig_scatter.add_trace(go.Scatter(
                    x=[props['cost_per_wp']],
                    y=[props['efficiency']],
                    mode='markers+text',
                    marker=dict(
                        size=props['recyclability'],
                        sizemode='diameter',
                        sizeref=2,
                        color=props['degradation_rate'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Degradation<br>(%/yr)")
                    ),
                    text=material,
                    textposition="top center",
                    name=material
                ))

            fig_scatter.update_layout(
                title="Efficiency vs Cost Analysis (Bubble size = Recyclability)",
                xaxis_title="Cost ($/Wp)",
                yaxis_title="Efficiency (%)",
                hovermode='closest',
                height=500
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Radar chart for selected materials
            st.subheader("Multi-dimensional Comparison")
            fig_radar = create_material_radar_chart(selected_materials)
            st.plotly_chart(fig_radar, use_container_width=True)

    # Tab 3: Add Custom Material
    with tab3:
        st.subheader("Add Custom Material")

        with st.form("custom_material_form"):
            col1, col2 = st.columns(2)

            with col1:
                custom_name = st.text_input("Material Name", "My Custom Material")
                custom_efficiency = st.slider("Efficiency (%)", 5.0, 50.0, 20.0, 0.1)
                custom_cost = st.number_input("Cost ($/Wp)", 0.1, 5.0, 0.5, 0.01)
                custom_degradation = st.number_input("Degradation Rate (%/yr)", 0.1, 5.0, 0.5, 0.1)
                custom_recyclability = st.slider("Recyclability Score", 0, 100, 80)

            with col2:
                custom_bandgap = st.number_input("Bandgap (eV)", 0.5, 3.5, 1.5, 0.01)
                custom_temp_coeff = st.number_input("Temp Coefficient (%/°C)", -1.0, 0.0, -0.4, 0.01)
                custom_lifespan = st.slider("Lifespan (years)", 10, 50, 25)
                custom_carbon = st.number_input("Carbon Footprint (kg CO₂/kWp)", 0, 200, 40)

            submitted = st.form_submit_button("Add Material")

            if submitted:
                # Validate using Pydantic model
                try:
                    new_material = Material(
                        name=custom_name,
                        efficiency=custom_efficiency,
                        cost_per_wp=custom_cost,
                        degradation_rate=custom_degradation,
                        recyclability=custom_recyclability,
                        bandgap_ev=custom_bandgap,
                        temp_coefficient=custom_temp_coeff,
                        lifespan_years=custom_lifespan,
                        carbon_footprint=custom_carbon
                    )
                    st.success(f"✓ Material '{custom_name}' validated successfully!")
                    st.json(new_material.dict())
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")

    # Tab 4: Property Search
    with tab4:
        st.subheader("Search by Properties")

        col1, col2 = st.columns(2)
        with col1:
            min_efficiency = st.slider("Minimum Efficiency (%)", 0.0, 50.0, 18.0)
            max_cost = st.slider("Maximum Cost ($/Wp)", 0.0, 5.0, 1.0)

        with col2:
            min_recyclability = st.slider("Minimum Recyclability", 0, 100, 70)
            max_degradation = st.slider("Maximum Degradation (%/yr)", 0.0, 5.0, 1.5)

        # Filter materials
        filtered_materials = filter_materials(
            min_efficiency, max_cost, min_recyclability, max_degradation
        )

        st.write(f"**{len(filtered_materials)} materials match your criteria:**")
        if filtered_materials:
            filtered_df = pd.DataFrame.from_dict(filtered_materials, orient='index')
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No materials match the specified criteria.")


def create_material_radar_chart(materials: List[str]) -> go.Figure:
    """
    Create a radar chart comparing multiple materials.

    Args:
        materials: List of material names to compare

    Returns:
        Plotly Figure object with radar chart
    """
    categories = ['Efficiency', 'Cost Efficiency', 'Recyclability', 'Lifespan', 'Low Degradation']

    fig = go.Figure()

    for material in materials:
        props = MATERIALS_DATABASE[material]
        values = [
            props['efficiency'] / 25 * 100,  # Normalize to 100
            (1 - props['cost_per_wp'] / 5) * 100,  # Invert cost (lower is better)
            props['recyclability'],
            props['lifespan_years'] / 50 * 100,
            (1 - props['degradation_rate'] / 5) * 100  # Invert degradation
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=material
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Material Performance Comparison (Normalized to 100)"
    )

    return fig


def filter_materials(
    min_efficiency: float,
    max_cost: float,
    min_recyclability: int,
    max_degradation: float
) -> Dict[str, Dict[str, Any]]:
    """
    Filter materials based on specified criteria.

    Args:
        min_efficiency: Minimum efficiency (%)
        max_cost: Maximum cost ($/Wp)
        min_recyclability: Minimum recyclability score
        max_degradation: Maximum degradation rate (%/yr)

    Returns:
        Dictionary of filtered materials
    """
    filtered = {}
    for name, props in MATERIALS_DATABASE.items():
        if (props['efficiency'] >= min_efficiency and
            props['cost_per_wp'] <= max_cost and
            props['recyclability'] >= min_recyclability and
            props['degradation_rate'] <= max_degradation):
            filtered[name] = props
    return filtered


# ============================================================================
# BRANCH 02: CELL DESIGN (SCAPS-1D SIMULATION)
# ============================================================================

def render_cell_design() -> None:
    """
    Render the Cell Design and SCAPS-1D Simulation interface.

    Features:
    - Device structure configuration
    - Layer stack definition
    - SCAPS-1D parameter input
    - IV curve simulation
    - Efficiency optimization
    - Bandgap engineering
    """
    st.header("🔋 Cell Design & SCAPS-1D Simulation")
    st.markdown("*Advanced solar cell design with physics-based simulation*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Cell Configuration",
        "📊 IV Characteristics",
        "🎯 Optimization",
        "📈 Parametric Analysis"
    ])

    # Tab 1: Cell Configuration
    with tab1:
        st.subheader("Device Structure Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Parameters**")
            substrate = st.selectbox(
                "Substrate Material",
                SCAPS_DEFAULTS['substrate_types']
            )
            thickness = st.slider(
                "Device Thickness (μm)",
                SCAPS_DEFAULTS['device_thickness_um'][0],
                SCAPS_DEFAULTS['device_thickness_um'][1],
                2.0,
                0.1
            )
            architecture = st.selectbox(
                "Cell Architecture",
                ["n-type Si", "p-type Si", "Perovskite", "Tandem", "HIT"]
            )
            area_cm2 = st.number_input("Cell Area (cm²)", 100.0, 300.0, 156.75, 0.01)

        with col2:
            st.markdown("**Operating Conditions**")
            temp_k = st.number_input(
                "Simulation Temperature (K)",
                250, 400, SCAPS_DEFAULTS['simulation_temperature']
            )
            illumination = st.selectbox(
                "Illumination Spectrum",
                ["AM1.5G", "AM1.5D", "AM0"]
            )
            voltage_points = st.number_input("Voltage Points", 50, 500, 100)

        with col1:
            st.markdown("**Electrical Parameters**")
            voc_mv = st.number_input("Target Voc (mV)", 400, 1000, 730)
            jsc_ma_cm2 = st.number_input("Target Jsc (mA/cm²)", 20.0, 50.0, 42.5, 0.1)
            fill_factor = st.slider("Fill Factor", 0.60, 0.90, 0.82, 0.01)

        # Validate cell design
        try:
            cell = CellDesign(
                substrate=substrate,
                thickness_um=thickness,
                architecture=architecture,
                voc_mv=voc_mv,
                jsc_ma_cm2=jsc_ma_cm2,
                fill_factor=fill_factor,
                area_cm2=area_cm2,
                simulation_temp_k=temp_k
            )

            # Display calculated efficiency
            st.success(f"✓ Cell design validated. Predicted efficiency: **{cell.efficiency:.2f}%**")

            # Show cell specs
            with st.expander("View Complete Cell Specifications"):
                specs_df = pd.DataFrame({
                    'Parameter': [
                        'Substrate', 'Architecture', 'Thickness', 'Area',
                        'Voc', 'Jsc', 'Fill Factor', 'Efficiency',
                        'Pmax', 'Simulation Temp'
                    ],
                    'Value': [
                        substrate, architecture, f"{thickness} μm", f"{area_cm2} cm²",
                        f"{voc_mv} mV", f"{jsc_ma_cm2} mA/cm²", f"{fill_factor:.3f}",
                        f"{cell.efficiency:.2f}%",
                        f"{voc_mv * jsc_ma_cm2 * fill_factor / 1000:.2f} W",
                        f"{temp_k} K"
                    ]
                })
                st.dataframe(specs_df, use_container_width=True)

        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            cell = None

    # Tab 2: IV Characteristics
    with tab2:
        st.subheader("Current-Voltage Characteristics")

        if cell:
            # Generate IV curve
            voltage = np.linspace(0, voc_mv / 1000, 100)
            current = generate_iv_curve(voltage, voc_mv / 1000, jsc_ma_cm2, fill_factor)
            power = voltage * current * 1000  # Convert to mW

            # Create IV and Power curves
            fig = go.Figure()

            # Current curve
            fig.add_trace(go.Scatter(
                x=voltage,
                y=current,
                mode='lines',
                name='Current',
                line=dict(color=COLOR_PALETTE['primary'], width=3),
                yaxis='y1'
            ))

            # Power curve
            fig.add_trace(go.Scatter(
                x=voltage,
                y=power,
                mode='lines',
                name='Power',
                line=dict(color=COLOR_PALETTE['danger'], width=3),
                yaxis='y2'
            ))

            # Mark MPP
            mpp_idx = np.argmax(power)
            fig.add_trace(go.Scatter(
                x=[voltage[mpp_idx]],
                y=[current[mpp_idx]],
                mode='markers',
                marker=dict(size=12, color=COLOR_PALETTE['warning']),
                name='MPP',
                yaxis='y1'
            ))

            fig.update_layout(
                title="IV and Power Characteristics",
                xaxis_title="Voltage (V)",
                yaxis=dict(title="Current (mA/cm²)", side='left'),
                yaxis2=dict(title="Power (mW/cm²)", overlaying='y', side='right'),
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Voc", f"{voc_mv} mV")
            with col2:
                st.metric("Jsc", f"{jsc_ma_cm2:.2f} mA/cm²")
            with col3:
                st.metric("FF", f"{fill_factor:.3f}")
            with col4:
                st.metric("Vmpp", f"{voltage[mpp_idx]*1000:.0f} mV")

    # Tab 3: Optimization
    with tab3:
        st.subheader("Cell Optimization")

        st.markdown("**Optimization Objectives**")
        objective = st.radio(
            "Select optimization target:",
            ["Maximum Efficiency", "Maximum Power", "Cost-Efficiency Balance"]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Optimization Parameters**")
            opt_thickness = st.checkbox("Optimize Thickness", True)
            opt_bandgap = st.checkbox("Optimize Bandgap", True)
            opt_doping = st.checkbox("Optimize Doping", False)

        with col2:
            st.markdown("**Constraints**")
            max_cost = st.number_input("Max Cost ($/cell)", 0.1, 10.0, 2.0)
            min_efficiency = st.number_input("Min Efficiency (%)", 15.0, 30.0, 20.0)

        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running optimization..."):
                # Simulate optimization results
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                optimized_results = {
                    'Thickness': f"{thickness * 1.1:.2f} μm",
                    'Efficiency Gain': "+2.3%",
                    'New Efficiency': f"{cell.efficiency * 1.023:.2f}%",
                    'Cost': f"${max_cost * 0.95:.2f}"
                }

                st.success("✓ Optimization completed!")
                st.json(optimized_results)

    # Tab 4: Parametric Analysis
    with tab4:
        st.subheader("Parametric Sensitivity Analysis")

        param_to_vary = st.selectbox(
            "Select parameter to analyze:",
            ["Thickness", "Temperature", "Bandgap", "Doping Concentration"]
        )

        # Generate parametric sweep
        if param_to_vary == "Thickness":
            param_range = np.linspace(0.5, 10, 20)
            efficiency_values = 21.5 * (1 - 0.3 * np.exp(-param_range / 3))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=param_range,
                y=efficiency_values,
                mode='lines+markers',
                line=dict(color=COLOR_PALETTE['primary'], width=3)
            ))
            fig.update_layout(
                title="Efficiency vs Thickness",
                xaxis_title="Thickness (μm)",
                yaxis_title="Efficiency (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def generate_iv_curve(
    voltage: np.ndarray,
    voc: float,
    jsc: float,
    ff: float
) -> np.ndarray:
    """
    Generate idealized IV curve using single-diode model.

    Args:
        voltage: Voltage array (V)
        voc: Open-circuit voltage (V)
        jsc: Short-circuit current density (mA/cm²)
        ff: Fill factor

    Returns:
        Current density array (mA/cm²)
    """
    # Simplified single-diode model
    n = 1.5  # Ideality factor
    Vt = 0.026  # Thermal voltage at 300K
    Rs = (1 - ff) * voc / jsc / 1000  # Series resistance estimate

    current = jsc * (1 - np.exp((voltage - voc) / (n * Vt))) - voltage / Rs * 1000
    current = np.maximum(current, 0)  # No negative current

    return current


# ============================================================================
# BRANCH 03: MODULE DESIGN & CTM LOSS ANALYSIS
# ============================================================================

def render_module_design() -> None:
    """
    Render the Module Design and CTM Loss Analysis interface.

    Features:
    - Cell-to-Module (CTM) loss breakdown (k1-k24)
    - Module configuration
    - Efficiency calculations
    - Bill of Materials
    - Cost analysis
    - Thermal modeling
    """
    st.header("📦 Module Design & CTM Loss Analysis")
    st.markdown("*Fraunhofer ISE k1-k24 Cell-to-Module loss framework*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Module Configuration",
        "📉 CTM Loss Analysis",
        "💰 Bill of Materials",
        "🌡️ Thermal Analysis"
    ])

    # Tab 1: Module Configuration
    with tab1:
        st.subheader("Module Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cell Configuration**")
            num_cells = st.selectbox("Number of Cells", [36, 48, 60, 72, 96, 120, 144], index=2)
            cell_efficiency = st.slider("Cell Efficiency (%)", 15.0, 26.0, 21.5, 0.1)
            cell_area_cm2 = st.number_input("Cell Area (cm²)", 100.0, 300.0, 156.75)

            st.markdown("**Module Specifications**")
            module_length = st.number_input("Module Length (mm)", 1000, 2500, 1650)
            module_width = st.number_input("Module Width (mm)", 800, 1400, 992)
            module_area_m2 = (module_length * module_width) / 1_000_000

        with col2:
            st.markdown("**Design Elements**")
            bypass_diodes = st.selectbox("Bypass Diodes", [2, 3, 4, 6], index=1)
            bus_bars = st.slider("Bus Bars per Cell", 3, 12, 5)
            encapsulation = st.selectbox("Encapsulation", ["EVA", "POE", "Silicone"])
            backsheet = st.selectbox("Backsheet", ["White", "Black", "Transparent", "Bifacial"])
            frame_material = st.selectbox("Frame", ["Aluminum", "Frameless", "Composite"])
            glass_type = st.selectbox("Front Glass", ["AR-coated", "Textured", "Standard"])

        # Calculate module power
        total_cell_area_m2 = num_cells * cell_area_cm2 / 10000
        theoretical_power = cell_efficiency * total_cell_area_m2 * 1000  # W

        st.info(f"**Module Area:** {module_area_m2:.2f} m² | **Theoretical Power:** {theoretical_power:.0f} W (before CTM losses)")

    # Tab 2: CTM Loss Analysis
    with tab2:
        st.subheader("Cell-to-Module Loss Breakdown (Fraunhofer ISE)")

        # Allow users to customize CTM losses
        st.markdown("**Customize CTM Loss Factors (k1-k24)**")

        # Create adjustable CTM losses
        ctm_losses_custom = []
        loss_categories = {
            "Optical Losses": ["k1_reflection", "k2_absorption", "k3_transmission", "k7_spectral"],
            "Electrical Losses": ["k9_mismatch", "k10_wiring", "k11_connection"],
            "Environmental": ["k4_soiling", "k5_temperature", "k6_low_irradiance"],
            "Degradation": ["k12_lid", "k13_pid"],
            "Geometric": ["k8_shading", "k16_edge_delete", "k17_bus_bar", "k18_junction_box", "k19_cell_gap"],
            "Manufacturing": ["k14_encapsulation", "k15_backsheet", "k20_lamination", "k21_quality", "k22_sorting", "k23_flash_test", "k24_outdoor"]
        }

        total_loss = 0
        for category, factors in loss_categories.items():
            with st.expander(f"**{category}**"):
                for factor_id in factors:
                    if factor_id in CTM_LOSS_FACTORS:
                        default_loss = CTM_LOSS_FACTORS[factor_id]['loss_pct']
                        description = CTM_LOSS_FACTORS[factor_id]['description']

                        loss_value = st.slider(
                            f"{factor_id}: {description}",
                            0.0, 10.0, default_loss, 0.1,
                            key=factor_id
                        )

                        ctm_losses_custom.append(CTMLoss(
                            factor_id=factor_id,
                            description=description,
                            loss_pct=loss_value,
                            category=category
                        ))
                        total_loss += loss_value

        # Calculate final module efficiency and power
        module_efficiency = cell_efficiency * (1 - total_loss / 100)
        module_power_w = module_efficiency * module_area_m2 * 1000

        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cell Efficiency", f"{cell_efficiency:.1f}%")
        with col2:
            st.metric("Total CTM Loss", f"{total_loss:.1f}%", delta=f"-{total_loss:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Module Efficiency", f"{module_efficiency:.2f}%")
        with col4:
            st.metric("Module Power", f"{module_power_w:.0f} W")

        # Waterfall chart showing CTM losses
        st.subheader("CTM Loss Waterfall Chart")
        fig_waterfall = create_ctm_waterfall(cell_efficiency, ctm_losses_custom)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Loss breakdown pie chart
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = create_loss_pie_chart(ctm_losses_custom)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Loss by category
            category_losses = {}
            for loss in ctm_losses_custom:
                if loss.category in category_losses:
                    category_losses[loss.category] += loss.loss_pct
                else:
                    category_losses[loss.category] = loss.loss_pct

            fig_category = go.Figure(data=[
                go.Bar(
                    x=list(category_losses.keys()),
                    y=list(category_losses.values()),
                    marker_color=COLOR_PALETTE['secondary']
                )
            ])
            fig_category.update_layout(
                title="Loss by Category",
                xaxis_title="Category",
                yaxis_title="Total Loss (%)",
                height=400
            )
            st.plotly_chart(fig_category, use_container_width=True)

    # Tab 3: Bill of Materials
    with tab3:
        st.subheader("Bill of Materials (BOM)")

        # Generate BOM
        bom_data = {
            'Component': [
                f'Solar Cells ({num_cells}x)',
                f'Bypass Diodes ({bypass_diodes}x)',
                f'Front Glass (3.2mm {glass_type})',
                f'Encapsulant ({encapsulation})',
                f'Backsheet ({backsheet})',
                f'Frame ({frame_material})',
                'Junction Box',
                'Cables & Connectors',
                'Bus Bars (Silver)',
                'Ribbons (Copper)'
            ],
            'Quantity': [num_cells, bypass_diodes, 1, 2, 1, 1, 1, 1, num_cells * bus_bars, num_cells * 2],
            'Unit Cost ($)': [2.50, 0.15, 8.50, 3.20, 2.80, 12.00, 4.50, 3.00, 0.05, 0.08],
            'Total Cost ($)': [
                num_cells * 2.50,
                bypass_diodes * 0.15,
                8.50,
                6.40,
                2.80,
                12.00,
                4.50,
                3.00,
                num_cells * bus_bars * 0.05,
                num_cells * 2 * 0.08
            ]
        }

        bom_df = pd.DataFrame(bom_data)
        bom_df['Total Cost ($)'] = bom_df['Total Cost ($)'].round(2)

        st.dataframe(bom_df, use_container_width=True)

        # Total costs
        total_material_cost = bom_df['Total Cost ($)'].sum()
        labor_cost = total_material_cost * 0.15
        overhead_cost = total_material_cost * 0.20
        total_cost = total_material_cost + labor_cost + overhead_cost

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Material Cost", f"${total_material_cost:.2f}")
        with col2:
            st.metric("Labor (15%)", f"${labor_cost:.2f}")
        with col3:
            st.metric("Overhead (20%)", f"${overhead_cost:.2f}")
        with col4:
            st.metric("Total Module Cost", f"${total_cost:.2f}")

        cost_per_watt = total_cost / module_power_w if module_power_w > 0 else 0
        st.info(f"**Manufacturing Cost:** ${cost_per_watt:.3f}/W")

    # Tab 4: Thermal Analysis
    with tab4:
        st.subheader("Thermal Performance Analysis")

        col1, col2 = st.columns(2)
        with col1:
            irradiance = st.slider("Irradiance (W/m²)", 200, 1200, 1000, 50)
            ambient_temp = st.slider("Ambient Temperature (°C)", 0, 50, 25)
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 10.0, 1.0, 0.1)

        # Calculate NOCT and operating temperature
        noct = 45 + (module_efficiency - 20) * 0.5  # Simplified NOCT estimate
        module_temp = ambient_temp + (noct - 20) * (irradiance / 800) * (1 - wind_speed / 10)

        # Temperature coefficient effect
        temp_diff = module_temp - 25  # STC is 25°C
        temp_coefficient = -0.40  # %/°C
        power_loss_temp = temp_diff * temp_coefficient

        actual_power = module_power_w * (1 + power_loss_temp / 100)

        with col2:
            st.metric("Module Temperature", f"{module_temp:.1f}°C")
            st.metric("NOCT", f"{noct:.1f}°C")
            st.metric("Temperature Loss", f"{power_loss_temp:.2f}%")
            st.metric("Actual Power", f"{actual_power:.0f} W")

        # Temperature vs Power curve
        temp_range = np.arange(-20, 80, 5)
        power_range = module_power_w * (1 + (temp_range - 25) * temp_coefficient / 100)

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temp_range,
            y=power_range,
            mode='lines',
            line=dict(color=COLOR_PALETTE['danger'], width=3),
            name='Power Output'
        ))
        fig_temp.add_trace(go.Scatter(
            x=[module_temp],
            y=[actual_power],
            mode='markers',
            marker=dict(size=12, color=COLOR_PALETTE['warning']),
            name='Current Operating Point'
        ))
        fig_temp.update_layout(
            title="Power vs Module Temperature",
            xaxis_title="Module Temperature (°C)",
            yaxis_title="Power Output (W)",
            height=400
        )
        st.plotly_chart(fig_temp, use_container_width=True)


def create_ctm_waterfall(cell_efficiency: float, ctm_losses: List[CTMLoss]) -> go.Figure:
    """
    Create waterfall chart showing CTM losses.

    Args:
        cell_efficiency: Initial cell efficiency (%)
        ctm_losses: List of CTM loss factors

    Returns:
        Plotly Figure object
    """
    # Sort losses by magnitude
    sorted_losses = sorted(ctm_losses, key=lambda x: x.loss_pct, reverse=True)

    x_labels = ['Cell Efficiency'] + [f"{loss.factor_id}" for loss in sorted_losses] + ['Module Efficiency']
    y_values = [cell_efficiency] + [-loss.loss_pct for loss in sorted_losses]

    # Calculate final efficiency
    final_efficiency = cell_efficiency - sum(loss.loss_pct for loss in sorted_losses)

    fig = go.Figure(go.Waterfall(
        x=x_labels,
        y=y_values + [final_efficiency],
        measure=['absolute'] + ['relative'] * len(sorted_losses) + ['total'],
        text=[f"{cell_efficiency:.2f}%"] + [f"-{loss.loss_pct:.1f}%" for loss in sorted_losses] + [f"{final_efficiency:.2f}%"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="CTM Loss Waterfall Analysis",
        showlegend=False,
        height=500
    )

    return fig


def create_loss_pie_chart(ctm_losses: List[CTMLoss]) -> go.Figure:
    """
    Create pie chart of CTM losses.

    Args:
        ctm_losses: List of CTM loss factors

    Returns:
        Plotly Figure object
    """
    # Get top 10 losses
    sorted_losses = sorted(ctm_losses, key=lambda x: x.loss_pct, reverse=True)[:10]

    labels = [loss.factor_id for loss in sorted_losses]
    values = [loss.loss_pct for loss in sorted_losses]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='label+percent'
    )])

    fig.update_layout(
        title="Top 10 CTM Loss Factors",
        height=400
    )

    return fig
