"""
Module Design & CTM Loss Analysis Module (Branch B03).

Features:
- Module specification and design
- CTM loss factors k1-k24 (Fraunhofer ISE standard)
- Cell-to-module efficiency calculations
- Module layout and optimization
- Encapsulation and materials selection
- Thermal analysis and NOCT calculations
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constants import (
    MATERIAL_PROPERTIES,
    CTM_LOSS_FACTORS,
    MODULE_COMPOSITION,
    MATERIAL_COLORS,
    STANDARD_TEMPERATURE,
    STANDARD_IRRADIANCE
)
from utils.validators import CTMLossFactors, ModuleSpecification
from utils.helpers import temperature_corrected_power, calculate_noct_temperature


class ModuleDesigner:
    """Module design and CTM loss analysis."""

    def __init__(self):
        """Initialize module designer."""
        self.encapsulation_types = {
            "glass_glass": {
                "name": "Glass-Glass",
                "front": "tempered_glass",
                "back": "tempered_glass",
                "weight_factor": 1.3,
                "cost_factor": 1.2,
                "bifacial": True,
                "durability": "excellent"
            },
            "glass_backsheet": {
                "name": "Glass-Backsheet",
                "front": "tempered_glass",
                "back": "polymer_backsheet",
                "weight_factor": 1.0,
                "cost_factor": 1.0,
                "bifacial": False,
                "durability": "good"
            },
            "glass_transparent": {
                "name": "Glass-Transparent Backsheet",
                "front": "tempered_glass",
                "back": "transparent_backsheet",
                "weight_factor": 1.05,
                "cost_factor": 1.1,
                "bifacial": True,
                "durability": "very_good"
            }
        }

        self.junction_box_types = {
            "standard": {"bypass_diodes": 3, "cost": 2.5, "ip_rating": "IP67"},
            "smart": {"bypass_diodes": 3, "cost": 8.0, "ip_rating": "IP68", "monitoring": True},
            "integrated": {"bypass_diodes": 6, "cost": 12.0, "ip_rating": "IP68", "power_optimizer": True}
        }

    def calculate_ctm_efficiency(
        self,
        cell_efficiency: float,
        ctm_losses: CTMLossFactors
    ) -> Dict[str, float]:
        """
        Calculate cell-to-module efficiency with detailed loss breakdown.

        Args:
            cell_efficiency: Cell efficiency (%)
            ctm_losses: CTM loss factors

        Returns:
            Dictionary with module efficiency and loss breakdown
        """
        # Calculate total CTM loss
        total_loss_percent = ctm_losses.total_loss()
        efficiency_factor = ctm_losses.efficiency_factor()

        # Module efficiency
        module_efficiency = cell_efficiency * efficiency_factor

        # Calculate individual loss impacts
        loss_breakdown = {}
        for field in ctm_losses.model_fields:
            if field.startswith('k'):
                loss_value = getattr(ctm_losses, field)
                loss_impact = cell_efficiency * (loss_value / 100)
                loss_name = field.split('_', 1)[1] if '_' in field else field
                loss_breakdown[loss_name] = loss_impact

        return {
            "cell_efficiency": cell_efficiency,
            "module_efficiency": module_efficiency,
            "total_ctm_loss_percent": total_loss_percent,
            "efficiency_factor": efficiency_factor,
            "ctm_ratio": module_efficiency / cell_efficiency if cell_efficiency > 0 else 0,
            "loss_breakdown": loss_breakdown
        }

    def design_module_layout(
        self,
        cell_size: float,
        num_cells: int,
        configuration: str = "6x12"
    ) -> Dict:
        """
        Design module physical layout.

        Args:
            cell_size: Cell edge length (mm)
            num_cells: Number of cells
            configuration: Cell configuration (e.g., "6x12", "6x10")

        Returns:
            Dictionary with layout specifications
        """
        # Parse configuration
        rows, cols = map(int, configuration.split('x'))

        # Validate cell count
        if rows * cols != num_cells:
            rows = int(np.sqrt(num_cells))
            cols = num_cells // rows

        # Cell spacing (mm)
        cell_spacing = 2.0  # Typical 2mm spacing

        # Module dimensions
        module_width = cols * cell_size + (cols + 1) * cell_spacing
        module_height = rows * cell_size + (rows + 1) * cell_spacing

        # Add frame
        frame_width = 35.0  # mm
        total_width = module_width + 2 * frame_width
        total_height = module_height + 2 * frame_width

        # Active area
        active_area = (cell_size ** 2) * num_cells / 1e6  # m²
        total_area = (total_width * total_height) / 1e6  # m²
        packing_factor = active_area / total_area

        return {
            "num_cells": num_cells,
            "rows": rows,
            "columns": cols,
            "cell_size": cell_size,
            "cell_spacing": cell_spacing,
            "module_width": total_width,
            "module_height": total_height,
            "active_area_m2": active_area,
            "total_area_m2": total_area,
            "packing_factor": packing_factor,
            "configuration": f"{rows}x{cols}"
        }

    def calculate_module_power(
        self,
        cell_efficiency: float,
        module_efficiency: float,
        module_area: float,
        irradiance: float = STANDARD_IRRADIANCE
    ) -> Dict[str, float]:
        """
        Calculate module power output.

        Args:
            cell_efficiency: Cell efficiency (%)
            module_efficiency: Module efficiency (%)
            module_area: Module area (m²)
            irradiance: Irradiance (W/m²)

        Returns:
            Dictionary with power calculations
        """
        # Power at cell level
        cell_power = (cell_efficiency / 100) * module_area * irradiance

        # Power at module level
        module_power = (module_efficiency / 100) * module_area * irradiance

        # Power loss
        power_loss = cell_power - module_power
        power_loss_percent = (power_loss / cell_power * 100) if cell_power > 0 else 0

        return {
            "cell_power_stc": cell_power,
            "module_power_stc": module_power,
            "power_loss": power_loss,
            "power_loss_percent": power_loss_percent,
            "watts_per_m2": module_power / module_area if module_area > 0 else 0
        }

    def calculate_temperature_coefficients(
        self,
        cell_temp_coeff_pmax: float,
        ctm_temp_losses: float = 3.2
    ) -> Dict[str, float]:
        """
        Calculate module temperature coefficients.

        Args:
            cell_temp_coeff_pmax: Cell temp coefficient of Pmax (%/°C)
            ctm_temp_losses: CTM temperature-related losses (%)

        Returns:
            Module temperature coefficients
        """
        # Module temperature coefficient is worse than cell due to CTM losses
        module_temp_coeff_pmax = cell_temp_coeff_pmax - (ctm_temp_losses / 100)

        # Typical relationships
        module_temp_coeff_voc = module_temp_coeff_pmax * 0.8  # Voc less sensitive
        module_temp_coeff_isc = abs(module_temp_coeff_pmax) * 0.1  # Isc slightly positive

        return {
            "temp_coeff_pmax": module_temp_coeff_pmax,
            "temp_coeff_voc": module_temp_coeff_voc,
            "temp_coeff_isc": module_temp_coeff_isc
        }

    def calculate_thermal_performance(
        self,
        ambient_temp: float,
        irradiance: float,
        wind_speed: float = 1.0,
        mounting_type: str = "rooftop"
    ) -> Dict[str, float]:
        """
        Calculate module thermal performance and NOCT.

        Args:
            ambient_temp: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            mounting_type: Mounting configuration

        Returns:
            Thermal performance metrics
        """
        # NOCT values by mounting type
        noct_base = {
            "rooftop": 47.0,
            "ground_mount": 45.0,
            "facade": 50.0,
            "tracking": 44.0
        }

        noct = noct_base.get(mounting_type, 45.0)

        # Wind cooling factor
        wind_factor = 1.0 - 0.02 * min(wind_speed, 10.0)  # Up to 20% reduction at 10 m/s

        # Adjusted NOCT
        noct_adjusted = noct * wind_factor

        # Module temperature
        module_temp = calculate_noct_temperature(
            ambient_temp=ambient_temp,
            irradiance=irradiance,
            noct=noct_adjusted,
            irradiance_noct=800.0
        )

        # Temperature rise
        temp_rise = module_temp - ambient_temp

        # Thermal resistance (K·m²/W)
        thermal_resistance = temp_rise / irradiance if irradiance > 0 else 0

        return {
            "module_temperature": module_temp,
            "ambient_temperature": ambient_temp,
            "temperature_rise": temp_rise,
            "noct": noct_adjusted,
            "thermal_resistance": thermal_resistance,
            "wind_speed": wind_speed
        }

    def calculate_module_weight(
        self,
        module_area: float,
        encapsulation_type: str = "glass_backsheet"
    ) -> Dict[str, float]:
        """
        Calculate module weight based on construction.

        Args:
            module_area: Module area (m²)
            encapsulation_type: Encapsulation type

        Returns:
            Weight breakdown
        """
        encap = self.encapsulation_types.get(encapsulation_type,
                                            self.encapsulation_types["glass_backsheet"])

        # Base weights (kg/m²)
        component_weights = {
            "front_glass": 10.0,  # 3.2mm glass
            "cells": 1.2,
            "eva": 0.8,
            "backsheet": 0.3 if not encap["bifacial"] else 10.0,
            "frame": 2.5,
            "junction_box": 0.3
        }

        # Calculate total
        total_weight_per_m2 = sum(component_weights.values()) * encap["weight_factor"]
        total_weight = total_weight_per_m2 * module_area

        # Component breakdown
        component_breakdown = {
            name: weight * module_area * encap["weight_factor"]
            for name, weight in component_weights.items()
        }

        return {
            "total_weight_kg": total_weight,
            "weight_per_m2": total_weight_per_m2,
            "component_breakdown": component_breakdown,
            "encapsulation_type": encapsulation_type
        }

    def optimize_string_configuration(
        self,
        module_voc: float,
        module_vmp: float,
        inverter_voltage_range: Tuple[float, float],
        max_string_length: int = 30
    ) -> Dict:
        """
        Optimize string configuration for inverter compatibility.

        Args:
            module_voc: Module open-circuit voltage (V)
            module_vmp: Module maximum power voltage (V)
            inverter_voltage_range: Inverter voltage range (V_min, V_max)
            max_string_length: Maximum modules per string

        Returns:
            Optimized configuration
        """
        v_min, v_max = inverter_voltage_range

        # Temperature correction factors (worst case)
        voc_cold = module_voc * 1.12  # -10°C
        vmp_cold = module_vmp * 1.08
        vmp_hot = module_vmp * 0.85   # +70°C

        # Calculate string lengths
        max_modules_by_voc = int(v_max / voc_cold)
        min_modules_by_vmp = int(v_min / vmp_hot) + 1

        # Optimal configuration
        optimal_modules = min(max_modules_by_voc, max_string_length)
        optimal_modules = max(optimal_modules, min_modules_by_vmp)

        string_voc = module_voc * optimal_modules
        string_vmp = module_vmp * optimal_modules
        string_voc_cold = voc_cold * optimal_modules
        string_vmp_cold = vmp_cold * optimal_modules
        string_vmp_hot = vmp_hot * optimal_modules

        return {
            "modules_per_string": optimal_modules,
            "string_voc_stc": string_voc,
            "string_vmp_stc": string_vmp,
            "string_voc_cold": string_voc_cold,
            "string_vmp_cold": string_vmp_cold,
            "string_vmp_hot": string_vmp_hot,
            "min_modules": min_modules_by_vmp,
            "max_modules": max_modules_by_voc,
            "voltage_margin_min": (string_vmp_hot - v_min) / v_min * 100,
            "voltage_margin_max": (v_max - string_voc_cold) / v_max * 100
        }

    def create_ctm_loss_waterfall(
        self,
        cell_efficiency: float,
        ctm_losses: CTMLossFactors
    ) -> go.Figure:
        """
        Create waterfall chart showing CTM losses.

        Args:
            cell_efficiency: Cell efficiency (%)
            ctm_losses: CTM loss factors

        Returns:
            Plotly waterfall figure
        """
        # Get loss breakdown
        result = self.calculate_ctm_efficiency(cell_efficiency, ctm_losses)
        loss_breakdown = result["loss_breakdown"]

        # Sort losses by magnitude
        sorted_losses = sorted(
            loss_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 10 losses
        top_losses = sorted_losses[:10]

        # Create waterfall data
        x_labels = ["Cell Efficiency"]
        y_values = [cell_efficiency]
        measures = ["absolute"]

        current_value = cell_efficiency
        for loss_name, loss_impact in top_losses:
            x_labels.append(loss_name.replace('_', ' ').title())
            y_values.append(-loss_impact)
            measures.append("relative")
            current_value -= loss_impact

        # Add final module efficiency
        x_labels.append("Module Efficiency")
        y_values.append(result["module_efficiency"])
        measures.append("total")

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Waterfall(
            name="CTM Losses",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=[f"{abs(v):.2f}%" for v in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#E74C3C"}},
            increasing={"marker": {"color": "#2ECC71"}},
            totals={"marker": {"color": "#3498DB"}}
        ))

        fig.update_layout(
            title="Cell-to-Module Efficiency Loss Waterfall",
            xaxis_title="Component",
            yaxis_title="Efficiency (%)",
            showlegend=False,
            height=500,
            template='plotly_white'
        )

        return fig

    def create_loss_breakdown_pie(
        self,
        cell_efficiency: float,
        ctm_losses: CTMLossFactors
    ) -> go.Figure:
        """
        Create pie chart of CTM loss breakdown.

        Args:
            cell_efficiency: Cell efficiency (%)
            ctm_losses: CTM loss factors

        Returns:
            Plotly pie chart
        """
        result = self.calculate_ctm_efficiency(cell_efficiency, ctm_losses)
        loss_breakdown = result["loss_breakdown"]

        # Categorize losses
        categories = {
            "Optical Losses": ["reflection", "glass_absorption", "optical", "iam"],
            "Electrical Losses": ["series_resistance", "mismatch", "wiring", "interconnection"],
            "Thermal Losses": ["temperature", "temp_non_uniformity"],
            "Shading Losses": ["ribbon_shading", "busbar_shading", "cell_spacing", "shading"],
            "Degradation": ["degradation", "lid", "pid"],
            "Other": ["soiling", "spectral", "encapsulation", "backsheet", "quality",
                     "manufacturing", "other"]
        }

        category_totals = {}
        for category, loss_names in categories.items():
            total = sum(
                loss_breakdown.get(name, 0)
                for name in loss_names
            )
            if total > 0:
                category_totals[category] = total

        # Create pie chart
        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=list(category_totals.keys()),
            values=list(category_totals.values()),
            hole=0.4,
            marker=dict(colors=['#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#E67E22', '#95A5A6']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Loss: %{value:.3f}%<br>Share: %{percent}<extra></extra>'
        ))

        fig.update_layout(
            title="CTM Loss Breakdown by Category",
            height=500,
            template='plotly_white',
            annotations=[dict(
                text=f'{result["total_ctm_loss_percent"]:.1f}%<br>Total Loss',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )

        return fig

    def create_thermal_heatmap(
        self,
        ambient_temps: np.ndarray,
        irradiances: np.ndarray,
        mounting_type: str = "rooftop"
    ) -> go.Figure:
        """
        Create heatmap of module temperature vs conditions.

        Args:
            ambient_temps: Array of ambient temperatures (°C)
            irradiances: Array of irradiances (W/m²)
            mounting_type: Mounting type

        Returns:
            Plotly heatmap
        """
        # Calculate module temperatures
        module_temps = np.zeros((len(ambient_temps), len(irradiances)))

        for i, t_amb in enumerate(ambient_temps):
            for j, irr in enumerate(irradiances):
                thermal_result = self.calculate_thermal_performance(
                    ambient_temp=t_amb,
                    irradiance=irr,
                    mounting_type=mounting_type
                )
                module_temps[i, j] = thermal_result["module_temperature"]

        # Create heatmap
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=module_temps,
            x=irradiances,
            y=ambient_temps,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Module<br>Temp (°C)"),
            hovertemplate='Irradiance: %{x} W/m²<br>Ambient: %{y}°C<br>Module: %{z:.1f}°C<extra></extra>'
        ))

        fig.update_layout(
            title=f"Module Temperature Map - {mounting_type.replace('_', ' ').title()}",
            xaxis_title="Irradiance (W/m²)",
            yaxis_title="Ambient Temperature (°C)",
            height=500,
            template='plotly_white'
        )

        return fig


def render_module_design():
    """Render module design interface in Streamlit."""
    st.header("📦 Module Design & CTM Loss Analysis")
    st.markdown("Advanced module design with cell-to-module efficiency analysis and optimization.")

    designer = ModuleDesigner()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Module Design",
        "📉 CTM Loss Analysis",
        "🌡️ Thermal Analysis",
        "⚡ String Configuration",
        "📊 Performance Summary"
    ])

    with tab1:
        st.subheader("Module Design Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cell Configuration**")

            cell_efficiency = st.slider(
                "Cell Efficiency (%):",
                min_value=15.0,
                max_value=28.0,
                value=23.5,
                step=0.1
            )

            cell_size = st.selectbox(
                "Cell Size (mm):",
                [156.75, 161.7, 166.0, 182.0, 210.0],
                index=0
            )

            num_cells = st.selectbox(
                "Number of Cells:",
                [60, 72, 78, 84, 96, 108, 120, 132, 144],
                index=1
            )

            configuration = st.selectbox(
                "Cell Layout:",
                ["6x10", "6x12", "6x13", "6x14", "6x16", "6x18", "6x20", "6x22", "6x24"],
                index=1
            )

        with col2:
            st.write("**Module Construction**")

            encapsulation = st.selectbox(
                "Encapsulation Type:",
                list(designer.encapsulation_types.keys()),
                format_func=lambda x: designer.encapsulation_types[x]["name"]
            )

            junction_box = st.selectbox(
                "Junction Box:",
                list(designer.junction_box_types.keys()),
                format_func=lambda x: x.title()
            )

            frame_type = st.selectbox(
                "Frame:",
                ["Aluminum Anodized", "Aluminum Black", "Frameless"],
                index=0
            )

            bifacial = designer.encapsulation_types[encapsulation]["bifacial"]
            if bifacial:
                bifaciality = st.slider(
                    "Bifaciality Factor:",
                    min_value=0.50,
                    max_value=0.95,
                    value=0.70,
                    step=0.05
                )

        # Calculate layout
        layout = designer.design_module_layout(cell_size, num_cells, configuration)

        st.subheader("Module Specifications")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Dimensions", f"{layout['module_width']:.0f} × {layout['module_height']:.0f} mm")
            st.metric("Configuration", layout['configuration'])

        with col2:
            st.metric("Total Area", f"{layout['total_area_m2']:.3f} m²")
            st.metric("Active Area", f"{layout['active_area_m2']:.3f} m²")

        with col3:
            st.metric("Packing Factor", f"{layout['packing_factor']:.1%}")
            weight = designer.calculate_module_weight(layout['total_area_m2'], encapsulation)
            st.metric("Weight", f"{weight['total_weight_kg']:.1f} kg")

        with col4:
            st.metric("Cell Spacing", f"{layout['cell_spacing']:.1f} mm")
            st.metric("Weight/Area", f"{weight['weight_per_m2']:.1f} kg/m²")

    with tab2:
        st.subheader("CTM Loss Factor Configuration")
        st.markdown("Configure Cell-to-Module loss factors (Fraunhofer ISE standard k1-k24)")

        # CTM Loss input
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Optical Losses**")
            k1 = st.slider("k1: Reflection (%)", 0.0, 10.0, 2.5, 0.1)
            k11 = st.slider("k11: Glass Absorption (%)", 0.0, 5.0, 1.5, 0.1)
            k22 = st.slider("k22: Optical (%)", 0.0, 5.0, 1.5, 0.1)
            k9 = st.slider("k9: IAM (%)", 0.0, 10.0, 2.0, 0.1)

            st.write("**Electrical Losses**")
            k4 = st.slider("k4: Series Resistance (%)", 0.0, 5.0, 2.1, 0.1)
            k5 = st.slider("k5: Mismatch (%)", 0.0, 5.0, 1.5, 0.1)
            k6 = st.slider("k6: Wiring (%)", 0.0, 5.0, 0.8, 0.1)
            k23 = st.slider("k23: Interconnection (%)", 0.0, 3.0, 1.0, 0.1)

        with col2:
            st.write("**Thermal Losses**")
            k3 = st.slider("k3: Temperature (%)", 0.0, 15.0, 3.2, 0.1)
            k16 = st.slider("k16: Temp Non-uniformity (%)", 0.0, 5.0, 1.0, 0.1)

            st.write("**Shading Losses**")
            k7 = st.slider("k7: External Shading (%)", 0.0, 20.0, 0.0, 0.5)
            k13 = st.slider("k13: Cell Spacing (%)", 0.0, 10.0, 3.0, 0.1)
            k14 = st.slider("k14: Ribbon Shading (%)", 0.0, 5.0, 2.0, 0.1)
            k15 = st.slider("k15: Busbar Shading (%)", 0.0, 5.0, 2.5, 0.1)

            st.write("**Degradation**")
            k17 = st.slider("k17: Degradation (%)", 0.0, 5.0, 1.5, 0.1)
            k18 = st.slider("k18: LID (%)", 0.0, 10.0, 2.0, 0.1)
            k19 = st.slider("k19: PID (%)", 0.0, 20.0, 0.0, 0.5)

        with col3:
            st.write("**Environmental**")
            k2 = st.slider("k2: Soiling (%)", 0.0, 10.0, 1.8, 0.1)
            k8 = st.slider("k8: Spectral (%)", 0.0, 5.0, 1.5, 0.1)

            st.write("**Manufacturing**")
            k10 = st.slider("k10: Encapsulation (%)", 0.0, 5.0, 1.0, 0.1)
            k12 = st.slider("k12: Backsheet (%)", 0.0, 3.0, 0.8, 0.1)
            k20 = st.slider("k20: Quality (%)", 0.0, 5.0, 1.0, 0.1)
            k21 = st.slider("k21: Manufacturing (%)", 0.0, 3.0, 1.0, 0.1)
            k24 = st.slider("k24: Other (%)", 0.0, 5.0, 1.0, 0.1)

        # Create CTM loss object
        ctm_losses = CTMLossFactors(
            k1_reflection=k1, k2_soiling=k2, k3_temperature=k3, k4_series_resistance=k4,
            k5_mismatch=k5, k6_wiring=k6, k7_shading=k7, k8_spectral=k8, k9_iam=k9,
            k10_encapsulation=k10, k11_glass_absorption=k11, k12_backsheet=k12,
            k13_cell_spacing=k13, k14_ribbon_shading=k14, k15_busbar_shading=k15,
            k16_temp_non_uniformity=k16, k17_degradation=k17, k18_lid=k18, k19_pid=k19,
            k20_quality=k20, k21_manufacturing=k21, k22_optical=k22,
            k23_interconnection=k23, k24_other=k24
        )

        # Calculate CTM efficiency
        ctm_result = designer.calculate_ctm_efficiency(cell_efficiency, ctm_losses)

        st.subheader("CTM Efficiency Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Cell Efficiency", f"{ctm_result['cell_efficiency']:.2f}%")

        with col2:
            st.metric("Module Efficiency", f"{ctm_result['module_efficiency']:.2f}%")

        with col3:
            st.metric("Total CTM Loss", f"{ctm_result['total_ctm_loss_percent']:.2f}%")

        with col4:
            st.metric("CTM Ratio", f"{ctm_result['ctm_ratio']:.3f}")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            fig_waterfall = designer.create_ctm_loss_waterfall(cell_efficiency, ctm_losses)
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with col2:
            fig_pie = designer.create_loss_breakdown_pie(cell_efficiency, ctm_losses)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Calculate module power
        power_result = designer.calculate_module_power(
            cell_efficiency,
            ctm_result['module_efficiency'],
            layout['total_area_m2']
        )

        st.subheader("Power Output (STC)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Module Power", f"{power_result['module_power_stc']:.1f} W")

        with col2:
            st.metric("Cell Power", f"{power_result['cell_power_stc']:.1f} W")

        with col3:
            st.metric("Power Loss", f"{power_result['power_loss']:.1f} W")

        with col4:
            st.metric("Power Density", f"{power_result['watts_per_m2']:.0f} W/m²")

    with tab3:
        st.subheader("Thermal Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            ambient_temp = st.slider(
                "Ambient Temperature (°C):",
                min_value=-20,
                max_value=50,
                value=25,
                step=1
            )

            irradiance = st.slider(
                "Irradiance (W/m²):",
                min_value=0,
                max_value=1200,
                value=1000,
                step=50
            )

        with col2:
            wind_speed = st.slider(
                "Wind Speed (m/s):",
                min_value=0.0,
                max_value=15.0,
                value=1.0,
                step=0.5
            )

            mounting = st.selectbox(
                "Mounting Type:",
                ["rooftop", "ground_mount", "facade", "tracking"],
                format_func=lambda x: x.replace('_', ' ').title()
            )

        # Calculate thermal performance
        thermal = designer.calculate_thermal_performance(
            ambient_temp, irradiance, wind_speed, mounting
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Module Temperature", f"{thermal['module_temperature']:.1f}°C")

        with col2:
            st.metric("Temperature Rise", f"{thermal['temperature_rise']:.1f}°C")

        with col3:
            st.metric("NOCT", f"{thermal['noct']:.1f}°C")

        with col4:
            st.metric("Thermal Resistance", f"{thermal['thermal_resistance']:.4f} K·m²/W")

        # Temperature coefficients
        temp_coeffs = designer.calculate_temperature_coefficients(
            cell_temp_coeff_pmax=-0.45,
            ctm_temp_losses=k3
        )

        st.subheader("Temperature Coefficients")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pmax Coefficient", f"{temp_coeffs['temp_coeff_pmax']:.3f}%/°C")

        with col2:
            st.metric("Voc Coefficient", f"{temp_coeffs['temp_coeff_voc']:.3f}%/°C")

        with col3:
            st.metric("Isc Coefficient", f"+{temp_coeffs['temp_coeff_isc']:.3f}%/°C")

        # Temperature-corrected power
        power_at_temp = temperature_corrected_power(
            power_result['module_power_stc'],
            thermal['module_temperature'],
            25.0,
            temp_coeffs['temp_coeff_pmax']
        )

        st.metric(
            "Temperature-Corrected Power",
            f"{power_at_temp:.1f} W",
            delta=f"{power_at_temp - power_result['module_power_stc']:.1f} W"
        )

        # Thermal heatmap
        st.subheader("Module Temperature Map")
        ambient_range = np.linspace(-10, 45, 12)
        irradiance_range = np.linspace(200, 1200, 11)

        fig_heatmap = designer.create_thermal_heatmap(
            ambient_range, irradiance_range, mounting
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab4:
        st.subheader("String Configuration Optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Module Electrical Parameters**")

            voc = st.number_input(
                "Module Voc (V):",
                min_value=30.0,
                max_value=60.0,
                value=49.5,
                step=0.1
            )

            vmp = st.number_input(
                "Module Vmp (V):",
                min_value=25.0,
                max_value=55.0,
                value=41.8,
                step=0.1
            )

            isc = st.number_input(
                "Module Isc (A):",
                min_value=5.0,
                max_value=15.0,
                value=11.5,
                step=0.1
            )

            imp = st.number_input(
                "Module Imp (A):",
                min_value=5.0,
                max_value=15.0,
                value=10.8,
                step=0.1
            )

        with col2:
            st.write("**Inverter Specifications**")

            v_min = st.number_input(
                "Min MPPT Voltage (V):",
                min_value=100,
                max_value=500,
                value=200,
                step=10
            )

            v_max = st.number_input(
                "Max MPPT Voltage (V):",
                min_value=500,
                max_value=1500,
                value=1000,
                step=50
            )

            max_string = st.slider(
                "Max String Length:",
                min_value=10,
                max_value=35,
                value=25,
                step=1
            )

        # Optimize configuration
        string_config = designer.optimize_string_configuration(
            voc, vmp, (v_min, v_max), max_string
        )

        st.subheader("Optimized String Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Modules per String", f"{string_config['modules_per_string']}")
            st.metric("String Voc (STC)", f"{string_config['string_voc_stc']:.1f} V")
            st.metric("String Vmp (STC)", f"{string_config['string_vmp_stc']:.1f} V")

        with col2:
            st.metric("String Voc (Cold)", f"{string_config['string_voc_cold']:.1f} V")
            st.metric("String Vmp (Cold)", f"{string_config['string_vmp_cold']:.1f} V")
            st.metric("String Vmp (Hot)", f"{string_config['string_vmp_hot']:.1f} V")

        with col3:
            st.metric("Min Modules", f"{string_config['min_modules']}")
            st.metric("Max Modules", f"{string_config['max_modules']}")
            st.metric("Voltage Margin", f"{string_config['voltage_margin_max']:.1f}%")

        # Voltage visualization
        fig = go.Figure()

        categories = ['V_min', 'Vmp Hot', 'Vmp STC', 'Vmp Cold', 'Voc STC', 'Voc Cold', 'V_max']
        values = [
            v_min,
            string_config['string_vmp_hot'],
            string_config['string_vmp_stc'],
            string_config['string_vmp_cold'],
            string_config['string_voc_stc'],
            string_config['string_voc_cold'],
            v_max
        ]
        colors = ['red', 'orange', 'green', 'blue', 'cyan', 'purple', 'red']

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.0f}V" for v in values],
            textposition='auto'
        ))

        # Add MPPT range
        fig.add_hrect(
            y0=v_min, y1=v_max,
            fillcolor="green", opacity=0.1,
            annotation_text="MPPT Range",
            annotation_position="top right"
        )

        fig.update_layout(
            title="String Voltage Analysis",
            yaxis_title="Voltage (V)",
            height=400,
            template='plotly_white',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Module Performance Summary")

        # Create comprehensive specification
        spec_data = {
            "Parameter": [
                "Module Power (STC)",
                "Module Efficiency",
                "Cell Efficiency",
                "CTM Ratio",
                "Open Circuit Voltage",
                "Maximum Power Voltage",
                "Short Circuit Current",
                "Maximum Power Current",
                "Module Dimensions",
                "Module Weight",
                "Temperature Coefficient (Pmax)",
                "Temperature Coefficient (Voc)",
                "Temperature Coefficient (Isc)",
                "NOCT",
                "Number of Cells",
                "Cell Configuration",
                "Encapsulation Type",
                "Bifacial",
                "Frame Type",
                "Junction Box"
            ],
            "Value": [
                f"{power_result['module_power_stc']:.1f} W",
                f"{ctm_result['module_efficiency']:.2f}%",
                f"{ctm_result['cell_efficiency']:.2f}%",
                f"{ctm_result['ctm_ratio']:.3f}",
                f"{voc:.2f} V",
                f"{vmp:.2f} V",
                f"{isc:.2f} A",
                f"{imp:.2f} A",
                f"{layout['module_width']:.0f} × {layout['module_height']:.0f} mm",
                f"{weight['total_weight_kg']:.1f} kg",
                f"{temp_coeffs['temp_coeff_pmax']:.3f}%/°C",
                f"{temp_coeffs['temp_coeff_voc']:.3f}%/°C",
                f"+{temp_coeffs['temp_coeff_isc']:.3f}%/°C",
                f"{thermal['noct']:.1f}°C",
                f"{num_cells}",
                f"{layout['configuration']}",
                designer.encapsulation_types[encapsulation]["name"],
                "Yes" if bifacial else "No",
                frame_type,
                junction_box.title()
            ]
        }

        df_spec = pd.DataFrame(spec_data)
        st.dataframe(df_spec, use_container_width=True, hide_index=True, height=700)

        # Download specification sheet
        csv = df_spec.to_csv(index=False)
        st.download_button(
            label="Download Specification Sheet (CSV)",
            data=csv,
            file_name="module_specification.csv",
            mime="text/csv"
        )

    st.divider()
    st.info("💡 **Module Design & CTM Loss Analysis** - Branch B03 | 5 Sessions Integrated")
