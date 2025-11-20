"""
Interactive Configurator for PV system layout design.

This module provides interactive tools for designing PV system layouts with
drag-drop functionality, real-time validation, and design optimization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from src.models.pv_components import (
    SystemDesign, StringConfiguration, MountingStructure,
    PVModule, SiteLocation
)


@dataclass
class ModulePosition:
    """Represents the position of a module in the layout."""
    row: int
    col: int
    x: float  # meters
    y: float  # meters
    z: float  # meters
    tilt: float  # degrees
    azimuth: float  # degrees
    module_id: str
    string_id: int = 0
    is_shaded: bool = False
    shade_fraction: float = 0.0


@dataclass
class ArrayLayout:
    """Represents a complete array layout."""
    modules: List[ModulePosition] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    row_spacing: float = 3.0  # meters
    col_spacing: float = 0.02  # meters (small gap between modules)
    total_area: float = 0.0  # m²


class InteractiveConfigurator:
    """
    Interactive configurator for PV system layout design.

    Provides drag-drop layout tools, real-time validation, constraint checking,
    auto-optimization triggers, and design comparison capabilities.
    """

    def __init__(self):
        """Initialize the InteractiveConfigurator."""
        self.initialize_layout_state()

    def initialize_layout_state(self) -> None:
        """Initialize session state for layout configuration."""
        if 'array_layout' not in st.session_state:
            st.session_state.array_layout = ArrayLayout()

        if 'layout_mode' not in st.session_state:
            st.session_state.layout_mode = "automatic"  # or "manual"

        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = {}

        if 'layout_constraints' not in st.session_state:
            st.session_state.layout_constraints = {
                'max_width': 100.0,  # meters
                'max_length': 100.0,  # meters
                'min_row_spacing': 1.5,  # meters
                'max_modules': 1000,
                'setback_requirements': {
                    'north': 5.0,
                    'south': 5.0,
                    'east': 5.0,
                    'west': 5.0
                }
            }

    def drag_drop_layout(self, module_config: StringConfiguration) -> None:
        """
        Create drag-and-drop layout interface.

        Provides interactive tools for manually positioning modules or
        automatically generating layouts.

        Args:
            module_config: String configuration for modules to layout
        """
        st.subheader("🎨 Array Layout Designer")

        # Layout mode selection
        col1, col2 = st.columns([1, 3])

        with col1:
            layout_mode = st.radio(
                "Layout Mode",
                options=["Automatic", "Manual"],
                index=0 if st.session_state.layout_mode == "automatic" else 1
            )
            st.session_state.layout_mode = layout_mode.lower()

        with col2:
            if layout_mode == "Automatic":
                self._render_automatic_layout(module_config)
            else:
                self._render_manual_layout(module_config)

        # Display current layout
        st.markdown("---")
        st.markdown("### Current Layout")

        if st.session_state.array_layout.modules:
            self._display_layout_visualization()
            self._display_layout_metrics()
        else:
            st.info("No layout configured. Use the tools above to create a layout.")

    def _render_automatic_layout(self, module_config: StringConfiguration) -> None:
        """Render automatic layout generation interface."""
        st.markdown("**Automatic Layout Generation**")

        col1, col2, col3 = st.columns(3)

        with col1:
            num_rows = st.number_input(
                "Number of Rows",
                min_value=1,
                max_value=100,
                value=5,
                help="Number of module rows"
            )

        with col2:
            modules_per_row = st.number_input(
                "Modules per Row",
                min_value=1,
                max_value=100,
                value=20,
                help="Number of modules in each row"
            )

        with col3:
            orientation = st.selectbox(
                "Module Orientation",
                options=["Landscape", "Portrait"],
                index=0
            )

        # Spacing parameters
        col1, col2 = st.columns(2)

        with col1:
            row_spacing = st.slider(
                "Row Spacing (m)",
                min_value=1.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Distance between rows"
            )

        with col2:
            col_spacing = st.slider(
                "Column Spacing (m)",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
                help="Gap between modules in a row"
            )

        # Generate layout button
        if st.button("Generate Layout", type="primary"):
            layout = self._generate_automatic_layout(
                module_config=module_config,
                num_rows=num_rows,
                modules_per_row=modules_per_row,
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                orientation=orientation
            )
            st.session_state.array_layout = layout
            st.success(f"✅ Generated layout with {len(layout.modules)} modules")
            st.rerun()

    def _render_manual_layout(self, module_config: StringConfiguration) -> None:
        """Render manual layout interface."""
        st.markdown("**Manual Layout Editor**")
        st.info("Manual drag-drop layout editor - Click on grid to place modules")

        # Grid parameters
        col1, col2 = st.columns(2)

        with col1:
            grid_width = st.number_input("Grid Width (m)", min_value=10.0, value=50.0, step=5.0)
            grid_height = st.number_input("Grid Height (m)", min_value=10.0, value=50.0, step=5.0)

        with col2:
            snap_to_grid = st.checkbox("Snap to Grid", value=True)
            grid_size = st.number_input("Grid Size (m)", min_value=0.1, value=0.5, step=0.1)

        # Interactive canvas would go here (requires more advanced Streamlit components)
        st.warning("Manual layout requires interactive canvas component - using simplified interface")

        # Simplified manual entry
        if st.button("Add Module Manually"):
            st.session_state.manual_add_mode = True

        if st.session_state.get('manual_add_mode', False):
            col1, col2, col3 = st.columns(3)
            with col1:
                x_pos = st.number_input("X Position (m)", value=0.0, step=0.1)
            with col2:
                y_pos = st.number_input("Y Position (m)", value=0.0, step=0.1)
            with col3:
                if st.button("Place Module"):
                    self._add_module_at_position(module_config, x_pos, y_pos)
                    st.success("Module added!")
                    st.rerun()

    def _generate_automatic_layout(
        self,
        module_config: StringConfiguration,
        num_rows: int,
        modules_per_row: int,
        row_spacing: float,
        col_spacing: float,
        orientation: str
    ) -> ArrayLayout:
        """
        Generate automatic array layout.

        Args:
            module_config: Module configuration
            num_rows: Number of rows
            modules_per_row: Modules per row
            row_spacing: Spacing between rows (m)
            col_spacing: Spacing between columns (m)
            orientation: Module orientation (Landscape/Portrait)

        Returns:
            ArrayLayout with positioned modules
        """
        layout = ArrayLayout()
        layout.num_rows = num_rows
        layout.num_cols = modules_per_row
        layout.row_spacing = row_spacing
        layout.col_spacing = col_spacing

        module = module_config.module

        # Module dimensions based on orientation
        if orientation == "Landscape":
            module_width = module.length
            module_height = module.width
        else:
            module_width = module.width
            module_height = module.length

        # Get tilt and azimuth from mounting config if available
        tilt = 25.0  # default
        azimuth = 180.0  # default
        if st.session_state.get('mounting_config'):
            tilt = st.session_state.mounting_config.tilt_angle
            azimuth = st.session_state.mounting_config.azimuth

        # Generate module positions
        modules = []
        string_id = 0
        modules_in_current_string = 0

        for row in range(num_rows):
            for col in range(modules_per_row):
                # Calculate position
                x = col * (module_width + col_spacing)
                y = row * row_spacing
                z = 0.5  # Default height above ground

                # Create module position
                module_pos = ModulePosition(
                    row=row,
                    col=col,
                    x=x,
                    y=y,
                    z=z,
                    tilt=tilt,
                    azimuth=azimuth,
                    module_id=f"M{row:03d}_{col:03d}",
                    string_id=string_id
                )

                modules.append(module_pos)

                # Track string assignment
                modules_in_current_string += 1
                if modules_in_current_string >= module_config.modules_per_string:
                    string_id += 1
                    modules_in_current_string = 0

        layout.modules = modules
        layout.total_area = num_rows * modules_per_row * module.area

        return layout

    def _add_module_at_position(
        self,
        module_config: StringConfiguration,
        x: float,
        y: float
    ) -> None:
        """Add a module at specified position."""
        module_pos = ModulePosition(
            row=len(st.session_state.array_layout.modules),
            col=0,
            x=x,
            y=y,
            z=0.5,
            tilt=25.0,
            azimuth=180.0,
            module_id=f"M{len(st.session_state.array_layout.modules):03d}",
            string_id=0
        )
        st.session_state.array_layout.modules.append(module_pos)

    def real_time_validation(self) -> Dict[str, Any]:
        """
        Perform real-time validation of current layout.

        Validates layout against design constraints, spacing requirements,
        electrical limits, and site boundaries.

        Returns:
            Dictionary with validation results and issues
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks': []
        }

        layout = st.session_state.array_layout

        if not layout.modules:
            validation_results['warnings'].append("No modules in layout")
            return validation_results

        # Check 1: Site boundary constraints
        validation_results['checks'].append("Site boundary check")
        max_x = max(m.x for m in layout.modules)
        max_y = max(m.y for m in layout.modules)

        if max_x > st.session_state.layout_constraints['max_width']:
            validation_results['errors'].append(
                f"Layout width ({max_x:.1f}m) exceeds site boundary "
                f"({st.session_state.layout_constraints['max_width']:.1f}m)"
            )
            validation_results['is_valid'] = False

        if max_y > st.session_state.layout_constraints['max_length']:
            validation_results['errors'].append(
                f"Layout length ({max_y:.1f}m) exceeds site boundary "
                f"({st.session_state.layout_constraints['max_length']:.1f}m)"
            )
            validation_results['is_valid'] = False

        # Check 2: Module count limits
        validation_results['checks'].append("Module count check")
        if len(layout.modules) > st.session_state.layout_constraints['max_modules']:
            validation_results['errors'].append(
                f"Module count ({len(layout.modules)}) exceeds limit "
                f"({st.session_state.layout_constraints['max_modules']})"
            )
            validation_results['is_valid'] = False

        # Check 3: Row spacing
        validation_results['checks'].append("Row spacing check")
        if layout.row_spacing < st.session_state.layout_constraints['min_row_spacing']:
            validation_results['errors'].append(
                f"Row spacing ({layout.row_spacing:.2f}m) below minimum "
                f"({st.session_state.layout_constraints['min_row_spacing']:.2f}m)"
            )
            validation_results['is_valid'] = False

        # Check 4: Module overlaps
        validation_results['checks'].append("Module overlap check")
        overlaps = self._check_module_overlaps(layout.modules)
        if overlaps:
            validation_results['errors'].append(
                f"Found {len(overlaps)} module overlaps"
            )
            validation_results['is_valid'] = False

        # Check 5: String configuration
        if st.session_state.get('selected_modules'):
            validation_results['checks'].append("String configuration check")
            string_issues = self._validate_string_configuration(layout)
            if string_issues:
                validation_results['warnings'].extend(string_issues)

        st.session_state.validation_results = validation_results
        return validation_results

    def design_constraints_check(self, design: Optional[SystemDesign] = None) -> Dict[str, Any]:
        """
        Check design against all constraints.

        Validates complete design including electrical limits, structural constraints,
        and regulatory requirements.

        Args:
            design: System design to validate (optional)

        Returns:
            Dictionary with constraint check results
        """
        results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

        # Electrical constraints
        if st.session_state.get('selected_modules') and st.session_state.get('selected_inverters'):
            # Check voltage limits
            for string_config in st.session_state.selected_modules:
                max_string_voltage = string_config.string_v_oc

                for inverter in st.session_state.selected_inverters:
                    if max_string_voltage > inverter.v_dc_max:
                        results['failed'].append(
                            f"String voltage ({max_string_voltage:.1f}V) exceeds "
                            f"inverter max ({inverter.v_dc_max:.1f}V)"
                        )
                    elif max_string_voltage < inverter.v_dc_min:
                        results['failed'].append(
                            f"String voltage ({max_string_voltage:.1f}V) below "
                            f"inverter min ({inverter.v_dc_min:.1f}V)"
                        )
                    else:
                        results['passed'].append("String voltage within inverter limits")

                    # Check MPPT range
                    string_vmp = string_config.string_v_mp
                    if string_vmp < inverter.v_mpp_min or string_vmp > inverter.v_mpp_max:
                        results['warnings'].append(
                            f"String Vmp ({string_vmp:.1f}V) outside optimal MPPT range "
                            f"({inverter.v_mpp_min:.1f}-{inverter.v_mpp_max:.1f}V)"
                        )

        # Layout constraints
        layout_validation = self.real_time_validation()
        if layout_validation['is_valid']:
            results['passed'].append("Layout constraints satisfied")
        else:
            results['failed'].extend(layout_validation['errors'])

        # Structural constraints
        if st.session_state.get('mounting_config'):
            mounting = st.session_state.mounting_config
            if mounting.tilt_angle < 0 or mounting.tilt_angle > 60:
                results['warnings'].append(
                    f"Unusual tilt angle ({mounting.tilt_angle}°) - typically 10-45°"
                )

        return results

    def auto_optimization_trigger(self) -> Dict[str, Any]:
        """
        Automatically trigger optimization when design changes.

        Monitors design changes and suggests optimizations for improved performance.

        Returns:
            Dictionary with optimization suggestions
        """
        suggestions = {
            'triggered': False,
            'optimizations': [],
            'estimated_improvements': {}
        }

        # Check if optimization should be triggered
        if not st.session_state.get('selected_modules'):
            return suggestions

        if not st.session_state.array_layout.modules:
            return suggestions

        suggestions['triggered'] = True

        # Analyze current design
        layout = st.session_state.array_layout

        # Suggestion 1: Row spacing optimization
        if layout.row_spacing < 2.5:
            suggestions['optimizations'].append({
                'type': 'row_spacing',
                'current': layout.row_spacing,
                'suggested': 3.0,
                'reason': 'Increased row spacing reduces shading losses',
                'estimated_gain': '+2.5% annual energy'
            })

        # Suggestion 2: DC/AC ratio optimization
        if st.session_state.get('selected_inverters'):
            dc_power = sum(
                config.total_power
                for config in st.session_state.selected_modules
            ) / 1000.0
            ac_power = sum(
                inv.p_ac_rated
                for inv in st.session_state.selected_inverters
            ) / 1000.0

            if ac_power > 0:
                dc_ac_ratio = dc_power / ac_power

                if dc_ac_ratio < 1.15:
                    suggestions['optimizations'].append({
                        'type': 'dc_ac_ratio',
                        'current': dc_ac_ratio,
                        'suggested': 1.25,
                        'reason': 'Higher DC/AC ratio improves capacity factor',
                        'estimated_gain': '+3.0% annual energy'
                    })
                elif dc_ac_ratio > 1.40:
                    suggestions['optimizations'].append({
                        'type': 'dc_ac_ratio',
                        'current': dc_ac_ratio,
                        'suggested': 1.30,
                        'reason': 'Lower DC/AC ratio reduces clipping losses',
                        'estimated_gain': '+1.5% annual energy'
                    })

        # Suggestion 3: Tilt angle optimization
        if st.session_state.get('mounting_config') and st.session_state.get('site_location'):
            mounting = st.session_state.mounting_config
            site = st.session_state.site_location

            # Simple rule: optimal tilt ≈ latitude
            optimal_tilt = abs(site.latitude)

            if abs(mounting.tilt_angle - optimal_tilt) > 5:
                suggestions['optimizations'].append({
                    'type': 'tilt_angle',
                    'current': mounting.tilt_angle,
                    'suggested': optimal_tilt,
                    'reason': f'Tilt angle closer to latitude ({site.latitude:.1f}°) maximizes annual irradiance',
                    'estimated_gain': '+1.8% annual energy'
                })

        return suggestions

    def design_comparison_view(
        self,
        designs: List[SystemDesign]
    ) -> pd.DataFrame:
        """
        Create comparison view for multiple designs.

        Args:
            designs: List of system designs to compare

        Returns:
            DataFrame with comparison metrics
        """
        if not designs:
            return pd.DataFrame()

        comparison_data = []

        for design in designs:
            metrics = {
                'Design Name': design.design_name,
                'DC Power (kW)': design.total_dc_power,
                'AC Power (kW)': design.total_ac_power,
                'DC/AC Ratio': design.dc_ac_ratio,
                'Module Count': design.total_modules_count,
                'System Losses (%)': design.total_system_losses,
                'Validated': '✅' if design.is_validated else '❌',
                'Errors': len(design.validation_errors),
                'Warnings': len(design.validation_warnings)
            }
            comparison_data.append(metrics)

        return pd.DataFrame(comparison_data)

    def _display_layout_visualization(self) -> None:
        """Display interactive 2D layout visualization."""
        layout = st.session_state.array_layout

        if not layout.modules:
            return

        # Create plotly figure
        fig = go.Figure()

        # Get module dimensions (assuming first module config)
        if st.session_state.get('selected_modules'):
            module = st.session_state.selected_modules[0].module
            module_length = module.length
            module_width = module.width
        else:
            module_length = 2.0
            module_width = 1.0

        # Plot each module as a rectangle
        for module_pos in layout.modules:
            # Rectangle corners
            x_coords = [
                module_pos.x,
                module_pos.x + module_length,
                module_pos.x + module_length,
                module_pos.x,
                module_pos.x
            ]
            y_coords = [
                module_pos.y,
                module_pos.y,
                module_pos.y + module_width,
                module_pos.y + module_width,
                module_pos.y
            ]

            # Color based on shading
            color = 'lightblue' if not module_pos.is_shaded else 'orange'

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill='toself',
                fillcolor=color,
                line=dict(color='darkblue', width=1),
                mode='lines',
                name=module_pos.module_id,
                showlegend=False,
                hovertemplate=(
                    f"<b>{module_pos.module_id}</b><br>"
                    f"Row: {module_pos.row}<br>"
                    f"Col: {module_pos.col}<br>"
                    f"Position: ({module_pos.x:.1f}, {module_pos.y:.1f}) m<br>"
                    f"String: {module_pos.string_id}<br>"
                    "<extra></extra>"
                )
            ))

        # Update layout
        fig.update_layout(
            title="Array Layout (Top View)",
            xaxis_title="East-West (m)",
            yaxis_title="North-South (m)",
            width=800,
            height=600,
            hovermode='closest',
            showlegend=False,
            yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_layout_metrics(self) -> None:
        """Display layout metrics."""
        layout = st.session_state.array_layout

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Modules", len(layout.modules))

        with col2:
            max_x = max(m.x for m in layout.modules) if layout.modules else 0
            max_y = max(m.y for m in layout.modules) if layout.modules else 0
            st.metric("Layout Size", f"{max_x:.1f}m × {max_y:.1f}m")

        with col3:
            st.metric("Row Spacing", f"{layout.row_spacing:.2f} m")

        with col4:
            st.metric("Total Area", f"{layout.total_area:.1f} m²")

    def _check_module_overlaps(self, modules: List[ModulePosition]) -> List[Tuple[str, str]]:
        """
        Check for module overlaps.

        Args:
            modules: List of module positions

        Returns:
            List of overlapping module ID pairs
        """
        overlaps = []

        # Simple overlap detection (can be optimized with spatial indexing)
        for i, mod1 in enumerate(modules):
            for mod2 in modules[i + 1:]:
                # Check if rectangles overlap
                # Assuming modules have standard dimensions
                module_size = 2.0  # meters (approximate)

                if (abs(mod1.x - mod2.x) < module_size and
                    abs(mod1.y - mod2.y) < module_size):
                    overlaps.append((mod1.module_id, mod2.module_id))

        return overlaps

    def _validate_string_configuration(self, layout: ArrayLayout) -> List[str]:
        """
        Validate string configuration in layout.

        Args:
            layout: Array layout to validate

        Returns:
            List of validation warnings
        """
        warnings = []

        if not st.session_state.get('selected_modules'):
            return warnings

        # Check if number of modules matches string configuration
        total_config_modules = sum(
            config.total_modules
            for config in st.session_state.selected_modules
        )

        if len(layout.modules) != total_config_modules:
            warnings.append(
                f"Layout has {len(layout.modules)} modules but configuration "
                f"specifies {total_config_modules} modules"
            )

        return warnings
