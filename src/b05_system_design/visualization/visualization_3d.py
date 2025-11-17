"""
3D Visualization module for PV system design.

This module provides comprehensive 3D visualization capabilities including
system rendering, animated sun path, shade visualization, terrain overlay,
and interactive camera controls.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from src.models.pv_components import SystemDesign, SiteLocation, MountingStructure
from src.b05_system_design.configurator.interactive_configurator import ArrayLayout, ModulePosition


@dataclass
class SunPosition:
    """Sun position at a specific time."""
    datetime: datetime
    altitude: float  # degrees above horizon
    azimuth: float  # degrees from north
    x: float  # 3D coordinates
    y: float
    z: float


class Visualization3D:
    """
    3D visualization engine for PV system design.

    Provides interactive 3D rendering of PV systems with sun path animation,
    shade analysis visualization, terrain overlay, and camera controls.
    """

    def __init__(self):
        """Initialize the Visualization3D engine."""
        self.initialize_visualization_state()

    def initialize_visualization_state(self) -> None:
        """Initialize session state for 3D visualization."""
        if 'camera_position' not in st.session_state:
            st.session_state.camera_position = {
                'eye': dict(x=1.5, y=1.5, z=1.5),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=0, z=1)
            }

        if 'show_sun_path' not in st.session_state:
            st.session_state.show_sun_path = True

        if 'show_shade_analysis' not in st.session_state:
            st.session_state.show_shade_analysis = False

        if 'current_time' not in st.session_state:
            st.session_state.current_time = datetime.now()

        if 'animation_speed' not in st.session_state:
            st.session_state.animation_speed = 1.0

    def render_system_3d(
        self,
        layout: ArrayLayout,
        mounting: Optional[MountingStructure] = None,
        site: Optional[SiteLocation] = None
    ) -> go.Figure:
        """
        Render complete 3D visualization of PV system.

        Args:
            layout: Array layout to render
            mounting: Mounting structure configuration
            site: Site location for sun path calculation

        Returns:
            Plotly 3D figure
        """
        fig = go.Figure()

        if not layout.modules:
            st.warning("No modules in layout to visualize")
            return fig

        # Get module dimensions
        if st.session_state.get('selected_modules'):
            module = st.session_state.selected_modules[0].module
            module_length = module.length
            module_width = module.width
            module_thickness = module.thickness
        else:
            module_length = 2.0
            module_width = 1.0
            module_thickness = 0.04

        # Render modules
        self._add_modules_to_figure(
            fig, layout.modules, module_length, module_width,
            module_thickness, mounting
        )

        # Render mounting structures
        if mounting:
            self._add_mounting_structures(fig, layout.modules, mounting)

        # Render ground plane
        self._add_ground_plane(fig, layout)

        # Add sun path if enabled
        if st.session_state.show_sun_path and site:
            self._add_sun_path(fig, site, st.session_state.current_time)

        # Add shade visualization if enabled
        if st.session_state.show_shade_analysis:
            self._add_shade_visualization(fig, layout, site)

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis_title="East-West (m)",
                yaxis_title="North-South (m)",
                zaxis_title="Height (m)",
                camera=st.session_state.camera_position,
                aspectmode='data'
            ),
            title="3D System Visualization",
            width=900,
            height=700,
            showlegend=True
        )

        return fig

    def _add_modules_to_figure(
        self,
        fig: go.Figure,
        modules: List[ModulePosition],
        length: float,
        width: float,
        thickness: float,
        mounting: Optional[MountingStructure]
    ) -> None:
        """Add PV modules to 3D figure."""
        tilt = mounting.tilt_angle if mounting else 25.0
        tilt_rad = np.radians(tilt)

        # Sample subset for performance (show every Nth module for large arrays)
        step = max(1, len(modules) // 100)  # Show max 100 modules
        sampled_modules = modules[::step]

        for mod in sampled_modules:
            # Calculate module corners with tilt
            # Module lies in XY plane, tilted around X-axis

            # Base corners (flat)
            corners = np.array([
                [0, 0, 0],
                [length, 0, 0],
                [length, width, 0],
                [0, width, 0]
            ])

            # Apply tilt rotation around X-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                [0, np.sin(tilt_rad), np.cos(tilt_rad)]
            ])

            tilted_corners = corners @ rotation_matrix.T

            # Translate to module position
            tilted_corners += np.array([mod.x, mod.y, mod.z])

            # Create mesh for module surface
            x = tilted_corners[:, 0].tolist() + [tilted_corners[0, 0]]
            y = tilted_corners[:, 1].tolist() + [tilted_corners[0, 1]]
            z = tilted_corners[:, 2].tolist() + [tilted_corners[0, 2]]

            # Color based on shading
            color = 'rgba(30, 80, 150, 0.7)' if not mod.is_shaded else 'rgba(200, 100, 50, 0.7)'

            fig.add_trace(go.Mesh3d(
                x=tilted_corners[:, 0],
                y=tilted_corners[:, 1],
                z=tilted_corners[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.8,
                name=f"Module {mod.module_id}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{mod.module_id}</b><br>"
                    f"Position: ({mod.x:.1f}, {mod.y:.1f}, {mod.z:.1f})<br>"
                    f"Tilt: {mod.tilt:.1f}°<br>"
                    f"String: {mod.string_id}<br>"
                    "<extra></extra>"
                )
            ))

    def _add_mounting_structures(
        self,
        fig: go.Figure,
        modules: List[ModulePosition],
        mounting: MountingStructure
    ) -> None:
        """Add mounting structure supports to figure."""
        # Add simplified support posts
        step = max(1, len(modules) // 50)  # Show max 50 posts
        sampled_modules = modules[::step]

        for mod in sampled_modules:
            # Add vertical post from ground to module
            fig.add_trace(go.Scatter3d(
                x=[mod.x, mod.x],
                y=[mod.y, mod.y],
                z=[0, mod.z],
                mode='lines',
                line=dict(color='gray', width=3),
                name='Support',
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_ground_plane(self, fig: go.Figure, layout: ArrayLayout) -> None:
        """Add ground plane to figure."""
        if not layout.modules:
            return

        # Calculate ground plane bounds
        x_coords = [m.x for m in layout.modules]
        y_coords = [m.y for m in layout.modules]

        x_min, x_max = min(x_coords) - 5, max(x_coords) + 10
        y_min, y_max = min(y_coords) - 5, max(y_coords) + 10

        # Create ground mesh
        x_ground = [x_min, x_max, x_max, x_min]
        y_ground = [y_min, y_min, y_max, y_max]
        z_ground = [0, 0, 0, 0]

        fig.add_trace(go.Mesh3d(
            x=x_ground,
            y=y_ground,
            z=z_ground,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='lightgreen',
            opacity=0.3,
            name='Ground',
            showlegend=True,
            hoverinfo='skip'
        ))

    def animated_sun_path(
        self,
        site: SiteLocation,
        date: datetime,
        num_points: int = 48
    ) -> List[SunPosition]:
        """
        Calculate animated sun path for a given date.

        Args:
            site: Site location
            date: Date for sun path calculation
            num_points: Number of points in sun path

        Returns:
            List of sun positions throughout the day
        """
        sun_positions = []

        # Calculate sun position for each hour of the day
        for hour in np.linspace(0, 24, num_points):
            time = date.replace(hour=int(hour), minute=int((hour % 1) * 60))

            # Calculate sun position (simplified calculation)
            altitude, azimuth = self._calculate_sun_position(
                site.latitude, site.longitude, time
            )

            # Convert to 3D coordinates (spherical to Cartesian)
            # Place sun at distance of 50m for visualization
            r = 50.0
            alt_rad = np.radians(altitude)
            az_rad = np.radians(azimuth)

            x = r * np.cos(alt_rad) * np.sin(az_rad)
            y = r * np.cos(alt_rad) * np.cos(az_rad)
            z = r * np.sin(alt_rad)

            sun_positions.append(SunPosition(
                datetime=time,
                altitude=altitude,
                azimuth=azimuth,
                x=x,
                y=y,
                z=z
            ))

        return sun_positions

    def _add_sun_path(
        self,
        fig: go.Figure,
        site: SiteLocation,
        current_time: datetime
    ) -> None:
        """Add sun path to 3D figure."""
        # Calculate sun path for current day
        sun_path = self.animated_sun_path(site, current_time)

        # Filter for sun above horizon
        visible_path = [pos for pos in sun_path if pos.altitude > 0]

        if not visible_path:
            return

        # Plot sun path arc
        x_path = [pos.x for pos in visible_path]
        y_path = [pos.y for pos in visible_path]
        z_path = [pos.z for pos in visible_path]

        fig.add_trace(go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='lines',
            line=dict(color='orange', width=4, dash='dash'),
            name='Sun Path',
            showlegend=True,
            hovertemplate=(
                "Time: %{text}<br>"
                "Altitude: %{customdata[0]:.1f}°<br>"
                "Azimuth: %{customdata[1]:.1f}°<br>"
                "<extra></extra>"
            ),
            text=[pos.datetime.strftime("%H:%M") for pos in visible_path],
            customdata=[[pos.altitude, pos.azimuth] for pos in visible_path]
        ))

        # Add current sun position
        current_pos = self._get_current_sun_position(site, current_time)
        if current_pos.altitude > 0:
            fig.add_trace(go.Scatter3d(
                x=[current_pos.x],
                y=[current_pos.y],
                z=[current_pos.z],
                mode='markers',
                marker=dict(size=15, color='yellow', symbol='circle'),
                name='Current Sun Position',
                showlegend=True,
                hovertemplate=(
                    f"<b>Current Sun Position</b><br>"
                    f"Time: {current_time.strftime('%H:%M')}<br>"
                    f"Altitude: {current_pos.altitude:.1f}°<br>"
                    f"Azimuth: {current_pos.azimuth:.1f}°<br>"
                    "<extra></extra>"
                )
            ))

    def _calculate_sun_position(
        self,
        latitude: float,
        longitude: float,
        time: datetime
    ) -> Tuple[float, float]:
        """
        Calculate sun position (simplified calculation).

        Args:
            latitude: Site latitude (degrees)
            longitude: Site longitude (degrees)
            time: Datetime for calculation

        Returns:
            Tuple of (altitude, azimuth) in degrees
        """
        # This is a simplified calculation
        # In production, use pvlib or similar for accurate solar position

        # Day of year
        n = time.timetuple().tm_yday

        # Declination angle (simplified)
        declination = 23.45 * np.sin(np.radians(360 * (284 + n) / 365))

        # Hour angle
        hour = time.hour + time.minute / 60.0
        hour_angle = 15 * (hour - 12)  # degrees

        # Convert to radians
        lat_rad = np.radians(latitude)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        # Solar altitude
        sin_alt = (np.sin(lat_rad) * np.sin(dec_rad) +
                   np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
        altitude = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))

        # Solar azimuth
        cos_az = ((np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) /
                  (np.cos(lat_rad) * np.cos(np.radians(altitude))))
        cos_az = np.clip(cos_az, -1, 1)
        azimuth = np.degrees(np.arccos(cos_az))

        # Adjust azimuth for afternoon
        if hour > 12:
            azimuth = 360 - azimuth

        return altitude, azimuth

    def _get_current_sun_position(
        self,
        site: SiteLocation,
        time: datetime
    ) -> SunPosition:
        """Get sun position at current time."""
        altitude, azimuth = self._calculate_sun_position(
            site.latitude, site.longitude, time
        )

        r = 50.0
        alt_rad = np.radians(altitude)
        az_rad = np.radians(azimuth)

        x = r * np.cos(alt_rad) * np.sin(az_rad)
        y = r * np.cos(alt_rad) * np.cos(az_rad)
        z = r * np.sin(alt_rad)

        return SunPosition(
            datetime=time,
            altitude=altitude,
            azimuth=azimuth,
            x=x,
            y=y,
            z=z
        )

    def shade_visualization(
        self,
        layout: ArrayLayout,
        site: SiteLocation,
        time: datetime
    ) -> ArrayLayout:
        """
        Calculate and visualize shading on modules.

        Args:
            layout: Array layout
            site: Site location
            time: Time for shade calculation

        Returns:
            ArrayLayout with updated shade information
        """
        # Get sun position
        sun_pos = self._get_current_sun_position(site, time)

        if sun_pos.altitude <= 0:
            # Sun below horizon - all modules shaded
            for module in layout.modules:
                module.is_shaded = True
                module.shade_fraction = 1.0
            return layout

        # Simplified shade calculation
        # In production, use ray tracing or detailed shadow analysis

        for i, module in enumerate(layout.modules):
            # Check if module is shaded by modules in front
            is_shaded = False
            shade_fraction = 0.0

            # Check modules in rows ahead (lower row numbers in typical N-S oriented system)
            for other in layout.modules:
                if other.row < module.row:  # Module is in front
                    # Simple shadow test based on sun altitude and row spacing
                    shadow_length = module.z / np.tan(np.radians(max(sun_pos.altitude, 5)))

                    # Check if shadow reaches current module
                    row_distance = (module.row - other.row) * layout.row_spacing

                    if shadow_length >= row_distance:
                        is_shaded = True
                        # Calculate shade fraction
                        overlap = min(1.0, shadow_length / row_distance)
                        shade_fraction = max(shade_fraction, overlap * 0.3)  # Max 30% shading

            module.is_shaded = is_shaded
            module.shade_fraction = shade_fraction

        return layout

    def _add_shade_visualization(
        self,
        fig: go.Figure,
        layout: ArrayLayout,
        site: Optional[SiteLocation]
    ) -> None:
        """Add shade visualization to figure."""
        if not site:
            return

        # Calculate shading for current time
        shaded_layout = self.shade_visualization(
            layout, site, st.session_state.current_time
        )

        # Modules are already colored based on is_shaded in render_system_3d
        # This method could add shadow projections on ground

    def terrain_overlay(
        self,
        fig: go.Figure,
        terrain_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Add terrain overlay to visualization.

        Args:
            fig: Plotly figure to add terrain to
            terrain_data: Terrain elevation data (optional)
        """
        if terrain_data is None:
            # Generate sample terrain (gentle slopes)
            x = np.linspace(0, 100, 50)
            y = np.linspace(0, 100, 50)
            X, Y = np.meshgrid(x, y)

            # Simple sinusoidal terrain
            Z = 2 * np.sin(X / 20) * np.cos(Y / 20)

            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Earth',
                opacity=0.5,
                name='Terrain',
                showscale=False,
                hoverinfo='skip'
            ))

    def interactive_camera_controls(self) -> None:
        """
        Render interactive camera controls.

        Provides UI controls for adjusting camera position and view.
        """
        st.sidebar.markdown("### 📷 Camera Controls")

        # Preset views
        view_preset = st.sidebar.selectbox(
            "View Preset",
            options=["Isometric", "Top View", "Front View", "Side View", "Custom"],
            index=0
        )

        if view_preset == "Isometric":
            st.session_state.camera_position = {
                'eye': dict(x=1.5, y=1.5, z=1.5),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=0, z=1)
            }
        elif view_preset == "Top View":
            st.session_state.camera_position = {
                'eye': dict(x=0, y=0, z=3),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=1, z=0)
            }
        elif view_preset == "Front View":
            st.session_state.camera_position = {
                'eye': dict(x=0, y=3, z=0.5),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=0, z=1)
            }
        elif view_preset == "Side View":
            st.session_state.camera_position = {
                'eye': dict(x=3, y=0, z=0.5),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=0, z=1)
            }
        elif view_preset == "Custom":
            st.sidebar.markdown("**Camera Position:**")
            eye_x = st.sidebar.slider("Eye X", -5.0, 5.0, 1.5, 0.1)
            eye_y = st.sidebar.slider("Eye Y", -5.0, 5.0, 1.5, 0.1)
            eye_z = st.sidebar.slider("Eye Z", 0.0, 5.0, 1.5, 0.1)

            st.session_state.camera_position = {
                'eye': dict(x=eye_x, y=eye_y, z=eye_z),
                'center': dict(x=0, y=0, z=0),
                'up': dict(x=0, y=0, z=1)
            }

        # Visualization options
        st.sidebar.markdown("### 🎨 Visualization Options")

        st.session_state.show_sun_path = st.sidebar.checkbox(
            "Show Sun Path",
            value=st.session_state.show_sun_path
        )

        st.session_state.show_shade_analysis = st.sidebar.checkbox(
            "Show Shade Analysis",
            value=st.session_state.show_shade_analysis
        )

        # Time control for sun position
        if st.session_state.show_sun_path:
            st.sidebar.markdown("### ⏰ Time Control")

            current_date = st.sidebar.date_input(
                "Date",
                value=st.session_state.current_time.date()
            )

            current_hour = st.sidebar.slider(
                "Hour",
                min_value=0,
                max_value=23,
                value=st.session_state.current_time.hour
            )

            st.session_state.current_time = datetime.combine(
                current_date,
                datetime.min.time()
            ).replace(hour=current_hour)

            # Animation controls
            if st.sidebar.button("▶️ Animate Day"):
                st.sidebar.info("Animation feature - cycle through 24 hours")
