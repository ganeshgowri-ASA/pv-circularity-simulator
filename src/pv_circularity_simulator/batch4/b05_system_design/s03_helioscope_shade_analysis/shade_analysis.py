"""
Shade Analysis Engine for comprehensive near and far shading calculations.

This module provides detailed shade analysis including row-to-row shading,
obstruction shading, and far shading from horizon obstacles.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helioscope_model import HelioscapeModel
from .horizon_profiler import HorizonProfiler
from .irradiance import IrradianceOnSurface
from .sun_position import SunPositionCalculator
from .models import (
    ArrayGeometry,
    IrradianceComponents,
    ShadeAnalysisConfig,
    ShadeAnalysisResult,
    ShadingLoss,
    SiteModel,
    SunPosition,
    TrackerType,
)

logger = logging.getLogger(__name__)


class ShadeAnalysisEngine:
    """
    Comprehensive shade analysis engine.

    Performs near shading (row-to-row, obstacles), far shading (horizon),
    and calculates irradiance losses from shading effects.
    """

    def __init__(
        self,
        site_model: SiteModel,
        array_geometry: ArrayGeometry,
        config: ShadeAnalysisConfig
    ):
        """
        Initialize the shade analysis engine.

        Args:
            site_model: Site model with terrain and obstacles
            array_geometry: PV array geometry configuration
            config: Shade analysis configuration
        """
        self.site_model = site_model
        self.array_geometry = array_geometry
        self.config = config

        # Initialize helper components
        self.sun_calculator = SunPositionCalculator(site_model.location)
        self.irradiance_calculator = IrradianceOnSurface(
            site_model.location,
            array_geometry,
            config.transposition_model,
            config.aoi_model
        )
        self.helioscope_model = HelioscapeModel(site_model)
        self.horizon_profiler = HorizonProfiler(site_model.location)

        # Load horizon profile if available
        if site_model.horizon_profile:
            self.horizon_profiler.horizon_profile = site_model.horizon_profile

        logger.info("Initialized ShadeAnalysisEngine")

    def near_shading_analysis(
        self,
        sun_position: SunPosition,
        row_index: int = 0,
        total_rows: int = 10
    ) -> float:
        """
        Calculate near shading from row-to-row and nearby obstructions.

        Args:
            sun_position: Current sun position
            row_index: Index of the row being analyzed
            total_rows: Total number of rows in the array

        Returns:
            Near shading loss fraction (0-1)
        """
        if sun_position.elevation <= 0:
            # Sun below horizon - fully shaded
            return 1.0

        # Calculate row-to-row shading
        row_shading = self._calculate_row_to_row_shading(sun_position, row_index, total_rows)

        # Calculate obstruction shading
        obstruction_shading = self._calculate_obstruction_shading(
            sun_position, row_index
        )

        # Combined shading (take maximum)
        total_shading = max(row_shading, obstruction_shading)

        logger.debug(
            f"Near shading for row {row_index}: {total_shading:.3f} "
            f"(row-to-row: {row_shading:.3f}, obstruction: {obstruction_shading:.3f})"
        )

        return total_shading

    def far_shading_analysis(self, sun_position: SunPosition) -> float:
        """
        Calculate far shading from horizon obstacles and mountains.

        Args:
            sun_position: Current sun position

        Returns:
            Far shading loss fraction (0-1, where 1.0 = fully shaded)
        """
        if not self.config.enable_far_shading:
            return 0.0

        if sun_position.elevation <= 0:
            # Sun below astronomical horizon
            return 1.0

        if not self.horizon_profiler.horizon_profile:
            # No horizon profile available - no far shading
            return 0.0

        # Check if sun is above horizon profile
        is_visible = self.horizon_profiler.is_sun_visible(
            sun_position.azimuth,
            sun_position.elevation
        )

        # If sun is behind horizon, it's fully shaded
        shading_loss = 0.0 if is_visible else 1.0

        logger.debug(
            f"Far shading at az={sun_position.azimuth:.1f}°, "
            f"el={sun_position.elevation:.1f}°: {shading_loss:.3f}"
        )

        return shading_loss

    def electrical_shading_effects(
        self,
        shading_fraction: float,
        module_id: int
    ) -> Dict[str, float]:
        """
        Calculate electrical effects of shading including bypass diode activation.

        Args:
            shading_fraction: Fraction of module that is shaded (0-1)
            module_id: Module identifier

        Returns:
            Dictionary with electrical loss factors
        """
        # This is a simplified interface - full implementation in ElectricalShadingModel
        # Here we provide a basic electrical loss estimation

        if shading_fraction <= 0:
            return {
                "power_loss": 0.0,
                "voltage_loss": 0.0,
                "current_loss": 0.0
            }

        # Simple non-linear shading loss model
        # Full shading of one cell can cause entire substring to be bypassed
        if shading_fraction < 0.1:
            # Minor shading - proportional loss
            power_loss = shading_fraction
        elif shading_fraction < 0.33:
            # Moderate shading - one bypass diode activated
            power_loss = 0.33
        elif shading_fraction < 0.67:
            # Significant shading - two bypass diodes activated
            power_loss = 0.67
        else:
            # Severe shading - entire module affected
            power_loss = 1.0

        return {
            "power_loss": power_loss,
            "voltage_loss": shading_fraction * 0.33,  # Approximate voltage drop
            "current_loss": shading_fraction
        }

    def backtracking_optimization(
        self,
        sun_position: SunPosition,
        nominal_tilt: Optional[float] = None
    ) -> float:
        """
        Calculate optimal tracker angle using backtracking algorithm.

        Backtracking reduces row-to-row shading by limiting tracker rotation
        when sun is at low angles.

        Args:
            sun_position: Current sun position
            nominal_tilt: Nominal tracking angle (if None, calculated from sun)

        Returns:
            Optimized tracker angle in degrees
        """
        if self.array_geometry.tracker_type != TrackerType.SINGLE_AXIS:
            # Not a tracker system - return fixed tilt
            return self.array_geometry.tilt

        if not self.array_geometry.enable_backtracking:
            # Backtracking disabled - use ideal tracking
            if nominal_tilt is not None:
                return nominal_tilt
            else:
                return self._ideal_tracker_angle(sun_position)

        # Calculate ideal tracker angle
        ideal_angle = self._ideal_tracker_angle(sun_position)

        # Calculate backtracking limit
        gcr = self.array_geometry.gcr
        sun_elevation_rad = np.radians(sun_position.elevation)

        # Backtracking angle formula
        # This prevents the trailing edge of one row from shading the next row
        if sun_elevation_rad > 0:
            backtrack_angle_rad = np.arctan(
                np.tan(sun_elevation_rad) * gcr
            )
            backtrack_angle = np.degrees(backtrack_angle_rad)
        else:
            backtrack_angle = 0.0

        # Use minimum of ideal angle and backtracking angle
        optimized_angle = min(abs(ideal_angle), backtrack_angle)

        # Restore sign
        if ideal_angle < 0:
            optimized_angle = -optimized_angle

        # Limit to max tracker angle
        optimized_angle = np.clip(
            optimized_angle,
            -self.array_geometry.tracker_max_angle,
            self.array_geometry.tracker_max_angle
        )

        logger.debug(
            f"Backtracking: ideal={ideal_angle:.1f}°, "
            f"backtrack_limit={backtrack_angle:.1f}°, "
            f"optimized={optimized_angle:.1f}°"
        )

        return optimized_angle

    def irradiance_loss_calculation(
        self,
        timestamp: datetime,
        ghi: float,
        dni: float,
        dhi: float,
        row_index: int = 0,
        total_rows: int = 10
    ) -> Tuple[float, IrradianceComponents]:
        """
        Calculate hourly/subhourly irradiance losses from shading.

        Args:
            timestamp: Datetime for calculation
            ghi: Global horizontal irradiance (W/m²)
            dni: Direct normal irradiance (W/m²)
            dhi: Diffuse horizontal irradiance (W/m²)
            row_index: Row being analyzed
            total_rows: Total number of rows

        Returns:
            Tuple of (total_shading_loss, irradiance_components)
        """
        # Calculate sun position
        sun_position = self.sun_calculator.solar_position_algorithm(timestamp)

        # Calculate tracker angle if applicable
        if self.array_geometry.tracker_type == TrackerType.SINGLE_AXIS:
            current_tilt = self.backtracking_optimization(sun_position)
        else:
            current_tilt = self.array_geometry.tilt

        # Calculate shading losses
        near_shading = self.near_shading_analysis(sun_position, row_index, total_rows)
        far_shading = self.far_shading_analysis(sun_position)

        # Combined shading loss (simplified - assumes independent effects)
        total_shading = 1.0 - (1.0 - near_shading) * (1.0 - far_shading)

        # Calculate POA irradiance
        irradiance = IrradianceComponents(
            timestamp=timestamp,
            ghi=ghi,
            dni=dni,
            dhi=dhi
        )

        irradiance = self.irradiance_calculator.poa_irradiance(
            irradiance,
            sun_position,
            surface_tilt=current_tilt,
            albedo=self.site_model.albedo
        )

        # Apply shading loss to direct component
        # (diffuse component is less affected by shading)
        irradiance.poa_direct *= (1.0 - total_shading)
        irradiance.poa_global = (
            irradiance.poa_direct + irradiance.poa_diffuse + irradiance.poa_ground
        )

        return total_shading, irradiance

    def shading_loss_by_month(
        self,
        irradiance_data: pd.DataFrame,
        row_index: int = 0,
        total_rows: int = 10
    ) -> Dict[int, float]:
        """
        Calculate monthly shading loss breakdown.

        Args:
            irradiance_data: DataFrame with columns ['timestamp', 'ghi', 'dni', 'dhi']
            row_index: Row being analyzed
            total_rows: Total number of rows

        Returns:
            Dictionary mapping month (1-12) to average shading loss fraction
        """
        monthly_losses = {}

        for month in range(1, 13):
            # Filter data for this month
            month_data = irradiance_data[
                irradiance_data['timestamp'].dt.month == month
            ]

            if len(month_data) == 0:
                monthly_losses[month] = 0.0
                continue

            # Calculate shading for each timestep
            shading_losses = []
            irradiance_weights = []

            for _, row in month_data.iterrows():
                shading_loss, irr = self.irradiance_loss_calculation(
                    row['timestamp'],
                    row['ghi'],
                    row['dni'],
                    row['dhi'],
                    row_index,
                    total_rows
                )

                shading_losses.append(shading_loss)
                # Weight by GHI (more important when there's more sun)
                irradiance_weights.append(row['ghi'])

            # Calculate weighted average
            total_weight = sum(irradiance_weights)
            if total_weight > 0:
                weighted_loss = sum(
                    loss * weight
                    for loss, weight in zip(shading_losses, irradiance_weights)
                ) / total_weight
            else:
                weighted_loss = 0.0

            monthly_losses[month] = weighted_loss

        logger.info(f"Calculated monthly shading losses: {monthly_losses}")

        return monthly_losses

    def worst_case_shading(
        self,
        irradiance_data: pd.DataFrame,
        num_worst_modules: int = 10,
        total_rows: int = 10
    ) -> List[int]:
        """
        Identify worst-shaded modules/strings.

        Args:
            irradiance_data: DataFrame with irradiance time series
            num_worst_modules: Number of worst modules to identify
            total_rows: Total number of rows

        Returns:
            List of module/row indices with highest shading losses
        """
        # Calculate annual shading loss for each row
        row_shading_losses = []

        for row_idx in range(total_rows):
            total_loss = 0.0
            total_weight = 0.0

            for _, row in irradiance_data.iterrows():
                shading_loss, _ = self.irradiance_loss_calculation(
                    row['timestamp'],
                    row['ghi'],
                    row['dni'],
                    row['dhi'],
                    row_idx,
                    total_rows
                )

                total_loss += shading_loss * row['ghi']
                total_weight += row['ghi']

            avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
            row_shading_losses.append((row_idx, avg_loss))

        # Sort by shading loss (highest first)
        row_shading_losses.sort(key=lambda x: x[1], reverse=True)

        # Return indices of worst modules
        worst_modules = [idx for idx, loss in row_shading_losses[:num_worst_modules]]

        logger.info(
            f"Identified {len(worst_modules)} worst-shaded modules: {worst_modules}"
        )

        return worst_modules

    # Private helper methods

    def _calculate_row_to_row_shading(
        self,
        sun_position: SunPosition,
        row_index: int,
        total_rows: int
    ) -> float:
        """Calculate row-to-row shading fraction."""
        if row_index == 0:
            # First row - no row-to-row shading
            return 0.0

        if sun_position.elevation <= 0:
            return 1.0

        # Get current tilt (may vary for trackers)
        if self.array_geometry.tracker_type == TrackerType.SINGLE_AXIS:
            tilt = self.backtracking_optimization(sun_position)
        else:
            tilt = self.array_geometry.tilt

        # Calculate shadow length from previous row
        module_height = self.array_geometry.module_height
        row_spacing = self.array_geometry.row_spacing

        # Effective height of tilted module
        effective_height = module_height * np.sin(np.radians(tilt))

        # Shadow length
        sun_elevation_rad = np.radians(sun_position.elevation)
        if sun_elevation_rad > 0:
            shadow_length = effective_height / np.tan(sun_elevation_rad)
        else:
            shadow_length = float('inf')

        # Check if shadow reaches this row
        # Need to consider sun azimuth relative to row orientation
        azimuth_diff = abs(sun_position.azimuth - self.array_geometry.azimuth)
        if azimuth_diff > 180:
            azimuth_diff = 360 - azimuth_diff

        # Project shadow length in row direction
        projected_shadow = shadow_length * np.cos(np.radians(azimuth_diff))

        # Calculate shading fraction
        if projected_shadow <= 0:
            shading_fraction = 0.0
        elif projected_shadow >= row_spacing:
            # Complete shading
            shading_fraction = 1.0
        else:
            # Partial shading
            shading_fraction = projected_shadow / row_spacing

        return min(1.0, shading_fraction)

    def _calculate_obstruction_shading(
        self,
        sun_position: SunPosition,
        row_index: int
    ) -> float:
        """Calculate shading from site obstacles."""
        if not self.site_model.obstacles:
            return 0.0

        # Simplified obstruction shading
        # Full implementation would ray-trace from module to sun through obstacles

        max_shading = 0.0

        for obstacle in self.site_model.obstacles:
            # Calculate if obstacle blocks sun for this row
            # This is simplified - real implementation needs 3D ray tracing
            shading = self._check_obstacle_shadow(obstacle, sun_position, row_index)
            max_shading = max(max_shading, shading)

        return max_shading

    def _check_obstacle_shadow(
        self,
        obstacle,
        sun_position: SunPosition,
        row_index: int
    ) -> float:
        """Check if obstacle casts shadow on module (simplified)."""
        # This is a placeholder for actual 3D shadow calculation
        # Real implementation would use ray tracing or shadow volume methods

        # For now, return 0 (no obstruction shading)
        # Full implementation would check obstacle geometry vs. sun vector
        return 0.0

    def _ideal_tracker_angle(self, sun_position: SunPosition) -> float:
        """Calculate ideal single-axis tracker angle for given sun position."""
        # For single-axis tracking (North-South axis)
        # Tracker angle to point directly at sun

        sun_azimuth_rad = np.radians(sun_position.azimuth)
        sun_elevation_rad = np.radians(sun_position.elevation)

        # Calculate angle from horizontal
        tracker_angle_rad = np.arctan(
            np.tan(sun_elevation_rad) / np.cos(sun_azimuth_rad - np.radians(180))
        )

        tracker_angle = np.degrees(tracker_angle_rad)

        return tracker_angle

    def run_full_analysis(
        self,
        irradiance_data: pd.DataFrame,
        total_rows: int = 10,
        analyze_all_rows: bool = False
    ) -> ShadeAnalysisResult:
        """
        Run complete shade analysis for the configured time period.

        Args:
            irradiance_data: DataFrame with columns ['timestamp', 'ghi', 'dni', 'dhi']
            total_rows: Total number of rows in the array
            analyze_all_rows: If True, analyze all rows; if False, analyze middle row

        Returns:
            ShadeAnalysisResult with complete analysis
        """
        logger.info("Starting full shade analysis")

        # Select row(s) to analyze
        if analyze_all_rows:
            rows_to_analyze = list(range(total_rows))
        else:
            # Analyze middle row as representative
            rows_to_analyze = [total_rows // 2]

        # Initialize results
        all_shading_losses = []
        all_irradiance_data = []

        # Analyze each row
        for row_idx in rows_to_analyze:
            logger.info(f"Analyzing row {row_idx + 1}/{total_rows}")

            for _, row in irradiance_data.iterrows():
                shading_loss, irr = self.irradiance_loss_calculation(
                    row['timestamp'],
                    row['ghi'],
                    row['dni'],
                    row['dhi'],
                    row_idx,
                    total_rows
                )

                # Calculate near and far shading separately
                sun_pos = self.sun_calculator.solar_position_algorithm(row['timestamp'])
                near_shading = self.near_shading_analysis(sun_pos, row_idx, total_rows)
                far_shading = self.far_shading_analysis(sun_pos)

                shading_loss_obj = ShadingLoss(
                    timestamp=row['timestamp'],
                    near_shading_loss=near_shading,
                    far_shading_loss=far_shading,
                    total_shading_loss=shading_loss,
                    shaded_modules=int(shading_loss > 0.5),
                    partially_shaded_modules=int(0 < shading_loss <= 0.5)
                )

                all_shading_losses.append(shading_loss_obj)
                all_irradiance_data.append(irr)

        # Calculate monthly losses
        monthly_losses = self.shading_loss_by_month(
            irradiance_data,
            rows_to_analyze[0],
            total_rows
        )

        # Calculate annual average
        annual_loss = sum(monthly_losses.values()) / 12

        # Identify worst-shaded modules
        worst_modules = self.worst_case_shading(irradiance_data, 10, total_rows)

        # Create result object
        result = ShadeAnalysisResult(
            site_model=self.site_model,
            array_geometry=self.array_geometry,
            config=self.config,
            shading_losses=all_shading_losses,
            monthly_losses=monthly_losses,
            annual_shading_loss=annual_loss,
            worst_shaded_modules=worst_modules,
            irradiance_data=all_irradiance_data
        )

        logger.info(
            f"Shade analysis complete. Annual shading loss: {annual_loss:.2%}"
        )

        return result
