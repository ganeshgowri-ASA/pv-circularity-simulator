"""
System Layout Optimizer for PV array design optimization.

This module optimizes array layout parameters including tilt angle, azimuth,
row spacing, and GCR to maximize energy yield while managing shading constraints.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from .models import (
    ArrayGeometry,
    LayoutOptimizationResult,
    Location,
    ShadeAnalysisConfig,
    SiteModel,
    TrackerType,
)
from .shade_analysis import ShadeAnalysisEngine
from .sun_position import SunPositionCalculator
from .irradiance import IrradianceOnSurface

logger = logging.getLogger(__name__)


class SystemLayoutOptimizer:
    """
    Optimize PV system layout for maximum energy yield.

    Performs multi-parameter optimization considering shading, land use,
    and energy production tradeoffs.
    """

    def __init__(
        self,
        site_model: SiteModel,
        config: ShadeAnalysisConfig
    ):
        """
        Initialize the layout optimizer.

        Args:
            site_model: Site model with location and terrain
            config: Shade analysis configuration
        """
        self.site_model = site_model
        self.config = config
        self.sun_calculator = SunPositionCalculator(site_model.location)

        logger.info("Initialized SystemLayoutOptimizer")

    def optimize_row_spacing(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        spacing_range: Tuple[float, float] = (2.0, 10.0),
        num_rows: int = 10
    ) -> LayoutOptimizationResult:
        """
        Optimize row spacing for optimal GCR vs shading tradeoff.

        Args:
            array_geometry: Base array geometry (will optimize spacing)
            irradiance_data: Time-series irradiance data
            spacing_range: (min_spacing, max_spacing) in meters
            num_rows: Number of rows in array

        Returns:
            LayoutOptimizationResult with optimal spacing
        """
        logger.info(f"Optimizing row spacing in range {spacing_range}")

        def objective(spacing: float) -> float:
            """Objective function: maximize annual energy yield."""
            # Update geometry with new spacing
            test_geometry = array_geometry.model_copy()
            test_geometry.row_spacing = spacing

            # Calculate GCR
            pitch = spacing
            module_width = array_geometry.module_width
            gcr = module_width * np.cos(np.radians(array_geometry.tilt)) / pitch
            test_geometry.gcr = gcr

            # Run shade analysis
            energy_yield = self._calculate_annual_energy(
                test_geometry, irradiance_data, num_rows
            )

            # Negative because we minimize (not maximize)
            return -energy_yield

        # Optimize using bounded minimization
        result = minimize(
            objective,
            x0=(spacing_range[0] + spacing_range[1]) / 2,
            bounds=[spacing_range],
            method='L-BFGS-B'
        )

        optimal_spacing = result.x[0]
        optimal_energy = -result.fun

        # Calculate final metrics
        optimal_geometry = array_geometry.model_copy()
        optimal_geometry.row_spacing = optimal_spacing
        optimal_gcr = (
            array_geometry.module_width *
            np.cos(np.radians(array_geometry.tilt)) /
            optimal_spacing
        )
        optimal_geometry.gcr = optimal_gcr

        # Run final shade analysis
        shade_engine = ShadeAnalysisEngine(
            self.site_model, optimal_geometry, self.config
        )
        shade_result = shade_engine.run_full_analysis(irradiance_data, num_rows)

        optimization_result = LayoutOptimizationResult(
            optimal_tilt=array_geometry.tilt,
            optimal_azimuth=array_geometry.azimuth,
            optimal_gcr=optimal_gcr,
            optimal_row_spacing=optimal_spacing,
            annual_energy_yield=optimal_energy,
            capacity_factor=optimal_energy / (365 * 24 * array_geometry.module_width * num_rows),
            shading_loss=shade_result.annual_shading_loss,
            aoi_loss=0.05,  # Placeholder
            soiling_loss=self.config.soiling_loss,
            land_use_efficiency=optimal_energy / (optimal_spacing * array_geometry.module_width * num_rows),
            optimization_iterations=result.nit if hasattr(result, 'nit') else 0,
            convergence_achieved=result.success
        )

        logger.info(
            f"Optimal row spacing: {optimal_spacing:.2f}m, "
            f"GCR: {optimal_gcr:.3f}, "
            f"Annual energy: {optimal_energy:.0f} kWh"
        )

        return optimization_result

    def optimize_tilt_angle(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        tilt_range: Tuple[float, float] = (0.0, 60.0),
        num_rows: int = 10
    ) -> LayoutOptimizationResult:
        """
        Optimize tilt angle for annual energy maximization.

        Args:
            array_geometry: Base array geometry
            irradiance_data: Time-series irradiance data
            tilt_range: (min_tilt, max_tilt) in degrees
            num_rows: Number of rows

        Returns:
            LayoutOptimizationResult with optimal tilt
        """
        logger.info(f"Optimizing tilt angle in range {tilt_range}")

        def objective(tilt: float) -> float:
            """Objective: maximize energy yield."""
            test_geometry = array_geometry.model_copy()
            test_geometry.tilt = tilt

            energy_yield = self._calculate_annual_energy(
                test_geometry, irradiance_data, num_rows
            )

            return -energy_yield

        # Optimize
        result = minimize(
            objective,
            x0=self.site_model.location.latitude,  # Start with latitude
            bounds=[tilt_range],
            method='L-BFGS-B'
        )

        optimal_tilt = result.x[0]
        optimal_energy = -result.fun

        optimal_geometry = array_geometry.model_copy()
        optimal_geometry.tilt = optimal_tilt

        optimization_result = LayoutOptimizationResult(
            optimal_tilt=optimal_tilt,
            optimal_azimuth=array_geometry.azimuth,
            optimal_gcr=array_geometry.gcr,
            optimal_row_spacing=array_geometry.row_spacing,
            annual_energy_yield=optimal_energy,
            capacity_factor=optimal_energy / (365 * 24 * 100),  # Simplified
            shading_loss=0.0,
            aoi_loss=0.05,
            soiling_loss=self.config.soiling_loss,
            land_use_efficiency=optimal_energy / (array_geometry.row_spacing * array_geometry.module_width * num_rows),
            optimization_iterations=result.nit if hasattr(result, 'nit') else 0,
            convergence_achieved=result.success
        )

        logger.info(
            f"Optimal tilt: {optimal_tilt:.1f}°, "
            f"Annual energy: {optimal_energy:.0f} kWh"
        )

        return optimization_result

    def optimize_azimuth(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        azimuth_range: Tuple[float, float] = (135.0, 225.0),
        num_rows: int = 10
    ) -> LayoutOptimizationResult:
        """
        Optimize azimuth for non-south-facing sites.

        Args:
            array_geometry: Base array geometry
            irradiance_data: Time-series irradiance data
            azimuth_range: (min_azimuth, max_azimuth) in degrees
            num_rows: Number of rows

        Returns:
            LayoutOptimizationResult with optimal azimuth
        """
        logger.info(f"Optimizing azimuth in range {azimuth_range}")

        def objective(azimuth: float) -> float:
            test_geometry = array_geometry.model_copy()
            test_geometry.azimuth = azimuth

            energy_yield = self._calculate_annual_energy(
                test_geometry, irradiance_data, num_rows
            )

            return -energy_yield

        result = minimize(
            objective,
            x0=180.0,  # Start with south-facing
            bounds=[azimuth_range],
            method='L-BFGS-B'
        )

        optimal_azimuth = result.x[0]
        optimal_energy = -result.fun

        optimization_result = LayoutOptimizationResult(
            optimal_tilt=array_geometry.tilt,
            optimal_azimuth=optimal_azimuth,
            optimal_gcr=array_geometry.gcr,
            optimal_row_spacing=array_geometry.row_spacing,
            annual_energy_yield=optimal_energy,
            capacity_factor=optimal_energy / (365 * 24 * 100),
            shading_loss=0.0,
            aoi_loss=0.05,
            soiling_loss=self.config.soiling_loss,
            land_use_efficiency=optimal_energy / (array_geometry.row_spacing * array_geometry.module_width * num_rows),
            optimization_iterations=result.nit if hasattr(result, 'nit') else 0,
            convergence_achieved=result.success
        )

        logger.info(
            f"Optimal azimuth: {optimal_azimuth:.1f}°, "
            f"Annual energy: {optimal_energy:.0f} kWh"
        )

        return optimization_result

    def east_west_optimization(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        num_rows: int = 10
    ) -> Dict[str, LayoutOptimizationResult]:
        """
        Optimize split for East-West systems.

        Args:
            array_geometry: Base array geometry
            irradiance_data: Time-series irradiance data
            num_rows: Number of rows

        Returns:
            Dictionary with 'east' and 'west' optimization results
        """
        logger.info("Optimizing East-West system configuration")

        # Optimize east-facing portion
        east_geometry = array_geometry.model_copy()
        east_result = self.optimize_azimuth(
            east_geometry,
            irradiance_data,
            azimuth_range=(45.0, 135.0),
            num_rows=num_rows // 2
        )

        # Optimize west-facing portion
        west_geometry = array_geometry.model_copy()
        west_result = self.optimize_azimuth(
            west_geometry,
            irradiance_data,
            azimuth_range=(225.0, 315.0),
            num_rows=num_rows // 2
        )

        logger.info(
            f"East-West optimization complete: "
            f"East={east_result.optimal_azimuth:.1f}°, "
            f"West={west_result.optimal_azimuth:.1f}°"
        )

        return {
            "east": east_result,
            "west": west_result
        }

    def tracker_optimization(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        num_rows: int = 10
    ) -> LayoutOptimizationResult:
        """
        Optimize single-axis tracker angle limits and backtracking parameters.

        Args:
            array_geometry: Array geometry with tracker configuration
            irradiance_data: Time-series irradiance data
            num_rows: Number of rows

        Returns:
            LayoutOptimizationResult with optimal tracker parameters
        """
        if array_geometry.tracker_type != TrackerType.SINGLE_AXIS:
            raise ValueError("Tracker optimization requires tracker_type=SINGLE_AXIS")

        logger.info("Optimizing tracker parameters")

        def objective(params: np.ndarray) -> float:
            """Objective: maximize energy with tracker parameters."""
            max_angle, gcr = params

            test_geometry = array_geometry.model_copy()
            test_geometry.tracker_max_angle = max_angle
            test_geometry.gcr = gcr

            # Update row spacing based on GCR
            test_geometry.row_spacing = (
                test_geometry.module_width *
                np.cos(np.radians(test_geometry.tilt)) / gcr
            )

            energy_yield = self._calculate_annual_energy(
                test_geometry, irradiance_data, num_rows
            )

            return -energy_yield

        # Optimize using differential evolution (global optimization)
        bounds = [
            (30.0, 60.0),  # max_angle
            (0.2, 0.6)     # gcr
        ]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            disp=False
        )

        optimal_max_angle, optimal_gcr = result.x
        optimal_energy = -result.fun

        optimal_spacing = (
            array_geometry.module_width *
            np.cos(np.radians(array_geometry.tilt)) / optimal_gcr
        )

        optimization_result = LayoutOptimizationResult(
            optimal_tilt=array_geometry.tilt,
            optimal_azimuth=array_geometry.azimuth,
            optimal_gcr=optimal_gcr,
            optimal_row_spacing=optimal_spacing,
            annual_energy_yield=optimal_energy,
            capacity_factor=optimal_energy / (365 * 24 * 100),
            shading_loss=0.0,
            aoi_loss=0.05,
            soiling_loss=self.config.soiling_loss,
            land_use_efficiency=optimal_energy / (optimal_spacing * array_geometry.module_width * num_rows),
            optimization_iterations=result.nit,
            convergence_achieved=result.success
        )

        logger.info(
            f"Optimal tracker parameters: max_angle={optimal_max_angle:.1f}°, "
            f"GCR={optimal_gcr:.3f}, energy={optimal_energy:.0f} kWh"
        )

        return optimization_result

    def shading_constrained_layout(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        max_shading_loss: float = 0.05,
        num_rows: int = 10
    ) -> LayoutOptimizationResult:
        """
        Layout optimization with shading loss constraint.

        Maximizes energy density while ensuring shading loss stays below threshold.

        Args:
            array_geometry: Base array geometry
            irradiance_data: Time-series irradiance data
            max_shading_loss: Maximum allowable shading loss fraction
            num_rows: Number of rows

        Returns:
            LayoutOptimizationResult meeting shading constraint
        """
        logger.info(f"Optimizing layout with shading constraint: {max_shading_loss:.1%}")

        def objective(params: np.ndarray) -> float:
            """Objective: maximize GCR subject to shading constraint."""
            tilt, spacing = params

            test_geometry = array_geometry.model_copy()
            test_geometry.tilt = tilt
            test_geometry.row_spacing = spacing
            test_geometry.gcr = (
                test_geometry.module_width *
                np.cos(np.radians(tilt)) / spacing
            )

            # Calculate shading loss
            shade_engine = ShadeAnalysisEngine(
                self.site_model, test_geometry, self.config
            )
            shade_result = shade_engine.run_full_analysis(irradiance_data, num_rows)

            # Penalty for exceeding shading constraint
            if shade_result.annual_shading_loss > max_shading_loss:
                penalty = 1000 * (shade_result.annual_shading_loss - max_shading_loss)
                return penalty
            else:
                # Maximize GCR (minimize negative GCR)
                return -test_geometry.gcr

        # Optimize
        bounds = [
            (10.0, 45.0),  # tilt
            (2.0, 15.0)    # spacing
        ]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            disp=False
        )

        optimal_tilt, optimal_spacing = result.x
        optimal_gcr = (
            array_geometry.module_width *
            np.cos(np.radians(optimal_tilt)) / optimal_spacing
        )

        # Calculate final energy
        optimal_geometry = array_geometry.model_copy()
        optimal_geometry.tilt = optimal_tilt
        optimal_geometry.row_spacing = optimal_spacing
        optimal_geometry.gcr = optimal_gcr

        optimal_energy = self._calculate_annual_energy(
            optimal_geometry, irradiance_data, num_rows
        )

        optimization_result = LayoutOptimizationResult(
            optimal_tilt=optimal_tilt,
            optimal_azimuth=array_geometry.azimuth,
            optimal_gcr=optimal_gcr,
            optimal_row_spacing=optimal_spacing,
            annual_energy_yield=optimal_energy,
            capacity_factor=optimal_energy / (365 * 24 * 100),
            shading_loss=min(-result.fun, max_shading_loss) if result.fun < 0 else max_shading_loss,
            aoi_loss=0.05,
            soiling_loss=self.config.soiling_loss,
            land_use_efficiency=optimal_energy / (optimal_spacing * array_geometry.module_width * num_rows),
            optimization_iterations=result.nit,
            convergence_achieved=result.success
        )

        logger.info(
            f"Shading-constrained optimization complete: "
            f"tilt={optimal_tilt:.1f}°, spacing={optimal_spacing:.2f}m, "
            f"GCR={optimal_gcr:.3f}"
        )

        return optimization_result

    # Private helper methods

    def _calculate_annual_energy(
        self,
        array_geometry: ArrayGeometry,
        irradiance_data: pd.DataFrame,
        num_rows: int
    ) -> float:
        """
        Calculate annual energy yield for given array geometry.

        Args:
            array_geometry: Array configuration
            irradiance_data: Time-series irradiance data
            num_rows: Number of rows

        Returns:
            Annual energy yield in kWh
        """
        # Initialize shade analysis engine
        shade_engine = ShadeAnalysisEngine(
            self.site_model, array_geometry, self.config
        )

        # Calculate energy for middle row (representative)
        middle_row = num_rows // 2
        total_energy = 0.0

        for _, row in irradiance_data.iterrows():
            shading_loss, irr = shade_engine.irradiance_loss_calculation(
                row['timestamp'],
                row['ghi'],
                row['dni'],
                row['dhi'],
                middle_row,
                num_rows
            )

            # Calculate instantaneous power
            # Simplified: power = POA irradiance * area * efficiency * (1 - losses)
            module_area = array_geometry.module_width * array_geometry.module_height
            efficiency = 0.20  # 20% module efficiency
            system_losses = 0.14  # 14% system losses (inverter, wiring, etc.)

            power_kw = (
                irr.poa_global *
                module_area *
                num_rows *
                array_geometry.modules_per_string *
                efficiency *
                (1 - system_losses) / 1000.0
            )

            # Calculate energy for this timestep (assuming hourly data)
            timestep_hours = self.config.timestep_minutes / 60.0
            energy_kwh = power_kw * timestep_hours

            total_energy += energy_kwh

        logger.debug(f"Calculated annual energy: {total_energy:.0f} kWh")

        return total_energy
