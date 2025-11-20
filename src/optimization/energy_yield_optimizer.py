"""
EnergyYieldOptimizer: Energy production optimization methods.

This module implements optimization methods focused on maximizing
energy yield, including bifacial gains, tracker optimization, and
seasonal adjustments.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from datetime import datetime, timedelta

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    DesignPoint,
)


class EnergyYieldOptimizer:
    """
    Energy yield optimization for PV systems.

    This class provides methods to maximize annual energy production
    through optimization of tilt angles, tracker configurations,
    bifacial gain, and seasonal adjustments.

    Attributes:
        parameters: PV system parameters
        constraints: Optimization constraints
    """

    def __init__(
        self,
        parameters: PVSystemParameters,
        constraints: OptimizationConstraints,
    ):
        """
        Initialize energy yield optimizer.

        Args:
            parameters: PV system parameters
            constraints: Optimization constraints
        """
        self.parameters = parameters
        self.constraints = constraints

    def maximize_annual_energy(
        self,
        method: str = "gradient",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Maximize annual energy production through tilt and azimuth optimization.

        This method optimizes the fixed tilt angle and azimuth to maximize
        annual energy yield based on site latitude and solar resource.

        Args:
            method: Optimization method ('gradient', 'grid_search', 'analytical')

        Returns:
            Tuple of (optimal_annual_energy_kwh, optimal_parameters)
        """
        if method == "analytical":
            # Analytical approximation: optimal tilt ≈ latitude
            optimal_tilt = abs(self.parameters.latitude)
            optimal_tilt = np.clip(
                optimal_tilt,
                self.constraints.min_tilt,
                self.constraints.max_tilt,
            )
            optimal_azimuth = 180.0 if self.parameters.latitude >= 0 else 0.0

            annual_energy = self._calculate_annual_energy(optimal_tilt, optimal_azimuth)

            return annual_energy, {
                "tilt_angle": optimal_tilt,
                "azimuth": optimal_azimuth,
                "method": "analytical",
            }

        elif method == "grid_search":
            # Grid search over tilt and azimuth space
            tilt_range = np.linspace(
                self.constraints.min_tilt,
                self.constraints.max_tilt,
                20,
            )
            azimuth_range = np.linspace(0, 360, 36)

            best_energy = 0.0
            best_tilt = self.parameters.tilt_angle
            best_azimuth = self.parameters.azimuth

            for tilt in tilt_range:
                for azimuth in azimuth_range:
                    energy = self._calculate_annual_energy(tilt, azimuth)
                    if energy > best_energy:
                        best_energy = energy
                        best_tilt = tilt
                        best_azimuth = azimuth

            return best_energy, {
                "tilt_angle": best_tilt,
                "azimuth": best_azimuth,
                "method": "grid_search",
            }

        else:  # gradient-based optimization
            def objective(x: np.ndarray) -> float:
                """Negative energy for minimization."""
                tilt, azimuth = x
                return -self._calculate_annual_energy(tilt, azimuth)

            # Bounds
            bounds = [
                (self.constraints.min_tilt, self.constraints.max_tilt),
                (0, 360),
            ]

            # Initial guess
            x0 = [self.parameters.tilt_angle, self.parameters.azimuth]

            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
            )

            optimal_tilt, optimal_azimuth = result.x
            optimal_energy = -result.fun

            return optimal_energy, {
                "tilt_angle": optimal_tilt,
                "azimuth": optimal_azimuth,
                "method": "gradient",
                "success": result.success,
            }

    def minimize_shading_losses(
        self,
        target_gcr: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Minimize shading losses through row spacing optimization.

        This method optimizes row spacing to minimize inter-row shading
        while considering the target ground coverage ratio.

        Args:
            target_gcr: Target GCR (if None, uses parameter value)

        Returns:
            Tuple of (minimum_shading_loss, optimal_parameters)
        """
        gcr = target_gcr if target_gcr is not None else self.parameters.gcr

        # Calculate optimal row spacing for minimal shading
        # Based on sun elevation angle at worst-case (winter solstice)
        latitude_rad = np.radians(abs(self.parameters.latitude))

        # Solar elevation at winter solstice, solar noon
        declination_winter = np.radians(-23.45)  # Dec 21
        solar_elevation = np.arcsin(
            np.sin(latitude_rad) * np.sin(declination_winter)
            + np.cos(latitude_rad) * np.cos(declination_winter)
        )

        # Module height (assuming landscape orientation)
        module_height = np.sqrt(self.parameters.module_area) * np.sin(
            np.radians(self.parameters.tilt_angle)
        )

        # Shadow length
        shadow_length = module_height / np.tan(solar_elevation) if solar_elevation > 0 else 0

        # Optimal row spacing (shadow length + module length)
        module_length = np.sqrt(self.parameters.module_area) * np.cos(
            np.radians(self.parameters.tilt_angle)
        )
        optimal_spacing = shadow_length + module_length

        # Calculate actual GCR with this spacing
        actual_gcr = module_length / optimal_spacing if optimal_spacing > 0 else gcr

        # Shading loss estimation
        if actual_gcr >= gcr:
            shading_loss = 0.01  # Minimal shading
        else:
            # Increase spacing to match target GCR
            adjusted_spacing = module_length / gcr
            shadow_overlap = max(0, shadow_length - (adjusted_spacing - module_length))
            shading_loss = min(0.3, shadow_overlap / module_length * 0.15)

        return shading_loss, {
            "optimal_row_spacing": optimal_spacing,
            "actual_gcr": actual_gcr,
            "shading_loss": shading_loss,
            "shadow_length": shadow_length,
        }

    def optimize_bifacial_gain(
        self,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize bifacial gain through height and tilt optimization.

        This method optimizes module height above ground and tilt angle
        to maximize rear-side irradiance collection for bifacial modules.

        Returns:
            Tuple of (bifacial_gain_fraction, optimal_parameters)
        """
        if not self.parameters.bifacial:
            return 0.0, {"bifacial_gain": 0.0, "note": "Modules are not bifacial"}

        def calculate_bifacial_gain(tilt: float, height: float) -> float:
            """
            Calculate bifacial gain based on empirical model.

            Args:
                tilt: Tilt angle in degrees
                height: Module height above ground in meters

            Returns:
                Bifacial gain fraction
            """
            # View factor to ground (simplified)
            view_factor = 0.5 * (1 + np.cos(np.radians(tilt)))

            # Height factor (higher modules see more reflected light)
            height_factor = np.tanh(height / 2.0)  # Saturates around 2m

            # Albedo contribution
            rear_irradiance_fraction = (
                view_factor * height_factor * self.parameters.albedo
            )

            # Bifacial gain
            gain = rear_irradiance_fraction * self.parameters.bifaciality

            return gain

        # Optimize over tilt angle
        def objective(x: float) -> float:
            """Negative gain for minimization."""
            tilt = x
            height = 1.5  # Fixed height assumption
            return -calculate_bifacial_gain(tilt, height)

        result = minimize_scalar(
            objective,
            bounds=(self.constraints.min_tilt, self.constraints.max_tilt),
            method='bounded',
        )

        optimal_tilt = result.x
        optimal_gain = -result.fun

        return optimal_gain, {
            "optimal_tilt_for_bifacial": optimal_tilt,
            "bifacial_gain": optimal_gain,
            "bifaciality": self.parameters.bifaciality,
            "albedo": self.parameters.albedo,
        }

    def optimize_tracker_angles(
        self,
        num_samples: int = 24,
    ) -> Dict[str, np.ndarray]:
        """
        Optimize tracker angles throughout the day for single-axis tracking.

        This method calculates optimal tracker rotation angles to maximize
        direct normal irradiance collection.

        Args:
            num_samples: Number of time samples per day

        Returns:
            Dictionary with hourly tracker angles and expected energy gain
        """
        if self.parameters.tracker_type == "fixed":
            return {
                "tracker_angles": np.zeros(num_samples),
                "energy_gain": 0.0,
                "note": "System uses fixed tilt",
            }

        # Generate hourly sun positions (simplified)
        hours = np.linspace(6, 18, num_samples)  # 6 AM to 6 PM
        tracker_angles = []

        for hour in hours:
            # Solar position calculation (simplified)
            hour_angle = 15 * (hour - 12)  # degrees from solar noon

            if self.parameters.tracker_type == "single_axis":
                # Rotate to track sun azimuth
                tracker_angle = hour_angle
                tracker_angles.append(np.clip(tracker_angle, -60, 60))
            elif self.parameters.tracker_type == "dual_axis":
                # Both azimuth and elevation tracking
                tracker_angle = hour_angle
                tracker_angles.append(tracker_angle)
            else:
                tracker_angles.append(0)

        # Estimate energy gain from tracking
        if self.parameters.tracker_type == "single_axis":
            tracking_gain = 0.20  # 20% typical gain
        elif self.parameters.tracker_type == "dual_axis":
            tracking_gain = 0.35  # 35% typical gain
        else:
            tracking_gain = 0.0

        return {
            "hours": hours,
            "tracker_angles": np.array(tracker_angles),
            "tracking_gain": tracking_gain,
            "tracker_type": self.parameters.tracker_type,
        }

    def seasonal_optimization(
        self,
    ) -> Dict[int, float]:
        """
        Optimize tilt angles for each season to maximize annual energy.

        This method calculates optimal tilt angles for different seasons,
        useful for seasonally-adjustable mounting systems.

        Returns:
            Dictionary mapping month to optimal tilt angle
        """
        # Season definitions (Northern hemisphere)
        seasons = {
            "winter": [12, 1, 2],  # Dec, Jan, Feb
            "spring": [3, 4, 5],   # Mar, Apr, May
            "summer": [6, 7, 8],   # Jun, Jul, Aug
            "fall": [9, 10, 11],   # Sep, Oct, Nov
        }

        # Solar declination by season
        declinations = {
            "winter": -23.45,  # Winter solstice
            "spring": 0.0,     # Equinox
            "summer": 23.45,   # Summer solstice
            "fall": 0.0,       # Equinox
        }

        seasonal_tilts = {}

        for season, months in seasons.items():
            declination = declinations[season]

            # Optimal tilt ≈ latitude - declination
            optimal_tilt = abs(self.parameters.latitude) - declination

            # Constrain to allowed range
            optimal_tilt = np.clip(
                optimal_tilt,
                self.constraints.min_tilt,
                self.constraints.max_tilt,
            )

            for month in months:
                seasonal_tilts[month] = optimal_tilt

        return seasonal_tilts

    def _calculate_annual_energy(
        self,
        tilt: float,
        azimuth: float,
    ) -> float:
        """
        Calculate annual energy production for given tilt and azimuth.

        This is a simplified model. For production use, integrate with
        pvlib or similar solar simulation library.

        Args:
            tilt: Tilt angle in degrees
            azimuth: Azimuth angle in degrees

        Returns:
            Annual energy in kWh
        """
        # System capacity
        capacity_kw = self.parameters.num_modules * self.parameters.module_power / 1000

        # Base peak sun hours (varies with latitude)
        base_psh = 5.0 - 0.02 * abs(self.parameters.latitude)

        # Tilt factor (optimal near latitude)
        tilt_factor = np.cos(np.radians(abs(tilt - abs(self.parameters.latitude))))

        # Azimuth factor (optimal is 180° in N hemisphere, 0° in S hemisphere)
        optimal_azimuth = 180.0 if self.parameters.latitude >= 0 else 0.0
        azimuth_factor = np.cos(np.radians(azimuth - optimal_azimuth))
        azimuth_factor = max(0.7, azimuth_factor)  # Minimum 70% even if poorly oriented

        # Effective peak sun hours
        effective_psh = base_psh * tilt_factor * azimuth_factor

        # Annual energy
        annual_energy_kwh = (
            capacity_kw
            * effective_psh
            * 365
            * self.parameters.inverter_efficiency
            * 0.95  # System losses
        )

        return annual_energy_kwh
