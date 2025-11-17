"""
LayoutOptimizer: Physical layout optimization for PV systems.

This module implements optimization methods for physical system layout
including GCR, land use, capacity maximization, and string configuration.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    DesignPoint,
)


class LayoutOptimizer:
    """
    Physical layout optimization for PV systems.

    This class provides methods to optimize the physical arrangement
    of PV modules including ground coverage ratio, land use minimization,
    capacity maximization, and electrical string configuration.

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
        Initialize layout optimizer.

        Args:
            parameters: PV system parameters
            constraints: Optimization constraints
        """
        self.parameters = parameters
        self.constraints = constraints

    def optimize_gcr(
        self,
        objective: str = "energy_per_area",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize ground coverage ratio (GCR).

        GCR is the ratio of module area to ground area. Higher GCR means
        more dense packing but potentially more shading losses.

        Args:
            objective: Optimization objective
                - 'energy_per_area': Maximize energy per land area
                - 'lcoe': Minimize LCOE
                - 'total_energy': Maximize total energy

        Returns:
            Tuple of (optimal_gcr, performance_metrics)
        """
        def calculate_objective(gcr: float) -> float:
            """Calculate objective for given GCR."""
            # Module area and land area
            module_area_total = self.parameters.num_modules * self.parameters.module_area
            land_area_m2 = module_area_total / gcr
            land_area_acres = land_area_m2 / 4046.86

            # Check land availability
            if land_area_acres > self.parameters.available_land_acres:
                return 1e9  # Penalty for exceeding available land

            # Energy calculation
            capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6

            # Shading loss increases with GCR
            shading_loss = max(0, (gcr - 0.35) * 0.2)

            # Base annual energy
            psh = 5.0 - 0.02 * abs(self.parameters.latitude)
            annual_energy_kwh = (
                capacity_mw
                * 1000
                * psh
                * 365
                * self.parameters.inverter_efficiency
                * (1 - shading_loss)
                * 0.95
            )

            if objective == "energy_per_area":
                # Maximize energy per land area
                energy_per_acre = annual_energy_kwh / land_area_acres
                return -energy_per_acre  # Negative for minimization

            elif objective == "lcoe":
                # Minimize LCOE
                land_cost = land_area_acres * self.parameters.land_cost_per_acre
                module_cost = self.parameters.num_modules * self.parameters.module_cost
                inverter_cost = (
                    capacity_mw * 1000 / self.parameters.dc_ac_ratio
                    * self.parameters.inverter_cost_per_kw
                )
                total_capex = module_cost + inverter_cost + land_cost + capacity_mw * 200000

                total_energy = sum(
                    annual_energy_kwh * (1 - self.parameters.degradation_rate) ** year
                    for year in range(self.parameters.project_lifetime)
                )

                lcoe = total_capex / total_energy if total_energy > 0 else 999
                return lcoe

            else:  # total_energy
                return -annual_energy_kwh

        # Optimize
        result = minimize_scalar(
            calculate_objective,
            bounds=(self.constraints.min_gcr, self.constraints.max_gcr),
            method='bounded',
        )

        optimal_gcr = result.x
        optimal_value = result.fun

        # Calculate metrics at optimal GCR
        module_area_total = self.parameters.num_modules * self.parameters.module_area
        land_area_acres = (module_area_total / optimal_gcr) / 4046.86
        shading_loss = max(0, (optimal_gcr - 0.35) * 0.2)

        capacity_mw = self.parameters.num_modules * self.parameters.module_power / 1e6
        psh = 5.0 - 0.02 * abs(self.parameters.latitude)
        annual_energy_kwh = (
            capacity_mw * 1000 * psh * 365
            * self.parameters.inverter_efficiency
            * (1 - shading_loss)
            * 0.95
        )

        return optimal_gcr, {
            "optimal_gcr": optimal_gcr,
            "land_use_acres": land_area_acres,
            "shading_loss": shading_loss,
            "annual_energy_kwh": annual_energy_kwh,
            "energy_per_acre": annual_energy_kwh / land_area_acres,
            "objective": objective,
            "success": result.success if hasattr(result, 'success') else True,
        }

    def minimize_land_use(
        self,
        min_capacity_mw: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Minimize land use while meeting capacity requirements.

        Args:
            min_capacity_mw: Minimum required capacity in MW

        Returns:
            Tuple of (minimum_land_acres, design_parameters)
        """
        target_capacity = (
            min_capacity_mw if min_capacity_mw is not None
            else self.parameters.num_modules * self.parameters.module_power / 1e6
        )

        # Calculate required number of modules
        required_modules = int(target_capacity * 1e6 / self.parameters.module_power)

        # Maximize GCR to minimize land use
        optimal_gcr = self.constraints.max_gcr

        # Calculate land use
        module_area_total = required_modules * self.parameters.module_area
        land_area_m2 = module_area_total / optimal_gcr
        land_area_acres = land_area_m2 / 4046.86

        # Check feasibility
        if land_area_acres > self.parameters.available_land_acres:
            return land_area_acres, {
                "feasible": False,
                "land_use_acres": land_area_acres,
                "available_acres": self.parameters.available_land_acres,
                "deficit_acres": land_area_acres - self.parameters.available_land_acres,
            }

        # Calculate shading impact
        shading_loss = max(0, (optimal_gcr - 0.35) * 0.2)

        return land_area_acres, {
            "feasible": True,
            "land_use_acres": land_area_acres,
            "gcr": optimal_gcr,
            "num_modules": required_modules,
            "capacity_mw": target_capacity,
            "shading_loss": shading_loss,
        }

    def maximize_capacity(
        self,
        available_land_acres: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Maximize system capacity within land constraints.

        Args:
            available_land_acres: Available land in acres (if None, uses parameter)

        Returns:
            Tuple of (maximum_capacity_mw, design_parameters)
        """
        land_acres = (
            available_land_acres if available_land_acres is not None
            else self.parameters.available_land_acres
        )

        # Maximize GCR to fit more modules
        optimal_gcr = self.constraints.max_gcr

        # Calculate maximum number of modules
        land_area_m2 = land_acres * 4046.86
        module_area_available = land_area_m2 * optimal_gcr
        max_modules = int(module_area_available / self.parameters.module_area)

        # System capacity
        max_capacity_mw = max_modules * self.parameters.module_power / 1e6

        # Inverter sizing
        inverter_capacity_mw = max_capacity_mw / self.parameters.dc_ac_ratio

        # Shading losses
        shading_loss = max(0, (optimal_gcr - 0.35) * 0.2)

        return max_capacity_mw, {
            "capacity_mw": max_capacity_mw,
            "dc_capacity_mw": max_capacity_mw,
            "ac_capacity_mw": inverter_capacity_mw,
            "num_modules": max_modules,
            "gcr": optimal_gcr,
            "land_use_acres": land_acres,
            "shading_loss": shading_loss,
        }

    def optimize_string_configuration(
        self,
        inverter_mppt_voltage_range: Tuple[float, float] = (600, 1500),
        module_voltage: float = 40.0,
    ) -> Dict[str, any]:
        """
        Optimize string configuration for electrical design.

        This method determines optimal number of modules per string
        and strings per inverter based on voltage constraints.

        Args:
            inverter_mppt_voltage_range: MPPT voltage range (Vmin, Vmax) in volts
            module_voltage: Module voltage at MPP in volts

        Returns:
            Dictionary with optimal string configuration
        """
        v_min, v_max = inverter_mppt_voltage_range

        # Temperature coefficients (simplified)
        temp_coef = -0.0035  # -0.35%/°C
        max_temp = 70  # °C (cell temperature)
        min_temp = -10  # °C
        ref_temp = 25  # °C

        # Voltage at temperature extremes
        v_module_cold = module_voltage * (1 + temp_coef * (min_temp - ref_temp))
        v_module_hot = module_voltage * (1 + temp_coef * (max_temp - ref_temp))

        # String sizing
        min_modules_per_string = int(np.ceil(v_min / v_module_cold))
        max_modules_per_string = int(np.floor(v_max / v_module_hot))

        # Choose optimal (middle of range for safety margin)
        optimal_modules_per_string = (min_modules_per_string + max_modules_per_string) // 2
        optimal_modules_per_string = max(optimal_modules_per_string, min_modules_per_string)
        optimal_modules_per_string = min(optimal_modules_per_string, max_modules_per_string)

        # String voltage
        string_voltage_nominal = optimal_modules_per_string * module_voltage
        string_voltage_cold = optimal_modules_per_string * v_module_cold
        string_voltage_hot = optimal_modules_per_string * v_module_hot

        # Number of strings
        total_modules = self.parameters.num_modules
        num_strings = int(np.ceil(total_modules / optimal_modules_per_string))

        # Strings per inverter
        capacity_mw = total_modules * self.parameters.module_power / 1e6
        inverter_capacity_kw = capacity_mw * 1000 / self.parameters.dc_ac_ratio
        num_inverters = max(1, int(np.ceil(inverter_capacity_kw / 1000)))  # Assume 1MW inverters
        strings_per_inverter = num_strings // num_inverters

        return {
            "modules_per_string": optimal_modules_per_string,
            "num_strings": num_strings,
            "num_inverters": num_inverters,
            "strings_per_inverter": strings_per_inverter,
            "string_voltage_nominal": string_voltage_nominal,
            "string_voltage_range": (string_voltage_hot, string_voltage_cold),
            "mppt_voltage_range": inverter_mppt_voltage_range,
            "total_modules": total_modules,
            "utilization": (total_modules / (num_strings * optimal_modules_per_string)),
        }

    def terrain_following_optimization(
        self,
        terrain_slope: float = 5.0,
        terrain_aspect: float = 180.0,
    ) -> Dict[str, float]:
        """
        Optimize layout for sloped terrain.

        This method adjusts module tilt and row spacing to account for
        terrain slope while maximizing energy yield and minimizing
        grading costs.

        Args:
            terrain_slope: Terrain slope in degrees (0-30)
            terrain_aspect: Terrain aspect/orientation in degrees (0-360)

        Returns:
            Dictionary with terrain-following layout parameters
        """
        # Adjust tilt angle to account for terrain slope
        base_tilt = self.parameters.tilt_angle

        # If terrain faces the sun, reduce tilt; if away, increase tilt
        optimal_azimuth = 180.0 if self.parameters.latitude >= 0 else 0.0
        azimuth_diff = abs(terrain_aspect - optimal_azimuth)

        if azimuth_diff < 90:  # Terrain faces sun
            tilt_adjustment = -terrain_slope * 0.7
        else:  # Terrain faces away
            tilt_adjustment = terrain_slope * 0.5

        adjusted_tilt = base_tilt + tilt_adjustment
        adjusted_tilt = np.clip(
            adjusted_tilt,
            self.constraints.min_tilt,
            self.constraints.max_tilt,
        )

        # Adjust row spacing for slope
        # Upslope rows can be closer, downslope need more spacing
        base_spacing = self.parameters.row_spacing

        if azimuth_diff < 90:  # Slope helps
            spacing_factor = 1.0 - terrain_slope / 100
        else:  # Slope hinders
            spacing_factor = 1.0 + terrain_slope / 50

        adjusted_spacing = base_spacing * spacing_factor

        # GCR adjustment
        module_length = np.sqrt(self.parameters.module_area)
        adjusted_gcr = module_length / adjusted_spacing

        # Grading cost (cost increases with slope)
        grading_cost_factor = 1.0 + (terrain_slope / 10) ** 2
        base_grading_cost_per_acre = 5000  # USD
        grading_cost_per_acre = base_grading_cost_per_acre * grading_cost_factor

        # Land area
        module_area_total = self.parameters.num_modules * self.parameters.module_area
        land_area_acres = (module_area_total / adjusted_gcr) / 4046.86

        total_grading_cost = land_area_acres * grading_cost_per_acre

        # Energy impact
        # Slope can increase or decrease energy depending on orientation
        energy_factor = 1.0 + np.cos(np.radians(azimuth_diff)) * terrain_slope / 100

        return {
            "adjusted_tilt": adjusted_tilt,
            "tilt_adjustment": tilt_adjustment,
            "adjusted_row_spacing": adjusted_spacing,
            "adjusted_gcr": adjusted_gcr,
            "grading_cost": total_grading_cost,
            "grading_cost_per_acre": grading_cost_per_acre,
            "terrain_slope": terrain_slope,
            "terrain_aspect": terrain_aspect,
            "energy_factor": energy_factor,
            "land_use_acres": land_area_acres,
        }
