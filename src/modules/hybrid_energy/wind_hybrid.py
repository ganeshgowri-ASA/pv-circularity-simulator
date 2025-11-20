"""
B12-S02: Wind Hybrid Systems
Production-ready wind-solar hybrid system design and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import differential_evolution, minimize
from scipy.stats import weibull_min

from ..core.data_models import (
    WindResourceData,
    HybridSystemConfiguration
)
from ..core.utilities import (
    validate_positive,
    validate_range,
    convert_power
)


class WindHybridDesigner:
    """
    Wind-solar hybrid system designer with resource analysis and optimization.
    """

    def __init__(self, config: HybridSystemConfiguration):
        """
        Initialize wind hybrid designer.

        Args:
            config: Hybrid system configuration
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.solar_capacity_kw <= 0 and self.config.wind_capacity_kw <= 0:
            raise ValueError("Must have either solar or wind capacity")

    def wind_resource(self, resource_data: WindResourceData,
                     hours: int = 8760) -> Dict[str, any]:
        """
        Analyze wind resource characteristics.

        Args:
            resource_data: Wind resource data
            hours: Number of hours to simulate

        Returns:
            Wind resource analysis results
        """
        validate_positive(resource_data.average_wind_speed_ms, "average_wind_speed")

        # Generate synthetic wind speed time series using Weibull distribution
        wind_speeds = weibull_min.rvs(
            resource_data.weibull_k,
            scale=resource_data.weibull_c,
            size=hours
        )

        # Apply height correction if needed (power law)
        if resource_data.hub_height_m != 100:  # Assuming data is at 100m
            reference_height = 100.0
            alpha = 1.0 / 7.0  # Power law exponent
            wind_speeds = wind_speeds * (
                resource_data.hub_height_m / reference_height
            ) ** alpha

        # Calculate wind power density
        wind_power_density = 0.5 * resource_data.air_density * wind_speeds ** 3

        # Statistics
        capacity_factor = self._estimate_wind_capacity_factor(wind_speeds)

        return {
            "wind_speed_timeseries": wind_speeds,
            "average_wind_speed_ms": np.mean(wind_speeds),
            "median_wind_speed_ms": np.median(wind_speeds),
            "p50_wind_speed": np.percentile(wind_speeds, 50),
            "p90_wind_speed": np.percentile(wind_speeds, 90),
            "max_wind_speed": np.max(wind_speeds),
            "average_power_density_w_m2": np.mean(wind_power_density),
            "estimated_capacity_factor": capacity_factor,
            "hours_above_cut_in": np.sum(wind_speeds > 3.0) / hours,
            "hours_above_rated": np.sum(wind_speeds > 12.0) / hours,
            "resource_quality": self._classify_wind_resource(
                resource_data.average_wind_speed_ms
            )
        }

    def _estimate_wind_capacity_factor(self, wind_speeds: np.ndarray) -> float:
        """
        Estimate capacity factor from wind speeds.

        Uses simplified wind turbine power curve.
        """
        # Simplified power curve parameters
        cut_in = 3.0  # m/s
        rated = 12.0  # m/s
        cut_out = 25.0  # m/s

        power_output = np.zeros_like(wind_speeds)

        # Linear region from cut-in to rated
        mask_linear = (wind_speeds >= cut_in) & (wind_speeds < rated)
        power_output[mask_linear] = (wind_speeds[mask_linear] - cut_in) / (rated - cut_in)

        # Rated power region
        mask_rated = (wind_speeds >= rated) & (wind_speeds < cut_out)
        power_output[mask_rated] = 1.0

        return np.mean(power_output)

    def _classify_wind_resource(self, avg_wind_speed: float) -> str:
        """Classify wind resource quality."""
        if avg_wind_speed < 5.0:
            return "Poor"
        elif avg_wind_speed < 6.5:
            return "Marginal"
        elif avg_wind_speed < 7.5:
            return "Good"
        elif avg_wind_speed < 8.5:
            return "Excellent"
        else:
            return "Outstanding"

    def capacity_mix(self,
                    solar_generation: np.ndarray,
                    wind_generation: np.ndarray,
                    load_profile: np.ndarray) -> Dict[str, float]:
        """
        Analyze capacity mix performance.

        Args:
            solar_generation: Solar generation profile (kW)
            wind_generation: Wind generation profile (kW)
            load_profile: Load demand profile (kW)

        Returns:
            Capacity mix analysis results
        """
        # Total generation
        total_generation = solar_generation + wind_generation

        # Calculate metrics
        total_energy_solar = np.sum(solar_generation)
        total_energy_wind = np.sum(wind_generation)
        total_energy = total_energy_solar + total_energy_wind
        total_load = np.sum(load_profile)

        # Penetration levels
        solar_penetration = total_energy_solar / total_energy if total_energy > 0 else 0
        wind_penetration = total_energy_wind / total_energy if total_energy > 0 else 0

        # Complementarity analysis
        correlation = np.corrcoef(solar_generation, wind_generation)[0, 1]

        # Reliability metrics
        shortfall = np.maximum(load_profile - total_generation, 0)
        total_shortfall = np.sum(shortfall)
        shortfall_hours = np.sum(shortfall > 0)

        surplus = np.maximum(total_generation - load_profile, 0)
        total_surplus = np.sum(surplus)

        # Loss of load probability
        lolp = shortfall_hours / len(load_profile)

        # Energy balance
        self_consumption = min(total_energy, total_load)
        self_sufficiency = self_consumption / total_load if total_load > 0 else 0

        return {
            "solar_energy_kwh": total_energy_solar,
            "wind_energy_kwh": total_energy_wind,
            "total_energy_kwh": total_energy,
            "solar_penetration": solar_penetration,
            "wind_penetration": wind_penetration,
            "solar_wind_correlation": correlation,
            "complementarity_score": 1 - abs(correlation),
            "total_shortfall_kwh": total_shortfall,
            "total_surplus_kwh": total_surplus,
            "loss_of_load_probability": lolp,
            "self_sufficiency_ratio": self_sufficiency,
            "capacity_factor_solar": np.mean(solar_generation) / self.config.solar_capacity_kw if self.config.solar_capacity_kw > 0 else 0,
            "capacity_factor_wind": np.mean(wind_generation) / self.config.wind_capacity_kw if self.config.wind_capacity_kw > 0 else 0
        }

    def hybrid_optimization(self,
                           solar_generation_per_kw: np.ndarray,
                           wind_generation_per_kw: np.ndarray,
                           load_profile: np.ndarray,
                           battery_capacity_kwh: float = 0,
                           constraints: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Optimize hybrid system sizing.

        Args:
            solar_generation_per_kw: Solar generation per kW installed
            wind_generation_per_kw: Wind generation per kW installed
            load_profile: Load demand profile
            battery_capacity_kwh: Battery capacity (if any)
            constraints: Optimization constraints

        Returns:
            Optimization results
        """
        if constraints is None:
            constraints = {
                "max_solar_kw": 100000,
                "max_wind_kw": 100000,
                "min_solar_kw": 0,
                "min_wind_kw": 0,
                "solar_cost_per_kw": 1000,
                "wind_cost_per_kw": 1500,
                "battery_cost_per_kwh": 300,
                "grid_electricity_price": 0.10,
                "feed_in_tariff": 0.05
            }

        # Objective function: minimize cost while meeting reliability
        def objective(x):
            solar_kw, wind_kw = x

            # Generation
            solar_gen = solar_generation_per_kw * solar_kw
            wind_gen = wind_generation_per_kw * wind_kw
            total_gen = solar_gen + wind_gen

            # Capital cost
            capex = (solar_kw * constraints["solar_cost_per_kw"] +
                    wind_kw * constraints["wind_cost_per_kw"] +
                    battery_capacity_kwh * constraints["battery_cost_per_kwh"])

            # Energy metrics
            shortfall = np.maximum(load_profile - total_gen, 0)
            surplus = np.maximum(total_gen - load_profile, 0)

            # Cost of unmet demand
            shortfall_cost = np.sum(shortfall) * constraints["grid_electricity_price"]

            # Revenue from surplus
            surplus_revenue = np.sum(surplus) * constraints["feed_in_tariff"]

            # Total cost (normalized)
            total_cost = capex + shortfall_cost * 10 - surplus_revenue

            return total_cost

        # Bounds
        bounds = [
            (constraints["min_solar_kw"], constraints["max_solar_kw"]),
            (constraints["min_wind_kw"], constraints["max_wind_kw"])
        ]

        # Optimization
        if self.config.optimization_objective == "cost":
            result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        else:
            # For other objectives, use multi-objective approach (simplified here)
            result = differential_evolution(objective, bounds, seed=42, maxiter=100)

        optimal_solar_kw, optimal_wind_kw = result.x

        # Evaluate optimal solution
        optimal_solar_gen = solar_generation_per_kw * optimal_solar_kw
        optimal_wind_gen = wind_generation_per_kw * optimal_wind_kw
        optimal_total_gen = optimal_solar_gen + optimal_wind_gen

        metrics = self.capacity_mix(optimal_solar_gen, optimal_wind_gen, load_profile)

        return {
            "optimal_solar_capacity_kw": optimal_solar_kw,
            "optimal_wind_capacity_kw": optimal_wind_kw,
            "solar_fraction": optimal_solar_kw / (optimal_solar_kw + optimal_wind_kw) if (optimal_solar_kw + optimal_wind_kw) > 0 else 0,
            "wind_fraction": optimal_wind_kw / (optimal_solar_kw + optimal_wind_kw) if (optimal_solar_kw + optimal_wind_kw) > 0 else 0,
            "total_capacity_kw": optimal_solar_kw + optimal_wind_kw,
            "capital_cost": (optimal_solar_kw * constraints["solar_cost_per_kw"] +
                           optimal_wind_kw * constraints["wind_cost_per_kw"]),
            "lcoe_estimate": result.fun / np.sum(load_profile) if np.sum(load_profile) > 0 else 0,
            "performance_metrics": metrics,
            "optimization_successful": result.success,
            "optimization_objective": self.config.optimization_objective
        }

    def temporal_complementarity(self,
                                solar_generation: np.ndarray,
                                wind_generation: np.ndarray,
                                time_resolution: str = "hourly") -> Dict[str, any]:
        """
        Analyze temporal complementarity between solar and wind.

        Args:
            solar_generation: Solar generation profile
            wind_generation: Wind generation profile
            time_resolution: Time resolution of data

        Returns:
            Complementarity analysis
        """
        # Normalize to 0-1
        solar_norm = solar_generation / np.max(solar_generation) if np.max(solar_generation) > 0 else solar_generation
        wind_norm = wind_generation / np.max(wind_generation) if np.max(wind_generation) > 0 else wind_generation

        # Overall correlation
        correlation = np.corrcoef(solar_norm, wind_norm)[0, 1]

        # Diurnal patterns
        if len(solar_generation) >= 24:
            hours_per_day = 24 if time_resolution == "hourly" else len(solar_generation) // 365

            # Reshape to analyze by hour of day
            if len(solar_generation) % hours_per_day == 0:
                solar_by_hour = solar_norm.reshape(-1, hours_per_day)
                wind_by_hour = wind_norm.reshape(-1, hours_per_day)

                hourly_correlation = np.array([
                    np.corrcoef(solar_by_hour[:, h], wind_by_hour[:, h])[0, 1]
                    for h in range(hours_per_day)
                ])

                best_complementarity_hour = np.argmin(hourly_correlation)
                worst_complementarity_hour = np.argmax(hourly_correlation)
            else:
                hourly_correlation = None
                best_complementarity_hour = None
                worst_complementarity_hour = None
        else:
            hourly_correlation = None
            best_complementarity_hour = None
            worst_complementarity_hour = None

        # Combined output variability
        solar_std = np.std(solar_norm)
        wind_std = np.std(wind_norm)
        combined_std = np.std(solar_norm + wind_norm)

        variability_reduction = 1 - (combined_std / (solar_std + wind_std)) if (solar_std + wind_std) > 0 else 0

        return {
            "overall_correlation": correlation,
            "complementarity_score": 1 - abs(correlation),
            "hourly_correlation_profile": hourly_correlation.tolist() if hourly_correlation is not None else None,
            "best_complementarity_hour": int(best_complementarity_hour) if best_complementarity_hour is not None else None,
            "worst_complementarity_hour": int(worst_complementarity_hour) if worst_complementarity_hour is not None else None,
            "variability_reduction": variability_reduction,
            "solar_variability": float(solar_std),
            "wind_variability": float(wind_std),
            "combined_variability": float(combined_std)
        }

    def seasonal_analysis(self,
                         solar_generation: np.ndarray,
                         wind_generation: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Analyze seasonal performance of hybrid system.

        Args:
            solar_generation: Annual solar generation (8760 hours)
            wind_generation: Annual wind generation (8760 hours)

        Returns:
            Seasonal analysis results
        """
        if len(solar_generation) != 8760 or len(wind_generation) != 8760:
            raise ValueError("Seasonal analysis requires 8760 hourly values")

        # Define seasons (Northern Hemisphere)
        seasons = {
            "winter": list(range(0, 2160)) + list(range(8016, 8760)),  # Dec-Feb
            "spring": list(range(2160, 4344)),  # Mar-May
            "summer": list(range(4344, 6552)),  # Jun-Aug
            "fall": list(range(6552, 8016))  # Sep-Nov
        }

        results = {}

        for season_name, hours in seasons.items():
            solar_season = solar_generation[hours]
            wind_season = wind_generation[hours]

            results[season_name] = {
                "solar_energy_kwh": float(np.sum(solar_season)),
                "wind_energy_kwh": float(np.sum(wind_season)),
                "total_energy_kwh": float(np.sum(solar_season + wind_season)),
                "solar_fraction": float(np.sum(solar_season) / np.sum(solar_season + wind_season)) if np.sum(solar_season + wind_season) > 0 else 0,
                "average_solar_kw": float(np.mean(solar_season)),
                "average_wind_kw": float(np.mean(wind_season)),
                "correlation": float(np.corrcoef(solar_season, wind_season)[0, 1])
            }

        return results


def calculate_wind_power_curve(wind_speeds: np.ndarray,
                               rated_power_kw: float,
                               cut_in_speed: float = 3.0,
                               rated_speed: float = 12.0,
                               cut_out_speed: float = 25.0) -> np.ndarray:
    """
    Calculate wind turbine power output from wind speeds.

    Args:
        wind_speeds: Wind speed array (m/s)
        rated_power_kw: Rated power of turbine
        cut_in_speed: Cut-in wind speed
        rated_speed: Rated wind speed
        cut_out_speed: Cut-out wind speed

    Returns:
        Power output array (kW)
    """
    power = np.zeros_like(wind_speeds)

    # Below cut-in: no power
    mask_below_cut_in = wind_speeds < cut_in_speed

    # Linear region: cut-in to rated
    mask_linear = (wind_speeds >= cut_in_speed) & (wind_speeds < rated_speed)
    power[mask_linear] = rated_power_kw * (
        (wind_speeds[mask_linear] - cut_in_speed) / (rated_speed - cut_in_speed)
    )

    # Rated region: rated to cut-out
    mask_rated = (wind_speeds >= rated_speed) & (wind_speeds < cut_out_speed)
    power[mask_rated] = rated_power_kw

    # Above cut-out: no power
    mask_above_cut_out = wind_speeds >= cut_out_speed

    return power


__all__ = ["WindHybridDesigner", "calculate_wind_power_curve"]
