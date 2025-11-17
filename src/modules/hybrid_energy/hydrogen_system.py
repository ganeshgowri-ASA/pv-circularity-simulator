"""
B12-S03: Hydrogen Integration & P2X (Power-to-X)
Production-ready hydrogen production, storage, and fuel cell integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..core.data_models import HydrogenSystemSpec
from ..core.utilities import (
    validate_positive,
    validate_range,
    convert_energy
)


class HydrogenSystem:
    """
    Comprehensive hydrogen production and storage system for renewable energy integration.
    """

    # Physical constants
    H2_LHV = 33.33  # kWh/kg - Lower Heating Value
    H2_HHV = 39.41  # kWh/kg - Higher Heating Value
    H2_DENSITY_STP = 0.0899  # kg/m³ at STP

    def __init__(self, spec: HydrogenSystemSpec):
        """
        Initialize hydrogen system.

        Args:
            spec: Hydrogen system specifications
        """
        self.spec = spec
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate system specifications."""
        validate_positive(self.spec.electrolyzer_capacity_kw, "electrolyzer_capacity")
        validate_range(self.spec.electrolyzer_efficiency, 0.5, 0.9, "efficiency")
        validate_positive(self.spec.h2_storage_capacity_kg, "storage_capacity")

    def electrolyzer_sizing(self,
                           excess_energy_profile: np.ndarray,
                           target_h2_production_kg_day: float,
                           operating_hours_per_day: float = 24) -> Dict[str, float]:
        """
        Size electrolyzer based on excess energy and hydrogen demand.

        Args:
            excess_energy_profile: Excess renewable energy (kWh) timeseries
            target_h2_production_kg_day: Target daily H2 production
            operating_hours_per_day: Expected operating hours per day

        Returns:
            Electrolyzer sizing recommendations
        """
        # Energy required to produce target H2
        energy_per_kg_h2 = self.H2_LHV / self.spec.electrolyzer_efficiency
        daily_energy_required = target_h2_production_kg_day * energy_per_kg_h2

        # Required electrolyzer power
        required_power_kw = daily_energy_required / operating_hours_per_day

        # Available excess energy
        daily_excess_energy = np.mean(excess_energy_profile) * 24  # kWh/day average

        # Achievable H2 production with available energy
        achievable_h2_kg_day = (daily_excess_energy * self.spec.electrolyzer_efficiency) / self.H2_LHV

        # Utilization factor
        utilization_factor = min(daily_energy_required / daily_excess_energy, 1.0) if daily_excess_energy > 0 else 0

        # Capacity factor
        capacity_factor = operating_hours_per_day / 24.0

        # Cost estimates
        capex = required_power_kw * self.spec.capex_per_kw
        annual_opex = capex * 0.03  # 3% of capex

        return {
            "required_electrolyzer_capacity_kw": required_power_kw,
            "target_h2_production_kg_day": target_h2_production_kg_day,
            "achievable_h2_production_kg_day": achievable_h2_kg_day,
            "energy_per_kg_h2_kwh": energy_per_kg_h2,
            "utilization_factor": utilization_factor,
            "capacity_factor": capacity_factor,
            "capital_cost": capex,
            "annual_opex": annual_opex,
            "is_oversized": achievable_h2_kg_day > target_h2_production_kg_day
        }

    def h2_storage(self,
                  production_profile: np.ndarray,
                  consumption_profile: np.ndarray,
                  initial_storage_kg: float = 0) -> Dict[str, any]:
        """
        Simulate hydrogen storage dynamics.

        Args:
            production_profile: H2 production rate (kg/hour)
            consumption_profile: H2 consumption rate (kg/hour)
            initial_storage_kg: Initial storage amount

        Returns:
            Storage simulation results
        """
        validate_range(initial_storage_kg, 0, self.spec.h2_storage_capacity_kg, "initial_storage")

        n_steps = len(production_profile)
        storage_level = np.zeros(n_steps)
        storage_level[0] = initial_storage_kg

        # Track statistics
        shortfall = np.zeros(n_steps)
        curtailment = np.zeros(n_steps)

        for t in range(1, n_steps):
            # Net change in storage
            net_change = production_profile[t] - consumption_profile[t]
            new_level = storage_level[t - 1] + net_change

            # Apply storage constraints
            if new_level < 0:
                shortfall[t] = abs(new_level)
                storage_level[t] = 0
            elif new_level > self.spec.h2_storage_capacity_kg:
                curtailment[t] = new_level - self.spec.h2_storage_capacity_kg
                storage_level[t] = self.spec.h2_storage_capacity_kg
            else:
                storage_level[t] = new_level

        # Calculate metrics
        total_production = np.sum(production_profile)
        total_consumption = np.sum(consumption_profile)
        total_shortfall = np.sum(shortfall)
        total_curtailment = np.sum(curtailment)

        # Storage utilization
        avg_storage_level = np.mean(storage_level)
        max_storage_level = np.max(storage_level)
        storage_utilization = avg_storage_level / self.spec.h2_storage_capacity_kg

        # Energy value
        stored_energy_kwh = storage_level * self.H2_LHV

        return {
            "storage_timeseries_kg": storage_level,
            "stored_energy_kwh": stored_energy_kwh,
            "total_production_kg": total_production,
            "total_consumption_kg": total_consumption,
            "total_shortfall_kg": total_shortfall,
            "total_curtailment_kg": total_curtailment,
            "average_storage_kg": avg_storage_level,
            "max_storage_kg": max_storage_level,
            "storage_utilization": storage_utilization,
            "supply_reliability": 1 - (total_shortfall / total_consumption) if total_consumption > 0 else 1.0,
            "curtailment_rate": total_curtailment / total_production if total_production > 0 else 0
        }

    def fuel_cell_integration(self,
                             h2_available_kg: float,
                             power_demand_kw: float,
                             duration_hours: float) -> Dict[str, float]:
        """
        Model fuel cell operation for power generation from hydrogen.

        Args:
            h2_available_kg: Available hydrogen
            power_demand_kw: Required power output
            duration_hours: Operating duration

        Returns:
            Fuel cell operation results
        """
        if self.spec.fuel_cell_capacity_kw is None:
            raise ValueError("Fuel cell capacity not specified")

        validate_positive(duration_hours, "duration_hours")

        # Check if fuel cell can meet demand
        actual_power_kw = min(power_demand_kw, self.spec.fuel_cell_capacity_kw)

        # Energy to be delivered
        energy_output_kwh = actual_power_kw * duration_hours

        # H2 consumption (accounting for efficiency)
        h2_consumed_kg = energy_output_kwh / (self.H2_LHV * self.spec.fuel_cell_efficiency)

        # Check H2 availability
        if h2_consumed_kg > h2_available_kg:
            h2_consumed_kg = h2_available_kg
            actual_energy_kwh = h2_available_kg * self.H2_LHV * self.spec.fuel_cell_efficiency
            actual_power_kw = actual_energy_kwh / duration_hours
            is_constrained = True
        else:
            actual_energy_kwh = energy_output_kwh
            is_constrained = False

        # Remaining H2
        h2_remaining_kg = h2_available_kg - h2_consumed_kg

        # Efficiency and heat recovery
        waste_heat_kwh = actual_energy_kwh * ((1 / self.spec.fuel_cell_efficiency) - 1)

        return {
            "power_output_kw": actual_power_kw,
            "energy_output_kwh": actual_energy_kwh,
            "h2_consumed_kg": h2_consumed_kg,
            "h2_remaining_kg": h2_remaining_kg,
            "fuel_cell_efficiency": self.spec.fuel_cell_efficiency,
            "waste_heat_kwh": waste_heat_kwh,
            "is_h2_constrained": is_constrained,
            "capacity_factor": actual_power_kw / self.spec.fuel_cell_capacity_kw if self.spec.fuel_cell_capacity_kw > 0 else 0
        }

    def p2x_economic_analysis(self,
                             annual_h2_production_kg: float,
                             electricity_cost_per_kwh: float,
                             h2_selling_price_per_kg: Optional[float] = None,
                             project_lifetime_years: int = 20,
                             discount_rate: float = 0.08) -> Dict[str, float]:
        """
        Economic analysis of power-to-X system.

        Args:
            annual_h2_production_kg: Annual hydrogen production
            electricity_cost_per_kwh: Cost of electricity for electrolysis
            h2_selling_price_per_kg: H2 selling price (if None, calculated)
            project_lifetime_years: Project lifetime
            discount_rate: Discount rate

        Returns:
            Economic analysis results
        """
        # Production cost
        energy_per_kg = self.H2_LHV / self.spec.electrolyzer_efficiency
        electricity_cost_per_kg = energy_per_kg * electricity_cost_per_kwh

        # Capital costs
        electrolyzer_capex = self.spec.electrolyzer_capacity_kw * self.spec.capex_per_kw
        storage_capex = self.spec.h2_storage_capacity_kg * 500  # $/kg storage cost
        total_capex = electrolyzer_capex + storage_capex

        if self.spec.fuel_cell_capacity_kw:
            total_capex += self.spec.fuel_cell_capacity_kw * 1500  # $/kW for fuel cell

        # Annual costs
        annual_opex = total_capex * 0.03  # 3% of capex
        annual_electricity_cost = annual_h2_production_kg * electricity_cost_per_kg
        total_annual_cost = annual_opex + annual_electricity_cost

        # Levelized cost of hydrogen (LCOH)
        lcoh = total_annual_cost / annual_h2_production_kg if annual_h2_production_kg > 0 else 0

        # NPV analysis (if selling price provided)
        if h2_selling_price_per_kg:
            annual_revenue = annual_h2_production_kg * h2_selling_price_per_kg
            annual_profit = annual_revenue - total_annual_cost

            npv = -total_capex
            for year in range(1, project_lifetime_years + 1):
                npv += annual_profit / ((1 + discount_rate) ** year)

            payback_period = total_capex / annual_profit if annual_profit > 0 else None
            roi = (npv + total_capex) / total_capex if total_capex > 0 else 0
        else:
            npv = None
            payback_period = None
            roi = None

        return {
            "levelized_cost_h2_per_kg": lcoh,
            "electricity_cost_per_kg_h2": electricity_cost_per_kg,
            "total_capex": total_capex,
            "electrolyzer_capex": electrolyzer_capex,
            "storage_capex": storage_capex,
            "annual_opex": annual_opex,
            "annual_electricity_cost": annual_electricity_cost,
            "total_annual_cost": total_annual_cost,
            "npv": npv,
            "payback_period_years": payback_period,
            "roi": roi,
            "is_profitable": npv > 0 if npv is not None else None
        }

    def calculate_round_trip_efficiency(self) -> float:
        """
        Calculate round-trip efficiency for H2 storage cycle.

        Returns:
            Round-trip efficiency (electricity -> H2 -> electricity)
        """
        if self.spec.fuel_cell_efficiency is None:
            raise ValueError("Fuel cell efficiency required for round-trip calculation")

        return self.spec.electrolyzer_efficiency * self.spec.fuel_cell_efficiency

    def storage_energy_density(self) -> Dict[str, float]:
        """
        Calculate hydrogen storage energy density metrics.

        Returns:
            Energy density metrics
        """
        # Mass-based
        mass_energy_density_kwh_kg = self.H2_LHV

        # Volume-based (depends on pressure)
        # Simplified calculation for compressed hydrogen
        if self.spec.storage_pressure_bar <= 200:
            volumetric_density_kg_m3 = 15  # kg/m³ at 200 bar
        elif self.spec.storage_pressure_bar <= 350:
            volumetric_density_kg_m3 = 23  # kg/m³ at 350 bar
        elif self.spec.storage_pressure_bar <= 700:
            volumetric_density_kg_m3 = 40  # kg/m³ at 700 bar
        else:
            volumetric_density_kg_m3 = 40  # Assume similar to 700 bar

        volumetric_energy_density_kwh_m3 = volumetric_density_kg_m3 * self.H2_LHV

        # Storage volume required
        storage_volume_m3 = self.spec.h2_storage_capacity_kg / volumetric_density_kg_m3

        return {
            "mass_energy_density_kwh_kg": mass_energy_density_kwh_kg,
            "volumetric_density_kg_m3": volumetric_density_kg_m3,
            "volumetric_energy_density_kwh_m3": volumetric_energy_density_kwh_m3,
            "storage_volume_m3": storage_volume_m3,
            "total_energy_capacity_kwh": self.spec.h2_storage_capacity_kg * self.H2_LHV,
            "storage_pressure_bar": self.spec.storage_pressure_bar
        }


def calculate_h2_compression_energy(mass_kg: float,
                                   initial_pressure_bar: float = 1.0,
                                   final_pressure_bar: float = 350) -> float:
    """
    Calculate energy required for hydrogen compression.

    Args:
        mass_kg: Mass of hydrogen to compress
        initial_pressure_bar: Initial pressure
        final_pressure_bar: Final pressure

    Returns:
        Compression energy in kWh
    """
    # Simplified isothermal compression work
    # W = n * R * T * ln(P2/P1)
    # For hydrogen: approximately 2.2 kWh per kg H2 for compression to 350 bar

    if final_pressure_bar <= initial_pressure_bar:
        return 0.0

    # Empirical approximation
    compression_ratio = final_pressure_bar / initial_pressure_bar

    if final_pressure_bar <= 350:
        specific_energy = 2.2  # kWh/kg
    elif final_pressure_bar <= 700:
        specific_energy = 3.0  # kWh/kg
    else:
        specific_energy = 3.5  # kWh/kg

    return mass_kg * specific_energy * np.log(compression_ratio) / np.log(350)


__all__ = ["HydrogenSystem", "calculate_h2_compression_energy"]
