"""
B12-S01: Battery Integration & Energy Storage
Production-ready battery system modeling with sizing, charge/discharge, and arbitrage optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from scipy.optimize import minimize, linprog

from ..core.data_models import (
    BatterySpecification,
    BatteryOperationSchedule,
    EnergyStorageTechnology
)
from ..core.utilities import (
    validate_positive,
    validate_range,
    present_value
)


class BatteryIntegrator:
    """
    Comprehensive battery energy storage system integration and optimization.
    """

    def __init__(self, specification: BatterySpecification):
        """
        Initialize battery integrator.

        Args:
            specification: Battery system specifications
        """
        self.spec = specification
        self._validate_specification()

    def _validate_specification(self) -> None:
        """Validate battery specification parameters."""
        validate_positive(self.spec.capacity_kwh, "capacity_kwh")
        validate_positive(self.spec.power_rating_kw, "power_rating_kw")
        validate_range(self.spec.round_trip_efficiency, 0.7, 1.0, "efficiency")
        validate_range(self.spec.depth_of_discharge, 0.1, 1.0, "depth_of_discharge")

    def sizing(self, load_profile: np.ndarray, generation_profile: np.ndarray,
               autonomy_hours: float = 4.0,
               sizing_method: str = "peak_shaving") -> Dict[str, float]:
        """
        Size battery system based on load and generation profiles.

        Args:
            load_profile: Load demand in kW (hourly)
            generation_profile: Generation output in kW (hourly)
            autonomy_hours: Required backup autonomy
            sizing_method: Method for sizing ('peak_shaving', 'autonomy', 'hybrid')

        Returns:
            Dictionary with sizing recommendations
        """
        net_load = load_profile - generation_profile

        if sizing_method == "peak_shaving":
            capacity_kwh = self._size_for_peak_shaving(net_load)
        elif sizing_method == "autonomy":
            capacity_kwh = self._size_for_autonomy(load_profile, autonomy_hours)
        elif sizing_method == "hybrid":
            peak_capacity = self._size_for_peak_shaving(net_load)
            autonomy_capacity = self._size_for_autonomy(load_profile, autonomy_hours)
            capacity_kwh = max(peak_capacity, autonomy_capacity)
        else:
            raise ValueError(f"Unknown sizing method: {sizing_method}")

        # Calculate power rating based on C-rate (typically 0.5C to 1C)
        c_rate = 0.5  # 2-hour discharge
        power_rating_kw = capacity_kwh * c_rate

        # Calculate usable capacity considering DoD
        usable_capacity_kwh = capacity_kwh * self.spec.depth_of_discharge

        # Estimate costs
        capital_cost = capacity_kwh * self.spec.initial_cost_per_kwh
        power_cost = power_rating_kw * 150  # $/kW for power electronics

        return {
            "recommended_capacity_kwh": capacity_kwh,
            "recommended_power_kw": power_rating_kw,
            "usable_capacity_kwh": usable_capacity_kwh,
            "c_rate": c_rate,
            "capital_cost_usd": capital_cost + power_cost,
            "cost_per_kwh": self.spec.initial_cost_per_kwh,
            "sizing_method": sizing_method,
            "autonomy_hours": autonomy_hours
        }

    def _size_for_peak_shaving(self, net_load: np.ndarray) -> float:
        """Size battery for peak shaving application."""
        # Find peak excess generation and peak demand
        excess_generation = np.maximum(-net_load, 0)
        excess_demand = np.maximum(net_load, 0)

        # Size to capture average of top 10% peaks
        gen_threshold = np.percentile(excess_generation, 90)
        dem_threshold = np.percentile(excess_demand, 90)

        avg_excess_gen = np.mean(excess_generation[excess_generation > gen_threshold])
        avg_excess_dem = np.mean(excess_demand[excess_demand > dem_threshold])

        # Battery should handle larger of the two
        return max(avg_excess_gen, avg_excess_dem)

    def _size_for_autonomy(self, load_profile: np.ndarray,
                          autonomy_hours: float) -> float:
        """Size battery for autonomy/backup requirement."""
        # Use average load for the autonomy period
        avg_load = np.mean(load_profile)
        capacity = avg_load * autonomy_hours / self.spec.depth_of_discharge
        return capacity

    def charge_discharge(self, initial_soc: float,
                        power_kw: float,
                        duration_hours: float) -> Dict[str, float]:
        """
        Simulate charge or discharge operation.

        Args:
            initial_soc: Initial state of charge (0-1)
            power_kw: Power (positive=discharge, negative=charge)
            duration_hours: Duration in hours

        Returns:
            Dictionary with operation results
        """
        validate_range(initial_soc, 0, 1, "initial_soc")

        # Determine if charging or discharging
        is_discharging = power_kw > 0

        # Apply efficiency
        if is_discharging:
            efficiency = np.sqrt(self.spec.round_trip_efficiency)  # Discharge efficiency
        else:
            efficiency = np.sqrt(self.spec.round_trip_efficiency)  # Charge efficiency

        # Calculate energy transferred
        if is_discharging:
            energy_from_battery = abs(power_kw) * duration_hours
            energy_delivered = energy_from_battery * efficiency
        else:
            energy_input = abs(power_kw) * duration_hours
            energy_to_battery = energy_input * efficiency

        # Calculate SoC change
        if is_discharging:
            soc_change = -energy_from_battery / self.spec.capacity_kwh
        else:
            soc_change = energy_to_battery / self.spec.capacity_kwh

        # Calculate final SoC
        final_soc = initial_soc + soc_change

        # Check constraints
        min_soc = 1 - self.spec.depth_of_discharge
        max_soc = 1.0

        if final_soc < min_soc:
            final_soc = min_soc
            actual_soc_change = final_soc - initial_soc
            is_constrained = True
        elif final_soc > max_soc:
            final_soc = max_soc
            actual_soc_change = final_soc - initial_soc
            is_constrained = True
        else:
            actual_soc_change = soc_change
            is_constrained = False

        # Calculate actual energy transferred
        if is_discharging:
            actual_energy = abs(actual_soc_change) * self.spec.capacity_kwh
            actual_power = actual_energy / duration_hours if duration_hours > 0 else 0
        else:
            actual_energy = abs(actual_soc_change) * self.spec.capacity_kwh / efficiency
            actual_power = actual_energy / duration_hours if duration_hours > 0 else 0

        # Calculate degradation
        cycles = abs(actual_soc_change) * self.spec.depth_of_discharge
        degradation = cycles / self.spec.cycle_life

        return {
            "initial_soc": initial_soc,
            "final_soc": final_soc,
            "soc_change": actual_soc_change,
            "energy_transferred_kwh": actual_energy,
            "actual_power_kw": actual_power,
            "efficiency": efficiency,
            "is_constrained": is_constrained,
            "cycles_consumed": cycles,
            "degradation_fraction": degradation,
            "operation_type": "discharge" if is_discharging else "charge"
        }

    def arbitrage_optimization(self,
                              price_profile: np.ndarray,
                              time_hours: np.ndarray,
                              initial_soc: float = 0.5) -> Dict[str, any]:
        """
        Optimize battery operation for energy arbitrage.

        Args:
            price_profile: Electricity prices ($/kWh) for each time step
            time_hours: Time array in hours
            initial_soc: Initial state of charge

        Returns:
            Optimization results including schedule and profit
        """
        n_steps = len(price_profile)
        dt = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 1.0

        # Decision variables: power at each time step (positive=discharge)
        # We'll use linear programming for simplification

        # Variables: [charge_power, discharge_power, soc] for each step
        # Simplified: just optimize charge/discharge decisions

        schedule = []
        current_soc = initial_soc
        total_profit = 0
        total_cycles = 0

        # Simple heuristic optimization: charge at low prices, discharge at high
        price_threshold = np.median(price_profile)

        for i, price in enumerate(price_profile):
            if price < price_threshold and current_soc < 0.9:
                # Charge
                power_kw = -min(self.spec.power_rating_kw,
                              (0.9 - current_soc) * self.spec.capacity_kwh / dt)
            elif price > price_threshold and current_soc > (1 - self.spec.depth_of_discharge):
                # Discharge
                max_discharge = (current_soc - (1 - self.spec.depth_of_discharge)) * \
                               self.spec.capacity_kwh / dt
                power_kw = min(self.spec.power_rating_kw, max_discharge)
            else:
                power_kw = 0

            # Execute operation
            result = self.charge_discharge(current_soc, power_kw, dt)
            current_soc = result["final_soc"]

            # Calculate profit/cost
            if power_kw > 0:  # Discharging (selling)
                arbitrage_profit = result["energy_transferred_kwh"] * price
            else:  # Charging (buying)
                arbitrage_profit = -result["energy_transferred_kwh"] * price

            total_profit += arbitrage_profit
            total_cycles += result["cycles_consumed"]

            # Record schedule
            timestamp = datetime.now() + timedelta(hours=float(time_hours[i]))
            schedule.append(BatteryOperationSchedule(
                timestamp=timestamp,
                power_kw=power_kw,
                state_of_charge=current_soc,
                grid_price=price,
                arbitrage_profit=arbitrage_profit
            ))

        # Calculate cycle life impact
        remaining_life_fraction = 1 - (total_cycles / self.spec.cycle_life)

        return {
            "total_profit_usd": total_profit,
            "total_cycles": total_cycles,
            "remaining_life_fraction": remaining_life_fraction,
            "final_soc": current_soc,
            "schedule": schedule,
            "average_profit_per_cycle": total_profit / total_cycles if total_cycles > 0 else 0,
            "optimization_method": "price_threshold_heuristic"
        }

    def calculate_degradation(self, years_operated: float,
                             annual_cycles: float) -> Dict[str, float]:
        """
        Calculate battery degradation over time.

        Args:
            years_operated: Years of operation
            annual_cycles: Average cycles per year

        Returns:
            Degradation metrics
        """
        # Calendar aging
        calendar_degradation = years_operated * self.spec.degradation_rate_per_year

        # Cycle aging
        total_cycles = years_operated * annual_cycles
        cycle_degradation = total_cycles / self.spec.cycle_life

        # Combined degradation (simplified model)
        total_degradation = min(calendar_degradation + cycle_degradation, 1.0)

        # Remaining capacity
        remaining_capacity_fraction = 1 - total_degradation
        remaining_capacity_kwh = self.spec.capacity_kwh * remaining_capacity_fraction

        # Estimated remaining life
        if annual_cycles > 0:
            cycle_life_remaining = max(0, self.spec.cycle_life - total_cycles) / annual_cycles
        else:
            cycle_life_remaining = float('inf')

        calendar_life_remaining = max(0, (0.8 / self.spec.degradation_rate_per_year) - years_operated)
        estimated_remaining_life = min(cycle_life_remaining, calendar_life_remaining)

        return {
            "calendar_degradation": calendar_degradation,
            "cycle_degradation": cycle_degradation,
            "total_degradation": total_degradation,
            "remaining_capacity_fraction": remaining_capacity_fraction,
            "remaining_capacity_kwh": remaining_capacity_kwh,
            "estimated_remaining_life_years": estimated_remaining_life,
            "end_of_life_reached": remaining_capacity_fraction < 0.8
        }

    def economic_analysis(self, annual_revenue: float,
                         annual_cycles: float,
                         electricity_price: float,
                         project_years: int = 10,
                         discount_rate: float = 0.08) -> Dict[str, float]:
        """
        Perform economic analysis of battery system.

        Args:
            annual_revenue: Annual revenue from battery operation
            annual_cycles: Annual charge/discharge cycles
            electricity_price: Average electricity price
            project_years: Project lifetime
            discount_rate: Discount rate for NPV

        Returns:
            Economic metrics
        """
        # Initial investment
        initial_cost = (self.spec.capacity_kwh * self.spec.initial_cost_per_kwh +
                       self.spec.power_rating_kw * 150)

        # Annual O&M (typically 1-2% of capex)
        annual_om = initial_cost * 0.015

        # Calculate NPV
        npv = -initial_cost
        cash_flows = []

        for year in range(1, project_years + 1):
            # Revenue accounting for degradation
            degradation = self.calculate_degradation(year, annual_cycles)
            revenue = annual_revenue * degradation["remaining_capacity_fraction"]

            # Net cash flow
            net_cash_flow = revenue - annual_om

            # Discount to present value
            pv = present_value(net_cash_flow, discount_rate, year)
            npv += pv
            cash_flows.append(net_cash_flow)

        # Calculate payback period
        cumulative = -initial_cost
        payback_period = None
        for year, cf in enumerate(cash_flows, 1):
            cumulative += cf
            if cumulative > 0 and payback_period is None:
                payback_period = year

        # ROI
        total_revenue = sum(cash_flows)
        roi = (total_revenue - initial_cost) / initial_cost

        return {
            "initial_investment": initial_cost,
            "npv": npv,
            "payback_period_years": payback_period,
            "roi": roi,
            "annual_om": annual_om,
            "total_revenue": total_revenue,
            "is_profitable": npv > 0
        }


def create_battery_system(capacity_kwh: float,
                          technology: EnergyStorageTechnology = EnergyStorageTechnology.LITHIUM_ION
                          ) -> BatteryIntegrator:
    """
    Factory function to create a battery system with default parameters.

    Args:
        capacity_kwh: Battery capacity
        technology: Battery technology type

    Returns:
        Configured BatteryIntegrator instance
    """
    # Technology-specific defaults
    tech_params = {
        EnergyStorageTechnology.LITHIUM_ION: {
            "round_trip_efficiency": 0.92,
            "depth_of_discharge": 0.9,
            "cycle_life": 5000,
            "cost_per_kwh": 300.0,
            "degradation_rate": 0.02
        },
        EnergyStorageTechnology.LEAD_ACID: {
            "round_trip_efficiency": 0.85,
            "depth_of_discharge": 0.5,
            "cycle_life": 2000,
            "cost_per_kwh": 150.0,
            "degradation_rate": 0.03
        },
        EnergyStorageTechnology.FLOW_BATTERY: {
            "round_trip_efficiency": 0.75,
            "depth_of_discharge": 1.0,
            "cycle_life": 10000,
            "cost_per_kwh": 400.0,
            "degradation_rate": 0.01
        }
    }

    params = tech_params.get(technology, tech_params[EnergyStorageTechnology.LITHIUM_ION])

    spec = BatterySpecification(
        technology=technology,
        capacity_kwh=capacity_kwh,
        power_rating_kw=capacity_kwh * 0.5,
        round_trip_efficiency=params["round_trip_efficiency"],
        depth_of_discharge=params["depth_of_discharge"],
        cycle_life=params["cycle_life"],
        initial_cost_per_kwh=params["cost_per_kwh"],
        degradation_rate_per_year=params["degradation_rate"]
    )

    return BatteryIntegrator(spec)


__all__ = ["BatteryIntegrator", "create_battery_system"]
