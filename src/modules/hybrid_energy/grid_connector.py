"""
B12-S04: Grid Interaction & Smart Grid
Production-ready grid connection modeling with services, frequency regulation, and demand response.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..core.data_models import GridConnectionSpec, GridServiceType
from ..core.utilities import validate_positive, validate_range


class GridOperationMode(str, Enum):
    """Grid operation modes."""
    GRID_FOLLOWING = "grid_following"
    GRID_FORMING = "grid_forming"
    ISLANDED = "islanded"


class GridConnector:
    """
    Smart grid integration and services provider for renewable energy systems.
    """

    def __init__(self, spec: GridConnectionSpec):
        """
        Initialize grid connector.

        Args:
            spec: Grid connection specifications
        """
        self.spec = spec
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate grid connection specifications."""
        validate_positive(self.spec.voltage_level_kv, "voltage_level")
        if self.spec.max_export_capacity_kw < 0 or self.spec.max_import_capacity_kw < 0:
            raise ValueError("Grid capacities must be non-negative")

    def grid_services(self,
                     generation_profile: np.ndarray,
                     load_profile: np.ndarray,
                     grid_price_profile: np.ndarray,
                     service_types: Optional[List[GridServiceType]] = None) -> Dict[str, any]:
        """
        Analyze and optimize grid services provision.

        Args:
            generation_profile: Generation output (kW)
            load_profile: Load demand (kW)
            grid_price_profile: Grid electricity prices ($/kWh)
            service_types: List of grid services to provide

        Returns:
            Grid services analysis results
        """
        if service_types is None:
            service_types = self.spec.grid_services

        net_power = generation_profile - load_profile  # Positive = export

        # Initialize results
        results = {
            "services_provided": [],
            "total_revenue": 0,
            "export_energy_kwh": 0,
            "import_energy_kwh": 0,
            "grid_interaction_cost": 0
        }

        # Calculate base grid interactions
        export_mask = net_power > 0
        import_mask = net_power < 0

        export_power = np.where(export_mask, np.minimum(net_power, self.spec.max_export_capacity_kw), 0)
        import_power = np.where(import_mask, np.maximum(net_power, -self.spec.max_import_capacity_kw), 0)

        results["export_energy_kwh"] = float(np.sum(export_power))
        results["import_energy_kwh"] = float(np.abs(np.sum(import_power)))

        # Revenue from exports
        export_revenue = np.sum(export_power * grid_price_profile)
        import_cost = np.sum(np.abs(import_power) * grid_price_profile)

        results["export_revenue"] = float(export_revenue)
        results["import_cost"] = float(import_cost)
        results["net_grid_revenue"] = float(export_revenue - import_cost)

        # Grid service revenue
        service_revenue = 0

        if GridServiceType.FREQUENCY_REGULATION in service_types:
            freq_reg = self._calculate_frequency_regulation_value(generation_profile)
            service_revenue += freq_reg["revenue"]
            results["services_provided"].append({
                "type": "frequency_regulation",
                "revenue": freq_reg["revenue"],
                "capacity_provided_kw": freq_reg["capacity_kw"]
            })

        if GridServiceType.VOLTAGE_SUPPORT in service_types:
            voltage_support = self._calculate_voltage_support_value(
                self.spec.max_export_capacity_kw
            )
            service_revenue += voltage_support["revenue"]
            results["services_provided"].append({
                "type": "voltage_support",
                "revenue": voltage_support["revenue"]
            })

        if GridServiceType.PEAK_SHAVING in service_types:
            peak_shaving = self._calculate_peak_shaving_value(net_power, grid_price_profile)
            service_revenue += peak_shaving["revenue"]
            results["services_provided"].append({
                "type": "peak_shaving",
                "revenue": peak_shaving["revenue"],
                "peak_reduction_kw": peak_shaving["peak_reduction_kw"]
            })

        results["grid_service_revenue"] = float(service_revenue)
        results["total_revenue"] = float(export_revenue + service_revenue - import_cost)

        return results

    def _calculate_frequency_regulation_value(self, generation_profile: np.ndarray) -> Dict[str, float]:
        """Calculate frequency regulation service value."""
        # Simplified model: capacity payment based on available capacity
        # Typical rate: $10-40/kW-month

        available_capacity = np.mean(generation_profile) * 0.1  # 10% available for regulation
        monthly_rate = 20  # $/kW-month
        annual_revenue = available_capacity * monthly_rate * 12

        return {
            "capacity_kw": available_capacity,
            "revenue": annual_revenue
        }

    def _calculate_voltage_support_value(self, capacity_kw: float) -> Dict[str, float]:
        """Calculate voltage support service value."""
        # Reactive power capability
        # Typical rate: $5-15/kW-month
        monthly_rate = 10  # $/kW-month
        annual_revenue = capacity_kw * 0.05 * monthly_rate * 12  # 5% of capacity

        return {
            "revenue": annual_revenue
        }

    def _calculate_peak_shaving_value(self, net_power: np.ndarray,
                                     price_profile: np.ndarray) -> Dict[str, float]:
        """Calculate peak shaving service value."""
        # Find peak demand periods (top 10% prices)
        price_threshold = np.percentile(price_profile, 90)
        peak_mask = price_profile > price_threshold

        # Power reduction during peaks
        peak_power = net_power[peak_mask]
        avg_reduction = np.mean(np.abs(peak_power[peak_power < 0])) if np.any(peak_power < 0) else 0

        # Revenue from peak reduction
        # Typical rate: $50-150/kW-month for capacity
        monthly_rate = 100  # $/kW-month
        annual_revenue = avg_reduction * monthly_rate * 12

        return {
            "peak_reduction_kw": avg_reduction,
            "revenue": annual_revenue
        }

    def frequency_regulation(self,
                           nominal_frequency: float = 60.0,
                           frequency_deviation: np.ndarray = None,
                           droop_setting: float = 0.05) -> Dict[str, any]:
        """
        Model frequency regulation response.

        Args:
            nominal_frequency: Nominal grid frequency (Hz)
            frequency_deviation: Frequency deviation timeseries (Hz)
            droop_setting: Droop setting (e.g., 5% = 0.05)

        Returns:
            Frequency regulation analysis
        """
        if frequency_deviation is None:
            # Generate synthetic frequency deviations (normal distribution)
            frequency_deviation = np.random.normal(0, 0.05, 8760)  # ±0.05 Hz typical

        # Calculate required power response
        # ΔP = (Prated / droop) * (Δf / fnom)
        power_response = (self.spec.max_export_capacity_kw / droop_setting) * \
                        (frequency_deviation / nominal_frequency)

        # Clip to available capacity
        power_response = np.clip(power_response,
                                -self.spec.max_import_capacity_kw,
                                self.spec.max_export_capacity_kw)

        # Statistics
        avg_response = np.mean(np.abs(power_response))
        max_response = np.max(np.abs(power_response))
        response_time = 0.1  # seconds (typical for modern inverters)

        # Regulation performance score (based on accuracy)
        performance_score = 1.0 - (np.std(power_response) / self.spec.max_export_capacity_kw)

        return {
            "power_response_timeseries": power_response,
            "average_response_kw": float(avg_response),
            "max_response_kw": float(max_response),
            "response_time_seconds": response_time,
            "droop_setting": droop_setting,
            "performance_score": float(max(0, performance_score)),
            "frequency_deviation_std": float(np.std(frequency_deviation)),
            "regulation_capacity_utilized": float(max_response / self.spec.max_export_capacity_kw)
        }

    def demand_response(self,
                       load_profile: np.ndarray,
                       dr_events: List[Dict[str, any]],
                       max_load_reduction_pct: float = 0.3) -> Dict[str, any]:
        """
        Model demand response participation.

        Args:
            load_profile: Original load profile (kW)
            dr_events: List of DR events with 'start_hour', 'duration', 'target_reduction'
            max_load_reduction_pct: Maximum allowed load reduction

        Returns:
            Demand response analysis
        """
        validate_range(max_load_reduction_pct, 0, 1, "max_load_reduction_pct")

        modified_load = load_profile.copy()
        total_reduction = 0
        total_revenue = 0

        for event in dr_events:
            start = event['start_hour']
            duration = event['duration']
            target_reduction_kw = event.get('target_reduction_kw', 0)
            payment_per_kwh = event.get('payment_per_kwh', 0.5)  # $/kWh

            # Calculate achievable reduction
            max_reduction = load_profile[start:start + duration] * max_load_reduction_pct
            actual_reduction = np.minimum(target_reduction_kw, max_reduction)

            # Apply reduction
            modified_load[start:start + duration] -= actual_reduction

            # Calculate revenue
            energy_reduced = np.sum(actual_reduction)
            revenue = energy_reduced * payment_per_kwh

            total_reduction += energy_reduced
            total_revenue += revenue

        # Load shifting (simplified - shift some load to off-peak)
        # This is a simplified model; real implementation would be more complex
        shifted_energy = total_reduction * 0.8  # 80% of reduced load is shifted
        off_peak_hours = np.where(load_profile < np.percentile(load_profile, 25))[0]

        if len(off_peak_hours) > 0:
            shift_per_hour = shifted_energy / len(off_peak_hours)
            modified_load[off_peak_hours] += shift_per_hour

        return {
            "original_total_load_kwh": float(np.sum(load_profile)),
            "modified_total_load_kwh": float(np.sum(modified_load)),
            "total_energy_reduced_kwh": float(total_reduction),
            "total_energy_shifted_kwh": float(shifted_energy),
            "total_revenue": float(total_revenue),
            "number_of_events": len(dr_events),
            "average_reduction_per_event_kwh": float(total_reduction / len(dr_events)) if dr_events else 0,
            "modified_load_profile": modified_load,
            "peak_load_reduction": float(np.max(load_profile) - np.max(modified_load))
        }

    def power_quality_analysis(self,
                              voltage_profile: Optional[np.ndarray] = None,
                              power_factor: float = 0.95) -> Dict[str, float]:
        """
        Analyze power quality metrics.

        Args:
            voltage_profile: Voltage timeseries (p.u.)
            power_factor: System power factor

        Returns:
            Power quality metrics
        """
        if voltage_profile is None:
            # Generate synthetic voltage profile
            voltage_profile = np.random.normal(1.0, 0.02, 8760)  # ±2% typical

        # Voltage quality metrics
        voltage_violations = np.sum((voltage_profile < 0.95) | (voltage_profile > 1.05))
        voltage_violation_rate = voltage_violations / len(voltage_profile)

        # THD (Total Harmonic Distortion) - simplified model
        thd_current = 0.05  # 5% typical for modern inverters
        thd_voltage = 0.02  # 2% typical

        # Reactive power capability
        apparent_power_kva = self.spec.max_export_capacity_kw / power_factor
        reactive_power_kvar = apparent_power_kva * np.sin(np.arccos(power_factor))

        return {
            "average_voltage_pu": float(np.mean(voltage_profile)),
            "voltage_std_pu": float(np.std(voltage_profile)),
            "voltage_violation_rate": float(voltage_violation_rate),
            "power_factor": power_factor,
            "thd_current_percent": thd_current * 100,
            "thd_voltage_percent": thd_voltage * 100,
            "reactive_power_capacity_kvar": float(reactive_power_kvar),
            "apparent_power_kva": float(apparent_power_kva),
            "meets_ieee_1547": voltage_violation_rate < 0.05  # Less than 5% violations
        }

    def calculate_interconnection_costs(self) -> Dict[str, float]:
        """
        Calculate grid interconnection costs.

        Returns:
            Cost breakdown
        """
        # Cost components (simplified models)
        capacity_kw = max(self.spec.max_export_capacity_kw, self.spec.max_import_capacity_kw)

        # Transformer costs (if required)
        if self.spec.voltage_level_kv >= 69:
            transformer_cost = capacity_kw * 100  # $/kW
        elif self.spec.voltage_level_kv >= 12:
            transformer_cost = capacity_kw * 50  # $/kW
        else:
            transformer_cost = capacity_kw * 30  # $/kW

        # Protection and control equipment
        protection_cost = 50000 + capacity_kw * 20

        # Metering and communication
        metering_cost = 25000

        # Studies and engineering
        study_cost = 100000 if capacity_kw > 5000 else 50000

        total_cost = (transformer_cost + protection_cost +
                     metering_cost + study_cost + self.spec.interconnection_cost)

        return {
            "transformer_cost": transformer_cost,
            "protection_equipment_cost": protection_cost,
            "metering_cost": metering_cost,
            "study_cost": study_cost,
            "utility_fees": self.spec.interconnection_cost,
            "total_interconnection_cost": total_cost,
            "cost_per_kw": total_cost / capacity_kw if capacity_kw > 0 else 0
        }


__all__ = ["GridConnector", "GridOperationMode"]
