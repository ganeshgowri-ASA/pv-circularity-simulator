"""
Performance metrics and monitoring utilities.

This module provides classes and functions for tracking and analyzing
the performance of hybrid energy systems.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class SystemMetrics:
    """
    Real-time system metrics snapshot.

    Attributes:
        timestamp: Time of measurement
        total_generation_kw: Total power generation
        total_consumption_kw: Total power consumption
        net_power_kw: Net power (generation - consumption)
        grid_power_kw: Grid import/export
        battery_soc: Battery state of charge
        component_status: Status of each component
    """

    timestamp: datetime
    total_generation_kw: float = 0.0
    total_consumption_kw: float = 0.0
    net_power_kw: float = 0.0
    grid_power_kw: float = 0.0
    battery_soc: float = 0.0
    component_status: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_generation_kw": self.total_generation_kw,
            "total_consumption_kw": self.total_consumption_kw,
            "net_power_kw": self.net_power_kw,
            "grid_power_kw": self.grid_power_kw,
            "battery_soc": self.battery_soc,
            "component_status": self.component_status,
        }


@dataclass
class PerformanceMetrics:
    """
    Aggregate performance metrics over a time period.

    This class calculates and stores performance indicators for
    hybrid energy system operation.

    Attributes:
        start_time: Period start time
        end_time: Period end time
        total_energy_generated_kwh: Total energy generated
        total_energy_consumed_kwh: Total energy consumed
        total_grid_import_kwh: Energy imported from grid
        total_grid_export_kwh: Energy exported to grid
        renewable_fraction: Fraction of load met by renewables
        self_consumption_ratio: Fraction of renewable energy self-consumed
        self_sufficiency_ratio: Fraction of load met by local generation
        battery_cycles: Number of battery charge/discharge cycles
        capacity_factor: System capacity factor
    """

    start_time: datetime
    end_time: datetime
    total_energy_generated_kwh: float = 0.0
    total_energy_consumed_kwh: float = 0.0
    total_grid_import_kwh: float = 0.0
    total_grid_export_kwh: float = 0.0
    renewable_fraction: float = 0.0
    self_consumption_ratio: float = 0.0
    self_sufficiency_ratio: float = 0.0
    battery_cycles: float = 0.0
    capacity_factor: float = 0.0

    def calculate_metrics(
        self,
        metrics_history: List[Dict],
        system_capacity_kw: float
    ) -> None:
        """
        Calculate performance metrics from historical data.

        Args:
            metrics_history: List of metric dictionaries
            system_capacity_kw: Total system capacity
        """
        if not metrics_history:
            return

        # Calculate energy totals
        for metric in metrics_history:
            gen = metric.get("pv_generation_kw", 0.0)
            load = metric.get("load_demand_kw", 0.0)
            grid = metric.get("grid_power_kw", 0.0)

            # Assume 5-minute intervals
            time_step_hours = 5.0 / 60.0

            self.total_energy_generated_kwh += gen * time_step_hours
            self.total_energy_consumed_kwh += load * time_step_hours

            if grid > 0:
                self.total_grid_import_kwh += grid * time_step_hours
            else:
                self.total_grid_export_kwh += abs(grid) * time_step_hours

        # Calculate ratios
        if self.total_energy_consumed_kwh > 0:
            self.self_sufficiency_ratio = min(
                1.0,
                (self.total_energy_consumed_kwh - self.total_grid_import_kwh)
                / self.total_energy_consumed_kwh,
            )

        if self.total_energy_generated_kwh > 0:
            self.self_consumption_ratio = min(
                1.0,
                (self.total_energy_generated_kwh - self.total_grid_export_kwh)
                / self.total_energy_generated_kwh,
            )

        # Calculate capacity factor
        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600.0
        if duration_hours > 0 and system_capacity_kw > 0:
            self.capacity_factor = self.total_energy_generated_kwh / (
                system_capacity_kw * duration_hours
            )

        # Renewable fraction (assuming all generation is renewable)
        if self.total_energy_consumed_kwh > 0:
            renewable_energy = min(
                self.total_energy_generated_kwh, self.total_energy_consumed_kwh
            )
            self.renewable_fraction = renewable_energy / self.total_energy_consumed_kwh

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_energy_generated_kwh": round(self.total_energy_generated_kwh, 2),
            "total_energy_consumed_kwh": round(self.total_energy_consumed_kwh, 2),
            "total_grid_import_kwh": round(self.total_grid_import_kwh, 2),
            "total_grid_export_kwh": round(self.total_grid_export_kwh, 2),
            "renewable_fraction": round(self.renewable_fraction, 3),
            "self_consumption_ratio": round(self.self_consumption_ratio, 3),
            "self_sufficiency_ratio": round(self.self_sufficiency_ratio, 3),
            "battery_cycles": round(self.battery_cycles, 2),
            "capacity_factor": round(self.capacity_factor, 3),
        }

    def get_summary_text(self) -> str:
        """
        Get a formatted text summary of performance metrics.

        Returns:
            Formatted string with key metrics
        """
        return f"""
Performance Summary
==================
Period: {self.start_time.strftime('%Y-%m-%d %H:%M')} to {self.end_time.strftime('%Y-%m-%d %H:%M')}

Energy:
  Generated: {self.total_energy_generated_kwh:.2f} kWh
  Consumed: {self.total_energy_consumed_kwh:.2f} kWh
  Grid Import: {self.total_grid_import_kwh:.2f} kWh
  Grid Export: {self.total_grid_export_kwh:.2f} kWh

Performance Indicators:
  Renewable Fraction: {self.renewable_fraction*100:.1f}%
  Self-Consumption: {self.self_consumption_ratio*100:.1f}%
  Self-Sufficiency: {self.self_sufficiency_ratio*100:.1f}%
  Capacity Factor: {self.capacity_factor*100:.1f}%
        """.strip()


class MetricsTracker:
    """
    Metrics tracking and analysis utility.

    This class provides methods for collecting, storing, and analyzing
    system metrics over time.
    """

    def __init__(self, max_history_points: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            max_history_points: Maximum number of historical points to retain
        """
        self.max_history_points = max_history_points
        self.metrics_history: List[SystemMetrics] = []
        self.simulation_results: List[Dict] = []

    def add_metric(self, metric: SystemMetrics) -> None:
        """
        Add a metric snapshot to history.

        Args:
            metric: SystemMetrics instance
        """
        self.metrics_history.append(metric)

        # Limit history size
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history.pop(0)

    def add_simulation_result(self, result: Dict) -> None:
        """
        Add simulation result to history.

        Args:
            result: Dictionary of simulation results
        """
        self.simulation_results.append(result)

        # Limit history size
        if len(self.simulation_results) > self.max_history_points:
            self.simulation_results.pop(0)

    def get_recent_metrics(self, count: int = 100) -> List[SystemMetrics]:
        """
        Get most recent metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent SystemMetrics
        """
        return self.metrics_history[-count:]

    def calculate_performance(
        self,
        start_time: datetime,
        end_time: datetime,
        system_capacity_kw: float,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a time period.

        Args:
            start_time: Period start
            end_time: Period end
            system_capacity_kw: System capacity

        Returns:
            PerformanceMetrics instance
        """
        perf = PerformanceMetrics(start_time=start_time, end_time=end_time)
        perf.calculate_metrics(self.simulation_results, system_capacity_kw)
        return perf

    def get_time_series_data(
        self, metric_name: str, count: Optional[int] = None
    ) -> List[float]:
        """
        Extract time series data for a specific metric.

        Args:
            metric_name: Name of the metric
            count: Number of recent points (None for all)

        Returns:
            List of metric values
        """
        results = self.simulation_results if count is None else self.simulation_results[-count:]
        return [r.get(metric_name, 0.0) for r in results]

    def clear_history(self) -> None:
        """Clear all historical data."""
        self.metrics_history = []
        self.simulation_results = []
