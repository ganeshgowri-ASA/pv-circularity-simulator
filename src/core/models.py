"""
Energy system models for hybrid energy simulation.

This module defines the core models for energy components and hybrid
energy systems, including their behavior and interactions.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EnergyComponent:
    """
    Base class for energy system components.

    This class represents a generic energy component in a hybrid system
    with common attributes and methods.

    Attributes:
        component_id: Unique identifier
        component_type: Type of component
        name: Human-readable name
        capacity: Nominal capacity
        capacity_unit: Unit of capacity
        efficiency: Operating efficiency (0.0-1.0)
        state: Current operating state
        power_output: Current power output
        cumulative_energy: Total energy produced/consumed
        parameters: Additional component-specific parameters
    """

    component_id: str
    component_type: str
    name: str
    capacity: float
    capacity_unit: str = "kW"
    efficiency: float = 0.95
    state: str = "idle"
    power_output: float = 0.0
    cumulative_energy: float = 0.0
    parameters: Dict = field(default_factory=dict)

    def calculate_output(
        self, input_conditions: Dict, time_step_minutes: int = 5
    ) -> float:
        """
        Calculate component output based on input conditions.

        Args:
            input_conditions: Dictionary of input conditions (weather, demand, etc.)
            time_step_minutes: Time step for calculation

        Returns:
            Power output in kW
        """
        # Base implementation - override in subclasses
        return 0.0

    def update_state(self, power: float, time_step_minutes: int = 5) -> None:
        """
        Update component state based on power output.

        Args:
            power: Power output in kW
            time_step_minutes: Time step for state update
        """
        self.power_output = power
        energy_kwh = power * (time_step_minutes / 60.0)
        self.cumulative_energy += energy_kwh

        if power > 0:
            self.state = "operating"
        else:
            self.state = "idle"

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics for the component.

        Returns:
            Dictionary of component metrics
        """
        return {
            "power_output_kw": self.power_output,
            "cumulative_energy_kwh": self.cumulative_energy,
            "capacity_factor": (
                self.power_output / self.capacity if self.capacity > 0 else 0.0
            ),
            "efficiency": self.efficiency,
        }


@dataclass
class PVArray(EnergyComponent):
    """
    Photovoltaic array component.

    Attributes:
        area_m2: Array area in square meters
        tilt_angle: Tilt angle in degrees
        azimuth_angle: Azimuth angle in degrees
    """

    area_m2: float = 50.0
    tilt_angle: float = 30.0
    azimuth_angle: float = 180.0

    def calculate_output(
        self, input_conditions: Dict, time_step_minutes: int = 5
    ) -> float:
        """
        Calculate PV output based on irradiance and temperature.

        Args:
            input_conditions: Must include 'irradiance_w_m2' and optionally 'temperature_c'
            time_step_minutes: Time step for calculation

        Returns:
            Power output in kW
        """
        irradiance = input_conditions.get("irradiance_w_m2", 0.0)
        temperature = input_conditions.get("temperature_c", 25.0)

        # Simple temperature coefficient model
        temp_coefficient = 0.004  # per degree C
        temp_factor = 1.0 - temp_coefficient * (temperature - 25.0)

        # Calculate power output
        power_kw = (irradiance * self.area_m2 * self.efficiency * temp_factor) / 1000.0
        power_kw = min(power_kw, self.capacity)

        self.update_state(power_kw, time_step_minutes)
        return power_kw


@dataclass
class BatteryStorage(EnergyComponent):
    """
    Battery energy storage system.

    Attributes:
        state_of_charge: Current state of charge (0.0-1.0)
        min_soc: Minimum allowable SOC
        max_soc: Maximum allowable SOC
        charge_rate_max_kw: Maximum charging rate
        discharge_rate_max_kw: Maximum discharging rate
    """

    state_of_charge: float = 0.5
    min_soc: float = 0.2
    max_soc: float = 0.9
    charge_rate_max_kw: float = 5.0
    discharge_rate_max_kw: float = 5.0

    def charge(self, power_kw: float, time_step_minutes: int = 5) -> float:
        """
        Charge the battery.

        Args:
            power_kw: Charging power in kW
            time_step_minutes: Time step for charging

        Returns:
            Actual power accepted (may be less than requested)
        """
        # Limit by charge rate
        actual_power = min(power_kw, self.charge_rate_max_kw)

        # Calculate energy in kWh
        energy_kwh = actual_power * (time_step_minutes / 60.0) * self.efficiency

        # Calculate new SOC
        new_soc = self.state_of_charge + (energy_kwh / self.capacity)

        # Limit by max SOC
        if new_soc > self.max_soc:
            energy_kwh = (self.max_soc - self.state_of_charge) * self.capacity
            actual_power = energy_kwh / (time_step_minutes / 60.0) / self.efficiency
            new_soc = self.max_soc

        self.state_of_charge = new_soc
        self.update_state(actual_power, time_step_minutes)
        return actual_power

    def discharge(self, power_kw: float, time_step_minutes: int = 5) -> float:
        """
        Discharge the battery.

        Args:
            power_kw: Discharge power in kW
            time_step_minutes: Time step for discharging

        Returns:
            Actual power delivered (may be less than requested)
        """
        # Limit by discharge rate
        actual_power = min(power_kw, self.discharge_rate_max_kw)

        # Calculate energy in kWh
        energy_kwh = actual_power * (time_step_minutes / 60.0) / self.efficiency

        # Calculate new SOC
        new_soc = self.state_of_charge - (energy_kwh / self.capacity)

        # Limit by min SOC
        if new_soc < self.min_soc:
            energy_kwh = (self.state_of_charge - self.min_soc) * self.capacity
            actual_power = (
                energy_kwh * self.efficiency / (time_step_minutes / 60.0)
            )
            new_soc = self.min_soc

        self.state_of_charge = new_soc
        self.update_state(-actual_power, time_step_minutes)
        return actual_power


class HybridEnergySystem:
    """
    Hybrid energy system composed of multiple energy components.

    This class manages the interactions between different energy components
    and implements control strategies for optimal operation.

    Attributes:
        system_name: Name of the hybrid system
        components: Dictionary of energy components
        operation_strategy: Current operation strategy
        current_time: Current simulation time
        system_state: Current system state
        metrics_history: Historical metrics data
    """

    def __init__(self, system_name: str = "Hybrid System"):
        """
        Initialize hybrid energy system.

        Args:
            system_name: Name for the system
        """
        self.system_name = system_name
        self.components: Dict[str, EnergyComponent] = {}
        self.operation_strategy: Optional[str] = None
        self.current_time: Optional[datetime] = None
        self.system_state: str = "initialized"
        self.metrics_history: List[Dict] = []

    def add_component(self, component: EnergyComponent) -> None:
        """
        Add a component to the hybrid system.

        Args:
            component: EnergyComponent instance to add
        """
        self.components[component.component_id] = component

    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from the system.

        Args:
            component_id: ID of component to remove

        Returns:
            True if removed, False if not found
        """
        if component_id in self.components:
            del self.components[component_id]
            return True
        return False

    def get_component(self, component_id: str) -> Optional[EnergyComponent]:
        """
        Retrieve a component by ID.

        Args:
            component_id: Component identifier

        Returns:
            EnergyComponent if found, None otherwise
        """
        return self.components.get(component_id)

    def get_total_capacity(self, component_type: Optional[str] = None) -> float:
        """
        Calculate total system capacity.

        Args:
            component_type: Filter by component type (optional)

        Returns:
            Total capacity in kW
        """
        total = 0.0
        for component in self.components.values():
            if component_type is None or component.component_type == component_type:
                total += component.capacity
        return total

    def get_current_power(self) -> Tuple[float, float]:
        """
        Get current generation and consumption.

        Returns:
            Tuple of (generation_kw, consumption_kw)
        """
        generation = 0.0
        consumption = 0.0

        for component in self.components.values():
            if component.power_output > 0:
                generation += component.power_output
            else:
                consumption += abs(component.power_output)

        return generation, consumption

    def simulate_step(
        self,
        load_demand_kw: float,
        input_conditions: Dict,
        time_step_minutes: int = 5,
    ) -> Dict[str, float]:
        """
        Simulate one time step of system operation.

        Args:
            load_demand_kw: Load demand in kW
            input_conditions: Environmental conditions
            time_step_minutes: Time step duration

        Returns:
            Dictionary of simulation results
        """
        results = {
            "load_demand_kw": load_demand_kw,
            "pv_generation_kw": 0.0,
            "battery_power_kw": 0.0,
            "grid_power_kw": 0.0,
            "unmet_load_kw": 0.0,
            "excess_power_kw": 0.0,
        }

        # Calculate PV generation
        pv_power = 0.0
        for component in self.components.values():
            if component.component_type == "pv_array":
                pv_power += component.calculate_output(
                    input_conditions, time_step_minutes
                )

        results["pv_generation_kw"] = pv_power

        # Calculate net load
        net_load = load_demand_kw - pv_power

        # Handle battery charging/discharging
        battery_power = 0.0
        if net_load < 0:  # Excess power - charge battery
            for component in self.components.values():
                if component.component_type == "battery":
                    battery = component
                    battery_power = -battery.charge(
                        abs(net_load), time_step_minutes
                    )
                    break
        elif net_load > 0:  # Deficit - discharge battery
            for component in self.components.values():
                if component.component_type == "battery":
                    battery = component
                    battery_power = battery.discharge(net_load, time_step_minutes)
                    break

        results["battery_power_kw"] = battery_power
        net_load -= battery_power

        # Handle remaining load/excess
        if net_load > 0:
            results["grid_power_kw"] = net_load  # Import from grid
        elif net_load < 0:
            results["excess_power_kw"] = abs(net_load)  # Export or curtail

        return results

    def get_system_metrics(self) -> Dict[str, any]:
        """
        Get current system-wide metrics.

        Returns:
            Dictionary of system metrics
        """
        generation, consumption = self.get_current_power()

        metrics = {
            "system_name": self.system_name,
            "total_generation_kw": generation,
            "total_consumption_kw": consumption,
            "net_power_kw": generation - consumption,
            "total_capacity_kw": self.get_total_capacity(),
            "num_components": len(self.components),
            "system_state": self.system_state,
        }

        # Add component-specific metrics
        for comp_id, component in self.components.items():
            comp_metrics = component.get_metrics()
            for key, value in comp_metrics.items():
                metrics[f"{comp_id}_{key}"] = value

        return metrics

    def reset(self) -> None:
        """Reset the system to initial state."""
        for component in self.components.values():
            component.power_output = 0.0
            component.cumulative_energy = 0.0
            component.state = "idle"

        self.metrics_history = []
        self.system_state = "initialized"
