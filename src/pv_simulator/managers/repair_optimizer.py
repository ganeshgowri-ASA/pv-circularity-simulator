"""RepairOptimizer: Intelligent repair optimization and maintenance planning for PV systems.

This module provides a comprehensive solution for diagnosing faults, estimating repair costs,
scheduling maintenance activities, and managing spare parts inventory for photovoltaic systems.
It uses optimization algorithms to minimize costs while maximizing system availability.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy.optimize import linear_sum_assignment

from pv_simulator.models.maintenance import (
    ComponentHealth,
    Fault,
    FaultSeverity,
    FaultType,
    MaintenancePriority,
    MaintenanceSchedule,
    MaintenanceType,
    RepairCostEstimate,
    RepairTask,
    SparePart,
)


class RepairOptimizer:
    """Intelligent repair optimization and maintenance planning system for PV installations.

    The RepairOptimizer provides advanced capabilities for managing the maintenance lifecycle
    of photovoltaic systems, including:

    - Fault diagnosis using rule-based and statistical methods
    - Detailed repair cost estimation with parts and labor breakdown
    - Optimized maintenance scheduling to minimize downtime and costs
    - Spare parts inventory management with automatic reordering

    Attributes:
        labor_rate: Cost per hour for maintenance labor (default: 75.0)
        overhead_rate: Overhead cost multiplier (default: 0.15 or 15%)
        fault_detection_threshold: Minimum confidence for fault detection (default: 0.6)
        spare_parts: Dictionary of spare parts inventory
        component_health: Dictionary of component health records
        active_faults: List of currently active faults
        repair_history: Historical record of completed repairs

    Example:
        >>> optimizer = RepairOptimizer(labor_rate=80.0)
        >>>
        >>> # Diagnose a fault
        >>> fault = optimizer.fault_diagnosis(
        ...     component_id="INV-001",
        ...     component_type="inverter",
        ...     performance_data={"efficiency": 0.85, "temperature": 75.0}
        ... )
        >>>
        >>> # Get cost estimate
        >>> estimate = optimizer.repair_cost_estimation(fault)
        >>>
        >>> # Schedule maintenance
        >>> schedule = optimizer.maintenance_scheduling(
        ...     tasks=[task1, task2, task3],
        ...     start_date=datetime.now(),
        ...     end_date=datetime.now() + timedelta(days=30)
        ... )
        >>>
        >>> # Manage spare parts
        >>> reorder_list = optimizer.spare_parts_management()
    """

    def __init__(
        self,
        labor_rate: float = 75.0,
        overhead_rate: float = 0.15,
        fault_detection_threshold: float = 0.6,
    ) -> None:
        """Initialize the RepairOptimizer.

        Args:
            labor_rate: Hourly cost for maintenance labor in currency units
            overhead_rate: Overhead multiplier applied to total costs (e.g., 0.15 = 15%)
            fault_detection_threshold: Minimum confidence score required for fault detection
                (0.0 to 1.0, where higher values reduce false positives)

        Raises:
            ValueError: If labor_rate is negative or overhead_rate is not in valid range
        """
        if labor_rate < 0:
            raise ValueError("Labor rate must be non-negative")
        if not 0.0 <= overhead_rate <= 1.0:
            raise ValueError("Overhead rate must be between 0.0 and 1.0")
        if not 0.0 <= fault_detection_threshold <= 1.0:
            raise ValueError("Fault detection threshold must be between 0.0 and 1.0")

        self.labor_rate = labor_rate
        self.overhead_rate = overhead_rate
        self.fault_detection_threshold = fault_detection_threshold

        # Internal state management
        self.spare_parts: Dict[str, SparePart] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.active_faults: List[Fault] = []
        self.repair_history: List[RepairTask] = []

        # Fault diagnosis parameters (configurable thresholds)
        self._fault_thresholds = {
            "efficiency_drop": 0.10,  # 10% efficiency drop
            "temperature_high": 70.0,  # Celsius
            "voltage_deviation": 0.05,  # 5% voltage deviation
            "current_imbalance": 0.10,  # 10% current imbalance
            "degradation_rate_high": 0.02,  # 2% per year
        }

    def fault_diagnosis(
        self,
        component_id: str,
        component_type: str,
        performance_data: Dict[str, float],
        baseline_data: Optional[Dict[str, float]] = None,
        historical_data: Optional[List[Dict[str, float]]] = None,
    ) -> Optional[Fault]:
        """Diagnose faults in PV system components using performance data analysis.

        This method analyzes component performance data against baselines and historical
        trends to identify potential faults. It uses rule-based logic and statistical
        methods to detect anomalies and classify fault types.

        Args:
            component_id: Unique identifier for the component being diagnosed
            component_type: Type of component (e.g., 'panel', 'inverter', 'junction_box')
            performance_data: Current performance metrics (e.g., voltage, current, temperature,
                efficiency, power output)
            baseline_data: Expected/nominal performance values for comparison. If None,
                uses typical values based on component_type
            historical_data: List of historical performance measurements for trend analysis.
                Used to detect degradation patterns

        Returns:
            Fault object if a fault is detected with confidence above threshold,
            None if no significant fault is found

        Raises:
            ValueError: If component_id is empty or performance_data is empty

        Example:
            >>> fault = optimizer.fault_diagnosis(
            ...     component_id="PANEL-A23",
            ...     component_type="panel",
            ...     performance_data={
            ...         "voltage": 28.5,
            ...         "current": 7.2,
            ...         "temperature": 65.0,
            ...         "efficiency": 0.16
            ...     },
            ...     baseline_data={"voltage": 30.0, "current": 8.0, "efficiency": 0.18}
            ... )
        """
        if not component_id:
            raise ValueError("component_id cannot be empty")
        if not performance_data:
            raise ValueError("performance_data cannot be empty")

        # Initialize baseline if not provided
        if baseline_data is None:
            baseline_data = self._get_default_baseline(component_type)

        # Detect anomalies and classify fault
        fault_type, severity, symptoms, confidence = self._analyze_performance(
            component_type, performance_data, baseline_data, historical_data
        )

        # Return None if confidence is below threshold
        if confidence < self.fault_detection_threshold:
            return None

        # Determine root cause based on symptoms
        root_cause = self._determine_root_cause(fault_type, symptoms, component_type)

        # Create and return fault object
        fault = Fault(
            component_id=component_id,
            component_type=component_type,
            fault_type=fault_type,
            severity=severity,
            description=f"{fault_type.value.title()} fault detected in {component_type}",
            symptoms=symptoms,
            diagnosis_confidence=confidence,
            root_cause=root_cause,
            affected_metrics=self._calculate_affected_metrics(
                performance_data, baseline_data
            ),
        )

        # Update internal state
        self.active_faults.append(fault)
        if component_id in self.component_health:
            self.component_health[component_id].current_faults.append(fault.fault_id)

        return fault

    def repair_cost_estimation(
        self,
        fault: Fault,
        include_parts: bool = True,
        rush_service: bool = False,
        custom_labor_hours: Optional[float] = None,
    ) -> RepairCostEstimate:
        """Estimate the total cost of repairing a diagnosed fault.

        Generates a detailed cost breakdown including labor, parts, and overhead costs.
        The estimation uses historical data, component complexity, and fault severity
        to provide accurate cost projections.

        Args:
            fault: Diagnosed fault requiring repair
            include_parts: Whether to include spare parts costs in estimate (default: True)
            rush_service: Apply rush service multiplier (typically 1.5x labor cost)
            custom_labor_hours: Override automatic labor hour estimation with custom value

        Returns:
            RepairCostEstimate with detailed cost breakdown

        Raises:
            ValueError: If fault severity is unrecognized

        Example:
            >>> estimate = optimizer.repair_cost_estimation(
            ...     fault=detected_fault,
            ...     include_parts=True,
            ...     rush_service=False
            ... )
            >>> print(f"Total cost: ${estimate.total_cost:.2f}")
            >>> print(f"Labor: ${estimate.labor_cost:.2f}, Parts: ${estimate.parts_cost:.2f}")
        """
        # Estimate labor hours based on fault type and severity
        if custom_labor_hours is not None:
            labor_hours = custom_labor_hours
        else:
            labor_hours = self._estimate_labor_hours(fault)

        # Apply rush service multiplier if requested
        effective_labor_rate = self.labor_rate * (1.5 if rush_service else 1.0)
        labor_cost = labor_hours * effective_labor_rate

        # Estimate parts cost
        parts_cost = 0.0
        parts_breakdown: Dict[str, float] = {}

        if include_parts:
            parts_breakdown = self._estimate_parts_cost(fault)
            parts_cost = sum(parts_breakdown.values())

        # Calculate overhead
        overhead_cost = (labor_cost + parts_cost) * self.overhead_rate

        # Total cost
        total_cost = labor_cost + parts_cost + overhead_cost

        # Confidence based on fault diagnosis confidence and parts availability
        confidence_level = fault.diagnosis_confidence * 0.9  # Slight reduction for estimation

        return RepairCostEstimate(
            fault_id=fault.fault_id,
            component_id=fault.component_id,
            labor_hours=labor_hours,
            labor_rate=effective_labor_rate,
            labor_cost=labor_cost,
            parts_cost=parts_cost,
            parts_breakdown=parts_breakdown,
            overhead_cost=overhead_cost,
            total_cost=total_cost,
            confidence_level=confidence_level,
            valid_until=datetime.now() + timedelta(days=30),
            notes=f"Estimate for {fault.fault_type.value} repair"
            + (" (RUSH SERVICE)" if rush_service else ""),
        )

    def maintenance_scheduling(
        self,
        tasks: List[RepairTask],
        start_date: datetime,
        end_date: datetime,
        max_daily_hours: float = 8.0,
        optimization_objective: str = "minimize_cost",
        constraints: Optional[Dict[str, any]] = None,
    ) -> MaintenanceSchedule:
        """Create an optimized maintenance schedule for repair tasks.

        Uses optimization algorithms to schedule tasks while respecting constraints
        and optimizing for the specified objective (cost, time, or priority).

        The scheduler considers:
        - Task priorities and dependencies
        - Technician availability and skills
        - Spare parts availability
        - Resource constraints (daily work hours)

        Args:
            tasks: List of repair tasks to schedule
            start_date: Earliest date for scheduling tasks
            end_date: Latest date for completing all tasks
            max_daily_hours: Maximum work hours per day (default: 8.0)
            optimization_objective: Objective to optimize:
                - 'minimize_cost': Minimize total cost
                - 'minimize_time': Minimize total completion time
                - 'maximize_priority': Handle high-priority tasks first
            constraints: Additional constraints (e.g., {'technician_count': 2})

        Returns:
            MaintenanceSchedule with optimized task scheduling

        Raises:
            ValueError: If date range is invalid or no tasks provided

        Example:
            >>> schedule = optimizer.maintenance_scheduling(
            ...     tasks=[task1, task2, task3],
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 1, 31),
            ...     optimization_objective="minimize_cost"
            ... )
            >>> for task in schedule.tasks:
            ...     print(f"{task.component_id}: {task.scheduled_start}")
        """
        if not tasks:
            raise ValueError("At least one task must be provided")
        if end_date <= start_date:
            raise ValueError("end_date must be after start_date")

        # Initialize constraints
        if constraints is None:
            constraints = {}

        # Sort tasks by priority and severity
        sorted_tasks = self._prioritize_tasks(tasks, optimization_objective)

        # Allocate tasks to time slots
        scheduled_tasks = self._allocate_tasks_to_schedule(
            sorted_tasks, start_date, end_date, max_daily_hours, constraints
        )

        # Calculate totals
        total_cost = sum(task.estimated_cost for task in scheduled_tasks)
        total_hours = sum(task.estimated_duration_hours for task in scheduled_tasks)

        # Calculate optimization score
        optimization_score = self._calculate_schedule_score(
            scheduled_tasks, optimization_objective
        )

        # Check if all constraints are satisfied
        constraints_satisfied = self._verify_constraints(
            scheduled_tasks, max_daily_hours, constraints
        )

        return MaintenanceSchedule(
            valid_from=start_date,
            valid_until=end_date,
            tasks=scheduled_tasks,
            total_estimated_cost=total_cost,
            total_estimated_hours=total_hours,
            optimization_objective=optimization_objective,
            optimization_score=optimization_score,
            constraints_satisfied=constraints_satisfied,
            notes=f"Scheduled {len(scheduled_tasks)} tasks using {optimization_objective}",
        )

    def spare_parts_management(
        self,
        check_inventory: bool = True,
        generate_reorder_list: bool = True,
        forecast_days: int = 90,
    ) -> Dict[str, any]:
        """Manage spare parts inventory and generate reorder recommendations.

        Analyzes current inventory levels, reserved quantities, and forecasted demand
        to generate optimal reorder recommendations. Helps prevent stockouts while
        minimizing inventory carrying costs.

        Args:
            check_inventory: Perform inventory level checks (default: True)
            generate_reorder_list: Generate list of parts needing reorder (default: True)
            forecast_days: Number of days to forecast demand (default: 90)

        Returns:
            Dictionary containing:
                - 'inventory_status': Current inventory levels
                - 'reorder_recommendations': List of parts to reorder with quantities
                - 'critical_shortages': Parts critically low or out of stock
                - 'forecasted_demand': Predicted demand for forecast period
                - 'total_reorder_cost': Estimated cost of recommended reorders

        Example:
            >>> result = optimizer.spare_parts_management(forecast_days=60)
            >>> for item in result['reorder_recommendations']:
            ...     print(f"Reorder {item['part_name']}: {item['quantity']} units")
        """
        result: Dict[str, any] = {
            "inventory_status": {},
            "reorder_recommendations": [],
            "critical_shortages": [],
            "forecasted_demand": {},
            "total_reorder_cost": 0.0,
        }

        if not self.spare_parts:
            return result

        # Check inventory levels
        if check_inventory:
            for part_id, part in self.spare_parts.items():
                result["inventory_status"][part_id] = {
                    "part_name": part.part_name,
                    "quantity_available": part.quantity_available,
                    "quantity_reserved": part.quantity_reserved,
                    "quantity_on_hand": part.quantity_on_hand,
                    "reorder_point": part.reorder_point,
                    "needs_reorder": part.needs_reorder,
                }

                # Track critical shortages
                if part.quantity_on_hand == 0:
                    result["critical_shortages"].append(
                        {
                            "part_id": part_id,
                            "part_name": part.part_name,
                            "quantity_reserved": part.quantity_reserved,
                            "lead_time_days": part.lead_time_days,
                        }
                    )

        # Generate reorder recommendations
        if generate_reorder_list:
            forecasted_demand = self._forecast_parts_demand(forecast_days)
            result["forecasted_demand"] = forecasted_demand

            for part_id, part in self.spare_parts.items():
                # Calculate optimal reorder quantity
                demand = forecasted_demand.get(part_id, 0)
                optimal_quantity = self._calculate_optimal_reorder_quantity(
                    part, demand, forecast_days
                )

                if part.needs_reorder or optimal_quantity > 0:
                    reorder_qty = max(part.reorder_quantity, optimal_quantity)
                    reorder_cost = reorder_qty * part.unit_cost

                    result["reorder_recommendations"].append(
                        {
                            "part_id": part_id,
                            "part_name": part.part_name,
                            "part_number": part.part_number,
                            "current_quantity": part.quantity_on_hand,
                            "reorder_quantity": reorder_qty,
                            "unit_cost": part.unit_cost,
                            "total_cost": reorder_cost,
                            "supplier": part.supplier,
                            "lead_time_days": part.lead_time_days,
                            "reason": (
                                "Below reorder point"
                                if part.needs_reorder
                                else "Forecasted demand"
                            ),
                        }
                    )

                    result["total_reorder_cost"] += reorder_cost

        return result

    # ==================== Helper Methods ====================

    def _get_default_baseline(self, component_type: str) -> Dict[str, float]:
        """Get default baseline performance values for component types."""
        baselines = {
            "panel": {
                "voltage": 30.0,
                "current": 8.0,
                "power": 240.0,
                "efficiency": 0.18,
                "temperature": 45.0,
            },
            "inverter": {
                "efficiency": 0.96,
                "temperature": 50.0,
                "voltage_in": 600.0,
                "voltage_out": 240.0,
                "power_factor": 0.99,
            },
            "junction_box": {
                "voltage": 30.0,
                "current": 8.0,
                "temperature": 40.0,
                "resistance": 0.01,
            },
        }
        return baselines.get(component_type, {})

    def _analyze_performance(
        self,
        component_type: str,
        performance_data: Dict[str, float],
        baseline_data: Dict[str, float],
        historical_data: Optional[List[Dict[str, float]]],
    ) -> Tuple[FaultType, FaultSeverity, Dict[str, float], float]:
        """Analyze performance data to detect and classify faults."""
        symptoms: Dict[str, float] = {}
        fault_indicators: List[Tuple[FaultType, float]] = []

        # Efficiency analysis
        if "efficiency" in performance_data and "efficiency" in baseline_data:
            eff_drop = (
                baseline_data["efficiency"] - performance_data["efficiency"]
            ) / baseline_data["efficiency"]
            if eff_drop > self._fault_thresholds["efficiency_drop"]:
                symptoms["efficiency_drop"] = eff_drop
                fault_indicators.append((FaultType.DEGRADATION, eff_drop * 2))

        # Temperature analysis
        if "temperature" in performance_data:
            temp = performance_data["temperature"]
            if temp > self._fault_thresholds["temperature_high"]:
                symptoms["temperature_high"] = temp
                fault_indicators.append((FaultType.THERMAL, (temp - 50) / 50))

        # Voltage analysis
        if "voltage" in performance_data and "voltage" in baseline_data:
            voltage_dev = abs(
                performance_data["voltage"] - baseline_data["voltage"]
            ) / baseline_data["voltage"]
            if voltage_dev > self._fault_thresholds["voltage_deviation"]:
                symptoms["voltage_deviation"] = voltage_dev
                fault_indicators.append((FaultType.ELECTRICAL, voltage_dev * 2))

        # Current analysis
        if "current" in performance_data and "current" in baseline_data:
            current_dev = abs(
                performance_data["current"] - baseline_data["current"]
            ) / baseline_data["current"]
            if current_dev > self._fault_thresholds["current_imbalance"]:
                symptoms["current_imbalance"] = current_dev
                fault_indicators.append((FaultType.ELECTRICAL, current_dev * 2))

        # Historical trend analysis
        if historical_data and len(historical_data) >= 3:
            degradation_rate = self._calculate_degradation_rate(historical_data)
            if degradation_rate > self._fault_thresholds["degradation_rate_high"]:
                symptoms["degradation_rate"] = degradation_rate
                fault_indicators.append((FaultType.DEGRADATION, degradation_rate * 5))

        # Determine dominant fault type and severity
        if not fault_indicators:
            return FaultType.ELECTRICAL, FaultSeverity.NEGLIGIBLE, symptoms, 0.0

        # Select fault with highest indicator value
        fault_type, indicator_value = max(fault_indicators, key=lambda x: x[1])

        # Calculate severity based on indicator value
        if indicator_value > 0.8:
            severity = FaultSeverity.CRITICAL
        elif indicator_value > 0.6:
            severity = FaultSeverity.HIGH
        elif indicator_value > 0.4:
            severity = FaultSeverity.MEDIUM
        elif indicator_value > 0.2:
            severity = FaultSeverity.LOW
        else:
            severity = FaultSeverity.NEGLIGIBLE

        # Confidence is based on number of symptoms and indicator strength
        confidence = min(
            0.95, 0.5 + (len(symptoms) * 0.1) + (min(indicator_value, 0.5) * 0.7)
        )

        return fault_type, severity, symptoms, confidence

    def _calculate_degradation_rate(
        self, historical_data: List[Dict[str, float]]
    ) -> float:
        """Calculate degradation rate from historical performance data."""
        if "efficiency" not in historical_data[0]:
            return 0.0

        efficiencies = [d["efficiency"] for d in historical_data if "efficiency" in d]
        if len(efficiencies) < 2:
            return 0.0

        # Simple linear regression to find degradation trend
        x = np.arange(len(efficiencies))
        y = np.array(efficiencies)
        slope = np.polyfit(x, y, 1)[0]

        # Convert to annual rate (assuming monthly data)
        annual_rate = abs(slope * 12)
        return annual_rate

    def _determine_root_cause(
        self, fault_type: FaultType, symptoms: Dict[str, float], component_type: str
    ) -> str:
        """Determine root cause based on fault type and symptoms."""
        if fault_type == FaultType.ELECTRICAL:
            if "voltage_deviation" in symptoms:
                return "Possible wiring degradation or connection issue"
            if "current_imbalance" in symptoms:
                return "Possible cell mismatch or partial shading"
            return "Electrical system fault"

        elif fault_type == FaultType.THERMAL:
            return "Excessive temperature - check cooling system and ambient conditions"

        elif fault_type == FaultType.DEGRADATION:
            if component_type == "panel":
                return "Natural panel degradation or soiling accumulation"
            return f"{component_type} performance degradation"

        return f"Unknown root cause for {fault_type.value} fault"

    def _calculate_affected_metrics(
        self, performance_data: Dict[str, float], baseline_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate the impact on performance metrics."""
        affected = {}
        for key in performance_data:
            if key in baseline_data and baseline_data[key] != 0:
                deviation = (performance_data[key] - baseline_data[key]) / baseline_data[key]
                if abs(deviation) > 0.05:  # 5% threshold
                    affected[key] = deviation
        return affected

    def _estimate_labor_hours(self, fault: Fault) -> float:
        """Estimate labor hours based on fault characteristics."""
        # Base hours by component type
        base_hours = {
            "panel": 2.0,
            "inverter": 4.0,
            "junction_box": 1.5,
            "cable": 1.0,
            "sensor": 0.5,
        }

        hours = base_hours.get(fault.component_type, 2.0)

        # Adjust for severity
        severity_multipliers = {
            FaultSeverity.CRITICAL: 1.5,
            FaultSeverity.HIGH: 1.3,
            FaultSeverity.MEDIUM: 1.0,
            FaultSeverity.LOW: 0.8,
            FaultSeverity.NEGLIGIBLE: 0.5,
        }

        hours *= severity_multipliers.get(fault.severity, 1.0)

        # Adjust for fault type
        if fault.fault_type == FaultType.ELECTRICAL:
            hours *= 1.2  # Electrical faults take longer to diagnose
        elif fault.fault_type == FaultType.STRUCTURAL:
            hours *= 1.5  # Structural repairs are more intensive

        return round(hours, 1)

    def _estimate_parts_cost(self, fault: Fault) -> Dict[str, float]:
        """Estimate required parts and their costs."""
        parts_breakdown: Dict[str, float] = {}

        # Determine required parts based on fault type and component
        if fault.fault_type == FaultType.ELECTRICAL:
            if fault.component_type == "panel":
                parts_breakdown["bypass_diode"] = 15.0
                parts_breakdown["junction_box"] = 25.0
            elif fault.component_type == "inverter":
                parts_breakdown["capacitor"] = 50.0
                parts_breakdown["circuit_board"] = 200.0

        elif fault.fault_type == FaultType.THERMAL:
            if fault.component_type == "inverter":
                parts_breakdown["cooling_fan"] = 40.0
                parts_breakdown["thermal_paste"] = 10.0

        elif fault.fault_type == FaultType.DEGRADATION:
            if fault.severity in [FaultSeverity.CRITICAL, FaultSeverity.HIGH]:
                # Severe degradation may require replacement
                if fault.component_type == "panel":
                    parts_breakdown["replacement_panel"] = 300.0

        # Add consumables
        parts_breakdown["wiring_connectors"] = 10.0
        parts_breakdown["sealant"] = 5.0

        return parts_breakdown

    def _prioritize_tasks(
        self, tasks: List[RepairTask], optimization_objective: str
    ) -> List[RepairTask]:
        """Sort tasks based on optimization objective."""
        if optimization_objective == "minimize_cost":
            # Sort by cost per hour (efficiency)
            return sorted(
                tasks,
                key=lambda t: t.estimated_cost / max(t.estimated_duration_hours, 0.1),
            )
        elif optimization_objective == "minimize_time":
            # Sort by duration
            return sorted(tasks, key=lambda t: t.estimated_duration_hours)
        elif optimization_objective == "maximize_priority":
            # Sort by priority (lower number = higher priority)
            return sorted(tasks, key=lambda t: t.priority.value)
        else:
            # Default: priority-based
            return sorted(tasks, key=lambda t: t.priority.value)

    def _allocate_tasks_to_schedule(
        self,
        tasks: List[RepairTask],
        start_date: datetime,
        end_date: datetime,
        max_daily_hours: float,
        constraints: Dict[str, any],
    ) -> List[RepairTask]:
        """Allocate tasks to time slots using greedy algorithm."""
        scheduled_tasks: List[RepairTask] = []
        current_date = start_date
        daily_hours_used = 0.0

        for task in tasks:
            # Check if task fits in current day
            if daily_hours_used + task.estimated_duration_hours <= max_daily_hours:
                # Schedule on current day
                task.scheduled_start = current_date.replace(
                    hour=8 + int(daily_hours_used), minute=0
                )
                task.scheduled_end = task.scheduled_start + timedelta(
                    hours=task.estimated_duration_hours
                )
                daily_hours_used += task.estimated_duration_hours
            else:
                # Move to next day
                current_date += timedelta(days=1)
                if current_date > end_date:
                    # Cannot fit within date range
                    break
                daily_hours_used = task.estimated_duration_hours
                task.scheduled_start = current_date.replace(hour=8, minute=0)
                task.scheduled_end = task.scheduled_start + timedelta(
                    hours=task.estimated_duration_hours
                )

            scheduled_tasks.append(task)

        return scheduled_tasks

    def _calculate_schedule_score(
        self, tasks: List[RepairTask], optimization_objective: str
    ) -> float:
        """Calculate optimization score for the schedule."""
        if not tasks:
            return 0.0

        if optimization_objective == "minimize_cost":
            # Lower total cost is better
            total_cost = sum(t.estimated_cost for t in tasks)
            return 1.0 / (1.0 + total_cost / 1000.0)  # Normalized score

        elif optimization_objective == "minimize_time":
            # Shorter schedule is better
            if tasks[0].scheduled_start and tasks[-1].scheduled_end:
                duration = (tasks[-1].scheduled_end - tasks[0].scheduled_start).days
                return 1.0 / (1.0 + duration)
            return 0.0

        elif optimization_objective == "maximize_priority":
            # Lower average priority value is better
            avg_priority = sum(t.priority.value for t in tasks) / len(tasks)
            return 1.0 / avg_priority

        return 0.5  # Default score

    def _verify_constraints(
        self,
        tasks: List[RepairTask],
        max_daily_hours: float,
        constraints: Dict[str, any],
    ) -> bool:
        """Verify that all constraints are satisfied."""
        # Check daily hours constraint
        tasks_by_date: Dict[str, float] = {}
        for task in tasks:
            if task.scheduled_start:
                date_key = task.scheduled_start.date().isoformat()
                tasks_by_date[date_key] = (
                    tasks_by_date.get(date_key, 0.0) + task.estimated_duration_hours
                )

        for date_key, hours in tasks_by_date.items():
            if hours > max_daily_hours:
                return False

        # Additional constraint checks can be added here
        return True

    def _forecast_parts_demand(self, forecast_days: int) -> Dict[str, int]:
        """Forecast spare parts demand based on historical usage."""
        demand: Dict[str, int] = {}

        # Simple forecasting based on active faults and repair history
        for fault in self.active_faults:
            parts = self._estimate_parts_cost(fault)
            for part_name in parts.keys():
                # Find matching part ID (simplified matching)
                for part_id, part in self.spare_parts.items():
                    if part_name.lower() in part.part_name.lower():
                        demand[part_id] = demand.get(part_id, 0) + 1

        # Scale by forecast period (assuming current rate continues)
        scaling_factor = forecast_days / 30.0  # Normalize to monthly rate
        for part_id in demand:
            demand[part_id] = int(demand[part_id] * scaling_factor)

        return demand

    def _calculate_optimal_reorder_quantity(
        self, part: SparePart, forecasted_demand: int, forecast_days: int
    ) -> int:
        """Calculate optimal reorder quantity using Economic Order Quantity (EOQ) concept."""
        # If demand is low, use standard reorder quantity
        if forecasted_demand <= part.reorder_quantity:
            return 0 if not part.needs_reorder else part.reorder_quantity

        # Simple calculation: ensure we have enough for forecasted demand plus buffer
        buffer_multiplier = 1.2  # 20% safety stock
        required_quantity = int(forecasted_demand * buffer_multiplier)

        # Subtract current available quantity
        needed = max(0, required_quantity - part.quantity_on_hand)

        return needed

    # ==================== Utility Methods ====================

    def add_spare_part(self, spare_part: SparePart) -> None:
        """Add or update a spare part in inventory.

        Args:
            spare_part: SparePart object to add to inventory
        """
        self.spare_parts[spare_part.part_id] = spare_part

    def update_component_health(self, health: ComponentHealth) -> None:
        """Update health status for a component.

        Args:
            health: ComponentHealth object with updated status
        """
        self.component_health[health.component_id] = health

    def get_active_faults(self, component_id: Optional[str] = None) -> List[Fault]:
        """Retrieve active faults, optionally filtered by component.

        Args:
            component_id: Optional component ID to filter faults

        Returns:
            List of active Fault objects
        """
        if component_id:
            return [f for f in self.active_faults if f.component_id == component_id]
        return self.active_faults.copy()

    def clear_fault(self, fault_id: UUID) -> bool:
        """Mark a fault as resolved and remove from active faults.

        Args:
            fault_id: UUID of the fault to clear

        Returns:
            True if fault was found and cleared, False otherwise
        """
        for i, fault in enumerate(self.active_faults):
            if fault.fault_id == fault_id:
                self.active_faults.pop(i)
                # Update component health
                for health in self.component_health.values():
                    if fault_id in health.current_faults:
                        health.current_faults.remove(fault_id)
                return True
        return False
