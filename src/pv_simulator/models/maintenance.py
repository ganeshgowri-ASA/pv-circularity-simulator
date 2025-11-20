"""Pydantic models for maintenance, repair, and spare parts management.

This module defines the data structures used throughout the RepairOptimizer system,
ensuring type safety and validation for all maintenance-related operations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class FaultType(str, Enum):
    """Types of faults that can occur in PV systems."""

    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    DEGRADATION = "degradation"
    CONNECTION = "connection"
    INVERTER = "inverter"
    SENSOR = "sensor"
    STRUCTURAL = "structural"


class FaultSeverity(str, Enum):
    """Severity levels for diagnosed faults."""

    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"  # Attention required soon
    MEDIUM = "medium"  # Should be addressed
    LOW = "low"  # Minor issue
    NEGLIGIBLE = "negligible"  # Monitoring only


class MaintenanceType(str, Enum):
    """Types of maintenance activities."""

    CORRECTIVE = "corrective"  # Fix a fault
    PREVENTIVE = "preventive"  # Prevent future faults
    PREDICTIVE = "predictive"  # Based on condition monitoring
    INSPECTION = "inspection"  # Regular inspection
    CALIBRATION = "calibration"  # Sensor/equipment calibration


class MaintenancePriority(int, Enum):
    """Priority levels for maintenance scheduling (1=highest, 5=lowest)."""

    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5


class ComponentStatus(str, Enum):
    """Operational status of PV system components."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class Fault(BaseModel):
    """Represents a diagnosed fault in a PV system component.

    Attributes:
        fault_id: Unique identifier for the fault
        component_id: Identifier of the affected component
        component_type: Type of component (e.g., 'panel', 'inverter', 'junction_box')
        fault_type: Category of the fault
        severity: Severity level of the fault
        description: Human-readable description of the fault
        detected_at: Timestamp when the fault was detected
        symptoms: Observable symptoms (e.g., voltage drop, temperature spike)
        diagnosis_confidence: Confidence level of the diagnosis (0.0 to 1.0)
        root_cause: Identified root cause if determined
        affected_metrics: Performance metrics affected by the fault
    """

    model_config = ConfigDict(use_enum_values=True)

    fault_id: UUID = Field(default_factory=uuid4)
    component_id: str
    component_type: str
    fault_type: FaultType
    severity: FaultSeverity
    description: str
    detected_at: datetime = Field(default_factory=datetime.now)
    symptoms: Dict[str, float] = Field(default_factory=dict)
    diagnosis_confidence: float = Field(ge=0.0, le=1.0)
    root_cause: Optional[str] = None
    affected_metrics: Dict[str, float] = Field(default_factory=dict)

    @field_validator("diagnosis_confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure diagnosis confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Diagnosis confidence must be between 0.0 and 1.0")
        return v


class RepairTask(BaseModel):
    """Represents a repair task for addressing a fault.

    Attributes:
        task_id: Unique identifier for the task
        fault_id: Reference to the associated fault
        component_id: Identifier of the component to be repaired
        task_type: Type of maintenance/repair activity
        priority: Scheduling priority
        estimated_duration_hours: Expected time to complete (hours)
        required_skills: List of required technician skills
        required_parts: List of spare part IDs needed
        estimated_cost: Estimated total cost (labor + parts)
        scheduled_start: Scheduled start time
        scheduled_end: Scheduled completion time
        actual_start: Actual start time (if started)
        actual_end: Actual completion time (if completed)
        status: Current status of the task
        assigned_technician: Technician assigned to the task
        notes: Additional notes or instructions
    """

    model_config = ConfigDict(use_enum_values=True)

    task_id: UUID = Field(default_factory=uuid4)
    fault_id: Optional[UUID] = None
    component_id: str
    task_type: MaintenanceType
    priority: MaintenancePriority
    estimated_duration_hours: float = Field(gt=0)
    required_skills: List[str] = Field(default_factory=list)
    required_parts: List[str] = Field(default_factory=list)
    estimated_cost: float = Field(ge=0.0)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    status: str = "pending"
    assigned_technician: Optional[str] = None
    notes: str = ""

    @field_validator("estimated_duration_hours")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        """Ensure duration is positive."""
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v


class MaintenanceSchedule(BaseModel):
    """Represents an optimized maintenance schedule.

    Attributes:
        schedule_id: Unique identifier for the schedule
        created_at: Timestamp when the schedule was created
        valid_from: Start of the schedule validity period
        valid_until: End of the schedule validity period
        tasks: List of scheduled repair tasks
        total_estimated_cost: Sum of all task costs
        total_estimated_hours: Sum of all task durations
        optimization_objective: Objective used for optimization
        optimization_score: Score achieved by this schedule
        constraints_satisfied: Whether all constraints are met
        notes: Additional schedule information
    """

    schedule_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    valid_from: datetime
    valid_until: datetime
    tasks: List[RepairTask] = Field(default_factory=list)
    total_estimated_cost: float = Field(ge=0.0)
    total_estimated_hours: float = Field(ge=0.0)
    optimization_objective: str = "minimize_cost"
    optimization_score: float = 0.0
    constraints_satisfied: bool = True
    notes: str = ""

    @field_validator("valid_until")
    @classmethod
    def validate_period(cls, v: datetime, info) -> datetime:
        """Ensure schedule period is valid."""
        if "valid_from" in info.data and v <= info.data["valid_from"]:
            raise ValueError("valid_until must be after valid_from")
        return v


class SparePart(BaseModel):
    """Represents a spare part in inventory.

    Attributes:
        part_id: Unique identifier for the part
        part_name: Name of the part
        part_number: Manufacturer part number
        category: Category (e.g., 'panel', 'inverter', 'cable')
        quantity_available: Current inventory quantity
        quantity_reserved: Quantity reserved for scheduled tasks
        unit_cost: Cost per unit
        lead_time_days: Days required to restock
        reorder_point: Inventory level triggering reorder
        reorder_quantity: Quantity to order when restocking
        supplier: Primary supplier name
        compatible_components: List of compatible component types
        location: Storage location
        last_restocked: Date of last restock
        notes: Additional part information
    """

    part_id: str
    part_name: str
    part_number: str
    category: str
    quantity_available: int = Field(ge=0)
    quantity_reserved: int = Field(ge=0, default=0)
    unit_cost: float = Field(ge=0.0)
    lead_time_days: int = Field(ge=0)
    reorder_point: int = Field(ge=0)
    reorder_quantity: int = Field(gt=0)
    supplier: str
    compatible_components: List[str] = Field(default_factory=list)
    location: str = ""
    last_restocked: Optional[datetime] = None
    notes: str = ""

    @property
    def quantity_on_hand(self) -> int:
        """Calculate available quantity not reserved."""
        return max(0, self.quantity_available - self.quantity_reserved)

    @property
    def needs_reorder(self) -> bool:
        """Check if inventory is below reorder point."""
        return self.quantity_on_hand <= self.reorder_point


class ComponentHealth(BaseModel):
    """Represents the health status of a PV system component.

    Attributes:
        component_id: Unique identifier for the component
        component_type: Type of component
        status: Current operational status
        health_score: Overall health score (0.0=failed, 1.0=perfect)
        performance_metrics: Current performance measurements
        degradation_rate: Rate of performance degradation per year
        estimated_remaining_life_years: Estimated years until replacement
        last_maintenance: Date of last maintenance
        next_maintenance_due: Date when next maintenance is due
        maintenance_history: List of past maintenance task IDs
        current_faults: List of active fault IDs
    """

    model_config = ConfigDict(use_enum_values=True)

    component_id: str
    component_type: str
    status: ComponentStatus
    health_score: float = Field(ge=0.0, le=1.0)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    degradation_rate: float = Field(ge=0.0)
    estimated_remaining_life_years: Optional[float] = Field(ge=0.0, default=None)
    last_maintenance: Optional[datetime] = None
    next_maintenance_due: Optional[datetime] = None
    maintenance_history: List[UUID] = Field(default_factory=list)
    current_faults: List[UUID] = Field(default_factory=list)


class RepairCostEstimate(BaseModel):
    """Represents a detailed cost estimate for a repair.

    Attributes:
        estimate_id: Unique identifier for the estimate
        fault_id: Associated fault ID
        component_id: Component to be repaired
        labor_hours: Estimated labor hours
        labor_rate: Labor cost per hour
        labor_cost: Total labor cost
        parts_cost: Total cost of required parts
        parts_breakdown: Detailed cost per part
        overhead_cost: Additional overhead costs
        total_cost: Total estimated cost
        confidence_level: Confidence in the estimate (0.0 to 1.0)
        estimated_at: When the estimate was created
        valid_until: When the estimate expires
        notes: Additional estimate details
    """

    estimate_id: UUID = Field(default_factory=uuid4)
    fault_id: UUID
    component_id: str
    labor_hours: float = Field(ge=0.0)
    labor_rate: float = Field(ge=0.0)
    labor_cost: float = Field(ge=0.0)
    parts_cost: float = Field(ge=0.0)
    parts_breakdown: Dict[str, float] = Field(default_factory=dict)
    overhead_cost: float = Field(ge=0.0, default=0.0)
    total_cost: float = Field(ge=0.0)
    confidence_level: float = Field(ge=0.0, le=1.0)
    estimated_at: datetime = Field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    notes: str = ""

    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(cls, v: float, info) -> float:
        """Ensure total cost matches component costs."""
        if "labor_cost" in info.data and "parts_cost" in info.data and "overhead_cost" in info.data:
            expected = info.data["labor_cost"] + info.data["parts_cost"] + info.data["overhead_cost"]
            if abs(v - expected) > 0.01:  # Allow small floating point differences
                raise ValueError(
                    f"Total cost {v} does not match sum of components {expected}"
                )
        return v
