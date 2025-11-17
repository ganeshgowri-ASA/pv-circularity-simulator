"""
Maintenance-related Pydantic models for PV system management.

This module defines models for maintenance scheduling, work orders, and spare parts.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, date

from pydantic import Field

from .base import BaseSchema, GeoLocation


class MaintenanceType(str, Enum):
    """
    Types of maintenance activities.
    """

    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    ROUTINE = "routine"
    INSPECTION = "inspection"


class MaintenancePriority(str, Enum):
    """
    Priority levels for maintenance activities.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class WorkOrderStatus(str, Enum):
    """
    Status of work orders.
    """

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    VERIFIED = "verified"


class SparePart(BaseSchema):
    """
    Spare part inventory item.

    Attributes:
        part_number: Unique part number
        part_name: Name of the part
        description: Part description
        category: Part category (panel, inverter, cable, etc.)
        quantity_available: Available quantity
        quantity_reserved: Reserved quantity
        reorder_level: Minimum quantity before reorder
        unit_cost: Cost per unit
        supplier: Supplier information
        lead_time_days: Lead time in days
        location: Storage location
    """

    part_number: str = Field(description="Part number")
    part_name: str = Field(description="Part name")
    description: Optional[str] = Field(None, description="Part description")
    category: str = Field(description="Part category")

    quantity_available: int = Field(default=0, ge=0, description="Available quantity")
    quantity_reserved: int = Field(default=0, ge=0, description="Reserved quantity")
    reorder_level: int = Field(default=5, ge=0, description="Reorder level")

    unit_cost: float = Field(ge=0.0, description="Unit cost")
    supplier: Optional[str] = Field(None, description="Supplier")
    lead_time_days: int = Field(default=30, ge=0, description="Lead time in days")
    location: Optional[str] = Field(None, description="Storage location")


class MaintenanceSchedule(BaseSchema):
    """
    Maintenance schedule entry.

    Attributes:
        schedule_name: Name of the schedule
        site_id: Site identifier
        maintenance_type: Type of maintenance
        priority: Priority level
        scheduled_date: Scheduled date
        estimated_duration_hours: Estimated duration in hours
        required_parts: List of required spare parts
        required_skills: List of required technician skills
        description: Maintenance description
        recurrence_pattern: Recurrence pattern (e.g., "monthly", "quarterly")
        last_performed: Last performed date
        next_due: Next due date
        related_defects: Related defect IDs
    """

    schedule_name: str = Field(description="Schedule name")
    site_id: str = Field(description="Site identifier")
    maintenance_type: MaintenanceType = Field(description="Maintenance type")
    priority: MaintenancePriority = Field(description="Priority level")

    scheduled_date: date = Field(description="Scheduled date")
    estimated_duration_hours: float = Field(ge=0.0, description="Estimated duration")

    required_parts: List[str] = Field(
        default_factory=list,
        description="Required part numbers"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Required skills"
    )

    description: Optional[str] = Field(None, description="Description")
    recurrence_pattern: Optional[str] = Field(None, description="Recurrence pattern")

    last_performed: Optional[date] = Field(None, description="Last performed date")
    next_due: Optional[date] = Field(None, description="Next due date")

    related_defects: List[str] = Field(
        default_factory=list,
        description="Related defect IDs"
    )


class Technician(BaseSchema):
    """
    Technician information.

    Attributes:
        technician_id: Unique technician identifier
        name: Technician name
        skills: List of skills/certifications
        availability: Availability status
        assigned_work_orders: List of assigned work order IDs
        location: Current location
        contact_info: Contact information
        hourly_rate: Hourly rate for cost calculations
    """

    technician_id: str = Field(description="Technician identifier")
    name: str = Field(description="Technician name")
    skills: List[str] = Field(default_factory=list, description="Skills")
    availability: bool = Field(default=True, description="Availability")

    assigned_work_orders: List[str] = Field(
        default_factory=list,
        description="Assigned work orders"
    )
    location: Optional[GeoLocation] = Field(None, description="Current location")
    contact_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Contact information"
    )
    hourly_rate: float = Field(default=50.0, ge=0.0, description="Hourly rate")


class WorkOrder(BaseSchema):
    """
    Comprehensive work order model.

    Attributes:
        work_order_number: Unique work order number
        title: Work order title
        site_id: Site identifier
        maintenance_schedule_id: Associated maintenance schedule ID
        maintenance_type: Type of maintenance
        priority: Priority level
        status: Current status
        assigned_technician_id: Assigned technician ID
        scheduled_start: Scheduled start datetime
        scheduled_end: Scheduled end datetime
        actual_start: Actual start datetime
        actual_end: Actual end datetime
        description: Detailed description
        required_parts: Required spare parts with quantities
        estimated_cost: Estimated total cost
        actual_cost: Actual total cost
        completion_notes: Notes upon completion
        verification_status: Verification status
        verified_by: User who verified the work
        related_defects: Related defect IDs
        photos: List of photo paths
        attachments: List of attachment paths
    """

    work_order_number: str = Field(description="Work order number")
    title: str = Field(description="Work order title")
    site_id: str = Field(description="Site identifier")

    maintenance_schedule_id: Optional[str] = Field(
        None,
        description="Maintenance schedule ID"
    )
    maintenance_type: MaintenanceType = Field(description="Maintenance type")
    priority: MaintenancePriority = Field(description="Priority level")
    status: WorkOrderStatus = Field(default=WorkOrderStatus.DRAFT, description="Status")

    assigned_technician_id: Optional[str] = Field(None, description="Assigned technician")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(None, description="Scheduled end")
    actual_start: Optional[datetime] = Field(None, description="Actual start")
    actual_end: Optional[datetime] = Field(None, description="Actual end")

    description: str = Field(description="Detailed description")
    required_parts: Dict[str, int] = Field(
        default_factory=dict,
        description="Required parts (part_number: quantity)"
    )

    estimated_cost: float = Field(default=0.0, ge=0.0, description="Estimated cost")
    actual_cost: float = Field(default=0.0, ge=0.0, description="Actual cost")

    completion_notes: Optional[str] = Field(None, description="Completion notes")
    verification_status: bool = Field(default=False, description="Verification status")
    verified_by: Optional[str] = Field(None, description="Verified by")

    related_defects: List[str] = Field(
        default_factory=list,
        description="Related defect IDs"
    )
    photos: List[str] = Field(default_factory=list, description="Photo paths")
    attachments: List[str] = Field(default_factory=list, description="Attachment paths")


class CorrectiveAction(BaseSchema):
    """
    Corrective action tracking.

    Attributes:
        action_id: Unique action identifier
        defect_id: Related defect identifier
        work_order_id: Related work order identifier
        action_type: Type of corrective action
        description: Action description
        status: Current status
        effectiveness_score: Effectiveness score (0-100)
        follow_up_required: Whether follow-up is required
        follow_up_date: Follow-up date
        notes: Additional notes
    """

    action_id: str = Field(default_factory=lambda: str(uuid4()), description="Action ID")
    defect_id: str = Field(description="Defect identifier")
    work_order_id: Optional[str] = Field(None, description="Work order ID")

    action_type: str = Field(description="Action type")
    description: str = Field(description="Action description")
    status: str = Field(default="planned", description="Status")

    effectiveness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Effectiveness score"
    )
    follow_up_required: bool = Field(default=False, description="Follow-up required")
    follow_up_date: Optional[date] = Field(None, description="Follow-up date")
    notes: Optional[str] = Field(None, description="Notes")


# Import uuid4 for CorrectiveAction
from uuid import uuid4
