"""
Work Order Management System for PV Maintenance (B08-S04).

This module provides comprehensive work order management including technician
assignment, task tracking, and completion verification.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from pydantic import BaseModel, Field

from ...models import (
    WorkOrder,
    WorkOrderStatus,
    MaintenanceSchedule,
    MaintenanceType,
    MaintenancePriority,
    Technician,
    SparePart,
)


class WorkOrderConfig(BaseModel):
    """
    Configuration for work order management.

    Attributes:
        auto_assign: Automatically assign work orders to technicians
        require_verification: Require verification after completion
        enable_notifications: Enable notifications for status changes
        max_concurrent_orders_per_technician: Maximum concurrent orders per technician
    """

    auto_assign: bool = Field(default=True, description="Auto-assign work orders")
    require_verification: bool = Field(default=True, description="Require verification")
    enable_notifications: bool = Field(default=True, description="Enable notifications")
    max_concurrent_orders_per_technician: int = Field(
        default=3,
        description="Max concurrent orders per technician"
    )


class WorkOrderManagement:
    """
    Comprehensive work order management system.

    This class manages the full lifecycle of work orders including creation,
    assignment, tracking, and verification.

    Attributes:
        config: Work order management configuration
        work_orders: Dictionary of work orders by ID
        technicians: Dictionary of technicians by ID
        assignment_history: History of technician assignments
    """

    def __init__(self, config: Optional[WorkOrderConfig] = None):
        """
        Initialize the WorkOrderManagement system.

        Args:
            config: Optional configuration
        """
        self.config = config or WorkOrderConfig()
        self.work_orders: Dict[str, WorkOrder] = {}
        self.technicians: Dict[str, Technician] = {}
        self.assignment_history: List[Dict] = []

    def create_work_order(
        self,
        maintenance_schedule: MaintenanceSchedule,
        work_order_number: Optional[str] = None,
        auto_assign: Optional[bool] = None,
    ) -> WorkOrder:
        """
        Create a work order from a maintenance schedule.

        Args:
            maintenance_schedule: Maintenance schedule to convert
            work_order_number: Optional custom work order number
            auto_assign: Override auto-assignment setting

        Returns:
            Created WorkOrder object

        Example:
            >>> wom = WorkOrderManagement()
            >>> schedule = MaintenanceSchedule(...)
            >>> work_order = wom.create_work_order(schedule)
            >>> print(f"Created work order: {work_order.work_order_number}")
        """
        if work_order_number is None:
            work_order_number = self._generate_work_order_number(
                maintenance_schedule.site_id
            )

        # Estimate cost based on maintenance type
        estimated_cost = self._estimate_work_order_cost(
            maintenance_schedule.maintenance_type,
            maintenance_schedule.estimated_duration_hours,
        )

        work_order = WorkOrder(
            work_order_number=work_order_number,
            title=maintenance_schedule.schedule_name,
            site_id=maintenance_schedule.site_id,
            maintenance_schedule_id=maintenance_schedule.id,
            maintenance_type=maintenance_schedule.maintenance_type,
            priority=maintenance_schedule.priority,
            status=WorkOrderStatus.DRAFT,
            scheduled_start=datetime.combine(
                maintenance_schedule.scheduled_date,
                datetime.min.time()
            ),
            scheduled_end=datetime.combine(
                maintenance_schedule.scheduled_date,
                datetime.min.time()
            ) + timedelta(hours=maintenance_schedule.estimated_duration_hours),
            description=maintenance_schedule.description or "",
            required_parts={
                part: 1 for part in maintenance_schedule.required_parts
            },
            estimated_cost=estimated_cost,
            related_defects=maintenance_schedule.related_defects,
        )

        self.work_orders[work_order.id] = work_order

        # Auto-assign if configured
        should_auto_assign = (
            auto_assign if auto_assign is not None else self.config.auto_assign
        )
        if should_auto_assign:
            self.technician_assignment(work_order.id)

        return work_order

    def technician_assignment(
        self,
        work_order_id: str,
        technician_id: Optional[str] = None,
    ) -> Optional[WorkOrder]:
        """
        Assign a technician to a work order.

        Automatically selects the best available technician if not specified,
        considering skills, availability, and workload.

        Args:
            work_order_id: Work order identifier
            technician_id: Optional specific technician to assign

        Returns:
            Updated WorkOrder or None if assignment failed

        Example:
            >>> wom = WorkOrderManagement()
            >>> wom.technician_assignment("WO-001", "TECH-123")
            >>> # Or let the system auto-assign
            >>> wom.technician_assignment("WO-002")
        """
        if work_order_id not in self.work_orders:
            return None

        work_order = self.work_orders[work_order_id]

        # If technician not specified, find the best match
        if technician_id is None:
            technician_id = self._find_best_technician(work_order)

        if technician_id is None:
            # No suitable technician found
            return None

        if technician_id not in self.technicians:
            return None

        technician = self.technicians[technician_id]

        # Check if technician is available and not overloaded
        if not self._can_assign_to_technician(technician):
            return None

        # Assign technician
        work_order.assigned_technician_id = technician_id
        work_order.status = WorkOrderStatus.ASSIGNED

        # Update technician's assigned work orders
        if work_order_id not in technician.assigned_work_orders:
            technician.assigned_work_orders.append(work_order_id)

        # Record assignment history
        self.assignment_history.append({
            "work_order_id": work_order_id,
            "technician_id": technician_id,
            "assigned_at": datetime.utcnow(),
            "assigned_by": "system",
        })

        return work_order

    def task_tracking(
        self,
        work_order_id: str,
        status: Optional[WorkOrderStatus] = None,
        actual_start: Optional[datetime] = None,
        actual_end: Optional[datetime] = None,
        notes: Optional[str] = None,
        actual_cost: Optional[float] = None,
    ) -> Optional[WorkOrder]:
        """
        Track work order progress and update status.

        Updates work order status, timestamps, costs, and notes as work
        progresses through its lifecycle.

        Args:
            work_order_id: Work order identifier
            status: New status
            actual_start: Actual start datetime
            actual_end: Actual end datetime
            notes: Progress notes
            actual_cost: Actual cost incurred

        Returns:
            Updated WorkOrder or None if not found

        Example:
            >>> wom = WorkOrderManagement()
            >>> # Start work
            >>> wom.task_tracking(
            ...     "WO-001",
            ...     status=WorkOrderStatus.IN_PROGRESS,
            ...     actual_start=datetime.now()
            ... )
            >>> # Complete work
            >>> wom.task_tracking(
            ...     "WO-001",
            ...     status=WorkOrderStatus.COMPLETED,
            ...     actual_end=datetime.now(),
            ...     actual_cost=450.50
            ... )
        """
        if work_order_id not in self.work_orders:
            return None

        work_order = self.work_orders[work_order_id]

        # Update status
        if status is not None:
            work_order.status = status

        # Update timestamps
        if actual_start is not None:
            work_order.actual_start = actual_start

        if actual_end is not None:
            work_order.actual_end = actual_end

        # Update cost
        if actual_cost is not None:
            work_order.actual_cost = actual_cost

        # Update notes
        if notes is not None:
            if work_order.completion_notes:
                work_order.completion_notes += f"\n\n{notes}"
            else:
                work_order.completion_notes = notes

        # Update timestamp
        work_order.updated_at = datetime.utcnow()

        return work_order

    def completion_verification(
        self,
        work_order_id: str,
        verified_by: str,
        verification_passed: bool,
        verification_notes: Optional[str] = None,
        photos: Optional[List[str]] = None,
    ) -> Optional[WorkOrder]:
        """
        Verify completion of a work order.

        Performs final verification of completed work, including quality
        checks, documentation review, and acceptance.

        Args:
            work_order_id: Work order identifier
            verified_by: User performing verification
            verification_passed: Whether verification passed
            verification_notes: Verification notes
            photos: Optional list of verification photo paths

        Returns:
            Updated WorkOrder or None if not found

        Example:
            >>> wom = WorkOrderManagement()
            >>> wom.completion_verification(
            ...     work_order_id="WO-001",
            ...     verified_by="supervisor@example.com",
            ...     verification_passed=True,
            ...     verification_notes="All work completed satisfactorily"
            ... )
        """
        if work_order_id not in self.work_orders:
            return None

        work_order = self.work_orders[work_order_id]

        # Check if work order is in a completable state
        if work_order.status not in [WorkOrderStatus.COMPLETED, WorkOrderStatus.IN_PROGRESS]:
            return None

        # Update verification status
        work_order.verification_status = verification_passed
        work_order.verified_by = verified_by

        # Add verification notes
        if verification_notes:
            verification_text = (
                f"\n\n=== VERIFICATION ({datetime.utcnow().isoformat()}) ===\n"
                f"Verified by: {verified_by}\n"
                f"Status: {'PASSED' if verification_passed else 'FAILED'}\n"
                f"Notes: {verification_notes}"
            )
            if work_order.completion_notes:
                work_order.completion_notes += verification_text
            else:
                work_order.completion_notes = verification_text

        # Add photos
        if photos:
            work_order.photos.extend(photos)

        # Update status based on verification
        if verification_passed:
            work_order.status = WorkOrderStatus.VERIFIED
            # Release technician
            if work_order.assigned_technician_id:
                self._release_technician(
                    work_order.assigned_technician_id,
                    work_order_id
                )
        else:
            # Send back for rework
            work_order.status = WorkOrderStatus.IN_PROGRESS

        work_order.updated_at = datetime.utcnow()

        return work_order

    def add_technician(self, technician: Technician) -> None:
        """
        Add a technician to the management system.

        Args:
            technician: Technician object to add
        """
        self.technicians[technician.technician_id] = technician

    def get_work_order_status(self, work_order_id: str) -> Optional[Dict]:
        """
        Get detailed status information for a work order.

        Args:
            work_order_id: Work order identifier

        Returns:
            Dictionary with work order status details or None
        """
        if work_order_id not in self.work_orders:
            return None

        work_order = self.work_orders[work_order_id]

        # Calculate progress metrics
        duration_planned = None
        duration_actual = None
        if work_order.scheduled_start and work_order.scheduled_end:
            duration_planned = (
                work_order.scheduled_end - work_order.scheduled_start
            ).total_seconds() / 3600.0

        if work_order.actual_start and work_order.actual_end:
            duration_actual = (
                work_order.actual_end - work_order.actual_start
            ).total_seconds() / 3600.0

        return {
            "work_order_id": work_order_id,
            "work_order_number": work_order.work_order_number,
            "status": work_order.status.value,
            "assigned_technician_id": work_order.assigned_technician_id,
            "scheduled_start": work_order.scheduled_start,
            "scheduled_end": work_order.scheduled_end,
            "actual_start": work_order.actual_start,
            "actual_end": work_order.actual_end,
            "duration_planned_hours": duration_planned,
            "duration_actual_hours": duration_actual,
            "estimated_cost": work_order.estimated_cost,
            "actual_cost": work_order.actual_cost,
            "verification_status": work_order.verification_status,
            "verified_by": work_order.verified_by,
        }

    def get_technician_workload(self, technician_id: str) -> Optional[Dict]:
        """
        Get workload information for a technician.

        Args:
            technician_id: Technician identifier

        Returns:
            Dictionary with workload details or None
        """
        if technician_id not in self.technicians:
            return None

        technician = self.technicians[technician_id]
        assigned_orders = [
            self.work_orders[wo_id]
            for wo_id in technician.assigned_work_orders
            if wo_id in self.work_orders
        ]

        active_orders = [
            wo for wo in assigned_orders
            if wo.status in [
                WorkOrderStatus.ASSIGNED,
                WorkOrderStatus.IN_PROGRESS,
            ]
        ]

        return {
            "technician_id": technician_id,
            "name": technician.name,
            "total_assigned": len(assigned_orders),
            "active_orders": len(active_orders),
            "availability": technician.availability,
            "skills": technician.skills,
            "work_orders": [wo.work_order_number for wo in active_orders],
        }

    def _generate_work_order_number(self, site_id: str) -> str:
        """Generate a unique work order number."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"WO-{site_id}-{timestamp}"

    def _estimate_work_order_cost(
        self,
        maintenance_type: MaintenanceType,
        estimated_hours: float,
    ) -> float:
        """Estimate work order cost based on type and duration."""
        base_labor_rate = 75.0  # Per hour
        labor_cost = estimated_hours * base_labor_rate

        # Add material costs based on maintenance type
        material_cost = 0.0
        if maintenance_type == MaintenanceType.CORRECTIVE:
            material_cost = 200.0  # Average parts cost
        elif maintenance_type == MaintenanceType.PREVENTIVE:
            material_cost = 100.0
        elif maintenance_type == MaintenanceType.ROUTINE:
            material_cost = 25.0

        return round(labor_cost + material_cost, 2)

    def _find_best_technician(self, work_order: WorkOrder) -> Optional[str]:
        """Find the best available technician for a work order."""
        # Get required skills from maintenance schedule
        required_skills = set()
        if work_order.maintenance_schedule_id:
            # In a real system, would look up the schedule
            pass

        best_technician_id = None
        best_score = -1

        for tech_id, technician in self.technicians.items():
            if not self._can_assign_to_technician(technician):
                continue

            # Calculate match score
            score = 0

            # Availability
            if technician.availability:
                score += 10

            # Workload (prefer less busy technicians)
            active_count = len([
                wo_id for wo_id in technician.assigned_work_orders
                if wo_id in self.work_orders
                and self.work_orders[wo_id].status in [
                    WorkOrderStatus.ASSIGNED,
                    WorkOrderStatus.IN_PROGRESS,
                ]
            ])
            score += (self.config.max_concurrent_orders_per_technician - active_count) * 5

            # Skills match
            tech_skills = set(technician.skills)
            matching_skills = tech_skills & required_skills
            score += len(matching_skills) * 3

            if score > best_score:
                best_score = score
                best_technician_id = tech_id

        return best_technician_id

    def _can_assign_to_technician(self, technician: Technician) -> bool:
        """Check if a technician can accept more work orders."""
        if not technician.availability:
            return False

        active_count = len([
            wo_id for wo_id in technician.assigned_work_orders
            if wo_id in self.work_orders
            and self.work_orders[wo_id].status in [
                WorkOrderStatus.ASSIGNED,
                WorkOrderStatus.IN_PROGRESS,
            ]
        ])

        return active_count < self.config.max_concurrent_orders_per_technician

    def _release_technician(self, technician_id: str, work_order_id: str) -> None:
        """Release a technician from a completed work order."""
        if technician_id in self.technicians:
            technician = self.technicians[technician_id]
            if work_order_id in technician.assigned_work_orders:
                technician.assigned_work_orders.remove(work_order_id)
