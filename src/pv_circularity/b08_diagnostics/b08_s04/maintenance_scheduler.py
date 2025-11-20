"""
Maintenance Scheduler for PV System Management (B08-S04).

This module provides intelligent maintenance scheduling with preventive maintenance
planning, corrective action tracking, and spare parts management capabilities.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import calendar

from pydantic import BaseModel, Field

from ...models import (
    Defect,
    DefectSeverity,
    DiagnosticResult,
    MaintenanceSchedule,
    MaintenanceType,
    MaintenancePriority,
    SparePart,
    CorrectiveAction,
)


class MaintenancePolicy(BaseModel):
    """
    Configuration for maintenance scheduling policies.

    Attributes:
        preventive_interval_days: Days between preventive maintenance
        inspection_interval_days: Days between inspections
        cleaning_interval_days: Days between cleaning operations
        critical_response_hours: Response time for critical issues
        high_response_days: Response time for high priority issues
        enable_predictive: Enable predictive maintenance
    """

    preventive_interval_days: int = Field(default=180, description="Preventive maintenance interval")
    inspection_interval_days: int = Field(default=90, description="Inspection interval")
    cleaning_interval_days: int = Field(default=60, description="Cleaning interval")
    critical_response_hours: int = Field(default=24, description="Critical response time")
    high_response_days: int = Field(default=7, description="High priority response time")
    enable_predictive: bool = Field(default=True, description="Enable predictive maintenance")


class MaintenanceScheduler:
    """
    Intelligent maintenance scheduling system for PV installations.

    This class provides comprehensive maintenance planning including preventive
    maintenance scheduling, corrective action tracking, and spare parts management.

    Attributes:
        policy: Maintenance scheduling policy
        spare_parts_inventory: Inventory of spare parts
        corrective_actions: Tracked corrective actions
    """

    def __init__(self, policy: Optional[MaintenancePolicy] = None):
        """
        Initialize the MaintenanceScheduler.

        Args:
            policy: Optional maintenance policy configuration
        """
        self.policy = policy or MaintenancePolicy()
        self.spare_parts_inventory: Dict[str, SparePart] = {}
        self.corrective_actions: Dict[str, CorrectiveAction] = {}

    def preventive_maintenance_planning(
        self,
        site_id: str,
        start_date: Optional[date] = None,
        planning_horizon_days: int = 365,
        panel_count: int = 100,
        last_maintenance_date: Optional[date] = None,
    ) -> List[MaintenanceSchedule]:
        """
        Generate a comprehensive preventive maintenance plan.

        Creates a schedule of preventive maintenance activities based on policy
        configuration, site characteristics, and historical maintenance data.

        Args:
            site_id: Identifier of the site
            start_date: Start date for planning (defaults to today)
            planning_horizon_days: Number of days to plan ahead
            panel_count: Number of panels at the site
            last_maintenance_date: Date of last maintenance (if known)

        Returns:
            List of MaintenanceSchedule objects

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> schedules = scheduler.preventive_maintenance_planning(
            ...     site_id="SITE-001",
            ...     panel_count=500,
            ...     planning_horizon_days=365
            ... )
            >>> print(f"Generated {len(schedules)} maintenance schedules")
        """
        if start_date is None:
            start_date = date.today()

        schedules: List[MaintenanceSchedule] = []
        end_date = start_date + timedelta(days=planning_horizon_days)

        # Schedule periodic inspections
        inspection_schedules = self._schedule_periodic_inspections(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            last_maintenance_date=last_maintenance_date,
        )
        schedules.extend(inspection_schedules)

        # Schedule preventive maintenance
        preventive_schedules = self._schedule_preventive_maintenance(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            last_maintenance_date=last_maintenance_date,
        )
        schedules.extend(preventive_schedules)

        # Schedule cleaning
        cleaning_schedules = self._schedule_cleaning(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            panel_count=panel_count,
        )
        schedules.extend(cleaning_schedules)

        # Schedule seasonal maintenance
        seasonal_schedules = self._schedule_seasonal_maintenance(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
        )
        schedules.extend(seasonal_schedules)

        return schedules

    def corrective_action_tracking(
        self,
        diagnostics: List[DiagnosticResult],
        site_id: str,
    ) -> List[CorrectiveAction]:
        """
        Track and manage corrective actions for identified defects.

        Creates corrective action records for defects requiring remediation
        and tracks their status through completion.

        Args:
            diagnostics: List of diagnostic results
            site_id: Site identifier

        Returns:
            List of CorrectiveAction objects

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> actions = scheduler.corrective_action_tracking(
            ...     diagnostics=diagnostic_results,
            ...     site_id="SITE-001"
            ... )
            >>> print(f"Created {len(actions)} corrective actions")
        """
        actions: List[CorrectiveAction] = []

        for diagnostic in diagnostics:
            # Only create actions for defects requiring remediation
            if diagnostic.priority <= 3:  # High priority or above
                action = self._create_corrective_action(diagnostic, site_id)
                actions.append(action)
                self.corrective_actions[action.action_id] = action

        return actions

    def spare_parts_management(
        self,
        maintenance_schedules: List[MaintenanceSchedule],
        work_orders: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Manage spare parts inventory and requirements.

        Analyzes maintenance schedules and work orders to determine spare parts
        requirements, check inventory levels, and identify reorder needs.

        Args:
            maintenance_schedules: List of maintenance schedules
            work_orders: Optional list of work orders

        Returns:
            Dictionary with spare parts analysis including:
            - required: Parts needed for scheduled work
            - available: Parts currently in stock
            - shortages: Parts that need to be ordered
            - reorder: Parts below reorder level

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> analysis = scheduler.spare_parts_management(schedules)
            >>> if analysis['shortages']:
            ...     print("Need to order parts:", analysis['shortages'])
        """
        required_parts: Dict[str, int] = defaultdict(int)
        available_parts: Dict[str, int] = {}
        shortages: Dict[str, int] = {}
        reorder_needed: Dict[str, int] = {}

        # Analyze maintenance schedules for required parts
        for schedule in maintenance_schedules:
            for part_number in schedule.required_parts:
                required_parts[part_number] += 1

        # Analyze work orders if provided
        if work_orders:
            for wo in work_orders:
                if "required_parts" in wo:
                    for part_number, quantity in wo["required_parts"].items():
                        required_parts[part_number] += quantity

        # Check inventory levels
        for part_number in required_parts:
            if part_number in self.spare_parts_inventory:
                part = self.spare_parts_inventory[part_number]
                available_qty = part.quantity_available - part.quantity_reserved
                available_parts[part_number] = available_qty

                # Check for shortages
                required_qty = required_parts[part_number]
                if available_qty < required_qty:
                    shortages[part_number] = required_qty - available_qty

                # Check reorder level
                if part.quantity_available <= part.reorder_level:
                    reorder_needed[part_number] = part.reorder_level * 2 - part.quantity_available
            else:
                # Part not in inventory - full quantity needed
                shortages[part_number] = required_parts[part_number]

        return {
            "required": dict(required_parts),
            "available": available_parts,
            "shortages": shortages,
            "reorder": reorder_needed,
        }

    def add_spare_part(self, spare_part: SparePart) -> None:
        """
        Add or update a spare part in the inventory.

        Args:
            spare_part: SparePart object to add or update
        """
        self.spare_parts_inventory[spare_part.part_number] = spare_part

    def update_corrective_action_status(
        self,
        action_id: str,
        status: str,
        effectiveness_score: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[CorrectiveAction]:
        """
        Update the status of a corrective action.

        Args:
            action_id: Action identifier
            status: New status
            effectiveness_score: Effectiveness rating (0-100)
            notes: Additional notes

        Returns:
            Updated CorrectiveAction or None if not found
        """
        if action_id in self.corrective_actions:
            action = self.corrective_actions[action_id]
            action.status = status
            if effectiveness_score is not None:
                action.effectiveness_score = effectiveness_score
            if notes:
                action.notes = notes
            return action
        return None

    def _schedule_periodic_inspections(
        self,
        site_id: str,
        start_date: date,
        end_date: date,
        last_maintenance_date: Optional[date],
    ) -> List[MaintenanceSchedule]:
        """Schedule periodic inspections."""
        schedules = []
        current_date = start_date

        # Adjust start date based on last maintenance
        if last_maintenance_date:
            days_since_last = (start_date - last_maintenance_date).days
            if days_since_last < self.policy.inspection_interval_days:
                current_date = last_maintenance_date + timedelta(
                    days=self.policy.inspection_interval_days
                )

        while current_date <= end_date:
            schedule = MaintenanceSchedule(
                schedule_name=f"Periodic Inspection - {current_date}",
                site_id=site_id,
                maintenance_type=MaintenanceType.INSPECTION,
                priority=MaintenancePriority.MEDIUM,
                scheduled_date=current_date,
                estimated_duration_hours=4.0,
                required_skills=["electrical_inspection", "visual_inspection"],
                description="Comprehensive periodic inspection of PV system",
                recurrence_pattern=f"every_{self.policy.inspection_interval_days}_days",
                next_due=current_date,
            )
            schedules.append(schedule)
            current_date += timedelta(days=self.policy.inspection_interval_days)

        return schedules

    def _schedule_preventive_maintenance(
        self,
        site_id: str,
        start_date: date,
        end_date: date,
        last_maintenance_date: Optional[date],
    ) -> List[MaintenanceSchedule]:
        """Schedule preventive maintenance."""
        schedules = []
        current_date = start_date

        if last_maintenance_date:
            days_since_last = (start_date - last_maintenance_date).days
            if days_since_last < self.policy.preventive_interval_days:
                current_date = last_maintenance_date + timedelta(
                    days=self.policy.preventive_interval_days
                )

        while current_date <= end_date:
            schedule = MaintenanceSchedule(
                schedule_name=f"Preventive Maintenance - {current_date}",
                site_id=site_id,
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.HIGH,
                scheduled_date=current_date,
                estimated_duration_hours=8.0,
                required_parts=["connectors", "junction_boxes", "cables"],
                required_skills=[
                    "electrical_maintenance",
                    "mechanical_maintenance",
                    "safety_certified",
                ],
                description="Comprehensive preventive maintenance including electrical "
                "checks, mechanical inspections, and performance testing",
                recurrence_pattern=f"every_{self.policy.preventive_interval_days}_days",
                next_due=current_date,
            )
            schedules.append(schedule)
            current_date += timedelta(days=self.policy.preventive_interval_days)

        return schedules

    def _schedule_cleaning(
        self,
        site_id: str,
        start_date: date,
        end_date: date,
        panel_count: int,
    ) -> List[MaintenanceSchedule]:
        """Schedule cleaning operations."""
        schedules = []
        current_date = start_date + timedelta(days=self.policy.cleaning_interval_days)

        # Estimate cleaning duration based on panel count
        hours_per_panel = 0.1  # 6 minutes per panel
        estimated_hours = max(panel_count * hours_per_panel, 2.0)

        while current_date <= end_date:
            schedule = MaintenanceSchedule(
                schedule_name=f"Panel Cleaning - {current_date}",
                site_id=site_id,
                maintenance_type=MaintenanceType.ROUTINE,
                priority=MaintenancePriority.MEDIUM,
                scheduled_date=current_date,
                estimated_duration_hours=estimated_hours,
                required_skills=["cleaning", "safety_certified"],
                description=f"Cleaning of {panel_count} solar panels",
                recurrence_pattern=f"every_{self.policy.cleaning_interval_days}_days",
                next_due=current_date,
            )
            schedules.append(schedule)
            current_date += timedelta(days=self.policy.cleaning_interval_days)

        return schedules

    def _schedule_seasonal_maintenance(
        self,
        site_id: str,
        start_date: date,
        end_date: date,
    ) -> List[MaintenanceSchedule]:
        """Schedule seasonal maintenance activities."""
        schedules = []

        # Find next occurrence of each season within the planning period
        seasons = {
            "spring": (3, "Spring Preparation - Check for winter damage, clean panels"),
            "summer": (6, "Summer Check - Verify cooling, check for overheating"),
            "fall": (9, "Fall Preparation - Prepare for winter, check weatherproofing"),
            "winter": (12, "Winter Check - Remove snow, check for ice damage"),
        }

        current_year = start_date.year
        end_year = end_date.year

        for year in range(current_year, end_year + 1):
            for season, (month, description) in seasons.items():
                # Schedule for the 15th of the season month
                scheduled_date = date(year, month, 15)

                if start_date <= scheduled_date <= end_date:
                    schedule = MaintenanceSchedule(
                        schedule_name=f"{season.capitalize()} Seasonal Maintenance",
                        site_id=site_id,
                        maintenance_type=MaintenanceType.PREVENTIVE,
                        priority=MaintenancePriority.MEDIUM,
                        scheduled_date=scheduled_date,
                        estimated_duration_hours=4.0,
                        required_skills=["general_maintenance", "safety_certified"],
                        description=description,
                        recurrence_pattern="seasonal",
                        next_due=scheduled_date,
                    )
                    schedules.append(schedule)

        return schedules

    def _create_corrective_action(
        self,
        diagnostic: DiagnosticResult,
        site_id: str,
    ) -> CorrectiveAction:
        """Create a corrective action from a diagnostic result."""
        action_type = self._determine_action_type(diagnostic)

        # Determine follow-up requirements
        follow_up_required = diagnostic.priority <= 2  # Critical or high priority
        follow_up_date = None
        if follow_up_required:
            follow_up_date = date.today() + timedelta(days=30)

        action = CorrectiveAction(
            defect_id=diagnostic.defect_id,
            action_type=action_type,
            description=f"{diagnostic.recommended_action.value}: {diagnostic.root_cause}",
            status="planned",
            follow_up_required=follow_up_required,
            follow_up_date=follow_up_date,
            notes=diagnostic.analysis_notes,
        )

        return action

    def _determine_action_type(self, diagnostic: DiagnosticResult) -> str:
        """Determine the action type from diagnostic result."""
        action_map = {
            "immediate_repair": "emergency_repair",
            "replace_panel": "panel_replacement",
            "replace_module": "module_replacement",
            "schedule_inspection": "inspection",
            "clean": "cleaning",
            "adjust": "adjustment",
            "monitor": "monitoring",
        }
        return action_map.get(
            diagnostic.recommended_action.value,
            "corrective_maintenance"
        )
