"""
Repair Optimization & Maintenance Strategies (B11-S03)

This module provides tools for optimizing repair decisions, prioritizing defects,
and planning preventive maintenance for PV modules and systems.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class DefectSeverity(str, Enum):
    """Severity levels for module defects."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class RepairAction(str, Enum):
    """Types of repair actions."""
    REPLACE_JUNCTION_BOX = "replace_junction_box"
    REPLACE_BYPASS_DIODE = "replace_bypass_diode"
    REFRAME = "reframe"
    CLEAN = "clean"
    REWIRE = "rewire"
    RELAMINATE = "relaminate"
    REPLACE_MODULE = "replace_module"
    NO_ACTION = "no_action"


class Defect(BaseModel):
    """Individual defect in a PV module."""

    defect_id: str = Field(description="Unique defect identifier")
    defect_type: str = Field(description="Type of defect")
    severity: DefectSeverity = Field(description="Severity level")
    location: str = Field(description="Location in module/system")
    power_loss_w: float = Field(ge=0, description="Estimated power loss in watts")
    safety_risk: bool = Field(default=False, description="Whether defect poses safety risk")
    progression_rate: float = Field(
        default=0.0,
        description="Rate of defect progression (power loss increase per year)"
    )
    detected_date: Optional[str] = Field(default=None, description="Detection date")


class RepairCost(BaseModel):
    """Cost structure for repair operations."""

    labor_cost: float = Field(ge=0, description="Labor cost in USD")
    parts_cost: float = Field(ge=0, description="Parts/materials cost in USD")
    equipment_cost: float = Field(ge=0, description="Equipment/tools cost in USD")
    downtime_cost: float = Field(ge=0, description="Cost of system downtime in USD")
    transportation_cost: float = Field(default=0, ge=0, description="Transportation cost in USD")

    @property
    def total_cost(self) -> float:
        """Total repair cost."""
        return (
            self.labor_cost + self.parts_cost + self.equipment_cost +
            self.downtime_cost + self.transportation_cost
        )


class RepairDecision(BaseModel):
    """Decision analysis for repair vs replace."""

    decision: str = Field(description="Recommended decision: 'repair' or 'replace'")
    confidence: float = Field(ge=0, le=1, description="Confidence in decision (0-1)")
    repair_cost: float = Field(ge=0, description="Total repair cost in USD")
    replacement_cost: float = Field(ge=0, description="Total replacement cost in USD")
    repair_npv: float = Field(description="Net present value of repair option")
    replace_npv: float = Field(description="Net present value of replacement option")
    payback_period_years: Optional[float] = Field(
        default=None,
        description="Payback period for repair investment"
    )
    reasoning: List[str] = Field(default_factory=list, description="Decision reasoning")


class MaintenanceSchedule(BaseModel):
    """Preventive maintenance schedule."""

    activity: str = Field(description="Maintenance activity")
    frequency_months: int = Field(ge=1, description="Frequency in months")
    estimated_cost: float = Field(ge=0, description="Estimated cost per occurrence")
    duration_hours: float = Field(ge=0, description="Duration in hours")
    priority: str = Field(description="Priority level: 'critical', 'high', 'medium', 'low'")
    next_due_date: Optional[str] = Field(default=None, description="Next scheduled date")


class PrioritizedDefect(BaseModel):
    """Defect with priority scoring."""

    defect: Defect
    priority_score: float = Field(ge=0, description="Priority score (higher = more urgent)")
    recommended_action: RepairAction
    estimated_repair_cost: float = Field(ge=0, description="Estimated repair cost")
    urgency_days: int = Field(ge=0, description="Recommended repair timeline in days")


class RepairOptimizer:
    """
    Optimizer for PV module repair decisions and maintenance planning.

    Provides methods for defect prioritization, repair vs replace analysis,
    and preventive maintenance scheduling.
    """

    def __init__(
        self,
        electricity_price_per_kwh: float = 0.12,
        discount_rate: float = 0.05,
        labor_rate_per_hour: float = 75.0,
        new_module_cost_per_watt: float = 0.40
    ):
        """
        Initialize the repair optimizer.

        Args:
            electricity_price_per_kwh: Electricity price in USD/kWh
            discount_rate: Annual discount rate for NPV calculations
            labor_rate_per_hour: Labor rate in USD/hour
            new_module_cost_per_watt: Cost of new module in USD/watt
        """
        self.electricity_price = electricity_price_per_kwh
        self.discount_rate = discount_rate
        self.labor_rate = labor_rate_per_hour
        self.new_module_cost_per_watt = new_module_cost_per_watt

    def defect_prioritization(
        self,
        defects: List[Defect],
        system_size_kw: float,
        age_years: float
    ) -> List[PrioritizedDefect]:
        """
        Prioritize defects for repair based on impact and urgency.

        Args:
            defects: List of detected defects
            system_size_kw: Total system size in kW
            age_years: System age in years

        Returns:
            List of PrioritizedDefect sorted by priority (highest first)
        """
        prioritized = []

        for defect in defects:
            # Calculate priority score based on multiple factors
            priority_score = self._calculate_priority_score(defect, system_size_kw, age_years)

            # Determine recommended action
            action = self._recommend_repair_action(defect)

            # Estimate repair cost
            repair_cost = self._estimate_repair_cost(defect, action)

            # Determine urgency (days until repair needed)
            urgency = self._calculate_urgency_days(defect)

            prioritized.append(PrioritizedDefect(
                defect=defect,
                priority_score=priority_score,
                recommended_action=action,
                estimated_repair_cost=repair_cost,
                urgency_days=urgency
            ))

        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x.priority_score, reverse=True)

        return prioritized

    def repair_vs_replace(
        self,
        module_age_years: float,
        current_power_w: float,
        rated_power_w: float,
        repair_costs: RepairCost,
        defects: List[Defect],
        expected_life_years: float = 25.0
    ) -> RepairDecision:
        """
        Analyze whether to repair or replace a module based on economics.

        Args:
            module_age_years: Current age of module in years
            current_power_w: Current power output in watts
            rated_power_w: Original rated power in watts
            repair_costs: Detailed repair costs
            defects: List of defects to be repaired
            expected_life_years: Expected total life of new module

        Returns:
            RepairDecision with detailed economic analysis
        """
        reasoning = []

        # Calculate remaining useful life after repair
        remaining_life_after_repair = max(0, expected_life_years - module_age_years)

        # Estimate power after repair
        total_power_loss = sum(d.power_loss_w for d in defects)
        power_after_repair = min(current_power_w + total_power_loss, rated_power_w)

        # Calculate annual energy production (assuming 1500 kWh/kW/year)
        annual_energy_after_repair = power_after_repair * 1.5  # kWh/year

        # Repair option NPV
        repair_benefit = self._calculate_npv(
            annual_benefit=annual_energy_after_repair * self.electricity_price,
            years=remaining_life_after_repair,
            discount_rate=self.discount_rate
        )
        repair_npv = repair_benefit - repair_costs.total_cost

        # Replacement option
        replacement_cost = rated_power_w * self.new_module_cost_per_watt
        new_annual_energy = rated_power_w * 1.5  # kWh/year
        replacement_benefit = self._calculate_npv(
            annual_benefit=new_annual_energy * self.electricity_price,
            years=expected_life_years,
            discount_rate=self.discount_rate
        )
        replace_npv = replacement_benefit - replacement_cost

        # Make decision
        if repair_npv > replace_npv:
            decision = "repair"
            reasoning.append(f"Repair NPV (${repair_npv:.2f}) exceeds replacement NPV (${replace_npv:.2f})")
            confidence = min(0.95, (repair_npv - replace_npv) / replacement_cost)
        else:
            decision = "replace"
            reasoning.append(f"Replacement NPV (${replace_npv:.2f}) exceeds repair NPV (${repair_npv:.2f})")
            confidence = min(0.95, (replace_npv - repair_npv) / replacement_cost)

        # Additional decision factors
        if module_age_years > 0.8 * expected_life_years:
            reasoning.append(f"Module age ({module_age_years:.1f} years) is near end of expected life")
            if decision == "replace":
                confidence += 0.1

        if any(d.safety_risk for d in defects):
            reasoning.append("Safety risks detected - replacement may be preferred")
            if decision == "replace":
                confidence += 0.15

        if repair_costs.total_cost > 0.5 * replacement_cost:
            reasoning.append(f"Repair cost (${repair_costs.total_cost:.2f}) is >50% of replacement cost")

        # Calculate payback period for repair
        if decision == "repair":
            annual_savings = (power_after_repair - current_power_w) * 1.5 * self.electricity_price
            payback_period = repair_costs.total_cost / annual_savings if annual_savings > 0 else None
        else:
            payback_period = None

        confidence = max(0, min(1, confidence))

        return RepairDecision(
            decision=decision,
            confidence=confidence,
            repair_cost=repair_costs.total_cost,
            replacement_cost=replacement_cost,
            repair_npv=repair_npv,
            replace_npv=replace_npv,
            payback_period_years=payback_period,
            reasoning=reasoning
        )

    def preventive_maintenance(
        self,
        system_size_kw: float,
        system_age_years: float,
        climate_zone: str = "temperate",
        system_type: str = "ground_mount"
    ) -> List[MaintenanceSchedule]:
        """
        Generate preventive maintenance schedule.

        Args:
            system_size_kw: System size in kW
            system_age_years: Current system age in years
            climate_zone: Climate zone ('desert', 'temperate', 'tropical', 'cold')
            system_type: System type ('ground_mount', 'rooftop', 'carport', 'floating')

        Returns:
            List of MaintenanceSchedule items
        """
        schedule = []

        # Visual inspection
        inspection_frequency = 6 if climate_zone in ["desert", "tropical"] else 12
        schedule.append(MaintenanceSchedule(
            activity="Visual inspection of modules and racking",
            frequency_months=inspection_frequency,
            estimated_cost=system_size_kw * 2.0,
            duration_hours=system_size_kw * 0.1,
            priority="high"
        ))

        # Cleaning
        cleaning_frequency = self._determine_cleaning_frequency(climate_zone, system_type)
        schedule.append(MaintenanceSchedule(
            activity="Module cleaning",
            frequency_months=cleaning_frequency,
            estimated_cost=system_size_kw * 5.0,
            duration_hours=system_size_kw * 0.2,
            priority="medium"
        ))

        # Electrical testing
        schedule.append(MaintenanceSchedule(
            activity="Electrical performance testing",
            frequency_months=12,
            estimated_cost=system_size_kw * 3.0,
            duration_hours=system_size_kw * 0.15,
            priority="high"
        ))

        # Infrared thermography
        schedule.append(MaintenanceSchedule(
            activity="Infrared thermography scan",
            frequency_months=24,
            estimated_cost=system_size_kw * 4.0,
            duration_hours=system_size_kw * 0.12,
            priority="medium"
        ))

        # Inverter maintenance
        schedule.append(MaintenanceSchedule(
            activity="Inverter inspection and filter cleaning",
            frequency_months=6,
            estimated_cost=100.0 + system_size_kw * 1.0,
            duration_hours=2.0,
            priority="high"
        ))

        # Connection tightness check
        schedule.append(MaintenanceSchedule(
            activity="Electrical connection tightness check",
            frequency_months=12,
            estimated_cost=system_size_kw * 2.5,
            duration_hours=system_size_kw * 0.1,
            priority="high"
        ))

        # Vegetation management (for ground-mount)
        if system_type == "ground_mount":
            schedule.append(MaintenanceSchedule(
                activity="Vegetation control and site maintenance",
                frequency_months=3 if climate_zone == "tropical" else 6,
                estimated_cost=system_size_kw * 1.5,
                duration_hours=system_size_kw * 0.05,
                priority="medium"
            ))

        # Grounding system check
        schedule.append(MaintenanceSchedule(
            activity="Grounding system integrity check",
            frequency_months=24,
            estimated_cost=system_size_kw * 1.0,
            duration_hours=system_size_kw * 0.08,
            priority="high"
        ))

        # Age-based additional maintenance
        if system_age_years > 10:
            schedule.append(MaintenanceSchedule(
                activity="Comprehensive system assessment (aging system)",
                frequency_months=12,
                estimated_cost=system_size_kw * 8.0,
                duration_hours=system_size_kw * 0.3,
                priority="critical"
            ))

        return schedule

    def _calculate_priority_score(
        self,
        defect: Defect,
        system_size_kw: float,
        age_years: float
    ) -> float:
        """Calculate priority score for a defect."""
        score = 0.0

        # Severity weight
        severity_weights = {
            DefectSeverity.CRITICAL: 100,
            DefectSeverity.HIGH: 75,
            DefectSeverity.MEDIUM: 50,
            DefectSeverity.LOW: 25,
            DefectSeverity.NEGLIGIBLE: 10
        }
        score += severity_weights.get(defect.severity, 50)

        # Safety risk
        if defect.safety_risk:
            score += 150

        # Power loss impact (relative to system size)
        power_loss_ratio = defect.power_loss_w / (system_size_kw * 1000)
        score += power_loss_ratio * 100

        # Progression rate
        score += defect.progression_rate * 10

        # Age factor (older systems get lower priority for minor defects)
        if age_years > 20 and defect.severity in [DefectSeverity.LOW, DefectSeverity.NEGLIGIBLE]:
            score *= 0.5

        return score

    def _recommend_repair_action(self, defect: Defect) -> RepairAction:
        """Recommend repair action for a defect."""
        defect_type_lower = defect.defect_type.lower()

        if "junction" in defect_type_lower or "j-box" in defect_type_lower:
            return RepairAction.REPLACE_JUNCTION_BOX
        elif "diode" in defect_type_lower or "bypass" in defect_type_lower:
            return RepairAction.REPLACE_BYPASS_DIODE
        elif "frame" in defect_type_lower or "mounting" in defect_type_lower:
            return RepairAction.REFRAME
        elif "soiling" in defect_type_lower or "dirt" in defect_type_lower:
            return RepairAction.CLEAN
        elif "wiring" in defect_type_lower or "cable" in defect_type_lower:
            return RepairAction.REWIRE
        elif "delamination" in defect_type_lower or "encapsulant" in defect_type_lower:
            if defect.severity in [DefectSeverity.CRITICAL, DefectSeverity.HIGH]:
                return RepairAction.REPLACE_MODULE
            return RepairAction.RELAMINATE
        elif defect.severity == DefectSeverity.CRITICAL or defect.safety_risk:
            return RepairAction.REPLACE_MODULE
        else:
            return RepairAction.NO_ACTION

    def _estimate_repair_cost(self, defect: Defect, action: RepairAction) -> float:
        """Estimate cost of repair action."""
        action_costs = {
            RepairAction.REPLACE_JUNCTION_BOX: 50.0 + self.labor_rate * 0.5,
            RepairAction.REPLACE_BYPASS_DIODE: 30.0 + self.labor_rate * 0.75,
            RepairAction.REFRAME: 80.0 + self.labor_rate * 1.5,
            RepairAction.CLEAN: 10.0 + self.labor_rate * 0.25,
            RepairAction.REWIRE: 60.0 + self.labor_rate * 1.0,
            RepairAction.RELAMINATE: 200.0 + self.labor_rate * 3.0,
            RepairAction.REPLACE_MODULE: 300.0 + self.labor_rate * 2.0,
            RepairAction.NO_ACTION: 0.0
        }

        return action_costs.get(action, 100.0)

    def _calculate_urgency_days(self, defect: Defect) -> int:
        """Calculate urgency in days for defect repair."""
        if defect.safety_risk:
            return 1  # Immediate

        severity_urgency = {
            DefectSeverity.CRITICAL: 7,
            DefectSeverity.HIGH: 30,
            DefectSeverity.MEDIUM: 90,
            DefectSeverity.LOW: 180,
            DefectSeverity.NEGLIGIBLE: 365
        }

        base_urgency = severity_urgency.get(defect.severity, 90)

        # Adjust for progression rate
        if defect.progression_rate > 10:  # >10W loss per year
            base_urgency = int(base_urgency * 0.5)

        return base_urgency

    @staticmethod
    def _calculate_npv(
        annual_benefit: float,
        years: float,
        discount_rate: float
    ) -> float:
        """Calculate net present value of annual benefits."""
        if years <= 0:
            return 0.0

        if discount_rate == 0:
            return annual_benefit * years

        # NPV formula for annuity
        npv = annual_benefit * ((1 - (1 + discount_rate) ** -years) / discount_rate)
        return npv

    @staticmethod
    def _determine_cleaning_frequency(climate_zone: str, system_type: str) -> int:
        """Determine optimal cleaning frequency based on climate and system type."""
        base_frequency = {
            "desert": 2,
            "tropical": 3,
            "temperate": 6,
            "cold": 12
        }

        frequency = base_frequency.get(climate_zone, 6)

        # Adjust for system type
        if system_type == "ground_mount":
            frequency = max(1, int(frequency * 0.75))  # More frequent for ground mount

        return frequency
