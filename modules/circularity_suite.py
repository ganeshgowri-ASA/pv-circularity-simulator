"""
Circularity Suite Module (B10-B12)
===================================
Integrates:
- B10: Revamp & Repower Planning
- B11: Circularity 3R Assessment (Reduce, Reuse, Recycle)
- B12: Hybrid Energy Storage Integration

This module provides comprehensive circular economy capabilities including
lifecycle extension, end-of-life management, and energy storage integration.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator


# ============================================================================
# B10: REVAMP & REPOWER PLANNING
# ============================================================================

class RevampStrategy(str, Enum):
    """Revamp/repower strategy types."""
    PARTIAL_REPOWER = "Partial Repower"
    FULL_REPOWER = "Full Repower"
    MODULE_UPGRADE = "Module Upgrade"
    INVERTER_UPGRADE = "Inverter Upgrade"
    BOS_UPGRADE = "Balance-of-System Upgrade"
    CAPACITY_EXPANSION = "Capacity Expansion"
    HYBRID_CONVERSION = "Hybrid Conversion"


class SystemCondition(str, Enum):
    """Current system condition assessment."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    CRITICAL = "Critical"


class RevampAssessment(BaseModel):
    """Revamp/repower assessment results."""

    assessment_id: str = Field(..., description="Assessment identifier")
    assessed_at: datetime = Field(..., description="Assessment timestamp")
    system_age_years: float = Field(..., ge=0, description="System age (years)")
    current_capacity_kw: float = Field(..., ge=0, description="Current capacity (kW)")
    current_condition: SystemCondition = Field(..., description="System condition")
    remaining_useful_life_years: float = Field(..., ge=0, description="Estimated RUL (years)")
    current_performance_ratio: float = Field(..., ge=0, le=100, description="Current PR (%)")
    degradation_rate_per_year: float = Field(..., ge=0, description="Annual degradation (%/yr)")
    recommended_strategy: RevampStrategy = Field(..., description="Recommended strategy")
    estimated_cost_usd: float = Field(..., ge=0, description="Estimated revamp cost ($)")
    expected_capacity_gain_kw: float = Field(default=0.0, ge=0, description="Expected capacity gain (kW)")
    payback_period_years: float = Field(..., ge=0, description="Payback period (years)")
    roi_percentage: float = Field(..., description="Return on investment (%)")
    co2_savings_tons_year: float = Field(default=0.0, ge=0, description="Annual CO2 savings (tons/year)")

    class Config:
        use_enum_values = True


class RevampPlanner:
    """
    PV System Revamp & Repower Planning Engine.
    Analyzes aging systems and recommends optimal upgrade strategies.
    """

    def __init__(self):
        """Initialize revamp planner."""
        self.assessments: List[RevampAssessment] = []

    def assess_system(
        self,
        system_age_years: float,
        current_capacity_kw: float,
        current_pr: float,
        annual_degradation: float,
        electricity_price_kwh: float = 0.12
    ) -> RevampAssessment:
        """
        Assess system for revamp/repower opportunities.

        Args:
            system_age_years: System age
            current_capacity_kw: Current system capacity
            current_pr: Current performance ratio
            annual_degradation: Annual degradation rate
            electricity_price_kwh: Electricity price for financial analysis

        Returns:
            Comprehensive revamp assessment
        """
        # Assess system condition
        condition = self._assess_condition(system_age_years, current_pr)

        # Calculate remaining useful life
        rul = self._calculate_rul(system_age_years, annual_degradation)

        # Determine optimal strategy
        strategy = self._recommend_strategy(system_age_years, condition, current_pr)

        # Estimate costs and benefits
        costs = self._estimate_revamp_costs(strategy, current_capacity_kw)
        capacity_gain = self._estimate_capacity_gain(strategy, current_capacity_kw)

        # Financial analysis
        annual_energy_gain = capacity_gain * 1500  # kWh (assuming 1500 FLH)
        annual_revenue_gain = annual_energy_gain * electricity_price_kwh
        payback_period = costs / annual_revenue_gain if annual_revenue_gain > 0 else 999
        roi = (annual_revenue_gain * 20 - costs) / costs * 100 if costs > 0 else 0

        # Environmental impact
        co2_savings = annual_energy_gain * 0.5 / 1000  # tons CO2/year (0.5 kg/kWh)

        assessment = RevampAssessment(
            assessment_id=f"REVAMP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assessed_at=datetime.now(),
            system_age_years=system_age_years,
            current_capacity_kw=current_capacity_kw,
            current_condition=condition,
            remaining_useful_life_years=rul,
            current_performance_ratio=current_pr,
            degradation_rate_per_year=annual_degradation,
            recommended_strategy=strategy,
            estimated_cost_usd=costs,
            expected_capacity_gain_kw=capacity_gain,
            payback_period_years=payback_period,
            roi_percentage=roi,
            co2_savings_tons_year=co2_savings
        )

        self.assessments.append(assessment)
        return assessment

    def _assess_condition(self, age: float, pr: float) -> SystemCondition:
        """Assess system condition based on age and performance."""
        if pr > 85 and age < 5:
            return SystemCondition.EXCELLENT
        elif pr > 80 and age < 10:
            return SystemCondition.GOOD
        elif pr > 75 and age < 15:
            return SystemCondition.FAIR
        elif pr > 70:
            return SystemCondition.POOR
        else:
            return SystemCondition.CRITICAL

    def _calculate_rul(self, age: float, degradation: float) -> float:
        """Calculate remaining useful life."""
        # Assume 80% performance threshold for end-of-life
        current_performance = 100 - (age * degradation)
        if current_performance <= 80:
            return 0
        remaining_degradation = current_performance - 80
        return remaining_degradation / degradation if degradation > 0 else 25 - age

    def _recommend_strategy(self, age: float, condition: SystemCondition, pr: float) -> RevampStrategy:
        """Recommend optimal revamp strategy."""
        if age > 20 or condition == SystemCondition.CRITICAL:
            return RevampStrategy.FULL_REPOWER
        elif age > 15 or pr < 75:
            return RevampStrategy.PARTIAL_REPOWER
        elif age > 10:
            return RevampStrategy.MODULE_UPGRADE
        elif pr < 85:
            return RevampStrategy.INVERTER_UPGRADE
        else:
            return RevampStrategy.BOS_UPGRADE

    def _estimate_revamp_costs(self, strategy: RevampStrategy, capacity_kw: float) -> float:
        """Estimate revamp costs based on strategy."""
        cost_per_kw = {
            RevampStrategy.FULL_REPOWER: 1000,
            RevampStrategy.PARTIAL_REPOWER: 600,
            RevampStrategy.MODULE_UPGRADE: 400,
            RevampStrategy.INVERTER_UPGRADE: 200,
            RevampStrategy.BOS_UPGRADE: 150,
            RevampStrategy.CAPACITY_EXPANSION: 800,
            RevampStrategy.HYBRID_CONVERSION: 1200
        }
        return cost_per_kw.get(strategy, 500) * capacity_kw

    def _estimate_capacity_gain(self, strategy: RevampStrategy, current_capacity: float) -> float:
        """Estimate capacity gain from revamp."""
        gain_percentage = {
            RevampStrategy.FULL_REPOWER: 0.30,
            RevampStrategy.PARTIAL_REPOWER: 0.20,
            RevampStrategy.MODULE_UPGRADE: 0.15,
            RevampStrategy.INVERTER_UPGRADE: 0.05,
            RevampStrategy.BOS_UPGRADE: 0.02,
            RevampStrategy.CAPACITY_EXPANSION: 0.50,
            RevampStrategy.HYBRID_CONVERSION: 0.40
        }
        return current_capacity * gain_percentage.get(strategy, 0.10)

    def compare_strategies(
        self,
        system_age_years: float,
        current_capacity_kw: float,
        current_pr: float,
        annual_degradation: float
    ) -> List[RevampAssessment]:
        """
        Compare multiple revamp strategies.

        Returns:
            List of assessments for different strategies
        """
        strategies = [
            RevampStrategy.FULL_REPOWER,
            RevampStrategy.PARTIAL_REPOWER,
            RevampStrategy.MODULE_UPGRADE,
            RevampStrategy.INVERTER_UPGRADE
        ]

        assessments = []
        for strategy in strategies:
            # Temporarily override strategy recommendation
            assessment = self.assess_system(
                system_age_years, current_capacity_kw, current_pr, annual_degradation
            )
            # Update strategy for comparison
            original_strategy = assessment.recommended_strategy
            assessment.recommended_strategy = strategy
            assessment.estimated_cost_usd = self._estimate_revamp_costs(strategy, current_capacity_kw)
            assessment.expected_capacity_gain_kw = self._estimate_capacity_gain(strategy, current_capacity_kw)
            assessments.append(assessment)

        return assessments


# ============================================================================
# B11: CIRCULARITY 3R ASSESSMENT (REDUCE, REUSE, RECYCLE)
# ============================================================================

class CircularityPhase(str, Enum):
    """Circular economy phase."""
    REDUCE = "Reduce"
    REUSE = "Reuse"
    RECYCLE = "Recycle"
    REFURBISH = "Refurbish"
    REMANUFACTURE = "Remanufacture"
    DISPOSE = "Dispose"


class ModuleEndOfLifeStatus(str, Enum):
    """Module end-of-life status."""
    FUNCTIONAL = "Functional"
    DEGRADED = "Degraded"
    DAMAGED = "Damaged"
    FAILED = "Failed"


class CircularityMetrics(BaseModel):
    """Circular economy metrics."""

    circularity_id: str = Field(..., description="Circularity assessment ID")
    assessed_at: datetime = Field(..., description="Assessment timestamp")
    module_id: str = Field(..., description="Module identifier")
    module_age_years: float = Field(..., ge=0, description="Module age (years)")
    current_efficiency: float = Field(..., ge=0, le=100, description="Current efficiency (%)")
    original_efficiency: float = Field(..., ge=0, le=100, description="Original efficiency (%)")
    eol_status: ModuleEndOfLifeStatus = Field(..., description="End-of-life status")

    # Reduce metrics
    lifetime_extension_years: float = Field(default=0.0, ge=0, description="Potential lifetime extension (years)")
    maintenance_cost_usd: float = Field(default=0.0, ge=0, description="Maintenance cost ($)")

    # Reuse metrics
    reuse_potential: float = Field(..., ge=0, le=100, description="Reuse potential score (0-100)")
    reuse_application: str = Field(default="", description="Recommended reuse application")
    reuse_market_value_usd: float = Field(default=0.0, ge=0, description="Reuse market value ($)")

    # Recycle metrics
    recyclable_materials_kg: Dict[str, float] = Field(default_factory=dict, description="Recyclable materials (kg)")
    recycling_efficiency: float = Field(..., ge=0, le=100, description="Recycling efficiency (%)")
    material_recovery_value_usd: float = Field(default=0.0, ge=0, description="Material recovery value ($)")

    # Overall circularity
    circularity_index: float = Field(..., ge=0, le=100, description="Overall circularity index (0-100)")
    recommended_phase: CircularityPhase = Field(..., description="Recommended circular phase")
    environmental_impact_score: float = Field(..., ge=0, le=100, description="Environmental impact score")

    class Config:
        use_enum_values = True


class CircularityAssessor:
    """
    Circular Economy 3R Assessment Engine.
    Evaluates reduce, reuse, and recycle opportunities for PV systems.
    """

    def __init__(self):
        """Initialize circularity assessor."""
        self.assessments: List[CircularityMetrics] = []

    def assess_module_circularity(
        self,
        module_id: str,
        module_age_years: float,
        current_efficiency: float,
        original_efficiency: float,
        physical_condition: str = "Good"
    ) -> CircularityMetrics:
        """
        Assess module circular economy potential.

        Args:
            module_id: Module identifier
            module_age_years: Module age
            current_efficiency: Current efficiency
            original_efficiency: Original efficiency
            physical_condition: Physical condition assessment

        Returns:
            Comprehensive circularity metrics
        """
        # Determine EOL status
        eol_status = self._determine_eol_status(current_efficiency, original_efficiency, physical_condition)

        # Calculate reduce potential (lifetime extension)
        lifetime_extension = self._calculate_lifetime_extension(eol_status, module_age_years)
        maintenance_cost = lifetime_extension * 50  # $50/year maintenance

        # Calculate reuse potential
        reuse_score = self._calculate_reuse_potential(current_efficiency, original_efficiency, eol_status)
        reuse_app = self._recommend_reuse_application(reuse_score)
        reuse_value = self._estimate_reuse_value(reuse_score, original_efficiency)

        # Calculate recycle potential
        recyclable_materials = self._calculate_recyclable_materials()
        recycling_eff = self._calculate_recycling_efficiency()
        recovery_value = self._estimate_recovery_value(recyclable_materials)

        # Calculate overall circularity index
        circularity_index = self._calculate_circularity_index(reuse_score, recycling_eff, lifetime_extension)

        # Recommend optimal phase
        recommended_phase = self._recommend_circular_phase(eol_status, reuse_score, module_age_years)

        # Environmental impact
        env_score = self._calculate_environmental_impact(recommended_phase, reuse_score)

        metrics = CircularityMetrics(
            circularity_id=f"CIRC_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assessed_at=datetime.now(),
            module_id=module_id,
            module_age_years=module_age_years,
            current_efficiency=current_efficiency,
            original_efficiency=original_efficiency,
            eol_status=eol_status,
            lifetime_extension_years=lifetime_extension,
            maintenance_cost_usd=maintenance_cost,
            reuse_potential=reuse_score,
            reuse_application=reuse_app,
            reuse_market_value_usd=reuse_value,
            recyclable_materials_kg=recyclable_materials,
            recycling_efficiency=recycling_eff,
            material_recovery_value_usd=recovery_value,
            circularity_index=circularity_index,
            recommended_phase=recommended_phase,
            environmental_impact_score=env_score
        )

        self.assessments.append(metrics)
        return metrics

    def _determine_eol_status(
        self,
        current_eff: float,
        original_eff: float,
        condition: str
    ) -> ModuleEndOfLifeStatus:
        """Determine module end-of-life status."""
        efficiency_ratio = current_eff / original_eff if original_eff > 0 else 0

        if efficiency_ratio > 0.90 and condition == "Good":
            return ModuleEndOfLifeStatus.FUNCTIONAL
        elif efficiency_ratio > 0.80:
            return ModuleEndOfLifeStatus.DEGRADED
        elif efficiency_ratio > 0.60:
            return ModuleEndOfLifeStatus.DAMAGED
        else:
            return ModuleEndOfLifeStatus.FAILED

    def _calculate_lifetime_extension(self, status: ModuleEndOfLifeStatus, age: float) -> float:
        """Calculate potential lifetime extension."""
        if status == ModuleEndOfLifeStatus.FUNCTIONAL:
            return max(0, 30 - age)
        elif status == ModuleEndOfLifeStatus.DEGRADED:
            return max(0, 10 - (age - 20))
        else:
            return 0

    def _calculate_reuse_potential(
        self,
        current_eff: float,
        original_eff: float,
        status: ModuleEndOfLifeStatus
    ) -> float:
        """Calculate reuse potential score (0-100)."""
        efficiency_score = (current_eff / original_eff * 100) if original_eff > 0 else 0

        status_multiplier = {
            ModuleEndOfLifeStatus.FUNCTIONAL: 1.0,
            ModuleEndOfLifeStatus.DEGRADED: 0.7,
            ModuleEndOfLifeStatus.DAMAGED: 0.4,
            ModuleEndOfLifeStatus.FAILED: 0.1
        }

        return efficiency_score * status_multiplier.get(status, 0.5)

    def _recommend_reuse_application(self, reuse_score: float) -> str:
        """Recommend reuse application based on score."""
        if reuse_score > 85:
            return "Residential rooftop (second-life)"
        elif reuse_score > 70:
            return "Off-grid applications"
        elif reuse_score > 50:
            return "Agricultural/greenhouse power"
        elif reuse_score > 30:
            return "Educational/demonstration"
        else:
            return "Not suitable for reuse - recycle"

    def _estimate_reuse_value(self, reuse_score: float, original_eff: float) -> float:
        """Estimate reuse market value."""
        # Assume $0.30/W for new modules, depreciate based on reuse score
        original_value_per_wp = 0.30
        module_power_wp = original_eff * 2.5  # Approximate module power
        return module_power_wp * original_value_per_wp * (reuse_score / 100)

    def _calculate_recyclable_materials(self) -> Dict[str, float]:
        """Calculate recyclable material quantities for typical module."""
        return {
            'silicon': 3.5,      # kg
            'glass': 12.0,       # kg
            'aluminum': 2.5,     # kg
            'copper': 0.5,       # kg
            'silver': 0.03,      # kg
            'plastic': 1.5,      # kg
            'other': 0.5         # kg
        }

    def _calculate_recycling_efficiency(self) -> float:
        """Calculate overall recycling efficiency."""
        # Based on current PV recycling technology
        return 92.0  # 92% material recovery rate

    def _estimate_recovery_value(self, materials: Dict[str, float]) -> float:
        """Estimate material recovery value."""
        # Commodity prices ($/kg)
        prices = {
            'silicon': 2.0,
            'glass': 0.05,
            'aluminum': 2.5,
            'copper': 8.0,
            'silver': 600.0,
            'plastic': 0.5,
            'other': 0.2
        }

        total_value = sum(materials.get(mat, 0) * prices.get(mat, 0) for mat in materials)
        return total_value

    def _calculate_circularity_index(
        self,
        reuse_score: float,
        recycling_eff: float,
        lifetime_ext: float
    ) -> float:
        """Calculate overall circularity index (0-100)."""
        # Weighted average: 40% reuse, 30% recycling, 30% lifetime extension
        lifetime_score = min(100, lifetime_ext / 10 * 100)
        return 0.4 * reuse_score + 0.3 * recycling_eff + 0.3 * lifetime_score

    def _recommend_circular_phase(
        self,
        status: ModuleEndOfLifeStatus,
        reuse_score: float,
        age: float
    ) -> CircularityPhase:
        """Recommend optimal circular economy phase."""
        if status == ModuleEndOfLifeStatus.FUNCTIONAL and age < 15:
            return CircularityPhase.REDUCE
        elif status == ModuleEndOfLifeStatus.FUNCTIONAL and reuse_score > 70:
            return CircularityPhase.REUSE
        elif status == ModuleEndOfLifeStatus.DEGRADED and reuse_score > 50:
            return CircularityPhase.REFURBISH
        elif status == ModuleEndOfLifeStatus.DAMAGED:
            return CircularityPhase.REMANUFACTURE
        else:
            return CircularityPhase.RECYCLE

    def _calculate_environmental_impact(self, phase: CircularityPhase, reuse_score: float) -> float:
        """Calculate environmental impact score (higher is better)."""
        phase_scores = {
            CircularityPhase.REDUCE: 95,
            CircularityPhase.REUSE: 90,
            CircularityPhase.REFURBISH: 85,
            CircularityPhase.REMANUFACTURE: 75,
            CircularityPhase.RECYCLE: 70,
            CircularityPhase.DISPOSE: 20
        }
        return phase_scores.get(phase, 50)


# ============================================================================
# B12: HYBRID ENERGY STORAGE INTEGRATION
# ============================================================================

class StorageType(str, Enum):
    """Energy storage technology types."""
    LITHIUM_ION = "Lithium-Ion"
    LEAD_ACID = "Lead-Acid"
    FLOW_BATTERY = "Flow Battery"
    SODIUM_ION = "Sodium-Ion"
    SOLID_STATE = "Solid-State"


class HybridConfiguration(BaseModel):
    """Hybrid PV + storage system configuration."""

    config_id: str = Field(..., description="Configuration identifier")
    pv_capacity_kw: float = Field(..., ge=0, description="PV capacity (kW)")
    storage_capacity_kwh: float = Field(..., ge=0, description="Storage capacity (kWh)")
    storage_power_kw: float = Field(..., ge=0, description="Storage power rating (kW)")
    storage_type: StorageType = Field(..., description="Storage technology")
    battery_efficiency: float = Field(default=90.0, ge=70, le=99, description="Round-trip efficiency (%)")
    depth_of_discharge: float = Field(default=80.0, ge=0, le=100, description="Usable DoD (%)")
    cycle_life: int = Field(..., ge=0, description="Expected cycle life")
    warranty_years: int = Field(default=10, ge=0, description="Warranty period (years)")
    installation_cost_usd: float = Field(..., ge=0, description="Installation cost ($)")

    class Config:
        use_enum_values = True


class HybridSystemDesigner:
    """
    Hybrid Energy Storage Integration Engine.
    Designs optimal PV + storage hybrid systems.
    """

    def __init__(self):
        """Initialize hybrid system designer."""
        self.configurations: List[HybridConfiguration] = []

    def design_hybrid_system(
        self,
        pv_capacity_kw: float,
        daily_load_kwh: float,
        peak_load_kw: float,
        storage_type: StorageType = StorageType.LITHIUM_ION,
        autonomy_hours: float = 4.0
    ) -> HybridConfiguration:
        """
        Design optimal hybrid PV + storage system.

        Args:
            pv_capacity_kw: PV system capacity
            daily_load_kwh: Daily energy consumption
            peak_load_kw: Peak power demand
            storage_type: Storage technology
            autonomy_hours: Desired backup duration

        Returns:
            Optimized hybrid configuration
        """
        # Size battery for autonomy requirement
        storage_capacity_kwh = peak_load_kw * autonomy_hours

        # Size inverter/storage power
        storage_power_kw = peak_load_kw * 1.2  # 20% margin

        # Get technology specifications
        specs = self._get_storage_specs(storage_type)

        # Calculate costs
        cost_per_kwh = specs['cost_per_kwh']
        installation_cost = storage_capacity_kwh * cost_per_kwh

        config = HybridConfiguration(
            config_id=f"HYBRID_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            pv_capacity_kw=pv_capacity_kw,
            storage_capacity_kwh=storage_capacity_kwh,
            storage_power_kw=storage_power_kw,
            storage_type=storage_type,
            battery_efficiency=specs['efficiency'],
            depth_of_discharge=specs['dod'],
            cycle_life=specs['cycle_life'],
            warranty_years=specs['warranty_years'],
            installation_cost_usd=installation_cost
        )

        self.configurations.append(config)
        return config

    def _get_storage_specs(self, storage_type: StorageType) -> Dict[str, Any]:
        """Get storage technology specifications."""
        specs = {
            StorageType.LITHIUM_ION: {
                'efficiency': 95.0,
                'dod': 90.0,
                'cycle_life': 6000,
                'warranty_years': 10,
                'cost_per_kwh': 400
            },
            StorageType.LEAD_ACID: {
                'efficiency': 80.0,
                'dod': 50.0,
                'cycle_life': 1500,
                'warranty_years': 5,
                'cost_per_kwh': 200
            },
            StorageType.FLOW_BATTERY: {
                'efficiency': 75.0,
                'dod': 100.0,
                'cycle_life': 10000,
                'warranty_years': 20,
                'cost_per_kwh': 500
            },
            StorageType.SODIUM_ION: {
                'efficiency': 92.0,
                'dod': 85.0,
                'cycle_life': 5000,
                'warranty_years': 10,
                'cost_per_kwh': 350
            },
            StorageType.SOLID_STATE: {
                'efficiency': 98.0,
                'dod': 95.0,
                'cycle_life': 10000,
                'warranty_years': 15,
                'cost_per_kwh': 600
            }
        }
        return specs.get(storage_type, specs[StorageType.LITHIUM_ION])

    def simulate_energy_flow(
        self,
        config: HybridConfiguration,
        pv_generation: List[float],
        load_demand: List[float]
    ) -> Dict[str, Any]:
        """
        Simulate energy flow in hybrid system.

        Args:
            config: Hybrid configuration
            pv_generation: Hourly PV generation (kWh)
            load_demand: Hourly load demand (kWh)

        Returns:
            Energy flow simulation results
        """
        battery_soc = []  # State of charge
        grid_import = []
        grid_export = []
        battery_charge = []
        battery_discharge = []

        current_soc = config.storage_capacity_kwh * 0.5  # Start at 50% SOC

        for pv, load in zip(pv_generation, load_demand):
            net_power = pv - load

            if net_power > 0:  # Excess PV
                # Charge battery
                charge_amount = min(net_power, config.storage_power_kw,
                                   config.storage_capacity_kwh - current_soc)
                current_soc += charge_amount * (config.battery_efficiency / 100)
                battery_charge.append(charge_amount)
                battery_discharge.append(0)

                # Export to grid if battery full
                export = max(0, net_power - charge_amount)
                grid_export.append(export)
                grid_import.append(0)

            else:  # Load exceeds PV
                deficit = abs(net_power)

                # Discharge battery
                discharge_amount = min(deficit, config.storage_power_kw, current_soc)
                current_soc -= discharge_amount / (config.battery_efficiency / 100)
                battery_discharge.append(discharge_amount)
                battery_charge.append(0)

                # Import from grid if needed
                import_amount = max(0, deficit - discharge_amount)
                grid_import.append(import_amount)
                grid_export.append(0)

            battery_soc.append(current_soc)

        return {
            'battery_soc': battery_soc,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'battery_charge': battery_charge,
            'battery_discharge': battery_discharge,
            'self_consumption_ratio': (sum(pv_generation) - sum(grid_export)) / sum(pv_generation) * 100 if sum(pv_generation) > 0 else 0,
            'self_sufficiency_ratio': (sum(load_demand) - sum(grid_import)) / sum(load_demand) * 100 if sum(load_demand) > 0 else 0
        }


# ============================================================================
# CIRCULARITY SUITE INTEGRATION INTERFACE
# ============================================================================

class CircularitySuite:
    """
    Unified Circularity Suite Interface integrating B10-B12.
    Provides comprehensive circular economy and hybrid system capabilities.
    """

    def __init__(self):
        """Initialize all circularity suite components."""
        self.revamp_planner = RevampPlanner()
        self.circularity_assessor = CircularityAssessor()
        self.hybrid_designer = HybridSystemDesigner()

    def complete_circularity_analysis(
        self,
        system_age_years: float,
        system_capacity_kw: float,
        current_pr: float,
        module_efficiency: float,
        original_efficiency: float
    ) -> Dict[str, Any]:
        """
        Execute complete circularity analysis.

        Args:
            system_age_years: System age
            system_capacity_kw: System capacity
            current_pr: Current performance ratio
            module_efficiency: Current module efficiency
            original_efficiency: Original module efficiency

        Returns:
            Complete circularity analysis results
        """
        # Revamp assessment
        revamp = self.revamp_planner.assess_system(
            system_age_years=system_age_years,
            current_capacity_kw=system_capacity_kw,
            current_pr=current_pr,
            annual_degradation=0.5
        )

        # Circularity assessment
        circularity = self.circularity_assessor.assess_module_circularity(
            module_id="MODULE_001",
            module_age_years=system_age_years,
            current_efficiency=module_efficiency,
            original_efficiency=original_efficiency
        )

        # Hybrid system design (if revamp recommended)
        hybrid = None
        if revamp.recommended_strategy == RevampStrategy.HYBRID_CONVERSION:
            hybrid = self.hybrid_designer.design_hybrid_system(
                pv_capacity_kw=system_capacity_kw,
                daily_load_kwh=system_capacity_kw * 4,
                peak_load_kw=system_capacity_kw * 0.8
            )

        return {
            'revamp_assessment': revamp.dict(),
            'circularity_metrics': circularity.dict(),
            'hybrid_configuration': hybrid.dict() if hybrid else None,
            'overall_circularity_score': circularity.circularity_index,
            'analysis_complete': True
        }


# Export main interface
__all__ = [
    'CircularitySuite',
    'RevampPlanner',
    'RevampAssessment',
    'CircularityAssessor',
    'CircularityMetrics',
    'HybridSystemDesigner',
    'HybridConfiguration',
    'StorageType'
]
