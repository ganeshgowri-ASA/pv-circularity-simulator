"""Repower Analysis & Feasibility Study for PV Systems.

This module provides comprehensive analysis tools for evaluating the technical and economic
feasibility of repowering existing photovoltaic systems. It includes capacity upgrade analysis,
component replacement planning, technical feasibility assessment, and economic viability analysis.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from pv_simulator.core.enums import (
    ComponentType,
    HealthStatus,
    ModuleTechnology,
    RepowerStrategy,
)
from pv_simulator.core.models import (
    CapacityUpgradeAnalysis,
    ComponentHealth,
    ComponentReplacementPlan,
    CostBreakdown,
    EconomicMetrics,
    EconomicViabilityResult,
    PVModule,
    PVSystem,
    RepowerScenario,
    TechnicalFeasibilityResult,
)


class RepowerAnalyzerConfig(BaseModel):
    """Configuration parameters for repower analysis.

    Attributes:
        electricity_rate: Electricity rate for energy valuation ($/kWh)
        discount_rate: Discount rate for NPV calculations (fraction)
        analysis_period: Analysis period for economic evaluation (years)
        escalation_rate: Annual electricity price escalation (fraction)
        structural_safety_factor: Safety factor for structural capacity (1.5 = 50% margin)
        max_inverter_loading_ratio: Maximum DC/AC ratio for inverter sizing
        min_roi_threshold: Minimum ROI for viability (fraction)
        min_payback_period: Maximum acceptable payback period (years)
        degradation_threshold: Performance ratio triggering replacement (fraction)
    """

    electricity_rate: float = Field(
        default=0.12, gt=0, description="Electricity rate ($/kWh)"
    )
    discount_rate: float = Field(
        default=0.06, gt=0, lt=1, description="Discount rate (fraction)"
    )
    analysis_period: int = Field(default=25, gt=0, description="Analysis period (years)")
    escalation_rate: float = Field(
        default=0.025, ge=0, lt=1, description="Electricity escalation (fraction/year)"
    )
    structural_safety_factor: float = Field(
        default=1.5, gt=1, description="Structural safety factor"
    )
    max_inverter_loading_ratio: float = Field(
        default=1.35, gt=1, le=1.5, description="Max DC/AC ratio"
    )
    min_roi_threshold: float = Field(
        default=0.10, gt=0, description="Minimum ROI threshold (fraction)"
    )
    max_payback_period: float = Field(
        default=12.0, gt=0, description="Max payback period (years)"
    )
    degradation_threshold: float = Field(
        default=0.80, gt=0, le=1, description="Degradation threshold (fraction)"
    )


class RepowerAnalyzer:
    """Comprehensive repower analysis and feasibility assessment for PV systems.

    This analyzer provides end-to-end analysis capabilities for evaluating whether and how
    to repower existing PV systems. It combines engineering analysis with financial modeling
    to support data-driven decision making.

    Key Features:
        - Capacity upgrade analysis with constraint identification
        - Component-level replacement planning with prioritization
        - Multi-dimensional technical feasibility assessment
        - Comprehensive economic viability analysis with sensitivity testing
        - Multiple repower strategy evaluation
        - Risk assessment and mitigation planning

    Example:
        >>> from pv_simulator import RepowerAnalyzer, PVSystem
        >>> analyzer = RepowerAnalyzer()
        >>> system = PVSystem(...)  # Define existing system
        >>> capacity_analysis = analyzer.capacity_upgrade_analysis(system)
        >>> feasibility = analyzer.technical_feasibility_check(system, target_capacity=150.0)
        >>> economics = analyzer.economic_viability_analysis(system, scenarios=[...])
    """

    def __init__(self, config: Optional[RepowerAnalyzerConfig] = None):
        """Initialize the RepowerAnalyzer.

        Args:
            config: Configuration parameters for analysis. Uses defaults if not provided.
        """
        self.config = config or RepowerAnalyzerConfig()

    def capacity_upgrade_analysis(
        self,
        system: PVSystem,
        available_roof_area: Optional[float] = None,
        structural_load_limit: Optional[float] = None,
        electrical_capacity_limit: Optional[float] = None,
    ) -> CapacityUpgradeAnalysis:
        """Analyze capacity upgrade potential and constraints for a PV system.

        Evaluates the maximum additional capacity that can be added to an existing system
        by analyzing spatial, structural, and electrical constraints. Identifies limiting
        factors and generates multiple upgrade scenarios.

        Args:
            system: Existing PV system to analyze
            available_roof_area: Additional available roof/ground area (m²). If None,
                assumes 50% expansion possible.
            structural_load_limit: Maximum additional structural load capacity (kg).
                If None, calculated based on typical racking standards.
            electrical_capacity_limit: Maximum additional electrical capacity (A).
                If None, calculated from inverter and wiring capacity.

        Returns:
            CapacityUpgradeAnalysis containing:
                - Maximum additional capacity possible
                - Available space, structural, and electrical capacity
                - Multiple upgrade scenarios (25%, 50%, 75%, 100% of maximum)
                - Recommended upgrade capacity
                - Identification of primary limiting factor

        Engineering Methodology:
            1. Spatial Analysis: Calculate available area and module fit
            2. Structural Analysis: Evaluate load-bearing capacity with safety factors
            3. Electrical Analysis: Assess inverter capacity and wiring limits
            4. Scenario Generation: Create upgrade options at different scales
            5. Optimization: Recommend optimal upgrade based on constraints
        """
        # Calculate current system metrics
        current_module_area = system.module.area * system.num_modules

        # Determine available roof area
        if available_roof_area is None:
            # Assume 50% expansion is possible (conservative estimate)
            available_roof_area = current_module_area * 0.5

        # Calculate space-based capacity
        modules_that_fit = int(available_roof_area / system.module.area)
        space_based_capacity = (modules_that_fit * system.module.rated_power) / 1000  # kW

        # Structural capacity analysis
        if structural_load_limit is None:
            # Typical module weight: 15-20 kg/m² for standard crystalline silicon
            typical_module_weight_per_m2 = 18.0  # kg/m²
            # Typical roof load capacity: 50-100 kg/m² for residential, 100-200 for commercial
            typical_load_capacity_per_m2 = 100.0  # kg/m² (conservative)
            structural_load_limit = (
                available_roof_area
                * (typical_load_capacity_per_m2 - typical_module_weight_per_m2)
                / self.config.structural_safety_factor
            )

        # Estimate module weight (typical: 15-25 kg for residential modules)
        module_weight = system.module.area * 18.0  # kg, based on typical weight
        structural_based_modules = int(structural_load_limit / module_weight)
        structural_based_capacity = (
            structural_based_modules * system.module.rated_power
        ) / 1000  # kW

        # Electrical capacity analysis
        if electrical_capacity_limit is None:
            # Calculate from inverter capacity and max loading ratio
            max_dc_from_inverter = (
                system.ac_capacity * self.config.max_inverter_loading_ratio
            )
            electrical_capacity_limit = max(0, max_dc_from_inverter - system.dc_capacity)
        else:
            electrical_capacity_limit = electrical_capacity_limit  # Already in kW

        # Determine maximum additional capacity (limited by constraints)
        max_additional_capacity = min(
            space_based_capacity, structural_based_capacity, electrical_capacity_limit
        )

        # Identify limiting factor
        limiting_factors = {
            "space": space_based_capacity,
            "structural": structural_based_capacity,
            "electrical": electrical_capacity_limit,
        }
        limiting_factor = min(limiting_factors, key=limiting_factors.get)

        # Generate upgrade scenarios
        upgrade_scenarios = []
        for percentage in [0.25, 0.5, 0.75, 1.0]:
            scenario_capacity = max_additional_capacity * percentage
            upgrade_scenarios.append(
                {
                    "upgrade_capacity_kw": round(scenario_capacity, 2),
                    "total_capacity_kw": round(system.dc_capacity + scenario_capacity, 2),
                    "capacity_increase_pct": round(
                        (scenario_capacity / system.dc_capacity) * 100, 1
                    ),
                    "num_additional_modules": int(
                        (scenario_capacity * 1000) / system.module.rated_power
                    ),
                }
            )

        # Recommend upgrade (typically 50-75% of maximum for optimal cost/benefit)
        recommended_upgrade = max_additional_capacity * 0.65

        return CapacityUpgradeAnalysis(
            current_capacity=system.dc_capacity,
            max_additional_capacity=round(max_additional_capacity, 2),
            space_available=round(available_roof_area, 2),
            structural_capacity_available=round(structural_load_limit, 2),
            electrical_capacity_available=round(electrical_capacity_limit, 2),
            upgrade_scenarios=upgrade_scenarios,
            recommended_upgrade=round(recommended_upgrade, 2),
            limiting_factor=f"{limiting_factor} (allows {limiting_factors[limiting_factor]:.2f} kW)",
        )

    def component_replacement_planning(
        self,
        system: PVSystem,
        planning_horizon_years: int = 5,
    ) -> ComponentReplacementPlan:
        """Develop a prioritized component replacement plan based on health and risk.

        Analyzes all system components to create a time-phased replacement plan that
        balances risk mitigation with budget optimization. Prioritizes replacements
        based on component health, failure probability, and system impact.

        Args:
            system: PV system with component health data
            planning_horizon_years: Planning horizon for replacement scheduling (years)

        Returns:
            ComponentReplacementPlan containing:
                - Immediate replacements (critical failures)
                - Short-term replacements (0-1 year)
                - Medium-term replacements (1-3 years)
                - Long-term replacements (3-5 years)
                - Total replacement costs
                - Priority order for replacements
                - Risk mitigation strategies

        Planning Methodology:
            1. Component Assessment: Evaluate health status and failure risk
            2. Criticality Analysis: Assess system impact of component failures
            3. Time-Phase Planning: Schedule replacements based on urgency
            4. Cost Optimization: Balance replacement timing with budget
            5. Risk Mitigation: Develop strategies for high-risk components
        """
        immediate = []
        short_term = []
        medium_term = []
        long_term = []
        total_cost = 0.0

        for component in system.component_health:
            total_cost += component.replacement_cost

            # Calculate remaining useful life
            remaining_life = max(
                0, component.expected_lifetime - component.age_years
            )

            # Determine replacement timing based on status and risk
            if component.status == HealthStatus.FAILED or component.status == HealthStatus.CRITICAL:
                immediate.append(component)
            elif (
                component.status == HealthStatus.POOR
                or component.failure_probability > 0.3
                or remaining_life < 1.0
            ):
                short_term.append(component)
            elif (
                component.status == HealthStatus.FAIR
                or component.failure_probability > 0.15
                or remaining_life < 3.0
            ):
                medium_term.append(component)
            elif remaining_life < planning_horizon_years:
                long_term.append(component)

        # Prioritize components by criticality
        priority_order = self._prioritize_components(system.component_health)

        # Develop risk mitigation plan
        risk_mitigation = {}
        for component in immediate + short_term:
            if component.component_type == ComponentType.INVERTER:
                risk_mitigation[component.component_type.value] = (
                    "Consider redundant inverter capacity or rapid replacement contracts"
                )
            elif component.component_type == ComponentType.MODULE:
                risk_mitigation[component.component_type.value] = (
                    "Monitor hot spots with thermal imaging; have replacement modules on-site"
                )
            elif component.component_type == ComponentType.MONITORING:
                risk_mitigation[component.component_type.value] = (
                    "Install backup monitoring system; increase manual inspection frequency"
                )

        return ComponentReplacementPlan(
            immediate_replacements=immediate,
            short_term_replacements=short_term,
            medium_term_replacements=medium_term,
            long_term_replacements=long_term,
            total_replacement_cost=round(total_cost, 2),
            priority_order=priority_order,
            risk_mitigation_plan=risk_mitigation,
        )

    def technical_feasibility_check(
        self,
        system: PVSystem,
        repower_scenario: Optional[RepowerScenario] = None,
        target_capacity: Optional[float] = None,
        new_module: Optional[PVModule] = None,
    ) -> TechnicalFeasibilityResult:
        """Assess technical feasibility of a repower project across multiple dimensions.

        Evaluates whether a proposed repower is technically feasible by analyzing
        structural, electrical, spatial, regulatory, and integration constraints.
        Provides a comprehensive feasibility score and identifies specific constraints.

        Args:
            system: Existing PV system
            repower_scenario: Complete repower scenario to evaluate. If None, uses
                target_capacity and new_module to create scenario.
            target_capacity: Target DC capacity after repower (kW). Required if
                repower_scenario is None.
            new_module: New module specifications. If None, uses system.module.

        Returns:
            TechnicalFeasibilityResult containing:
                - Overall feasibility determination (boolean)
                - Overall feasibility score (0-100)
                - Individual scores for structural, electrical, spatial,
                  regulatory, and integration feasibility
                - List of identified constraints
                - List of identified risks
                - Specific recommendations for addressing issues

        Assessment Dimensions:
            1. Structural: Load capacity, mounting compatibility, structural integrity
            2. Electrical: Inverter capacity, wiring adequacy, grid connection limits
            3. Spatial: Available space, layout optimization, shading impacts
            4. Regulatory: Code compliance, permitting requirements, utility approval
            5. Integration: Component compatibility, system integration, monitoring
        """
        # Use provided scenario or create from parameters
        if repower_scenario is None:
            if target_capacity is None:
                raise ValueError(
                    "Either repower_scenario or target_capacity must be provided"
                )
            if new_module is None:
                new_module = system.module

        capacity_to_check = (
            repower_scenario.new_dc_capacity if repower_scenario else target_capacity
        )
        module_to_check = (
            repower_scenario.new_module if repower_scenario and repower_scenario.new_module
            else new_module or system.module
        )

        constraints = []
        risks = []
        recommendations = []

        # 1. Structural Feasibility Analysis
        structural_score = self._assess_structural_feasibility(
            system, capacity_to_check, module_to_check, constraints, risks
        )

        # 2. Electrical Feasibility Analysis
        electrical_score = self._assess_electrical_feasibility(
            system, capacity_to_check, constraints, risks
        )

        # 3. Spatial Feasibility Analysis
        spatial_score = self._assess_spatial_feasibility(
            system, capacity_to_check, module_to_check, constraints, risks
        )

        # 4. Regulatory Feasibility Analysis
        regulatory_score = self._assess_regulatory_feasibility(
            system, capacity_to_check, constraints, recommendations
        )

        # 5. Integration Feasibility Analysis
        integration_score = self._assess_integration_feasibility(
            system, module_to_check, constraints, risks
        )

        # Calculate overall feasibility score (weighted average)
        weights = {
            "structural": 0.25,
            "electrical": 0.25,
            "spatial": 0.15,
            "regulatory": 0.20,
            "integration": 0.15,
        }

        overall_score = (
            structural_score * weights["structural"]
            + electrical_score * weights["electrical"]
            + spatial_score * weights["spatial"]
            + regulatory_score * weights["regulatory"]
            + integration_score * weights["integration"]
        )

        # Determine overall feasibility (score >= 60 is feasible)
        is_feasible = overall_score >= 60.0 and len([c for c in constraints if "CRITICAL" in c]) == 0

        # Add general recommendations
        if not is_feasible:
            recommendations.append(
                "Consider alternative repower strategies or phased implementation"
            )
        if overall_score < 80:
            recommendations.append(
                "Conduct detailed site assessment before proceeding"
            )

        return TechnicalFeasibilityResult(
            is_feasible=is_feasible,
            feasibility_score=round(overall_score, 1),
            structural_feasibility=round(structural_score, 1),
            electrical_feasibility=round(electrical_score, 1),
            spatial_feasibility=round(spatial_score, 1),
            regulatory_feasibility=round(regulatory_score, 1),
            integration_feasibility=round(integration_score, 1),
            constraints=constraints,
            risks=risks,
            recommendations=recommendations,
        )

    def economic_viability_analysis(
        self,
        system: PVSystem,
        repower_scenarios: List[RepowerScenario],
        electricity_rate: Optional[float] = None,
        incentives: Optional[Dict[str, float]] = None,
    ) -> EconomicViabilityResult:
        """Analyze economic viability of repower scenarios with comprehensive financial modeling.

        Evaluates the economic performance of multiple repower scenarios using standard
        financial metrics including NPV, IRR, ROI, and LCOE. Performs sensitivity analysis
        and identifies break-even conditions.

        Args:
            system: Existing PV system
            repower_scenarios: List of repower scenarios to evaluate
            electricity_rate: Electricity rate for energy valuation ($/kWh). Uses config
                default if not provided.
            incentives: Available incentives and rebates as dict of {name: amount ($)}

        Returns:
            EconomicViabilityResult containing:
                - Overall economic viability determination
                - Viability score (0-100)
                - All scenarios with calculated economic metrics
                - Best performing scenario
                - Sensitivity analysis results
                - Break-even conditions
                - Financing options
                - Available incentives

        Financial Methodology:
            1. Cash Flow Modeling: Project annual revenues and costs over analysis period
            2. Metric Calculation: Compute NPV, IRR, ROI, payback period, and LCOE
            3. Scenario Comparison: Rank scenarios by multiple financial metrics
            4. Sensitivity Analysis: Vary key parameters (electricity rate, costs, production)
            5. Risk Assessment: Evaluate financial risks and uncertainty
            6. Optimization: Identify optimal scenario and financing structure
        """
        rate = electricity_rate or self.config.electricity_rate
        incentives_dict = incentives or {}

        analyzed_scenarios = []
        best_scenario = None
        best_npv = float("-inf")

        # Analyze each scenario
        for scenario in repower_scenarios:
            # Calculate economic metrics
            metrics = self._calculate_economic_metrics(
                system=system,
                scenario=scenario,
                electricity_rate=rate,
                incentives=incentives_dict,
            )

            # Update scenario with metrics
            scenario.economic_metrics = metrics

            # Apply total CAPEX to cost breakdown
            scenario.cost_breakdown.total_capex = scenario.cost_breakdown.calculate_total()

            analyzed_scenarios.append(scenario)

            # Track best scenario by NPV
            if metrics.npv > best_npv:
                best_npv = metrics.npv
                best_scenario = scenario

        # Perform sensitivity analysis on best scenario
        sensitivity_analysis = {}
        if best_scenario:
            sensitivity_analysis = self._perform_sensitivity_analysis(
                system, best_scenario, rate, incentives_dict
            )

        # Calculate break-even scenarios
        break_even = {}
        if best_scenario and best_scenario.economic_metrics:
            break_even = self._calculate_break_even(
                system, best_scenario, best_scenario.economic_metrics
            )

        # Determine overall viability
        is_viable = False
        if best_scenario and best_scenario.economic_metrics:
            metrics = best_scenario.economic_metrics
            is_viable = (
                metrics.npv > 0
                and metrics.roi >= self.config.min_roi_threshold
                and metrics.payback_period <= self.config.max_payback_period
            )

        # Calculate viability score
        viability_score = self._calculate_viability_score(analyzed_scenarios, best_scenario)

        # Suggest financing options
        financing_options = self._suggest_financing_options(best_scenario)

        return EconomicViabilityResult(
            is_viable=is_viable,
            viability_score=round(viability_score, 1),
            scenarios_analyzed=analyzed_scenarios,
            best_scenario=best_scenario,
            sensitivity_analysis=sensitivity_analysis,
            break_even_scenarios=break_even,
            financing_options=financing_options,
            incentives_available=incentives_dict,
        )

    # ==================== Private Helper Methods ====================

    def _prioritize_components(
        self, components: List[ComponentHealth]
    ) -> List[ComponentType]:
        """Prioritize components for replacement based on criticality.

        Args:
            components: List of component health statuses

        Returns:
            Prioritized list of component types
        """
        # Define criticality weights (higher = more critical)
        criticality = {
            ComponentType.INVERTER: 10,  # Most critical - affects entire system
            ComponentType.MODULE: 8,  # High criticality - direct energy production
            ComponentType.TRANSFORMER: 7,
            ComponentType.COMBINER_BOX: 6,
            ComponentType.WIRING: 5,
            ComponentType.MONITORING: 4,
            ComponentType.DISCONNECT: 3,
            ComponentType.RACKING: 2,
        }

        # Score components by health status and criticality
        component_scores = []
        for component in components:
            health_score = {
                HealthStatus.FAILED: 100,
                HealthStatus.CRITICAL: 80,
                HealthStatus.POOR: 60,
                HealthStatus.FAIR: 40,
                HealthStatus.GOOD: 20,
                HealthStatus.EXCELLENT: 10,
            }.get(component.status, 50)

            total_score = (
                health_score
                + criticality.get(component.component_type, 5) * 5
                + component.failure_probability * 100
            )

            component_scores.append((component.component_type, total_score))

        # Sort by score (descending) and return unique types
        sorted_components = sorted(component_scores, key=lambda x: x[1], reverse=True)
        seen = set()
        priority_order = []
        for comp_type, _ in sorted_components:
            if comp_type not in seen:
                priority_order.append(comp_type)
                seen.add(comp_type)

        return priority_order

    def _assess_structural_feasibility(
        self,
        system: PVSystem,
        target_capacity: float,
        new_module: PVModule,
        constraints: List[str],
        risks: List[str],
    ) -> float:
        """Assess structural feasibility for repower."""
        score = 100.0

        # Calculate load increase
        old_modules_needed = int((system.dc_capacity * 1000) / system.module.rated_power)
        new_modules_needed = int((target_capacity * 1000) / new_module.rated_power)

        module_increase_pct = ((new_modules_needed - old_modules_needed) / old_modules_needed) * 100

        # Assess structural impacts
        if module_increase_pct > 50:
            score -= 30
            constraints.append("CRITICAL: >50% increase in module count may exceed structural capacity")
            risks.append("Structural reinforcement required")
        elif module_increase_pct > 25:
            score -= 15
            constraints.append("Moderate: >25% increase requires structural assessment")
            risks.append("Potential structural upgrades needed")

        # Consider system age
        system_age = (date.today() - system.installation_date).days / 365.25
        if system_age > 15:
            score -= 10
            risks.append("Aging structure may need reinforcement")

        return max(0, score)

    def _assess_electrical_feasibility(
        self,
        system: PVSystem,
        target_capacity: float,
        constraints: List[str],
        risks: List[str],
    ) -> float:
        """Assess electrical feasibility for repower."""
        score = 100.0

        # Check inverter capacity
        max_dc_capacity = system.ac_capacity * self.config.max_inverter_loading_ratio
        if target_capacity > max_dc_capacity:
            score -= 40
            constraints.append(
                f"CRITICAL: Target capacity {target_capacity:.1f}kW exceeds "
                f"max inverter capacity {max_dc_capacity:.1f}kW"
            )
            risks.append("Inverter replacement required")
        elif target_capacity > system.ac_capacity * 1.15:
            score -= 10
            constraints.append("Inverter may need upgrading for optimal performance")

        # Check capacity increase rate
        capacity_increase = ((target_capacity - system.dc_capacity) / system.dc_capacity) * 100
        if capacity_increase > 100:
            score -= 20
            constraints.append("Major electrical infrastructure upgrades required")
            risks.append("Utility interconnection approval needed")

        return max(0, score)

    def _assess_spatial_feasibility(
        self,
        system: PVSystem,
        target_capacity: float,
        new_module: PVModule,
        constraints: List[str],
        risks: List[str],
    ) -> float:
        """Assess spatial feasibility for repower."""
        score = 100.0

        # Calculate area requirements
        old_modules_needed = int((system.dc_capacity * 1000) / system.module.rated_power)
        new_modules_needed = int((target_capacity * 1000) / new_module.rated_power)

        old_area = old_modules_needed * system.module.area
        new_area = new_modules_needed * new_module.area

        area_increase_pct = ((new_area - old_area) / old_area) * 100

        if area_increase_pct > 50:
            score -= 35
            constraints.append(
                f"CRITICAL: {area_increase_pct:.1f}% area increase may not be available"
            )
            risks.append("Insufficient space for planned capacity")
        elif area_increase_pct > 25:
            score -= 15
            constraints.append(f"Moderate: {area_increase_pct:.1f}% area increase required")

        return max(0, score)

    def _assess_regulatory_feasibility(
        self,
        system: PVSystem,
        target_capacity: float,
        constraints: List[str],
        recommendations: List[str],
    ) -> float:
        """Assess regulatory and permitting feasibility."""
        score = 100.0

        # Check if capacity increase triggers utility review
        capacity_increase_pct = ((target_capacity - system.dc_capacity) / system.dc_capacity) * 100

        if capacity_increase_pct > 50:
            score -= 20
            constraints.append("Utility interconnection study required")
            recommendations.append("Apply for utility approval early in planning process")
        elif capacity_increase_pct > 20:
            score -= 10
            constraints.append("Utility notification required")

        # Check system age for code compliance
        system_age = (date.today() - system.installation_date).days / 365.25
        if system_age > 10:
            score -= 15
            constraints.append("Must comply with current building and electrical codes")
            recommendations.append("Conduct code compliance review before design")

        return max(0, score)

    def _assess_integration_feasibility(
        self,
        system: PVSystem,
        new_module: PVModule,
        constraints: List[str],
        risks: List[str],
    ) -> float:
        """Assess component integration feasibility."""
        score = 100.0

        # Check module technology compatibility
        if system.module.technology != new_module.technology:
            score -= 10
            constraints.append("Different module technology requires system reconfiguration")
            risks.append("Integration complexity with mixed technologies")

        # Check voltage compatibility (modules with very different Voc)
        # Simplified check based on efficiency as proxy
        efficiency_diff = abs(new_module.efficiency - system.module.efficiency) / system.module.efficiency
        if efficiency_diff > 0.3:
            score -= 15
            constraints.append("Significant module performance difference may affect system design")
            risks.append("String reconfiguration may be needed")

        return max(0, score)

    def _calculate_economic_metrics(
        self,
        system: PVSystem,
        scenario: RepowerScenario,
        electricity_rate: float,
        incentives: Dict[str, float],
    ) -> EconomicMetrics:
        """Calculate comprehensive economic metrics for a scenario."""
        # Total capital cost
        total_capex = scenario.cost_breakdown.calculate_total()

        # Apply incentives
        total_incentives = sum(incentives.values())
        net_capex = total_capex - total_incentives

        # Annual production and value
        annual_production = scenario.estimated_annual_production
        annual_energy_value = annual_production * electricity_rate

        # Annual O&M costs (typically 1-2% of CAPEX or $15-25/kW/year)
        annual_opex = max(
            scenario.new_dc_capacity * 20,  # $20/kW/year
            total_capex * 0.015  # 1.5% of CAPEX
        )

        # Calculate NPV
        npv = self._calculate_npv(
            initial_cost=net_capex,
            annual_revenue=annual_energy_value,
            annual_opex=annual_opex,
            discount_rate=self.config.discount_rate,
            years=self.config.analysis_period,
            escalation_rate=self.config.escalation_rate,
        )

        # Calculate IRR
        irr = self._calculate_irr(
            initial_cost=net_capex,
            annual_revenue=annual_energy_value,
            annual_opex=annual_opex,
            years=self.config.analysis_period,
            escalation_rate=self.config.escalation_rate,
        )

        # Calculate simple payback period
        net_annual_cashflow = annual_energy_value - annual_opex
        payback_period = net_capex / net_annual_cashflow if net_annual_cashflow > 0 else 999

        # Calculate ROI
        total_revenue = annual_energy_value * self.config.analysis_period
        roi = (total_revenue - total_capex - (annual_opex * self.config.analysis_period)) / total_capex

        # Calculate LCOE
        lcoe = self._calculate_lcoe(
            capex=total_capex,
            annual_opex=annual_opex,
            annual_production=annual_production,
            discount_rate=self.config.discount_rate,
            years=self.config.analysis_period,
        )

        # Benefit-cost ratio
        total_benefits = total_revenue
        total_costs = total_capex + (annual_opex * self.config.analysis_period)
        bcr = total_benefits / total_costs if total_costs > 0 else 0

        return EconomicMetrics(
            lcoe=round(lcoe, 4),
            npv=round(npv, 2),
            irr=round(irr, 4),
            payback_period=round(payback_period, 2),
            roi=round(roi, 4),
            benefit_cost_ratio=round(bcr, 3),
            annual_energy_value=round(annual_energy_value, 2),
            annual_opex=round(annual_opex, 2),
            discount_rate=self.config.discount_rate,
            analysis_period=self.config.analysis_period,
        )

    def _calculate_npv(
        self,
        initial_cost: float,
        annual_revenue: float,
        annual_opex: float,
        discount_rate: float,
        years: int,
        escalation_rate: float,
    ) -> float:
        """Calculate net present value."""
        npv = -initial_cost

        for year in range(1, years + 1):
            # Revenue escalates with electricity prices
            escalated_revenue = annual_revenue * ((1 + escalation_rate) ** year)
            # OPEX typically escalates with inflation (use escalation rate as proxy)
            escalated_opex = annual_opex * ((1 + escalation_rate * 0.5) ** year)

            net_cashflow = escalated_revenue - escalated_opex
            discount_factor = (1 + discount_rate) ** year
            npv += net_cashflow / discount_factor

        return npv

    def _calculate_irr(
        self,
        initial_cost: float,
        annual_revenue: float,
        annual_opex: float,
        years: int,
        escalation_rate: float,
    ) -> float:
        """Calculate internal rate of return using Newton-Raphson method."""
        # Create cash flow array
        cash_flows = [-initial_cost]

        for year in range(1, years + 1):
            escalated_revenue = annual_revenue * ((1 + escalation_rate) ** year)
            escalated_opex = annual_opex * ((1 + escalation_rate * 0.5) ** year)
            cash_flows.append(escalated_revenue - escalated_opex)

        # Use numpy's IRR calculation
        try:
            irr = np.irr(cash_flows)
            return irr if not np.isnan(irr) else 0.0
        except:
            # Fallback to approximation if numpy fails
            return self._approximate_irr(cash_flows)

    def _approximate_irr(self, cash_flows: List[float]) -> float:
        """Approximate IRR using bisection method."""
        def npv_at_rate(rate: float) -> float:
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))

        # Bisection method
        low, high = -0.5, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            npv_mid = npv_at_rate(mid)

            if abs(npv_mid) < 0.01:
                return mid
            elif npv_mid > 0:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def _calculate_lcoe(
        self,
        capex: float,
        annual_opex: float,
        annual_production: float,
        discount_rate: float,
        years: int,
    ) -> float:
        """Calculate levelized cost of energy."""
        # Present value of costs
        pv_costs = capex
        for year in range(1, years + 1):
            pv_costs += annual_opex / ((1 + discount_rate) ** year)

        # Present value of production
        pv_production = 0
        for year in range(1, years + 1):
            # Account for degradation (typically 0.5%/year)
            degraded_production = annual_production * ((1 - 0.005) ** year)
            pv_production += degraded_production / ((1 + discount_rate) ** year)

        lcoe = pv_costs / pv_production if pv_production > 0 else 999
        return lcoe

    def _perform_sensitivity_analysis(
        self,
        system: PVSystem,
        scenario: RepowerScenario,
        base_electricity_rate: float,
        incentives: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on key variables."""
        sensitivity = {}

        if not scenario.economic_metrics:
            return sensitivity

        base_npv = scenario.economic_metrics.npv

        # Electricity rate sensitivity (-20% to +20%)
        electricity_sensitivity = {}
        for variation in [-0.2, -0.1, 0, 0.1, 0.2]:
            rate = base_electricity_rate * (1 + variation)
            metrics = self._calculate_economic_metrics(system, scenario, rate, incentives)
            electricity_sensitivity[f"{variation*100:+.0f}%"] = metrics.npv - base_npv

        sensitivity["electricity_rate"] = electricity_sensitivity

        # CAPEX sensitivity (-20% to +20%)
        capex_sensitivity = {}
        original_costs = {
            "module": scenario.cost_breakdown.module_costs,
            "inverter": scenario.cost_breakdown.inverter_costs,
            "bos": scenario.cost_breakdown.bos_costs,
            "labor": scenario.cost_breakdown.labor_costs,
        }

        for variation in [-0.2, -0.1, 0, 0.1, 0.2]:
            scenario.cost_breakdown.module_costs = original_costs["module"] * (1 + variation)
            scenario.cost_breakdown.inverter_costs = original_costs["inverter"] * (1 + variation)
            scenario.cost_breakdown.bos_costs = original_costs["bos"] * (1 + variation)
            scenario.cost_breakdown.labor_costs = original_costs["labor"] * (1 + variation)

            metrics = self._calculate_economic_metrics(
                system, scenario, base_electricity_rate, incentives
            )
            capex_sensitivity[f"{variation*100:+.0f}%"] = metrics.npv - base_npv

        # Restore original costs
        scenario.cost_breakdown.module_costs = original_costs["module"]
        scenario.cost_breakdown.inverter_costs = original_costs["inverter"]
        scenario.cost_breakdown.bos_costs = original_costs["bos"]
        scenario.cost_breakdown.labor_costs = original_costs["labor"]

        sensitivity["capex"] = capex_sensitivity

        # Production sensitivity (-20% to +20%)
        production_sensitivity = {}
        original_production = scenario.estimated_annual_production

        for variation in [-0.2, -0.1, 0, 0.1, 0.2]:
            scenario.estimated_annual_production = original_production * (1 + variation)
            metrics = self._calculate_economic_metrics(
                system, scenario, base_electricity_rate, incentives
            )
            production_sensitivity[f"{variation*100:+.0f}%"] = metrics.npv - base_npv

        # Restore original production
        scenario.estimated_annual_production = original_production

        sensitivity["production"] = production_sensitivity

        return sensitivity

    def _calculate_break_even(
        self,
        system: PVSystem,
        scenario: RepowerScenario,
        metrics: EconomicMetrics,
    ) -> Dict[str, float]:
        """Calculate break-even conditions."""
        break_even = {}

        total_capex = scenario.cost_breakdown.calculate_total()
        annual_production = scenario.estimated_annual_production

        # Break-even electricity rate (LCOE)
        break_even["electricity_rate_$/kwh"] = round(metrics.lcoe, 4)

        # Break-even production (to achieve target ROI)
        target_roi = self.config.min_roi_threshold
        required_revenue = total_capex * (1 + target_roi) + (
            metrics.annual_opex * self.config.analysis_period
        )
        required_annual_revenue = required_revenue / self.config.analysis_period
        break_even_production = required_annual_revenue / self.config.electricity_rate
        break_even["production_kwh"] = round(break_even_production, 0)

        # Break-even payback period
        break_even["payback_years"] = round(metrics.payback_period, 2)

        return break_even

    def _calculate_viability_score(
        self,
        scenarios: List[RepowerScenario],
        best_scenario: Optional[RepowerScenario],
    ) -> float:
        """Calculate overall viability score."""
        if not best_scenario or not best_scenario.economic_metrics:
            return 0.0

        score = 0.0
        metrics = best_scenario.economic_metrics

        # NPV component (0-30 points)
        if metrics.npv > 0:
            # Normalize to CAPEX
            capex = best_scenario.cost_breakdown.calculate_total()
            npv_ratio = metrics.npv / capex if capex > 0 else 0
            score += min(30, npv_ratio * 100)

        # ROI component (0-25 points)
        if metrics.roi >= self.config.min_roi_threshold:
            roi_score = min(25, (metrics.roi / self.config.min_roi_threshold) * 12.5)
            score += roi_score

        # Payback period component (0-25 points)
        if metrics.payback_period <= self.config.max_payback_period:
            payback_score = 25 * (
                1 - (metrics.payback_period / self.config.max_payback_period)
            )
            score += payback_score

        # IRR component (0-20 points)
        if metrics.irr > self.config.discount_rate:
            irr_score = min(20, ((metrics.irr - self.config.discount_rate) / self.config.discount_rate) * 20)
            score += irr_score

        return min(100, score)

    def _suggest_financing_options(
        self, scenario: Optional[RepowerScenario]
    ) -> List[Dict[str, str]]:
        """Suggest financing options based on project scale."""
        if not scenario:
            return []

        options = []
        capex = scenario.cost_breakdown.calculate_total()

        # Cash purchase
        options.append(
            {
                "type": "Cash Purchase",
                "description": f"Direct purchase of ${capex:,.0f} system",
                "pros": "Maximum long-term savings, full ownership, all incentives",
                "cons": "High upfront cost, opportunity cost of capital",
            }
        )

        # Loan financing
        options.append(
            {
                "type": "Loan Financing",
                "description": "5-15 year loan with typical rates 4-8%",
                "pros": "Preserve capital, potential tax benefits, own system",
                "cons": "Interest costs, monthly payments, credit requirements",
            }
        )

        # PPA/lease (for larger systems)
        if capex > 100000:
            options.append(
                {
                    "type": "Power Purchase Agreement (PPA)",
                    "description": "Third party owns system, purchase power at fixed rate",
                    "pros": "No upfront cost, predictable rates, O&M included",
                    "cons": "Don't own system, some incentives go to owner, long-term contract",
                }
            )

        # PACE financing (for commercial)
        if capex > 50000:
            options.append(
                {
                    "type": "PACE Financing",
                    "description": "Property Assessed Clean Energy long-term financing",
                    "pros": "Low rates, long terms, transferable with property",
                    "cons": "Limited availability, property requirements, assessment lien",
                }
            )

        return options
