"""
Data models for PV circularity metrics and assessment.

This module provides comprehensive data structures for tracking material flows,
circularity metrics, reuse/repair/recycling strategies, policy compliance,
and environmental impact scorecards.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class MaterialType(Enum):
    """Enumeration of PV material types."""
    SILICON = "silicon"
    GLASS = "glass"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    SILVER = "silver"
    POLYMER = "polymer"
    OTHER_METALS = "other_metals"


class ProcessStage(Enum):
    """Enumeration of lifecycle stages."""
    MANUFACTURING = "manufacturing"
    INSTALLATION = "installation"
    OPERATION = "operation"
    END_OF_LIFE = "end_of_life"
    RECYCLING = "recycling"
    REUSE = "reuse"
    REPAIR = "repair"


class ComplianceStatus(Enum):
    """Policy compliance status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class MaterialFlow:
    """
    Represents material flow through the PV lifecycle.

    Attributes:
        material_type: Type of material being tracked
        stage: Current lifecycle stage
        input_mass_kg: Input mass in kilograms
        output_mass_kg: Output mass in kilograms
        loss_mass_kg: Mass lost during processing
        timestamp: Time of measurement
        location: Geographic location identifier
        efficiency: Process efficiency (0-1)
        metadata: Additional contextual information
    """
    material_type: MaterialType
    stage: ProcessStage
    input_mass_kg: float
    output_mass_kg: float
    loss_mass_kg: float
    timestamp: datetime = field(default_factory=datetime.now)
    location: str = "Unknown"
    efficiency: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate efficiency after initialization."""
        if self.input_mass_kg > 0:
            self.efficiency = self.output_mass_kg / self.input_mass_kg


@dataclass
class ReuseMetrics:
    """
    Metrics for PV module reuse strategies.

    Attributes:
        total_modules_collected: Total modules collected for reuse
        modules_suitable_for_reuse: Modules meeting reuse criteria
        modules_reused: Modules successfully reused
        reuse_rate: Percentage of collected modules reused (0-100)
        avg_residual_capacity_pct: Average remaining power capacity (0-100)
        avg_extension_years: Average operational life extension in years
        cost_savings_usd: Cost savings compared to new modules
        co2_avoided_kg: CO2 emissions avoided through reuse
        quality_grade_distribution: Distribution of quality grades
    """
    total_modules_collected: int = 0
    modules_suitable_for_reuse: int = 0
    modules_reused: int = 0
    reuse_rate: float = 0.0
    avg_residual_capacity_pct: float = 0.0
    avg_extension_years: float = 0.0
    cost_savings_usd: float = 0.0
    co2_avoided_kg: float = 0.0
    quality_grade_distribution: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate reuse rate after initialization."""
        if self.total_modules_collected > 0:
            self.reuse_rate = (self.modules_reused / self.total_modules_collected) * 100


@dataclass
class RepairMetrics:
    """
    Metrics for PV module repair operations.

    Attributes:
        total_modules_assessed: Total modules assessed for repair
        modules_repairable: Modules determined to be repairable
        modules_repaired: Modules successfully repaired
        repair_success_rate: Success rate of repair attempts (0-100)
        avg_repair_cost_usd: Average cost per repair
        avg_performance_recovery_pct: Average performance recovery (0-100)
        common_failure_modes: Most common failure types
        repair_time_hours: Average repair time in hours
        warranty_extension_months: Average warranty extension period
    """
    total_modules_assessed: int = 0
    modules_repairable: int = 0
    modules_repaired: int = 0
    repair_success_rate: float = 0.0
    avg_repair_cost_usd: float = 0.0
    avg_performance_recovery_pct: float = 0.0
    common_failure_modes: Dict[str, int] = field(default_factory=dict)
    repair_time_hours: float = 0.0
    warranty_extension_months: int = 0

    def __post_init__(self):
        """Calculate repair success rate after initialization."""
        if self.modules_repairable > 0:
            self.repair_success_rate = (self.modules_repaired / self.modules_repairable) * 100


@dataclass
class RecyclingMetrics:
    """
    Metrics for PV module recycling operations.

    Attributes:
        total_mass_processed_kg: Total mass processed for recycling
        material_recovery_rates: Recovery rate by material type (0-100)
        total_mass_recovered_kg: Total mass of materials recovered
        recovery_efficiency: Overall recovery efficiency (0-100)
        recycling_cost_per_kg: Cost per kilogram recycled
        revenue_per_kg: Revenue per kilogram from recovered materials
        energy_consumption_kwh: Energy consumed in recycling process
        water_usage_liters: Water usage in recycling process
        hazardous_waste_kg: Hazardous waste generated
    """
    total_mass_processed_kg: float = 0.0
    material_recovery_rates: Dict[str, float] = field(default_factory=dict)
    total_mass_recovered_kg: float = 0.0
    recovery_efficiency: float = 0.0
    recycling_cost_per_kg: float = 0.0
    revenue_per_kg: float = 0.0
    energy_consumption_kwh: float = 0.0
    water_usage_liters: float = 0.0
    hazardous_waste_kg: float = 0.0

    def __post_init__(self):
        """Calculate recovery efficiency after initialization."""
        if self.total_mass_processed_kg > 0:
            self.recovery_efficiency = (self.total_mass_recovered_kg / self.total_mass_processed_kg) * 100


@dataclass
class PolicyCompliance:
    """
    Policy and regulatory compliance tracking.

    Attributes:
        policy_name: Name of the policy/regulation
        jurisdiction: Geographic jurisdiction (EU, US, China, etc.)
        compliance_status: Current compliance status
        required_collection_rate_pct: Required collection rate (0-100)
        actual_collection_rate_pct: Actual achieved collection rate (0-100)
        required_recovery_rate_pct: Required recovery rate (0-100)
        actual_recovery_rate_pct: Actual achieved recovery rate (0-100)
        penalties_usd: Penalties for non-compliance
        compliance_deadline: Deadline for compliance
        notes: Additional compliance notes
    """
    policy_name: str
    jurisdiction: str
    compliance_status: ComplianceStatus
    required_collection_rate_pct: float = 0.0
    actual_collection_rate_pct: float = 0.0
    required_recovery_rate_pct: float = 0.0
    actual_recovery_rate_pct: float = 0.0
    penalties_usd: float = 0.0
    compliance_deadline: Optional[datetime] = None
    notes: str = ""


@dataclass
class ImpactScorecard:
    """
    Environmental and economic impact scorecard.

    Attributes:
        category: Impact category name
        baseline_value: Baseline value (linear economy)
        circular_value: Value with circular strategies
        improvement_pct: Percentage improvement (0-100)
        unit: Unit of measurement
        target_value: Target value to achieve
        target_year: Target year for goal achievement
        sub_metrics: Detailed sub-metrics breakdown
        data_quality: Data quality indicator (1-5, 5=best)
    """
    category: str
    baseline_value: float
    circular_value: float
    improvement_pct: float = 0.0
    unit: str = ""
    target_value: Optional[float] = None
    target_year: Optional[int] = None
    sub_metrics: Dict[str, float] = field(default_factory=dict)
    data_quality: int = 3

    def __post_init__(self):
        """Calculate improvement percentage after initialization."""
        if self.baseline_value != 0:
            self.improvement_pct = ((self.baseline_value - self.circular_value) / abs(self.baseline_value)) * 100


@dataclass
class CircularityMetrics:
    """
    Comprehensive circularity assessment metrics.

    Attributes:
        assessment_id: Unique identifier for this assessment
        timestamp: Time of assessment
        material_flows: List of material flow records
        reuse_metrics: Reuse strategy metrics
        repair_metrics: Repair strategy metrics
        recycling_metrics: Recycling strategy metrics
        policy_compliance: List of policy compliance records
        impact_scorecards: List of impact scorecards
        circularity_index: Overall circularity index (0-100)
        metadata: Additional assessment metadata
    """
    assessment_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    material_flows: List[MaterialFlow] = field(default_factory=list)
    reuse_metrics: Optional[ReuseMetrics] = None
    repair_metrics: Optional[RepairMetrics] = None
    recycling_metrics: Optional[RecyclingMetrics] = None
    policy_compliance: List[PolicyCompliance] = field(default_factory=list)
    impact_scorecards: List[ImpactScorecard] = field(default_factory=list)
    circularity_index: float = 0.0
    metadata: Dict = field(default_factory=dict)
