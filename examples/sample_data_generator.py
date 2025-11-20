"""
Sample data generator for testing the Circularity Dashboard.

This module provides functions to generate realistic sample data for testing
and demonstrating the PV Circularity Assessment Dashboard functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pv_circularity_simulator.core.data_models import (
    CircularityMetrics,
    MaterialFlow,
    ReuseMetrics,
    RepairMetrics,
    RecyclingMetrics,
    PolicyCompliance,
    ImpactScorecard,
    MaterialType,
    ProcessStage,
    ComplianceStatus,
)


def generate_sample_material_flows(num_flows: int = 20) -> list[MaterialFlow]:
    """
    Generate sample material flow data.

    Args:
        num_flows: Number of material flow records to generate

    Returns:
        List of MaterialFlow objects with realistic sample data
    """
    flows = []
    materials = list(MaterialType)
    stages = list(ProcessStage)
    locations = ["Germany", "China", "USA", "Japan", "India"]

    base_time = datetime.now() - timedelta(days=365)

    for i in range(num_flows):
        material = random.choice(materials)
        stage = random.choice(stages)

        # Generate realistic masses based on material type
        if material == MaterialType.GLASS:
            input_mass = random.uniform(10000, 50000)
        elif material == MaterialType.SILICON:
            input_mass = random.uniform(5000, 15000)
        elif material == MaterialType.ALUMINUM:
            input_mass = random.uniform(3000, 10000)
        else:
            input_mass = random.uniform(500, 5000)

        # Efficiency varies by stage
        if stage == ProcessStage.RECYCLING:
            efficiency = random.uniform(0.75, 0.95)
        elif stage == ProcessStage.MANUFACTURING:
            efficiency = random.uniform(0.85, 0.98)
        else:
            efficiency = random.uniform(0.90, 0.99)

        output_mass = input_mass * efficiency
        loss_mass = input_mass - output_mass

        flow = MaterialFlow(
            material_type=material,
            stage=stage,
            input_mass_kg=input_mass,
            output_mass_kg=output_mass,
            loss_mass_kg=loss_mass,
            timestamp=base_time + timedelta(days=i * 18),
            location=random.choice(locations),
            metadata={
                "batch_id": f"BATCH-{i:04d}",
                "facility": f"Facility-{random.randint(1, 5)}"
            }
        )
        flows.append(flow)

    return flows


def generate_sample_reuse_metrics() -> ReuseMetrics:
    """
    Generate sample reuse metrics.

    Returns:
        ReuseMetrics object with realistic sample data
    """
    total_collected = random.randint(5000, 10000)
    suitable = int(total_collected * random.uniform(0.65, 0.85))
    reused = int(suitable * random.uniform(0.75, 0.95))

    return ReuseMetrics(
        total_modules_collected=total_collected,
        modules_suitable_for_reuse=suitable,
        modules_reused=reused,
        avg_residual_capacity_pct=random.uniform(75, 90),
        avg_extension_years=random.uniform(8, 15),
        cost_savings_usd=reused * random.uniform(100, 250),
        co2_avoided_kg=reused * random.uniform(150, 300),
        quality_grade_distribution={
            "Grade A (>90%)": int(reused * 0.25),
            "Grade B (80-90%)": int(reused * 0.40),
            "Grade C (70-80%)": int(reused * 0.25),
            "Grade D (<70%)": int(reused * 0.10)
        }
    )


def generate_sample_repair_metrics() -> RepairMetrics:
    """
    Generate sample repair metrics.

    Returns:
        RepairMetrics object with realistic sample data
    """
    total_assessed = random.randint(2000, 5000)
    repairable = int(total_assessed * random.uniform(0.50, 0.75))
    repaired = int(repairable * random.uniform(0.80, 0.95))

    return RepairMetrics(
        total_modules_assessed=total_assessed,
        modules_repairable=repairable,
        modules_repaired=repaired,
        avg_repair_cost_usd=random.uniform(50, 150),
        avg_performance_recovery_pct=random.uniform(85, 98),
        common_failure_modes={
            "Junction Box Issues": random.randint(200, 500),
            "Cell Cracks": random.randint(150, 400),
            "Bypass Diode Failure": random.randint(100, 300),
            "Frame Damage": random.randint(80, 250),
            "Backsheet Delamination": random.randint(60, 200),
            "Connector Problems": random.randint(50, 150)
        },
        repair_time_hours=random.uniform(2, 8),
        warranty_extension_months=random.randint(12, 36)
    )


def generate_sample_recycling_metrics() -> RecyclingMetrics:
    """
    Generate sample recycling metrics.

    Returns:
        RecyclingMetrics object with realistic sample data
    """
    total_processed = random.uniform(50000, 150000)
    recovery_rate = random.uniform(0.85, 0.95)

    return RecyclingMetrics(
        total_mass_processed_kg=total_processed,
        material_recovery_rates={
            "glass": random.uniform(95, 99),
            "aluminum": random.uniform(90, 98),
            "silicon": random.uniform(80, 92),
            "copper": random.uniform(85, 95),
            "silver": random.uniform(90, 98),
            "polymer": random.uniform(50, 75)
        },
        total_mass_recovered_kg=total_processed * recovery_rate,
        recycling_cost_per_kg=random.uniform(0.15, 0.35),
        revenue_per_kg=random.uniform(0.25, 0.55),
        energy_consumption_kwh=total_processed * random.uniform(0.3, 0.6),
        water_usage_liters=total_processed * random.uniform(2, 5),
        hazardous_waste_kg=total_processed * random.uniform(0.02, 0.05)
    )


def generate_sample_policy_compliance() -> list[PolicyCompliance]:
    """
    Generate sample policy compliance records.

    Returns:
        List of PolicyCompliance objects representing various regulations
    """
    policies = [
        PolicyCompliance(
            policy_name="EU WEEE Directive",
            jurisdiction="European Union",
            compliance_status=ComplianceStatus.COMPLIANT,
            required_collection_rate_pct=85.0,
            actual_collection_rate_pct=87.5,
            required_recovery_rate_pct=80.0,
            actual_recovery_rate_pct=85.3,
            penalties_usd=0.0,
            compliance_deadline=datetime.now() + timedelta(days=180),
            notes="On track to meet all requirements for 2025"
        ),
        PolicyCompliance(
            policy_name="China PV Recycling Standard",
            jurisdiction="China",
            compliance_status=ComplianceStatus.PARTIALLY_COMPLIANT,
            required_collection_rate_pct=70.0,
            actual_collection_rate_pct=65.8,
            required_recovery_rate_pct=75.0,
            actual_recovery_rate_pct=78.2,
            penalties_usd=15000.0,
            compliance_deadline=datetime.now() + timedelta(days=90),
            notes="Collection rate below target; recovery exceeds requirement"
        ),
        PolicyCompliance(
            policy_name="US EPA Guidelines",
            jurisdiction="United States",
            compliance_status=ComplianceStatus.COMPLIANT,
            required_collection_rate_pct=65.0,
            actual_collection_rate_pct=72.1,
            required_recovery_rate_pct=70.0,
            actual_recovery_rate_pct=76.8,
            penalties_usd=0.0,
            compliance_deadline=datetime.now() + timedelta(days=365),
            notes="Exceeding federal requirements"
        ),
        PolicyCompliance(
            policy_name="Japan Resource Circulation Law",
            jurisdiction="Japan",
            compliance_status=ComplianceStatus.COMPLIANT,
            required_collection_rate_pct=80.0,
            actual_collection_rate_pct=83.5,
            required_recovery_rate_pct=85.0,
            actual_recovery_rate_pct=88.7,
            penalties_usd=0.0,
            compliance_deadline=datetime.now() + timedelta(days=270),
            notes="Strong performance across all metrics"
        )
    ]

    return policies


def generate_sample_impact_scorecards() -> list[ImpactScorecard]:
    """
    Generate sample impact scorecard data.

    Returns:
        List of ImpactScorecard objects for various impact categories
    """
    scorecards = [
        ImpactScorecard(
            category="Carbon Footprint",
            baseline_value=1250000.0,
            circular_value=875000.0,
            unit="kg CO₂e",
            target_value=750000.0,
            target_year=2030,
            sub_metrics={
                "Manufacturing emissions": 350000.0,
                "Transportation": 125000.0,
                "End-of-life processing": 400000.0
            },
            data_quality=4
        ),
        ImpactScorecard(
            category="Virgin Material Consumption",
            baseline_value=85000.0,
            circular_value=42000.0,
            unit="kg",
            target_value=30000.0,
            target_year=2030,
            sub_metrics={
                "Silicon": 15000.0,
                "Glass": 18000.0,
                "Metals": 7000.0,
                "Polymers": 2000.0
            },
            data_quality=5
        ),
        ImpactScorecard(
            category="Water Consumption",
            baseline_value=450000.0,
            circular_value=280000.0,
            unit="liters",
            target_value=200000.0,
            target_year=2030,
            sub_metrics={
                "Manufacturing": 150000.0,
                "Recycling": 130000.0
            },
            data_quality=4
        ),
        ImpactScorecard(
            category="Waste Generation",
            baseline_value=35000.0,
            circular_value=8500.0,
            unit="kg",
            target_value=5000.0,
            target_year=2030,
            sub_metrics={
                "Manufacturing waste": 3500.0,
                "Non-recyclable EoL": 4000.0,
                "Hazardous waste": 1000.0
            },
            data_quality=4
        ),
        ImpactScorecard(
            category="Energy Consumption",
            baseline_value=2800000.0,
            circular_value=1950000.0,
            unit="kWh",
            target_value=1500000.0,
            target_year=2030,
            sub_metrics={
                "Manufacturing": 850000.0,
                "Recycling": 600000.0,
                "Transportation": 500000.0
            },
            data_quality=5
        ),
        ImpactScorecard(
            category="Economic Value Retained",
            baseline_value=5000000.0,
            circular_value=8750000.0,
            unit="USD",
            target_value=10000000.0,
            target_year=2030,
            sub_metrics={
                "Material recovery": 3500000.0,
                "Reuse revenue": 2750000.0,
                "Repair services": 2500000.0
            },
            data_quality=4
        )
    ]

    return scorecards


def generate_sample_circularity_data(
    assessment_id: str = "SAMPLE-2025-001"
) -> CircularityMetrics:
    """
    Generate a complete sample circularity assessment.

    Args:
        assessment_id: Unique identifier for this assessment

    Returns:
        CircularityMetrics object with comprehensive sample data

    Example:
        >>> metrics = generate_sample_circularity_data()
        >>> print(f"Circularity Index: {metrics.circularity_index}")
        Circularity Index: 73.5
    """
    # Calculate circularity index based on metrics
    reuse_metrics = generate_sample_reuse_metrics()
    repair_metrics = generate_sample_repair_metrics()
    recycling_metrics = generate_sample_recycling_metrics()

    # Weighted circularity index calculation
    circularity_index = (
        reuse_metrics.reuse_rate * 0.35 +
        repair_metrics.repair_success_rate * 0.25 +
        recycling_metrics.recovery_efficiency * 0.40
    )

    metrics = CircularityMetrics(
        assessment_id=assessment_id,
        timestamp=datetime.now(),
        material_flows=generate_sample_material_flows(25),
        reuse_metrics=reuse_metrics,
        repair_metrics=repair_metrics,
        recycling_metrics=recycling_metrics,
        policy_compliance=generate_sample_policy_compliance(),
        impact_scorecards=generate_sample_impact_scorecards(),
        circularity_index=circularity_index,
        metadata={
            "assessment_type": "annual_review",
            "scope": "global_operations",
            "data_sources": ["ERP", "MES", "recycling_partners"],
            "validated_by": "sustainability_team",
            "notes": "Automated sample data for demonstration purposes"
        }
    )

    return metrics


if __name__ == "__main__":
    """Generate and display sample data for testing."""
    print("Generating sample circularity data...")
    metrics = generate_sample_circularity_data()

    print(f"\n✅ Generated Sample Assessment: {metrics.assessment_id}")
    print(f"   Timestamp: {metrics.timestamp}")
    print(f"   Circularity Index: {metrics.circularity_index:.1f}/100")
    print(f"\n📊 Data Summary:")
    print(f"   - Material Flows: {len(metrics.material_flows)}")
    print(f"   - Reuse Rate: {metrics.reuse_metrics.reuse_rate:.1f}%")
    print(f"   - Repair Success: {metrics.repair_metrics.repair_success_rate:.1f}%")
    print(f"   - Recovery Efficiency: {metrics.recycling_metrics.recovery_efficiency:.1f}%")
    print(f"   - Policy Compliance Records: {len(metrics.policy_compliance)}")
    print(f"   - Impact Scorecards: {len(metrics.impact_scorecards)}")
    print("\n✨ Sample data generated successfully!")
