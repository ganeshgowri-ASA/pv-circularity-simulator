"""Integration tests for circularity system."""

import pytest
from src.circularity import (
    MaterialRecoveryCalculator,
    ReuseAnalyzer,
    RepairOptimizer,
    RecyclingEconomics,
    LCAAnalyzer,
    CircularityUI,
)
from src.circularity.material_recovery import ModuleComposition
from src.circularity.reuse_analyzer import ModuleTestResults, ModuleCondition, DefectType
from src.circularity.repair_optimizer import Defect, DefectSeverity
from src.circularity.recycling_economics import MaterialPrices


def test_end_to_end_circularity_analysis():
    """Test complete circularity analysis workflow."""
    # 1. Define module
    module = ModuleComposition(
        glass=15.0,
        aluminum=2.5,
        silicon=0.5,
        silver=0.005,
        copper=0.2,
        eva_polymer=1.0,
        backsheet=0.5,
        junction_box=0.3
    )

    # 2. Material recovery analysis
    recovery_calc = MaterialRecoveryCalculator()
    recovery_results = recovery_calc.full_recovery_analysis(
        composition=module,
        num_modules=100,
        transport_distance_km=150.0
    )
    assert recovery_results["overall_recovery_rate"] > 0.7

    # 3. Recycling economics
    economics = RecyclingEconomics()
    roi_analysis = economics.recycling_roi(
        num_modules=100,
        avg_module_weight_kg=module.total_mass,
        recycling_cost_per_module=15.0,
        recovered_materials={
            "aluminum": recovery_results["metal_recovery"]["aluminum_kg"],
            "glass": recovery_results["glass_recovery"]["recovered_glass_kg"],
            "silicon": recovery_results["silicon_recovery"]["recovered_silicon_kg"],
            "silver": recovery_results["metal_recovery"]["silver_kg"],
            "copper": recovery_results["metal_recovery"]["copper_kg"],
        }
    )
    assert roi_analysis.total_revenue > 0

    # 4. Environmental analysis
    lca = LCAAnalyzer()
    carbon_footprint = lca.carbon_footprint(
        module_power_w=400,
        module_weight_kg=module.total_mass
    )
    assert carbon_footprint.total_kg_co2eq > 0

    # 5. Circular economy score
    ui = CircularityUI()
    ce_score = ui.circular_economy_score(
        material_circularity_index=0.75,
        recovery_rate=recovery_results["overall_recovery_rate"],
        reuse_rate=0.15,
        lifetime_extension_factor=1.2,
        carbon_footprint_kg=carbon_footprint.total_kg_co2eq,
        roi_percent=roi_analysis.roi_percent
    )
    assert 0 <= ce_score.overall_score <= 100
    assert ce_score.rating in ["A+", "A", "B", "C", "D", "F"]


def test_reuse_to_recycling_pathway():
    """Test pathway from reuse assessment to recycling."""
    # Test module that fails reuse criteria
    test_results = ModuleTestResults(
        visual_inspection_passed=True,
        electrical_test_passed=True,
        insulation_test_passed=True,
        current_power_w=280,  # 70% capacity retention
        rated_power_w=400,
        voltage_v=30.0,
        current_a=9.3,
        fill_factor=0.75,
        insulation_resistance_mohm=45.0,
        defects=[DefectType.DISCOLORATION],
        condition=ModuleCondition.FAIR
    )

    # Assess for reuse
    reuse_analyzer = ReuseAnalyzer(min_capacity_retention=0.80)
    eligibility = reuse_analyzer.module_testing(test_results, age_years=12)

    # Should not be eligible for reuse (70% < 80% threshold)
    assert not eligibility.is_eligible

    # Route to recycling
    module = ModuleComposition(
        glass=15.0, aluminum=2.5, silicon=0.5,
        silver=0.005, copper=0.2, eva_polymer=1.0,
        backsheet=0.5, junction_box=0.3
    )

    recovery = MaterialRecoveryCalculator()
    recycling_results = recovery.full_recovery_analysis(module, num_modules=1)

    assert recycling_results["overall_recovery_rate"] > 0


def test_repair_vs_recycle_decision():
    """Test decision logic for repair vs recycling."""
    # Define defects
    defects = [
        Defect(
            defect_id="D001",
            defect_type="junction_box_failure",
            severity=DefectSeverity.MEDIUM,
            location="Module A1",
            power_loss_w=15.0,
            safety_risk=False,
            progression_rate=2.0
        )
    ]

    # Repair optimization
    optimizer = RepairOptimizer()

    # Prioritize defects
    prioritized = optimizer.defect_prioritization(
        defects=defects,
        system_size_kw=10.0,
        age_years=8.0
    )
    assert len(prioritized) == 1

    # If module is young and defect is minor, should recommend repair
    # If module is old or severely degraded, should recommend replace/recycle
    assert prioritized[0].recommended_action is not None


def test_lca_with_recycling_benefit():
    """Test LCA analysis showing recycling benefits."""
    lca = LCAAnalyzer()

    # Calculate carbon footprint with recycling
    with_recycling = lca.carbon_footprint(
        module_power_w=400,
        module_weight_kg=20.0,
        recycling_at_eol=True
    )

    # Calculate without recycling
    without_recycling = lca.carbon_footprint(
        module_power_w=400,
        module_weight_kg=20.0,
        recycling_at_eol=False
    )

    # Recycling should reduce total carbon footprint
    assert with_recycling.total_kg_co2eq < without_recycling.total_kg_co2eq


def test_visualization_data_generation():
    """Test that visualization components generate valid data."""
    ui = CircularityUI()

    # Test material flow diagram
    fig = ui.material_flow_diagrams(
        input_materials={"glass": 15.0, "aluminum": 2.5, "silicon": 0.5},
        recovered_materials={"glass": 14.0, "aluminum": 2.4, "silicon": 0.4},
        waste_materials={"mixed": 1.2}
    )
    assert fig is not None
    assert fig.data is not None


def test_economic_environmental_correlation():
    """Test correlation between economic and environmental benefits."""
    economics = RecyclingEconomics()

    # Calculate environmental credits
    env_credits = economics.environmental_credits(
        num_modules=1000,
        avg_module_weight_kg=20.0,
        region="EU"
    )

    # Higher environmental benefits should correlate with economic value
    assert env_credits.carbon_credits_usd > 0
    assert env_credits.carbon_offset_tons > 0
    assert env_credits.total_environmental_value > 0
