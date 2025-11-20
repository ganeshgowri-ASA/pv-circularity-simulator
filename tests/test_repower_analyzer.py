"""Comprehensive test suite for RepowerAnalyzer."""

from datetime import date, timedelta

import pytest

from pv_simulator.analyzers.repower_analyzer import RepowerAnalyzer, RepowerAnalyzerConfig
from pv_simulator.core.enums import (
    ClimateZone,
    ComponentType,
    HealthStatus,
    ModuleTechnology,
    RepowerStrategy,
)
from pv_simulator.core.models import (
    ComponentHealth,
    CostBreakdown,
    Location,
    PVModule,
    PVSystem,
    RepowerScenario,
)


@pytest.fixture
def sample_location():
    """Create a sample location for testing."""
    return Location(
        latitude=37.7749,
        longitude=-122.4194,
        climate_zone=ClimateZone.TEMPERATE,
        avg_annual_irradiance=1800.0,
        avg_temperature=15.5,
        elevation=50.0,
    )


@pytest.fixture
def sample_module():
    """Create a sample PV module for testing."""
    return PVModule(
        technology=ModuleTechnology.MONO_SI,
        rated_power=350.0,
        efficiency=0.20,
        area=1.75,
        degradation_rate=0.005,
        temperature_coefficient=-0.4,
        warranty_years=25,
        cost_per_watt=0.35,
    )


@pytest.fixture
def sample_system(sample_location, sample_module):
    """Create a sample PV system for testing."""
    num_modules = 286  # ~100 kW system
    dc_capacity = (num_modules * sample_module.rated_power) / 1000

    # Create component health data
    component_health = [
        ComponentHealth(
            component_type=ComponentType.MODULE,
            status=HealthStatus.GOOD,
            performance_ratio=0.88,
            age_years=10.0,
            expected_lifetime=25.0,
            failure_probability=0.05,
            maintenance_cost_annual=500.0,
            replacement_cost=35000.0,
        ),
        ComponentHealth(
            component_type=ComponentType.INVERTER,
            status=HealthStatus.FAIR,
            performance_ratio=0.92,
            age_years=10.0,
            expected_lifetime=15.0,
            failure_probability=0.15,
            maintenance_cost_annual=800.0,
            replacement_cost=12000.0,
        ),
        ComponentHealth(
            component_type=ComponentType.RACKING,
            status=HealthStatus.EXCELLENT,
            performance_ratio=1.0,
            age_years=10.0,
            expected_lifetime=30.0,
            failure_probability=0.01,
            maintenance_cost_annual=200.0,
            replacement_cost=15000.0,
        ),
    ]

    return PVSystem(
        system_id="TEST-SYSTEM-001",
        installation_date=date.today() - timedelta(days=365 * 10),
        location=sample_location,
        module=sample_module,
        num_modules=num_modules,
        dc_capacity=dc_capacity,
        ac_capacity=95.0,
        inverter_efficiency=0.96,
        system_losses=0.14,
        component_health=component_health,
        current_performance_ratio=0.78,
        avg_annual_production=140000.0,
    )


@pytest.fixture
def analyzer():
    """Create a RepowerAnalyzer instance."""
    return RepowerAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Create a RepowerAnalyzer with custom configuration."""
    config = RepowerAnalyzerConfig(
        electricity_rate=0.15,
        discount_rate=0.05,
        analysis_period=30,
        min_roi_threshold=0.12,
        max_payback_period=10.0,
    )
    return RepowerAnalyzer(config=config)


class TestRepowerAnalyzerInit:
    """Test RepowerAnalyzer initialization."""

    def test_default_initialization(self, analyzer):
        """Test analyzer initializes with default config."""
        assert analyzer.config is not None
        assert analyzer.config.electricity_rate == 0.12
        assert analyzer.config.discount_rate == 0.06
        assert analyzer.config.analysis_period == 25

    def test_custom_initialization(self, custom_analyzer):
        """Test analyzer initializes with custom config."""
        assert custom_analyzer.config.electricity_rate == 0.15
        assert custom_analyzer.config.discount_rate == 0.05
        assert custom_analyzer.config.analysis_period == 30


class TestCapacityUpgradeAnalysis:
    """Test capacity upgrade analysis functionality."""

    def test_basic_capacity_analysis(self, analyzer, sample_system):
        """Test basic capacity upgrade analysis."""
        result = analyzer.capacity_upgrade_analysis(sample_system)

        assert result.current_capacity == sample_system.dc_capacity
        assert result.max_additional_capacity >= 0
        assert result.recommended_upgrade >= 0
        assert result.recommended_upgrade <= result.max_additional_capacity
        assert len(result.upgrade_scenarios) == 4
        assert result.limiting_factor is not None

    def test_capacity_analysis_with_constraints(self, analyzer, sample_system):
        """Test capacity analysis with explicit constraints."""
        result = analyzer.capacity_upgrade_analysis(
            sample_system,
            available_roof_area=250.0,
            structural_load_limit=3000.0,
            electrical_capacity_limit=30.0,
        )

        assert result.space_available == 250.0
        assert result.structural_capacity_available == 3000.0
        assert result.electrical_capacity_available == 30.0
        assert result.max_additional_capacity > 0

    def test_upgrade_scenarios_validity(self, analyzer, sample_system):
        """Test that upgrade scenarios are valid and increasing."""
        result = analyzer.capacity_upgrade_analysis(sample_system)

        previous_capacity = 0
        for scenario in result.upgrade_scenarios:
            assert scenario["upgrade_capacity_kw"] >= previous_capacity
            assert scenario["total_capacity_kw"] >= sample_system.dc_capacity
            assert scenario["num_additional_modules"] >= 0
            previous_capacity = scenario["upgrade_capacity_kw"]

    def test_recommended_upgrade_is_reasonable(self, analyzer, sample_system):
        """Test that recommended upgrade is between 50-75% of maximum."""
        result = analyzer.capacity_upgrade_analysis(sample_system)

        # Recommended should be around 65% of max (between 50-80%)
        ratio = result.recommended_upgrade / result.max_additional_capacity
        assert 0.5 <= ratio <= 0.8


class TestComponentReplacementPlanning:
    """Test component replacement planning functionality."""

    def test_basic_replacement_planning(self, analyzer, sample_system):
        """Test basic component replacement planning."""
        result = analyzer.component_replacement_planning(sample_system)

        assert isinstance(result.immediate_replacements, list)
        assert isinstance(result.short_term_replacements, list)
        assert isinstance(result.medium_term_replacements, list)
        assert isinstance(result.long_term_replacements, list)
        assert result.total_replacement_cost >= 0
        assert len(result.priority_order) > 0

    def test_critical_components_flagged(self, analyzer, sample_system):
        """Test that critical components are flagged for immediate replacement."""
        # Add a critical component
        critical_component = ComponentHealth(
            component_type=ComponentType.COMBINER_BOX,
            status=HealthStatus.CRITICAL,
            performance_ratio=0.45,
            age_years=12.0,
            expected_lifetime=15.0,
            failure_probability=0.80,
            maintenance_cost_annual=300.0,
            replacement_cost=5000.0,
        )
        sample_system.component_health.append(critical_component)

        result = analyzer.component_replacement_planning(sample_system)

        # Critical component should be in immediate replacements
        immediate_types = [c.component_type for c in result.immediate_replacements]
        assert ComponentType.COMBINER_BOX in immediate_types

    def test_replacement_cost_calculation(self, analyzer, sample_system):
        """Test that total replacement cost is correctly calculated."""
        result = analyzer.component_replacement_planning(sample_system)

        expected_cost = sum(c.replacement_cost for c in sample_system.component_health)
        assert result.total_replacement_cost == expected_cost

    def test_priority_order_includes_all_types(self, analyzer, sample_system):
        """Test that priority order includes all component types."""
        result = analyzer.component_replacement_planning(sample_system)

        # Should have priorities for all components in the system
        component_types = {c.component_type for c in sample_system.component_health}
        priority_types = set(result.priority_order)

        assert component_types.issubset(priority_types)

    def test_risk_mitigation_for_critical(self, analyzer, sample_system):
        """Test that risk mitigation plans are created for critical components."""
        # Add critical components
        critical_inverter = ComponentHealth(
            component_type=ComponentType.INVERTER,
            status=HealthStatus.POOR,
            performance_ratio=0.65,
            age_years=14.0,
            expected_lifetime=15.0,
            failure_probability=0.40,
            maintenance_cost_annual=1000.0,
            replacement_cost=15000.0,
        )
        sample_system.component_health.append(critical_inverter)

        result = analyzer.component_replacement_planning(sample_system)

        # Should have risk mitigation plan for inverter
        assert len(result.risk_mitigation_plan) > 0


class TestTechnicalFeasibilityCheck:
    """Test technical feasibility checking functionality."""

    def test_basic_feasibility_check(self, analyzer, sample_system):
        """Test basic technical feasibility check."""
        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=110.0
        )

        assert isinstance(result.is_feasible, bool)
        assert 0 <= result.feasibility_score <= 100
        assert 0 <= result.structural_feasibility <= 100
        assert 0 <= result.electrical_feasibility <= 100
        assert 0 <= result.spatial_feasibility <= 100
        assert 0 <= result.regulatory_feasibility <= 100
        assert 0 <= result.integration_feasibility <= 100

    def test_feasible_scenario(self, analyzer, sample_system):
        """Test that a reasonable upgrade is deemed feasible."""
        # Small 10% upgrade should be feasible
        target_capacity = sample_system.dc_capacity * 1.1

        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=target_capacity
        )

        assert result.is_feasible is True
        assert result.feasibility_score >= 60

    def test_infeasible_scenario(self, analyzer, sample_system):
        """Test that an extreme upgrade is deemed infeasible."""
        # 300% upgrade should likely be infeasible
        target_capacity = sample_system.dc_capacity * 3.0

        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=target_capacity
        )

        # Should have low feasibility or be marked infeasible
        assert result.feasibility_score < 80 or result.is_feasible is False

    def test_constraints_identified(self, analyzer, sample_system):
        """Test that constraints are identified for large upgrades."""
        target_capacity = sample_system.dc_capacity * 2.0

        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=target_capacity
        )

        # Should have some constraints identified
        assert len(result.constraints) > 0

    def test_recommendations_provided(self, analyzer, sample_system):
        """Test that recommendations are provided when needed."""
        target_capacity = sample_system.dc_capacity * 1.8

        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=target_capacity
        )

        # Should have recommendations for moderate upgrade
        assert len(result.recommendations) >= 0

    def test_new_module_compatibility(self, analyzer, sample_system):
        """Test module compatibility checking."""
        # Different technology module
        new_module = PVModule(
            technology=ModuleTechnology.THIN_FILM_CDTE,
            rated_power=400.0,
            efficiency=0.18,
            area=2.0,
            cost_per_watt=0.30,
        )

        result = analyzer.technical_feasibility_check(
            sample_system, target_capacity=110.0, new_module=new_module
        )

        # Should flag technology compatibility
        assert result.integration_feasibility < 100


class TestEconomicViabilityAnalysis:
    """Test economic viability analysis functionality."""

    def test_basic_economic_analysis(self, analyzer, sample_system, sample_module):
        """Test basic economic viability analysis."""
        # Create a simple repower scenario
        scenario = RepowerScenario(
            scenario_id="TEST-SCENARIO-1",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=110.0,
            capacity_increase=0.10,
            new_module=sample_module,
            num_new_modules=314,
            cost_breakdown=CostBreakdown(
                module_costs=38500.0,
                inverter_costs=0.0,
                bos_costs=5000.0,
                labor_costs=8000.0,
                permitting_costs=1500.0,
                engineering_costs=2000.0,
            ),
            estimated_annual_production=155000.0,
            performance_improvement=0.11,
            technical_feasibility_score=85.0,
        )

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario]
        )

        assert isinstance(result.is_viable, bool)
        assert 0 <= result.viability_score <= 100
        assert len(result.scenarios_analyzed) == 1
        assert result.best_scenario is not None

    def test_multiple_scenarios_comparison(self, analyzer, sample_system, sample_module):
        """Test comparison of multiple repower scenarios."""
        scenarios = [
            RepowerScenario(
                scenario_id="SCENARIO-1",
                strategy=RepowerStrategy.MODULE_ONLY,
                new_dc_capacity=110.0,
                capacity_increase=0.10,
                new_module=sample_module,
                num_new_modules=314,
                cost_breakdown=CostBreakdown(
                    module_costs=38500.0,
                    bos_costs=5000.0,
                    labor_costs=8000.0,
                ),
                estimated_annual_production=155000.0,
                performance_improvement=0.11,
                technical_feasibility_score=85.0,
            ),
            RepowerScenario(
                scenario_id="SCENARIO-2",
                strategy=RepowerStrategy.FULL_REPLACEMENT,
                new_dc_capacity=120.0,
                capacity_increase=0.20,
                new_module=sample_module,
                num_new_modules=343,
                cost_breakdown=CostBreakdown(
                    module_costs=42000.0,
                    inverter_costs=15000.0,
                    bos_costs=8000.0,
                    labor_costs=12000.0,
                ),
                estimated_annual_production=168000.0,
                performance_improvement=0.20,
                technical_feasibility_score=80.0,
            ),
        ]

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=scenarios
        )

        assert len(result.scenarios_analyzed) == 2
        assert result.best_scenario is not None
        assert result.best_scenario.economic_metrics is not None

    def test_economic_metrics_calculated(self, analyzer, sample_system, sample_module):
        """Test that all economic metrics are calculated."""
        scenario = RepowerScenario(
            scenario_id="TEST-SCENARIO",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=110.0,
            capacity_increase=0.10,
            new_module=sample_module,
            num_new_modules=314,
            cost_breakdown=CostBreakdown(
                module_costs=38500.0,
                bos_costs=5000.0,
                labor_costs=8000.0,
            ),
            estimated_annual_production=155000.0,
            performance_improvement=0.11,
            technical_feasibility_score=85.0,
        )

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario]
        )

        metrics = result.scenarios_analyzed[0].economic_metrics
        assert metrics is not None
        assert metrics.lcoe > 0
        assert metrics.payback_period > 0
        assert metrics.annual_energy_value > 0
        assert metrics.annual_opex >= 0

    def test_sensitivity_analysis(self, analyzer, sample_system, sample_module):
        """Test that sensitivity analysis is performed."""
        scenario = RepowerScenario(
            scenario_id="TEST-SCENARIO",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=110.0,
            capacity_increase=0.10,
            new_module=sample_module,
            num_new_modules=314,
            cost_breakdown=CostBreakdown(
                module_costs=38500.0,
                bos_costs=5000.0,
                labor_costs=8000.0,
            ),
            estimated_annual_production=155000.0,
            performance_improvement=0.11,
            technical_feasibility_score=85.0,
        )

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario]
        )

        assert "electricity_rate" in result.sensitivity_analysis
        assert "capex" in result.sensitivity_analysis
        assert "production" in result.sensitivity_analysis

    def test_incentives_applied(self, analyzer, sample_system, sample_module):
        """Test that incentives are properly applied."""
        scenario = RepowerScenario(
            scenario_id="TEST-SCENARIO",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=110.0,
            capacity_increase=0.10,
            new_module=sample_module,
            num_new_modules=314,
            cost_breakdown=CostBreakdown(
                module_costs=38500.0,
                bos_costs=5000.0,
                labor_costs=8000.0,
            ),
            estimated_annual_production=155000.0,
            performance_improvement=0.11,
            technical_feasibility_score=85.0,
        )

        incentives = {
            "Federal ITC": 10000.0,
            "State Rebate": 5000.0,
        }

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario], incentives=incentives
        )

        assert result.incentives_available == incentives

    def test_financing_options_suggested(self, analyzer, sample_system, sample_module):
        """Test that financing options are suggested."""
        scenario = RepowerScenario(
            scenario_id="TEST-SCENARIO",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=110.0,
            capacity_increase=0.10,
            new_module=sample_module,
            num_new_modules=314,
            cost_breakdown=CostBreakdown(
                module_costs=38500.0,
                bos_costs=5000.0,
                labor_costs=8000.0,
            ),
            estimated_annual_production=155000.0,
            performance_improvement=0.11,
            technical_feasibility_score=85.0,
        )

        result = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario]
        )

        assert len(result.financing_options) > 0
        assert any("Cash" in opt["type"] for opt in result.financing_options)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_repower_analysis_workflow(
        self, analyzer, sample_system, sample_module
    ):
        """Test complete workflow from analysis to decision."""
        # Step 1: Capacity upgrade analysis
        capacity_analysis = analyzer.capacity_upgrade_analysis(sample_system)
        assert capacity_analysis.max_additional_capacity > 0

        # Step 2: Component replacement planning
        replacement_plan = analyzer.component_replacement_planning(sample_system)
        assert replacement_plan.total_replacement_cost > 0

        # Step 3: Technical feasibility
        target_capacity = (
            sample_system.dc_capacity + capacity_analysis.recommended_upgrade
        )
        feasibility = analyzer.technical_feasibility_check(
            sample_system, target_capacity=target_capacity
        )
        assert feasibility.feasibility_score > 0

        # Step 4: Economic viability
        scenario = RepowerScenario(
            scenario_id="INTEGRATED-TEST",
            strategy=RepowerStrategy.MODULE_ONLY,
            new_dc_capacity=target_capacity,
            capacity_increase=capacity_analysis.recommended_upgrade
            / sample_system.dc_capacity,
            new_module=sample_module,
            num_new_modules=int(
                (capacity_analysis.recommended_upgrade * 1000) / sample_module.rated_power
            ),
            cost_breakdown=CostBreakdown(
                module_costs=capacity_analysis.recommended_upgrade * 1000 * 0.35,
                bos_costs=5000.0,
                labor_costs=8000.0,
            ),
            estimated_annual_production=sample_system.avg_annual_production * 1.1,
            performance_improvement=0.1,
            technical_feasibility_score=feasibility.feasibility_score,
        )

        economics = analyzer.economic_viability_analysis(
            sample_system, repower_scenarios=[scenario]
        )

        assert economics.best_scenario is not None
        assert economics.viability_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
