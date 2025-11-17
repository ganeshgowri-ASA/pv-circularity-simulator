"""
Comprehensive tests for WindHybridIntegrator class.
"""

import pytest
import numpy as np
from datetime import datetime

from pv_simulator.integrators.hybrid_integrator import WindHybridIntegrator
from pv_simulator.core.models import (
    HybridSystemConfig,
    WindResourceData,
    TurbineSpecifications,
    CoordinationStrategy,
    OptimizationObjective,
)


class TestWindHybridIntegratorInitialization:
    """Test WindHybridIntegrator initialization and configuration."""

    def test_initialization_with_valid_config(self, sample_hybrid_config):
        """Test successful initialization with valid configuration."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        assert integrator.config == sample_hybrid_config
        assert integrator.metadata.integrator_id == sample_hybrid_config.system_id
        assert not integrator.is_initialized()
        assert integrator.wind_assessment is None
        assert integrator.turbine_performance is None

    def test_initialize_sets_initialized_flag(self, sample_hybrid_config):
        """Test that initialize() sets the initialized flag."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assert integrator.is_initialized()
        assert integrator.coordination_strategy is not None

    def test_validate_configuration_success(self, sample_hybrid_config):
        """Test successful configuration validation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        assert integrator.validate_configuration() is True

    def test_validate_configuration_capacity_mismatch(
        self, sample_hybrid_config, sample_turbine_specs
    ):
        """Test validation fails when wind capacity doesn't match turbine config."""
        # Modify wind capacity to not match turbine specs
        sample_hybrid_config.wind_capacity_mw = 999.0

        integrator = WindHybridIntegrator(sample_hybrid_config)

        with pytest.raises(ValueError, match="doesn't match turbine configuration"):
            integrator.validate_configuration()

    def test_validate_configuration_insufficient_power_curve(
        self, sample_hybrid_config, sample_turbine_specs
    ):
        """Test validation fails with insufficient power curve points."""
        # Modify turbine specs to have too few power curve points
        sample_turbine_specs.power_curve_speeds_ms = [0, 10]
        sample_turbine_specs.power_curve_kw = [0, 3000]

        integrator = WindHybridIntegrator(sample_hybrid_config)

        with pytest.raises(ValueError, match="Power curve must have at least 3 points"):
            integrator.validate_configuration()

    def test_reset_clears_initialized_flag(self, sample_hybrid_config):
        """Test that reset() clears the initialized flag."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assert integrator.is_initialized()

        integrator.reset()

        assert not integrator.is_initialized()


class TestWindResourceAssessment:
    """Test wind resource assessment functionality."""

    def test_wind_resource_assessment_basic(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test basic wind resource assessment."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assessment = integrator.wind_resource_assessment(sample_wind_data)

        assert assessment is not None
        assert assessment.mean_wind_speed_ms > 0
        assert 0 < assessment.capacity_factor_estimate <= 1
        assert assessment.annual_energy_potential_mwh > 0
        assert assessment.weibull_k > 0
        assert assessment.weibull_c > 0
        assert 0 <= assessment.prevailing_direction_deg < 360

    def test_wind_resource_assessment_stores_result(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test that assessment result is stored in integrator."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assessment = integrator.wind_resource_assessment(sample_wind_data)

        assert integrator.wind_assessment == assessment

    def test_wind_resource_assessment_with_custom_hub_height(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test assessment with custom hub height."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assessment = integrator.wind_resource_assessment(
            sample_wind_data,
            target_hub_height=100.0
        )

        # Wind speed should be higher at 100m than at turbine hub height (80m)
        assert assessment.mean_wind_speed_ms > 0

    def test_wind_resource_assessment_insufficient_data(
        self, sample_hybrid_config
    ):
        """Test assessment fails with insufficient data."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        # Create wind data with too few points
        insufficient_data = WindResourceData(
            site_id="test",
            latitude=45.0,
            longitude=-95.0,
            elevation_m=300.0,
            wind_speeds_ms=[5.0] * 50,  # Only 50 points
            wind_directions_deg=[180.0] * 50,
            measurement_height_m=10.0,
            assessment_period_days=1,
        )

        with pytest.raises(ValueError, match="Insufficient wind data"):
            integrator.wind_resource_assessment(insufficient_data)

    def test_wind_resource_assessment_realistic_values(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test that assessment produces realistic values."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        assessment = integrator.wind_resource_assessment(sample_wind_data)

        # Check realistic ranges
        assert 3 < assessment.mean_wind_speed_ms < 20  # m/s
        assert 1 < assessment.weibull_k < 4  # Shape parameter
        assert 0 < assessment.capacity_factor_estimate < 0.6  # CF typically < 60%
        assert 0 < assessment.turbulence_intensity < 0.5  # TI typically < 50%


class TestTurbineModeling:
    """Test wind turbine modeling functionality."""

    def test_turbine_modeling_basic(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test basic turbine modeling."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        performance = integrator.turbine_modeling(sample_wind_data)

        assert performance is not None
        assert 0 < performance.capacity_factor <= 1
        assert performance.annual_energy_production_mwh > 0
        assert len(performance.power_output_timeseries_kw) > 0

    def test_turbine_modeling_with_losses(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test turbine modeling includes losses."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        performance = integrator.turbine_modeling(
            sample_wind_data,
            include_losses=True
        )

        # Net capacity factor should be less than gross capacity factor
        assert performance.net_capacity_factor < performance.capacity_factor
        assert performance.wake_losses_percent >= 0
        assert performance.electrical_losses_percent > 0
        assert performance.availability_factor <= 1

    def test_turbine_modeling_without_losses(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test turbine modeling without losses."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        performance = integrator.turbine_modeling(
            sample_wind_data,
            include_losses=False
        )

        # Net and gross capacity factor should be equal
        assert performance.net_capacity_factor == performance.capacity_factor
        assert performance.wake_losses_percent == 0
        assert performance.availability_factor == 1.0

    def test_turbine_modeling_stores_result(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test that modeling result is stored in integrator."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        performance = integrator.turbine_modeling(sample_wind_data)

        assert integrator.turbine_performance == performance

    def test_turbine_modeling_respects_power_curve(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test that turbine modeling respects power curve limits."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        performance = integrator.turbine_modeling(sample_wind_data)

        # Maximum power should not exceed rated power
        max_power = max(performance.power_output_timeseries_kw)
        rated_power = sample_hybrid_config.turbine_specs.rated_power_kw

        assert max_power <= rated_power * 1.01  # Allow 1% tolerance


class TestHybridOptimization:
    """Test hybrid system optimization functionality."""

    def test_hybrid_optimization_maximize_energy(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test optimization with maximize energy objective."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        # Need to run assessment first
        integrator.wind_resource_assessment(sample_wind_data)

        result = integrator.hybrid_optimization(
            wind_data=sample_wind_data,
            objective="maximize_energy"
        )

        assert result is not None
        assert result.optimal_pv_capacity_mw >= 0
        assert result.optimal_wind_capacity_mw >= 0
        assert result.total_annual_energy_mwh > 0
        assert result.optimization_objective == OptimizationObjective.MAXIMIZE_ENERGY

    def test_hybrid_optimization_minimize_cost(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test optimization with minimize cost objective."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()
        integrator.wind_resource_assessment(sample_wind_data)

        result = integrator.hybrid_optimization(
            wind_data=sample_wind_data,
            objective="minimize_cost"
        )

        assert result.optimization_objective == OptimizationObjective.MINIMIZE_COST
        assert result.levelized_cost_of_energy > 0

    def test_hybrid_optimization_with_constraints(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test optimization with custom constraints."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()
        integrator.wind_resource_assessment(sample_wind_data)

        constraints = {
            "max_pv_capacity_mw": 5.0,
            "max_wind_capacity_mw": 10.0,
            "max_storage_capacity_mwh": 10.0,
        }

        result = integrator.hybrid_optimization(
            wind_data=sample_wind_data,
            objective="maximize_energy",
            constraints=constraints
        )

        # Results should respect constraints
        assert result.optimal_pv_capacity_mw <= constraints["max_pv_capacity_mw"]
        assert result.optimal_wind_capacity_mw <= constraints["max_wind_capacity_mw"]
        if result.optimal_storage_capacity_mwh:
            assert result.optimal_storage_capacity_mwh <= constraints["max_storage_capacity_mwh"]

    def test_hybrid_optimization_stores_result(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test that optimization result is stored in integrator."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()
        integrator.wind_resource_assessment(sample_wind_data)

        result = integrator.hybrid_optimization(
            wind_data=sample_wind_data,
            objective="maximize_energy"
        )

        assert integrator.optimization_result == result

    def test_hybrid_optimization_invalid_objective(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test optimization fails with invalid objective."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()
        integrator.wind_resource_assessment(sample_wind_data)

        with pytest.raises(ValueError, match="Invalid objective"):
            integrator.hybrid_optimization(
                wind_data=sample_wind_data,
                objective="invalid_objective"
            )

    def test_hybrid_optimization_not_initialized(
        self, sample_hybrid_config, sample_wind_data
    ):
        """Test optimization fails if integrator not initialized."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        with pytest.raises(RuntimeError, match="must be initialized"):
            integrator.hybrid_optimization(
                wind_data=sample_wind_data,
                objective="maximize_energy"
            )


class TestWindPVCoordination:
    """Test wind-PV coordination functionality."""

    def test_wind_pv_coordination_basic(
        self, sample_hybrid_config, sample_generation_timeseries
    ):
        """Test basic wind-PV coordination."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        wind_gen, pv_gen = sample_generation_timeseries

        results = integrator.wind_pv_coordination(wind_gen, pv_gen)

        assert len(results) == len(wind_gen)
        assert all(r.total_dispatch_mw >= 0 for r in results)
        assert all(r.coordination_efficiency >= 0 for r in results)

    def test_wind_pv_coordination_with_custom_strategy(
        self, sample_hybrid_config, sample_generation_timeseries,
        sample_coordination_strategy
    ):
        """Test coordination with custom strategy."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        wind_gen, pv_gen = sample_generation_timeseries

        results = integrator.wind_pv_coordination(
            wind_gen,
            pv_gen,
            strategy=sample_coordination_strategy
        )

        assert len(results) > 0

    def test_wind_pv_coordination_respects_grid_capacity(
        self, sample_hybrid_config, sample_generation_timeseries
    ):
        """Test that coordination respects grid capacity limits."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        wind_gen, pv_gen = sample_generation_timeseries

        results = integrator.wind_pv_coordination(wind_gen, pv_gen)

        grid_capacity = sample_hybrid_config.grid_connection_capacity_mw

        # Grid export should not exceed grid capacity
        assert all(r.grid_export_mw <= grid_capacity * 1.01 for r in results)

    def test_wind_pv_coordination_handles_curtailment(
        self, sample_hybrid_config, sample_generation_timeseries
    ):
        """Test that coordination handles curtailment when necessary."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        # Create high generation scenario
        wind_gen = [15.0] * 100  # High wind
        pv_gen = [10.0] * 100    # High solar

        results = integrator.wind_pv_coordination(wind_gen, pv_gen)

        # Should have some curtailment when generation exceeds grid capacity
        total_curtailed = sum(r.curtailed_energy_mw for r in results)
        grid_capacity = sample_hybrid_config.grid_connection_capacity_mw

        if wind_gen[0] + pv_gen[0] > grid_capacity:
            assert total_curtailed > 0

    def test_wind_pv_coordination_mismatched_lengths(
        self, sample_hybrid_config
    ):
        """Test coordination fails with mismatched timeseries lengths."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        wind_gen = [10.0] * 100
        pv_gen = [5.0] * 50  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            integrator.wind_pv_coordination(wind_gen, pv_gen)

    def test_wind_pv_coordination_with_demand(
        self, sample_hybrid_config, sample_generation_timeseries
    ):
        """Test coordination with grid demand data."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        wind_gen, pv_gen = sample_generation_timeseries
        demand = [12.0] * len(wind_gen)

        results = integrator.wind_pv_coordination(
            wind_gen,
            pv_gen,
            grid_demand_mw=demand
        )

        assert len(results) == len(wind_gen)


class TestRunSimulation:
    """Test complete simulation workflow."""

    def test_run_simulation_complete(
        self, sample_hybrid_config
    ):
        """Test complete simulation workflow."""
        integrator = WindHybridIntegrator(sample_hybrid_config)
        integrator.initialize()

        results = integrator.run_simulation()

        assert results is not None
        assert "system_id" in results
        assert "site_name" in results
        assert "simulation_timestamp" in results
        assert results["system_id"] == sample_hybrid_config.system_id

    def test_run_simulation_not_initialized(
        self, sample_hybrid_config
    ):
        """Test simulation fails if not initialized."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        with pytest.raises(RuntimeError, match="must be initialized"):
            integrator.run_simulation()


class TestHelperMethods:
    """Test private helper methods."""

    def test_extrapolate_wind_speeds(self, sample_hybrid_config):
        """Test wind speed extrapolation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        speeds = [5.0, 6.0, 7.0, 8.0]
        extrapolated = integrator._extrapolate_wind_speeds(
            speeds, 10.0, 80.0, 0.14
        )

        # Wind speeds should increase with height
        assert all(extrapolated > speeds)

    def test_fit_weibull_distribution(self, sample_hybrid_config, sample_wind_data):
        """Test Weibull distribution fitting."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        speeds = np.array(sample_wind_data.wind_speeds_ms)
        k, c = integrator._fit_weibull_distribution(speeds)

        # Check parameter ranges
        assert k > 0
        assert c > 0
        assert 1 < k < 4  # Typical range for wind

    def test_calculate_wind_power_density(self, sample_hybrid_config):
        """Test wind power density calculation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        speeds = np.array([5.0, 6.0, 7.0, 8.0])
        power_density = integrator._calculate_wind_power_density(speeds, 1.225)

        # Power density should be positive
        assert power_density > 0

    def test_calculate_prevailing_direction(self, sample_hybrid_config):
        """Test prevailing direction calculation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        # Test with westerly winds (270°)
        directions = [260, 270, 280, 270, 265, 275]
        prevailing = integrator._calculate_prevailing_direction(directions)

        assert 0 <= prevailing < 360
        assert 260 <= prevailing <= 280  # Should be around 270

    def test_estimate_capacity_factor(
        self, sample_hybrid_config, sample_turbine_specs
    ):
        """Test capacity factor estimation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        speeds = np.array([7.0] * 100)  # Constant wind speed
        cf = integrator._estimate_capacity_factor(speeds, sample_turbine_specs)

        assert 0 <= cf <= 1

    def test_calculate_wake_losses(self, sample_hybrid_config):
        """Test wake loss calculation."""
        integrator = WindHybridIntegrator(sample_hybrid_config)

        # Single turbine should have no wake losses
        assert integrator._calculate_wake_losses(1) == 0.0

        # Multiple turbines should have wake losses
        assert integrator._calculate_wake_losses(10) > 0

        # More turbines should have higher losses
        assert integrator._calculate_wake_losses(20) > integrator._calculate_wake_losses(5)
