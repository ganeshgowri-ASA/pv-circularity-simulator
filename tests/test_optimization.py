"""
Comprehensive tests for PV system optimization engine.

This test suite covers all optimization components including:
- System optimizer (GA, PSO, LP)
- Energy yield optimizer
- Economic optimizer
- Layout optimizer
- Design space explorer
"""

import pytest
import numpy as np
from typing import Callable

from src.models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    OptimizationObjectives,
    DesignPoint,
)
from src.optimization.system_optimizer import SystemOptimizer
from src.optimization.energy_yield_optimizer import EnergyYieldOptimizer
from src.optimization.economic_optimizer import EconomicOptimizer
from src.optimization.layout_optimizer import LayoutOptimizer
from src.optimization.design_space_explorer import DesignSpaceExplorer, ParameterRange


# Fixtures
@pytest.fixture
def default_parameters() -> PVSystemParameters:
    """Default PV system parameters for testing."""
    return PVSystemParameters(
        module_power=450.0,
        module_efficiency=0.20,
        module_area=2.5,
        module_cost=150.0,
        bifacial=False,
        bifaciality=0.7,
        tracker_type="fixed",
        tilt_angle=25.0,
        azimuth=180.0,
        gcr=0.4,
        dc_ac_ratio=1.25,
        inverter_efficiency=0.98,
        inverter_cost_per_kw=100.0,
        land_cost_per_acre=5000.0,
        available_land_acres=100.0,
        latitude=35.0,
        longitude=-120.0,
        elevation=0.0,
        albedo=0.2,
        num_modules=10000,
        row_spacing=5.0,
        string_length=20,
        discount_rate=0.08,
        project_lifetime=25,
        degradation_rate=0.005,
        om_cost_per_kw_year=15.0,
    )


@pytest.fixture
def default_constraints() -> OptimizationConstraints:
    """Default optimization constraints for testing."""
    return OptimizationConstraints(
        min_gcr=0.2,
        max_gcr=0.6,
        min_dc_ac_ratio=1.1,
        max_dc_ac_ratio=1.5,
        min_tilt=10.0,
        max_tilt=40.0,
        max_shading_loss=0.1,
    )


@pytest.fixture
def default_objectives() -> OptimizationObjectives:
    """Default optimization objectives for testing."""
    return OptimizationObjectives(
        maximize_energy=1.0,
        minimize_lcoe=1.0,
        minimize_land_use=0.5,
        maximize_npv=0.8,
    )


# SystemOptimizer Tests
class TestSystemOptimizer:
    """Test suite for SystemOptimizer class."""

    def test_initialization(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test SystemOptimizer initialization."""
        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )
        assert optimizer.parameters == default_parameters
        assert optimizer.constraints == default_constraints
        assert optimizer.objectives is not None

    def test_genetic_algorithm_optimizer(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test genetic algorithm optimization."""
        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        result = optimizer.genetic_algorithm_optimizer(
            population_size=20,
            num_generations=10,
        )

        assert result.success
        assert result.algorithm == "genetic"
        assert result.best_solution.gcr >= default_constraints.min_gcr
        assert result.best_solution.gcr <= default_constraints.max_gcr
        assert result.best_solution.lcoe > 0
        assert result.execution_time_seconds > 0

    def test_particle_swarm_optimizer(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test particle swarm optimization."""
        pytest.importorskip("pyswarm")  # Skip if pyswarm not available

        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        result = optimizer.particle_swarm_optimizer(
            swarm_size=20,
            max_iterations=20,
        )

        assert result.success
        assert result.algorithm == "pso"
        assert result.best_solution.dc_ac_ratio >= default_constraints.min_dc_ac_ratio
        assert result.best_solution.dc_ac_ratio <= default_constraints.max_dc_ac_ratio

    def test_linear_programming_optimizer(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test linear programming optimization."""
        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        result = optimizer.linear_programming_optimizer()

        assert result.algorithm == "linear"
        assert result.best_solution is not None
        assert result.num_evaluations >= 1

    def test_multi_objective_optimization(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test multi-objective optimization."""
        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        result = optimizer.multi_objective_optimization(
            population_size=30,
            num_generations=10,
        )

        assert result.success
        assert result.algorithm == "multi_objective"
        assert len(result.pareto_front) > 0
        assert all(sol.rank == 0 for sol in result.pareto_front)

    def test_optimization_constraints(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test constraint retrieval."""
        optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        constraints = optimizer.optimization_constraints()

        assert "gcr_range" in constraints
        assert "dc_ac_ratio_range" in constraints
        assert constraints["gcr_range"] == (default_constraints.min_gcr, default_constraints.max_gcr)


# EnergyYieldOptimizer Tests
class TestEnergyYieldOptimizer:
    """Test suite for EnergyYieldOptimizer class."""

    def test_maximize_annual_energy_analytical(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test analytical energy maximization."""
        optimizer = EnergyYieldOptimizer(default_parameters, default_constraints)

        energy, params = optimizer.maximize_annual_energy(method="analytical")

        assert energy > 0
        assert "tilt_angle" in params
        assert "azimuth" in params
        assert params["method"] == "analytical"

    def test_maximize_annual_energy_gradient(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test gradient-based energy maximization."""
        optimizer = EnergyYieldOptimizer(default_parameters, default_constraints)

        energy, params = optimizer.maximize_annual_energy(method="gradient")

        assert energy > 0
        assert params["success"]
        assert default_constraints.min_tilt <= params["tilt_angle"] <= default_constraints.max_tilt

    def test_minimize_shading_losses(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test shading loss minimization."""
        optimizer = EnergyYieldOptimizer(default_parameters, default_constraints)

        shading_loss, params = optimizer.minimize_shading_losses()

        assert 0 <= shading_loss <= 1
        assert "optimal_row_spacing" in params
        assert params["optimal_row_spacing"] > 0

    def test_optimize_bifacial_gain(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test bifacial gain optimization."""
        # Test with bifacial modules
        params_bifacial = default_parameters.model_copy(deep=True)
        params_bifacial.bifacial = True

        optimizer = EnergyYieldOptimizer(params_bifacial, default_constraints)
        gain, results = optimizer.optimize_bifacial_gain()

        assert gain >= 0
        assert "bifacial_gain" in results

        # Test with non-bifacial modules
        optimizer_mono = EnergyYieldOptimizer(default_parameters, default_constraints)
        gain_mono, results_mono = optimizer_mono.optimize_bifacial_gain()

        assert gain_mono == 0
        assert "note" in results_mono

    def test_optimize_tracker_angles(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test tracker angle optimization."""
        # Test fixed tilt
        optimizer_fixed = EnergyYieldOptimizer(default_parameters, default_constraints)
        result_fixed = optimizer_fixed.optimize_tracker_angles()

        assert "tracker_angles" in result_fixed
        assert result_fixed["energy_gain"] == 0

        # Test single-axis tracking
        params_tracking = default_parameters.model_copy(deep=True)
        params_tracking.tracker_type = "single_axis"

        optimizer_tracking = EnergyYieldOptimizer(params_tracking, default_constraints)
        result_tracking = optimizer_tracking.optimize_tracker_angles()

        assert result_tracking["energy_gain"] > 0
        assert len(result_tracking["tracker_angles"]) > 0

    def test_seasonal_optimization(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test seasonal tilt optimization."""
        optimizer = EnergyYieldOptimizer(default_parameters, default_constraints)

        seasonal_tilts = optimizer.seasonal_optimization()

        assert len(seasonal_tilts) == 12  # 12 months
        assert all(1 <= month <= 12 for month in seasonal_tilts.keys())
        assert all(
            default_constraints.min_tilt <= tilt <= default_constraints.max_tilt
            for tilt in seasonal_tilts.values()
        )


# EconomicOptimizer Tests
class TestEconomicOptimizer:
    """Test suite for EconomicOptimizer class."""

    def test_minimize_lcoe(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test LCOE minimization."""
        optimizer = EconomicOptimizer(default_parameters, default_constraints)

        lcoe, params = optimizer.minimize_lcoe(vary_dc_ac_ratio=True, vary_gcr=True)

        assert lcoe > 0
        assert "lcoe" in params
        assert "dc_ac_ratio" in params
        assert "gcr" in params
        assert params["success"]

    def test_maximize_npv(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test NPV maximization."""
        optimizer = EconomicOptimizer(default_parameters, default_constraints)

        npv, params = optimizer.maximize_npv(electricity_price=0.06)

        assert "npv" in params
        assert "num_modules" in params
        assert "capacity_mw" in params

    def test_optimize_dc_ac_ratio(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test DC/AC ratio optimization."""
        optimizer = EconomicOptimizer(default_parameters, default_constraints)

        dc_ac, params = optimizer.optimize_dc_ac_ratio()

        assert default_constraints.min_dc_ac_ratio <= dc_ac <= default_constraints.max_dc_ac_ratio
        assert "clipping_loss" in params
        assert params["inverter_capacity_kw"] > 0

    def test_optimize_module_selection(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test module selection optimization."""
        optimizer = EconomicOptimizer(default_parameters, default_constraints)

        module_options = [
            {"power": 400, "efficiency": 0.19, "cost": 140, "area": 2.4},
            {"power": 450, "efficiency": 0.20, "cost": 150, "area": 2.5},
            {"power": 500, "efficiency": 0.21, "cost": 165, "area": 2.6},
        ]

        best_idx, result = optimizer.optimize_module_selection(module_options)

        assert 0 <= best_idx < len(module_options)
        assert "lcoe" in result
        assert "annual_energy_kwh" in result

    def test_balance_of_system_optimization(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test BOS optimization."""
        optimizer = EconomicOptimizer(default_parameters, default_constraints)

        bos_costs = optimizer.balance_of_system_optimization()

        assert "total_bos_cost" in bos_costs
        assert "racking_cost_per_w" in bos_costs
        assert bos_costs["total_bos_cost"] > 0


# LayoutOptimizer Tests
class TestLayoutOptimizer:
    """Test suite for LayoutOptimizer class."""

    def test_optimize_gcr_energy_per_area(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test GCR optimization for energy per area."""
        optimizer = LayoutOptimizer(default_parameters, default_constraints)

        gcr, params = optimizer.optimize_gcr(objective="energy_per_area")

        assert default_constraints.min_gcr <= gcr <= default_constraints.max_gcr
        assert "energy_per_acre" in params
        assert params["success"]

    def test_minimize_land_use(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test land use minimization."""
        optimizer = LayoutOptimizer(default_parameters, default_constraints)

        land_acres, params = optimizer.minimize_land_use(min_capacity_mw=4.5)

        assert land_acres > 0
        assert params["feasible"]
        assert params["capacity_mw"] >= 4.5

    def test_maximize_capacity(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test capacity maximization."""
        optimizer = LayoutOptimizer(default_parameters, default_constraints)

        capacity, params = optimizer.maximize_capacity(available_land_acres=50)

        assert capacity > 0
        assert params["land_use_acres"] <= 50
        assert params["num_modules"] > 0

    def test_optimize_string_configuration(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test string configuration optimization."""
        optimizer = LayoutOptimizer(default_parameters, default_constraints)

        config = optimizer.optimize_string_configuration(
            inverter_mppt_voltage_range=(600, 1500),
            module_voltage=40.0,
        )

        assert "modules_per_string" in config
        assert "num_strings" in config
        assert config["modules_per_string"] > 0
        assert config["string_voltage_nominal"] > 0

    def test_terrain_following_optimization(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test terrain-following optimization."""
        optimizer = LayoutOptimizer(default_parameters, default_constraints)

        result = optimizer.terrain_following_optimization(
            terrain_slope=5.0,
            terrain_aspect=180.0,
        )

        assert "adjusted_tilt" in result
        assert "adjusted_row_spacing" in result
        assert "grading_cost" in result
        assert result["grading_cost"] > 0


# DesignSpaceExplorer Tests
class TestDesignSpaceExplorer:
    """Test suite for DesignSpaceExplorer class."""

    def test_parameter_sweep_1d(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test 1D parameter sweep."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        param_ranges = [ParameterRange("gcr", 0.2, 0.6, 10)]

        results = explorer.parameter_sweep(
            param_ranges,
            output_metric="lcoe",
            parallel=False,
        )

        assert results["dimension"] == 1
        assert len(results["values"]) == 10
        assert len(results["results"]) == 10

    def test_parameter_sweep_2d(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test 2D parameter sweep."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        param_ranges = [
            ParameterRange("gcr", 0.2, 0.6, 5),
            ParameterRange("dc_ac_ratio", 1.1, 1.5, 5),
        ]

        results = explorer.parameter_sweep(
            param_ranges,
            output_metric="lcoe",
            parallel=False,
        )

        assert results["dimension"] == 2
        assert len(results["param1_values"]) == 5
        assert len(results["param2_values"]) == 5

    def test_sensitivity_analysis(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test sensitivity analysis."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        results = explorer.sensitivity_analysis(
            parameters_to_vary=["gcr", "dc_ac_ratio"],
            variation_percent=10.0,
        )

        assert len(results) > 0
        for result in results:
            assert result.sensitivity_index >= 0
            assert -1 <= result.correlation <= 1

    def test_monte_carlo_simulation(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test Monte Carlo simulation."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        results = explorer.monte_carlo_simulation(
            num_samples=100,
            parallel=False,
        )

        assert results["num_samples"] == 100
        assert "statistics" in results
        assert "lcoe" in results["statistics"]
        assert "mean" in results["statistics"]["lcoe"]
        assert "std" in results["statistics"]["lcoe"]

    def test_pareto_frontier_analysis(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test Pareto frontier analysis."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        pareto_solutions = explorer.pareto_frontier_analysis(
            objective1="lcoe",
            objective2="annual_energy_kwh",
            num_points=30,
        )

        assert len(pareto_solutions) > 0
        assert all(sol.rank == 0 for sol in pareto_solutions)

    def test_constraint_handling(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
    ) -> None:
        """Test constraint violation checking."""
        explorer = DesignSpaceExplorer(default_parameters, default_constraints)

        # Create a feasible design
        feasible_design = DesignPoint(
            gcr=0.4,
            dc_ac_ratio=1.25,
            tilt_angle=25.0,
            num_modules=10000,
            row_spacing=5.0,
            annual_energy_kwh=20000000,
            capacity_mw=4.5,
            lcoe=0.05,
            npv=5000000,
            land_use_acres=50,
            shading_loss=0.05,
            capex=6000000,
        )

        is_feasible, violations = explorer.constraint_handling(feasible_design)

        assert is_feasible
        assert len(violations) == 0

        # Create an infeasible design
        infeasible_design = DesignPoint(
            gcr=0.8,  # Exceeds max_gcr
            dc_ac_ratio=1.25,
            tilt_angle=25.0,
            num_modules=10000,
            row_spacing=5.0,
            annual_energy_kwh=20000000,
            capacity_mw=4.5,
            lcoe=0.05,
            npv=5000000,
            land_use_acres=50,
            shading_loss=0.05,
            capex=6000000,
        )

        is_feasible2, violations2 = explorer.constraint_handling(infeasible_design)

        assert not is_feasible2
        assert len(violations2) > 0


# Integration Tests
class TestIntegration:
    """Integration tests for the full optimization workflow."""

    def test_full_optimization_workflow(
        self,
        default_parameters: PVSystemParameters,
        default_constraints: OptimizationConstraints,
        default_objectives: OptimizationObjectives,
    ) -> None:
        """Test complete optimization workflow."""
        # Step 1: Run system optimization
        system_optimizer = SystemOptimizer(
            default_parameters,
            default_constraints,
            default_objectives,
        )

        ga_result = system_optimizer.genetic_algorithm_optimizer(
            population_size=20,
            num_generations=5,
        )

        assert ga_result.success

        # Step 2: Analyze energy yield
        energy_optimizer = EnergyYieldOptimizer(
            default_parameters,
            default_constraints,
        )

        max_energy, _ = energy_optimizer.maximize_annual_energy()
        assert max_energy > 0

        # Step 3: Economic analysis
        econ_optimizer = EconomicOptimizer(
            default_parameters,
            default_constraints,
        )

        min_lcoe, _ = econ_optimizer.minimize_lcoe()
        assert min_lcoe > 0

        # Step 4: Layout optimization
        layout_optimizer = LayoutOptimizer(
            default_parameters,
            default_constraints,
        )

        optimal_gcr, _ = layout_optimizer.optimize_gcr()
        assert default_constraints.min_gcr <= optimal_gcr <= default_constraints.max_gcr

        # Step 5: Design space exploration
        explorer = DesignSpaceExplorer(
            default_parameters,
            default_constraints,
        )

        sensitivity_results = explorer.sensitivity_analysis(
            parameters_to_vary=["gcr"],
            variation_percent=10.0,
        )

        assert len(sensitivity_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
