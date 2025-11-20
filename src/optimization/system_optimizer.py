"""
SystemOptimizer: Multi-algorithm optimization engine for PV system design.

This module implements genetic algorithms, particle swarm optimization,
and linear programming for single and multi-objective optimization.
"""

import time
from typing import List, Tuple, Callable, Optional, Dict, Any
import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint
from deap import base, creator, tools, algorithms
import warnings

try:
    from pyswarm import pso
except ImportError:
    pso = None  # Will raise error if PSO is attempted without pyswarm

from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, lpSum, value, PULP_CBC_CMD

from ..models.optimization_models import (
    PVSystemParameters,
    OptimizationConstraints,
    OptimizationObjectives,
    OptimizationResult,
    DesignPoint,
    ParetoSolution,
)


class SystemOptimizer:
    """
    Multi-algorithm optimization engine for PV system design.

    This class implements multiple optimization algorithms including:
    - Genetic algorithms (GA) via DEAP
    - Particle swarm optimization (PSO) via pyswarm
    - Linear programming (LP) via PuLP
    - Multi-objective optimization with Pareto frontier analysis

    Attributes:
        parameters: PV system parameters
        constraints: Optimization constraints
        objectives: Optimization objectives and weights
    """

    def __init__(
        self,
        parameters: PVSystemParameters,
        constraints: OptimizationConstraints,
        objectives: OptimizationObjectives,
    ):
        """
        Initialize the system optimizer.

        Args:
            parameters: PV system base parameters
            constraints: Optimization constraints
            objectives: Optimization objectives and weights
        """
        self.parameters = parameters
        self.constraints = constraints
        self.objectives = objectives.normalize()

        # Performance evaluator (will be set by specialized optimizers)
        self._evaluator: Optional[Callable[[PVSystemParameters], DesignPoint]] = None

    def set_evaluator(
        self, evaluator: Callable[[PVSystemParameters], DesignPoint]
    ) -> None:
        """
        Set the performance evaluation function.

        Args:
            evaluator: Function that takes PVSystemParameters and returns DesignPoint
        """
        self._evaluator = evaluator

    def genetic_algorithm_optimizer(
        self,
        population_size: int = 100,
        num_generations: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
    ) -> OptimizationResult:
        """
        Optimize using genetic algorithm (NSGA-II for multi-objective).

        This method uses DEAP library to implement a genetic algorithm with:
        - Tournament selection
        - Two-point crossover
        - Gaussian mutation
        - Elitism

        Args:
            population_size: Number of individuals in population
            num_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Tournament selection size

        Returns:
            OptimizationResult with best solution and convergence history
        """
        start_time = time.time()

        # Define bounds for design variables: [gcr, dc_ac_ratio, tilt_angle]
        bounds = [
            (self.constraints.min_gcr, self.constraints.max_gcr),
            (self.constraints.min_dc_ac_ratio, self.constraints.max_dc_ac_ratio),
            (self.constraints.min_tilt, self.constraints.max_tilt),
        ]

        def objective_function(x: np.ndarray) -> float:
            """Weighted multi-objective function."""
            gcr, dc_ac_ratio, tilt = x

            # Create modified parameters
            params = self.parameters.model_copy(deep=True)
            params.gcr = float(gcr)
            params.dc_ac_ratio = float(dc_ac_ratio)
            params.tilt_angle = float(tilt)

            # Evaluate design point
            if self._evaluator is None:
                # Fallback to simple analytical model
                design = self._evaluate_simple(params)
            else:
                design = self._evaluator(params)

            # Apply constraints as penalties
            penalty = 0.0
            if self.constraints.max_land_use_acres and design.land_use_acres > self.constraints.max_land_use_acres:
                penalty += 1e6 * (design.land_use_acres - self.constraints.max_land_use_acres)
            if design.shading_loss > self.constraints.max_shading_loss:
                penalty += 1e6 * (design.shading_loss - self.constraints.max_shading_loss)

            # Compute weighted objective (to minimize)
            obj = 0.0
            obj += self.objectives.minimize_lcoe * design.lcoe
            obj -= self.objectives.maximize_energy * design.annual_energy_kwh / 1e6
            obj += self.objectives.minimize_land_use * design.land_use_acres / 100.0
            obj -= self.objectives.maximize_npv * design.npv / 1e6
            obj += self.objectives.minimize_shading * design.shading_loss

            return obj + penalty

        # Use scipy's differential evolution (a form of genetic algorithm)
        convergence = []

        def callback(xk: np.ndarray, convergence_val: float = 0.0) -> bool:
            """Callback to track convergence."""
            convergence.append(objective_function(xk))
            return False

        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=num_generations,
            popsize=population_size,
            mutation=(0.5, 1.0),
            recombination=crossover_prob,
            callback=callback,
            workers=1,
            updating='deferred',
            polish=True,
        )

        # Create best solution
        best_params = self.parameters.model_copy(deep=True)
        best_params.gcr = float(result.x[0])
        best_params.dc_ac_ratio = float(result.x[1])
        best_params.tilt_angle = float(result.x[2])

        if self._evaluator is None:
            best_design = self._evaluate_simple(best_params)
        else:
            best_design = self._evaluator(best_params)

        execution_time = time.time() - start_time

        return OptimizationResult(
            algorithm="genetic",
            best_solution=best_design,
            convergence_history=convergence,
            execution_time_seconds=execution_time,
            num_iterations=num_generations,
            num_evaluations=result.nfev,
            success=result.success,
            message=result.message,
            metadata={
                "population_size": population_size,
                "crossover_prob": crossover_prob,
                "mutation_prob": mutation_prob,
            },
        )

    def particle_swarm_optimizer(
        self,
        swarm_size: int = 50,
        max_iterations: int = 100,
        omega: float = 0.5,
        phi_p: float = 0.5,
        phi_g: float = 0.5,
    ) -> OptimizationResult:
        """
        Optimize using particle swarm optimization.

        PSO is a population-based stochastic optimization technique inspired
        by social behavior of bird flocking or fish schooling.

        Args:
            swarm_size: Number of particles in swarm
            max_iterations: Maximum number of iterations
            omega: Inertia weight
            phi_p: Cognitive parameter (personal best)
            phi_g: Social parameter (global best)

        Returns:
            OptimizationResult with best solution and convergence history
        """
        if pso is None:
            raise ImportError("pyswarm is required for PSO. Install with: pip install pyswarm")

        start_time = time.time()

        # Define bounds
        lower_bounds = np.array([
            self.constraints.min_gcr,
            self.constraints.min_dc_ac_ratio,
            self.constraints.min_tilt,
        ])
        upper_bounds = np.array([
            self.constraints.max_gcr,
            self.constraints.max_dc_ac_ratio,
            self.constraints.max_tilt,
        ])

        convergence = []

        def objective_function(x: np.ndarray) -> float:
            """Objective function for PSO."""
            gcr, dc_ac_ratio, tilt = x

            params = self.parameters.model_copy(deep=True)
            params.gcr = float(gcr)
            params.dc_ac_ratio = float(dc_ac_ratio)
            params.tilt_angle = float(tilt)

            if self._evaluator is None:
                design = self._evaluate_simple(params)
            else:
                design = self._evaluator(params)

            # Penalty for constraint violations
            penalty = 0.0
            if self.constraints.max_land_use_acres and design.land_use_acres > self.constraints.max_land_use_acres:
                penalty += 1e6 * (design.land_use_acres - self.constraints.max_land_use_acres)
            if design.shading_loss > self.constraints.max_shading_loss:
                penalty += 1e6 * (design.shading_loss - self.constraints.max_shading_loss)

            # Weighted objective
            obj = 0.0
            obj += self.objectives.minimize_lcoe * design.lcoe
            obj -= self.objectives.maximize_energy * design.annual_energy_kwh / 1e6
            obj += self.objectives.minimize_land_use * design.land_use_acres / 100.0
            obj -= self.objectives.maximize_npv * design.npv / 1e6

            total_obj = obj + penalty
            convergence.append(total_obj)
            return total_obj

        # Run PSO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xopt, fopt = pso(
                objective_function,
                lower_bounds,
                upper_bounds,
                swarmsize=swarm_size,
                maxiter=max_iterations,
                omega=omega,
                phip=phi_p,
                phig=phi_g,
            )

        # Create best solution
        best_params = self.parameters.model_copy(deep=True)
        best_params.gcr = float(xopt[0])
        best_params.dc_ac_ratio = float(xopt[1])
        best_params.tilt_angle = float(xopt[2])

        if self._evaluator is None:
            best_design = self._evaluate_simple(best_params)
        else:
            best_design = self._evaluator(best_params)

        execution_time = time.time() - start_time

        return OptimizationResult(
            algorithm="pso",
            best_solution=best_design,
            convergence_history=convergence[:max_iterations],  # Trim to iterations
            execution_time_seconds=execution_time,
            num_iterations=max_iterations,
            num_evaluations=swarm_size * max_iterations,
            success=True,
            message="PSO optimization completed",
            metadata={
                "swarm_size": swarm_size,
                "omega": omega,
                "phi_p": phi_p,
                "phi_g": phi_g,
                "final_objective": float(fopt),
            },
        )

    def linear_programming_optimizer(self) -> OptimizationResult:
        """
        Optimize using linear programming.

        This method formulates a simplified LP problem for cases where
        the objective and constraints can be reasonably approximated
        as linear functions.

        Note: LP is best suited for problems with linear objectives and
        constraints. For complex non-linear PV optimization, consider
        GA or PSO methods instead.

        Returns:
            OptimizationResult with optimal solution
        """
        start_time = time.time()

        # Create LP problem
        prob = LpProblem("PV_System_Optimization", LpMaximize)

        # Decision variables (normalized to [0, 1])
        gcr_var = LpVariable(
            "gcr",
            lowBound=self.constraints.min_gcr,
            upBound=self.constraints.max_gcr,
        )
        dc_ac_var = LpVariable(
            "dc_ac_ratio",
            lowBound=self.constraints.min_dc_ac_ratio,
            upBound=self.constraints.max_dc_ac_ratio,
        )
        tilt_var = LpVariable(
            "tilt",
            lowBound=self.constraints.min_tilt,
            upBound=self.constraints.max_tilt,
        )

        # Simplified linear objective (approximation)
        # Maximize energy (approximated as linear in tilt near optimal)
        # Minimize LCOE (approximated as increasing with gcr)
        prob += (
            self.objectives.maximize_energy * tilt_var
            - self.objectives.minimize_lcoe * gcr_var * 100
            - self.objectives.minimize_land_use * gcr_var * 10
        )

        # Solve
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        # Extract solution
        gcr_opt = value(gcr_var)
        dc_ac_opt = value(dc_ac_var)
        tilt_opt = value(tilt_var)

        # Evaluate actual performance
        best_params = self.parameters.model_copy(deep=True)
        best_params.gcr = float(gcr_opt) if gcr_opt is not None else self.constraints.min_gcr
        best_params.dc_ac_ratio = float(dc_ac_opt) if dc_ac_opt is not None else self.constraints.min_dc_ac_ratio
        best_params.tilt_angle = float(tilt_opt) if tilt_opt is not None else self.constraints.min_tilt

        if self._evaluator is None:
            best_design = self._evaluate_simple(best_params)
        else:
            best_design = self._evaluator(best_params)

        execution_time = time.time() - start_time

        return OptimizationResult(
            algorithm="linear",
            best_solution=best_design,
            convergence_history=[],
            execution_time_seconds=execution_time,
            num_iterations=1,
            num_evaluations=1,
            success=prob.status == 1,
            message=f"LP Status: {prob.status}",
            metadata={"lp_objective": value(prob.objective) if prob.objective else 0.0},
        )

    def multi_objective_optimization(
        self,
        population_size: int = 100,
        num_generations: int = 50,
    ) -> OptimizationResult:
        """
        Multi-objective optimization using NSGA-II algorithm.

        This method finds the Pareto frontier of non-dominated solutions
        for multiple competing objectives (energy, LCOE, land use, etc.).

        Args:
            population_size: Size of population
            num_generations: Number of generations

        Returns:
            OptimizationResult with Pareto front solutions
        """
        start_time = time.time()

        # Setup DEAP for multi-objective optimization
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Attribute generators
        toolbox.register(
            "attr_gcr",
            np.random.uniform,
            self.constraints.min_gcr,
            self.constraints.max_gcr,
        )
        toolbox.register(
            "attr_dc_ac",
            np.random.uniform,
            self.constraints.min_dc_ac_ratio,
            self.constraints.max_dc_ac_ratio,
        )
        toolbox.register(
            "attr_tilt",
            np.random.uniform,
            self.constraints.min_tilt,
            self.constraints.max_tilt,
        )

        # Individual and population
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_gcr, toolbox.attr_dc_ac, toolbox.attr_tilt),
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate_multi(individual: List[float]) -> Tuple[float, float, float]:
            """Multi-objective evaluation: (LCOE, -Energy, LandUse)."""
            gcr, dc_ac_ratio, tilt = individual

            params = self.parameters.model_copy(deep=True)
            params.gcr = float(gcr)
            params.dc_ac_ratio = float(dc_ac_ratio)
            params.tilt_angle = float(tilt)

            if self._evaluator is None:
                design = self._evaluate_simple(params)
            else:
                design = self._evaluator(params)

            # Return objectives (all to minimize with NSGA-II weights)
            return (
                design.lcoe,
                -design.annual_energy_kwh / 1e6,  # Negative for maximization
                design.land_use_acres,
            )

        toolbox.register("evaluate", evaluate_multi)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)

        # Run NSGA-II
        population = toolbox.population(n=population_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            population, logbook = algorithms.eaMuPlusLambda(
                population,
                toolbox,
                mu=population_size,
                lambda_=population_size,
                cxpb=0.8,
                mutpb=0.2,
                ngen=num_generations,
                stats=stats,
                verbose=False,
            )

        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        pareto_solutions = []
        for ind in pareto_front:
            params = self.parameters.model_copy(deep=True)
            params.gcr = float(ind[0])
            params.dc_ac_ratio = float(ind[1])
            params.tilt_angle = float(ind[2])

            if self._evaluator is None:
                design = self._evaluate_simple(params)
            else:
                design = self._evaluator(params)

            pareto_sol = ParetoSolution(
                design=design,
                objectives={
                    "lcoe": design.lcoe,
                    "energy_mwh": design.annual_energy_kwh / 1000,
                    "land_acres": design.land_use_acres,
                },
                rank=0,
                crowding_distance=0.0,
            )
            pareto_solutions.append(pareto_sol)

        # Best solution (first in Pareto front)
        best_ind = pareto_front[0]
        best_params = self.parameters.model_copy(deep=True)
        best_params.gcr = float(best_ind[0])
        best_params.dc_ac_ratio = float(best_ind[1])
        best_params.tilt_angle = float(best_ind[2])

        if self._evaluator is None:
            best_design = self._evaluate_simple(best_params)
        else:
            best_design = self._evaluator(best_params)

        execution_time = time.time() - start_time

        return OptimizationResult(
            algorithm="multi_objective",
            best_solution=best_design,
            pareto_front=pareto_solutions,
            convergence_history=[],
            execution_time_seconds=execution_time,
            num_iterations=num_generations,
            num_evaluations=population_size * num_generations,
            success=True,
            message=f"Found {len(pareto_front)} Pareto-optimal solutions",
            metadata={"pareto_front_size": len(pareto_front)},
        )

    def optimization_constraints(self) -> Dict[str, Any]:
        """
        Get formatted optimization constraints for display.

        Returns:
            Dictionary of constraint descriptions and values
        """
        return {
            "gcr_range": (self.constraints.min_gcr, self.constraints.max_gcr),
            "dc_ac_ratio_range": (
                self.constraints.min_dc_ac_ratio,
                self.constraints.max_dc_ac_ratio,
            ),
            "tilt_range": (self.constraints.min_tilt, self.constraints.max_tilt),
            "max_land_use_acres": self.constraints.max_land_use_acres,
            "max_shading_loss": self.constraints.max_shading_loss,
            "min_capacity_mw": self.constraints.min_capacity_mw,
            "max_capacity_mw": self.constraints.max_capacity_mw,
        }

    def _evaluate_simple(self, params: PVSystemParameters) -> DesignPoint:
        """
        Simple analytical performance evaluation (fallback).

        This provides a basic performance model when no external evaluator
        is provided. For production use, set a proper evaluator.

        Args:
            params: System parameters to evaluate

        Returns:
            DesignPoint with estimated performance
        """
        # System capacity
        capacity_mw = params.num_modules * params.module_power / 1e6

        # Annual energy (simplified model)
        # Peak sun hours approximation based on latitude
        psh = 5.0 - 0.02 * abs(params.latitude)  # Rough approximation
        annual_energy_kwh = (
            capacity_mw
            * 1000
            * psh
            * 365
            * params.inverter_efficiency
            * (1 - 0.02)  # Temperature loss
            * (1 - 0.05)  # Other losses
        )

        # Shading loss (increases with GCR)
        shading_loss = max(0, (params.gcr - 0.3) * 0.2)
        annual_energy_kwh *= 1 - shading_loss

        # Bifacial gain
        bifacial_gain = 0.0
        if params.bifacial:
            bifacial_gain = params.bifaciality * params.albedo * 0.1
            annual_energy_kwh *= 1 + bifacial_gain

        # Land use (simplified)
        module_area_total = params.num_modules * params.module_area
        land_area_m2 = module_area_total / params.gcr
        land_use_acres = land_area_m2 / 4046.86  # m² to acres

        # CAPEX
        module_capex = params.num_modules * params.module_cost
        inverter_capacity_kw = capacity_mw * 1000 / params.dc_ac_ratio
        inverter_capex = inverter_capacity_kw * params.inverter_cost_per_kw
        land_capex = land_use_acres * params.land_cost_per_acre
        bos_capex = capacity_mw * 200000  # $200k/MW for BOS
        total_capex = module_capex + inverter_capex + land_capex + bos_capex

        # LCOE calculation
        total_energy = sum(
            annual_energy_kwh * (1 - params.degradation_rate) ** year
            for year in range(params.project_lifetime)
        )
        discount_factor = sum(
            1 / (1 + params.discount_rate) ** year
            for year in range(1, params.project_lifetime + 1)
        )
        om_costs_pv = params.om_cost_per_kw_year * capacity_mw * 1000 * discount_factor
        lcoe = (total_capex + om_costs_pv) / total_energy if total_energy > 0 else 999

        # NPV calculation
        annual_revenue = annual_energy_kwh * 0.05  # $0.05/kWh assumption
        revenues_pv = sum(
            annual_revenue * (1 - params.degradation_rate) ** year / (1 + params.discount_rate) ** year
            for year in range(1, params.project_lifetime + 1)
        )
        npv = revenues_pv - total_capex - om_costs_pv

        return DesignPoint(
            gcr=params.gcr,
            dc_ac_ratio=params.dc_ac_ratio,
            tilt_angle=params.tilt_angle,
            num_modules=params.num_modules,
            row_spacing=params.row_spacing,
            annual_energy_kwh=annual_energy_kwh,
            capacity_mw=capacity_mw,
            lcoe=lcoe,
            npv=npv,
            land_use_acres=land_use_acres,
            shading_loss=shading_loss,
            bifacial_gain=bifacial_gain,
            capex=total_capex,
        )
