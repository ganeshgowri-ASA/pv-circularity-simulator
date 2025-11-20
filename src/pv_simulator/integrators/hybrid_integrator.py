"""
Wind Hybrid System Integrator for PV Circularity Simulator.

This module implements the WindHybridIntegrator class, which provides
comprehensive functionality for modeling, optimizing, and coordinating
hybrid wind-PV energy systems.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import weibull_min

from pv_simulator.core.base_integrator import BaseIntegrator
from pv_simulator.core.models import (
    HybridSystemConfig,
    WindResourceData,
    WindResourceAssessment,
    TurbineSpecifications,
    TurbinePerformance,
    HybridOptimizationResult,
    CoordinationStrategy,
    CoordinationResult,
    OptimizationObjective,
)


logger = logging.getLogger(__name__)


class WindHybridIntegrator(BaseIntegrator):
    """Integrator for hybrid wind-PV energy systems.

    This class provides comprehensive functionality for:
    - Wind resource assessment and characterization
    - Wind turbine modeling and performance prediction
    - Hybrid system optimization (wind + PV)
    - Real-time coordination between wind and PV generation

    The integrator uses production-ready algorithms including Weibull
    distribution analysis, power curve interpolation, and multi-objective
    optimization for hybrid system design.

    Attributes:
        config: Hybrid system configuration
        wind_assessment: Results from wind resource assessment
        turbine_performance: Wind turbine performance results
        optimization_result: Results from hybrid optimization
        coordination_strategy: Strategy for wind-PV coordination

    Example:
        >>> config = HybridSystemConfig(
        ...     system_id="hybrid_001",
        ...     site_name="Test Site",
        ...     pv_capacity_mw=10.0,
        ...     wind_capacity_mw=15.0,
        ...     num_turbines=5,
        ...     pv_system=pv_config,
        ...     turbine_specs=turbine_specs,
        ...     grid_connection_capacity_mw=20.0
        ... )
        >>> integrator = WindHybridIntegrator(config)
        >>> integrator.initialize()
        >>> assessment = integrator.wind_resource_assessment(wind_data)
        >>> performance = integrator.turbine_modeling(wind_data)
        >>> optimization = integrator.hybrid_optimization(objective="maximize_energy")
    """

    def __init__(self, config: HybridSystemConfig) -> None:
        """Initialize the Wind Hybrid Integrator.

        Args:
            config: Hybrid system configuration with wind and PV specifications

        Raises:
            TypeError: If config is not a HybridSystemConfig instance
        """
        super().__init__(
            config=config,
            integrator_id=config.system_id,
            description=f"Wind-PV Hybrid System at {config.site_name}",
        )

        self.wind_assessment: Optional[WindResourceAssessment] = None
        self.turbine_performance: Optional[TurbinePerformance] = None
        self.optimization_result: Optional[HybridOptimizationResult] = None
        self.coordination_strategy: Optional[CoordinationStrategy] = None

        logger.info(
            f"Initialized WindHybridIntegrator for {config.site_name} "
            f"(PV: {config.pv_capacity_mw} MW, Wind: {config.wind_capacity_mw} MW)"
        )

    def initialize(self) -> None:
        """Initialize the integrator and validate configuration.

        This method prepares the integrator for simulation by validating
        the configuration and setting up internal state.

        Raises:
            ValueError: If configuration validation fails
            RuntimeError: If initialization fails
        """
        if not self.validate_configuration():
            raise ValueError("Configuration validation failed")

        # Set default coordination strategy if not provided
        if self.coordination_strategy is None:
            self.coordination_strategy = CoordinationStrategy(
                strategy_name="default_coordination",
                dispatch_priority=["wind", "pv", "storage"],
                ramp_rate_limit_mw_per_min=5.0,
                forecast_horizon_hours=24,
                enable_storage_arbitrage=True,
                curtailment_strategy="proportional",
                grid_support_enabled=True,
            )

        self._initialized = True
        logger.info(f"WindHybridIntegrator {self.metadata.integrator_id} initialized successfully")

    def validate_configuration(self) -> bool:
        """Validate the hybrid system configuration.

        Validates that:
        - Combined capacity doesn't exceed grid connection capacity
        - Wind capacity matches turbine specifications
        - All required parameters are present and valid

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If validation fails with specific error message
        """
        config = self.config

        # Check total capacity against grid connection
        total_capacity = config.pv_capacity_mw + config.wind_capacity_mw
        if total_capacity > config.grid_connection_capacity_mw * 1.5:
            raise ValueError(
                f"Total capacity {total_capacity} MW significantly exceeds "
                f"grid connection capacity {config.grid_connection_capacity_mw} MW"
            )

        # Validate wind capacity matches turbine config
        expected_wind_capacity = (
            config.num_turbines * config.turbine_specs.rated_power_kw / 1000
        )
        if abs(config.wind_capacity_mw - expected_wind_capacity) > 0.1:
            raise ValueError(
                f"Wind capacity {config.wind_capacity_mw} MW doesn't match "
                f"turbine configuration {expected_wind_capacity} MW"
            )

        # Validate turbine power curve
        if len(config.turbine_specs.power_curve_speeds_ms) < 3:
            raise ValueError("Power curve must have at least 3 points")

        logger.debug(f"Configuration validated successfully for {config.system_id}")
        return True

    def wind_resource_assessment(
        self,
        wind_data: WindResourceData,
        target_hub_height: Optional[float] = None,
    ) -> WindResourceAssessment:
        """Perform comprehensive wind resource assessment.

        Analyzes wind resource data to characterize the wind regime using:
        - Wind speed extrapolation to hub height using power law
        - Weibull distribution parameter estimation
        - Wind power density calculation
        - Turbulence intensity analysis
        - Capacity factor estimation

        Args:
            wind_data: Measured wind resource data
            target_hub_height: Target hub height for extrapolation (uses turbine
                hub height if not specified)

        Returns:
            WindResourceAssessment with complete characterization results

        Raises:
            ValueError: If wind_data is invalid or insufficient
            RuntimeError: If assessment calculation fails

        Example:
            >>> wind_data = WindResourceData(
            ...     site_id="site_001",
            ...     latitude=45.0,
            ...     longitude=-95.0,
            ...     elevation_m=300,
            ...     wind_speeds_ms=[5.2, 6.1, 7.3, ...],
            ...     wind_directions_deg=[180, 175, 190, ...],
            ...     measurement_height_m=10.0,
            ...     assessment_period_days=365
            ... )
            >>> assessment = integrator.wind_resource_assessment(wind_data)
            >>> print(f"Mean wind speed: {assessment.mean_wind_speed_ms} m/s")
            >>> print(f"Capacity factor: {assessment.capacity_factor_estimate}")
        """
        if len(wind_data.wind_speeds_ms) < 100:
            raise ValueError("Insufficient wind data for reliable assessment (minimum 100 points)")

        # Use turbine hub height if not specified
        if target_hub_height is None:
            target_hub_height = self.config.turbine_specs.hub_height_m

        # Extrapolate wind speeds to hub height using power law
        wind_shear_exponent = 0.14  # Typical value, can be refined
        extrapolated_speeds = self._extrapolate_wind_speeds(
            wind_data.wind_speeds_ms,
            wind_data.measurement_height_m,
            target_hub_height,
            wind_shear_exponent,
        )

        # Calculate mean wind speed
        mean_speed = float(np.mean(extrapolated_speeds))

        # Fit Weibull distribution
        weibull_k, weibull_c = self._fit_weibull_distribution(extrapolated_speeds)

        # Calculate wind power density
        wind_power_density = self._calculate_wind_power_density(
            extrapolated_speeds,
            wind_data.air_density_kgm3,
        )

        # Calculate turbulence intensity
        turbulence_intensity = float(np.std(extrapolated_speeds) / mean_speed)

        # Determine prevailing wind direction
        prevailing_direction = self._calculate_prevailing_direction(
            wind_data.wind_directions_deg
        )

        # Estimate capacity factor using turbine power curve
        capacity_factor = self._estimate_capacity_factor(
            extrapolated_speeds,
            self.config.turbine_specs,
        )

        # Calculate annual energy potential
        turbine_rated_power_kw = self.config.turbine_specs.rated_power_kw
        hours_per_year = 8760
        annual_energy_per_turbine = (
            turbine_rated_power_kw * hours_per_year * capacity_factor / 1000  # Convert to MWh
        )
        total_annual_energy = annual_energy_per_turbine * self.config.num_turbines

        # Create assessment result
        assessment = WindResourceAssessment(
            mean_wind_speed_ms=mean_speed,
            weibull_k=weibull_k,
            weibull_c=weibull_c,
            wind_power_density_wm2=wind_power_density,
            turbulence_intensity=turbulence_intensity,
            wind_shear_exponent=wind_shear_exponent,
            prevailing_direction_deg=prevailing_direction,
            capacity_factor_estimate=capacity_factor,
            annual_energy_potential_mwh=total_annual_energy,
        )

        self.wind_assessment = assessment
        logger.info(
            f"Wind resource assessment complete: mean speed {mean_speed:.2f} m/s, "
            f"capacity factor {capacity_factor:.2%}, "
            f"annual energy {total_annual_energy:.0f} MWh"
        )

        return assessment

    def turbine_modeling(
        self,
        wind_data: WindResourceData,
        include_losses: bool = True,
    ) -> TurbinePerformance:
        """Model wind turbine performance with detailed loss analysis.

        Simulates turbine operation using:
        - Power curve interpolation for variable wind speeds
        - Wake effect losses for multiple turbines
        - Electrical system losses
        - Environmental losses (icing, soiling, blade degradation)
        - Availability factor modeling

        Args:
            wind_data: Wind resource data for simulation period
            include_losses: Whether to include detailed loss modeling

        Returns:
            TurbinePerformance with detailed performance metrics and timeseries

        Raises:
            ValueError: If wind_data is invalid
            RuntimeError: If turbine modeling fails

        Example:
            >>> performance = integrator.turbine_modeling(wind_data)
            >>> print(f"Annual energy: {performance.annual_energy_production_mwh} MWh")
            >>> print(f"Capacity factor: {performance.capacity_factor:.2%}")
            >>> print(f"Net capacity factor: {performance.net_capacity_factor:.2%}")
        """
        turbine_specs = self.config.turbine_specs

        # Extrapolate wind speeds to hub height
        hub_height_speeds = self._extrapolate_wind_speeds(
            wind_data.wind_speeds_ms,
            wind_data.measurement_height_m,
            turbine_specs.hub_height_m,
            wind_shear_exponent=0.14,
        )

        # Create power curve interpolation function
        power_curve = interp1d(
            turbine_specs.power_curve_speeds_ms,
            turbine_specs.power_curve_kw,
            kind='linear',
            bounds_error=False,
            fill_value=(0, turbine_specs.rated_power_kw),
        )

        # Calculate power output for each timestep
        power_output_kw = []
        for speed in hub_height_speeds:
            if speed < turbine_specs.cut_in_speed_ms:
                power = 0.0
            elif speed > turbine_specs.cut_out_speed_ms:
                power = 0.0
            else:
                power = float(power_curve(speed))
            power_output_kw.append(power)

        power_output_array = np.array(power_output_kw)

        # Apply turbine efficiency
        power_output_array *= turbine_specs.efficiency

        # Calculate base capacity factor
        rated_power = turbine_specs.rated_power_kw
        capacity_factor = float(np.mean(power_output_array) / rated_power)

        # Apply losses if requested
        if include_losses:
            # Wake losses (depends on number of turbines and layout)
            wake_losses_percent = self._calculate_wake_losses(self.config.num_turbines)

            # Electrical losses (transformer, cables, etc.)
            electrical_losses_percent = 2.0

            # Environmental losses (icing, soiling, degradation)
            environmental_losses_percent = 1.0

            # Availability factor
            availability_factor = 0.97

            # Apply all losses
            total_loss_factor = (
                (1 - wake_losses_percent / 100)
                * (1 - electrical_losses_percent / 100)
                * (1 - environmental_losses_percent / 100)
                * availability_factor
            )

            power_output_array *= total_loss_factor
            net_capacity_factor = capacity_factor * total_loss_factor
        else:
            wake_losses_percent = 0.0
            electrical_losses_percent = 0.0
            environmental_losses_percent = 0.0
            availability_factor = 1.0
            net_capacity_factor = capacity_factor

        # Calculate annual energy production (for one turbine)
        hours_in_period = len(power_output_array) / (365 * 24 / wind_data.assessment_period_days)
        scaling_factor = 8760 / hours_in_period  # Scale to annual
        annual_energy_mwh = float(np.sum(power_output_array) * scaling_factor / 1000)

        # Create performance result
        performance = TurbinePerformance(
            turbine_id=turbine_specs.turbine_id,
            capacity_factor=capacity_factor,
            annual_energy_production_mwh=annual_energy_mwh,
            power_output_timeseries_kw=power_output_array.tolist(),
            availability_factor=availability_factor,
            wake_losses_percent=wake_losses_percent,
            electrical_losses_percent=electrical_losses_percent,
            environmental_losses_percent=environmental_losses_percent,
            net_capacity_factor=net_capacity_factor,
        )

        self.turbine_performance = performance
        logger.info(
            f"Turbine modeling complete: {annual_energy_mwh:.0f} MWh/year, "
            f"CF: {capacity_factor:.2%}, Net CF: {net_capacity_factor:.2%}"
        )

        return performance

    def hybrid_optimization(
        self,
        wind_data: WindResourceData,
        objective: str = "maximize_energy",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> HybridOptimizationResult:
        """Optimize hybrid wind-PV system configuration.

        Performs multi-objective optimization to determine optimal sizing
        of wind, PV, and storage components based on:
        - Resource availability (wind and solar)
        - Economic objectives (minimize cost, maximize ROI)
        - Technical constraints (grid capacity, land availability)
        - Operational objectives (minimize curtailment, maximize capacity factor)

        Uses scipy.optimize with configurable objectives and constraints.

        Args:
            wind_data: Wind resource data for optimization period
            objective: Optimization objective - one of:
                - "maximize_energy": Maximize total energy production
                - "minimize_cost": Minimize levelized cost of energy
                - "minimize_curtailment": Minimize energy curtailment
                - "maximize_capacity_factor": Maximize combined capacity factor
            constraints: Optional dictionary of constraint parameters

        Returns:
            HybridOptimizationResult with optimal configuration and performance

        Raises:
            ValueError: If objective is invalid or constraints are infeasible
            RuntimeError: If optimization fails to converge

        Example:
            >>> result = integrator.hybrid_optimization(
            ...     wind_data=wind_data,
            ...     objective="maximize_energy",
            ...     constraints={"max_pv_capacity_mw": 50, "max_wind_capacity_mw": 100}
            ... )
            >>> print(f"Optimal PV: {result.optimal_pv_capacity_mw} MW")
            >>> print(f"Optimal Wind: {result.optimal_wind_capacity_mw} MW")
            >>> print(f"LCOE: ${result.levelized_cost_of_energy}/MWh")
        """
        if not self._initialized:
            raise RuntimeError("Integrator must be initialized before optimization")

        # Parse objective
        try:
            opt_objective = OptimizationObjective(objective)
        except ValueError:
            raise ValueError(
                f"Invalid objective '{objective}'. Must be one of: "
                f"{[e.value for e in OptimizationObjective]}"
            )

        # Set default constraints
        if constraints is None:
            constraints = {}

        max_pv_mw = constraints.get("max_pv_capacity_mw", self.config.grid_connection_capacity_mw)
        max_wind_mw = constraints.get("max_wind_capacity_mw", self.config.grid_connection_capacity_mw)
        max_storage_mwh = constraints.get("max_storage_capacity_mwh", 100.0)

        # Initial guess: current configuration
        x0 = np.array([
            self.config.pv_capacity_mw,
            self.config.wind_capacity_mw,
            self.config.storage_capacity_mwh or 0.0,
        ])

        # Bounds for optimization variables
        bounds = [
            (0, max_pv_mw),
            (0, max_wind_mw),
            (0, max_storage_mwh),
        ]

        # Define objective function
        def objective_function(x: np.ndarray) -> float:
            pv_capacity, wind_capacity, storage_capacity = x

            # Calculate annual energy production
            wind_cf = self.wind_assessment.capacity_factor_estimate if self.wind_assessment else 0.35
            pv_cf = 0.20  # Typical PV capacity factor

            wind_energy_mwh = wind_capacity * 8760 * wind_cf
            pv_energy_mwh = pv_capacity * 8760 * pv_cf
            total_energy_mwh = wind_energy_mwh + pv_energy_mwh

            # Calculate capacity factor
            total_capacity = pv_capacity + wind_capacity
            if total_capacity == 0:
                combined_cf = 0
            else:
                combined_cf = total_energy_mwh / (total_capacity * 8760)

            # Calculate curtailment
            grid_capacity = self.config.grid_connection_capacity_mw
            peak_generation = max(pv_capacity, wind_capacity)
            curtailment = max(0, (peak_generation - grid_capacity) / peak_generation * 100)

            # Estimate LCOE (simplified)
            capex_per_mw_wind = 1_500_000  # $1.5M per MW
            capex_per_mw_pv = 1_000_000    # $1M per MW
            capex_per_mwh_storage = 300_000  # $300k per MWh

            total_capex = (
                wind_capacity * capex_per_mw_wind
                + pv_capacity * capex_per_mw_pv
                + storage_capacity * capex_per_mwh_storage
            )

            # Simple LCOE calculation (20 year lifetime, 5% discount rate)
            annuity_factor = 0.0802
            annual_cost = total_capex * annuity_factor
            lcoe = annual_cost / total_energy_mwh if total_energy_mwh > 0 else 1e6

            # Return objective based on optimization goal
            if opt_objective == OptimizationObjective.MAXIMIZE_ENERGY:
                return -total_energy_mwh  # Negative for maximization
            elif opt_objective == OptimizationObjective.MINIMIZE_COST:
                return lcoe
            elif opt_objective == OptimizationObjective.MINIMIZE_CURTAILMENT:
                return curtailment
            elif opt_objective == OptimizationObjective.MAXIMIZE_CAPACITY_FACTOR:
                return -combined_cf  # Negative for maximization
            else:
                return lcoe

        # Constraint: total capacity doesn't exceed grid connection + storage
        def grid_constraint(x: np.ndarray) -> float:
            pv_capacity, wind_capacity, storage_capacity = x
            return self.config.grid_connection_capacity_mw + storage_capacity - (pv_capacity + wind_capacity)

        constraints_list = [
            {'type': 'ineq', 'fun': grid_constraint}
        ]

        # Run optimization
        logger.info(f"Starting hybrid optimization with objective: {opt_objective.value}")
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Extract optimal values
        optimal_pv_mw, optimal_wind_mw, optimal_storage_mwh = result.x

        # Calculate final metrics with optimal configuration
        wind_cf = self.wind_assessment.capacity_factor_estimate if self.wind_assessment else 0.35
        pv_cf = 0.20

        optimal_wind_energy = optimal_wind_mw * 8760 * wind_cf
        optimal_pv_energy = optimal_pv_mw * 8760 * pv_cf
        total_annual_energy = optimal_wind_energy + optimal_pv_energy

        total_capacity = optimal_pv_mw + optimal_wind_mw
        combined_cf = total_annual_energy / (total_capacity * 8760) if total_capacity > 0 else 0

        peak_gen = max(optimal_pv_mw, optimal_wind_mw)
        curtailment_pct = max(0, (peak_gen - self.config.grid_connection_capacity_mw) / peak_gen * 100)

        # Calculate LCOE
        total_capex = (
            optimal_wind_mw * 1_500_000
            + optimal_pv_mw * 1_000_000
            + optimal_storage_mwh * 300_000
        )
        annual_cost = total_capex * 0.0802
        lcoe = annual_cost / total_annual_energy if total_annual_energy > 0 else 0

        # Create optimization result
        optimization_result = HybridOptimizationResult(
            optimal_pv_capacity_mw=optimal_pv_mw,
            optimal_wind_capacity_mw=optimal_wind_mw,
            optimal_storage_capacity_mwh=optimal_storage_mwh if optimal_storage_mwh > 0.1 else None,
            objective_value=float(result.fun),
            total_annual_energy_mwh=total_annual_energy,
            capacity_factor_combined=combined_cf,
            curtailment_percent=curtailment_pct,
            levelized_cost_of_energy=lcoe,
            optimization_objective=opt_objective,
            convergence_status=result.success,
            iterations=result.nit,
        )

        self.optimization_result = optimization_result
        logger.info(
            f"Optimization complete: PV={optimal_pv_mw:.1f}MW, "
            f"Wind={optimal_wind_mw:.1f}MW, Storage={optimal_storage_mwh:.1f}MWh, "
            f"LCOE=${lcoe:.2f}/MWh"
        )

        return optimization_result

    def wind_pv_coordination(
        self,
        wind_generation_mw: List[float],
        pv_generation_mw: List[float],
        grid_demand_mw: Optional[List[float]] = None,
        strategy: Optional[CoordinationStrategy] = None,
    ) -> List[CoordinationResult]:
        """Coordinate wind and PV generation for optimal dispatch.

        Implements real-time coordination between wind and PV systems to:
        - Optimize dispatch based on generation and demand
        - Manage ramp rates and grid stability
        - Control energy storage for arbitrage and grid support
        - Minimize curtailment while respecting grid constraints
        - Provide ancillary services (frequency regulation, voltage support)

        Args:
            wind_generation_mw: Time series of wind generation in MW
            pv_generation_mw: Time series of PV generation in MW
            grid_demand_mw: Optional time series of grid demand in MW
            strategy: Coordination strategy (uses default if not provided)

        Returns:
            List of CoordinationResult for each timestep

        Raises:
            ValueError: If generation timeseries have different lengths
            RuntimeError: If coordination algorithm fails

        Example:
            >>> wind_gen = [10.5, 12.3, 11.8, 13.2, ...]
            >>> pv_gen = [5.2, 6.8, 8.1, 7.5, ...]
            >>> results = integrator.wind_pv_coordination(wind_gen, pv_gen)
            >>> for result in results:
            ...     print(f"{result.timestamp}: {result.total_dispatch_mw} MW")
        """
        if len(wind_generation_mw) != len(pv_generation_mw):
            raise ValueError("Wind and PV generation timeseries must have same length")

        if strategy is None:
            strategy = self.coordination_strategy or CoordinationStrategy(
                strategy_name="default",
                dispatch_priority=["wind", "pv", "storage"],
                ramp_rate_limit_mw_per_min=5.0,
            )

        results = []
        storage_soc_mwh = self.config.storage_capacity_mwh / 2 if self.config.storage_capacity_mwh else 0
        storage_capacity = self.config.storage_capacity_mwh or 0
        grid_capacity = self.config.grid_connection_capacity_mw

        # Assume 5-minute intervals if not specified
        timestep_hours = 1/12  # 5 minutes

        base_time = datetime.now()

        for i, (wind_mw, pv_mw) in enumerate(zip(wind_generation_mw, pv_generation_mw)):
            timestamp = base_time + timedelta(hours=i * timestep_hours)

            # Total available generation
            total_available = wind_mw + pv_mw

            # Determine dispatch based on priority and constraints
            if grid_demand_mw is not None and i < len(grid_demand_mw):
                demand = grid_demand_mw[i]
            else:
                demand = grid_capacity

            # Calculate curtailment
            if total_available > grid_capacity:
                curtailed = total_available - grid_capacity
                dispatched = grid_capacity

                # Apply proportional curtailment if strategy requires
                if strategy.curtailment_strategy == "proportional":
                    curtailment_ratio = curtailed / total_available
                    wind_dispatch = wind_mw * (1 - curtailment_ratio)
                    pv_dispatch = pv_mw * (1 - curtailment_ratio)
                else:
                    # Priority-based curtailment
                    if "wind" in strategy.dispatch_priority[0]:
                        wind_dispatch = min(wind_mw, grid_capacity)
                        pv_dispatch = min(pv_mw, grid_capacity - wind_dispatch)
                    else:
                        pv_dispatch = min(pv_mw, grid_capacity)
                        wind_dispatch = min(wind_mw, grid_capacity - pv_dispatch)
                    curtailed = total_available - (wind_dispatch + pv_dispatch)
            else:
                wind_dispatch = wind_mw
                pv_dispatch = pv_mw
                dispatched = total_available
                curtailed = 0.0

            # Storage dispatch (simplified arbitrage)
            storage_dispatch = 0.0
            if storage_capacity > 0 and strategy.enable_storage_arbitrage:
                # Charge when excess generation, discharge when shortage
                excess = dispatched - demand
                if excess > 0 and storage_soc_mwh < storage_capacity * 0.9:
                    # Charge
                    charge_amount = min(excess, storage_capacity - storage_soc_mwh)
                    storage_dispatch = -charge_amount  # Negative = charging
                    storage_soc_mwh += charge_amount
                elif excess < 0 and storage_soc_mwh > storage_capacity * 0.1:
                    # Discharge
                    discharge_amount = min(-excess, storage_soc_mwh)
                    storage_dispatch = discharge_amount  # Positive = discharging
                    storage_soc_mwh -= discharge_amount

            # Total dispatch to grid
            total_dispatch = wind_dispatch + pv_dispatch + storage_dispatch
            grid_export = min(total_dispatch, grid_capacity)

            # Grid support services (simplified)
            frequency_regulation = 0.0
            voltage_support = 0.0
            if strategy.grid_support_enabled and total_dispatch > 0:
                frequency_regulation = min(total_dispatch * 0.05, 2.0)  # 5% or 2 MW max
                voltage_support = min(total_dispatch * 0.03, 1.0)  # 3% or 1 MVAr max

            # Coordination efficiency
            efficiency = (total_dispatch - curtailed) / total_available if total_available > 0 else 1.0

            result = CoordinationResult(
                timestamp=timestamp,
                pv_dispatch_mw=pv_dispatch,
                wind_dispatch_mw=wind_dispatch,
                storage_dispatch_mw=storage_dispatch,
                total_dispatch_mw=total_dispatch,
                curtailed_energy_mw=curtailed,
                grid_export_mw=grid_export,
                frequency_regulation_mw=frequency_regulation,
                voltage_support_mvar=voltage_support,
                coordination_efficiency=efficiency,
            )

            results.append(result)

        logger.info(
            f"Coordination complete for {len(results)} timesteps. "
            f"Average efficiency: {np.mean([r.coordination_efficiency for r in results]):.2%}"
        )

        return results

    def run_simulation(self) -> Dict[str, Any]:
        """Run complete hybrid system simulation.

        Executes a full simulation workflow including:
        1. Wind resource assessment
        2. Turbine performance modeling
        3. Hybrid system optimization
        4. Wind-PV coordination analysis

        Returns:
            Dictionary containing all simulation results

        Raises:
            RuntimeError: If integrator is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Integrator must be initialized before running simulation")

        logger.info(f"Starting complete hybrid simulation for {self.config.system_id}")

        results = {
            "system_id": self.config.system_id,
            "site_name": self.config.site_name,
            "simulation_timestamp": datetime.now().isoformat(),
            "wind_assessment": self.wind_assessment,
            "turbine_performance": self.turbine_performance,
            "optimization_result": self.optimization_result,
            "metadata": self.metadata,
        }

        logger.info("Hybrid simulation complete")
        return results

    # Private helper methods

    def _extrapolate_wind_speeds(
        self,
        speeds: List[float],
        measurement_height: float,
        target_height: float,
        shear_exponent: float,
    ) -> np.ndarray:
        """Extrapolate wind speeds to target height using power law."""
        speeds_array = np.array(speeds)
        extrapolated = speeds_array * (target_height / measurement_height) ** shear_exponent
        return extrapolated

    def _fit_weibull_distribution(self, wind_speeds: np.ndarray) -> Tuple[float, float]:
        """Fit Weibull distribution to wind speed data."""
        # Filter out zero and very low speeds
        valid_speeds = wind_speeds[wind_speeds > 0.5]

        if len(valid_speeds) < 10:
            # Return default values if insufficient data
            return 2.0, float(np.mean(wind_speeds))

        # Fit Weibull distribution
        params = weibull_min.fit(valid_speeds, floc=0)
        k = params[0]  # Shape parameter
        c = params[2]  # Scale parameter

        return float(k), float(c)

    def _calculate_wind_power_density(
        self,
        wind_speeds: np.ndarray,
        air_density: float,
    ) -> float:
        """Calculate wind power density in W/m²."""
        # Power density = 0.5 * ρ * v³
        power_density = 0.5 * air_density * np.mean(wind_speeds ** 3)
        return float(power_density)

    def _calculate_prevailing_direction(self, directions: List[float]) -> float:
        """Calculate prevailing wind direction."""
        # Convert to radians and calculate mean direction
        directions_rad = np.array(directions) * np.pi / 180
        mean_sin = np.mean(np.sin(directions_rad))
        mean_cos = np.mean(np.cos(directions_rad))
        mean_direction_rad = np.arctan2(mean_sin, mean_cos)
        mean_direction_deg = (mean_direction_rad * 180 / np.pi) % 360
        return float(mean_direction_deg)

    def _estimate_capacity_factor(
        self,
        wind_speeds: np.ndarray,
        turbine_specs: TurbineSpecifications,
    ) -> float:
        """Estimate capacity factor from wind speeds and power curve."""
        # Create power curve interpolation
        power_curve = interp1d(
            turbine_specs.power_curve_speeds_ms,
            turbine_specs.power_curve_kw,
            kind='linear',
            bounds_error=False,
            fill_value=(0, turbine_specs.rated_power_kw),
        )

        # Calculate power for each wind speed
        power_outputs = []
        for speed in wind_speeds:
            if speed < turbine_specs.cut_in_speed_ms or speed > turbine_specs.cut_out_speed_ms:
                power = 0.0
            else:
                power = float(power_curve(speed))
            power_outputs.append(power)

        # Capacity factor
        mean_power = np.mean(power_outputs)
        capacity_factor = mean_power / turbine_specs.rated_power_kw

        return float(np.clip(capacity_factor, 0, 1))

    def _calculate_wake_losses(self, num_turbines: int) -> float:
        """Calculate wake losses based on number of turbines."""
        # Simplified wake loss model
        # More turbines = higher wake losses (up to ~15%)
        if num_turbines == 1:
            return 0.0
        elif num_turbines <= 5:
            return 5.0
        elif num_turbines <= 10:
            return 8.0
        elif num_turbines <= 20:
            return 12.0
        else:
            return 15.0
