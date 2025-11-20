"""
Hydrogen System Integrator

This module provides the core HydrogenIntegrator class for modeling and analyzing
hydrogen systems including electrolyzers, storage, fuel cells, and power-to-X pathways.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from .models import (
    ElectrolyzerConfig,
    ElectrolyzerResults,
    ElectrolyzerType,
    StorageConfig,
    StorageResults,
    StorageType,
    FuelCellConfig,
    FuelCellResults,
    FuelCellType,
    PowerToXConfig,
    PowerToXResults,
    PowerToXPathway,
)


class HydrogenIntegrator:
    """
    Comprehensive hydrogen system integration and analysis.

    The HydrogenIntegrator provides production-ready methods for modeling
    electrolyzers, designing hydrogen storage systems, integrating fuel cells,
    and analyzing power-to-X pathways for renewable energy integration.

    This class handles:
    - Dynamic electrolyzer modeling with degradation
    - Hydrogen storage design and optimization
    - Fuel cell integration and performance analysis
    - Power-to-X pathway techno-economic analysis

    Example:
        >>> from pv_circularity_simulator.hydrogen import HydrogenIntegrator
        >>> from pv_circularity_simulator.hydrogen import ElectrolyzerConfig, ElectrolyzerType
        >>>
        >>> integrator = HydrogenIntegrator()
        >>> config = ElectrolyzerConfig(
        ...     electrolyzer_type=ElectrolyzerType.PEM,
        ...     rated_power_kw=1000.0,
        ...     efficiency=0.68
        ... )
        >>> power_profile = [800.0] * 8760  # 1 year hourly data
        >>> results = integrator.electrolyzer_modeling(
        ...     config=config,
        ...     power_input_profile=power_profile,
        ...     timestep_hours=1.0
        ... )
        >>> print(f"Annual H2 production: {results.annual_h2_production_kg:.2f} kg")
    """

    def __init__(
        self,
        discount_rate: float = 0.05,
        electricity_price_kwh: float = 0.05,
        project_lifetime_years: int = 25,
    ):
        """
        Initialize the HydrogenIntegrator.

        Args:
            discount_rate: Annual discount rate for economic calculations (default: 0.05)
            electricity_price_kwh: Electricity price in $/kWh (default: 0.05)
            project_lifetime_years: Project lifetime for LCOH calculations (default: 25)
        """
        self.discount_rate = discount_rate
        self.electricity_price_kwh = electricity_price_kwh
        self.project_lifetime_years = project_lifetime_years

    def electrolyzer_modeling(
        self,
        config: ElectrolyzerConfig,
        power_input_profile: List[float],
        timestep_hours: float = 1.0,
        ambient_temperature_c: Optional[List[float]] = None,
    ) -> ElectrolyzerResults:
        """
        Model electrolyzer performance with dynamic operation and degradation.

        This method simulates electrolyzer operation over time, accounting for:
        - Part-load efficiency curves
        - Dynamic ramping and response times
        - Temperature effects on performance
        - Stack degradation over lifetime
        - Economic metrics (LCOH)

        Args:
            config: Electrolyzer configuration parameters
            power_input_profile: Time series of input power in kW
            timestep_hours: Time step duration in hours (default: 1.0)
            ambient_temperature_c: Optional ambient temperature profile in Celsius

        Returns:
            ElectrolyzerResults: Comprehensive results including H2 production,
                efficiency, degradation, and economic metrics

        Raises:
            ValueError: If power_input_profile is empty or contains negative values

        Example:
            >>> config = ElectrolyzerConfig(
            ...     electrolyzer_type=ElectrolyzerType.PEM,
            ...     rated_power_kw=1000.0,
            ...     efficiency=0.68,
            ...     min_load_fraction=0.1
            ... )
            >>> # Simulate variable renewable power input
            >>> power_profile = [500 + 300 * np.sin(i/24 * 2*np.pi)
            ...                  for i in range(8760)]
            >>> results = integrator.electrolyzer_modeling(config, power_profile)
        """
        if not power_input_profile:
            raise ValueError("power_input_profile cannot be empty")
        if any(p < 0 for p in power_input_profile):
            raise ValueError("power_input_profile cannot contain negative values")

        n_steps = len(power_input_profile)
        h2_lhv_kwh_per_kg = 33.33  # Lower heating value of hydrogen

        # Initialize tracking variables
        total_h2_kg = 0.0
        total_energy_kwh = 0.0
        operating_hours = 0.0
        equivalent_full_load_hours = 0.0
        degradation_factor = 1.0

        # Temperature coefficients (technology-specific)
        temp_coeff = self._get_temperature_coefficient(config.electrolyzer_type)

        for step in range(n_steps):
            power_available = power_input_profile[step]

            # Calculate load fraction
            load_fraction = power_available / config.rated_power_kw

            # Check operating limits
            if load_fraction < config.min_load_fraction:
                continue  # Not operating
            if load_fraction > config.max_load_fraction:
                load_fraction = config.max_load_fraction
                power_available = config.rated_power_kw * load_fraction

            # Part-load efficiency curve
            efficiency = self._calculate_part_load_efficiency(
                config=config,
                load_fraction=load_fraction,
                degradation_factor=degradation_factor,
            )

            # Temperature correction if ambient temperature provided
            if ambient_temperature_c is not None and step < len(ambient_temperature_c):
                temp_delta = ambient_temperature_c[step] - config.operating_temperature_c
                efficiency *= (1 + temp_coeff * temp_delta)
                efficiency = np.clip(efficiency, 0.1, 0.95)

            # Calculate H2 production for this timestep
            h2_produced = (power_available * efficiency * timestep_hours) / h2_lhv_kwh_per_kg

            # Update totals
            total_h2_kg += h2_produced
            total_energy_kwh += power_available * timestep_hours
            operating_hours += timestep_hours
            equivalent_full_load_hours += load_fraction * timestep_hours

            # Update degradation (cumulative)
            degradation_factor -= (
                config.degradation_rate_per_year * (timestep_hours / 8760)
            )
            degradation_factor = max(degradation_factor, 0.7)  # Min 70% of initial

        # Calculate average metrics
        average_efficiency = (
            (total_h2_kg * h2_lhv_kwh_per_kg) / total_energy_kwh
            if total_energy_kwh > 0
            else 0.0
        )
        capacity_factor = equivalent_full_load_hours / (n_steps * timestep_hours)

        # Calculate specific energy consumption
        specific_energy_consumption = (
            total_energy_kwh / total_h2_kg if total_h2_kg > 0 else 0.0
        )

        # Economic calculations
        levelized_cost_h2 = self._calculate_lcoh_electrolyzer(
            config=config,
            annual_h2_production_kg=total_h2_kg,
            annual_energy_kwh=total_energy_kwh,
            capacity_factor=capacity_factor,
        )

        # Annual production (scale based on profile duration)
        hours_per_year = 8760
        profile_duration_hours = n_steps * timestep_hours
        annual_h2_production_kg = (
            total_h2_kg * (hours_per_year / profile_duration_hours)
            if profile_duration_hours > 0
            else 0.0
        )

        # Additional performance metrics
        performance_metrics = {
            "stack_utilization": capacity_factor,
            "degradation_loss_percent": (1 - degradation_factor) * 100,
            "average_load_fraction": equivalent_full_load_hours / operating_hours
            if operating_hours > 0
            else 0.0,
            "starts_count": self._count_start_stop_cycles(
                power_input_profile, config.rated_power_kw, config.min_load_fraction
            ),
            "rated_h2_production_kg_h": config.h2_production_rate_kg_h,
        }

        return ElectrolyzerResults(
            h2_production_kg=total_h2_kg,
            energy_consumption_kwh=total_energy_kwh,
            average_efficiency=average_efficiency,
            capacity_factor=capacity_factor,
            degradation_factor=degradation_factor,
            operating_hours=operating_hours,
            equivalent_full_load_hours=equivalent_full_load_hours,
            levelized_cost_h2=levelized_cost_h2,
            specific_energy_consumption=specific_energy_consumption,
            annual_h2_production_kg=annual_h2_production_kg,
            performance_metrics=performance_metrics,
        )

    def h2_storage_design(
        self,
        config: StorageConfig,
        charge_profile: List[float],
        discharge_profile: List[float],
        timestep_hours: float = 1.0,
        initial_soc_fraction: float = 0.5,
    ) -> StorageResults:
        """
        Design and simulate hydrogen storage system operation.

        This method performs detailed storage system simulation including:
        - State of charge (SOC) dynamics
        - Charging and discharging efficiency
        - Self-discharge losses
        - Cycling analysis
        - Storage economics

        Args:
            config: Storage system configuration
            charge_profile: Time series of H2 charging rate in kg/h
            discharge_profile: Time series of H2 discharge rate in kg/h
            timestep_hours: Time step duration in hours (default: 1.0)
            initial_soc_fraction: Initial state of charge (0-1, default: 0.5)

        Returns:
            StorageResults: Comprehensive storage performance and economic metrics

        Raises:
            ValueError: If profiles have different lengths or invalid SOC

        Example:
            >>> storage_config = StorageConfig(
            ...     storage_type=StorageType.COMPRESSED_GAS,
            ...     capacity_kg=1000.0,
            ...     pressure_bar=350.0,
            ...     charging_rate_kg_h=50.0,
            ...     discharging_rate_kg_h=50.0
            ... )
            >>> charge = [10.0] * 4380 + [0.0] * 4380  # Charge first half
            >>> discharge = [0.0] * 4380 + [10.0] * 4380  # Discharge second half
            >>> results = integrator.h2_storage_design(
            ...     storage_config, charge, discharge
            ... )
        """
        if len(charge_profile) != len(discharge_profile):
            raise ValueError("charge_profile and discharge_profile must have same length")
        if not (0 <= initial_soc_fraction <= 1):
            raise ValueError("initial_soc_fraction must be between 0 and 1")

        n_steps = len(charge_profile)

        # Initialize state variables
        soc_kg = config.capacity_kg * initial_soc_fraction
        soc_history = []
        total_charged = 0.0
        total_discharged = 0.0
        total_losses = 0.0
        cycle_count = 0.0

        # Track SOC for cycle counting
        previous_soc = soc_kg
        soc_max_this_cycle = soc_kg
        soc_min_this_cycle = soc_kg

        for step in range(n_steps):
            charge_rate = charge_profile[step]
            discharge_rate = discharge_profile[step]

            # Apply charging (with efficiency losses)
            charge_amount = min(
                charge_rate * timestep_hours,
                config.charging_rate_kg_h * timestep_hours,
                (config.capacity_kg * config.max_soc_fraction - soc_kg)
                / np.sqrt(config.round_trip_efficiency),
            )
            charge_stored = charge_amount * np.sqrt(config.round_trip_efficiency)

            # Apply discharging (with efficiency losses)
            discharge_demand = min(
                discharge_rate * timestep_hours,
                config.discharging_rate_kg_h * timestep_hours,
            )
            discharge_available = (
                soc_kg - config.capacity_kg * config.min_soc_fraction
            ) * np.sqrt(config.round_trip_efficiency)
            discharge_amount = min(discharge_demand, max(0, discharge_available))
            discharge_from_storage = discharge_amount / np.sqrt(
                config.round_trip_efficiency
            )

            # Apply self-discharge
            self_discharge_rate = config.self_discharge_rate_per_day / 24  # per hour
            self_discharge_loss = soc_kg * self_discharge_rate * timestep_hours

            # Update SOC
            soc_kg += charge_stored - discharge_from_storage - self_discharge_loss
            soc_kg = np.clip(
                soc_kg,
                config.capacity_kg * config.min_soc_fraction,
                config.capacity_kg * config.max_soc_fraction,
            )

            # Track totals
            total_charged += charge_amount
            total_discharged += discharge_amount
            total_losses += (charge_amount - charge_stored) + (
                discharge_from_storage - discharge_amount
            ) + self_discharge_loss

            # Cycle counting (rainflow simplified: count full depth cycles)
            if soc_kg > soc_max_this_cycle:
                soc_max_this_cycle = soc_kg
            if soc_kg < soc_min_this_cycle:
                soc_min_this_cycle = soc_kg

            # Detect cycle completion (SOC crosses initial point)
            if (previous_soc <= initial_soc_fraction * config.capacity_kg < soc_kg) or (
                previous_soc >= initial_soc_fraction * config.capacity_kg > soc_kg
            ):
                cycle_depth = (soc_max_this_cycle - soc_min_this_cycle) / config.capacity_kg
                cycle_count += cycle_depth
                soc_max_this_cycle = soc_kg
                soc_min_this_cycle = soc_kg

            previous_soc = soc_kg
            soc_history.append(soc_kg / config.capacity_kg)

        # Calculate average metrics
        average_soc = np.mean(soc_history)
        average_efficiency = (
            total_discharged / total_charged if total_charged > 0 else 0.0
        )
        capacity_utilization = (
            max(soc_history) - min(soc_history) if soc_history else 0.0
        )

        # Economic calculations
        levelized_cost_storage = self._calculate_lcoh_storage(
            config=config, annual_throughput_kg=total_discharged
        )

        # Additional metrics
        storage_metrics = {
            "peak_soc": max(soc_history) if soc_history else 0.0,
            "min_soc": min(soc_history) if soc_history else 0.0,
            "self_discharge_losses_kg": sum(
                soc_history[i] * config.capacity_kg * config.self_discharge_rate_per_day / 24 * timestep_hours
                for i in range(len(soc_history))
            ),
            "efficiency_losses_kg": total_losses,
            "average_charge_rate_kg_h": total_charged / (n_steps * timestep_hours)
            if n_steps > 0
            else 0.0,
            "average_discharge_rate_kg_h": total_discharged / (n_steps * timestep_hours)
            if n_steps > 0
            else 0.0,
        }

        return StorageResults(
            total_capacity_kg=config.capacity_kg,
            usable_capacity_kg=config.usable_capacity_kg,
            average_soc=average_soc,
            total_charged_kg=total_charged,
            total_discharged_kg=total_discharged,
            total_losses_kg=total_losses,
            average_efficiency=average_efficiency,
            cycling_count=cycle_count,
            capacity_utilization=capacity_utilization,
            levelized_cost_storage=levelized_cost_storage,
            storage_metrics=storage_metrics,
        )

    def fuel_cell_integration(
        self,
        config: FuelCellConfig,
        power_demand_profile: List[float],
        h2_supply_profile: Optional[List[float]] = None,
        timestep_hours: float = 1.0,
        heat_demand_profile: Optional[List[float]] = None,
    ) -> FuelCellResults:
        """
        Integrate and analyze fuel cell system performance.

        This method simulates fuel cell operation including:
        - Part-load efficiency characteristics
        - Hydrogen consumption dynamics
        - Stack degradation modeling
        - Combined heat and power (CHP) if enabled
        - Economic analysis (LCOE)

        Args:
            config: Fuel cell configuration parameters
            power_demand_profile: Time series of electrical power demand in kW
            h2_supply_profile: Optional H2 availability in kg/h (unlimited if None)
            timestep_hours: Time step duration in hours (default: 1.0)
            heat_demand_profile: Optional thermal demand for CHP analysis in kW

        Returns:
            FuelCellResults: Comprehensive fuel cell performance and economics

        Raises:
            ValueError: If power_demand_profile is empty or invalid

        Example:
            >>> fc_config = FuelCellConfig(
            ...     fuel_cell_type=FuelCellType.PEMFC,
            ...     rated_power_kw=500.0,
            ...     efficiency=0.55,
            ...     cogeneration_enabled=True
            ... )
            >>> demand = [300.0] * 8760  # Constant 300 kW demand
            >>> results = integrator.fuel_cell_integration(fc_config, demand)
            >>> print(f"Total electricity: {results.electrical_output_kwh:.2f} kWh")
            >>> print(f"Total heat: {results.thermal_output_kwh:.2f} kWh")
        """
        if not power_demand_profile:
            raise ValueError("power_demand_profile cannot be empty")
        if any(p < 0 for p in power_demand_profile):
            raise ValueError("power_demand_profile cannot contain negative values")

        n_steps = len(power_demand_profile)
        h2_lhv_kwh_per_kg = 33.33

        # Initialize tracking variables
        total_electrical_kwh = 0.0
        total_thermal_kwh = 0.0
        total_h2_consumed = 0.0
        operating_hours = 0.0
        degradation_factor = 1.0
        equivalent_full_load_hours = 0.0

        for step in range(n_steps):
            power_demand = power_demand_profile[step]

            # Calculate load fraction
            load_fraction = power_demand / config.rated_power_kw

            # Check operating limits
            if load_fraction < config.min_load_fraction:
                continue  # Not operating
            if load_fraction > config.max_load_fraction:
                load_fraction = config.max_load_fraction
                power_demand = config.rated_power_kw * load_fraction

            # Part-load efficiency
            efficiency = self._calculate_fc_part_load_efficiency(
                config=config,
                load_fraction=load_fraction,
                degradation_factor=degradation_factor,
            )

            # Calculate H2 consumption
            h2_required_kg_h = power_demand / (efficiency * h2_lhv_kwh_per_kg)
            h2_consumed_timestep = h2_required_kg_h * timestep_hours

            # Check H2 availability
            if h2_supply_profile is not None and step < len(h2_supply_profile):
                h2_available = h2_supply_profile[step] * timestep_hours
                if h2_consumed_timestep > h2_available:
                    # H2 limited operation
                    h2_consumed_timestep = h2_available
                    power_output = h2_consumed_timestep * efficiency * h2_lhv_kwh_per_kg / timestep_hours
                else:
                    power_output = power_demand
            else:
                power_output = power_demand

            # Calculate thermal output (if CHP enabled)
            thermal_output = 0.0
            if config.cogeneration_enabled:
                total_input_power = power_output / efficiency
                waste_heat = total_input_power - power_output
                thermal_output = waste_heat * config.heat_recovery_fraction

                # Limit to heat demand if specified
                if heat_demand_profile is not None and step < len(heat_demand_profile):
                    thermal_output = min(thermal_output, heat_demand_profile[step])

            # Update totals
            total_electrical_kwh += power_output * timestep_hours
            total_thermal_kwh += thermal_output * timestep_hours
            total_h2_consumed += h2_consumed_timestep
            operating_hours += timestep_hours
            equivalent_full_load_hours += load_fraction * timestep_hours

            # Update degradation
            degradation_factor -= config.degradation_rate_per_hour * timestep_hours
            degradation_factor = max(degradation_factor, 0.7)  # Min 70% of initial

        # Calculate metrics
        average_efficiency = (
            total_electrical_kwh / (total_h2_consumed * h2_lhv_kwh_per_kg)
            if total_h2_consumed > 0
            else 0.0
        )
        capacity_factor = equivalent_full_load_hours / (n_steps * timestep_hours)
        specific_h2_consumption = (
            total_h2_consumed / total_electrical_kwh if total_electrical_kwh > 0 else 0.0
        )

        # Cogeneration efficiency
        cogeneration_efficiency = None
        if config.cogeneration_enabled and total_h2_consumed > 0:
            total_useful_output = total_electrical_kwh + total_thermal_kwh
            cogeneration_efficiency = total_useful_output / (
                total_h2_consumed * h2_lhv_kwh_per_kg
            )

        # Economic calculations
        levelized_cost_electricity = self._calculate_lcoe_fuel_cell(
            config=config,
            annual_electrical_output_kwh=total_electrical_kwh,
            annual_h2_consumption_kg=total_h2_consumed,
            capacity_factor=capacity_factor,
        )

        # Additional performance metrics
        performance_metrics = {
            "stack_utilization": capacity_factor,
            "degradation_loss_percent": (1 - degradation_factor) * 100,
            "average_load_fraction": equivalent_full_load_hours / operating_hours
            if operating_hours > 0
            else 0.0,
            "thermal_efficiency": (
                total_thermal_kwh / (total_h2_consumed * h2_lhv_kwh_per_kg)
                if total_h2_consumed > 0 and config.cogeneration_enabled
                else 0.0
            ),
            "rated_h2_consumption_kg_h": config.h2_consumption_rate_kg_h,
        }

        return FuelCellResults(
            electrical_output_kwh=total_electrical_kwh,
            thermal_output_kwh=total_thermal_kwh,
            h2_consumed_kg=total_h2_consumed,
            average_efficiency=average_efficiency,
            capacity_factor=capacity_factor,
            operating_hours=operating_hours,
            degradation_factor=degradation_factor,
            levelized_cost_electricity=levelized_cost_electricity,
            specific_h2_consumption=specific_h2_consumption,
            cogeneration_efficiency=cogeneration_efficiency,
            performance_metrics=performance_metrics,
        )

    def power_to_x_analysis(
        self,
        config: PowerToXConfig,
        power_input_profile: List[float],
        timestep_hours: float = 1.0,
        co2_availability_profile: Optional[List[float]] = None,
        grid_carbon_intensity: float = 0.5,
    ) -> PowerToXResults:
        """
        Analyze Power-to-X conversion pathways and techno-economics.

        This comprehensive method evaluates power-to-X pathways including:
        - Electrolyzer operation for H2 production
        - Catalytic conversion to target products
        - Mass and energy balances
        - CO2 and N2 requirements
        - Economic analysis (levelized cost)
        - Environmental impact (carbon intensity)

        Args:
            config: Power-to-X pathway configuration
            power_input_profile: Time series of input power in kW
            timestep_hours: Time step duration in hours (default: 1.0)
            co2_availability_profile: Optional CO2 availability in kg/h
            grid_carbon_intensity: Grid carbon intensity in kg CO2/kWh (default: 0.5)

        Returns:
            PowerToXResults: Comprehensive pathway analysis including production,
                efficiency, economics, and environmental metrics

        Raises:
            ValueError: If required inputs are missing for the pathway

        Example:
            >>> from pv_circularity_simulator.hydrogen import PowerToXPathway
            >>> ptx_config = PowerToXConfig(
            ...     pathway=PowerToXPathway.POWER_TO_METHANOL,
            ...     electrolyzer_config=ElectrolyzerConfig(
            ...         electrolyzer_type=ElectrolyzerType.PEM,
            ...         rated_power_kw=5000.0,
            ...         efficiency=0.68
            ...     ),
            ...     conversion_efficiency=0.75,
            ...     co2_source="DAC"
            ... )
            >>> power = [4000.0] * 8760
            >>> co2_available = [100.0] * 8760  # kg/h
            >>> results = integrator.power_to_x_analysis(
            ...     ptx_config, power, co2_availability_profile=co2_available
            ... )
        """
        if not power_input_profile:
            raise ValueError("power_input_profile cannot be empty")

        # Check pathway requirements
        if config.requires_co2 and co2_availability_profile is None:
            raise ValueError(
                f"Pathway {config.pathway} requires CO2 input. "
                "Provide co2_availability_profile."
            )

        # Step 1: Electrolyzer modeling for H2 production
        electrolyzer_results = self.electrolyzer_modeling(
            config=config.electrolyzer_config,
            power_input_profile=power_input_profile,
            timestep_hours=timestep_hours,
        )

        h2_produced_kg = electrolyzer_results.h2_production_kg
        h2_lhv_kwh_per_kg = 33.33

        # Step 2: Conversion to final product
        conversion_results = self._convert_h2_to_product(
            pathway=config.pathway,
            h2_available_kg=h2_produced_kg,
            conversion_efficiency=config.conversion_efficiency,
            co2_availability_profile=co2_availability_profile,
            timestep_hours=timestep_hours,
        )

        product_output_kg = conversion_results["product_kg"]
        co2_consumed_kg = conversion_results.get("co2_consumed_kg", 0.0)
        n2_consumed_kg = conversion_results.get("n2_consumed_kg", 0.0)

        # Step 3: Energy balance
        energy_input_kwh = electrolyzer_results.energy_consumption_kwh
        overall_efficiency = (
            (product_output_kg * config.product_lhv_kwh_per_kg) / energy_input_kwh
            if energy_input_kwh > 0
            else 0.0
        )
        specific_energy_consumption = (
            energy_input_kwh / product_output_kg if product_output_kg > 0 else 0.0
        )

        # Step 4: Economic analysis
        levelized_cost_product = self._calculate_lcop(
            config=config,
            electrolyzer_results=electrolyzer_results,
            annual_product_output_kg=product_output_kg,
            annual_co2_consumed_kg=co2_consumed_kg,
        )

        # Step 5: Environmental analysis
        carbon_intensity = self._calculate_carbon_intensity(
            pathway=config.pathway,
            energy_input_kwh=energy_input_kwh,
            co2_consumed_kg=co2_consumed_kg,
            product_output_kg=product_output_kg,
            grid_carbon_intensity=grid_carbon_intensity,
        )

        # Capacity factor
        capacity_factor = electrolyzer_results.capacity_factor

        # Economic metrics
        economic_metrics = {
            "electrolyzer_capex": (
                config.electrolyzer_config.rated_power_kw
                * config.electrolyzer_config.capex_per_kw
            ),
            "conversion_unit_capex": (
                config.electrolyzer_config.rated_power_kw * config.capex_conversion_per_kw
            ),
            "total_capex": (
                config.electrolyzer_config.rated_power_kw
                * config.electrolyzer_config.capex_per_kw
                + config.electrolyzer_config.rated_power_kw
                * config.capex_conversion_per_kw
            ),
            "electricity_cost_per_kg_product": (
                energy_input_kwh * self.electricity_price_kwh / product_output_kg
                if product_output_kg > 0
                else 0.0
            ),
            "co2_cost_per_kg_product": (
                co2_consumed_kg * config.co2_capture_cost_per_ton / 1000 / product_output_kg
                if product_output_kg > 0
                else 0.0
            ),
        }

        # Environmental metrics
        environmental_metrics = {
            "carbon_intensity_kg_co2_per_kg": carbon_intensity,
            "renewable_energy_fraction": 1.0,  # Assuming 100% renewable input
            "co2_utilization_kg": co2_consumed_kg,
            "avoided_emissions_kg_co2": self._calculate_avoided_emissions(
                pathway=config.pathway,
                product_output_kg=product_output_kg,
                carbon_intensity=carbon_intensity,
            ),
        }

        return PowerToXResults(
            product_output_kg=product_output_kg,
            h2_intermediate_kg=h2_produced_kg,
            energy_input_kwh=energy_input_kwh,
            co2_consumed_kg=co2_consumed_kg,
            n2_consumed_kg=n2_consumed_kg,
            overall_efficiency=overall_efficiency,
            specific_energy_consumption=specific_energy_consumption,
            levelized_cost_product=levelized_cost_product,
            carbon_intensity=carbon_intensity,
            capacity_factor=capacity_factor,
            economic_metrics=economic_metrics,
            environmental_metrics=environmental_metrics,
        )

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _get_temperature_coefficient(self, electrolyzer_type: ElectrolyzerType) -> float:
        """Get temperature coefficient for efficiency adjustment."""
        coefficients = {
            ElectrolyzerType.PEM: 0.001,  # 0.1% per °C
            ElectrolyzerType.ALKALINE: 0.0015,  # 0.15% per °C
            ElectrolyzerType.SOEC: 0.002,  # 0.2% per °C (more sensitive)
            ElectrolyzerType.AEM: 0.0012,
        }
        return coefficients.get(electrolyzer_type, 0.001)

    def _calculate_part_load_efficiency(
        self,
        config: ElectrolyzerConfig,
        load_fraction: float,
        degradation_factor: float,
    ) -> float:
        """Calculate part-load efficiency using empirical curve."""
        # Typical part-load curve: efficiency drops at low loads
        # Using polynomial approximation
        base_efficiency = config.efficiency

        if config.electrolyzer_type == ElectrolyzerType.PEM:
            # PEM has better part-load performance
            efficiency_factor = 0.95 + 0.05 * load_fraction
        elif config.electrolyzer_type == ElectrolyzerType.ALKALINE:
            # Alkaline has more pronounced part-load losses
            efficiency_factor = 0.85 + 0.15 * load_fraction
        elif config.electrolyzer_type == ElectrolyzerType.SOEC:
            # SOEC has good part-load performance
            efficiency_factor = 0.93 + 0.07 * load_fraction
        else:
            efficiency_factor = 0.90 + 0.10 * load_fraction

        return base_efficiency * efficiency_factor * degradation_factor

    def _calculate_fc_part_load_efficiency(
        self,
        config: FuelCellConfig,
        load_fraction: float,
        degradation_factor: float,
    ) -> float:
        """Calculate fuel cell part-load efficiency."""
        base_efficiency = config.efficiency

        if config.fuel_cell_type == FuelCellType.PEMFC:
            # PEMFC efficiency peaks around 50-70% load
            if load_fraction < 0.5:
                efficiency_factor = 0.85 + 0.3 * load_fraction
            else:
                efficiency_factor = 1.0 - 0.05 * (load_fraction - 0.5)
        elif config.fuel_cell_type == FuelCellType.SOFC:
            # SOFC has flatter efficiency curve
            efficiency_factor = 0.92 + 0.08 * load_fraction
        else:
            efficiency_factor = 0.88 + 0.12 * load_fraction

        return base_efficiency * efficiency_factor * degradation_factor

    def _count_start_stop_cycles(
        self, power_profile: List[float], rated_power: float, min_load: float
    ) -> int:
        """Count start-stop cycles in operation."""
        threshold = rated_power * min_load
        was_on = power_profile[0] >= threshold
        starts = 0

        for power in power_profile[1:]:
            is_on = power >= threshold
            if is_on and not was_on:
                starts += 1
            was_on = is_on

        return starts

    def _calculate_lcoh_electrolyzer(
        self,
        config: ElectrolyzerConfig,
        annual_h2_production_kg: float,
        annual_energy_kwh: float,
        capacity_factor: float,
    ) -> float:
        """Calculate levelized cost of hydrogen from electrolyzer."""
        if annual_h2_production_kg <= 0:
            return 0.0

        # CAPEX
        capex = config.rated_power_kw * config.capex_per_kw

        # Annuity factor for capital recovery
        annuity_factor = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.project_lifetime_years
        ) / (((1 + self.discount_rate) ** self.project_lifetime_years) - 1)

        annual_capital_cost = capex * annuity_factor

        # OPEX
        annual_opex = capex * config.opex_fraction

        # Electricity cost
        annual_electricity_cost = annual_energy_kwh * self.electricity_price_kwh

        # Stack replacement (if needed)
        stack_replacement_cost = 0.0
        if capacity_factor > 0:
            annual_operating_hours = 8760 * capacity_factor
            replacements_needed = (
                self.project_lifetime_years * annual_operating_hours
            ) / config.stack_lifetime_hours
            if replacements_needed > 1:
                stack_replacement_cost = (
                    capex * 0.4 * (replacements_needed - 1) * annuity_factor
                )  # Stack ~40% of CAPEX

        # Total annual cost
        total_annual_cost = (
            annual_capital_cost
            + annual_opex
            + annual_electricity_cost
            + stack_replacement_cost
        )

        # LCOH
        lcoh = total_annual_cost / annual_h2_production_kg

        return lcoh

    def _calculate_lcoh_storage(
        self, config: StorageConfig, annual_throughput_kg: float
    ) -> float:
        """Calculate levelized cost of hydrogen storage."""
        if annual_throughput_kg <= 0:
            return 0.0

        # CAPEX
        capex = config.capacity_kg * config.capex_per_kg

        # Annuity factor
        annuity_factor = (
            self.discount_rate * (1 + self.discount_rate) ** config.lifetime_years
        ) / (((1 + self.discount_rate) ** config.lifetime_years) - 1)

        annual_capital_cost = capex * annuity_factor

        # OPEX
        annual_opex = capex * config.opex_fraction

        # Total annual cost
        total_annual_cost = annual_capital_cost + annual_opex

        # Levelized cost per kg throughput
        lcos = total_annual_cost / annual_throughput_kg

        return lcos

    def _calculate_lcoe_fuel_cell(
        self,
        config: FuelCellConfig,
        annual_electrical_output_kwh: float,
        annual_h2_consumption_kg: float,
        capacity_factor: float,
    ) -> float:
        """Calculate levelized cost of electricity from fuel cell."""
        if annual_electrical_output_kwh <= 0:
            return 0.0

        # CAPEX
        capex = config.rated_power_kw * config.capex_per_kw

        # Annuity factor
        annuity_factor = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.project_lifetime_years
        ) / (((1 + self.discount_rate) ** self.project_lifetime_years) - 1)

        annual_capital_cost = capex * annuity_factor

        # OPEX
        annual_opex = capex * config.opex_fraction

        # Hydrogen cost (assume $3/kg as placeholder)
        h2_price_per_kg = 3.0
        annual_h2_cost = annual_h2_consumption_kg * h2_price_per_kg

        # Stack replacement
        stack_replacement_cost = 0.0
        if capacity_factor > 0:
            annual_operating_hours = 8760 * capacity_factor
            replacements_needed = (
                self.project_lifetime_years * annual_operating_hours
            ) / config.stack_lifetime_hours
            if replacements_needed > 1:
                stack_replacement_cost = (
                    capex * 0.5 * (replacements_needed - 1) * annuity_factor
                )

        # Total annual cost
        total_annual_cost = (
            annual_capital_cost + annual_opex + annual_h2_cost + stack_replacement_cost
        )

        # LCOE
        lcoe = total_annual_cost / annual_electrical_output_kwh

        return lcoe

    def _calculate_lcop(
        self,
        config: PowerToXConfig,
        electrolyzer_results: ElectrolyzerResults,
        annual_product_output_kg: float,
        annual_co2_consumed_kg: float,
    ) -> float:
        """Calculate levelized cost of product for Power-to-X."""
        if annual_product_output_kg <= 0:
            return 0.0

        # Electrolyzer CAPEX
        electrolyzer_capex = (
            config.electrolyzer_config.rated_power_kw
            * config.electrolyzer_config.capex_per_kw
        )

        # Conversion unit CAPEX
        conversion_capex = (
            config.electrolyzer_config.rated_power_kw * config.capex_conversion_per_kw
        )

        total_capex = electrolyzer_capex + conversion_capex

        # Annuity factor
        annuity_factor = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.project_lifetime_years
        ) / (((1 + self.discount_rate) ** self.project_lifetime_years) - 1)

        annual_capital_cost = total_capex * annuity_factor

        # OPEX
        annual_opex = total_capex * config.opex_fraction

        # Electricity cost
        annual_electricity_cost = (
            electrolyzer_results.energy_consumption_kwh * self.electricity_price_kwh
        )

        # CO2 cost
        annual_co2_cost = annual_co2_consumed_kg * config.co2_capture_cost_per_ton / 1000

        # Catalyst replacement
        catalyst_cost = total_capex * 0.05  # ~5% of CAPEX
        annual_catalyst_cost = (
            catalyst_cost * 8760 / config.catalyst_lifetime_hours * annuity_factor
        )

        # Total annual cost
        total_annual_cost = (
            annual_capital_cost
            + annual_opex
            + annual_electricity_cost
            + annual_co2_cost
            + annual_catalyst_cost
        )

        # Levelized cost of product
        lcop = total_annual_cost / annual_product_output_kg

        return lcop

    def _convert_h2_to_product(
        self,
        pathway: PowerToXPathway,
        h2_available_kg: float,
        conversion_efficiency: float,
        co2_availability_profile: Optional[List[float]],
        timestep_hours: float,
    ) -> Dict[str, float]:
        """Convert hydrogen to target product based on pathway."""
        # Stoichiometric ratios and product yields
        conversion_factors = {
            PowerToXPathway.POWER_TO_H2: {
                "product_per_h2": 1.0,
                "co2_per_product": 0.0,
                "n2_per_product": 0.0,
            },
            PowerToXPathway.POWER_TO_METHANE: {
                "product_per_h2": 2.0,  # CH4: 16 kg / (4 * 2 kg H2) = 2
                "co2_per_product": 2.75,  # CO2: 44 kg / 16 kg CH4
                "n2_per_product": 0.0,
            },
            PowerToXPathway.POWER_TO_METHANOL: {
                "product_per_h2": 5.33,  # CH3OH: 32 kg / (3 * 2 kg H2)
                "co2_per_product": 1.375,  # CO2: 44 kg / 32 kg CH3OH
                "n2_per_product": 0.0,
            },
            PowerToXPathway.POWER_TO_AMMONIA: {
                "product_per_h2": 5.67,  # NH3: 17 kg / (1.5 * 2 kg H2)
                "co2_per_product": 0.0,
                "n2_per_product": 4.67,  # N2: 14 kg / 17 kg NH3 * 2
            },
            PowerToXPathway.POWER_TO_LIQUID: {
                "product_per_h2": 3.5,  # Simplified Fischer-Tropsch
                "co2_per_product": 3.14,
                "n2_per_product": 0.0,
            },
            PowerToXPathway.POWER_TO_SNG: {
                "product_per_h2": 2.0,  # Similar to methane
                "co2_per_product": 2.75,
                "n2_per_product": 0.0,
            },
        }

        factors = conversion_factors[pathway]

        # Calculate theoretical product output
        theoretical_product = h2_available_kg * factors["product_per_h2"]

        # Apply conversion efficiency
        actual_product = theoretical_product * conversion_efficiency

        # Calculate CO2 requirement
        co2_required = actual_product * factors["co2_per_product"]
        co2_consumed = 0.0

        if co2_required > 0 and co2_availability_profile:
            # Check CO2 availability
            total_co2_available = sum(co2_availability_profile) * timestep_hours
            co2_consumed = min(co2_required, total_co2_available)

            # Adjust product output if CO2 limited
            if co2_consumed < co2_required:
                actual_product *= co2_consumed / co2_required

        # Calculate N2 requirement
        n2_consumed = actual_product * factors["n2_per_product"]

        return {
            "product_kg": actual_product,
            "co2_consumed_kg": co2_consumed,
            "n2_consumed_kg": n2_consumed,
        }

    def _calculate_carbon_intensity(
        self,
        pathway: PowerToXPathway,
        energy_input_kwh: float,
        co2_consumed_kg: float,
        product_output_kg: float,
        grid_carbon_intensity: float,
    ) -> float:
        """Calculate carbon intensity of the product."""
        if product_output_kg <= 0:
            return 0.0

        # Emissions from electricity generation (assuming renewable, so minimal)
        # If grid power used, this would be: energy_input_kwh * grid_carbon_intensity
        electricity_emissions = energy_input_kwh * grid_carbon_intensity * 0.01  # Assume 1% grid

        # CO2 credit for utilization
        co2_credit = co2_consumed_kg

        # Net emissions
        net_emissions = electricity_emissions - co2_credit

        # Carbon intensity per kg product
        carbon_intensity = net_emissions / product_output_kg

        return carbon_intensity

    def _calculate_avoided_emissions(
        self, pathway: PowerToXPathway, product_output_kg: float, carbon_intensity: float
    ) -> float:
        """Calculate avoided emissions compared to fossil alternative."""
        # Fossil fuel carbon intensities (kg CO2 / kg fuel)
        fossil_intensities = {
            PowerToXPathway.POWER_TO_H2: 11.0,  # vs. SMR hydrogen
            PowerToXPathway.POWER_TO_METHANE: 2.75,  # vs. natural gas
            PowerToXPathway.POWER_TO_METHANOL: 1.375,  # vs. fossil methanol
            PowerToXPathway.POWER_TO_AMMONIA: 2.0,  # vs. Haber-Bosch with fossil
            PowerToXPathway.POWER_TO_LIQUID: 3.2,  # vs. crude oil products
            PowerToXPathway.POWER_TO_SNG: 2.75,  # vs. natural gas
        }

        fossil_intensity = fossil_intensities.get(pathway, 0.0)
        avoided = (fossil_intensity - carbon_intensity) * product_output_kg

        return max(avoided, 0.0)  # No negative avoidance
