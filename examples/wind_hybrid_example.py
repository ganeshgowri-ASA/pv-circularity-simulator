#!/usr/bin/env python3
"""
Example: Wind-PV Hybrid System Integration

This example demonstrates how to use the WindHybridIntegrator to:
1. Assess wind resources
2. Model turbine performance
3. Optimize hybrid system configuration
4. Coordinate wind and PV generation
"""

import numpy as np
from datetime import datetime

from pv_simulator import (
    WindHybridIntegrator,
    HybridSystemConfig,
    WindResourceData,
    TurbineSpecifications,
    PVSystemConfig,
    TurbineType,
    CoordinationStrategy,
)


def main():
    """Run wind-PV hybrid system integration example."""
    print("=" * 80)
    print("Wind-PV Hybrid System Integration Example")
    print("=" * 80)
    print()

    # Step 1: Create turbine specifications
    print("Step 1: Defining turbine specifications...")
    turbine_specs = TurbineSpecifications(
        turbine_id="vestas_v110_2000",
        manufacturer="Vestas",
        model="V110-2.0 MW",
        rated_power_kw=2000.0,
        rotor_diameter_m=110.0,
        hub_height_m=80.0,
        cut_in_speed_ms=3.0,
        rated_speed_ms=12.0,
        cut_out_speed_ms=25.0,
        power_curve_speeds_ms=[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25],
        power_curve_kw=[0, 66, 154, 282, 460, 696, 996, 1341, 1661, 1866, 2000, 2000, 2000, 2000, 2000, 0],
        turbine_type=TurbineType.ONSHORE,
        efficiency=0.95,
    )
    print(f"  Turbine: {turbine_specs.manufacturer} {turbine_specs.model}")
    print(f"  Rated Power: {turbine_specs.rated_power_kw} kW")
    print(f"  Hub Height: {turbine_specs.hub_height_m} m")
    print()

    # Step 2: Create PV system configuration
    print("Step 2: Defining PV system configuration...")
    pv_system = PVSystemConfig(
        system_id="pv_array_001",
        capacity_mw=15.0,
        module_efficiency=0.20,
        inverter_efficiency=0.98,
        tilt_angle_deg=35.0,
        azimuth_deg=180.0,
        temperature_coefficient=-0.004,
    )
    print(f"  PV Capacity: {pv_system.capacity_mw} MW")
    print(f"  Module Efficiency: {pv_system.module_efficiency * 100:.1f}%")
    print()

    # Step 3: Create hybrid system configuration
    print("Step 3: Creating hybrid system configuration...")
    num_turbines = 10
    wind_capacity_mw = num_turbines * turbine_specs.rated_power_kw / 1000

    hybrid_config = HybridSystemConfig(
        system_id="hybrid_facility_001",
        site_name="Midwest Renewable Energy Park",
        pv_capacity_mw=pv_system.capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        num_turbines=num_turbines,
        pv_system=pv_system,
        turbine_specs=turbine_specs,
        shared_infrastructure=True,
        storage_capacity_mwh=25.0,
        grid_connection_capacity_mw=30.0,
    )
    print(f"  Site: {hybrid_config.site_name}")
    print(f"  Wind Capacity: {hybrid_config.wind_capacity_mw} MW ({num_turbines} turbines)")
    print(f"  PV Capacity: {hybrid_config.pv_capacity_mw} MW")
    print(f"  Storage: {hybrid_config.storage_capacity_mwh} MWh")
    print(f"  Grid Connection: {hybrid_config.grid_connection_capacity_mw} MW")
    print()

    # Step 4: Initialize integrator
    print("Step 4: Initializing WindHybridIntegrator...")
    integrator = WindHybridIntegrator(hybrid_config)
    integrator.initialize()
    print(f"  Integrator ID: {integrator.metadata.integrator_id}")
    print(f"  Status: {'Initialized' if integrator.is_initialized() else 'Not initialized'}")
    print()

    # Step 5: Generate sample wind resource data (1 year, hourly)
    print("Step 5: Generating sample wind resource data...")
    np.random.seed(42)
    num_hours = 8760  # 1 year

    # Generate realistic wind speeds (Weibull distribution, mean ~7 m/s)
    wind_speeds = (np.random.weibull(2.0, num_hours) * 7.0).tolist()

    # Generate wind directions (prevailing from west with variation)
    wind_directions = (np.random.normal(270, 30, num_hours) % 360).tolist()

    wind_data = WindResourceData(
        site_id="midwest_site_001",
        latitude=41.5,
        longitude=-93.5,
        elevation_m=350.0,
        wind_speeds_ms=wind_speeds,
        wind_directions_deg=wind_directions,
        air_density_kgm3=1.225,
        temperature_c=12.0,
        pressure_pa=101325.0,
        measurement_height_m=10.0,
        assessment_period_days=365,
        data_quality_score=0.95,
    )
    print(f"  Location: {wind_data.latitude}°N, {wind_data.longitude}°E")
    print(f"  Data points: {len(wind_data.wind_speeds_ms)}")
    print(f"  Mean wind speed (at 10m): {np.mean(wind_speeds):.2f} m/s")
    print()

    # Step 6: Perform wind resource assessment
    print("Step 6: Performing wind resource assessment...")
    assessment = integrator.wind_resource_assessment(wind_data)
    print(f"  Mean wind speed (at hub height): {assessment.mean_wind_speed_ms:.2f} m/s")
    print(f"  Weibull parameters: k={assessment.weibull_k:.2f}, c={assessment.weibull_c:.2f}")
    print(f"  Wind power density: {assessment.wind_power_density_wm2:.1f} W/m²")
    print(f"  Turbulence intensity: {assessment.turbulence_intensity:.2%}")
    print(f"  Prevailing direction: {assessment.prevailing_direction_deg:.0f}°")
    print(f"  Capacity factor estimate: {assessment.capacity_factor_estimate:.2%}")
    print(f"  Annual energy potential: {assessment.annual_energy_potential_mwh:,.0f} MWh")
    print()

    # Step 7: Model turbine performance
    print("Step 7: Modeling turbine performance...")
    performance = integrator.turbine_modeling(wind_data, include_losses=True)
    print(f"  Turbine ID: {performance.turbine_id}")
    print(f"  Gross capacity factor: {performance.capacity_factor:.2%}")
    print(f"  Net capacity factor: {performance.net_capacity_factor:.2%}")
    print(f"  Annual energy (per turbine): {performance.annual_energy_production_mwh:,.0f} MWh")
    print(f"  Availability factor: {performance.availability_factor:.2%}")
    print(f"  Wake losses: {performance.wake_losses_percent:.1f}%")
    print(f"  Electrical losses: {performance.electrical_losses_percent:.1f}%")
    print(f"  Environmental losses: {performance.environmental_losses_percent:.1f}%")
    print()

    # Step 8: Optimize hybrid system configuration
    print("Step 8: Optimizing hybrid system configuration...")
    optimization = integrator.hybrid_optimization(
        wind_data=wind_data,
        objective="maximize_energy",
        constraints={
            "max_pv_capacity_mw": 50.0,
            "max_wind_capacity_mw": 50.0,
            "max_storage_capacity_mwh": 50.0,
        }
    )
    print(f"  Optimization objective: {optimization.optimization_objective.value}")
    print(f"  Convergence status: {'Converged' if optimization.convergence_status else 'Not converged'}")
    print(f"  Iterations: {optimization.iterations}")
    print(f"  Optimal PV capacity: {optimization.optimal_pv_capacity_mw:.1f} MW")
    print(f"  Optimal wind capacity: {optimization.optimal_wind_capacity_mw:.1f} MW")
    if optimization.optimal_storage_capacity_mwh:
        print(f"  Optimal storage capacity: {optimization.optimal_storage_capacity_mwh:.1f} MWh")
    print(f"  Combined capacity factor: {optimization.capacity_factor_combined:.2%}")
    print(f"  Total annual energy: {optimization.total_annual_energy_mwh:,.0f} MWh")
    print(f"  Curtailment: {optimization.curtailment_percent:.1f}%")
    print(f"  LCOE: ${optimization.levelized_cost_of_energy:.2f}/MWh")
    print()

    # Step 9: Coordinate wind and PV generation
    print("Step 9: Coordinating wind and PV generation...")

    # Generate sample generation profiles (24 hours, 5-minute intervals)
    num_intervals = 288
    time_hours = np.linspace(0, 24, num_intervals)

    # Wind generation (more constant, with variation)
    wind_gen = (np.random.normal(12, 3, num_intervals)).clip(0, 20).tolist()

    # PV generation (follows solar curve)
    solar_curve = np.maximum(0, np.sin((time_hours - 6) * np.pi / 12))
    pv_gen = (solar_curve * 10 + np.random.normal(0, 0.5, num_intervals)).clip(0, 15).tolist()

    # Define coordination strategy
    strategy = CoordinationStrategy(
        strategy_name="advanced_coordination",
        dispatch_priority=["wind", "pv", "storage"],
        ramp_rate_limit_mw_per_min=5.0,
        forecast_horizon_hours=24,
        enable_storage_arbitrage=True,
        curtailment_strategy="proportional",
        grid_support_enabled=True,
    )

    coordination_results = integrator.wind_pv_coordination(
        wind_generation_mw=wind_gen,
        pv_generation_mw=pv_gen,
        strategy=strategy
    )

    # Analyze coordination results
    total_wind_dispatch = sum(r.wind_dispatch_mw for r in coordination_results)
    total_pv_dispatch = sum(r.pv_dispatch_mw for r in coordination_results)
    total_curtailed = sum(r.curtailed_energy_mw for r in coordination_results)
    avg_efficiency = np.mean([r.coordination_efficiency for r in coordination_results])

    print(f"  Strategy: {strategy.strategy_name}")
    print(f"  Time periods analyzed: {len(coordination_results)}")
    print(f"  Total wind dispatch: {total_wind_dispatch:,.1f} MW·interval")
    print(f"  Total PV dispatch: {total_pv_dispatch:,.1f} MW·interval")
    print(f"  Total curtailed energy: {total_curtailed:,.1f} MW·interval")
    print(f"  Average coordination efficiency: {avg_efficiency:.2%}")
    print()

    # Step 10: Run complete simulation
    print("Step 10: Running complete simulation...")
    simulation_results = integrator.run_simulation()
    print(f"  System ID: {simulation_results['system_id']}")
    print(f"  Site: {simulation_results['site_name']}")
    print(f"  Simulation timestamp: {simulation_results['simulation_timestamp']}")
    print()

    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Wind resource assessed: {assessment.mean_wind_speed_ms:.2f} m/s at hub height")
    print(f"  - Turbine performance: {performance.net_capacity_factor:.2%} net capacity factor")
    print(f"  - Optimal configuration: {optimization.optimal_wind_capacity_mw:.1f} MW wind + "
          f"{optimization.optimal_pv_capacity_mw:.1f} MW PV")
    print(f"  - Annual energy production: {optimization.total_annual_energy_mwh:,.0f} MWh")
    print(f"  - LCOE: ${optimization.levelized_cost_of_energy:.2f}/MWh")
    print(f"  - Coordination efficiency: {avg_efficiency:.2%}")
    print()


if __name__ == "__main__":
    main()
