"""
Hydrogen System Integration Example

This example demonstrates the use of the HydrogenIntegrator class for
comprehensive hydrogen system modeling and power-to-X analysis.
"""

import numpy as np
from pv_circularity_simulator.hydrogen import (
    HydrogenIntegrator,
    ElectrolyzerConfig,
    ElectrolyzerType,
    StorageConfig,
    StorageType,
    FuelCellConfig,
    FuelCellType,
    PowerToXConfig,
    PowerToXPathway,
)


def example_electrolyzer_modeling():
    """Example: Electrolyzer modeling with variable renewable power."""
    print("=" * 70)
    print("EXAMPLE 1: Electrolyzer Modeling")
    print("=" * 70)

    # Initialize integrator
    integrator = HydrogenIntegrator(
        discount_rate=0.05,
        electricity_price_kwh=0.04,
        project_lifetime_years=25,
    )

    # Configure PEM electrolyzer
    config = ElectrolyzerConfig(
        electrolyzer_type=ElectrolyzerType.PEM,
        rated_power_kw=5000.0,
        efficiency=0.68,
        min_load_fraction=0.1,
        max_load_fraction=1.0,
        operating_pressure_bar=30.0,
        capex_per_kw=1000.0,
        stack_lifetime_hours=80000.0,
    )

    print(f"\nElectrolyzer Configuration:")
    print(f"  Type: {config.electrolyzer_type.value}")
    print(f"  Rated Power: {config.rated_power_kw} kW")
    print(f"  Efficiency: {config.efficiency * 100:.1f}%")
    print(f"  Nominal H2 Production: {config.h2_production_rate_kg_h:.2f} kg/h")

    # Create variable power profile (simulating solar/wind)
    hours_per_year = 8760
    power_profile = []
    for i in range(hours_per_year):
        # Simulate daily and seasonal variation
        hour_of_day = i % 24
        day_of_year = i // 24

        # Daily pattern (solar-like)
        daily_factor = max(0, np.sin((hour_of_day - 6) * np.pi / 12))

        # Seasonal variation
        seasonal_factor = 0.7 + 0.3 * np.sin(day_of_year * 2 * np.pi / 365)

        # Random variations
        random_factor = 0.8 + 0.4 * np.random.random()

        power = config.rated_power_kw * daily_factor * seasonal_factor * random_factor
        power_profile.append(power)

    # Run electrolyzer modeling
    results = integrator.electrolyzer_modeling(
        config=config,
        power_input_profile=power_profile,
        timestep_hours=1.0,
    )

    print(f"\nResults:")
    print(f"  Total H2 Production: {results.h2_production_kg:,.0f} kg")
    print(f"  Annual H2 Production: {results.annual_h2_production_kg:,.0f} kg/year")
    print(f"  Energy Consumption: {results.energy_consumption_kwh:,.0f} kWh")
    print(f"  Average Efficiency: {results.average_efficiency * 100:.2f}%")
    print(f"  Capacity Factor: {results.capacity_factor * 100:.2f}%")
    print(f"  Operating Hours: {results.operating_hours:,.0f} h")
    print(f"  Degradation Factor: {results.degradation_factor:.4f}")
    print(f"  Specific Energy: {results.specific_energy_consumption:.2f} kWh/kg H2")
    print(f"  LCOH: ${results.levelized_cost_h2:.2f}/kg")
    print(f"  Start/Stop Cycles: {results.performance_metrics['starts_count']:.0f}")


def example_storage_design():
    """Example: Hydrogen storage system design and operation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Hydrogen Storage Design")
    print("=" * 70)

    integrator = HydrogenIntegrator()

    # Configure compressed gas storage
    storage_config = StorageConfig(
        storage_type=StorageType.COMPRESSED_GAS,
        capacity_kg=5000.0,
        pressure_bar=350.0,
        charging_rate_kg_h=100.0,
        discharging_rate_kg_h=80.0,
        round_trip_efficiency=0.92,
        self_discharge_rate_per_day=0.001,
        capex_per_kg=600.0,
        min_soc_fraction=0.1,
        max_soc_fraction=0.95,
    )

    print(f"\nStorage Configuration:")
    print(f"  Type: {storage_config.storage_type.value}")
    print(f"  Total Capacity: {storage_config.capacity_kg} kg")
    print(f"  Usable Capacity: {storage_config.usable_capacity_kg:.0f} kg")
    print(f"  Pressure: {storage_config.pressure_bar} bar")
    print(f"  Round-trip Efficiency: {storage_config.round_trip_efficiency * 100:.1f}%")

    # Create charge/discharge profiles
    hours = 8760
    charge_profile = []
    discharge_profile = []

    for i in range(hours):
        hour_of_day = i % 24

        # Charge during day (excess renewable)
        if 8 <= hour_of_day <= 16:
            charge = 60.0 + 30.0 * np.random.random()
            discharge = 0.0
        # Discharge during evening/night
        elif 17 <= hour_of_day <= 23 or hour_of_day <= 6:
            charge = 0.0
            discharge = 40.0 + 20.0 * np.random.random()
        else:
            charge = 10.0 * np.random.random()
            discharge = 10.0 * np.random.random()

        charge_profile.append(charge)
        discharge_profile.append(discharge)

    # Run storage simulation
    results = integrator.h2_storage_design(
        config=storage_config,
        charge_profile=charge_profile,
        discharge_profile=discharge_profile,
        timestep_hours=1.0,
        initial_soc_fraction=0.5,
    )

    print(f"\nResults:")
    print(f"  Total Charged: {results.total_charged_kg:,.0f} kg")
    print(f"  Total Discharged: {results.total_discharged_kg:,.0f} kg")
    print(f"  Total Losses: {results.total_losses_kg:,.0f} kg")
    print(f"  Average SOC: {results.average_soc * 100:.1f}%")
    print(f"  Average Efficiency: {results.average_efficiency * 100:.2f}%")
    print(f"  Cycling Count: {results.cycling_count:.1f} cycles")
    print(f"  Capacity Utilization: {results.capacity_utilization * 100:.1f}%")
    print(f"  LCOS: ${results.levelized_cost_storage:.2f}/kg")


def example_fuel_cell_integration():
    """Example: Fuel cell system integration with CHP."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Fuel Cell Integration (CHP)")
    print("=" * 70)

    integrator = HydrogenIntegrator()

    # Configure PEMFC with cogeneration
    fc_config = FuelCellConfig(
        fuel_cell_type=FuelCellType.PEMFC,
        rated_power_kw=1000.0,
        efficiency=0.55,
        min_load_fraction=0.05,
        max_load_fraction=1.0,
        capex_per_kw=1500.0,
        stack_lifetime_hours=40000.0,
        heat_recovery_fraction=0.35,
        cogeneration_enabled=True,
    )

    print(f"\nFuel Cell Configuration:")
    print(f"  Type: {fc_config.fuel_cell_type.value}")
    print(f"  Rated Power: {fc_config.rated_power_kw} kW")
    print(f"  Electrical Efficiency: {fc_config.efficiency * 100:.1f}%")
    print(f"  H2 Consumption Rate: {fc_config.h2_consumption_rate_kg_h:.2f} kg/h")
    print(f"  Thermal Power: {fc_config.thermal_power_kw:.1f} kW")
    print(f"  Cogeneration: {fc_config.cogeneration_enabled}")

    # Create power demand profile (commercial building)
    hours = 8760
    power_demand = []
    heat_demand = []

    for i in range(hours):
        hour_of_day = i % 24
        day_of_week = (i // 24) % 7

        # Base load
        if day_of_week < 5:  # Weekday
            if 6 <= hour_of_day <= 18:  # Business hours
                power = 600.0 + 200.0 * np.random.random()
                heat = 300.0 + 100.0 * np.random.random()
            else:
                power = 200.0 + 100.0 * np.random.random()
                heat = 100.0 + 50.0 * np.random.random()
        else:  # Weekend
            power = 150.0 + 50.0 * np.random.random()
            heat = 80.0 + 30.0 * np.random.random()

        power_demand.append(power)
        heat_demand.append(heat)

    # Run fuel cell simulation
    results = integrator.fuel_cell_integration(
        config=fc_config,
        power_demand_profile=power_demand,
        timestep_hours=1.0,
        heat_demand_profile=heat_demand,
    )

    print(f"\nResults:")
    print(f"  Electrical Output: {results.electrical_output_kwh:,.0f} kWh")
    print(f"  Thermal Output: {results.thermal_output_kwh:,.0f} kWh")
    print(f"  H2 Consumed: {results.h2_consumed_kg:,.0f} kg")
    print(f"  Average Efficiency: {results.average_efficiency * 100:.2f}%")
    print(f"  Cogeneration Efficiency: {results.cogeneration_efficiency * 100:.2f}%")
    print(f"  Capacity Factor: {results.capacity_factor * 100:.2f}%")
    print(f"  Operating Hours: {results.operating_hours:,.0f} h")
    print(f"  Degradation Factor: {results.degradation_factor:.4f}")
    print(f"  LCOE: ${results.levelized_cost_electricity:.4f}/kWh")
    print(f"  Specific H2 Consumption: {results.specific_h2_consumption:.4f} kg/kWh")


def example_power_to_x_analysis():
    """Example: Power-to-Methanol pathway analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Power-to-X Analysis (Methanol)")
    print("=" * 70)

    integrator = HydrogenIntegrator(
        electricity_price_kwh=0.03,  # Cheap renewable electricity
        discount_rate=0.06,
    )

    # Configure electrolyzer for P2X
    electrolyzer_config = ElectrolyzerConfig(
        electrolyzer_type=ElectrolyzerType.PEM,
        rated_power_kw=10000.0,  # 10 MW
        efficiency=0.70,
        min_load_fraction=0.2,
        capex_per_kw=900.0,
    )

    # Configure Power-to-Methanol pathway
    ptx_config = PowerToXConfig(
        pathway=PowerToXPathway.POWER_TO_METHANOL,
        electrolyzer_config=electrolyzer_config,
        conversion_efficiency=0.75,
        co2_source="Direct Air Capture",
        co2_capture_cost_per_ton=600.0,
        process_temperature_c=250.0,
        process_pressure_bar=50.0,
        catalyst_type="Cu/ZnO/Al2O3",
        capex_conversion_per_kw=800.0,
        product_lhv_kwh_per_kg=5.54,  # Methanol LHV
    )

    print(f"\nPower-to-X Configuration:")
    print(f"  Pathway: {ptx_config.pathway.value}")
    print(f"  Electrolyzer Power: {electrolyzer_config.rated_power_kw} kW")
    print(f"  Electrolyzer Efficiency: {electrolyzer_config.efficiency * 100:.1f}%")
    print(f"  Conversion Efficiency: {ptx_config.conversion_efficiency * 100:.1f}%")
    print(f"  Overall Efficiency: {ptx_config.overall_efficiency * 100:.2f}%")
    print(f"  CO2 Source: {ptx_config.co2_source}")
    print(f"  CO2 Required: {ptx_config.requires_co2}")

    # Create power input profile (steady renewable supply)
    hours = 8760
    power_profile = [8500.0 + 1000.0 * np.random.random() for _ in range(hours)]

    # CO2 availability (from DAC)
    co2_profile = [250.0 + 50.0 * np.random.random() for _ in range(hours)]  # kg/h

    # Run Power-to-X analysis
    results = integrator.power_to_x_analysis(
        config=ptx_config,
        power_input_profile=power_profile,
        timestep_hours=1.0,
        co2_availability_profile=co2_profile,
        grid_carbon_intensity=0.05,  # Low carbon grid
    )

    print(f"\nResults:")
    print(f"  Methanol Production: {results.product_output_kg:,.0f} kg")
    print(f"  H2 Intermediate: {results.h2_intermediate_kg:,.0f} kg")
    print(f"  Energy Input: {results.energy_input_kwh:,.0f} kWh")
    print(f"  CO2 Consumed: {results.co2_consumed_kg:,.0f} kg")
    print(f"  Overall Efficiency: {results.overall_efficiency * 100:.2f}%")
    print(f"  Specific Energy: {results.specific_energy_consumption:.2f} kWh/kg")
    print(f"  Capacity Factor: {results.capacity_factor * 100:.2f}%")
    print(f"  LCOP: ${results.levelized_cost_product:.2f}/kg methanol")
    print(f"  Carbon Intensity: {results.carbon_intensity:.3f} kg CO2/kg methanol")

    print(f"\nEconomic Metrics:")
    print(f"  Total CAPEX: ${results.economic_metrics['total_capex']:,.0f}")
    print(f"  Electricity Cost: ${results.economic_metrics['electricity_cost_per_kg_product']:.2f}/kg")
    print(f"  CO2 Cost: ${results.economic_metrics['co2_cost_per_kg_product']:.2f}/kg")

    print(f"\nEnvironmental Metrics:")
    print(f"  Avoided Emissions: {results.environmental_metrics['avoided_emissions_kg_co2']:,.0f} kg CO2")
    print(f"  Renewable Fraction: {results.environmental_metrics['renewable_energy_fraction'] * 100:.1f}%")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "HYDROGEN SYSTEM INTEGRATION EXAMPLES" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run examples
    example_electrolyzer_modeling()
    example_storage_design()
    example_fuel_cell_integration()
    example_power_to_x_analysis()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
