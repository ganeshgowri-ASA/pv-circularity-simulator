"""
Example usage of PV Circularity Simulator Pydantic models.

This script demonstrates how to create and use all the core models
in the PV circularity simulator.
"""

from pv_circularity_simulator.models import (
    # Material models
    MaterialProperties,
    SiliconMaterial,
    ContactMaterial,
    MaterialType,
    CrystalType,
    # Cell models
    CellModel,
    CellType,
    CellArchitecture,
    CellGeometry,
    CellElectricalCharacteristics,
    CellDesign,
    # Module models
    ModuleModel,
    ModuleConfiguration,
    ElectricalParameters,
    MechanicalProperties,
    ThermalProperties,
    # System models
    SystemModel,
    SystemConfiguration,
    LocationCoordinates,
    Orientation,
    InverterConfiguration,
    MountingStructure,
    ElectricalProtection,
    MountingType,
    InverterType,
    GridConnectionType,
    # Performance models
    PerformanceModel,
    PerformanceMetrics,
    TemperatureCoefficients,
    DegradationModel,
    LossAnalysis,
    # Financial models
    FinancialModel,
    CapitalCost,
    OperatingCost,
    FinancialAnalysis,
    EndOfLifeScenario,
    EndOfLifeCost,
    MaterialRecoveryData,
    CircularityMetrics,
)


def create_example_cell():
    """Create an example PV cell with all components."""
    print("Creating example PV cell...")

    # 1. Create silicon substrate material
    silicon_props = MaterialProperties(
        name="Monocrystalline Silicon Properties",
        density_kg_m3=2330.0,
        thermal_conductivity_w_mk=148.0,
        specific_heat_j_kgk=700.0,
        band_gap_ev=1.12,
        refractive_index=3.5,
        recyclability_percentage=95.0,
        environmental_impact_kg_co2_eq=50.0,
    )

    substrate = SiliconMaterial(
        name="High-Purity Monocrystalline Silicon",
        material_type=MaterialType.SILICON,
        properties=silicon_props,
        crystal_type=CrystalType.MONOCRYSTALLINE,
        purity_percentage=99.9999,
        doping_type="P-type",
        doping_concentration_cm3=1e16,
        minority_carrier_lifetime_us=500.0,
        wafer_thickness_um=180.0,
    )

    # 2. Create contact material
    contact_props = MaterialProperties(
        name="Silver Contact Properties",
        density_kg_m3=10490.0,
        thermal_conductivity_w_mk=429.0,
        specific_heat_j_kgk=235.0,
        recyclability_percentage=98.0,
    )

    contact = ContactMaterial(
        name="Silver Screen-Printed Contact",
        material_type=MaterialType.CONTACT,
        properties=contact_props,
        conductivity_s_m=6.3e7,
        contact_resistance_ohm_cm2=0.001,
        metal_type="Ag",
        layer_thickness_um=20.0,
        finger_width_um=50.0,
        finger_spacing_mm=2.0,
    )

    # 3. Create cell geometry
    geometry = CellGeometry(
        name="M6 166mm Cell",
        width_mm=166.0,
        height_mm=166.0,
        thickness_um=180.0,
        busbar_count=9,
        busbar_width_mm=1.2,
    )

    # 4. Create electrical characteristics
    electrical = CellElectricalCharacteristics(
        name="STC Performance",
        voc_v=0.68,
        isc_a=9.5,
        vmpp_v=0.58,
        impp_a=9.0,
        pmpp_w=5.22,
        fill_factor=0.80,
        efficiency_percentage=20.0,
        series_resistance_ohm=0.005,
        shunt_resistance_ohm=1000.0,
    )

    # 5. Create cell design
    design = CellDesign(
        name="PERC Design",
        architecture=CellArchitecture.PERC,
        front_contact_fraction=0.08,
        rear_contact_fraction=0.85,
        texture_type="pyramid",
        anti_reflective_coating=True,
    )

    # 6. Create complete cell model
    cell = CellModel(
        name="High-Efficiency PERC Cell",
        cell_type=CellType.PERC,
        substrate_material=substrate,
        contact_front=contact,
        contact_rear=contact,
        geometry=geometry,
        electrical=electrical,
        design=design,
        manufacturer="ExampleSolar Inc.",
    )

    print(f"✓ Created cell: {cell.name}")
    print(f"  Power: {cell.electrical.pmpp_w:.2f} W")
    print(f"  Efficiency: {cell.electrical.efficiency_percentage:.1f}%")
    print(f"  Area: {cell.geometry.area_cm2:.1f} cm²")

    return cell


def create_example_module(cell):
    """Create an example PV module using the cell."""
    print("\nCreating example PV module...")

    # 1. Module configuration
    config = ModuleConfiguration(
        name="72-Cell Configuration",
        cells_in_series=72,
        cells_in_parallel=1,
        total_cells=72,
        bypass_diodes=3,
        cells_per_bypass_diode=24,
        half_cut_cells=False,
    )

    # 2. Electrical parameters
    electrical = ElectricalParameters(
        name="Module STC Parameters",
        pmax_w=375.0,
        voc_v=48.96,
        isc_a=9.5,
        vmpp_v=41.76,
        impp_a=8.98,
        efficiency_percentage=19.2,
        temperature_coefficient_pmax=-0.38,
        temperature_coefficient_voc=-0.29,
        temperature_coefficient_isc=0.05,
    )

    # 3. Mechanical properties
    mechanical = MechanicalProperties(
        name="Module Dimensions",
        length_mm=1956.0,
        width_mm=992.0,
        thickness_mm=40.0,
        weight_kg=21.5,
        glass_thickness_mm=3.2,
    )

    # 4. Thermal properties
    thermal = ThermalProperties(
        name="Module Thermal",
        noct_c=45.0,
        operating_temp_min_c=-40.0,
        operating_temp_max_c=85.0,
    )

    # 5. Create module
    module = ModuleModel(
        name="ExampleSolar 375W PERC Module",
        model_number="ES-375M-60P",
        cell_reference=cell,
        configuration=config,
        electrical=electrical,
        mechanical=mechanical,
        thermal=thermal,
        manufacturer="ExampleSolar Inc.",
        warranty_years_product=12,
        warranty_years_performance=25,
        performance_guarantee_25y_percent=84.8,
    )

    print(f"✓ Created module: {module.name}")
    print(f"  Power: {module.electrical.pmax_w:.0f} W")
    print(f"  Efficiency: {module.electrical.efficiency_percentage:.1f}%")
    print(f"  Dimensions: {module.mechanical.length_mm}mm × {module.mechanical.width_mm}mm")

    return module


def create_example_system(module):
    """Create an example PV system using the module."""
    print("\nCreating example PV system...")

    # 1. Location
    location = LocationCoordinates(
        name="Installation Site",
        latitude=37.7749,
        longitude=-122.4194,
        altitude_m=16.0,
        timezone="America/Los_Angeles",
        location_name="San Francisco, CA, USA",
    )

    # 2. Orientation
    orientation = Orientation(
        name="Optimal Tilt",
        tilt_angle_deg=25.0,
        azimuth_angle_deg=180.0,  # South-facing
    )

    # 3. System configuration
    configuration = SystemConfiguration(
        name="10 kW Residential System",
        modules_per_string=10,
        strings_in_parallel=3,
        total_modules=30,
        dc_capacity_w=11250.0,
        ac_capacity_w=10000.0,
        dc_ac_ratio=1.125,
        string_voltage_voc=489.6,
        string_voltage_vmpp=417.6,
        string_current_isc=9.5,
    )

    # 4. Inverter
    inverter = InverterConfiguration(
        name="String Inverter",
        inverter_type=InverterType.STRING,
        rated_power_w=10000.0,
        max_dc_voltage_v=600.0,
        mppt_voltage_range_min_v=150.0,
        mppt_voltage_range_max_v=550.0,
        max_dc_current_a=30.0,
        number_of_mppt=2,
        efficiency_max_percentage=98.5,
        efficiency_weighted_percentage=98.0,
        manufacturer="InverterCo",
        model="INV-10K",
    )

    # 5. Mounting
    mounting = MountingStructure(
        name="Roof Mount",
        mounting_type=MountingType.ROOF_MOUNTED,
        structure_material="aluminum",
    )

    # 6. Protection
    protection = ElectricalProtection(
        dc_disconnect=True,
        ac_disconnect=True,
        surge_protection_dc=True,
        surge_protection_ac=True,
        ground_fault_protection=True,
        arc_fault_protection=True,
        rapid_shutdown=True,
    )

    # 7. Create system
    system = SystemModel(
        name="Residential 10kW PV System",
        module=module,
        location=location,
        orientation=orientation,
        configuration=configuration,
        inverter=inverter,
        mounting=mounting,
        protection=protection,
        grid_connection=GridConnectionType.ON_GRID,
        installer="ExampleSolar Installers",
        installation_date="2024-01-15",
    )

    print(f"✓ Created system: {system.name}")
    print(f"  DC Capacity: {system.configuration.dc_capacity_w / 1000:.1f} kW")
    print(f"  AC Capacity: {system.configuration.ac_capacity_w / 1000:.1f} kW")
    print(f"  Location: {system.location.location_name}")
    print(f"  Total area: {system.calculate_total_area_m2():.1f} m²")

    return system


def create_example_performance():
    """Create an example performance model."""
    print("\nCreating example performance model...")

    # 1. Performance metrics
    metrics = PerformanceMetrics(
        power_output_w=350.0,
        voltage_v=40.0,
        current_a=8.75,
        irradiance_w_m2=950.0,
        cell_temperature_c=50.0,
        ambient_temperature_c=28.0,
        wind_speed_m_s=3.5,
    )

    # 2. Temperature coefficients
    temp_coeffs = TemperatureCoefficients(
        name="Module Temp Coefficients",
        alpha_isc_percent_c=0.05,
        beta_voc_percent_c=-0.29,
        gamma_pmax_percent_c=-0.38,
    )

    # 3. Degradation model
    degradation = DegradationModel(
        name="Standard Degradation",
        initial_degradation_percentage=2.0,
        annual_degradation_rate_percentage=0.5,
        lifetime_years=25,
        age_years=5.0,
    )

    # 4. Loss analysis
    losses = LossAnalysis(
        name="System Losses",
        soiling_loss_percentage=2.0,
        shading_loss_percentage=1.5,
        temperature_loss_percentage=5.0,
        inverter_loss_percentage=2.5,
        wiring_loss_percentage=2.0,
    )

    # 5. Create performance model
    performance = PerformanceModel(
        name="System Performance Record",
        metrics=metrics,
        temperature_coefficients=temp_coeffs,
        degradation=degradation,
        losses=losses,
        performance_ratio=0.85,
    )

    print(f"✓ Created performance model")
    print(f"  Current output: {metrics.power_output_w:.0f} W")
    print(f"  Performance ratio: {performance.performance_ratio:.2%}")
    print(f"  Total losses: {losses.total_loss_percentage:.1f}%")
    print(f"  Power retention (5y): {degradation.calculate_power_retention():.1%}")

    return performance


def create_example_financial():
    """Create an example financial model."""
    print("\nCreating example financial model...")

    # 1. Capital costs
    capex = CapitalCost(
        name="System CAPEX",
        module_cost_usd_per_wp=0.30,
        inverter_cost_usd=1500.0,
        mounting_structure_cost_usd=1000.0,
        electrical_bos_cost_usd=800.0,
        installation_labor_cost_usd=2500.0,
        permit_fees_usd=500.0,
        total_capex_usd=10000.0,
    )

    # 2. Operating costs
    opex = OperatingCost(
        name="Annual OPEX",
        maintenance_annual_usd=150.0,
        insurance_annual_usd=100.0,
        monitoring_annual_usd=50.0,
        total_opex_annual_usd=300.0,
    )

    # 3. Financial analysis
    financial_analysis = FinancialAnalysis(
        name="25-Year Analysis",
        investment_cost_usd=10000.0,
        annual_revenue_usd=0.0,
        annual_savings_usd=1500.0,
        electricity_price_usd_kwh=0.15,
        discount_rate_percentage=5.0,
        analysis_period_years=25,
        npv_usd=15000.0,
        irr_percentage=12.5,
        payback_period_years=7.2,
        discounted_payback_period_years=8.5,
        lcoe_usd_kwh=0.08,
        benefit_cost_ratio=2.5,
        profitability_index=2.5,
        return_on_investment_percentage=150.0,
    )

    # 4. End-of-life scenario
    eol_scenario = EndOfLifeScenario(
        name="Circular Economy Scenario",
        recycling_percentage=85.0,
        refurbishment_percentage=10.0,
        reuse_percentage=5.0,
        landfill_percentage=0.0,
    )

    # 5. End-of-life costs
    eol_costs = EndOfLifeCost(
        name="EOL Costs & Revenues",
        decommissioning_cost_usd=500.0,
        transportation_cost_usd=200.0,
        recycling_cost_usd=300.0,
        total_eol_cost_usd=1000.0,
        material_recovery_revenue_usd=800.0,
        total_eol_revenue_usd=800.0,
        net_eol_cost_usd=200.0,
    )

    # 6. Material recovery
    material_recovery = [
        MaterialRecoveryData(
            name="Silicon Recovery",
            material_type="silicon",
            total_mass_kg=50.0,
            recovery_rate_percentage=95.0,
            recovered_mass_kg=47.5,
            market_value_usd_per_kg=5.0,
            recovery_cost_usd_per_kg=2.0,
            environmental_impact_avoided_kg_co2_eq=50.0,
        ),
        MaterialRecoveryData(
            name="Aluminum Recovery",
            material_type="aluminum",
            total_mass_kg=30.0,
            recovery_rate_percentage=98.0,
            recovered_mass_kg=29.4,
            market_value_usd_per_kg=2.0,
            recovery_cost_usd_per_kg=0.5,
            environmental_impact_avoided_kg_co2_eq=8.0,
        ),
    ]

    # 7. Circularity metrics
    circularity = CircularityMetrics(
        name="System Circularity",
        material_circularity_indicator=0.85,
        recycled_content_percentage=15.0,
        recyclability_rate_percentage=90.0,
        total_waste_kg=100.0,
        waste_diverted_from_landfill_kg=95.0,
        circular_economy_value_usd=500.0,
        environmental_footprint_saved_kg_co2_eq=3000.0,
    )

    # 8. Create financial model
    financial = FinancialModel(
        name="10kW System Financial Model",
        capex=capex,
        opex=opex,
        financial_analysis=financial_analysis,
        eol_scenario=eol_scenario,
        eol_costs=eol_costs,
        material_recovery=material_recovery,
        circularity_metrics=circularity,
    )

    print(f"✓ Created financial model")
    print(f"  Total investment: ${financial.capex.total_capex_usd:,.0f}")
    print(f"  NPV (25y): ${financial.financial_analysis.npv_usd:,.0f}")
    print(f"  IRR: {financial.financial_analysis.irr_percentage:.1f}%")
    print(f"  Payback: {financial.financial_analysis.payback_period_years:.1f} years")
    print(f"  LCOE: ${financial.financial_analysis.lcoe_usd_kwh:.3f}/kWh")
    print(f"  Circularity: {financial.circularity_metrics.material_circularity_indicator:.0%}")

    return financial


def demonstrate_serialization(model):
    """Demonstrate model serialization capabilities."""
    print("\n" + "=" * 60)
    print("SERIALIZATION DEMONSTRATION")
    print("=" * 60)

    # Export to dictionary
    print("\n1. Export to dictionary:")
    data_dict = model.model_dump()
    print(f"   Keys: {list(data_dict.keys())[:5]}... ({len(data_dict)} total)")

    # Export to JSON
    print("\n2. Export to JSON:")
    json_str = model.model_dump_json(indent=2)
    print(f"   JSON length: {len(json_str)} characters")
    print(f"   First 200 chars: {json_str[:200]}...")

    # Recreate from dictionary
    print("\n3. Recreate from dictionary:")
    recreated = model.__class__(**data_dict)
    print(f"   ✓ Successfully recreated {model.__class__.__name__}")

    return recreated


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PV CIRCULARITY SIMULATOR - MODEL EXAMPLES")
    print("=" * 60)

    # Create all models
    cell = create_example_cell()
    module = create_example_module(cell)
    system = create_example_system(module)
    performance = create_example_performance()
    financial = create_example_financial()

    # Demonstrate serialization
    demonstrate_serialization(financial)

    print("\n" + "=" * 60)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe PV Circularity Simulator models are production-ready with:")
    print("  • Comprehensive Pydantic v2 validation")
    print("  • Full type hints and docstrings")
    print("  • Serialization/deserialization support")
    print("  • Business logic and calculations")
    print("  • Circular economy metrics (3R: Recycle, Refurbish, Reuse)")
    print("\nReady for integration into your PV lifecycle simulation platform!")


if __name__ == "__main__":
    main()
