"""
Module Temperature Calculation Examples

This script demonstrates the various features of the module temperature
calculation system including:
- NOCT calculations
- Multiple temperature models
- Mounting configuration effects
- Temperature coefficient losses
- Seasonal and time-of-day variations
- Comprehensive temperature analysis

Author: PV Circularity Simulator Team
"""

import sys
sys.path.insert(0, '/home/user/pv-circularity-simulator')

from src.modules.module_temperature import (
    ModuleTemperatureModel,
    MountingType,
    TemperatureModelType,
    ModuleTechnology,
    ModuleSpecification,
    get_default_temp_coefficient,
    estimate_noct_from_mounting,
    calculate_power_at_temperature,
)


def demo_basic_noct_calculation():
    """Demonstrate basic NOCT calculation."""
    print("=" * 70)
    print("DEMO 1: Basic NOCT Calculation")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Standard conditions
    noct = model.calculate_noct(
        ambient_temp=20.0,
        irradiance=800.0,
        wind_speed=1.0,
        mounting=MountingType.OPEN_RACK,
        base_noct=45.0
    )

    print(f"NOCT at standard conditions: {noct:.2f}°C")
    print()

    # Different conditions
    conditions = [
        (25.0, 1000.0, 2.0, MountingType.OPEN_RACK),
        (30.0, 900.0, 1.5, MountingType.CLOSE_ROOF),
        (22.0, 850.0, 0.5, MountingType.BUILDING_INTEGRATED),
    ]

    for ambient, irr, wind, mounting in conditions:
        noct = model.calculate_noct(
            ambient_temp=ambient,
            irradiance=irr,
            wind_speed=wind,
            mounting=mounting
        )
        print(f"Conditions: {ambient}°C, {irr} W/m², {wind} m/s, {mounting.value}")
        print(f"  → NOCT: {noct:.2f}°C")
        print()


def demo_temperature_models():
    """Demonstrate different temperature calculation models."""
    print("=" * 70)
    print("DEMO 2: Temperature Models Comparison")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Test conditions
    ambient = 25.0
    irradiance = 1000.0
    wind_speed = 2.5
    noct = 45.0

    models = [
        TemperatureModelType.SIMPLE_LINEAR,
        TemperatureModelType.ROSS_FAIMAN,
        TemperatureModelType.SANDIA,
        TemperatureModelType.KING_SAPM,
    ]

    print(f"Conditions: {ambient}°C ambient, {irradiance} W/m², {wind_speed} m/s wind")
    print()

    for model_type in models:
        temp = model.calculate_module_temp(
            ambient=ambient,
            irr=irradiance,
            wind=wind_speed,
            noct=noct,
            mounting=MountingType.OPEN_RACK,
            model_type=model_type
        )
        print(f"{model_type.value:20s}: {temp:.2f}°C (ΔT = {temp - ambient:.2f}°C)")

    print()


def demo_mounting_configurations():
    """Demonstrate mounting configuration effects."""
    print("=" * 70)
    print("DEMO 3: Mounting Configuration Effects")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Fixed conditions
    ambient = 30.0
    irradiance = 900.0
    wind_speed = 1.5
    noct = 45.0

    mountings = [
        MountingType.TRACKER_DUAL_AXIS,
        MountingType.TRACKER_SINGLE_AXIS,
        MountingType.GROUND_MOUNT,
        MountingType.OPEN_RACK,
        MountingType.CLOSE_ROOF,
        MountingType.BUILDING_INTEGRATED,
    ]

    print(f"Conditions: {ambient}°C ambient, {irradiance} W/m², {wind_speed} m/s wind")
    print()

    for mounting in mountings:
        temp = model.calculate_module_temp(
            ambient=ambient,
            irr=irradiance,
            wind=wind_speed,
            noct=noct,
            mounting=mounting,
            model_type=TemperatureModelType.SIMPLE_LINEAR
        )
        print(f"{mounting.value:25s}: {temp:.2f}°C (ΔT = {temp - ambient:.2f}°C)")

    print()


def demo_temperature_coefficients():
    """Demonstrate temperature coefficient effects."""
    print("=" * 70)
    print("DEMO 4: Temperature Coefficient Effects")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Module operating at 55°C
    module_temp = 55.0
    stc_power = 400.0  # W

    technologies = [
        ModuleTechnology.HJT,
        ModuleTechnology.THIN_FILM_CDTE,
        ModuleTechnology.TOPCON,
        ModuleTechnology.PERC,
        ModuleTechnology.MONO_SI,
        ModuleTechnology.POLY_SI,
    ]

    print(f"Module temperature: {module_temp}°C")
    print(f"STC power rating: {stc_power} W")
    print()

    for tech in technologies:
        temp_coeff = get_default_temp_coefficient(tech)
        loss_factor = model.calculate_temp_coefficient_losses(
            module_temp=module_temp,
            technology=tech
        )
        actual_power = stc_power * loss_factor

        print(f"{tech.value:20s}: {temp_coeff:+.2f}%/°C → "
              f"{loss_factor:.3f} → {actual_power:.1f} W ({loss_factor*100:.1f}% of STC)")

    print()


def demo_seasonal_variations():
    """Demonstrate seasonal temperature variations."""
    print("=" * 70)
    print("DEMO 5: Seasonal Temperature Variations")
    print("=" * 70)

    model = ModuleTemperatureModel()

    base_ambient = 15.0  # Annual average
    latitude = 40.0  # Mid-latitude

    # Sample days throughout the year
    seasons = [
        (15, "Mid-January (Winter)"),
        (105, "Mid-April (Spring)"),
        (195, "Mid-July (Summer)"),
        (285, "Mid-October (Fall)"),
    ]

    print(f"Base annual average temperature: {base_ambient}°C")
    print(f"Latitude: {latitude}°")
    print()

    for day, season_name in seasons:
        temp = model.calculate_seasonal_adjustment(
            day_of_year=day,
            latitude=latitude,
            base_ambient=base_ambient
        )
        adjustment = temp - base_ambient
        print(f"Day {day:3d} ({season_name:25s}): {temp:.1f}°C ({adjustment:+.1f}°C)")

    print()


def demo_time_of_day_variations():
    """Demonstrate time-of-day temperature variations."""
    print("=" * 70)
    print("DEMO 6: Time-of-Day Temperature Variations")
    print("=" * 70)

    model = ModuleTemperatureModel()

    daily_amplitude = 8.0  # °C

    # Sample hours throughout the day
    hours = [0, 3, 6, 9, 12, 15, 18, 21]

    print(f"Daily temperature amplitude: ±{daily_amplitude}°C")
    print()

    for hour in hours:
        adjustment = model.calculate_time_of_day_adjustment(
            hour=hour,
            daily_amplitude=daily_amplitude
        )
        print(f"Hour {hour:2d}:00 → {adjustment:+.1f}°C adjustment")

    print()


def demo_thermal_time_constant():
    """Demonstrate thermal time constant calculation."""
    print("=" * 70)
    print("DEMO 7: Thermal Time Constant")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Different module configurations
    modules = [
        {
            "name": "Standard 60-cell",
            "area": 1.7,
            "mass": 18.0,
            "specific_heat": 900.0,
            "technology": ModuleTechnology.MONO_SI,
            "noct": 45.0,
        },
        {
            "name": "Large 72-cell",
            "area": 2.0,
            "mass": 22.0,
            "specific_heat": 900.0,
            "technology": ModuleTechnology.MONO_SI,
            "noct": 45.0,
        },
        {
            "name": "Lightweight thin-film",
            "area": 1.8,
            "mass": 12.0,
            "specific_heat": 850.0,
            "technology": ModuleTechnology.THIN_FILM_CDTE,
            "noct": 46.0,
        },
    ]

    for mod_config in modules:
        name = mod_config.pop("name")
        mod_spec = ModuleSpecification(**mod_config)
        tau = model.model_thermal_time_constant(mod_spec)
        tau_minutes = tau / 60.0

        print(f"{name}:")
        print(f"  Area: {mod_spec.area:.1f} m², Mass: {mod_spec.mass:.1f} kg")
        print(f"  Thermal time constant: {tau:.0f} seconds ({tau_minutes:.1f} minutes)")
        print()


def demo_comprehensive_analysis():
    """Demonstrate comprehensive temperature analysis."""
    print("=" * 70)
    print("DEMO 8: Comprehensive Temperature Analysis")
    print("=" * 70)

    model = ModuleTemperatureModel()

    # Module specification
    module_spec = ModuleSpecification(
        area=1.7,
        mass=18.0,
        technology=ModuleTechnology.MONO_SI,
        noct=45.0,
        temp_coeff_power=-0.40,
    )

    # Operating conditions
    ambient_base = 20.0
    irradiance = 950.0
    wind_speed = 2.0
    day_of_year = 195  # Mid-summer
    hour = 14.0  # 2 PM
    latitude = 35.0

    result = model.calculate_comprehensive_temperature(
        ambient_base=ambient_base,
        irradiance=irradiance,
        wind_speed=wind_speed,
        module_spec=module_spec,
        day_of_year=day_of_year,
        hour=hour,
        latitude=latitude,
        mounting=MountingType.OPEN_RACK,
        model_type=TemperatureModelType.SIMPLE_LINEAR
    )

    print("Operating Conditions:")
    print(f"  Date: Day {day_of_year} (mid-summer), Hour: {hour:.1f}")
    print(f"  Latitude: {latitude}°")
    print(f"  Base ambient: {ambient_base:.1f}°C")
    print(f"  Irradiance: {irradiance} W/m²")
    print(f"  Wind speed: {wind_speed} m/s")
    print(f"  Mounting: {MountingType.OPEN_RACK.value}")
    print()

    print("Results:")
    print(f"  Module temperature: {result.module_temperature:.2f}°C")
    print(f"  Temperature above ambient: {result.temp_above_ambient:.2f}°C")
    print(f"  Power loss factor: {result.power_loss_factor:.4f} "
          f"({result.power_loss_factor*100:.1f}% of STC)")
    print()

    print("Temperature Breakdown:")
    meta = result.metadata
    print(f"  Base ambient: {meta['ambient_base']:.1f}°C")
    print(f"  + Seasonal adjustment: {meta['seasonal_adjustment']:+.1f}°C")
    print(f"  + Time-of-day adjustment: {meta['tod_adjustment']:+.1f}°C")
    print(f"  = Adjusted ambient: {meta['ambient_adjusted']:.1f}°C")
    print(f"  + Irradiance heating: {result.temp_above_ambient - result.mounting_adjustment:.1f}°C")
    print(f"  + Mounting adjustment: {result.mounting_adjustment:+.1f}°C")
    print(f"  = Module temperature: {result.module_temperature:.2f}°C")
    print()


def demo_power_calculation():
    """Demonstrate power calculation with temperature effects."""
    print("=" * 70)
    print("DEMO 9: Power Calculation with Temperature Effects")
    print("=" * 70)

    # Module specifications
    stc_power = 400.0  # W
    temp_coeff = -0.40  # %/°C

    # Different operating temperatures
    temperatures = [15, 25, 35, 45, 55, 65, 75]

    print(f"STC power rating: {stc_power} W")
    print(f"Temperature coefficient: {temp_coeff}%/°C")
    print()

    print("Module Temp (°C)  |  Power Output (W)  |  % of STC")
    print("-" * 55)

    for module_temp in temperatures:
        actual_power = calculate_power_at_temperature(
            stc_power=stc_power,
            module_temp=module_temp,
            temp_coeff=temp_coeff
        )
        pct_of_stc = (actual_power / stc_power) * 100

        print(f"{module_temp:8.0f}          |  {actual_power:8.1f}          |  {pct_of_stc:6.1f}%")

    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "MODULE TEMPERATURE CALCULATION DEMONSTRATIONS" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_basic_noct_calculation()
    demo_temperature_models()
    demo_mounting_configurations()
    demo_temperature_coefficients()
    demo_seasonal_variations()
    demo_time_of_day_variations()
    demo_thermal_time_constant()
    demo_comprehensive_analysis()
    demo_power_calculation()

    print("=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
