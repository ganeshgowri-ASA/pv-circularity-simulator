"""Complete example of irradiance modeling and solar resource assessment.

This script demonstrates all major features of the irradiance modeling system:
1. Solar position calculation
2. GHI decomposition into DNI/DHI
3. POA irradiance calculation with multiple transposition models
4. Spectral and AOI loss calculations
5. Statistical resource analysis
6. Interactive visualizations

Run with: python examples/complete_irradiance_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

from src.irradiance.calculator import IrradianceCalculator
from src.irradiance.poa_model import POAIrradianceModel
from src.irradiance.resource_analyzer import SolarResourceAnalyzer
from src.irradiance.models import LocationConfig, SurfaceConfig
from src.ui.visualizations import SolarResourceVisualizer


def generate_sample_data(location: LocationConfig, days: int = 365) -> pd.DataFrame:
    """Generate synthetic irradiance data for demonstration.

    In production, this would be replaced with actual measured or satellite data.

    Args:
        location: Location configuration
        days: Number of days to generate

    Returns:
        DataFrame with GHI, DNI, DHI columns
    """
    # Create hourly time series
    times = pd.date_range(
        start="2024-01-01",
        periods=days * 24,
        freq="h",
        tz=location.timezone,
    )

    calc = IrradianceCalculator(location)
    solar_pos = calc.get_solar_position(times)

    # Generate synthetic clear-sky irradiance
    from pvlib.irradiance import get_extra_radiation
    from pvlib.atmosphere import get_relative_airmass

    dni_extra = get_extra_radiation(times)
    airmass = get_relative_airmass(solar_pos.zenith)

    # Simple clear-sky model (simplified for demo)
    dni = dni_extra * 0.85 ** (airmass ** 0.678)
    dni = dni.clip(lower=0)

    # Add clouds (random variation)
    cloud_factor = 0.7 + 0.3 * np.random.random(len(times))
    dni = dni * cloud_factor

    # Calculate GHI from DNI
    dhi = 0.1 * dni + 50  # Simplified diffuse component
    ghi = dni * np.cos(np.radians(solar_pos.zenith)) + dhi
    ghi = ghi.clip(lower=0)

    return pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi}, index=times)


def main():
    """Run complete irradiance analysis demonstration."""

    print("=" * 80)
    print("PV CIRCULARITY SIMULATOR - IRRADIANCE MODELING & RESOURCE ASSESSMENT")
    print("=" * 80)
    print()

    # ======================
    # 1. CONFIGURATION
    # ======================
    print("1. Setting up location and surface configuration...")

    location = LocationConfig(
        latitude=39.7555,  # Golden, CO
        longitude=-105.2211,
        altitude=1829,
        timezone="America/Denver",
        name="NREL Golden Campus",
    )

    surface = SurfaceConfig(
        tilt=30.0,  # Optimal tilt for this latitude
        azimuth=180.0,  # South-facing
        albedo=0.2,  # Typical ground reflectance
    )

    print(f"   Location: {location.name}")
    print(f"   Coordinates: {location.latitude}°N, {location.longitude}°E")
    print(f"   Surface: {surface.tilt}° tilt, {surface.azimuth}° azimuth")
    print()

    # ======================
    # 2. SOLAR POSITION
    # ======================
    print("2. Calculating solar position...")

    calc = IrradianceCalculator(location)

    # Calculate for a specific day
    times_day = pd.date_range(
        start="2024-06-21 00:00",
        end="2024-06-21 23:00",
        freq="h",
        tz=location.timezone,
    )

    solar_pos = calc.get_solar_position(times_day)

    print(f"   Summer solstice (June 21):")
    print(f"   Maximum elevation: {solar_pos.elevation.max():.1f}°")
    print(f"   Solar noon zenith: {solar_pos.zenith[solar_pos.elevation.idxmax()]:.1f}°")
    print()

    # ======================
    # 3. IRRADIANCE DATA
    # ======================
    print("3. Loading/generating irradiance data...")

    # Generate synthetic data (in production, load actual data)
    irradiance_data = generate_sample_data(location, days=365)

    print(f"   Generated {len(irradiance_data)} hours of data")
    print(f"   GHI range: {irradiance_data['ghi'].min():.1f} - {irradiance_data['ghi'].max():.1f} W/m²")
    print(f"   Annual GHI total: {irradiance_data['ghi'].sum()/1000:.1f} kWh/m²")
    print()

    # ======================
    # 4. GHI DECOMPOSITION
    # ======================
    print("4. Testing GHI decomposition models...")

    # Take a week of data for comparison
    week_data = irradiance_data.iloc[:168]

    for model in ["dirint", "disc", "erbs"]:
        components = calc.ghi_dni_dhi_decomposition(
            week_data["ghi"], times=week_data.index, model=model
        )

        dni_mean = components.dni.mean()
        dhi_mean = components.dhi.mean()
        print(f"   {model.upper():8s}: DNI={dni_mean:.1f} W/m², DHI={dhi_mean:.1f} W/m²")

    # Use DIRINT for further analysis (most accurate for hourly data)
    from src.irradiance.models import IrradianceComponents

    irrad_components = IrradianceComponents(
        ghi=irradiance_data["ghi"],
        dni=irradiance_data["dni"],
        dhi=irradiance_data["dhi"],
    )
    print()

    # ======================
    # 5. POA IRRADIANCE
    # ======================
    print("5. Calculating plane-of-array irradiance...")

    poa_model = POAIrradianceModel(location, surface, irradiance_calculator=calc)
    solar_pos_annual = calc.get_solar_position(irradiance_data.index)

    # Compare transposition models
    print("   Comparing transposition models (annual average POA):")

    for trans_model in ["isotropic", "haydavies", "perez"]:
        poa_components = poa_model.calculate_poa_components(
            irrad_components,
            solar_pos_annual,
            transposition_model=trans_model,
            include_spectral=False,
            include_aoi=False,
        )

        avg_poa = poa_components.poa_global.mean()
        annual_poa = poa_components.poa_global.sum() / 1000

        print(f"   {trans_model.capitalize():12s}: {avg_poa:.1f} W/m², {annual_poa:.1f} kWh/m²/year")

    # Use Perez for detailed analysis
    poa_components = poa_model.calculate_poa_components(
        irrad_components,
        solar_pos_annual,
        transposition_model="perez",
        include_spectral=True,
        include_aoi=True,
        module_type="multisi",
    )

    # Component breakdown
    print("\n   POA component breakdown (with losses):")
    print(f"   Direct beam:      {poa_components.poa_direct.sum()/1000:.1f} kWh/m²/year")
    print(f"   Sky diffuse:      {poa_components.poa_diffuse.sum()/1000:.1f} kWh/m²/year")
    print(f"   Ground reflected: {poa_components.poa_ground.sum()/1000:.1f} kWh/m²/year")
    print(f"   Total POA:        {poa_components.poa_global.sum()/1000:.1f} kWh/m²/year")
    print()

    # ======================
    # 6. LOSS ANALYSIS
    # ======================
    print("6. Analyzing optical losses...")

    # Calculate losses
    aoi_factor = poa_model.aoi_losses(solar_pos_annual)
    avg_aoi_loss = (1 - aoi_factor.mean()) * 100

    print(f"   Average AOI loss: {avg_aoi_loss:.2f}%")

    # Compare with and without losses
    poa_no_losses = poa_model.calculate_poa_components(
        irrad_components,
        solar_pos_annual,
        transposition_model="perez",
        include_spectral=False,
        include_aoi=False,
    )

    total_loss = (
        1 - poa_components.poa_global.sum() / poa_no_losses.poa_global.sum()
    ) * 100
    print(f"   Total optical losses (spectral + AOI): {total_loss:.2f}%")
    print()

    # ======================
    # 7. RESOURCE ANALYSIS
    # ======================
    print("7. Performing solar resource statistical analysis...")

    analyzer = SolarResourceAnalyzer(poa_components.poa_global, data_label="POA Global")

    # Monthly statistics
    monthly_stats = analyzer.monthly_averages()
    print("\n   Monthly averages (W/m²):")
    print(f"   Best month:  {monthly_stats['Mean'].idxmax()} ({monthly_stats['Mean'].max():.1f})")
    print(f"   Worst month: {monthly_stats['Mean'].idxmin()} ({monthly_stats['Mean'].min():.1f})")

    # Seasonal patterns
    seasonal_stats = analyzer.seasonal_patterns(hemishere="north")
    print("\n   Seasonal totals (kWh/m²):")
    for season in seasonal_stats.index:
        print(f"   {season:8s}: {seasonal_stats.loc[season, 'Total']/1000:.1f}")

    # P50/P90 analysis
    p_analysis = analyzer.p50_p90_analysis(time_aggregation="monthly")
    print("\n   P-value analysis (monthly kWh/m²):")
    for percentile in ["P10", "P50", "P90"]:
        value = p_analysis["summary"][p_analysis["summary"]["Percentile"] == percentile][
            "Value"
        ].values[0]
        print(f"   {percentile}: {value/1000:.1f}")

    # Resource summary
    summary = analyzer.generate_resource_summary()
    print(f"\n   Resource statistics:")
    print(f"   Mean: {summary.mean:.1f} W/m²")
    print(f"   Coefficient of variation: {summary.coefficient_of_variation:.3f}")
    print(f"   P90/P50 ratio: {summary.p90/summary.p50:.3f}")

    # Capacity factor estimation
    cf_range = analyzer.calculate_capacity_factor_range(
        system_capacity_kw=100.0, performance_ratio=0.85
    )
    print(f"\n   Estimated capacity factors (100 kW system, PR=0.85):")
    print(f"   P50: {cf_range['p50']:.2%} ({cf_range['p50_annual_kwh']:.0f} kWh/year)")
    print(f"   P90: {cf_range['p90']:.2%} ({cf_range['p90_annual_kwh']:.0f} kWh/year)")
    print()

    # ======================
    # 8. VISUALIZATIONS
    # ======================
    print("8. Generating interactive visualizations...")

    viz = SolarResourceVisualizer(theme="plotly_white")

    # Time series plot (one month)
    month_data = poa_components.poa_global.loc["2024-06"]
    fig1 = viz.plot_irradiance_timeseries(
        month_data, title="POA Irradiance - June 2024", show_average=True
    )
    fig1.write_html("/tmp/poa_timeseries.html")
    print("   ✓ Time series chart -> /tmp/poa_timeseries.html")

    # Component breakdown
    week_components = POAComponents(
        poa_global=poa_components.poa_global.loc["2024-06-15":"2024-06-21"],
        poa_direct=poa_components.poa_direct.loc["2024-06-15":"2024-06-21"],
        poa_diffuse=poa_components.poa_diffuse.loc["2024-06-15":"2024-06-21"],
        poa_ground=poa_components.poa_ground.loc["2024-06-15":"2024-06-21"],
    )
    fig2 = viz.plot_poa_components(week_components, title="POA Components - Summer Week")
    fig2.write_html("/tmp/poa_components.html")
    print("   ✓ Component breakdown -> /tmp/poa_components.html")

    # Heat map
    resource_maps = analyzer.solar_resource_maps()
    fig3 = viz.plot_resource_heatmap(
        resource_maps["hourly_by_month"],
        title="Solar Resource Heat Map - Hour × Month",
        xlabel="Month",
        ylabel="Hour of Day",
    )
    fig3.write_html("/tmp/resource_heatmap.html")
    print("   ✓ Resource heat map -> /tmp/resource_heatmap.html")

    # Annual profile
    daily_poa = poa_components.poa_global.resample("D").sum()
    fig4 = viz.plot_annual_profile(daily_poa, title="Annual POA Irradiance Profile")
    fig4.write_html("/tmp/annual_profile.html")
    print("   ✓ Annual profile -> /tmp/annual_profile.html")

    # P50/P90 analysis
    fig5 = viz.plot_p50_p90_analysis(p_analysis, title="P50/P90 Exceedance Analysis")
    fig5.write_html("/tmp/p50_p90_analysis.html")
    print("   ✓ P50/P90 analysis -> /tmp/p50_p90_analysis.html")

    # Comparison chart
    comparison_data = {
        "GHI": irradiance_data["ghi"].loc["2024-06-21"],
        "POA (no losses)": poa_no_losses.poa_global.loc["2024-06-21"],
        "POA (with losses)": poa_components.poa_global.loc["2024-06-21"],
    }
    fig6 = viz.plot_comparison_chart(
        comparison_data,
        title="Irradiance Comparison - Summer Solstice",
        chart_type="line",
    )
    fig6.write_html("/tmp/comparison.html")
    print("   ✓ Comparison chart -> /tmp/comparison.html")

    # Dashboard
    fig7 = viz.create_dashboard(
        poa_components.poa_global, poa_components=None, resource_stats=summary
    )
    fig7.write_html("/tmp/dashboard.html")
    print("   ✓ Complete dashboard -> /tmp/dashboard.html")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • Location: {location.name}")
    print(f"  • Annual GHI: {irradiance_data['ghi'].sum()/1000:.0f} kWh/m²")
    print(f"  • Annual POA: {poa_components.poa_global.sum()/1000:.0f} kWh/m²")
    print(f"  • Optical losses: {total_loss:.1f}%")
    print(f"  • P90 capacity factor: {cf_range['p90']:.1%}")
    print()
    print("View the generated HTML files in /tmp/ for interactive visualizations.")
    print()


if __name__ == "__main__":
    main()
