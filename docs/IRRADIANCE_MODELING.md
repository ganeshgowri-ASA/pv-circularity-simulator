# Irradiance Modeling & Solar Resource Assessment

Comprehensive solar irradiance modeling system with production-ready implementations of industry-standard algorithms.

## Overview

This module provides complete functionality for:

- **Solar Position Calculation**: High-accuracy solar position algorithms (NREL SPA, ephemeris)
- **Irradiance Decomposition**: Separate GHI into DNI and DHI components using validated models
- **Transposition Models**: Convert horizontal irradiance to tilted plane-of-array (POA) values
- **Loss Modeling**: Account for spectral mismatch and angle-of-incidence losses
- **Statistical Analysis**: P50/P90 resource assessment for financial modeling
- **Visualizations**: Interactive Plotly charts for professional reporting

## Features

### IrradianceCalculator

Core calculator for irradiance decomposition and transposition.

**Key Methods:**

- `get_solar_position()`: Calculate solar zenith, azimuth, and elevation angles
- `ghi_dni_dhi_decomposition()`: Decompose GHI using DIRINT, DISC, or Erbs models
- `perez_transposition()`: Perez anisotropic sky model (most accurate)
- `hay_davies_model()`: Hay-Davies transposition model
- `isotropic_sky()`: Simple isotropic diffuse model
- `anisotropic_corrections()`: Quantify anisotropic effects
- `calculate_clearness_index()`: Atmospheric transparency metric

**Example:**

```python
from src.irradiance.calculator import IrradianceCalculator
from src.irradiance.models import LocationConfig

location = LocationConfig(
    latitude=39.7555,
    longitude=-105.2211,
    altitude=1829,
    timezone="America/Denver",
    name="NREL Golden"
)

calc = IrradianceCalculator(location)

# Calculate solar position
times = pd.date_range('2024-01-01', periods=24, freq='h', tz='America/Denver')
solar_pos = calc.get_solar_position(times)

# Decompose GHI into DNI and DHI
components = calc.ghi_dni_dhi_decomposition(ghi_data, times=times, model='dirint')
```

### POAIrradianceModel

Plane-of-array irradiance calculation with loss factors.

**Key Methods:**

- `direct_beam()`: Direct beam irradiance on tilted surface
- `sky_diffuse()`: Sky diffuse component (multiple transposition models)
- `ground_reflected()`: Ground-reflected irradiance
- `spectral_corrections()`: Spectral mismatch factor
- `aoi_losses()`: Angle-of-incidence loss factor
- `calculate_poa_components()`: Complete POA breakdown
- `calculate_effective_irradiance()`: POA with all loss factors

**Example:**

```python
from src.irradiance.poa_model import POAIrradianceModel
from src.irradiance.models import SurfaceConfig

surface = SurfaceConfig(
    tilt=30.0,      # degrees
    azimuth=180.0,  # south-facing
    albedo=0.2      # ground reflectance
)

poa_model = POAIrradianceModel(location, surface)

# Calculate POA components
poa_components = poa_model.calculate_poa_components(
    irradiance_components,
    solar_position,
    transposition_model='perez',
    include_spectral=True,
    include_aoi=True,
    module_type='multisi'
)

# Access results
total_poa = poa_components.poa_global
direct_component = poa_components.poa_direct
diffuse_component = poa_components.poa_diffuse
```

### SolarResourceAnalyzer

Statistical analysis for solar resource assessment.

**Key Methods:**

- `monthly_averages()`: Monthly resource statistics
- `seasonal_patterns()`: Seasonal variability analysis
- `interannual_variability()`: Year-to-year variation
- `p50_p90_analysis()`: Exceedance probability for financial modeling
- `solar_resource_maps()`: Heat map data preparation
- `calculate_capacity_factor_range()`: Expected system capacity factors
- `identify_resource_anomalies()`: Data quality and extreme events

**Example:**

```python
from src.irradiance.resource_analyzer import SolarResourceAnalyzer

analyzer = SolarResourceAnalyzer(poa_data, data_label="POA Global")

# Monthly statistics
monthly_stats = analyzer.monthly_averages()
print(monthly_stats)

# P50/P90 analysis for financial modeling
p_analysis = analyzer.p50_p90_analysis(time_aggregation='monthly')
p50_value = p_analysis['summary'].loc[p_analysis['summary']['Percentile'] == 'P50', 'Value'].values[0]
p90_value = p_analysis['summary'].loc[p_analysis['summary']['Percentile'] == 'P90', 'Value'].values[0]

# Capacity factor estimation
cf_range = analyzer.calculate_capacity_factor_range(
    system_capacity_kw=100.0,
    performance_ratio=0.85
)
print(f"P50 CF: {cf_range['p50']:.2%}")
print(f"P90 CF: {cf_range['p90']:.2%}")
```

### SolarResourceVisualizer

Interactive Plotly visualizations for professional reporting.

**Key Methods:**

- `plot_irradiance_timeseries()`: Time series plots
- `plot_poa_components()`: Stacked area chart of POA components
- `plot_resource_heatmap()`: Hour-by-month heat maps
- `plot_annual_profile()`: Annual daily totals with smoothing
- `plot_monthly_boxplot()`: Monthly distribution box plots
- `plot_p50_p90_analysis()`: Exceedance probability visualization
- `plot_comparison_chart()`: Multi-series comparison
- `create_dashboard()`: Comprehensive multi-panel dashboard

**Example:**

```python
from src.ui.visualizations import SolarResourceVisualizer

viz = SolarResourceVisualizer(theme='plotly_white')

# Create heat map
resource_maps = analyzer.solar_resource_maps()
fig = viz.plot_resource_heatmap(
    resource_maps['hourly_by_month'],
    title="Solar Resource Heat Map"
)
fig.write_html("resource_heatmap.html")
fig.show()

# Create dashboard
dashboard = viz.create_dashboard(
    poa_data,
    poa_components=poa_components,
    resource_stats=summary
)
dashboard.write_html("solar_dashboard.html")
```

## Technical Details

### Decomposition Models

**DIRINT** (Direct Insolation Radiation INTegration):
- Most accurate for hourly data
- Recommended by NREL
- Uses clearness index and solar geometry

**DISC** (Direct Insolation Simulation Code):
- Faster, simpler model
- Good for preliminary analysis
- Based on Maxwellian distribution

**Erbs**:
- Works for hourly and daily data
- Uses clearness index correlation
- Good general-purpose model

### Transposition Models

**Perez** (Recommended):
- Accounts for circumsolar and horizon brightening
- Most accurate for POA calculations
- Industry standard for PV modeling

**Hay-Davies**:
- Separates isotropic and circumsolar diffuse
- Good balance of accuracy and simplicity
- Well-validated for tilted surfaces

**Isotropic**:
- Simplest model
- Assumes uniform sky radiance
- Conservative estimates

### Loss Factors

**Spectral Mismatch**:
- Accounts for varying solar spectrum
- Module-type specific corrections
- Based on First Solar model
- Typical impact: ±2-5%

**Angle of Incidence (AOI)**:
- Reflection losses at module surface
- Uses ASHRAE or physical model
- Maximum at low sun angles
- Typical annual impact: 2-4%

## Data Models

All components use Pydantic models for type safety and validation:

- `LocationConfig`: Geographic coordinates and timezone
- `SurfaceConfig`: Tilt, azimuth, albedo
- `IrradianceComponents`: GHI, DNI, DHI
- `POAComponents`: POA global, direct, diffuse, ground
- `SolarPosition`: Zenith, azimuth, elevation
- `ResourceStatistics`: Mean, median, P-values, CV

## Performance Considerations

- **Vectorized Operations**: All calculations use NumPy/pandas for efficiency
- **pvlib Integration**: Leverages well-tested pvlib-python library
- **Memory Efficient**: Handles multi-year hourly datasets
- **Caching**: Solar position can be pre-calculated and reused

## Validation

The implementation follows these standards:

- **pvlib-python**: Industry-standard open-source library
- **NREL Validation**: Algorithms validated against NREL data
- **IEEE Standards**: Compliant with IEEE Std 1562-2007
- **Comprehensive Tests**: Unit tests for all major components

## References

1. Perez, R., et al. (1990). "Modeling daylight availability and irradiance components from direct and global irradiance." Solar Energy.
2. Hay, J. E., & Davies, J. A. (1980). "Calculation of the solar radiation incident on an inclined surface." Proceedings of First Canadian Solar Radiation Data Workshop.
3. Holmgren, W. F., et al. (2018). "pvlib python: A python package for modeling solar energy systems." Journal of Open Source Software.
4. Maxwell, E. L. (1987). "A quasi-physical model for converting hourly global horizontal to direct normal insolation."

## Complete Example

See `examples/complete_irradiance_analysis.py` for a comprehensive demonstration of all features.

```bash
python examples/complete_irradiance_analysis.py
```

This generates:
- Time series plots
- Component breakdowns
- Resource heat maps
- Annual profiles
- P50/P90 analysis
- Interactive dashboards
