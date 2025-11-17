# Thermal Modeling Documentation

## Overview

This document provides detailed information about the thermal modeling capabilities in the PV Circularity Simulator.

## Temperature Prediction Models

### 1. Sandia Array Performance Model

**Description:** Empirical model developed by Sandia National Laboratories based on extensive outdoor testing.

**Equation:**
```
T_module = T_ambient + (E / E0) * exp(a + b * ws) + ΔT
```

**Parameters:**
- `a`: Empirical coefficient (dimensionless), typically -3.47
- `b`: Empirical coefficient (s/m), typically -0.0594
- `ΔT`: Temperature difference between cell and module back (°C), typically 2-5°C
- `E`: Solar irradiance (W/m²)
- `E0`: Reference irradiance (1000 W/m²)
- `ws`: Wind speed (m/s)

**Best for:** General purpose, well-validated for field conditions

**Reference:** King et al. (2004), SAND2004-3535

### 2. PVsyst Temperature Model

**Description:** Heat loss factor model used in PVsyst software.

**Equation:**
```
T_cell = T_ambient + (E / (u_c + u_v * ws)) * (1 - η)
```

**Parameters:**
- `u_c`: Constant heat loss factor (W/(m²·K)), typically 20-30
- `u_v`: Wind-dependent heat loss factor (W/(m²·K)/(m/s)), typically 0-6
- `η`: Module efficiency (fraction)

**Best for:** PVsyst software users, simulation consistency

**Reference:** Mermoud (2012), PVsyst User's Manual

### 3. Faiman Temperature Model

**Description:** Two-parameter model based on heat transfer principles.

**Equation:**
```
T_module = T_ambient + (E * α) / (u0 + u1 * ws)
```

**Parameters:**
- `u0`: Constant heat transfer coefficient (W/(m²·K)), typically 20-30
- `u1`: Wind-dependent heat transfer coefficient (W/(m²·K)/(m/s)), typically 5-10
- `α`: Solar absorptivity (dimensionless), typically 0.9

**Best for:** Physics-based understanding, parameter optimization

**Reference:** Faiman (2008), Progress in Photovoltaics

### 4. NOCT-based Model

**Description:** Simple model based on Nominal Operating Cell Temperature rating.

**Equation:**
```
T_cell = T_ambient + (NOCT - 20) * (E / 800) * wind_correction
```

**Parameters:**
- `NOCT`: Nominal Operating Cell Temperature (°C)
- Standard conditions: 800 W/m², 20°C ambient, 1 m/s wind

**Best for:** Quick estimates, datasheet-based modeling

**Reference:** IEC 61215

## Heat Transfer Physics

### Convective Heat Transfer

**Front Surface:**
```python
h_conv_front = h_base + 4.0 * wind_speed
```

**Back Surface:**
```python
h_conv_back = h_conv_front * reduction_factor
```

Reduction factors by mounting type:
- Open rack: 0.7
- Roof mounted: 0.35
- Ground mounted: 0.7
- Building integrated: 0.21

### Radiative Heat Transfer

**Linearized radiation coefficient:**
```
h_rad = 4 * σ * ε * T_mean³
```

Where:
- σ = Stefan-Boltzmann constant (5.67×10⁻⁸ W/(m²·K⁴))
- ε = Surface emissivity (typically 0.85)
- T_mean = Mean temperature (K)

### Thermal Time Constants

**Heating time constant:**
```
τ_heating = C / h_front
```

**Cooling time constant:**
```
τ_cooling = C / (h_front + h_back)
```

Where:
- C = Heat capacity per unit area (J/(m²·K))
- h = Total heat transfer coefficient (W/(m²·K))

**Response times:**
- 63% response: τ
- 95% response: 3τ

## Mounting Configuration Effects

### Open Rack
- Best cooling performance
- Full exposure to wind on both sides
- Typical ΔT from ambient: 20-25°C at 1000 W/m²

### Roof Mounted
- Moderate cooling
- Restricted back surface airflow
- Typical ΔT from ambient: 25-30°C at 1000 W/m²

### Ground Mounted
- Good cooling performance
- Enhanced convection from ground proximity
- Typical ΔT from ambient: 22-27°C at 1000 W/m²

### Building Integrated
- Poorest cooling
- Minimal airflow on back surface
- Typical ΔT from ambient: 30-40°C at 1000 W/m²

## Wind Speed Effects

### Low Wind (0-2 m/s)
- Natural convection dominates
- Temperature highly dependent on mounting
- Large temperature variations

### Moderate Wind (2-5 m/s)
- Mixed convection regime
- Moderate cooling improvement
- Most common operating conditions

### High Wind (>5 m/s)
- Forced convection dominates
- Significant cooling effect
- Temperature less dependent on mounting

## Performance Impact

### Temperature Coefficients

Typical values:
- Power: -0.3% to -0.5% per °C
- Voltage (Voc): -0.25% to -0.35% per °C
- Current (Isc): +0.03% to +0.06% per °C

### Power Loss Calculation

```python
ΔP = P_rated * β * (T_cell - 25)
```

Where:
- β = Temperature coefficient of power (%/°C)
- T_cell = Cell temperature (°C)

Example: 400W module, β = -0.4%/°C, T_cell = 50°C
```
ΔP = 400 * (-0.004) * (50 - 25) = -40W
```

## Best Practices

### Model Selection

1. **General purpose:** Sandia model
2. **Software compatibility:** PVsyst model
3. **Physics understanding:** Faiman model
4. **Quick estimates:** NOCT-based model

### Data Requirements

**Minimum:**
- Ambient temperature
- Solar irradiance
- Wind speed
- Module NOCT rating

**Recommended:**
- Module thermal properties
- Mounting configuration details
- Local environmental data
- Temperature coefficients

### Validation

1. Compare multiple models
2. Validate against field measurements
3. Check physical reasonableness
4. Account for uncertainty

## Example Use Cases

### Case 1: Design Optimization

**Objective:** Select mounting configuration for maximum energy yield

**Approach:**
1. Run all models for each mounting type
2. Calculate temperature distributions
3. Estimate energy losses
4. Compare cost vs. performance

### Case 2: Performance Troubleshooting

**Objective:** Diagnose underperformance

**Approach:**
1. Measure actual temperatures
2. Compare with model predictions
3. Identify cooling issues
4. Recommend improvements

### Case 3: Energy Forecasting

**Objective:** Predict daily/annual energy production

**Approach:**
1. Use time series weather data
2. Calculate hourly temperatures
3. Apply temperature derating
4. Integrate for energy estimate

## Advanced Topics

### Custom Thermal Models

Users can implement custom models:

```python
def my_thermal_model(conditions, thermal_params, mounting, k=1.0):
    """Custom temperature model."""
    return conditions.ambient_temp + k * conditions.irradiance / 30.0

result = model.custom_thermal_models(
    model_func=my_thermal_model,
    model_params={'k': 1.2}
)
```

### Multi-layer Thermal Modeling

For advanced analysis:
1. Glass layer temperature
2. Cell layer temperature
3. Backsheet temperature
4. Frame temperature

### Spectral Effects

Consider wavelength-dependent absorption:
- Direct vs. diffuse irradiance
- Angle of incidence effects
- Spectral mismatch

## Troubleshooting

### Issue: Unrealistic temperatures

**Possible causes:**
- Incorrect input parameters
- Missing wind speed data
- Wrong mounting configuration

**Solutions:**
- Validate input ranges
- Check model parameters
- Compare multiple models

### Issue: Large model discrepancies

**Possible causes:**
- Different model assumptions
- Parameter uncertainty
- Extreme conditions

**Solutions:**
- Use ensemble average
- Validate with measurements
- Document assumptions

## References

### Key Papers

1. King, D. L., et al. (2004). "Sandia Photovoltaic Array Performance Model." Sandia Report SAND2004-3535.

2. Faiman, D. (2008). "Assessing the outdoor operating temperature of photovoltaic modules." Progress in Photovoltaics: Research and Applications, 16(4), 307-315.

3. Mermoud, A., & Lejeune, T. (2010). "Partial shadings on PV arrays: By-pass diode benefits analysis." Proceedings of 25th EU PVSEC, Valencia, Spain.

### Standards

1. IEC 61215 - Terrestrial photovoltaic (PV) modules - Design qualification and type approval

2. IEC 61853 - Photovoltaic (PV) module performance testing and energy rating

### Software

1. pvlib-python: https://pvlib-python.readthedocs.io/
2. PVsyst: https://www.pvsyst.com/
3. SAM (System Advisor Model): https://sam.nrel.gov/

## Support

For questions or issues:
- Open an issue on GitHub
- Consult the API documentation
- Review example notebooks
