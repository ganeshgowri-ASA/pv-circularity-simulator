# Module Temperature & NOCT Calculations

Comprehensive photovoltaic module temperature modeling system with support for multiple industry-standard calculation methods, mounting configurations, and thermal effects.

## Features

### 1. NOCT (Nominal Operating Cell Temperature) Calculation
- **Standard conditions**: 800 W/m², 20°C ambient, 1 m/s wind, open rack
- **Mounting configuration factors**: Open rack, close roof, building-integrated, ground mount, trackers
- **Wind speed effects**: Convective cooling modeling
- **Module backside ventilation**: Different thermal resistance values

### 2. Operating Temperature Models

#### Simple Linear Model
Basic NOCT-based calculation:
```
Tmod = Tamb + (NOCT - 20) * (G / 800)
```

#### Ross/Faiman Model
Empirical model with wind speed dependency:
```
Tmod = Tamb + (G / (a + b * v))
```

#### Sandia Model
Detailed model from Sandia National Laboratories:
```
Tmod = G * exp(a + b*v) + Tamb + G/1000 * DeltaC
```

#### King Model (SAPM)
Sandia Array Performance Model temperature equations with NOCT-derived coefficients.

### 3. Mounting Configuration Effects

| Mounting Type | Temperature Adjustment |
|--------------|------------------------|
| Open Rack | Baseline (0°C) |
| Ground Mount | -2°C (optimized clearance) |
| Single-axis Tracker | -3°C (better ventilation) |
| Dual-axis Tracker | -4°C (best ventilation) |
| Close Roof Mount | +12.5°C (limited airflow) |
| Building Integrated (BIPV) | +25°C (minimal ventilation) |

### 4. Temperature Coefficient Application

Default temperature coefficients by technology:

| Technology | Temp Coefficient (%/°C) |
|-----------|-------------------------|
| HJT (Heterojunction) | -0.25 |
| CdTe Thin Film | -0.25 |
| TOPCon | -0.35 |
| PERC | -0.37 |
| Bifacial | -0.38 |
| Mono-Si | -0.40 |
| Poly-Si | -0.43 |

### 5. Additional Features
- **Seasonal variations**: Latitude-based temperature adjustments
- **Time-of-day modeling**: Diurnal temperature patterns
- **Thermal time constant**: Module thermal response dynamics
- **System yield integration**: Power loss calculations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic NOCT Calculation

```python
from src.modules import ModuleTemperatureModel, MountingType

model = ModuleTemperatureModel()

noct = model.calculate_noct(
    ambient_temp=25.0,
    irradiance=1000.0,
    wind_speed=2.0,
    mounting=MountingType.OPEN_RACK
)

print(f"NOCT: {noct:.2f}°C")
```

### Module Temperature Calculation

```python
from src.modules import (
    ModuleTemperatureModel,
    TemperatureModelType,
    MountingType
)

model = ModuleTemperatureModel()

module_temp = model.calculate_module_temp(
    ambient=30.0,
    irr=950.0,
    wind=1.5,
    noct=45.0,
    mounting=MountingType.CLOSE_ROOF,
    model_type=TemperatureModelType.SANDIA
)

print(f"Module temperature: {module_temp:.2f}°C")
```

### Temperature Coefficient Losses

```python
from src.modules import ModuleTemperatureModel, ModuleTechnology

model = ModuleTemperatureModel()

loss_factor = model.calculate_temp_coefficient_losses(
    module_temp=55.0,
    technology=ModuleTechnology.MONO_SI
)

stc_power = 400.0  # W
actual_power = stc_power * loss_factor

print(f"Power at 55°C: {actual_power:.1f} W ({loss_factor*100:.1f}% of STC)")
```

### Comprehensive Analysis

```python
from src.modules import (
    ModuleTemperatureModel,
    ModuleSpecification,
    ModuleTechnology,
    MountingType,
    TemperatureModelType
)

model = ModuleTemperatureModel()

module_spec = ModuleSpecification(
    area=1.7,
    mass=18.0,
    technology=ModuleTechnology.MONO_SI,
    noct=45.0,
    temp_coeff_power=-0.40
)

result = model.calculate_comprehensive_temperature(
    ambient_base=20.0,
    irradiance=950.0,
    wind_speed=2.0,
    module_spec=module_spec,
    day_of_year=195,  # Mid-summer
    hour=14.0,  # 2 PM
    latitude=35.0,
    mounting=MountingType.OPEN_RACK,
    model_type=TemperatureModelType.SIMPLE_LINEAR
)

print(f"Module temperature: {result.module_temperature:.2f}°C")
print(f"Power loss factor: {result.power_loss_factor:.4f}")
print(f"Metadata: {result.metadata}")
```

## Examples

Run the comprehensive demonstration:

```bash
python examples/module_temperature_demo.py
```

This will show:
1. Basic NOCT calculations
2. Temperature model comparisons
3. Mounting configuration effects
4. Temperature coefficient impacts
5. Seasonal variations
6. Time-of-day variations
7. Thermal time constant calculations
8. Comprehensive temperature analysis
9. Power calculations with temperature effects

## API Reference

### Main Classes

#### `ModuleTemperatureModel`
Main class for temperature calculations.

**Methods:**
- `calculate_noct()`: Calculate NOCT with adjustments
- `calculate_module_temp()`: Calculate operating temperature
- `calculate_temp_coefficient_losses()`: Calculate power loss factor
- `model_thermal_time_constant()`: Calculate thermal time constant
- `calculate_seasonal_adjustment()`: Seasonal temperature variation
- `calculate_time_of_day_adjustment()`: Diurnal temperature variation
- `calculate_comprehensive_temperature()`: Complete analysis

### Enumerations

#### `MountingType`
- `OPEN_RACK`: Full ventilation, baseline temperature
- `CLOSE_ROOF`: <6 inches clearance, limited airflow
- `BUILDING_INTEGRATED`: Minimal ventilation, highest temperature
- `GROUND_MOUNT`: Optimized clearance
- `TRACKER_SINGLE_AXIS`: Better ventilation
- `TRACKER_DUAL_AXIS`: Best ventilation and orientation

#### `TemperatureModelType`
- `SIMPLE_LINEAR`: Basic NOCT-based model
- `ROSS_FAIMAN`: Empirical model with wind effects
- `SANDIA`: Detailed Sandia National Labs model
- `KING_SAPM`: Sandia Array Performance Model

#### `ModuleTechnology`
- `MONO_SI`: Monocrystalline silicon
- `POLY_SI`: Polycrystalline silicon
- `HJT`: Heterojunction (best temp coefficient)
- `PERC`: Passivated Emitter Rear Cell
- `TOPCON`: Tunnel Oxide Passivated Contact
- `THIN_FILM_CDTE`: CdTe thin film
- `THIN_FILM_CIGS`: CIGS thin film
- `BIFACIAL`: Bifacial modules

### Pydantic Models

All input/output data is validated using Pydantic models:
- `NOCTCalculationInput`: NOCT calculation parameters
- `ModuleTemperatureInput`: Temperature calculation parameters
- `TemperatureCoefficientInput`: Temp coefficient calculation parameters
- `ModuleSpecification`: Complete module specifications
- `TemperatureCalculationResult`: Comprehensive calculation results

## References

### Standards
- **IEC 61215**: Crystalline silicon terrestrial photovoltaic modules
- **ASTM E1036**: Standard Test Methods for Electrical Performance of NOCT

### Scientific Literature
1. **Faiman, D. (2008)**: "Assessing the outdoor operating temperature of PV modules"
2. **King, D.L., et al. (2004)**: "Sandia Photovoltaic Array Performance Model"
3. **Kurnik, J., et al. (2011)**: "Outdoor testing of PV module temperature and performance under different mounting and operational conditions"
4. **Ross, R.G., & Smokler, M.I. (1986)**: "Flat-plate solar array project final report"

### Industry Resources
- Sandia National Laboratories PV Performance Modeling Collaborative
- NREL System Advisor Model (SAM) documentation
- PVsyst temperature modeling methodology

## Advanced Usage

### Custom Temperature Model Coefficients

```python
# Ross/Faiman with custom coefficients
temp = model.calculate_module_temp(
    ambient=28.0,
    irr=1000.0,
    wind=3.0,
    noct=45.0,
    model_type=TemperatureModelType.ROSS_FAIMAN,
    ross_a=22.0,  # Custom coefficient
    ross_b=7.2    # Custom coefficient
)
```

### Thermal Time Constant for Transient Analysis

```python
module_spec = ModuleSpecification(
    area=1.7,
    mass=18.0,
    specific_heat=900.0,
    emissivity=0.84,
    technology=ModuleTechnology.MONO_SI,
    noct=45.0
)

tau = model.model_thermal_time_constant(module_spec)
print(f"Thermal time constant: {tau:.0f} seconds ({tau/60:.1f} minutes)")
```

### Seasonal and Geographic Analysis

```python
# Compare different latitudes
latitudes = [0, 30, 45, 60]  # Equator to high latitude

for lat in latitudes:
    summer_temp = model.calculate_seasonal_adjustment(
        day_of_year=195,  # Mid-summer
        latitude=lat,
        base_ambient=20.0
    )
    winter_temp = model.calculate_seasonal_adjustment(
        day_of_year=15,  # Mid-winter
        latitude=lat,
        base_ambient=20.0
    )
    print(f"Latitude {lat}°: Summer={summer_temp:.1f}°C, Winter={winter_temp:.1f}°C")
```

## Testing

```bash
# Run tests (when test suite is created)
pytest tests/modules/test_module_temperature.py -v

# Run with coverage
pytest tests/modules/test_module_temperature.py --cov=src.modules.module_temperature
```

## Contributing

Contributions are welcome! Please ensure:
1. All new features include comprehensive docstrings
2. Type hints are provided for all functions
3. Pydantic models validate all inputs
4. No placeholder implementations
5. Code follows PEP 8 style guidelines

## License

See LICENSE file in repository root.

## Authors

PV Circularity Simulator Team

## Version History

- **0.1.0** (2025-11-17): Initial implementation
  - NOCT calculations
  - Multiple temperature models
  - Mounting configuration effects
  - Temperature coefficient losses
  - Seasonal and time-of-day variations
  - Thermal time constant modeling
