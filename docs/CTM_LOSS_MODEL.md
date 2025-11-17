# CTM Loss Modeling Engine Documentation

## Overview

The CTM (Cell-to-Module) Loss Modeling Engine implements the **Fraunhofer ISE SmartCalc methodology** for comprehensive analysis of power losses and gains when transitioning from individual solar cells to complete PV modules.

This implementation includes all **k1-k24 factors** covering optical, electrical, coupling, and environmental effects, with support for advanced module architectures.

## Features

### Comprehensive k-Factor Analysis

#### Optical Losses/Gains (k1-k7)
- **k1**: Glass reflection gain (0.5-2%)
- **k2**: Encapsulant current gain (yellow EVA effect)
- **k3**: Front grid shading correction
- **k4**: Inactive cell area losses
- **k5**: Front glass absorption
- **k6**: Encapsulant absorption and UV yellowing
- **k7**: Rear side optical properties (bifacial gain)

#### Coupling Effects (k8-k11)
- **k8**: Cell-to-cell gap losses
- **k9**: Internal cell mismatch
- **k10**: Module-to-module mismatch
- **k11**: LID/LETID degradation effects

#### Electrical Losses (k12-k15)
- **k12**: Resistive losses (ribbon, busbar, junction box)
- **k13**: Cell interconnection resistance
- **k14**: Manufacturing damage and quality losses
- **k15**: Inactive cell area electrical loss

#### Environmental Factors (k21-k24)
- **k21**: Temperature coefficient effects
- **k22**: Low irradiance performance
- **k23**: Spectral response differences
- **k24**: Angle of incidence (AOI) effects

### Supported Module Architectures

- **Standard**: Traditional full-cell modules (60/72 cells)
- **Half-Cut**: Reduced resistive losses, better shading performance
- **Quarter-Cut**: Further improved electrical characteristics
- **Shingled**: Overlapping cells, no gap losses
- **IBC (Interdigitated Back Contact)**: No front grid shading
- **Bifacial**: Front + rear power generation

### Key Capabilities

1. **Complete CTM Analysis**: Calculate module power from cell parameters
2. **Loss Breakdown**: Categorized loss/gain analysis
3. **Waterfall Visualization**: Visual representation of cumulative effects
4. **Sensitivity Analysis**: Parameter impact assessment
5. **Environmental Modeling**: Temperature, irradiance, AOI effects
6. **Production-Ready**: Full type hints, validation, documentation

## Quick Start

### Basic Usage

```python
from src.modules.ctm_loss_model import (
    CTMLossModel,
    CellParameters,
    ModuleParameters,
    ModuleType,
)

# Define cell parameters (PERC cell example)
cell = CellParameters(
    power_stc=5.25,           # Cell power at STC (W)
    voltage_mpp=0.650,        # Voltage at MPP (V)
    current_mpp=8.08,         # Current at MPP (A)
    voltage_oc=0.720,         # Open circuit voltage (V)
    current_sc=8.60,          # Short circuit current (A)
    efficiency=22.8,          # Cell efficiency (%)
    width=166,                # Cell width (mm)
    height=166,               # Cell height (mm)
)

# Define module parameters
module = ModuleParameters(
    cells_in_series=60,       # 60 cells in series
    module_type=ModuleType.STANDARD,
)

# Create model and analyze
model = CTMLossModel(cell, module)

# Calculate all k-factors
k_factors = model.calculate_all_k_factors()

# Get module power
module_power = model.calculate_module_power()
print(f"Module Power: {module_power:.2f} W")

# Get CTM ratio
ctm_ratio = model.get_ctm_ratio()
print(f"CTM Ratio: {ctm_ratio:.4f} ({(ctm_ratio-1)*100:+.2f}%)")

# Generate report
print(model.generate_report())
```

### Half-Cut Module Analysis

```python
# Half-cut modules reduce current and resistive losses
module_halfcut = ModuleParameters(
    cells_in_series=60,
    cells_in_parallel=2,      # 2 parallel strings
    module_type=ModuleType.HALF_CUT,
)

model_hc = CTMLossModel(cell, module_halfcut)
power_hc = model_hc.calculate_module_power()

# Half-cut typically shows 0.5-1% improvement
```

### Bifacial Module with Rear Gain

```python
module_bifacial = ModuleParameters(
    cells_in_series=60,
    is_bifacial=True,
    bifaciality_factor=0.75,   # Rear efficiency = 75% of front
    rear_glass=True,           # Glass-glass construction
)

model_bif = CTMLossModel(cell, module_bifacial)
power_bif = model_bif.calculate_module_power()

# Bifacial can provide 15-30% additional power depending on albedo
```

### Visualization

```python
# Generate waterfall chart
fig = model.generate_loss_waterfall(title="CTM Loss Analysis")
fig.savefig('ctm_waterfall.png', dpi=150, bbox_inches='tight')

# With Plotly (interactive)
fig_plotly = model.generate_loss_waterfall(use_plotly=True)
fig_plotly.show()
```

### Sensitivity Analysis

```python
# Analyze impact of cell efficiency variation
results = model.sensitivity_analysis(
    parameter='cell.efficiency',
    variation_range=(0.9, 1.1),  # ±10%
    num_points=20
)

# Plot results
import matplotlib.pyplot as plt
plt.plot(results['parameter_values'], results['module_power'])
plt.xlabel('Cell Efficiency (%)')
plt.ylabel('Module Power (W)')
plt.show()

# Multi-parameter analysis
params = ['cell.efficiency', 'module.glass_thickness', 'module.cell_gap']
multi_results = model.multi_parameter_sensitivity(params)
```

## Detailed Parameter Reference

### CellParameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `power_stc` | float | required | 0.5-10 W | Cell power at STC |
| `voltage_mpp` | float | required | >0 V | Voltage at max power |
| `current_mpp` | float | required | >0 A | Current at max power |
| `voltage_oc` | float | required | >0 V | Open circuit voltage |
| `current_sc` | float | required | >0 A | Short circuit current |
| `efficiency` | float | required | 0-100% | Cell efficiency |
| `width` | float | required | >0 mm | Cell width |
| `height` | float | required | >0 mm | Cell height |
| `thickness` | float | 180 μm | >0 | Cell thickness |
| `front_grid_coverage` | float | 2.5% | 0-100 | Metallization shading |
| `inactive_area_fraction` | float | 0.5% | 0-100 | Inactive edges |
| `temp_coeff_power` | float | -0.40 %/°C | - | Power temperature coefficient |
| `temp_coeff_voltage` | float | -0.30 %/°C | - | Voltage temperature coefficient |
| `temp_coeff_current` | float | 0.05 %/°C | - | Current temperature coefficient |
| `lid_factor` | float | 1.5% | 0-10 | Light-induced degradation |
| `letid_factor` | float | 0.0% | 0-10 | LETID degradation |
| `low_irradiance_loss` | float | 0.5% | 0-5 | Low light loss |
| `spectral_mismatch` | float | 0.0% | -5 to +5 | Spectral response variation |

### ModuleParameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `module_type` | ModuleType | STANDARD | enum | Module architecture |
| `cells_in_series` | int | required | >0 | Cells in series string |
| `cells_in_parallel` | int | 1 | >0 | Parallel strings |
| `cell_gap` | float | 2.0 mm | ≥0 | Gap between cells |
| `border_width` | float | 10.0 mm | ≥0 | Module border |
| `glass_thickness` | float | 3.2 mm | >0 | Front glass thickness |
| `glass_transmittance` | float | 91.5% | 0-100 | Glass transmission |
| `glass_ar_coating` | bool | True | - | AR coating present |
| `glass_reflection_gain` | float | 1.0% | 0-5 | Reflection gain |
| `encapsulant_type` | EncapsulantType | STANDARD_EVA | enum | Encapsulant material |
| `encapsulant_thickness` | float | 0.45 mm | >0 | Encapsulant thickness |
| `encapsulant_yellowing` | float | 0.0% | 0-10 | UV yellowing loss |
| `encapsulant_current_gain` | float | 0.5% | 0-5 | Current gain |
| `is_bifacial` | bool | False | - | Bifacial module |
| `rear_glass` | bool | False | - | Glass-glass construction |
| `bifaciality_factor` | float | 0.70 | 0-1 | Rear/front efficiency ratio |
| `ribbon_width` | float | 1.5 mm | >0 | Interconnect ribbon width |
| `ribbon_thickness` | float | 0.2 mm | >0 | Ribbon thickness |
| `busbar_count` | int | 5 | >0 | Busbars per cell |
| `junction_box_loss` | float | 0.5% | 0-5 | J-box resistive loss |
| `cell_mismatch` | float | 1.0% | 0-5 | Cell-to-cell mismatch |
| `module_mismatch` | float | 0.5% | 0-3 | Module-to-module mismatch |
| `manufacturing_damage` | float | 0.5% | 0-5 | Manufacturing losses |
| `operating_temperature` | float | 25°C | - | Operating temperature |
| `irradiance` | float | 1000 W/m² | >0 | Irradiance level |
| `aoi_angle` | float | 0° | 0-90 | Angle of incidence |

## k-Factor Details

### Optical Factors

#### k1: Glass Reflection Gain
- **Typical Range**: 1.005 - 1.020 (0.5% - 2% gain)
- **Physics**: Light reflected from glass back to cell
- **Influencing Factors**: AR coating, glass quality
- **Implementation**: Enhanced by AR coating (+0.5%)

#### k2: Encapsulant Gain
- **Typical Range**: 1.003 - 1.015 (0.3% - 1.5% gain)
- **Physics**: Spectral shifting, optical coupling
- **Influencing Factors**: Encapsulant type, thickness
- **Implementation**: Yellow EVA provides +0.5% bonus

#### k3: Grid Correction
- **Typical Range**: 0.97 - 0.99 (1% - 3% loss)
- **Physics**: Front metallization shading differences
- **Influencing Factors**: Busbar count, module design
- **Implementation**: IBC shows gain (no front grid)

#### k4: Inactive Area
- **Typical Range**: 0.995 - 1.0 (0% - 0.5% loss)
- **Physics**: Cell edge regions with reduced activity
- **Influencing Factors**: Cell design, module architecture
- **Implementation**: Shingled/cut cells recover some area

#### k5: Glass Absorption
- **Typical Range**: 0.96 - 0.985 (1.5% - 4% loss)
- **Physics**: Glass material absorption, especially UV
- **Influencing Factors**: Glass thickness, AR coating
- **Implementation**: Thicker glass = more absorption

#### k6: Encapsulant Absorption
- **Typical Range**: 0.985 - 1.0 (0% - 1.5% loss)
- **Physics**: UV-induced yellowing (EVA)
- **Influencing Factors**: Material type, UV exposure
- **Implementation**: POE/silicone more stable than EVA

#### k7: Rear Optical
- **Typical Range**:
  - Monofacial: 1.0 - 1.01
  - Bifacial: 1.1 - 1.3 (10% - 30% gain)
- **Physics**: Rear side light collection
- **Influencing Factors**: Bifaciality, albedo, mounting
- **Implementation**: Assumes 20% rear irradiance for bifacial

### Coupling Factors

#### k8: Cell Gap Losses
- **Typical Range**: 0.95 - 0.99 (1% - 5% loss)
- **Physics**: Inactive area between cells
- **Influencing Factors**: Gap width, cell count
- **Implementation**: Shingled = 1.0 (no gaps)

#### k9: Internal Mismatch
- **Typical Range**: 0.98 - 0.995 (0.5% - 2% loss)
- **Physics**: Cells in series limited by weakest
- **Influencing Factors**: Binning quality, design
- **Implementation**: Half/quarter-cut reduce impact

#### k10: Module Mismatch
- **Typical Range**: 0.995 - 0.998 (0.2% - 0.5% loss)
- **Physics**: Module-to-module variation in arrays
- **Influencing Factors**: Manufacturing consistency
- **Implementation**: System-level factor

#### k11: LID/LETID
- **Typical Range**: 0.97 - 0.99 (1% - 3% loss)
- **Physics**: Boron-oxygen defects, elevated temp effects
- **Influencing Factors**: Cell type, compensation
- **Implementation**: Multiplicative combination

### Electrical Factors

#### k12: Resistive Losses
- **Typical Range**: 0.97 - 0.99 (1% - 3% loss)
- **Physics**: I²R losses in interconnects
- **Influencing Factors**: Current, ribbon size
- **Implementation**: Half-cut shows improvement

#### k13: Interconnection
- **Typical Range**: 0.995 - 0.999 (0.1% - 0.5% loss)
- **Physics**: Contact resistance
- **Influencing Factors**: Soldering quality, design
- **Implementation**: Shingled higher loss (adhesive)

#### k14: Manufacturing Damage
- **Typical Range**: 0.995 - 0.998 (0.2% - 0.5% loss)
- **Physics**: Micro-cracks, handling damage
- **Influencing Factors**: Process quality
- **Implementation**: Cell type dependent

#### k15: Inactive Electrical
- **Typical Range**: 0.997 - 1.0 (0% - 0.3% loss)
- **Physics**: Electrical inactive regions
- **Influencing Factors**: Similar to k4 but electrical
- **Implementation**: Smaller than optical inactive

### Environmental Factors

#### k21: Temperature
- **Typical Range**: 0.85 - 1.0 (based on operating temp)
- **Physics**: Band gap temperature dependence
- **Influencing Factors**: Temp coefficient, ΔT
- **Implementation**: Linear from STC (25°C)

#### k22: Low Irradiance
- **Typical Range**: 0.95 - 1.0 (at <800 W/m²)
- **Physics**: Shunt resistance, recombination effects
- **Influencing Factors**: Irradiance level
- **Implementation**: Loss increases as irradiance decreases

#### k23: Spectral Response
- **Typical Range**: 0.98 - 1.02 (±2%)
- **Physics**: Spectrum variations vs. AM1.5G
- **Influencing Factors**: Weather, location, cell type
- **Implementation**: User-defined mismatch factor

#### k24: Angle of Incidence
- **Typical Range**: 0.5 - 1.0 (angle dependent)
- **Physics**: Fresnel reflection, cosine loss
- **Influencing Factors**: AOI angle, AR coating
- **Implementation**: Cosine + reflection correction

## Validation

The implementation has been validated against:

1. **Cell-to-Module.com** reference data
2. **Fraunhofer ISE SmartCalc** methodology
3. **Industry standard CTM ratios**:
   - Standard 60-cell PERC: 96-97%
   - Half-cut modules: 97-98%
   - Shingled modules: 97-99%
   - Bifacial modules: 110-130% (with rear irradiance)

## Examples

See `/examples/ctm_demo.py` for comprehensive demonstrations including:
- Standard module analysis
- Half-cut comparison
- Bifacial rear gain
- Shingled advantages
- Sensitivity analysis
- Environmental effects
- Architecture comparisons

## Testing

Run comprehensive test suite:

```bash
pytest tests/test_ctm_loss_model.py -v
```

Test coverage includes:
- Parameter validation
- Individual k-factor calculations
- Module power calculations
- Advanced architectures
- Sensitivity analysis
- Known value validation

## References

1. Fraunhofer ISE, "SmartCalc CTM", https://www.ise.fraunhofer.de
2. Cell-to-Module.com, CTM Calculator and Database
3. IEC 61853 series: PV module performance testing
4. Photovoltaic Module Power Rating per IEC 61853-1

## License

See LICENSE file in repository root.

## Authors

PV Circularity Simulator Development Team

## Version

v0.1.0 - Initial implementation with full k1-k24 factors
