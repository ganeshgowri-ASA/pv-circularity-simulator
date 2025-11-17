# Module Configuration Builder

Comprehensive PV module design, configuration, and analysis tools.

## Features

### 1. Module Layout Configurations

Support for all modern module architectures:

- **Standard**: Traditional full-cell modules (60, 72, 120, 132, 144 cells)
- **Half-Cut**: Reduced resistive losses with 2 sub-modules
- **Quarter-Cut**: 4 sub-modules for even better performance
- **Shingled**: Overlapping cells, no gaps, maximum efficiency
- **IBC (Interdigitated Back Contact)**: No visible busbars
- **Bifacial**: Glass-glass construction with rear power generation

### 2. Cell Technologies

Built-in support for:

- Mono PERC
- Mono TOPCon
- Mono HJT (Heterojunction)
- Mono IBC
- Multi-Si
- Perovskite
- Tandem cells

### 3. Module Specifications Calculator

Calculates complete electrical and thermal specifications:

- **Power**: Pmax, Voc, Isc, Vmpp, Impp at STC
- **Temperature Coefficients**: Pmax, Voc, Isc, Vmpp, Impp
- **Thermal**: NOCT calculation
- **Efficiency**: Module efficiency with CTM losses
- **Fill Factor**: Automatic calculation
- **CTM Losses**: Cell-to-Module loss breakdown
  - Resistance losses
  - Reflection losses
  - Mismatch losses
  - Inactive area losses

### 4. PVsyst PAN File Generator

Generate industry-standard PVsyst PAN files with:

- Complete electrical parameters
- Mechanical specifications
- Temperature behavior
- Low irradiance performance
- IAM (Incidence Angle Modifier) curves
- Bifacial parameters (when applicable)

### 5. Design Validation

Comprehensive validation against industry standards:

- Electrical parameter ranges
- Thermal characteristics
- Mechanical specifications
- Layout configuration checks
- Safety requirements
- Performance metrics

### 6. Layout Optimization

Intelligent optimization for different objectives:

- **Efficiency**: Maximize module efficiency
- **Cost**: Minimize cost per watt
- **Performance**: Balanced optimization

Considers:
- Target power output
- Voltage and current constraints
- Physical size limitations
- Cost factors

### 7. Export Capabilities

Export to multiple formats:

- **JSON**: Complete configuration and specifications
- **CSV**: Batch comparison of multiple modules
- **PAN**: PVsyst simulation files

## Quick Start

### Basic Module Creation

```python
from src.modules import (
    CellType,
    LayoutType,
    CellDesign,
    ModuleConfigBuilder
)

# Create a cell design
cell = CellDesign(
    cell_type=CellType.MONO_PERC,
    efficiency=0.225,
    area=0.0244,  # 156mm x 156mm M6 cell
    voltage_oc=0.68,
    current_sc=10.3,
    voltage_mpp=0.58,
    current_mpp=9.8,
    temp_coeff_voc=-0.28,
    temp_coeff_isc=0.05,
    temp_coeff_pmax=-0.35,
    series_resistance=0.005,
    shunt_resistance=500,
    ideality_factor=1.2,
    busbar_count=9
)

# Create module builder
builder = ModuleConfigBuilder()

# Define layout
layout = {
    'layout_type': LayoutType.HALF_CUT,
    'cells_series': 120,
    'cells_parallel': 2,
    'submodules': 2,
    'bypass_diodes': 3
}

# Create module configuration
module = builder.create_module_config(
    cell_design=cell,
    layout=layout,
    name="My 450W Module",
    manufacturer="My Solar Company"
)

# Calculate specifications
specs = builder.calculate_module_specs(module)
print(f"Module Power: {specs.pmax:.1f} W")
print(f"Efficiency: {specs.efficiency*100:.2f}%")
```

### Using Standard Module Templates

```python
from src.modules import create_standard_module, CellType, LayoutType

# Quick creation of standard modules
module = create_standard_module(
    power_class=450,  # Target power in watts
    cell_type=CellType.MONO_PERC,
    layout_type=LayoutType.HALF_CUT,
    manufacturer="Solar Inc."
)
```

### Generate PVsyst PAN File

```python
# Generate PAN file
pan_content = builder.generate_pvsyst_pan_file(module)

# Save to file
with open('my_module.PAN', 'w') as f:
    f.write(pan_content)
```

### Validate Design

```python
# Validate module design
report = builder.validate_module_design(module)

if report.is_valid:
    print("Design is valid!")
else:
    print(f"Design has {report.error_count} errors")
    for issue in report.issues:
        print(f"  {issue.level}: {issue.message}")
```

### Optimize Layout

```python
# Optimize layout for specific constraints
constraints = {
    'target_power': 550,
    'max_voltage': 50,
    'max_current': 15,
    'optimize_for': 'efficiency',  # or 'cost', 'performance'
    'allow_half_cut': True,
    'allow_shingled': True
}

optimal = builder.optimize_cell_layout(cell, constraints)
print(f"Optimal: {optimal.layout.layout_type.value}")
print(f"Cells: {optimal.layout.total_cells}")
print(f"Efficiency gain: {optimal.efficiency_gain:.2f}%")
```

### Export to JSON/CSV

```python
# Export single module to JSON
json_str = builder.export_to_json(module, include_specs=True)

# Export multiple modules to CSV for comparison
modules = [module1, module2, module3]
csv_str = builder.export_to_csv(modules, filepath="comparison.csv")
```

## Advanced Features

### Multi-Busbar (MBB) Support

```python
cell = CellDesign(
    # ... other parameters ...
    busbar_count=9  # or 12, 16 for advanced MBB
)
```

### Bifacial Modules

```python
cell = CellDesign(
    # ... other parameters ...
    is_bifacial=True,
    bifacial_factor=0.75  # 75% rear/front ratio
)

module = builder.create_module_config(
    cell_design=cell,
    layout=layout,
    glass_thickness_rear=2.0  # Required for bifacial
)
```

### CTM Loss Analysis

```python
specs = builder.calculate_module_specs(module)
print(f"Resistance loss: {specs.ctm_loss_resistance:.1f}%")
print(f"Reflection loss: {specs.ctm_loss_reflection:.1f}%")
print(f"Mismatch loss: {specs.ctm_loss_mismatch:.1f}%")
print(f"Inactive area loss: {specs.ctm_loss_inactive:.1f}%")
print(f"Total CTM loss: {specs.ctm_total_loss:.1f}%")
```

## Data Models

### CellDesign

Complete cell-level specifications including electrical parameters, temperature coefficients, and physical properties.

### ModuleLayout

Layout configuration specifying:
- Layout type (standard, half-cut, etc.)
- Series/parallel cell arrangement
- Submodules and bypass diodes
- Cell gaps and overlaps
- Connection type

### ModuleConfig

Complete module definition including:
- Cell design
- Layout configuration
- Mechanical specifications
- Operating conditions
- Certifications

### ModuleSpecs

Calculated specifications including:
- Electrical parameters
- Temperature coefficients
- CTM losses
- Performance characteristics

## Examples

See `examples/module_builder_demo.py` for comprehensive demonstrations of all features.

Run the demo:

```bash
python3 examples/module_builder_demo.py
```

## CTM (Cell-to-Module) Loss Model

The builder includes a comprehensive CTM loss model that accounts for:

1. **Resistance Losses** (0.8-2.5%)
   - Reduced in half-cut configurations
   - Minimal in shingled modules
   - Lower with multi-busbar designs

2. **Reflection Losses** (1.5-3.0%)
   - AR coating effects
   - Glass properties

3. **Mismatch Losses** (0.3-0.8%)
   - Cell-to-cell variations
   - Parallel string effects

4. **Inactive Area Losses** (0.5-1.2%)
   - Cell gaps
   - Frame area
   - Minimal in shingled designs

## PVsyst Integration

Generated PAN files are compatible with PVsyst 7.x and include:

- All electrical parameters at STC
- Temperature coefficients
- NOCT thermal behavior
- IAM profile for angle-dependent performance
- Low irradiance characteristics
- Bifacial parameters (when applicable)

## License

MIT License - see LICENSE file for details.
