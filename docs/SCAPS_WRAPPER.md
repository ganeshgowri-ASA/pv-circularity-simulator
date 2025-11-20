# SCAPS-1D Python Wrapper Documentation

## Overview

The SCAPS-1D Python Wrapper provides a comprehensive interface to the Solar Cell Capacitance Simulator (SCAPS-1D) for simulating photovoltaic device physics. This module enables:

- **Device Physics Simulation**: J-V characteristics, quantum efficiency, band diagrams
- **Multiple Architectures**: PERC, TOPCon, HJT, BSF, PERT, IBC
- **Batch Processing**: Parametric sweeps and optimization
- **Results Management**: Caching, export to JSON/CSV
- **Standard Templates**: Pre-configured cell designs

## Installation

```bash
pip install -r requirements.txt
```

Or with optional dependencies:

```bash
pip install -e ".[dev,viz]"
```

## Quick Start

### Basic Usage

```python
from src.modules import SCAPSInterface, CellTemplates

# Create SCAPS interface
scaps = SCAPSInterface(
    working_directory="./simulations",
    cache_directory="./.scaps_cache",
    enable_cache=True
)

# Create a PERC cell
perc_cell = CellTemplates.create_perc_cell()

# Run simulation
results = scaps.run_simulation(perc_cell)

# Print results
print(f"Efficiency: {results.efficiency*100:.2f}%")
print(f"Voc: {results.voc:.4f} V")
print(f"Jsc: {results.jsc:.4f} mA/cm²")
print(f"FF: {results.ff:.4f}")

# Export results
scaps.export_results(results, "results.json", format="json")
```

### Using Cell Templates

```python
from src.modules import CellTemplates

# PERC cell
perc = CellTemplates.create_perc_cell(
    wafer_thickness=180000.0,  # nm
    emitter_doping=1e19,
    bsf_doping=5e18
)

# TOPCon cell
topcon = CellTemplates.create_topcon_cell(
    wafer_thickness=180000.0,
    tunnel_oxide_thickness=1.5
)

# HJT cell
hjt = CellTemplates.create_hjt_cell(
    wafer_thickness=180000.0
)
```

## Device Physics Parameters

### Material Properties

```python
from src.modules import MaterialProperties, MaterialType

silicon = MaterialProperties(
    material=MaterialType.SILICON,
    bandgap=1.12,
    electron_affinity=4.05,
    dielectric_constant=11.7,
    electron_mobility=1400.0,
    hole_mobility=450.0,
    nc=2.8e19,
    nv=1.04e19,
    electron_lifetime=1e-3,
    hole_lifetime=1e-3,
    auger_electron=2.8e-31,
    auger_hole=9.9e-32,
)
```

### Layer Structure

```python
from src.modules import Layer, DopingProfile, DopingType

layer = Layer(
    name="n+ emitter",
    thickness=500.0,  # nm
    material_properties=silicon,
    doping=DopingProfile(
        doping_type=DopingType.N_TYPE,
        concentration=1e19,
        uniform=False,
        profile_type="gaussian",
        characteristic_length=100.0
    )
)
```

### Defect States

```python
from src.modules import DefectDistribution, DefectType

bulk_defect = DefectDistribution(
    defect_type=DefectType.BULK,
    energy_level=0.56,  # eV from band edge
    total_density=1e12,  # cm⁻³
    electron_capture_cross_section=1e-15,
    hole_capture_cross_section=1e-15,
    energetic_distribution="single",
    donor_type=True
)

interface_defect = DefectDistribution(
    defect_type=DefectType.INTERFACE,
    energy_level=0.56,
    total_density=1e11,  # cm⁻²
    electron_capture_cross_section=1e-15,
    hole_capture_cross_section=1e-15,
)
```

### Contacts

```python
from src.modules import Contact, ContactType

front_contact = Contact(
    contact_type=ContactType.FRONT,
    work_function=4.3,  # eV
    surface_recombination_electron=1e6,  # cm/s
    surface_recombination_hole=1e6,
    series_resistance=0.5,  # Ω·cm²
    shunt_resistance=1e10,
)
```

### Optical Properties

```python
from src.modules import OpticalProperties

optics = OpticalProperties(
    arc_enabled=True,
    arc_thickness=75.0,  # nm
    arc_refractive_index=2.0,
    illumination_spectrum="AM1.5G",
    light_intensity=1000.0,  # W/m²
    front_reflection=0.03,
    back_reflection=0.9,
    wavelength_min=300.0,
    wavelength_max=1200.0,
    wavelength_step=10.0,
)
```

## Advanced Features

### Batch Processing

```python
# Parametric sweep of emitter doping
simulations = []
for doping in [1e18, 5e18, 1e19, 5e19, 1e20]:
    params = CellTemplates.create_perc_cell(emitter_doping=doping)
    simulations.append(params.model_dump())

# Run in parallel
results = scaps.execute_scaps_batch(simulations, max_workers=4)

# Analyze results
for doping, result in zip([1e18, 5e18, 1e19, 5e19, 1e20], results):
    print(f"Doping {doping:.1e}: Efficiency {result.efficiency*100:.2f}%")
```

### Temperature Coefficients

```python
coefficients = scaps.calculate_temperature_coefficients(
    params=perc_cell,
    temp_range=(273.0, 343.0),  # K
    temp_step=5.0
)

print(f"TC Voc: {coefficients['temperature_coefficient_voc']*1000:.2f} mV/K")
print(f"TC Jsc: {coefficients['temperature_coefficient_jsc']:.4f} mA/cm²/K")
print(f"TC Eff: {coefficients['temperature_coefficient_efficiency']*100:.4f} %/K")
```

### Efficiency Optimization

```python
# Define parameters to optimize
optimization_params = {
    'layers.0.doping.concentration': (1e18, 1e20),  # Emitter doping
    'layers.2.doping.concentration': (1e18, 5e19),  # BSF doping
    'layers.0.thickness': (300.0, 800.0),  # Emitter thickness
}

# Run optimization
opt_params, best_results = scaps.optimize_efficiency(
    base_params=perc_cell,
    optimization_params=optimization_params,
    max_iterations=50
)

print(f"Optimized efficiency: {best_results.efficiency*100:.2f}%")
```

### Caching

The wrapper automatically caches simulation results based on parameter hash:

```python
# First run - executes simulation
results1 = scaps.run_simulation(perc_cell)

# Second run - returns cached result (instant)
results2 = scaps.run_simulation(perc_cell)
```

Disable caching:

```python
scaps = SCAPSInterface(enable_cache=False)
```

## Simulation Results

### J-V Characteristics

```python
results = scaps.run_simulation(perc_cell)

# Access J-V data
voltage = results.voltage  # List[float]
current_density = results.current_density  # List[float]
power_density = results.power_density  # List[float]

# Performance metrics
voc = results.voc  # Open-circuit voltage (V)
jsc = results.jsc  # Short-circuit current density (mA/cm²)
ff = results.ff  # Fill factor (0-1)
efficiency = results.efficiency  # Efficiency (0-1)
vmp = results.vmp  # Voltage at max power (V)
jmp = results.jmp  # Current at max power (mA/cm²)
pmax = results.pmax  # Maximum power (mW/cm²)
```

### Quantum Efficiency

```python
# QE data (if available)
wavelength = results.wavelength  # nm
eqe = results.eqe  # External quantum efficiency
iqe = results.iqe  # Internal quantum efficiency
reflectance = results.reflectance
```

### Band Diagrams

```python
# Band diagram data (if available)
position = results.position  # nm
ec = results.ec  # Conduction band (eV)
ev = results.ev  # Valence band (eV)
ef = results.ef  # Fermi level (eV)
```

### Generation/Recombination

```python
# Generation and recombination profiles
generation_rate = results.generation_rate  # cm⁻³·s⁻¹
recombination_rate = results.recombination_rate  # cm⁻³·s⁻¹
```

## Export Formats

### JSON Export

```python
scaps.export_results(results, "output.json", format="json")
```

JSON structure:
```json
{
  "voltage": [...],
  "current_density": [...],
  "voc": 0.72,
  "jsc": 42.0,
  "ff": 0.82,
  "efficiency": 0.248,
  ...
}
```

### CSV Export

```python
scaps.export_results(results, "output.csv", format="csv")
```

CSV structure:
```
Voltage (V),Current Density (mA/cm²),Power Density (mW/cm²)
0.0,42.0,0.0
0.01,41.95,0.4195
...

Metric,Value
Voc (V),0.72
Jsc (mA/cm²),42.0
FF,0.82
Efficiency (%),24.8
```

## Cell Architectures

### PERC (Passivated Emitter and Rear Cell)

```python
perc = CellTemplates.create_perc_cell(
    wafer_thickness=180000.0,  # 180 µm
    emitter_doping=1e19,
    bsf_doping=5e18
)
```

**Structure:**
- n+ front emitter (500 nm)
- p-type base (180 µm)
- p+ BSF (2 µm)
- Front ARC (Si₃N₄)
- Rear reflector

### TOPCon (Tunnel Oxide Passivated Contact)

```python
topcon = CellTemplates.create_topcon_cell(
    wafer_thickness=180000.0,
    tunnel_oxide_thickness=1.5
)
```

**Structure:**
- n+ front emitter (500 nm)
- p-type base (180 µm)
- Tunnel oxide (1.5 nm)
- n++ poly-Si contact (100 nm)
- Front ARC

### HJT (Heterojunction Technology)

```python
hjt = CellTemplates.create_hjt_cell(
    wafer_thickness=180000.0
)
```

**Structure:**
- Front ITO (80 nm)
- n-type a-Si (5 nm)
- i-type a-Si (5 nm)
- n-type c-Si base (180 µm)
- i-type a-Si (5 nm)
- p-type a-Si (5 nm)
- Rear ITO (80 nm)

## Custom Device Creation

### Complete Example

```python
from src.modules import (
    DeviceParams,
    CellArchitecture,
    Layer,
    MaterialProperties,
    MaterialType,
    DopingProfile,
    DopingType,
    Contact,
    ContactType,
    OpticalProperties,
    SimulationSettings,
    InterfaceProperties,
)

# Define materials
si_props = MaterialProperties(
    material=MaterialType.SILICON,
    bandgap=1.12,
    electron_affinity=4.05,
    dielectric_constant=11.7,
    electron_mobility=1400.0,
    hole_mobility=450.0,
    nc=2.8e19,
    nv=1.04e19,
)

# Create layers
emitter = Layer(
    name="emitter",
    thickness=500.0,
    material_properties=si_props,
    doping=DopingProfile(
        doping_type=DopingType.N_TYPE,
        concentration=1e19,
        uniform=True
    )
)

base = Layer(
    name="base",
    thickness=180000.0,
    material_properties=si_props,
    doping=DopingProfile(
        doping_type=DopingType.P_TYPE,
        concentration=1.5e16,
        uniform=True
    )
)

# Create interfaces
interface = InterfaceProperties(
    name="emitter-base",
    layer1_index=0,
    layer2_index=1,
    sn=1e3,
    sp=1e3,
)

# Create device
device = DeviceParams(
    architecture=CellArchitecture.BSF,
    device_name="Custom Silicon Cell",
    layers=[emitter, base],
    interfaces=[interface],
    front_contact=Contact(
        contact_type=ContactType.FRONT,
        work_function=4.3,
        surface_recombination_electron=1e6,
        surface_recombination_hole=1e6,
    ),
    back_contact=Contact(
        contact_type=ContactType.BACK,
        work_function=4.3,
        surface_recombination_electron=1e6,
        surface_recombination_hole=1e6,
    ),
    optics=OpticalProperties(
        arc_enabled=True,
        arc_thickness=75.0,
        arc_refractive_index=2.0,
        illumination_spectrum="AM1.5G",
        light_intensity=1000.0,
    ),
    settings=SimulationSettings(
        temperature=300.0,
        voltage_min=0.0,
        voltage_max=0.8,
        voltage_step=0.01,
    )
)

# Run simulation
results = scaps.run_simulation(device)
```

## Error Handling

```python
from pydantic import ValidationError

try:
    # Invalid parameters will raise ValidationError
    layer = Layer(
        name="invalid",
        thickness=-100.0,  # Negative!
        material_properties=si_props,
        doping=DopingProfile(
            doping_type=DopingType.P_TYPE,
            concentration=1e16,
            uniform=True
        )
    )
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    # Simulation errors raise RuntimeError
    results = scaps.run_simulation(invalid_params)
except RuntimeError as e:
    print(f"Simulation failed: {e}")
```

## Testing

Run unit tests:

```bash
pytest tests/unit/test_scaps_wrapper.py -v
```

Run with coverage:

```bash
pytest tests/unit/test_scaps_wrapper.py --cov=src.modules.scaps_wrapper --cov-report=html
```

## Performance Notes

1. **Caching**: Enable caching for repeated simulations with identical parameters
2. **Batch Processing**: Use `execute_scaps_batch()` for parametric sweeps (parallel execution)
3. **Mock Mode**: Currently uses mock SCAPS output for development/testing
4. **Real SCAPS**: To use actual SCAPS executable, provide path during initialization

## API Reference

See inline docstrings for detailed API documentation:

```python
help(SCAPSInterface)
help(CellTemplates)
help(DeviceParams)
```

## Contributing

When extending the wrapper:

1. Add comprehensive docstrings
2. Include type hints for all functions
3. Write unit tests for new features
4. Update this documentation
5. Follow PEP 8 style guidelines

## License

See LICENSE file for details.
