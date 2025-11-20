# IEC 61215 Test Simulator - Technical Documentation

## Overview

The IEC 61215 Test Simulator provides comprehensive simulation of module qualification tests (MQT) as defined in the IEC 61215 international standard for design qualification and type approval of terrestrial photovoltaic (PV) modules.

## IEC 61215 Standard

IEC 61215 is the international standard that specifies requirements for the design qualification and type approval of terrestrial photovoltaic modules suitable for long-term operation in general open-air climates.

### Test Sequence Overview

The simulator implements the following key tests from IEC 61215:

| Test ID | Test Name | Description | Standard Requirement |
|---------|-----------|-------------|---------------------|
| MQT-01 | Visual Inspection | Pre-test visual examination | No defects |
| MQT-02 | Maximum Power Determination | STC power measurement | Baseline measurement |
| MQT-10 | Thermal Cycling | -40°C to +85°C cycling | 200 cycles |
| MQT-11 | Humidity Freeze | 85°C/85%RH + -40°C | 10 cycles |
| MQT-12 | Damp Heat | 85°C/85%RH exposure | 1000 hours |
| MQT-13 | UV Preconditioning | UV radiation exposure | 15 kWh/m² |
| MQT-17 | Hail Impact | Ice ball impact test | 25mm @ 23 m/s |
| MQT-18 | Mechanical Load | Static/cyclic load test | ±2400 Pa |

## Architecture

### Class Structure

```
IEC61215TestSimulator
├── Initialization
│   ├── module: ModuleConfig
│   ├── random_seed: Optional[int]
│   └── strictness_factor: float
│
├── Test Methods
│   ├── simulate_thermal_cycling()
│   ├── simulate_humidity_freeze()
│   ├── simulate_damp_heat()
│   ├── simulate_uv_preconditioning()
│   ├── simulate_hail_impact()
│   └── simulate_mechanical_load()
│
├── Report Generation
│   └── generate_qualification_report()
│
├── Visualization
│   ├── plot_power_degradation_timeline()
│   └── plot_iv_curve_comparison()
│
└── Export
    ├── export_report_to_excel()
    └── export_report_to_pdf()
```

### Data Models

#### ModuleConfig
Comprehensive module configuration including:
- Electrical characteristics (Voc, Isc, Vmp, Imp, Pmax)
- Physical properties (dimensions, weight, area)
- Materials (glass, encapsulant, backsheet, frame)
- Temperature coefficients
- Operating conditions

#### TestResults
Individual test result containing:
- Test identification and metadata
- Power measurements (before/after)
- Visual defect observations
- Electrical measurements (insulation, leakage)
- I-V curve data
- Test-specific parameters

#### QualificationReport
Comprehensive qualification report including:
- All test results
- Overall pass/fail status
- Total degradation calculation
- Compliance checks
- Recommendations

## Degradation Modeling

### Physical Degradation Mechanisms

The simulator models realistic degradation mechanisms based on scientific literature and field experience:

#### 1. Thermal Cycling (MQT-10)
**Mechanism**: Thermomechanical stress from coefficient of thermal expansion (CTE) mismatch

**Degradation factors**:
- Base degradation: 0.05% per cycle
- Material modifiers:
  - POE encapsulant: 0.7× (better thermal stability)
  - Glass-glass construction: 0.8× (better CTE matching)

**Typical degradation**: 0.8-1.2% for 200 cycles

**Common defects**:
- Cell micro-cracks
- Solder bond fatigue
- Delamination at interfaces

#### 2. Humidity Freeze (MQT-11)
**Mechanism**: Moisture ingress + thermal stress + freeze-thaw cycling

**Degradation factors**:
- Base degradation: 0.06% per cycle
- Material modifiers:
  - Tedlar backsheet: 0.9× (better moisture barrier)
  - Glass-glass: 0.6× (excellent moisture barrier)

**Typical degradation**: 0.5-0.7% for 10 cycles

**Common defects**:
- Edge delamination
- Corrosion of metallization
- Encapsulant discoloration

#### 3. Damp Heat (MQT-12)
**Mechanism**: Accelerated aging from combined heat and humidity

**Degradation factors**:
- Base degradation: 5.0% per 1000 hours
- Material modifiers:
  - POE encapsulant: 0.7×
  - Glass-glass: 0.5×
  - Tedlar backsheet: 0.9×

**Typical degradation**: 2.0-4.0% for 1000 hours

**Common defects**:
- Encapsulant yellowing/browning
- Delamination
- Reduced insulation resistance

#### 4. UV Preconditioning (MQT-13)
**Mechanism**: Photodegradation of polymeric materials

**Degradation factors**:
- Base degradation: 2.0% for 15 kWh/m²
- Material modifiers:
  - POE encapsulant: 0.6× (excellent UV resistance)
  - Tedlar backsheet: 0.8×

**Typical degradation**: 0.8-1.6% for standard dose

**Common defects**:
- Backsheet discoloration
- Encapsulant yellowing

#### 5. Hail Impact (MQT-17)
**Mechanism**: Kinetic energy transfer causing glass fracture and cell damage

**Damage threshold**:
- Energy = 0.5 × mass × velocity²
- Mass ∝ diameter³
- Critical threshold ∝ glass thickness

**Degradation levels**:
- < 50% threshold: No damage
- 50-80% threshold: Minor cracks (0.1-0.5% degradation)
- 80-100% threshold: Moderate damage (1-3% degradation)
- > 100% threshold: Failure (>5% degradation)

#### 6. Mechanical Load (MQT-18)
**Mechanism**: Bending stress causing cell cracks and interconnect failure

**Deflection calculation**:
- δ ∝ F × L³ / (E × I)
- Frame support reduces deflection by 70%

**Degradation levels**:
- < 50% max deflection: No damage
- 50-100% max deflection: Minor damage (0.1-0.5%)
- 100-150% max deflection: Moderate damage (1-3%)
- > 150% max deflection: Failure (>5%)

### Degradation Distribution

Power degradation is distributed between voltage and current parameters:
- **60%** voltage degradation (Voc, Vmp reduction)
- **40%** current degradation (Isc, Imp reduction)

This distribution reflects typical field observations where voltage degradation is dominant.

## Pass/Fail Criteria

### Overall Requirements

| Requirement | Threshold | Criticality |
|-------------|-----------|-------------|
| Total Power Degradation | ≤ 5.0% | Critical |
| Power Retention | ≥ 95.0% | Critical |
| Insulation Resistance | ≥ 40 MΩ·m² | Critical |
| Wet Leakage Current | < 1 mA (Class A) | Critical |
| Hotspot Temperature | < 20°C above average | Major |
| Visual Defects | No major/critical defects | Major |

### Test-Specific Criteria

Each test has specific pass/fail criteria:

1. **Thermal Cycling**
   - Degradation ≤ 5%
   - No major visual defects
   - Insulation resistance maintained

2. **Humidity Freeze**
   - Degradation ≤ 5%
   - Insulation ≥ 40 MΩ·m²
   - Leakage current < 1 mA
   - No major delamination

3. **Damp Heat**
   - Degradation ≤ 5%
   - Insulation ≥ 40 MΩ·m²
   - Leakage current < 1 mA

4. **UV Preconditioning**
   - Degradation ≤ 5%
   - Acceptable discoloration

5. **Hail Impact**
   - No glass penetration
   - No critical cell damage
   - Insulation maintained

6. **Mechanical Load**
   - Deflection within limits
   - No structural failure
   - No broken cells

### Status Categories

- **PASSED**: All criteria met
- **CONDITIONAL**: Minor issues, degradation 3-5%
- **FAILED**: Critical criteria not met or degradation > 5%

## Usage Examples

### Basic Test Sequence

```python
from src.models import ModuleConfig, CellTechnology
from src.modules.iec_61215_simulator import IEC61215TestSimulator

# Define module configuration
module = ModuleConfig(
    name="Example-400W",
    technology=CellTechnology.PERC,
    rated_power=400.0,
    voc=49.5,
    isc=10.5,
    vmp=41.2,
    imp=9.71,
    efficiency=20.5,
    area=1.95,
    cells_in_series=72,
    dimensions=[1980, 990, 40],
    weight=22.5,
)

# Create simulator
simulator = IEC61215TestSimulator(module, random_seed=42)

# Run tests
thermal_test = simulator.simulate_thermal_cycling(module, cycles=200)
damp_heat_test = simulator.simulate_damp_heat(module, hours=1000)
hail_test = simulator.simulate_hail_impact(module)

# Generate report
all_tests = [thermal_test, damp_heat_test, hail_test]
report = simulator.generate_qualification_report(all_tests)

print(f"Overall Status: {report.overall_status.value}")
print(f"Total Degradation: {report.total_power_degradation:.2f}%")
```

### Complete Qualification Testing

```python
# Run all standard tests
tests = [
    simulator.simulate_thermal_cycling(module, cycles=200),
    simulator.simulate_humidity_freeze(module, cycles=10),
    simulator.simulate_damp_heat(module, hours=1000),
    simulator.simulate_uv_preconditioning(module, hours=48, dose=15.0),
    simulator.simulate_hail_impact(module, diameter=25.0, velocity=23.0),
    simulator.simulate_mechanical_load(module, front_load=2400, back_load=2400),
]

# Generate comprehensive report
report = simulator.generate_qualification_report(tests)

# Export reports
from pathlib import Path
output_dir = Path("output")
simulator.export_report_to_excel(report, output_dir / "report.xlsx")
simulator.export_report_to_pdf(report, output_dir / "report.pdf")

# Generate visualizations
simulator.plot_power_degradation_timeline(tests, output_dir / "timeline.png")
simulator.plot_iv_curve_comparison(tests[0], output_dir / "iv_curve.png")
```

### Custom Test Scenarios

```python
# Extreme thermal cycling
extreme_thermal = simulator.simulate_thermal_cycling(module, cycles=400)

# Extended damp heat
extended_dh = simulator.simulate_damp_heat(module, hours=2000)

# Severe hail
severe_hail = simulator.simulate_hail_impact(module, diameter=35.0, velocity=30.0)

# High mechanical load
high_load = simulator.simulate_mechanical_load(module, front_load=4800, back_load=4800)
```

## Visualization Capabilities

### 1. Power Degradation Timeline
Shows cumulative power degradation across all tests with:
- Initial power baseline
- Test-by-test degradation
- 95% threshold line
- Percentage labels

### 2. I-V Curve Comparison
Before/after comparison showing:
- I-V curves (current vs voltage)
- P-V curves (power vs voltage)
- MPP markers
- Key metrics (Pmax, FF, degradation)

### 3. Export Formats

**Excel Report**:
- Summary sheet (module info, overall status)
- Test results sheet (all tests)
- Visual defects sheet (detailed defect log)
- Compliance sheet (pass/fail checklist)

**PDF Report**:
- Professional formatting
- Module information table
- Test summary table
- Detailed test results
- Recommendations

## Advanced Features

### Material-Dependent Degradation

The simulator accounts for material properties:

| Material | Property | Impact |
|----------|----------|--------|
| POE encapsulant | UV resistance | -40% UV degradation |
| POE encapsulant | Moisture resistance | -30% damp heat degradation |
| Glass-glass | Moisture barrier | -50% humidity degradation |
| Glass-glass | Thermal stability | -20% thermal degradation |
| Tedlar backsheet | Moisture barrier | -10% humidity degradation |
| Aluminum frame | Structural support | -70% deflection |

### Strictness Factor

Control simulation severity:
- `strictness_factor = 0.5`: Lenient (50% degradation)
- `strictness_factor = 1.0`: Standard (100% degradation)
- `strictness_factor = 1.5`: Strict (150% degradation)

### Random Seed

Set `random_seed` for:
- Reproducible results (testing, validation)
- Consistent comparisons between module designs
- Monte Carlo analysis (vary seed for statistical distribution)

## Validation and Accuracy

The simulator is based on:
- IEC 61215:2021 standard specifications
- Published degradation data from field studies
- Industry testing laboratory results
- Material science research

Typical accuracy:
- **Degradation trends**: ±20% (matches field variability)
- **Relative comparisons**: High accuracy (consistent physics)
- **Material effects**: Qualitatively accurate

## Limitations

1. **Simplified physics**: Uses empirical models, not detailed finite element analysis
2. **Combined effects**: Tests simulated independently (actual testing may have interactions)
3. **Manufacturing variability**: Does not model cell-to-cell variation within module
4. **Long-term degradation**: Does not include time-dependent mechanisms beyond test duration
5. **Environmental factors**: Does not include wind, rain, soiling, or other field conditions

## Best Practices

1. **Always use random seed** for reproducible results
2. **Run complete test sequences** for accurate total degradation
3. **Compare similar designs** using same strictness factor
4. **Validate against real data** when available
5. **Use conservative assumptions** for safety-critical applications

## References

1. IEC 61215:2021 - Terrestrial photovoltaic (PV) modules - Design qualification and type approval
2. IEC 61730 - Photovoltaic (PV) module safety qualification
3. NREL - PV Module Reliability Testing and Modeling
4. PV QA Task Force - Testing Standards and Best Practices

## Support

For questions, issues, or contributions:
- Documentation: `/docs`
- Examples: `/examples`
- Issue tracking: Project repository
