# IEC 61730 Safety Testing Module

Comprehensive implementation of IEC 61730-1 (Construction Requirements) and IEC 61730-2 (Testing Requirements) standards for PV module safety qualification.

## Overview

This module provides a complete safety testing and qualification system for photovoltaic modules per international standards IEC 61730-1:2016 and IEC 61730-2:2016.

## Features

### Core Components

1. **IEC61730SafetyTester** - Main orchestrator class
   - Coordinates all safety testing activities
   - Manages test execution and results
   - Generates safety classifications
   - Issues safety certificates

2. **ElectricalSafetyTest** - Electrical safety testing (MST 01-05)
   - Insulation resistance test (MST 01)
   - Wet leakage current test (MST 02)
   - Dielectric strength test (MST 03)
   - Ground continuity test (MST 04)
   - Bypass diode thermal test (MST 05)

3. **FireSafetyClassification** - Fire safety testing (Annex C / UL 790)
   - Spread of flame test
   - Fire penetration test
   - Flying brand test
   - Fire classification (Class A, B, C)
   - Roof mounting fire safety assessment

4. **SafetyQualificationReport** - Report generation and documentation
   - Comprehensive test result documentation
   - Pass/fail criteria validation
   - Safety classification summary
   - PDF and JSON export
   - Certificate generation

5. **SafetyTestUI** - Interactive Streamlit interface
   - Test configuration
   - Real-time test execution
   - Results visualization
   - Certificate generation
   - Report export

### Safety Classifications

- **Safety Class**
  - Class I: Protective earthing required
  - Class II: Double/reinforced insulation
  - Class III: Extra-low voltage (SELV/PELV)

- **Application Class**
  - Class A: Hazardous voltage, not accessible
  - Class B: Hazardous voltage, accessible
  - Class C: Safe voltage

- **Fire Class** (per UL 790)
  - Class A: Highest fire resistance
  - Class B: Medium fire resistance
  - Class C: Basic fire resistance
  - Not Rated: Does not meet minimum requirements

## Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Python API

```python
from datetime import datetime
from src.models.safety_models import (
    SafetyTestConfig,
    SafetyClass,
    ApplicationClass,
    FireClass,
)
from src.safety.iec61730_tester import IEC61730SafetyTester
from src.safety.safety_report import SafetyQualificationReport

# Configure test
config = SafetyTestConfig(
    module_id="MOD-001",
    manufacturer="SolarTech Inc.",
    model_number="ST-400-72MH",
    max_system_voltage_v=1000.0,
    module_area_m2=2.0,
    application_class=ApplicationClass.CLASS_B,
    target_safety_class=SafetyClass.CLASS_II,
    target_fire_class=FireClass.CLASS_A,
    test_laboratory="TUV Rheinland",
    test_date=datetime.now(),
    perform_electrical_tests=True,
    perform_mechanical_tests=True,
    perform_fire_tests=True,
    perform_environmental_tests=True,
)

# Run tests
tester = IEC61730SafetyTester(config)
results = tester.run_all_tests()

print(f"Overall Pass: {results.overall_pass}")
print(f"Safety Class: {results.classification.safety_class}")
print(f"Fire Class: {results.classification.fire_class}")

# Generate certificate (if passed)
if results.overall_pass:
    certificate = tester.export_safety_certificate(
        certification_body="TUV Rheinland"
    )
    print(f"Certificate: {certificate.certificate_number}")

# Generate report
report = SafetyQualificationReport(results, certificate)
report.export_to_pdf("safety_report.pdf")
report.export_to_json("safety_report.json")
```

### Streamlit UI

```bash
# Run the interactive UI
streamlit run src/ui/safety_test_ui.py
```

Then navigate to http://localhost:8501 in your browser.

## Test Categories

### Electrical Safety Tests (IEC 61730-2 MST 01-05)

- **MST 01: Insulation Resistance**
  - Test voltage: 500V DC (Vmax ≤ 50V) or 1000V DC (Vmax > 50V)
  - Minimum resistance: 40 MΩ
  - Duration: 60 seconds

- **MST 02: Wet Leakage Current**
  - Test voltage: 1.25 × Vmax,dc
  - Maximum current: 275 μA
  - Water spray: 10 minutes

- **MST 03: Dielectric Strength**
  - Test voltage: 1.5 × Vmax,dc + 1000V
  - Duration: 60 seconds
  - Pass: No breakdown

- **MST 04: Ground Continuity** (Class I only)
  - Test current: 10A
  - Maximum resistance: 0.1 Ω

- **MST 05: Bypass Diode Thermal**
  - Fault current: Per design
  - Duration: 2 hours
  - Pass: No thermal runaway

### Mechanical Safety Tests (IEC 61730-2 MST 06-08)

- **MST 06: Mechanical Load**
  - Load: 2400 Pa (or design load)
  - Cycles: 3 (front, rear, front)
  - Pass: No visual defects

- **MST 07: Impact Resistance**
  - Ice ball: 25mm diameter
  - Locations: 11 points
  - Pass: Electrical safety maintained

- **MST 08: Robustness of Terminations**
  - Pull force: 100N
  - Torque: 1 Nm
  - Pass: No displacement or damage

### Fire Safety Tests (IEC 61730-2 Annex C / UL 790)

- **Spread of Flame**
  - Class A: Spread < 183cm (6 ft)
  - Class B: Spread < 244cm (8 ft)
  - Class C: Spread < 396cm (13 ft)

- **Fire Penetration**
  - Class A: No burn-through for 90 min
  - Class B: No burn-through for 60 min
  - Class C: No burn-through for 20 min

- **Flying Brand**
  - Class A: 12" × 12" brand
  - Class B: 6" × 6" brand
  - Class C: 1.5" diameter brand

### Environmental Safety Tests (IEC 61730-2 MST 09-11)

- **MST 09: UV Preconditioning**
  - UV dose: 15 kWh/m²
  - Pass: No visual degradation

- **MST 10: Thermal Cycling**
  - Cycles: 200
  - Range: -40°C to +85°C
  - Pass: No electrical failure

- **MST 11: Humidity-Freeze**
  - Cycles: 10
  - Conditions: 85°C/85%RH to -40°C
  - Pass: No electrical failure

## Data Models

All data is validated using Pydantic models:

- `SafetyTestConfig` - Test configuration
- `SafetyTestResult` - Complete test results
- `ElectricalSafetyTestResult` - Electrical test results
- `MechanicalSafetyTestResult` - Mechanical test results
- `FireSafetyTestResult` - Fire test results
- `EnvironmentalSafetyTestResult` - Environmental test results
- `SafetyClassification` - Safety class determination
- `SafetyCertificate` - Safety certificate

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/safety/test_electrical_safety.py
pytest tests/safety/test_fire_safety.py
pytest tests/safety/test_iec61730_tester.py
```

## Standards Compliance

This implementation follows:

- **IEC 61730-1:2016** - Photovoltaic (PV) module safety qualification - Part 1: Requirements for construction
- **IEC 61730-2:2016** - Photovoltaic (PV) module safety qualification - Part 2: Requirements for testing
- **UL 790** - Standard Test Methods for Fire Tests of Roof Coverings

## Documentation

All functions include comprehensive docstrings with:
- Parameter descriptions
- Return type information
- Usage examples
- Standard references

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure:
- Full type hints on all functions
- Comprehensive docstrings
- Unit tests for new features
- No placeholder code or TODOs

## Support

For issues and questions, please use the GitHub issue tracker.
