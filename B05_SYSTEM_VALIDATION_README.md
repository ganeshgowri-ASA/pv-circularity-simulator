# BATCH4-B05-S06: System Validation & Engineering Reports

Comprehensive system validation module with automated design checks, engineering calculations verification, compliance reporting, and professional documentation generation for all PV system types.

## Features

### Core Components

1. **SystemValidator** - Main orchestrator for complete system validation
   - `validate_complete_design()` - Comprehensive design validation
   - `check_electrical_codes()` - NEC/IEC compliance checking
   - `verify_structural_requirements()` - Building/fire code validation
   - `validate_performance_metrics()` - Performance validation
   - `generate_validation_report()` - Professional report generation

2. **CodeComplianceChecker** - Code compliance validation
   - `nec_690_compliance()` - NEC Article 690 compliance
   - `iec_60364_compliance()` - IEC 60364-7-712 compliance
   - `local_code_verification()` - Jurisdiction-specific codes
   - `building_code_checks()` - Structural/wind/snow loads
   - `fire_safety_compliance()` - IFC fire safety requirements

3. **EngineeringCalculationVerifier** - Electrical calculations
   - `verify_string_calculations()` - String voltage/current validation
   - `check_voltage_drop()` - DC voltage drop calculations
   - `validate_short_circuit()` - Short circuit current analysis
   - `verify_grounding()` - Grounding conductor sizing
   - `confirm_overcurrent_protection()` - OCPD selection

4. **PerformanceValidator** - Performance metrics validation
   - `energy_yield_sanity_check()` - Energy yield validation
   - `pr_range_validation()` - Performance ratio validation
   - `loss_budget_verification()` - Loss budget analysis
   - `compare_to_benchmarks()` - Industry benchmark comparison
   - `flag_unrealistic_results()` - Unrealistic result detection

5. **DocumentationGenerator** - Professional documentation
   - `generate_engineering_package()` - Complete engineering PDF
   - `create_stamped_drawings()` - Stamped engineering drawings
   - `produce_specification_sheets()` - Equipment specifications
   - `generate_O_and_M_manual()` - Operations & Maintenance manual
   - `create_commissioning_checklist()` - Commissioning procedures
   - `export_cad_drawing()` - CAD/DXF export
   - `create_calculations_spreadsheet()` - Excel calculations

6. **SystemValidationUI** - Streamlit dashboard
   - Interactive validation dashboard
   - Automated checks list
   - Compliance matrix display
   - Issue tracking interface
   - Report generation UI
   - Export package management

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Basic Validation Example

```python
from src.b05_system_validation import SystemValidator
from src.models.validation_models import SystemConfiguration, SystemType

# Define system configuration
config = SystemConfiguration(
    system_type=SystemType.COMMERCIAL,
    system_name="Downtown Solar Array",
    location="San Francisco, CA",
    capacity_kw=100.0,
    module_count=250,
    string_count=20,
    modules_per_string=12,
    system_voltage_vdc=600.0,
    max_voltage_voc=800.0,
    operating_voltage_vmp=650.0,
    max_current_isc=10.0,
    operating_current_imp=9.5,
    ambient_temp_min=-10.0,
    ambient_temp_max=45.0,
    wind_speed_max=40.0,
    jurisdiction="San Francisco",
    applicable_codes=["NEC 2020", "IEC 60364"],
)

# Run validation
validator = SystemValidator(config)
report = validator.validate_complete_design()

# Print results
print(f"Overall Status: {report.overall_status}")
print(f"Total Issues: {report.total_issues}")
print(f"Compliance Rate: {len([c for c in report.code_compliance if c.status.value == 'passed'])}/{len(report.code_compliance)}")
```

### Streamlit UI

```bash
# Launch the Streamlit UI
streamlit run ui/system_validation_ui.py
```

Navigate to `http://localhost:8501` to access the interactive dashboard.

## Usage Examples

### Complete Validation with Performance Analysis

```python
from src.b05_system_validation import SystemValidator
from src.models.validation_models import PerformanceMetrics, SystemConfiguration, SystemType

# System configuration
config = SystemConfiguration(
    system_type=SystemType.UTILITY_SCALE,
    system_name="Utility Solar Farm",
    location="Phoenix, AZ",
    capacity_kw=5000.0,
    module_count=12500,
    string_count=500,
    modules_per_string=25,
    # ... other parameters
)

# Performance metrics
metrics = PerformanceMetrics(
    annual_energy_yield_kwh=9500000.0,
    specific_yield_kwh_kwp=1900.0,
    performance_ratio=0.85,
    capacity_factor=0.25,
    loss_temperature=6.0,
    loss_soiling=3.0,
    loss_shading=0.5,
    loss_mismatch=2.0,
    loss_wiring=1.5,
    loss_inverter=2.5,
    loss_degradation=0.5,
    total_losses=16.0,
    is_energy_yield_realistic=True,
    is_pr_in_range=True,
    is_loss_budget_valid=True,
)

# Validate
validator = SystemValidator(config, metrics)
report = validator.validate_complete_design()
```

### Generate Documentation Package

```python
from src.b05_system_validation.documentation_generator import DocumentationGenerator

# Create documentation generator
doc_gen = DocumentationGenerator(
    config=config,
    validation_report=report,
    output_dir="./exports",
    include_pe_stamp=True
)

# Generate complete package
package = doc_gen.generate_complete_package()

print(f"Package ID: {package.package_id}")
print(f"Documents: {package.document_count}")
print(f"Size: {package.total_size_mb} MB")
```

### Individual Compliance Checks

```python
from src.b05_system_validation import CodeComplianceChecker

checker = CodeComplianceChecker(config)

# NEC compliance
nec_results = checker.nec_690_compliance()
for result in nec_results:
    print(f"{result.section}: {result.status.value}")

# IEC compliance
iec_results = checker.iec_60364_compliance()

# Building codes
building_results = checker.building_code_checks()

# Fire safety
fire_results = checker.fire_safety_compliance()
```

### Engineering Calculations

```python
from src.b05_system_validation import EngineeringCalculationVerifier

verifier = EngineeringCalculationVerifier(config)

# String calculations
string_calcs = verifier.verify_string_calculations()

# Voltage drop
vdrop = verifier.check_voltage_drop(
    current=50.0,
    distance=30.0,
    wire_gauge="10AWG"
)

# Short circuit
isc = verifier.validate_short_circuit(
    parallel_strings=20,
    string_isc=10.0
)

# Grounding
grounding = verifier.verify_grounding()

# OCPD sizing
ocpd = verifier.confirm_overcurrent_protection(
    continuous_current=50.0
)
```

## API Reference

### SystemConfiguration

```python
SystemConfiguration(
    system_type: SystemType,           # Type of PV system
    system_name: str,                  # System identifier
    location: str,                     # Installation location
    capacity_kw: float,                # System capacity (kW)
    module_count: int,                 # Total modules
    string_count: int,                 # Number of strings
    modules_per_string: int,           # Modules per string
    system_voltage_vdc: float,         # DC system voltage
    max_voltage_voc: float,            # Maximum Voc
    operating_voltage_vmp: float,      # Operating Vmp
    max_current_isc: float,            # Maximum Isc
    operating_current_imp: float,      # Operating Imp
    ambient_temp_min: float,           # Min temperature (°C)
    ambient_temp_max: float,           # Max temperature (°C)
    wind_speed_max: float,             # Max wind speed (m/s)
    jurisdiction: str,                 # Local jurisdiction
    applicable_codes: List[str],       # Applicable codes
)
```

### ValidationReport

The validation report contains:
- `report_id`: Unique identifier
- `overall_status`: PASSED / WARNING / FAILED
- `electrical_validation`: Electrical validation results
- `structural_validation`: Structural validation results
- `performance_validation`: Performance validation results
- `code_compliance`: All compliance check results
- `total_issues`: Count of all issues
- `critical_issues`: Count of critical issues
- `recommendations`: List of recommendations

## Compliance Standards

### Supported Codes

- **NEC 2020** - National Electrical Code Article 690
  - Voltage limits (690.7)
  - Circuit sizing (690.8)
  - Overcurrent protection (690.9)
  - Rapid shutdown (690.12)
  - Disconnecting means (690.13)
  - Wiring methods (690.31)
  - Grounding (690.35)

- **IEC 60364-7-712** - Solar PV power supply systems
  - Protection against electric shock (712.410.3)
  - Overcurrent protection (712.433)
  - Overvoltage protection (712.444)
  - Equipment selection (712.5)

- **IBC 2021** - International Building Code
  - Wind loads (1609)
  - Snow loads (1608)
  - Roof loading (1607)

- **IFC 2021** - International Fire Code
  - Roof access pathways (605.11.3.3)
  - Rapid shutdown for firefighter safety (1204.4)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=ui --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
```

## Code Quality

```bash
# Format code
black src/ ui/ tests/

# Lint code
ruff src/ ui/ tests/

# Type checking
mypy src/ ui/
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── models/
│   │   └── validation_models.py      # Pydantic data models
│   └── b05_system_validation/
│       ├── system_validator.py        # Main validator
│       ├── code_compliance_checker.py # Compliance checking
│       ├── engineering_calculation_verifier.py # Calculations
│       ├── performance_validator.py   # Performance validation
│       └── documentation_generator.py # Documentation generation
├── ui/
│   └── system_validation_ui.py       # Streamlit dashboard
├── tests/
│   └── test_validation.py            # Test suite
├── examples/
│   └── basic_validation_example.py   # Usage examples
├── requirements.txt                   # Dependencies
└── pyproject.toml                     # Project configuration
```

## Contributing

Production-ready code with:
- Full type hints (mypy compliant)
- Comprehensive docstrings (Google style)
- Pydantic models for data validation
- Unit tests with pytest
- Code formatting with black
- Linting with ruff

## License

MIT License - See LICENSE file for details.

## Support

For issues, questions, or contributions, please refer to the main project repository.
