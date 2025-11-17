# pv-circularity-simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Overview

The PV Circularity Simulator is a comprehensive platform for modeling and analyzing the complete lifecycle of photovoltaic systems, from manufacturing through operation to end-of-life circular economy considerations.

## Features

### Utility Library (`src/pv_simulator/utils/`)

The project includes a comprehensive utility library with the following modules:

#### 1. Unit Conversions (`unit_conversions.py`)
- **Energy conversions**: Wh, kWh, MWh, GWh, J, kJ, MJ, GJ
- **Power conversions**: W, kW, MW, GW
- **Area conversions**: m², cm², mm², km², hectares
- **Mass conversions**: kg, g, mg, ton, lb, oz
- **Length conversions**: m, cm, mm, km, inch, ft, yard, mile
- **Temperature conversions**: Celsius, Fahrenheit, Kelvin
- **Efficiency conversions**: decimal, percent, ppm
- **Specialized PV calculations**: Energy from power, specific yield

#### 2. Data Validation (`data_validation.py`)
- **Pydantic-based validators** for type safety and data integrity
- **Basic validators**: positive, non-negative, percentage, efficiency, range
- **PV-specific models**: `PVModuleSpecs`, `EnergyProductionData`, `MaterialComposition`
- **Batch validation** and safe validation utilities
- **Email, date range, and list/dict validators**

#### 3. File I/O (`file_io.py`)
- **Multi-format support**: JSON, YAML, CSV
- **Pandas integration** for efficient data processing
- **Auto-detection** of file format from extension
- **Utility functions**: file existence, directory creation, file size, backup
- **File listing** with glob pattern support

#### 4. Calculation Helpers (`calculations.py`)
- **Statistical functions**: mean, median, standard deviation, variance, percentiles, weighted average
- **Financial calculations**: NPV, IRR, payback period, LCOE
- **PV technical calculations**: panel efficiency, temperature derating, performance ratio, capacity factor, degradation
- **Circular economy**: material recovery rate, CE score, carbon footprint reduction
- **Math utilities**: clamp, interpolation, significant figures

#### 5. Formatting Functions (`formatting.py`)
- **Number formatting**: decimals, percentages, currency, scientific notation, SI units
- **Date/time formatting**: dates, timestamps, durations
- **Data structure formatting**: tables, lists, key-value pairs
- **Report generation**: headers, sections, summary boxes, progress bars
- **String utilities**: truncation, compact notation

## Installation

### Using pip

```bash
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Usage

### Unit Conversions

```python
from pv_simulator.utils.unit_conversions import convert_energy, convert_power

# Convert energy units
energy_kwh = convert_energy(5000, "Wh", "kWh")  # 5.0 kWh

# Convert power units
power_kw = convert_power(2000, "W", "kW")  # 2.0 kW

# Calculate specific yield
from pv_simulator.utils.unit_conversions import calculate_specific_yield
yield_kwh_kwp = calculate_specific_yield(1000, "kWh", 5, "kW")  # 200.0
```

### Data Validation

```python
from pv_simulator.utils.data_validation import PVModuleSpecs, validate_positive

# Validate PV module specifications
module = PVModuleSpecs(
    name="Solar Panel 300W",
    power_rating_w=300,
    efficiency=0.18,
    area_m2=1.67,
    voltage_voc=45.0,
    current_isc=9.5,
    temperature_coeff_power=-0.4,
    warranty_years=25
)

# Simple validation
value = validate_positive(5.0)  # Returns 5.0
```

### File I/O

```python
from pv_simulator.utils.file_io import load_data, save_data

# Auto-detect format from extension
config = load_data("config.yaml")
data = load_data("measurements.csv")

# Save data
save_data({"key": "value"}, "output.json")
save_data([{"a": 1}, {"a": 2}], "data.csv")
```

### Calculations

```python
from pv_simulator.utils.calculations import (
    calculate_performance_ratio,
    calculate_lcoe,
    calculate_material_recovery_rate
)

# Performance ratio
pr = calculate_performance_ratio(8500, 10000)  # 0.85

# Levelized Cost of Energy
lcoe = calculate_lcoe(10000, 50000, 0.05, 25)

# Material recovery rate for circular economy
recovery = calculate_material_recovery_rate(85, 100)  # 0.85
```

### Formatting

```python
from pv_simulator.utils.formatting import (
    format_currency,
    format_si_unit,
    format_table,
    format_summary_box
)

# Format numbers
cost = format_currency(1234.56)  # "$1,234.56"
power = format_si_unit(5000, "W")  # "5.00 kW"

# Format data as table
data = [
    {"module": "A", "power": 300, "efficiency": 18.5},
    {"module": "B", "power": 350, "efficiency": 20.0}
]
print(format_table(data))

# Create summary box
metrics = {"Total Energy": "1000 kWh", "Efficiency": "18.5%"}
print(format_summary_box("System Metrics", metrics))
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/pv_simulator --cov-report=html
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_simulator/
│       ├── __init__.py
│       └── utils/
│           ├── __init__.py
│           ├── unit_conversions.py
│           ├── data_validation.py
│           ├── file_io.py
│           ├── calculations.py
│           └── formatting.py
├── tests/
│   └── utils/
│       ├── __init__.py
│       ├── test_unit_conversions.py
│       ├── test_data_validation.py
│       ├── test_file_io.py
│       ├── test_calculations.py
│       └── test_formatting.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

## Documentation

All functions include comprehensive docstrings with:
- Parameter descriptions and types
- Return value descriptions
- Raised exceptions
- Usage examples
- Type hints for IDE support

## Requirements

- Python >= 3.9
- pydantic >= 2.0.0
- pyyaml >= 6.0
- numpy >= 1.24.0
- pandas >= 2.0.0

## Development

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

### Running Quality Checks

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass
2. Code is formatted with Black
3. Type hints are included
4. Docstrings are comprehensive
5. New functionality includes tests

## License

MIT License - See LICENSE file for details

## Author

PV Simulator Team

## Version

0.1.0 - Initial release with comprehensive utility library
