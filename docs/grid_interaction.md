# GridInteraction Component Documentation

## Overview

The `GridInteraction` component provides comprehensive grid integration capabilities for photovoltaic (PV) systems within the PV Circularity Simulator. It implements production-ready functionality for:

- **Grid Code Compliance**: Monitoring and enforcement of international grid standards
- **Reactive Power Control**: Multiple control strategies for voltage support
- **Frequency Regulation**: Droop-based frequency support
- **Smart Grid Communication**: SCADA integration with multiple industrial protocols

## Table of Contents

1. [Features](#features)
2. [Supported Grid Codes](#supported-grid-codes)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Components](#core-components)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [Testing](#testing)
10. [Best Practices](#best-practices)

---

## Features

### Grid Code Compliance

Validates PV system operation against major international grid codes:
- Voltage limits (under/over voltage protection)
- Frequency limits (under/over frequency protection)
- Power quality metrics (THD, power factor)
- Reactive power capability verification

### Reactive Power Control

Multiple control strategies:
- **Fixed Power Factor**: Constant power factor operation
- **Fixed Q**: Constant reactive power injection/absorption
- **Volt-VAR**: Voltage-dependent reactive power (IEEE 1547-2018)
- **Volt-Watt**: Voltage-dependent active power curtailment

### Frequency Regulation

- Frequency-watt droop control
- Configurable deadband and droop coefficient
- Ramp rate limiting for smooth transitions
- Support for both over-frequency and under-frequency response

### Smart Grid Communication

Support for industrial SCADA protocols:
- **Modbus TCP/RTU**: Industrial standard
- **DNP3**: Utility-grade SCADA
- **IEC 61850**: Substation automation
- **OPC UA**: Industry 4.0 standard
- **MQTT**: IoT and cloud integration
- **SunSpec**: PV-specific Modbus extension

---

## Supported Grid Codes

| Standard | Region | Voltage Range | Frequency Range | Notes |
|----------|--------|---------------|-----------------|-------|
| IEEE 1547 | North America | 0.88-1.10 p.u. | 59.3-60.5 Hz | Latest 2018 revision |
| VDE-AR-N 4105 | Germany | 0.90-1.10 p.u. | 47.5-51.5 Hz | Medium voltage |
| G99 | United Kingdom | 0.90-1.10 p.u. | 47.0-52.0 Hz | Replaces G83/G59 |
| IEC 61727 | International | 0.85-1.10 p.u. | 47.0-53.0 Hz | General standard |
| NRS 097-2-1 | South Africa | 0.88-1.10 p.u. | 47.0-52.0 Hz | Grid connection |
| AS 4777 | Australia | 0.85-1.15 p.u. | 47.0-52.0 Hz | Partial support |

---

## Architecture

### Component Structure

```
pv_circularity_simulator/grid/
├── __init__.py
└── grid_interaction.py        # Main GridInteraction class

Data Flow:
┌─────────────┐
│ PV System   │
│ Measurements│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     GridInteraction Component       │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Grid Code Compliance Check  │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │  Reactive Power Control      │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │  Frequency Regulation        │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │  SCADA Communication         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Control    │
│  Setpoints  │
└─────────────┘
```

### Key Classes

- **`GridInteraction`**: Main class for grid interaction
- **`GridState`**: Current grid measurements (Pydantic model)
- **`GridCodeLimits`**: Grid code compliance limits (Pydantic model)
- **`ReactivePowerControlConfig`**: Reactive power control configuration
- **`FrequencyRegulationConfig`**: Frequency regulation configuration
- **`SCADAConfig`**: SCADA communication configuration

---

## Installation

### Requirements

- Python >= 3.9
- pydantic >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0 (optional, for data analysis)
- scipy >= 1.10.0 (optional, for advanced features)

### Install Package

```bash
# From source
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Basic Example

```python
from pv_circularity_simulator.grid.grid_interaction import (
    GridInteraction,
    GridCodeStandard,
    GridState,
)

# Initialize grid interaction system
grid = GridInteraction(
    grid_code_standard=GridCodeStandard.IEEE_1547,
    nominal_voltage=230.0,  # Volts
    rated_power=5000.0,     # Watts
)

# Create grid state measurement
grid_state = GridState(
    voltage_l1=235.0,
    current_l1=10.5,
    frequency=60.02,
    active_power=2400.0,
    reactive_power=100.0,
)

# Check grid code compliance
result = grid.grid_code_compliance(grid_state)
print(f"Compliant: {result.compliant}")

# Calculate reactive power setpoint
q_setpoint = grid.reactive_power_control(
    grid_state=grid_state,
    pv_power=2400.0
)
print(f"Reactive Power Setpoint: {q_setpoint:.2f} VAR")

# Calculate active power setpoint (frequency regulation)
p_setpoint = grid.frequency_regulation(
    grid_state=grid_state,
    available_power=2400.0
)
print(f"Active Power Setpoint: {p_setpoint:.2f} W")
```

---

## Core Components

### 1. Grid Code Compliance

The `grid_code_compliance()` method validates grid conditions against configured standards.

**Checked Parameters:**
- Voltage (over/under voltage)
- Frequency (over/under frequency)
- Voltage THD (Total Harmonic Distortion)
- Current THD
- Power factor
- Reactive power capability

**Example:**

```python
grid_state = GridState(
    voltage_l1=260.0,  # High voltage!
    current_l1=10.0,
    frequency=60.0,
    active_power=2300.0,
    reactive_power=0.0,
)

result = grid.grid_code_compliance(grid_state)

if not result.compliant:
    print("Violations detected:")
    for violation in result.violations:
        print(f"  - {violation}")
```

### 2. Reactive Power Control

The `reactive_power_control()` method calculates reactive power setpoints based on control strategy.

#### Fixed Power Factor Mode

```python
from pv_circularity_simulator.grid.grid_interaction import ReactivePowerControlConfig

config = ReactivePowerControlConfig(
    mode="fixed_pf",
    target_power_factor=0.95
)

grid = GridInteraction(
    grid_code_standard=GridCodeStandard.IEEE_1547,
    nominal_voltage=230.0,
    rated_power=5000.0,
    reactive_power_config=config,
)

q_setpoint = grid.reactive_power_control(grid_state, pv_power=2400.0)
```

#### Volt-VAR Mode (IEEE 1547-2018)

```python
config = ReactivePowerControlConfig(
    mode="volt_var",
    volt_var_v1=0.92,   # Low voltage threshold
    volt_var_v2=0.98,   # Low voltage deadband
    volt_var_v3=1.02,   # High voltage deadband
    volt_var_v4=1.08,   # High voltage threshold
    volt_var_q1=0.44,   # Max capacitive Q
    volt_var_q2=0.0,    # No Q in deadband
    volt_var_q3=0.0,    # No Q in deadband
    volt_var_q4=-0.44,  # Max inductive Q
)
```

**Volt-VAR Curve:**

```
Q (p.u.)
  │
  │  0.44├─────┐
  │      │     │
  │      │     │
  │  0.0 ├────────┼────────┤
  │      │     │  │     │
  │      │     │  │     │
-0.44├─────────────────┘
  │
  └────┼────┼────┼────┼──── V (p.u.)
     0.92 0.98 1.02 1.08
```

### 3. Frequency Regulation

The `frequency_regulation()` method implements frequency-watt droop control.

**Droop Equation:**

```
ΔP = -droop × Δf × P_rated

where:
  ΔP = Power adjustment (W)
  Δf = Frequency deviation from nominal (Hz)
  droop = Droop coefficient (p.u./Hz)
  P_rated = Rated power (W)
```

**Example:**

```python
from pv_circularity_simulator.grid.grid_interaction import FrequencyRegulationConfig

config = FrequencyRegulationConfig(
    enabled=True,
    droop=0.05,              # 5% droop
    deadband=0.036,          # ±0.036 Hz
    nominal_frequency=60.0,  # 60 Hz
    max_power_ramp_rate=0.2, # 20% per second
)

grid = GridInteraction(
    grid_code_standard=GridCodeStandard.IEEE_1547,
    nominal_voltage=230.0,
    rated_power=5000.0,
    frequency_regulation_config=config,
)

p_setpoint = grid.frequency_regulation(grid_state, available_power=3000.0)
```

### 4. SCADA Communication

The `smart_grid_communication()` method handles telemetry and control data exchange.

**Supported Protocols:**
- Modbus TCP/RTU
- DNP3
- IEC 61850
- OPC UA
- MQTT
- SunSpec Modbus

**Example:**

```python
from pv_circularity_simulator.grid.grid_interaction import (
    SCADAConfig,
    SCADAProtocol,
)

scada_config = SCADAConfig(
    protocol=SCADAProtocol.MODBUS_TCP,
    host="192.168.1.100",
    port=502,
    device_id="PV_INV_001",
    polling_interval=1.0,
    timeout=5.0,
)

grid = GridInteraction(
    grid_code_standard=GridCodeStandard.IEEE_1547,
    nominal_voltage=230.0,
    rated_power=5000.0,
    scada_config=scada_config,
)

# Send telemetry data
result = grid.smart_grid_communication({
    "telemetry": {
        "active_power": 2400.0,
        "reactive_power": 100.0,
        "voltage": 235.0,
        "frequency": 60.02,
    }
})

if result["success"]:
    print("SCADA communication successful")
    print(f"Received commands: {result['received_data']}")
```

---

## Usage Examples

### Example 1: Complete Grid Integration Workflow

See [examples/grid_integration_example.py](../examples/grid_integration_example.py) for a complete workflow demonstration.

### Example 2: Voltage Support with Volt-VAR

```python
# Configure volt-var control
config = ReactivePowerControlConfig(mode="volt_var")

grid = GridInteraction(
    grid_code_standard=GridCodeStandard.IEEE_1547,
    nominal_voltage=230.0,
    rated_power=10000.0,  # 10 kW
    reactive_power_config=config,
)

# Simulate voltage sag scenario
grid_state = GridState(
    voltage_l1=210.0,  # 0.913 p.u. - low voltage
    current_l1=20.0,
    frequency=60.0,
    active_power=4200.0,
    reactive_power=0.0,
)

# Calculate Q setpoint to support voltage
q_setpoint = grid.reactive_power_control(grid_state, pv_power=4200.0)
print(f"Voltage support: Inject {q_setpoint:.2f} VAR (capacitive)")
```

### Example 3: Frequency Support During Disturbance

```python
# Configure frequency regulation
config = FrequencyRegulationConfig(
    enabled=True,
    droop=0.04,  # 4% droop (more aggressive)
    deadband=0.02,
    nominal_frequency=50.0,  # 50 Hz system
)

grid = GridInteraction(
    grid_code_standard=GridCodeStandard.VDE_AR_N_4105,
    nominal_voltage=400.0,  # 400V three-phase
    rated_power=100000.0,   # 100 kW
    frequency_regulation_config=config,
)

# Simulate over-frequency event
grid_state = GridState(
    voltage_l1=400.0,
    current_l1=50.0,
    frequency=50.4,  # High frequency - excess generation
    active_power=20000.0,
    reactive_power=0.0,
)

# Calculate P setpoint to reduce frequency
p_setpoint = grid.frequency_regulation(grid_state, available_power=100000.0)
curtailment = 100000.0 - p_setpoint
print(f"Frequency support: Curtail {curtailment:.2f} W ({curtailment/1000:.1f}%)")
```

---

## API Reference

### GridInteraction Class

#### Constructor

```python
GridInteraction(
    grid_code_standard: GridCodeStandard = GridCodeStandard.IEEE_1547,
    nominal_voltage: float = 230.0,
    rated_power: float = 5000.0,
    reactive_power_config: Optional[ReactivePowerControlConfig] = None,
    frequency_regulation_config: Optional[FrequencyRegulationConfig] = None,
    scada_config: Optional[SCADAConfig] = None,
)
```

#### Methods

##### `grid_code_compliance(grid_state: GridState) -> ComplianceCheckResult`

Check grid code compliance for current grid state.

**Returns:** `ComplianceCheckResult` with violations and warnings

##### `reactive_power_control(grid_state: GridState, pv_power: float) -> float`

Calculate reactive power setpoint based on control strategy.

**Returns:** Reactive power setpoint (VAR)

##### `frequency_regulation(grid_state: GridState, available_power: float) -> float`

Calculate active power setpoint for frequency regulation.

**Returns:** Active power setpoint (W)

##### `smart_grid_communication(data_points: Dict[str, Any]) -> Dict[str, Any]`

Communicate with SCADA system.

**Returns:** Communication result with status and data

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest

# Run grid interaction tests only
pytest tests/unit/test_grid/test_grid_interaction.py

# Run with coverage
pytest --cov=src/pv_circularity_simulator/grid

# Run integration tests
pytest -m integration
```

### Test Coverage

The test suite includes:
- ✓ Pydantic model validation tests
- ✓ Grid code compliance tests (all standards)
- ✓ Reactive power control tests (all modes)
- ✓ Frequency regulation tests
- ✓ SCADA communication tests (all protocols)
- ✓ Integration tests

**Coverage target:** > 90%

---

## Best Practices

### 1. Grid Code Selection

Choose the appropriate grid code for your region:

```python
# North America
grid = GridInteraction(grid_code_standard=GridCodeStandard.IEEE_1547, ...)

# Germany
grid = GridInteraction(grid_code_standard=GridCodeStandard.VDE_AR_N_4105, ...)

# UK
grid = GridInteraction(grid_code_standard=GridCodeStandard.G99, ...)
```

### 2. Error Handling

Always handle potential violations:

```python
result = grid.grid_code_compliance(grid_state)

if not result.compliant:
    # Trigger fault response
    for violation in result.violations:
        logger.error(f"Grid code violation: {violation}")
    # Disconnect if required
    disconnect_inverter()
```

### 3. Reactive Power Coordination

Use volt-var for modern grid support:

```python
# IEEE 1547-2018 compliant volt-var
config = ReactivePowerControlConfig(
    mode="volt_var",
    # Use default curve or customize
)
```

### 4. Frequency Response Tuning

Tune droop and deadband based on grid requirements:

```python
# Conservative settings for weak grids
config = FrequencyRegulationConfig(
    droop=0.05,      # Higher droop = less aggressive
    deadband=0.036,  # Standard deadband
)

# Aggressive settings for strong grids
config = FrequencyRegulationConfig(
    droop=0.02,      # Lower droop = more aggressive
    deadband=0.02,   # Tighter deadband
)
```

### 5. SCADA Security

Enable encryption for production systems:

```python
scada_config = SCADAConfig(
    protocol=SCADAProtocol.OPC_UA,  # Modern protocol
    host="scada.example.com",
    port=4840,
    device_id="PV_INV_001",
    enable_encryption=True,  # Always enable in production
)
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Compliance Violations

**Problem:** Frequent grid code violations

**Solution:**
- Check measurement accuracy
- Verify grid code standard matches local requirements
- Adjust control parameters

#### Issue 2: Reactive Power Oscillations

**Problem:** Q setpoint oscillates

**Solution:**
- Widen volt-var deadband (V2-V3 range)
- Add filtering to voltage measurements
- Reduce control loop update rate

#### Issue 3: SCADA Communication Failures

**Problem:** SCADA connection drops

**Solution:**
- Verify network connectivity
- Check firewall settings
- Increase timeout and retry attempts
- Validate protocol configuration

---

## References

### Standards

- IEEE Std 1547-2018: Standard for Interconnection and Interoperability of Distributed Energy Resources
- VDE-AR-N 4105: Technical Requirements for the Connection and Operation of Customer Installations
- G99: Requirements for the connection of generation equipment
- IEC 61727: Photovoltaic (PV) systems - Characteristics of the utility interface

### Further Reading

- [IEEE 1547 Overview](https://standards.ieee.org/standard/1547-2018.html)
- [Smart Inverter Functions](https://www.nrel.gov/grid/solar-integration-data.html)
- [SCADA Protocols Guide](https://en.wikipedia.org/wiki/SCADA)

---

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- GitHub Issues: [pv-circularity-simulator/issues](https://github.com/your-org/pv-circularity-simulator/issues)
- Email: support@pv-circularity-simulator.org
