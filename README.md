# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Overview

The PV Circularity Simulator is a comprehensive platform for modeling and optimizing the complete lifecycle of photovoltaic systems, from initial design through end-of-life circularity strategies.

## Features

### RepairOptimizer - Intelligent Maintenance Planning

The RepairOptimizer module provides production-ready tools for managing PV system maintenance and repairs:

#### Core Capabilities

- **Fault Diagnosis**: Automated detection and classification of component faults
  - Rule-based and statistical analysis
  - Multiple fault types: electrical, thermal, mechanical, degradation
  - Severity assessment with confidence scoring
  - Root cause identification

- **Repair Cost Estimation**: Detailed cost breakdowns for repairs
  - Labor cost calculation with customizable rates
  - Parts cost estimation with inventory integration
  - Overhead cost modeling
  - Rush service options

- **Maintenance Scheduling**: Optimized task scheduling
  - Multiple optimization objectives (cost, time, priority)
  - Resource constraint management
  - Technician skill matching
  - Spare parts availability checking

- **Spare Parts Management**: Inventory optimization
  - Real-time inventory tracking
  - Automatic reorder point detection
  - Demand forecasting
  - Economic order quantity calculations

## Installation

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from datetime import datetime, timedelta
from pv_simulator import RepairOptimizer
from pv_simulator.models.maintenance import (
    MaintenancePriority,
    MaintenanceType,
    RepairTask,
    SparePart
)

# Initialize the optimizer
optimizer = RepairOptimizer(
    labor_rate=75.0,
    overhead_rate=0.15,
    fault_detection_threshold=0.6
)

# Diagnose a fault
fault = optimizer.fault_diagnosis(
    component_id="PANEL-A23",
    component_type="panel",
    performance_data={
        "voltage": 25.0,
        "current": 7.0,
        "efficiency": 0.15,
        "temperature": 48.0
    },
    baseline_data={
        "voltage": 30.0,
        "current": 8.0,
        "efficiency": 0.18,
        "temperature": 45.0
    }
)

# Get cost estimate
if fault:
    estimate = optimizer.repair_cost_estimation(fault)
    print(f"Estimated repair cost: ${estimate.total_cost:.2f}")

    # Create repair task
    task = RepairTask(
        fault_id=fault.fault_id,
        component_id=fault.component_id,
        task_type=MaintenanceType.CORRECTIVE,
        priority=MaintenancePriority.HIGH,
        estimated_duration_hours=estimate.labor_hours,
        estimated_cost=estimate.total_cost
    )

    # Schedule maintenance
    schedule = optimizer.maintenance_scheduling(
        tasks=[task],
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14)
    )

# Manage spare parts
optimizer.add_spare_part(SparePart(
    part_id="BP-001",
    part_name="Bypass Diode",
    part_number="BD-12V-10A",
    category="electrical",
    quantity_available=100,
    unit_cost=15.0,
    lead_time_days=7,
    reorder_point=30,
    reorder_quantity=100,
    supplier="ElectroSupply Inc"
))

parts_report = optimizer.spare_parts_management(forecast_days=90)
print(f"Parts needing reorder: {len(parts_report['reorder_recommendations'])}")
```

### Running the Example

See the [examples](examples/) directory for a complete workflow demonstration:

```bash
python examples/repair_optimizer_example.py
```

## Testing

The project includes comprehensive unit and integration tests.

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=pv_simulator --cov-report=html
```

### Run Specific Test Files

```bash
# Unit tests
pytest tests/unit/test_repair_optimizer.py -v

# Integration tests
pytest tests/integration/test_repair_optimizer_integration.py -v
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_simulator/
│       ├── models/          # Pydantic data models
│       │   └── maintenance.py
│       ├── managers/        # Business logic classes
│       │   └── repair_optimizer.py
│       └── core/           # Core utilities
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example scripts
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies
└── README.md
```

## Architecture

### Data Models (Pydantic)

All data structures use Pydantic for validation and type safety:

- `Fault`: Diagnosed component faults
- `RepairTask`: Maintenance tasks
- `RepairCostEstimate`: Cost breakdowns
- `MaintenanceSchedule`: Optimized schedules
- `SparePart`: Inventory items
- `ComponentHealth`: Component status tracking

### Optimization Algorithms

The RepairOptimizer uses various optimization techniques:

- **Scheduling**: Greedy algorithms with priority-based sorting
- **Cost Optimization**: Multi-objective optimization
- **Inventory Management**: Economic Order Quantity (EOQ) principles
- **Forecasting**: Linear regression for degradation analysis

## Configuration

### Optimizer Parameters

- `labor_rate`: Hourly cost for maintenance labor (default: $75.00)
- `overhead_rate`: Overhead multiplier (default: 0.15 or 15%)
- `fault_detection_threshold`: Minimum confidence for fault detection (default: 0.6)

### Fault Detection Thresholds

Configurable thresholds for different fault indicators:
- Efficiency drop: 10%
- High temperature: 70°C
- Voltage deviation: 5%
- Current imbalance: 10%
- High degradation rate: 2% per year

## API Documentation

### RepairOptimizer Class

#### Methods

- `fault_diagnosis()`: Diagnose component faults from performance data
- `repair_cost_estimation()`: Estimate repair costs with detailed breakdown
- `maintenance_scheduling()`: Create optimized maintenance schedules
- `spare_parts_management()`: Manage inventory and generate reorder lists
- `add_spare_part()`: Add parts to inventory
- `update_component_health()`: Update component health status
- `get_active_faults()`: Retrieve current active faults
- `clear_fault()`: Mark faults as resolved

See inline docstrings for detailed parameter descriptions.

## Contributing

Contributions are welcome! Please ensure:

1. All new code includes comprehensive docstrings
2. Unit tests cover new functionality
3. Code passes linting (black, isort, ruff)
4. Type hints are provided for all functions

## License

MIT License - see LICENSE file for details.

## Version

Current version: 0.1.0

## Contact

For questions or issues, please open an issue on GitHub.
