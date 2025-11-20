# Examples

This directory contains example scripts demonstrating how to use the PV Circularity Simulator components.

## Repair Optimizer Example

The `repair_optimizer_example.py` script demonstrates a complete workflow for PV system maintenance and repair optimization.

### Running the Example

```bash
# From the project root directory
python examples/repair_optimizer_example.py
```

### What the Example Demonstrates

1. **Initialization**: Setting up the RepairOptimizer with custom parameters
2. **Inventory Setup**: Adding spare parts to the inventory system
3. **Fault Diagnosis**: Detecting and diagnosing various types of faults:
   - Electrical faults (voltage/current deviations)
   - Thermal faults (overheating)
   - Degradation faults (performance decline over time)
4. **Cost Estimation**: Calculating detailed repair costs including labor, parts, and overhead
5. **Task Creation**: Converting diagnosed faults into actionable repair tasks
6. **Schedule Optimization**: Creating an optimized maintenance schedule based on priorities
7. **Inventory Management**: Checking stock levels and generating reorder recommendations

### Sample Output

The script provides detailed output showing:
- Detected faults with confidence levels and root causes
- Detailed cost breakdowns for each repair
- Optimized maintenance schedule with task priorities
- Spare parts inventory status and reorder recommendations
- Complete summary of the maintenance workflow

### Customization

You can modify the example to test different scenarios:

- Adjust `labor_rate` and `overhead_rate` to match your cost structure
- Add different types of components and faults
- Experiment with different optimization objectives:
  - `minimize_cost`: Optimize for lowest total cost
  - `minimize_time`: Complete tasks as quickly as possible
  - `maximize_priority`: Handle critical tasks first
- Change the scheduling constraints (daily hours, date ranges)
- Modify spare parts inventory levels to test reordering logic

### Requirements

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```
