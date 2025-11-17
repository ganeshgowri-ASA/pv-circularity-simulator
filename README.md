# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Griddler Pro Integration & Metallization Optimization

Advanced metallization pattern design and optimization for photovoltaic solar cells.

**Key Features:**
- Multi-objective optimization (resistance, shading, cost, fill factor)
- Support for advanced patterns (MBB, IBC, bifacial, shingled, SmartWire)
- Comprehensive cost analysis (silver consumption, processing costs)
- Series resistance calculation with detailed component breakdown
- CAD export (JSON, SVG, DXF, GDSII)
- Module-level impact analysis

**Quick Start:**
```python
from src.modules.griddler_integration import GriddlerInterface, OptimizationObjective

griddler = GriddlerInterface()
optimized = griddler.optimize_metallization(
    {'cell_width': 156.75, 'cell_length': 156.75, 'jsc': 0.042, 'voc': 0.68},
    objective=OptimizationObjective.BALANCED
)
print(f"Efficiency: {optimized.combined_efficiency:.2%}")
```

See [docs/GRIDDLER_INTEGRATION.md](docs/GRIDDLER_INTEGRATION.md) for full documentation.

## Installation

```bash
pip install -r requirements.txt
```

## Examples

Run the Griddler integration examples:
```bash
python examples/griddler_example.py
```

## Documentation

- [Griddler Integration Guide](docs/GRIDDLER_INTEGRATION.md)

## License

See [LICENSE](LICENSE) for details.
