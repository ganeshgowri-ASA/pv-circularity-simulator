# pv-circularity-simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Bifacial Module Modeling (`src/modules/bifacial_model.py`)

Comprehensive bifacial PV module modeling and simulation system featuring:

- **Backside Irradiance Calculation**: Multiple view factor models (Perez, Durusoy, Simple)
- **Bifacial Gain Analysis**: Ground albedo effects, mounting structure optimization
- **View Factor Modeling**: Row-to-row shading, edge effects, inter-row reflections
- **Advanced Loss Mechanisms**: Mismatch, temperature, soiling impacts
- **Performance Simulation**: Time-series analysis with TMY data
- **Row Spacing Optimization**: Maximize energy yield per land area
- **Multiple Mounting Types**: Fixed tilt, single-axis tracker, dual-axis, vertical, east-west

**Ground Albedo Support**:
- Grass (0.20), Concrete (0.30), White membrane (0.70), Sand (0.40), Snow (0.80)
- Seasonal variation modeling
- Custom albedo values

**Module Bifaciality**: 0.65-0.95 (typical n-type and p-type modules)

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example analysis
python examples/bifacial_analysis_example.py

# Run tests
pytest tests/test_bifacial_model.py -v
```

### Example Usage

```python
from src.modules.bifacial_model import (
    BifacialModuleModel,
    BifacialSystemConfig,
    BifacialModuleParams,
    MountingStructure,
    GroundSurface,
    AlbedoType
)

# Configure system
config = BifacialSystemConfig(
    module=BifacialModuleParams(bifaciality=0.70, front_efficiency=0.21),
    structure=MountingStructure(
        mounting_type="fixed_tilt",
        tilt=30.0,
        clearance_height=1.0,
        row_spacing=4.0,
        row_width=1.1,
        n_rows=10
    ),
    ground=GroundSurface(albedo_type=AlbedoType.WHITE_MEMBRANE),
    location_latitude=35.0,
    location_longitude=-106.0
)

# Calculate bifacial performance
model = BifacialModuleModel(config)
back_irr = model.calculate_backside_irradiance(
    ground_albedo=0.70,
    tilt=30.0,
    clearance=1.0,
    front_poa_global=1000.0
)
gain = model.calculate_bifacial_gain(1000.0, back_irr, 0.70)
print(f"Bifacial gain: {gain*100:.1f}%")  # ~25% with white membrane
```

## Documentation

- **Bifacial Model**: See `docs/BIFACIAL_MODEL_DOCUMENTATION.md` for comprehensive documentation
- **Examples**: See `examples/bifacial_analysis_example.py` for detailed usage examples
- **Tests**: See `tests/test_bifacial_model.py` for validation and test cases

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── modules/
│       ├── __init__.py
│       └── bifacial_model.py      # Bifacial module modeling
├── tests/
│   ├── __init__.py
│   └── test_bifacial_model.py     # Comprehensive test suite
├── examples/
│   └── bifacial_analysis_example.py
├── docs/
│   └── BIFACIAL_MODEL_DOCUMENTATION.md
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Pydantic >= 2.0.0
- SciPy >= 1.10.0
- pytest >= 7.4.0 (for testing)

## License

MIT License - See LICENSE file for details
