# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### BATCH4-B05-S03: Helioscope Integration & Advanced Shade Analysis

Comprehensive 3D shade analysis and system design tools including terrain modeling, horizon profiling, near/far shading analysis, and advanced electrical modeling for accurate energy yield predictions.

#### Core Components

1. **HelioscapeModel** - 3D site modeling and terrain analysis
2. **ShadeAnalysisEngine** - Comprehensive near/far shade analysis
3. **SunPositionCalculator** - NREL SPA solar position algorithm
4. **IrradianceOnSurface** - POA irradiance with Perez transposition
5. **ElectricalShadingModel** - Bypass diode and mismatch simulation
6. **SystemLayoutOptimizer** - Multi-parameter layout optimization
7. **HorizonProfiler** - Horizon profile management
8. **ShadeAnalysisUI** - Interactive Streamlit interface

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis import (
    Location, SiteModel, ArrayGeometry, ShadeAnalysisConfig, ShadeAnalysisEngine
)

# Define site and array
location = Location(latitude=37.7749, longitude=-122.4194)
site_model = SiteModel(location=location, albedo=0.2)
array_geometry = ArrayGeometry(tilt=20.0, azimuth=180.0, gcr=0.4,
                               module_width=1.0, module_height=2.0,
                               modules_per_string=20, row_spacing=5.0)

# Run shade analysis
config = ShadeAnalysisConfig(
    start_date=datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")),
    end_date=datetime(2024, 12, 31, tzinfo=ZoneInfo("UTC"))
)
engine = ShadeAnalysisEngine(site_model, array_geometry, config)
```

### Streamlit UI

```bash
streamlit run src/pv_circularity_simulator/batch4/b05_system_design/s03_helioscope_shade_analysis/ui.py
```

## Documentation

See full documentation in module docstrings. All classes and methods include comprehensive type hints and Pydantic validation.

## Testing

```bash
pytest tests/
```

## License

MIT License
