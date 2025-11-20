# PV Circularity Simulator

End-to-end photovoltaic lifecycle simulation platform with comprehensive thermal imaging analysis and IV curve diagnostics.

## Overview

The PV Circularity Simulator is a complete platform for analyzing and optimizing photovoltaic systems across their entire lifecycle:
- **Cell Design** → **Module Engineering** → **System Planning** → **Performance Monitoring** → **Circularity (3R)**

Key capabilities include:
- **Thermal Imaging Analysis**: Hotspot detection, IR defect detection, temperature distribution analysis
- **IV Curve Analysis**: Electrical parameter extraction, degradation analysis, fault diagnostics
- CTM loss analysis
- SCAPS integration
- Reliability testing
- Energy forecasting
- Circular economy modeling

## Features

### 🔥 Thermal Imaging Analysis (BATCH6-B08-S02)

Production-ready thermal imaging analysis with comprehensive IR defect detection:

#### `ThermalImageAnalyzer`
- **Hotspot Detection**: Multi-method detection (threshold, Z-score, clustering)
- **Temperature Distribution Analysis**: Statistical analysis of thermal patterns
- **Thermal Anomaly Identification**: Detection of cold spots, edge heating, and abnormal patterns
- **Bypass Diode Failure Detection**: Identify failed bypass diodes from thermal signatures

#### `IRImageProcessing`
- **Temperature Calibration**: Atmospheric and distance corrections
- **Emissivity Correction**: Adjust for material emissivity differences
- **Background Subtraction**: Adaptive and fixed background removal
- **Image Denoising**: Gaussian, median, and bilateral filtering

#### `HotspotSeverityClassifier`
- **Severity Classification**: Normal, Warning, Moderate, Severe, Critical levels
- **Power Loss Estimation**: Quantify energy losses from hotspots
- **Failure Prediction**: Estimate failure probability and time-to-failure

### ⚡ IV Curve Analysis (BATCH6-B08-S03)

Complete electrical diagnostics with IV curve modeling and analysis:

#### `IVCurveAnalyzer`
- **Curve Tracing**: Smoothing, outlier removal, interpolation
- **Parameter Extraction**: Voc, Isc, Vmp, Imp, FF, Rs, Rsh, ideality factor
- **Degradation Analysis**: Quantify performance losses vs baseline
- **Mismatch Detection**: Identify cell mismatches from curve irregularities

#### `ElectricalDiagnostics`
- **String Underperformance**: Detect underperforming modules in arrays
- **Cell Failures**: Identify shunting, high resistance, and other defects
- **Bypass Diode Issues**: Detect activated or failed bypass diodes

#### `CurveComparison`
- **Baseline Comparison**: Compare current vs expected performance
- **Trend Analysis**: Multi-year degradation rate calculation
- **Anomaly Detection**: Statistical outlier detection with Z-scores

## Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,docs,flir]"
```

## Quick Start

### Thermal Imaging Analysis

```python
from datetime import datetime
import numpy as np
from pv_circularity_simulator.diagnostics import ThermalImageAnalyzer
from pv_circularity_simulator.core.models import ThermalImageData, ThermalImageMetadata

# Create thermal image data
metadata = ThermalImageMetadata(
    timestamp=datetime.now(),
    camera_model="FLIR E95",
    ambient_temp=25.0,
    measurement_distance=5.0,
    emissivity=0.90,
    irradiance=1000.0
)

thermal_data = ThermalImageData(
    temperature_matrix=your_temperature_array,
    metadata=metadata,
    width=width,
    height=height
)

# Analyze thermal image
analyzer = ThermalImageAnalyzer()
result = analyzer.analyze(thermal_data)

print(f"Hotspots detected: {len(result.hotspots)}")
print(f"Overall severity: {result.overall_severity}")
print(f"Temperature uniformity: {result.temperature_uniformity:.3f}")
```

### IV Curve Analysis

```python
from datetime import datetime
import numpy as np
from pv_circularity_simulator.diagnostics import IVCurveAnalyzer
from pv_circularity_simulator.core.models import IVCurveData

# Create IV curve data
iv_data = IVCurveData(
    voltage=voltage_array,
    current=current_array,
    temperature=25.0,
    irradiance=1000.0,
    timestamp=datetime.now()
)

# Extract electrical parameters
analyzer = IVCurveAnalyzer()
params = analyzer.parameter_extraction(iv_data)

print(f"Voc: {params.voc:.2f} V")
print(f"Isc: {params.isc:.2f} A")
print(f"Pmp: {params.pmp:.2f} W")
print(f"Fill Factor: {params.fill_factor:.4f}")
```

## Examples

Comprehensive examples are provided in the `examples/` directory:

- **`thermal_analysis_example.py`**: Complete thermal imaging workflow
- **`iv_curve_analysis_example.py`**: Complete IV curve analysis workflow

Run examples:
```bash
python examples/thermal_analysis_example.py
python examples/iv_curve_analysis_example.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_circularity_simulator --cov-report=html

# Run specific test file
pytest tests/unit/test_thermal.py
pytest tests/unit/test_iv_curve.py
```

## Architecture

```
pv-circularity-simulator/
├── src/pv_circularity_simulator/
│   ├── core/                    # Core models and utilities
│   │   ├── models.py            # Pydantic data models
│   │   ├── constants.py         # Physical constants and thresholds
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── utils.py             # Utility functions
│   │
│   └── diagnostics/             # Diagnostic modules
│       ├── thermal.py           # Thermal imaging analysis
│       └── iv_curve.py          # IV curve analysis
│
├── tests/                       # Unit and integration tests
├── examples/                    # Usage examples
└── docs/                        # Documentation
```

## Technology Stack

- **Core**: Python 3.9+, NumPy, SciPy
- **Data Validation**: Pydantic 2.0+
- **Image Processing**: OpenCV, scikit-image
- **Analysis**: scikit-learn, pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Testing**: pytest, pytest-cov

## Key Technologies

- **Thermal Image Processing**: OpenCV, FLIR integration
- **IV Curve Modeling**: Single-diode model, parameter extraction algorithms
- **Statistical Analysis**: Z-score anomaly detection, trend analysis
- **Machine Learning**: DBSCAN clustering for hotspot detection
- **Data Validation**: Pydantic models with comprehensive validation

## Documentation

Full API documentation is available in the `docs/` directory. Each module includes comprehensive docstrings with:
- Detailed parameter descriptions
- Return value documentation
- Usage examples
- Exception documentation

## Contributing

Contributions are welcome! Please ensure:
- All code includes comprehensive docstrings
- Unit tests are provided for new features
- Code follows PEP 8 style guidelines (enforced by ruff)
- Type hints are used throughout

## License

MIT License - see LICENSE file for details

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator},
  author = {PV Circularity Team},
  year = {2025},
  url = {https://github.com/ganeshgowri-ASA/pv-circularity-simulator}
}
```

## Acknowledgments

This project implements state-of-the-art thermal imaging and electrical diagnostics techniques for photovoltaic systems, incorporating best practices from industry standards (IEC 61215, IEC 62446) and academic research.
