# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### ✅ Irradiance Modeling & Solar Resource Assessment

Production-ready solar irradiance modeling with comprehensive analysis tools:

- **Solar Position Calculation**: High-accuracy algorithms (NREL SPA)
- **Irradiance Decomposition**: GHI → DNI/DHI using DIRINT, DISC, Erbs models
- **Transposition Models**: Perez, Hay-Davies, Isotropic sky models
- **POA Irradiance**: Complete plane-of-array calculations with loss factors
- **Spectral & AOI Losses**: Module-specific corrections
- **Resource Analysis**: P50/P90 assessment, monthly/seasonal statistics
- **Interactive Visualizations**: Plotly charts, heat maps, dashboards

**Quick Start:**

```python
from src.irradiance import IrradianceCalculator, POAIrradianceModel, SolarResourceAnalyzer
from src.irradiance.models import LocationConfig, SurfaceConfig

# Configure location and surface
location = LocationConfig(latitude=39.74, longitude=-105.18, timezone="America/Denver")
surface = SurfaceConfig(tilt=30.0, azimuth=180.0, albedo=0.2)

# Calculate solar position
calc = IrradianceCalculator(location)
solar_pos = calc.get_solar_position(times)

# Calculate POA irradiance with losses
poa_model = POAIrradianceModel(location, surface)
poa_components = poa_model.calculate_poa_components(
    irradiance_data, solar_pos, transposition_model='perez',
    include_spectral=True, include_aoi=True
)

# Analyze solar resource
analyzer = SolarResourceAnalyzer(poa_components.poa_global)
p_analysis = analyzer.p50_p90_analysis()
```

See [`docs/IRRADIANCE_MODELING.md`](docs/IRRADIANCE_MODELING.md) for complete documentation.

Run example: `python examples/complete_irradiance_analysis.py`

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run example
python examples/complete_irradiance_analysis.py
```

## Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── irradiance/          # Irradiance modeling components
│   │   ├── calculator.py    # Solar position & decomposition
│   │   ├── poa_model.py     # POA irradiance calculations
│   │   ├── resource_analyzer.py  # Statistical analysis
│   │   └── models.py        # Pydantic data models
│   └── ui/
│       └── visualizations.py  # Plotly charts
├── tests/
│   └── test_irradiance/     # Comprehensive test suite
├── examples/
│   └── complete_irradiance_analysis.py  # Full demonstration
├── docs/
│   └── IRRADIANCE_MODELING.md  # Technical documentation
└── requirements.txt         # Dependencies (pvlib, plotly, etc.)
```

## Technology Stack

- **pvlib-python**: Industry-standard solar modeling library
- **NumPy/pandas**: Vectorized numerical computations
- **Plotly**: Interactive visualizations
- **Pydantic**: Type-safe data models
- **pytest**: Comprehensive testing

## Documentation

- [Irradiance Modeling Documentation](docs/IRRADIANCE_MODELING.md)
- [API Reference](docs/) (coming soon)
- [Examples](examples/)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_irradiance/test_calculator.py -v
```

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- Code follows Black formatting
- Docstrings for all public methods
- Type hints where applicable

## License

MIT License - see [LICENSE](LICENSE) file

## References

- pvlib-python: https://pvlib-python.readthedocs.io/
- NREL Solar Position Algorithms: https://midcdmz.nrel.gov/spa/
- Perez Model: Perez, R., et al. (1990). Solar Energy, 44(5), 271-289.
