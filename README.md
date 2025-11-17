# ☀️ PV Circularity Simulator

**End-to-end PV lifecycle simulation platform with advanced thermal modeling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io)

## 🎯 Overview

The PV Circularity Simulator is a comprehensive platform for modeling photovoltaic systems throughout their lifecycle, from cell design to circular economy analysis. The platform features advanced thermal modeling capabilities with multiple industry-standard temperature prediction models and integration with the B03 NOCT database.

### Key Features

- 🌡️ **Advanced Thermal Modeling**
  - Multiple temperature models (Sandia, PVsyst, Faiman, NOCT-based)
  - Heat transfer physics calculations
  - Wind speed and mounting configuration effects
  - Thermal time constant analysis

- 📊 **Interactive Dashboards**
  - Real-time temperature predictions
  - Cooling analysis and optimization
  - Heat transfer coefficient breakdowns
  - Time series analysis

- 🗄️ **B03 NOCT Database**
  - 20+ verified module specifications
  - Multiple technologies (mono-Si, CdTe, HJT, bifacial, perovskite)
  - Real-world thermal performance data

- 🔬 **Production-Ready Code**
  - Full type hints and Pydantic models
  - Comprehensive docstrings
  - Extensive test coverage
  - pvlib integration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Running the Application

```bash
# Launch Streamlit app
streamlit run ui/app.py
```

Navigate to `http://localhost:8501` in your browser.

## 📖 Documentation

### Temperature Modeling

The simulator provides four main temperature prediction models:

#### 1. Sandia Model (King et al. 2004)

Empirical model based on outdoor testing:

```python
from pv_simulator.core.cell_temperature import CellTemperatureModel
from pv_simulator.models.thermal import TemperatureConditions, MountingConfiguration

conditions = TemperatureConditions(
    ambient_temp=25.0,
    irradiance=1000.0,
    wind_speed=3.0
)
mounting = MountingConfiguration(mounting_type="open_rack")

model = CellTemperatureModel(conditions=conditions, mounting=mounting)
result = model.sandia_model()

print(f"Cell Temperature: {result.cell_temperature:.1f}°C")
```

#### 2. PVsyst Model

Heat loss factor based model:

```python
result = model.pvsyst_model(u_c=29.0, u_v=0.0)
```

#### 3. Faiman Model

Two-parameter heat transfer model:

```python
result = model.faiman_model(u0=25.0, u1=6.84)
```

#### 4. NOCT-based Model

Simple NOCT-based temperature estimation:

```python
result = model.noct_based(noct=45.0)
```

### Using B03 NOCT Database

```python
from pv_simulator.data.loaders import load_b03_noct_database

# Load database
loader = load_b03_noct_database()

# Get module by ID
module = loader.get_module_by_id("B03-00001")
print(f"NOCT: {module.noct_spec.noct_celsius}°C")
print(f"Power: {module.rated_power_stc}W")

# Search by manufacturer
modules = loader.get_modules_by_manufacturer("SunPower")

# Get statistics
stats = loader.get_statistics()
print(f"Total modules: {stats['total_modules']}")
```

### Heat Transfer Analysis

```python
from pv_simulator.core.cell_temperature import ModuleTemperatureCalculator

calculator = ModuleTemperatureCalculator(conditions=conditions, mounting=mounting)

# Calculate heat transfer coefficients
coeffs = calculator.heat_transfer_coefficients()
print(f"Front convective: {coeffs.convective_front:.2f} W/(m²·K)")

# Analyze wind speed effects
wind_effects = calculator.wind_speed_effects()

# Compare mounting configurations
mount_comparison = calculator.mounting_configuration_effects()

# Calculate thermal time constants
tau = calculator.thermal_time_constants(wind_speed=3.0)
print(f"Heating time constant: {tau['tau_heating_minutes']:.1f} minutes")
```

## 🏗️ Project Structure

```
pv-circularity-simulator/
├── src/pv_simulator/          # Core library
│   ├── models/                # Pydantic data models
│   │   ├── thermal.py        # Thermal modeling models
│   │   └── noct.py           # NOCT specifications
│   ├── core/                  # Core simulation logic
│   │   └── cell_temperature.py  # Temperature modeling
│   ├── data/                  # Data management
│   │   └── loaders.py        # NOCT data loaders
│   └── utils/                 # Utility functions
│       ├── constants.py      # Physical constants
│       └── helpers.py        # Helper functions
├── ui/                        # Streamlit interface
│   ├── app.py                # Main application
│   ├── pages/                # Dashboard pages
│   │   └── 1_🌡️_Thermal_Modeling.py
│   └── components/           # Reusable components
│       └── thermal_viz.py   # Visualization components
├── data/                      # Data files
│   └── raw/noct/
│       └── b03_noct_data.csv  # B03 NOCT database
├── tests/                     # Test suite
│   ├── test_models/          # Model tests
│   └── test_core/            # Core logic tests
└── examples/                  # Usage examples
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_simulator --cov-report=html

# Run specific test file
pytest tests/test_core/test_cell_temperature.py
```

## 📊 B03 NOCT Database

The simulator includes a comprehensive database of 20+ PV modules with verified NOCT data:

| Technology | Count | NOCT Range | Efficiency Range |
|-----------|-------|------------|------------------|
| Mono-Si | 15 | 41-45.5°C | 19.8-23.5% |
| Bifacial | 3 | 42.5-44.2°C | 20.6-21.8% |
| HJT | 2 | 41-43.5°C | 21.7-22.0% |
| CdTe | 1 | 46.0°C | 18.5% |
| Perovskite | 1 | 40.0°C | 24.5% |

## 🔧 Technical Details

### Temperature Models

**Sandia Model:**
```
T_module = T_ambient + (E/E0) * exp(a + b*ws) + ΔT
```

**PVsyst Model:**
```
T_cell = T_ambient + (E / (u_c + u_v * ws)) * (1 - η)
```

**Faiman Model:**
```
T_module = T_ambient + (E * α) / (u0 + u1 * ws)
```

**NOCT-based:**
```
T_cell = T_ambient + (NOCT - 20) * (E / 800) * wind_correction
```

### Heat Transfer Calculations

- Convective heat transfer (forced and natural convection)
- Radiative heat transfer (sky and ground exchange)
- Mounting configuration effects
- Thermal time constants (heating/cooling response)

## 📚 References

1. **King, D. L., et al. (2004).** "Sandia Photovoltaic Array Performance Model." SAND2004-3535.
2. **Faiman, D. (2008).** "Assessing the outdoor operating temperature of photovoltaic modules." Progress in Photovoltaics, 16(4), 307-315.
3. **Mermoud, A. (2012).** "PVsyst User's Manual."
4. **IEC 61215** - Terrestrial photovoltaic (PV) modules - Design qualification and type approval

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- pvlib-python for photovoltaic modeling tools
- Streamlit for the interactive dashboard framework
- The PV research community for thermal modeling methodologies

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for the solar energy community**
