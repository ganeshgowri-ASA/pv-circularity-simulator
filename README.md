# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🚀 Features

### BATCH4-B05: PVsyst Integration & System Design Engine

Comprehensive PV system design with PVsyst-level accuracy for utility-scale, commercial, and residential applications.

#### Core Components

1. **PVsystIntegration** - Parse and integrate PVsyst files
   - Parse .PAN (module), .OND (inverter), .MET (weather) files
   - Generate PVsyst-compatible project files
   - Import/export PVsyst simulation results

2. **SystemDesignEngine** - Complete system design orchestration
   - Configure array layouts, strings, inverters, and transformers
   - Optimize system layout for maximum energy yield
   - Calculate comprehensive system losses
   - Design DC/AC collection systems

3. **StringSizingCalculator** - NEC 690 & IEC 60364 compliant string sizing
   - Calculate max/min string lengths based on temperature
   - Validate string configurations
   - Optimize MPPT utilization
   - NEC-compliant fuse sizing

4. **InverterSelector** - Database-driven inverter selection
   - Search 10+ major manufacturers (SMA, Fronius, Huawei, etc.)
   - DC/AC ratio optimization
   - Clipping loss analysis
   - Central vs. string inverter comparison

5. **ArrayLayoutDesigner** - Multi-mounting type layouts
   - Ground-mounted (fixed-tilt, single-axis, dual-axis trackers)
   - Rooftop (flat, sloped) with fire setbacks
   - Carport & canopy structures
   - Floating solar systems
   - Agrivoltaic systems
   - BIPV facades

6. **SystemLossModel** - Comprehensive loss modeling
   - Soiling (geographic database)
   - Shading (near/far, backtracking)
   - DC/AC wiring (resistance calculations)
   - Inverter efficiency curves
   - Transformer losses
   - Clipping analysis
   - Availability & curtailment

7. **SystemDesignUI** - Interactive Streamlit interface
   - Project configuration
   - Module & inverter selection
   - System sizing & layout
   - Loss waterfall visualization
   - Performance metrics
   - PVsyst export

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🎯 Quick Start

### Python API

```python
from pv_simulator.system_design import (
    SystemDesignEngine,
    ModuleParameters,
    InverterParameters,
    SystemType,
    MountingType,
)

# Create system design engine
engine = SystemDesignEngine(
    project_name="Solar Farm 1",
    system_type=SystemType.UTILITY,
    location="Phoenix, AZ",
    latitude=33.45,
    longitude=-112.07,
    elevation=340.0,
)

# Define module
module = ModuleParameters(
    manufacturer="Trina Solar",
    model="TSM-DEG21C.20",
    pmax=670.0,
    voc=45.9,
    isc=18.52,
    vmp=38.4,
    imp=17.45,
    temp_coeff_pmax=-0.34,
    temp_coeff_voc=-0.25,
    temp_coeff_isc=0.05,
    length=2.384,
    width=1.303,
    thickness=0.035,
    weight=34.6,
    cells_in_series=132,
    efficiency=21.5,
)

# Define inverter (or load from database)
inverter = InverterParameters(
    manufacturer="SMA",
    model="SC-2750-EV",
    inverter_type="central",
    pac_max=2750000,  # 2.75 MW
    vac_nom=480,
    pdc_max=2860000,
    vdc_max=1500,
    num_mppt=6,
    mppt_vmin=580,
    mppt_vmax=1300,
    max_efficiency=98.8,
)

# Design complete system
system_config = engine.design_system_configuration(
    module=module,
    inverter=inverter,
    target_dc_capacity_kw=100000,  # 100 MW
    mounting_type=MountingType.GROUND_SINGLE_AXIS,
    site_temp_min=-10.0,
    site_temp_max=70.0,
    target_dc_ac_ratio=1.25,
)

# View results
print(f"DC Capacity: {system_config.dc_capacity:.1f} kW")
print(f"AC Capacity: {system_config.ac_capacity:.1f} kW")
print(f"Modules: {system_config.num_modules:,}")
print(f"Inverters: {system_config.num_inverters}")
print(f"DC/AC Ratio: {system_config.dc_ac_ratio:.2f}")
print(f"Total Losses: {system_config.losses.total_losses():.1f}%")
```

### Streamlit UI

```bash
# Launch interactive UI
streamlit run pv_simulator/ui/system_design_ui.py
```

### String Sizing Calculator

```python
from pv_simulator.system_design import StringSizingCalculator

calculator = StringSizingCalculator(
    module=module,
    inverter=inverter,
    site_temp_min=-10.0,
    site_temp_max=70.0,
)

# Get optimal string configuration
string_config = calculator.design_optimal_string()

print(f"Modules per string: {string_config.modules_per_string}")
print(f"Strings per MPPT: {string_config.strings_per_mppt}")
print(f"String Voc (STC): {string_config.voc_stc:.1f}V")
print(f"String Vmp (STC): {string_config.vmp_stc:.1f}V")

# Validate custom configuration
is_valid, msg = calculator.validate_string_configuration(
    modules_per_string=20,
    strings_per_mppt=2,
)
print(f"Valid: {is_valid}, {msg}")
```

### PVsyst File Parsing

```python
from pv_simulator.system_design import PVsystIntegration

pvsyst = PVsystIntegration()

# Parse PVsyst .PAN file
module = pvsyst.parse_pvsyst_pan_file("path/to/module.PAN")

# Parse PVsyst .OND file
inverter = pvsyst.parse_pvsyst_ond_file("path/to/inverter.OND")

# Parse meteorological data
weather = pvsyst.parse_pvsyst_meteo_file("path/to/weather.MET")

# Generate PVsyst project
prj_file = pvsyst.generate_pvsyst_project(
    project_name="MyProject",
    module=module,
    inverter=inverter,
    output_path="./output",
    num_modules=1000,
    num_inverters=10,
)
```

### Inverter Selection

```python
from pv_simulator.system_design import InverterSelector

selector = InverterSelector(
    module=module,
    system_type=SystemType.UTILITY,
    database_path="pv_simulator/data/inverter_database.json",
)

# Search for suitable inverters
candidates = selector.search_inverter_database(
    dc_power_kw=100000,
    inverter_type="central",
    manufacturer="SMA",
)

print(f"Found {len(candidates)} suitable inverters")

# Optimize DC/AC ratio
import numpy as np
dc_profile = np.random.rand(8760) * 100000  # Hourly DC power profile
optimal_ratio, analysis = selector.optimize_dc_ac_ratio(
    dc_power_profile=dc_profile,
    target_clipping_percent=2.0,
)

print(f"Optimal DC/AC ratio: {optimal_ratio:.2f}")
print(f"Clipping losses: {analysis['clipping_loss_percent']:.2f}%")
```

## 🧪 Testing

```bash
# Run all tests
pytest pv_simulator/tests/

# Run with coverage
pytest --cov=pv_simulator pv_simulator/tests/

# Run specific test file
pytest pv_simulator/tests/test_string_sizing_calculator.py
```

## 📊 System Loss Model

The simulator includes comprehensive loss modeling:

- **Soiling**: Geographic database (desert: 4.5%, temperate: 2.0%, etc.)
- **Shading**: Near/far shading with backtracking support
- **DC Wiring**: Resistance-based calculations (copper/aluminum)
- **AC Wiring**: Three-phase collection system losses
- **Inverter**: Load-dependent efficiency curves
- **Transformer**: No-load and load losses
- **Clipping**: DC-side and AC-side clipping analysis
- **Availability**: Grid curtailment and maintenance downtime
- **LID**: Light-induced degradation
- **Mismatch**: Module-level and string-level mismatch

## 🏗️ Architecture

```
pv_simulator/
├── system_design/
│   ├── models.py                      # Pydantic data models
│   ├── system_design_engine.py        # Main orchestrator
│   ├── string_sizing_calculator.py    # NEC/IEC compliant string sizing
│   ├── inverter_selector.py           # Inverter database & selection
│   ├── array_layout_designer.py       # Multi-mounting layouts
│   ├── system_loss_model.py           # Comprehensive loss modeling
│   └── pvsyst_integration.py          # PVsyst file parsing
├── data/
│   └── inverter_database.json         # 10+ inverter manufacturers
├── ui/
│   └── system_design_ui.py            # Streamlit interface
└── tests/
    ├── test_string_sizing_calculator.py
    └── test_system_design_engine.py
```

## 📋 Requirements

- Python 3.9+
- NumPy, Pandas, SciPy
- Pydantic v2+
- PVLib
- Streamlit
- Plotly
- See `requirements.txt` for complete list

## 🔧 Configuration

### Inverter Database

The inverter database (`pv_simulator/data/inverter_database.json`) includes:

- **Central Inverters**: SMA SC-2750-EV, Power Electronics FS3450, ABB PVS980
- **String Inverters**: SMA Sunny Tripower, Fronius Symo, Huawei SUN2000, Sungrow SG250HX
- **Microinverters**: Enphase IQ8PLUS
- **Power Optimizers**: SolarEdge SE100K

Add custom inverters by editing the JSON file following the schema.

## 📖 Documentation

### Key Classes

- **SystemDesignEngine**: Main system design orchestrator
- **StringSizingCalculator**: NEC 690 & IEC 60364 compliant string sizing
- **InverterSelector**: Database search & DC/AC optimization
- **ArrayLayoutDesigner**: Multi-mounting type layouts
- **SystemLossModel**: Comprehensive loss calculations
- **PVsystIntegration**: PVsyst file parsing & generation

### Data Models (Pydantic)

- **ModuleParameters**: Module electrical & physical specs
- **InverterParameters**: Inverter specifications
- **SystemConfiguration**: Complete system design
- **StringConfiguration**: String sizing results
- **ArrayLayout**: Array layout parameters
- **SystemLosses**: Loss breakdown

## 🌟 Features by System Type

### Utility-Scale (MW+)
- Central inverters (1-5 MW)
- Single-axis tracker support
- GCR optimization
- Combiner box placement
- MV collection systems

### Commercial (100kW - 5MW)
- String inverters
- Rooftop & carport layouts
- Fire setback compliance
- 480V collection

### Residential (<100kW)
- String inverters / microinverters
- Rooftop optimization
- Module-level MPPT
- 240V single-phase

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

PV Circularity Team

## 🔗 Links

- Repository: https://github.com/ganeshgowri-ASA/pv-circularity-simulator
- Issues: https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues

## 🙏 Acknowledgments

- PVsyst for industry-leading simulation methodology
- NREL for PVLib and solar resource data
- SMA, Fronius, Huawei, Enphase, and other manufacturers for inverter specifications

## 📈 Roadmap

- [ ] Weather data integration (NSRDB, Meteonorm)
- [ ] Bifacial module support
- [ ] Advanced shading analysis (3D modeling)
- [ ] Energy storage integration
- [ ] Financial modeling (LCOE, NPV, IRR)
- [ ] Degradation modeling
- [ ] Real-time monitoring integration
- [ ] API for external tools
