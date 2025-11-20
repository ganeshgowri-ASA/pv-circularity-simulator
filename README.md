# 🌞 PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🆕 BATCH4-B05-S05: System Design UI & Interactive Configurator

A comprehensive system design interface integrating all B05 modules (PVsyst, mounting, shade analysis, optimization) with interactive configurator, 3D visualization, and real-time design validation.

### ✨ Features

#### Core Components

1. **SystemDesignDashboard** - Main dashboard with comprehensive configuration panels
   - `create_main_layout()` - Multi-tab interface with site, modules, inverters, mounting, optimization
   - `module_selection_panel()` - Interactive module database with filtering and configuration
   - `inverter_configuration()` - Inverter selection with MPPT configuration
   - `mounting_structure_selector()` - Mounting type, tilt, azimuth, tracking configuration
   - `optimization_controls()` - System optimization with multi-objective functions

2. **InteractiveConfigurator** - Drag-drop layout with real-time validation
   - `drag_drop_layout()` - Interactive array layout designer (automatic & manual modes)
   - `real_time_validation()` - Live validation against design constraints
   - `design_constraints_check()` - Electrical, structural, and regulatory compliance
   - `auto_optimization_trigger()` - Automatic optimization suggestions
   - `design_comparison_view()` - Multi-design comparison interface

3. **Visualization3D** - Advanced 3D rendering and analysis
   - `render_system_3d()` - Complete 3D system visualization with Plotly
   - `animated_sun_path()` - Dynamic sun path calculation and display
   - `shade_visualization()` - Real-time shade analysis and visualization
   - `terrain_overlay()` - Terrain integration in 3D view
   - `interactive_camera_controls()` - Multi-view camera controls (isometric, top, front, side)

4. **DesignValidation** - Comprehensive NEC compliance and validation
   - `check_nec_compliance()` - NEC 2023 Article 690 validation
   - `validate_string_sizing()` - String voltage and MPPT range validation
   - `check_voltage_limits()` - Temperature-corrected voltage limit checks
   - `verify_current_limits()` - Current rating and safety factor validation
   - `flag_design_issues()` - Best practice violations and optimization opportunities

5. **PerformancePreview** - Energy and financial analysis
   - `annual_energy_estimate()` - Monthly and annual energy production
   - `pr_calculation()` - Performance ratio with detailed loss breakdown
   - `shading_loss_summary()` - Comprehensive shading loss analysis
   - `financial_preview()` - LCOE, NPV, IRR, payback period calculations
   - `export_design_report()` - Professional PDF/Excel/JSON reports

6. **SystemDesignUI** - Streamlit multi-page application
   - 🏠 Home - Welcome and quick start guide
   - 📍 Site Configuration - Location and environmental parameters
   - 🔲 Module Selection - Interactive module database and configuration
   - ⚡ Inverters - Inverter selection and MPPT configuration
   - 📐 Layout Designer - Drag-drop array layout with optimization
   - 🏗️ Mounting - Structure configuration (fixed, tracking, rooftop)
   - 🎯 Optimization - Multi-objective optimization controls
   - ✅ Validation - NEC compliance and design validation
   - 📊 Performance - Energy, PR, and financial analysis
   - 🎨 3D Visualization - Interactive 3D rendering with sun path
   - 📈 Results & Export - Comprehensive reports (PDF, Excel, JSON)

### 🚀 Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pv-circularity-simulator.git
cd pv-circularity-simulator

# Run the startup script (creates venv, installs dependencies, launches app)
./RUN.sh
```

#### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run ui/system_design_app.py
```

### 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── models/
│   │   └── pv_components.py           # Pydantic data models
│   └── b05_system_design/
│       ├── dashboard/
│       │   └── system_design_dashboard.py
│       ├── configurator/
│       │   └── interactive_configurator.py
│       ├── visualization/
│       │   └── visualization_3d.py
│       ├── validation/
│       │   └── design_validation.py
│       └── performance/
│           └── performance_preview.py
├── ui/
│   └── system_design_app.py           # Main Streamlit application
├── tests/
│   └── b05_system_design/             # Unit tests
├── docs/                              # Documentation
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── RUN.sh                            # Startup script
└── README.md                         # This file
```

### 🎯 Usage Examples

#### 1. Basic System Design Workflow

```python
from src.models.pv_components import PVModule, Inverter, SystemDesign, SiteLocation
from src.b05_system_design.dashboard.system_design_dashboard import SystemDesignDashboard

# Initialize dashboard
dashboard = SystemDesignDashboard()

# Configure site
site = SiteLocation(
    name="My Solar Site",
    latitude=37.7749,
    longitude=-122.4194,
    elevation=10.0,
    albedo=0.2
)

# Select module (from database)
module = dashboard.module_database["Trina Solar TSM-DEG21C.20"]

# Create string configuration
string_config = StringConfiguration(
    modules_per_string=20,
    num_strings=10,
    module=module
)

# Select inverter
inverter = dashboard.inverter_database["SMA Sunny Tripower CORE1 110"]

# Create system design
design = SystemDesign(
    design_id="design_001",
    design_name="My PV System",
    site=site,
    modules=[string_config],
    inverters=[inverter],
    mounting=mounting_config,
    dc_ac_ratio=1.25
)
```

#### 2. Design Validation

```python
from src.b05_system_design.validation.design_validation import DesignValidation

# Initialize validator
validator = DesignValidation(nec_version="2023")

# Run comprehensive validation
validation_result = validator.validate_complete_design(design)

if validation_result.is_valid:
    print("✅ Design is valid!")
else:
    print(f"❌ Found {validation_result.error_count} errors")
    for error in validation_result.errors:
        print(f"  - {error}")
```

#### 3. Performance Analysis

```python
from src.b05_system_design.performance.performance_preview import PerformancePreview

# Initialize performance analyzer
performance = PerformancePreview()

# Calculate energy estimate
energy = performance.annual_energy_estimate(design, site)
print(f"Annual Energy: {energy.annual_energy:,.0f} kWh")
print(f"Specific Yield: {energy.specific_yield:.0f} kWh/kWp")
print(f"Capacity Factor: {energy.capacity_factor:.1f}%")

# Calculate performance ratio
pr = performance.pr_calculation(design, site)
print(f"Performance Ratio: {pr.pr_value:.3f}")

# Financial analysis
financial = performance.financial_preview(design, site)
print(f"LCOE: ${financial.lcoe:.3f}/kWh")
print(f"Payback Period: {financial.payback_period:.1f} years")
print(f"NPV: ${financial.npv:,.0f}")
```

#### 4. 3D Visualization

```python
from src.b05_system_design.visualization.visualization_3d import Visualization3D

# Initialize 3D visualizer
viz = Visualization3D()

# Render 3D system
fig = viz.render_system_3d(
    layout=array_layout,
    mounting=mounting_config,
    site=site
)

# Calculate sun path
sun_path = viz.animated_sun_path(site, datetime.now(), num_points=48)

# Perform shade analysis
shaded_layout = viz.shade_visualization(array_layout, site, datetime.now())
```

### 🔧 Technical Requirements

- **Python**: >=3.9
- **Core Libraries**:
  - `streamlit` - Web application framework
  - `plotly` - Interactive 3D visualization
  - `pydantic` - Data validation and modeling
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scipy` - Scientific computing
- **Optional**:
  - `pvlib-python` - Advanced solar calculations
  - `pyvista` - Advanced 3D rendering

### 📊 Data Models

All components use Pydantic models for type safety and validation:

- **PVModule** - Complete module specifications (electrical, physical, thermal)
- **Inverter** - Inverter characteristics (power, voltage, MPPT, efficiency)
- **MountingStructure** - Mounting configuration (type, tilt, azimuth, tracking)
- **StringConfiguration** - String design (modules per string, parallel strings)
- **SiteLocation** - Geographic and environmental parameters
- **SystemDesign** - Complete system design with all components
- **ValidationResult** - Validation outcomes and issues
- **EnergyEstimate** - Energy production calculations
- **PerformanceRatio** - PR calculations with loss breakdown
- **FinancialMetrics** - Cost and revenue analysis

### ✅ Validation Features

#### NEC 2023 Compliance Checks

- **NEC 690.7** - Maximum voltage calculations (temperature-corrected)
- **NEC 690.8** - Circuit sizing and current (1.25× safety factor)
- **NEC 690.9** - Overcurrent protection requirements
- **NEC 690.12** - Rapid shutdown compliance
- **NEC 690.13** - Disconnecting means verification
- **NEC 690.35** - Three-phase balance checking

#### Design Validation

- String voltage within MPPT range
- Temperature-corrected voltage limits
- Current rating compliance
- DC/AC ratio optimization
- Module orientation analysis
- System loss assessment

### 📈 Performance Metrics

- **Annual Energy Production** - kWh/year with monthly breakdown
- **Specific Yield** - kWh/kWp/year
- **Capacity Factor** - Percentage of theoretical maximum
- **Performance Ratio** - Industry-standard PR calculation
- **Loss Analysis** - Detailed waterfall of all system losses
- **Shading Analysis** - Near, far, and row-to-row shading
- **Financial Metrics** - LCOE, NPV, IRR, payback period

### 🎨 Visualization Features

- **3D System Rendering** - Interactive Plotly 3D visualization
- **Sun Path Animation** - Annual and daily sun path display
- **Shade Analysis** - Real-time shading calculation and display
- **Camera Controls** - Isometric, top, front, side views
- **Layout Visualization** - 2D array layout with module positioning
- **Loss Waterfall** - Interactive loss breakdown chart
- **Monthly Production** - Bar charts of monthly energy

### 📄 Export Formats

- **Excel** - Multi-sheet workbook with summary, monthly data, losses, components
- **JSON** - Complete design data in JSON format
- **PDF** - Professional design report (requires additional configuration)

### 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/b05_system_design/test_validation.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### 📚 Documentation

Full documentation available at: [https://pv-circularity-simulator.readthedocs.io](https://pv-circularity-simulator.readthedocs.io)

Build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

### 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

### 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments

- **NEC 2023** - National Electrical Code for compliance standards
- **NREL** - National Renewable Energy Laboratory for solar data
- **Streamlit** - For the excellent web framework
- **Plotly** - For interactive visualization capabilities

### 📞 Support

For issues, questions, or contributions:
- GitHub Issues: [https://github.com/your-org/pv-circularity-simulator/issues](https://github.com/your-org/pv-circularity-simulator/issues)
- Email: support@pv-circularity.com
- Documentation: [https://pv-circularity-simulator.readthedocs.io](https://pv-circularity-simulator.readthedocs.io)

---

**Status**: Production-ready ✅
**Version**: 1.0.0
**Last Updated**: November 2025
