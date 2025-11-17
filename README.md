# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🚀 Features

### Hybrid Energy System UI & Configuration (BATCH10-PENDING-S68)

A comprehensive Streamlit-based interface for configuring, controlling, and monitoring hybrid energy systems.

**Core Components:**

- **System Configuration Wizard** - Step-by-step guided setup for hybrid energy systems
- **Component Selector** - Interactive interface for adding and configuring energy components (PV arrays, batteries, etc.)
- **Operation Strategy Builder** - Define control strategies and component priorities
- **Performance Monitoring Dashboard** - Real-time visualization and analysis

**Key Features:**

- Interactive Streamlit widgets and selections
- Real-time performance metrics and KPIs
- Power flow visualization with Plotly charts
- Component status monitoring
- Energy balance analysis
- Configurable operation strategies (rule-based, optimal, predictive)
- Export/import configuration management
- Production-ready with full docstrings

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Running the Hybrid Energy System UI

```bash
# Launch the Streamlit application
streamlit run app.py
```

### Using the UI Programmatically

```python
from src.ui.hybrid_system_ui import HybridSystemUI
from src.core.config import ConfigManager

# Create UI instance with default configuration
ui = HybridSystemUI()

# Or load from a configuration file
config = ConfigManager.load_config("config/my_system.yaml")
ui = HybridSystemUI(config=config)

# Render the complete interface
ui.render()

# Or use individual components
ui.system_configuration_wizard()
ui.component_selector()
ui.operation_strategy_builder()
ui.performance_monitoring_dashboard()
```

### Creating a Custom Configuration

```python
from src.core.config import SystemConfiguration, ComponentConfig, OperationStrategy

# Create system configuration
config = SystemConfiguration(
    system_name="My Hybrid System",
    system_description="Custom hybrid energy system",
    components=[
        ComponentConfig(
            component_id="pv_001",
            component_type="pv_array",
            name="Rooftop PV",
            capacity=15.0,
            capacity_unit="kW",
            efficiency=0.85,
            parameters={
                "area_m2": 75.0,
                "tilt_angle": 25.0,
                "azimuth_angle": 180.0
            }
        ),
        ComponentConfig(
            component_id="battery_001",
            component_type="battery",
            name="Energy Storage",
            capacity=30.0,
            capacity_unit="kWh",
            efficiency=0.90,
            parameters={
                "initial_soc": 0.5,
                "min_soc": 0.2,
                "max_soc": 0.9,
                "charge_rate_max_kw": 7.5,
                "discharge_rate_max_kw": 7.5
            }
        )
    ],
    operation_strategy=OperationStrategy(
        strategy_name="Maximize Self-Consumption",
        strategy_type="rule_based",
        priority_order=["pv_001", "battery_001"]
    )
)

# Save configuration
config.to_yaml("my_config.yaml")
```

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── core/              # Core models and configuration
│   │   ├── config.py      # Configuration management (Pydantic models)
│   │   └── models.py      # Energy system models
│   ├── ui/                # User interface modules
│   │   └── hybrid_system_ui.py  # Main UI class
│   ├── monitoring/        # Monitoring and metrics
│   │   └── metrics.py     # Performance metrics tracking
│   └── utils/             # Utility functions
│       └── helpers.py     # Helper functions
├── config/                # Configuration files
├── tests/                 # Test suite
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 UI Components

### 1. System Configuration Wizard

A step-by-step guided interface for setting up hybrid energy systems:

- **Step 1:** Basic system information
- **Step 2:** Component selection and configuration
- **Step 3:** Operation strategy setup
- **Step 4:** Simulation parameters
- **Step 5:** Review and confirmation

### 2. Component Selector

Interactive interface for managing energy components:

- Add components (PV arrays, batteries, wind turbines, etc.)
- Configure component-specific parameters
- Edit existing components
- Remove components
- View component status

### 3. Operation Strategy Builder

Define and configure control strategies:

- Strategy type selection (rule-based, optimal, predictive)
- Component priority ordering
- Control algorithm parameters
- Operating constraints
- Strategy validation

### 4. Performance Monitoring Dashboard

Real-time monitoring and visualization:

- Key Performance Indicators (KPIs)
- Component status indicators
- Power flow diagrams
- Energy balance charts
- Time-series analysis
- Performance metrics (self-sufficiency, renewable fraction, etc.)

## 🔧 Configuration

### System Configuration

The system uses Pydantic models for configuration validation. Configuration files can be in YAML format:

```yaml
system_name: "Example Hybrid System"
system_description: "Demonstration system"
components:
  - component_id: "pv_001"
    component_type: "pv_array"
    name: "PV Array 1"
    capacity: 10.0
    capacity_unit: "kW"
    efficiency: 0.85
    parameters:
      area_m2: 50.0
      tilt_angle: 30.0
      azimuth_angle: 180.0

operation_strategy:
  strategy_name: "Basic Rule-Based"
  strategy_type: "rule_based"
  priority_order: ["pv_001", "battery_001"]

simulation:
  time_step_minutes: 5
  simulation_duration_hours: 24
```

## 📊 Performance Metrics

The system tracks and displays various performance metrics:

- **Energy Metrics:**
  - Total generation (kWh)
  - Total consumption (kWh)
  - Grid import/export (kWh)

- **Performance Indicators:**
  - Renewable fraction (%)
  - Self-consumption ratio (%)
  - Self-sufficiency ratio (%)
  - Capacity factor (%)

- **Component Metrics:**
  - Battery state of charge
  - PV generation profiles
  - Component efficiency
  - Operating status

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📝 API Documentation

Full API documentation is available in the docstrings of each module. Key classes:

- `HybridSystemUI` - Main UI class (src/ui/hybrid_system_ui.py)
- `SystemConfiguration` - Configuration management (src/core/config.py)
- `HybridEnergySystem` - System simulation (src/core/models.py)
- `PerformanceMetrics` - Metrics tracking (src/monitoring/metrics.py)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- Advanced optimization algorithms
- Machine learning-based forecasting
- Integration with real weather APIs
- Multi-site system management
- Economic analysis and ROI calculations
- Circular economy metrics and reporting
- IoT device integration
- Cloud-based deployment options
