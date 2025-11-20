# PV Circularity Simulator 🌞

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Overview

The PV Circularity Simulator is a comprehensive Streamlit-based application designed to simulate and analyze the complete lifecycle of photovoltaic (solar) systems. From initial cell design through end-of-life circular economy considerations, this platform provides researchers, engineers, and sustainability professionals with powerful tools for optimization and analysis.

## Features

### 🔬 Design & Engineering
- **Cell Design**: PV cell design optimization with SCAPS integration
- **Module Engineering**: Module configuration and design tools
- **CTM Loss Analysis**: Cell-to-module power loss analysis and optimization

### 🗺️ System & Operations
- **System Planning**: Architecture design and system configuration
- **Performance Monitoring**: Real-time performance tracking and analytics
- **Energy Forecasting**: Production forecasting with weather integration
- **Reliability Testing**: Durability and reliability assessment tools

### ♻️ Circularity & Sustainability
- **Circularity Analysis**: 3R (Reduce, Reuse, Recycle) framework analysis
- **Circular Economy Modeling**: Economic and environmental impact modeling

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pv-circularity-simulator

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`

## Architecture

### Core Components

The application is built around four main functions in `app.py`:

1. **`streamlit_app()`** - Main application entry point
   - Page configuration and setup
   - Application orchestration
   - Lifecycle management

2. **`session_state_management()`** - State management
   - Initializes session variables
   - Maintains state across navigation
   - Workflow progress tracking

3. **`page_routing()`** - Navigation routing
   - Handles page transitions
   - Navigation history management
   - Dynamic page loading

4. **`sidebar_navigation()`** - UI navigation
   - Sidebar menu rendering
   - Project information display
   - Quick action buttons

### Project Structure

```
pv-circularity-simulator/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── SETUP.md              # Detailed setup guide
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── pages/
    ├── __init__.py       # Pages package
    └── README.md         # Pages documentation
```

## Technology Stack

- **Framework**: Streamlit 1.28+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Validation**: Pydantic
- **Scientific Computing**: SciPy

## Session State Management

The application uses a sophisticated session state structure:

```python
session_state = {
    # Core state
    "current_page": str,
    "initialized": bool,
    "project_name": str,

    # User preferences
    "user_data": {
        "preferences": {
            "theme": str,
            "units": str,
            "language": str
        }
    },

    # Simulation data
    "simulation_data": {
        "cell_design": {},
        "module_engineering": {},
        "ctm_losses": {},
        "system_config": {},
        "performance_data": {},
        "forecast_data": {},
        "reliability_data": {},
        "circularity_metrics": {},
        "circular_economy_model": {}
    },

    # Navigation
    "workflow_progress": Dict[str, bool],
    "navigation_history": List[str]
}
```

## Development

### Code Quality Standards

- **Docstrings**: Comprehensive Google-style docstrings for all functions
- **Type Hints**: Full type annotations throughout the codebase
- **Error Handling**: Graceful error handling with user-friendly messages
- **Production-Ready**: Enterprise-grade code structure and patterns

### Adding New Pages

See `pages/README.md` for detailed guidelines on implementing new page modules.

## Documentation

- **SETUP.md**: Detailed setup and installation guide
- **pages/README.md**: Page development guidelines
- **Inline Documentation**: Comprehensive docstrings in code

## Roadmap

- [ ] Implement individual page modules
- [ ] Add data export functionality
- [ ] Integrate SCAPS simulation engine
- [ ] Add authentication and user management
- [ ] Implement project save/load functionality
- [ ] Add automated testing suite
- [ ] Create Docker deployment configuration
- [ ] Add API endpoints for external integration

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the established patterns and style
2. All functions have comprehensive docstrings
3. Type hints are included
4. Changes are tested thoroughly

## License

See LICENSE file for details.

## Support

For questions, issues, or feature requests, please use the GitHub issue tracker.

---

**Version**: 1.0.0
**Status**: Initial Release
**Last Updated**: 2024
