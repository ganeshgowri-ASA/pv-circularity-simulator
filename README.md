# 🌞 PV Circularity Simulator

> End-to-end photovoltaic lifecycle simulation platform with integrated circular economy modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📋 Overview

The PV Circularity Simulator is a comprehensive platform for simulating the complete lifecycle of photovoltaic systems, from cell design through end-of-life circularity. The platform integrates advanced simulation tools, performance monitoring, and circular economy modeling to support sustainable solar energy development.

## ✨ Key Features

### 🔬 Design & Engineering
- **Cell Design**: Optimize photovoltaic cell configurations with SCAPS integration
- **Module Engineering**: Design and analyze solar modules with CTM loss analysis
- **System Planning**: Plan complete PV installations with energy forecasting

### 📊 Analysis & Monitoring
- **Performance Monitoring**: Real-time and historical performance tracking
- **Reliability Testing**: Comprehensive reliability and degradation analysis
- **Circularity (3R)**: Model circular economy practices (Reduce, Reuse, Recycle)

### 🧭 Advanced Navigation System
- **Page Registry**: Organized page registration with metadata
- **Route Handling**: URL parameter support and session routing
- **Breadcrumbs**: Hierarchical navigation breadcrumbs
- **Deep Linking**: Direct URL links to specific pages and states

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pv-circularity-simulator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

```bash
streamlit run src/main.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── main.py                    # Main Streamlit application
│   ├── navigation/                # Navigation & Routing System
│   │   ├── __init__.py
│   │   └── navigation_manager.py  # Core navigation manager
│   ├── pages/                     # Application pages
│   │   ├── home.py
│   │   ├── cell_design.py
│   │   ├── module_engineering.py
│   │   ├── system_planning.py
│   │   ├── performance_monitoring.py
│   │   └── circularity.py
│   ├── components/                # Reusable UI components
│   └── utils/                     # Utility functions
├── tests/                         # Test suite
│   └── test_navigation.py
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── .streamlit/                    # Streamlit configuration
│   └── config.toml
└── README.md
```

## 🧭 Navigation System

The platform features a production-ready navigation system with:

### Core Components

- **NavigationManager**: Central routing and navigation management
- **PageConfig**: Page configuration with metadata
- **Route**: Route representation with parameters
- **AccessLevel**: Authorization levels (PUBLIC, AUTHENTICATED, ADMIN, CUSTOM)

### Key Features

#### 1. Page Registry
```python
@nav.page_registry(
    name="dashboard",
    title="Dashboard",
    icon="📊",
    description="Main dashboard",
    parent="home",
    keywords=["dashboard", "overview"]
)
def dashboard_page():
    st.write("Dashboard content")
```

#### 2. Route Handler
```python
# In main.py
nav.route_handler()  # Handles routing and renders current page
```

#### 3. Breadcrumbs
```python
# Display hierarchical breadcrumbs
nav.breadcrumbs(separator=" → ", show_icons=True)
```

#### 4. Deep Linking
```python
# Access via URL: http://localhost:8501/?page=dashboard&id=123
route = nav.deep_linking()
# route.page_name = "dashboard"
# route.params = {"id": "123"}
```

### Navigation API

```python
# Navigate to a page
nav.navigate("dashboard", params={"filter": "active"})

# Get current page
current = nav.get_current_page()

# Get navigation history
history = nav.get_navigation_history()

# Go back
nav.go_back()

# Set URL parameters
nav.set_query_params({"sort": "date", "order": "desc"})
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

Run specific test file:

```bash
pytest tests/test_navigation.py -v
```

## 📚 Documentation

### Page Development

To add a new page:

1. **Create page module** in `src/pages/your_page.py`:
   ```python
   def render():
       st.title("Your Page")
       st.write("Content here")
   ```

2. **Register page** in `src/main.py`:
   ```python
   @nav.page_registry(
       name="your_page",
       title="Your Page",
       icon="🎯",
       description="Page description"
   )
   def your_page():
       from pages import your_page
       your_page.render()
   ```

### Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server settings
- Browser behavior

Edit `.env` to configure:
- Application settings
- Database connections
- Feature flags

## 🔧 Technology Stack

- **Frontend**: Streamlit 1.28+
- **Data Processing**: NumPy, Pandas, SciPy
- **Visualization**: Plotly, Matplotlib
- **Testing**: pytest, pytest-cov
- **Type Checking**: Pydantic

## 📊 Features by Module

### Cell Design
- Cell parameter optimization
- SCAPS integration
- I-V curve analysis
- Efficiency calculations

### Module Engineering
- CTM loss analysis
- Module configuration
- Performance modeling
- Specification generation

### System Planning
- System sizing
- Layout design
- Energy forecasting
- Economic analysis

### Performance Monitoring
- Real-time monitoring
- Historical analysis
- Reliability testing
- Degradation tracking

### Circularity (3R)
- Material reduction strategies
- Reuse pathways
- Recycling processes
- Economic impact analysis

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Inspired by sustainable energy and circular economy principles
- SCAPS integration for detailed cell simulation

## 📞 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Version**: 0.1.0
**Status**: Production-ready Navigation System ✅
**Last Updated**: 2025-11-17
