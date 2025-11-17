# PV Circularity Simulator

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### 📊 Comprehensive Data Visualization Library

A production-ready visualization library specifically designed for photovoltaic system analysis:

- **Chart Templates**: Pre-configured charts optimized for PV data
  - Time series analysis
  - Bar charts and comparisons
  - Scatter plots with trendlines
  - Heatmaps for multi-dimensional data
  - Pie/donut charts for distributions
  - Multi-axis charts for correlated metrics
  - Dashboard grids for comprehensive monitoring

- **Interactive Plots**: Advanced interactivity features
  - Interactive time series with range selectors
  - Cross-filtering dashboards
  - Drill-down capabilities
  - Animated visualizations
  - Linked brushing for multi-view analysis
  - Real-time plot configurations

- **Export Functionality**: Multi-format export support
  - Static images (PNG, JPEG, SVG, PDF)
  - Interactive HTML
  - JSON data export
  - Multi-figure batch export
  - Image grid creation
  - Data + visualization bundles

- **Custom Themes**: Professional theming system
  - 6 pre-defined themes (Solar, Circularity, Performance, Technical, Dark, Default)
  - Custom theme creation
  - Plotly and Altair integration
  - Color palette management
  - Consistent styling across visualizations

- **PV-Specific Components**: Specialized visualizations
  - I-V and P-V curve plotting
  - Efficiency heatmaps
  - Degradation trend analysis
  - Material flow Sankey diagrams
  - Energy balance visualizations

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or with all extras
pip install -e ".[dev,docs]"
```

## Quick Start

### Basic Usage

```python
from pv_circularity.visualization import VisualizationLib
import pandas as pd
import numpy as np

# Initialize the visualization library
viz = VisualizationLib(default_theme='solar')

# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'power': np.random.rand(100) * 1000
})

# Create a time series chart
fig = viz.templates.time_series(
    df, 'date', 'power',
    title='PV Power Output'
)

# Export the chart
viz.export(fig, 'output.png', width=1600, height=900)
```

### Using Chart Templates

```python
from pv_circularity.visualization import chart_templates

templates = chart_templates()

# Bar chart
fig1 = templates.bar_chart(df, 'category', 'value')

# Scatter plot with trendline
fig2 = templates.scatter_plot(df, 'x', 'y', trendline=True)

# Heatmap
fig3 = templates.heatmap(matrix_data, title='Efficiency Map')
```

### Interactive Visualizations

```python
from pv_circularity.visualization import interactive_plots

interactive = interactive_plots()

# Interactive time series with range controls
fig = interactive.create_interactive_timeseries(
    df, 'date', 'power',
    enable_rangeslider=True,
    enable_rangeselector=True
)

# Animated scatter plot
fig_animated = interactive.create_animated_scatter(
    df, 'x', 'y', animation_frame='year'
)
```

### Custom Themes

```python
from pv_circularity.visualization import custom_themes

themes = custom_themes()

# List available themes
print(themes.list_themes())
# ['default', 'solar', 'circularity', 'performance', 'technical', 'dark']

# Set active theme
themes.set_theme('solar')

# Create custom theme
themes.create_custom_theme(
    name='corporate',
    primary='#003366',
    secondary='#66B2FF',
    accent='#FFD700'
)
```

### PV-Specific Components

```python
from pv_circularity.visualization.components import (
    IVCurveVisualizer,
    EfficiencyHeatmap,
    DegradationAnalyzer,
    SankeyFlowDiagram
)

# I-V curve visualization
iv_viz = IVCurveVisualizer()
fig = iv_viz.plot_iv_curve(voltage, current, voc=40, isc=8.5)

# Degradation analysis
deg_analyzer = DegradationAnalyzer()
fig = deg_analyzer.plot_degradation_trend(
    years, performance_ratio,
    calculate_rate=True,
    add_forecast=True
)

# Material flow diagram
sankey = SankeyFlowDiagram()
fig = sankey.create_material_flow(sources, targets, values)
```

## Examples

Comprehensive examples are available in the `examples/` directory:

- **basic_usage.py**: Fundamental visualization patterns
- **advanced_usage.py**: Advanced features and specialized components

Run examples:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_circularity --cov-report=html

# Run specific test file
pytest tests/test_visualization/test_core.py
```

## Documentation

### API Reference

The visualization library provides four main interfaces:

1. **VisualizationLib**: Main integrated interface
2. **chart_templates()**: Pre-configured chart templates
3. **interactive_plots()**: Interactive visualization components
4. **export_functionality()**: Multi-format export capabilities
5. **custom_themes()**: Theme management and customization

### Available Themes

- **solar**: Optimized for solar energy data (orange/blue palette)
- **circularity**: For lifecycle and circular economy (green/blue palette)
- **performance**: For monitoring and analytics (purple/teal palette)
- **technical**: For engineering analysis (dark gray/cyan palette)
- **dark**: For dark mode interfaces (cyan/pink palette)
- **default**: Clean and professional (matplotlib-inspired palette)

### Export Formats

- **PNG**: High-quality raster images
- **JPEG**: Compressed raster images
- **SVG**: Scalable vector graphics
- **PDF**: Print-ready documents
- **HTML**: Interactive web visualizations
- **JSON**: Chart data and configuration

## Project Structure

```
pv-circularity-simulator/
├── src/
│   └── pv_circularity/
│       ├── __init__.py
│       └── visualization/          # Visualization library
│           ├── __init__.py
│           ├── core.py             # Main VisualizationLib class
│           ├── templates.py        # Chart templates
│           ├── interactive.py      # Interactive plots
│           ├── exports.py          # Export functionality
│           ├── themes.py           # Theme management
│           └── components/         # PV-specific components
│               ├── __init__.py
│               └── pv_specific.py
├── tests/
│   └── test_visualization/
│       ├── test_core.py
│       ├── test_themes.py
│       └── test_templates.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Dependencies

Core dependencies:
- plotly >= 5.18.0
- altair >= 5.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- kaleido >= 0.2.1
- pillow >= 10.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Plotly and Altair visualization libraries
- Designed for photovoltaic system lifecycle analysis
- Optimized for circular economy modeling

## Contact

For questions or feedback, please open an issue on GitHub.
