"""
PV Circularity Visualization Library.

A comprehensive visualization library for photovoltaic system lifecycle analysis,
providing chart templates, interactive plots, export functionality, and custom themes.

Main Components:
    - VisualizationLib: Main interface for all visualization capabilities
    - chart_templates(): Access to pre-configured chart templates
    - interactive_plots(): Interactive visualization components
    - export_functionality(): Multi-format export capabilities
    - custom_themes(): Theme management and customization

Quick Start:
    >>> from pv_circularity.visualization import VisualizationLib
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create visualization library instance
    >>> viz = VisualizationLib(default_theme='solar')
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2024-01-01', periods=100),
    ...     'power': np.random.rand(100) * 1000
    ... })
    >>>
    >>> # Create a chart
    >>> fig = viz.templates.time_series(df, 'date', 'power', title='Power Output')
    >>>
    >>> # Export it
    >>> viz.export(fig, 'output.png', width=1600, height=900)

Examples:
    Using chart templates:
        >>> from pv_circularity.visualization import chart_templates
        >>> templates = chart_templates()
        >>> fig = templates.bar_chart(df, 'category', 'value')

    Creating interactive plots:
        >>> from pv_circularity.visualization import interactive_plots
        >>> interactive = interactive_plots()
        >>> fig = interactive.create_interactive_timeseries(df, 'date', 'power')

    Customizing themes:
        >>> from pv_circularity.visualization import custom_themes
        >>> themes = custom_themes()
        >>> themes.set_theme('solar')
        >>> print(themes.list_themes())

    Exporting visualizations:
        >>> from pv_circularity.visualization import export_functionality
        >>> exporter = export_functionality()
        >>> exporter.export_plotly(fig, 'chart.png', width=1200, height=800)
"""

from pv_circularity.visualization.core import (
    VisualizationLib,
    chart_templates,
    interactive_plots,
    export_functionality,
    custom_themes,
)

from pv_circularity.visualization.templates import ChartTemplates
from pv_circularity.visualization.interactive import InteractivePlots
from pv_circularity.visualization.exports import ExportManager
from pv_circularity.visualization.themes import ThemeManager, ColorPalette

from pv_circularity.visualization.components import (
    IVCurveVisualizer,
    EfficiencyHeatmap,
    DegradationAnalyzer,
    SankeyFlowDiagram,
)

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "VisualizationLib",
    # Convenience functions
    "chart_templates",
    "interactive_plots",
    "export_functionality",
    "custom_themes",
    # Core classes
    "ChartTemplates",
    "InteractivePlots",
    "ExportManager",
    "ThemeManager",
    "ColorPalette",
    # Custom components
    "IVCurveVisualizer",
    "EfficiencyHeatmap",
    "DegradationAnalyzer",
    "SankeyFlowDiagram",
    # Version
    "__version__",
]
