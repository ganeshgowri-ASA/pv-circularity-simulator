"""
Tests for core visualization library functionality.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pv_circularity.visualization import (
    VisualizationLib,
    chart_templates,
    interactive_plots,
    export_functionality,
    custom_themes,
)


class TestVisualizationLib:
    """Test suite for VisualizationLib class."""

    def test_initialization_default_theme(self) -> None:
        """Test initialization with default theme."""
        viz = VisualizationLib()
        assert viz.themes._current_theme == "default"

    def test_initialization_custom_theme(self) -> None:
        """Test initialization with custom theme."""
        viz = VisualizationLib(default_theme="solar")
        assert viz.themes._current_theme == "solar"

    def test_property_access(self) -> None:
        """Test access to all properties."""
        viz = VisualizationLib()
        assert viz.templates is not None
        assert viz.interactive is not None
        assert viz.exporter is not None
        assert viz.themes is not None

    def test_quick_plot_auto_detection(self) -> None:
        """Test quick plot with automatic chart type detection."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'power': np.random.rand(30) * 1000
        })

        viz = VisualizationLib()
        fig = viz.quick_plot(df, x='date', y='power')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_quick_plot_explicit_type(self) -> None:
        """Test quick plot with explicit chart type."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 15]
        })

        viz = VisualizationLib()
        fig = viz.quick_plot(df, chart_type='bar', x='category', y='value')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_quick_plot_interactive(self) -> None:
        """Test quick plot with interactive mode."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'power': np.random.rand(50) * 1000
        })

        viz = VisualizationLib()
        fig = viz.quick_plot(df, x='date', y='power', interactive=True)

        assert isinstance(fig, go.Figure)
        # Interactive plots should have rangeslider
        assert fig.layout.xaxis.rangeslider is not None

    def test_pv_performance_dashboard(self) -> None:
        """Test PV performance dashboard creation."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'power': np.random.rand(100) * 1000,
            'irradiance': np.random.rand(100) * 1000,
            'temperature': np.random.rand(100) * 40 + 10
        })

        viz = VisualizationLib()
        dashboard = viz.create_pv_performance_dashboard(df)

        assert isinstance(dashboard, go.Figure)
        assert len(dashboard.data) > 0

    def test_circularity_analysis(self) -> None:
        """Test circular economy analysis visualization."""
        df = pd.DataFrame({
            'stage': ['Manufacturing', 'Use', 'Collection', 'Recycling'],
            'material_kg': [1000, 950, 800, 700]
        })

        viz = VisualizationLib()
        fig = viz.create_circularity_analysis(df, 'stage', 'material_kg')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_apply_theme(self) -> None:
        """Test applying theme to existing figure."""
        viz = VisualizationLib()
        fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6])])

        styled_fig = viz.apply_theme(fig, 'solar')

        assert isinstance(styled_fig, go.Figure)
        assert styled_fig.layout.template is not None

    def test_set_theme(self) -> None:
        """Test setting active theme."""
        viz = VisualizationLib()
        viz.set_theme('solar')

        assert viz.themes._current_theme == 'solar'

    def test_list_themes(self) -> None:
        """Test listing available themes."""
        viz = VisualizationLib()
        themes = viz.list_themes()

        assert isinstance(themes, list)
        assert 'solar' in themes
        assert 'circularity' in themes
        assert 'performance' in themes

    def test_get_info(self) -> None:
        """Test getting library information."""
        viz = VisualizationLib()
        info = viz.get_info()

        assert isinstance(info, dict)
        assert 'version' in info
        assert 'capabilities' in info
        assert 'current_theme' in info


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_chart_templates_function(self) -> None:
        """Test chart_templates() convenience function."""
        templates = chart_templates()
        assert templates is not None
        assert hasattr(templates, 'time_series')
        assert hasattr(templates, 'bar_chart')

    def test_interactive_plots_function(self) -> None:
        """Test interactive_plots() convenience function."""
        interactive = interactive_plots()
        assert interactive is not None
        assert hasattr(interactive, 'create_interactive_timeseries')

    def test_export_functionality_function(self) -> None:
        """Test export_functionality() convenience function."""
        exporter = export_functionality()
        assert exporter is not None
        assert hasattr(exporter, 'export_plotly')

    def test_custom_themes_function(self) -> None:
        """Test custom_themes() convenience function."""
        themes = custom_themes()
        assert themes is not None
        assert hasattr(themes, 'set_theme')
        assert hasattr(themes, 'list_themes')


class TestIntegration:
    """Integration tests for visualization library."""

    def test_full_workflow(self, tmp_path) -> None:
        """Test complete visualization workflow."""
        # Create data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'power': np.random.rand(100) * 1000,
            'efficiency': np.random.rand(100) * 25
        })

        # Initialize library
        viz = VisualizationLib(default_theme='solar')

        # Create chart
        fig = viz.templates.time_series(
            df, 'date', ['power', 'efficiency'],
            title='PV System Performance'
        )

        # Apply theme
        styled_fig = viz.apply_theme(fig)

        # Export (using tmp_path for testing)
        output_file = tmp_path / "test_chart.png"
        viz.export(styled_fig, str(output_file), width=1200, height=600)

        # Verify file was created
        assert output_file.exists()

    def test_multiple_chart_types(self) -> None:
        """Test creating multiple chart types."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value1': [10, 20, 15, 25],
            'value2': [5, 15, 10, 20]
        })

        viz = VisualizationLib()

        # Bar chart
        bar_fig = viz.templates.bar_chart(df, 'category', 'value1')
        assert isinstance(bar_fig, go.Figure)

        # Pie chart
        pie_fig = viz.templates.pie_chart(df, 'value1', 'category')
        assert isinstance(pie_fig, go.Figure)

        # Scatter plot
        scatter_fig = viz.templates.scatter_plot(df, 'value1', 'value2')
        assert isinstance(scatter_fig, go.Figure)
