"""
Tests for chart templates functionality.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pv_circularity.visualization.templates import ChartTemplates, chart_templates


class TestChartTemplates:
    """Test suite for ChartTemplates class."""

    @pytest.fixture
    def sample_timeseries_data(self) -> pd.DataFrame:
        """Create sample time series data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'power': np.random.rand(50) * 1000,
            'temperature': np.random.rand(50) * 40 + 10
        })

    @pytest.fixture
    def sample_categorical_data(self) -> pd.DataFrame:
        """Create sample categorical data."""
        return pd.DataFrame({
            'category': ['Type A', 'Type B', 'Type C', 'Type D'],
            'value': [85, 92, 78, 88],
            'count': [10, 15, 8, 12]
        })

    def test_time_series_single_column(self, sample_timeseries_data) -> None:
        """Test time series chart with single column."""
        templates = ChartTemplates()
        fig = templates.time_series(
            sample_timeseries_data, 'date', 'power'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].mode == 'lines'

    def test_time_series_multiple_columns(self, sample_timeseries_data) -> None:
        """Test time series chart with multiple columns."""
        templates = ChartTemplates()
        fig = templates.time_series(
            sample_timeseries_data, 'date', ['power', 'temperature']
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_time_series_with_markers(self, sample_timeseries_data) -> None:
        """Test time series with markers enabled."""
        templates = ChartTemplates()
        fig = templates.time_series(
            sample_timeseries_data, 'date', 'power', show_markers=True
        )

        assert fig.data[0].mode == 'lines+markers'

    def test_bar_chart(self, sample_categorical_data) -> None:
        """Test bar chart creation."""
        templates = ChartTemplates()
        fig = templates.bar_chart(
            sample_categorical_data, 'category', 'value'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_bar_chart_horizontal(self, sample_categorical_data) -> None:
        """Test horizontal bar chart."""
        templates = ChartTemplates()
        fig = templates.bar_chart(
            sample_categorical_data, 'category', 'value', orientation='h'
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].orientation == 'h'

    def test_heatmap_from_dataframe(self) -> None:
        """Test heatmap from DataFrame."""
        df = pd.DataFrame(np.random.rand(10, 10))
        templates = ChartTemplates()
        fig = templates.heatmap(df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_heatmap_from_array(self) -> None:
        """Test heatmap from numpy array."""
        data = np.random.rand(5, 5)
        templates = ChartTemplates()
        fig = templates.heatmap(data)

        assert isinstance(fig, go.Figure)

    def test_scatter_plot(self) -> None:
        """Test scatter plot creation."""
        df = pd.DataFrame({
            'x': np.random.rand(50),
            'y': np.random.rand(50)
        })
        templates = ChartTemplates()
        fig = templates.scatter_plot(df, 'x', 'y')

        assert isinstance(fig, go.Figure)
        assert fig.data[0].mode == 'markers'

    def test_scatter_plot_with_trendline(self) -> None:
        """Test scatter plot with trendline."""
        df = pd.DataFrame({
            'x': np.linspace(0, 10, 50),
            'y': 2 * np.linspace(0, 10, 50) + np.random.randn(50)
        })
        templates = ChartTemplates()
        fig = templates.scatter_plot(df, 'x', 'y', trendline=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Data + trendline

    def test_pie_chart(self, sample_categorical_data) -> None:
        """Test pie chart creation."""
        templates = ChartTemplates()
        fig = templates.pie_chart(
            sample_categorical_data, 'value', 'category'
        )

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Pie)

    def test_donut_chart(self, sample_categorical_data) -> None:
        """Test donut chart (pie with hole)."""
        templates = ChartTemplates()
        fig = templates.pie_chart(
            sample_categorical_data, 'value', 'category', hole=0.4
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].hole == 0.4

    def test_multi_axis_chart(self, sample_timeseries_data) -> None:
        """Test multi-axis chart creation."""
        templates = ChartTemplates()
        fig = templates.multi_axis_chart(
            sample_timeseries_data, 'date', 'power', 'temperature'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_dashboard_grid(self, sample_timeseries_data) -> None:
        """Test dashboard grid creation."""
        templates = ChartTemplates()

        # Create individual figures
        fig1 = templates.time_series(sample_timeseries_data, 'date', 'power')
        fig2 = templates.time_series(sample_timeseries_data, 'date', 'temperature')

        # Create dashboard
        dashboard = templates.dashboard_grid([fig1, fig2], 2, 1)

        assert isinstance(dashboard, go.Figure)


class TestChartTemplatesCustomization:
    """Test chart template customization options."""

    def test_custom_dimensions(self) -> None:
        """Test charts with custom dimensions."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        templates = ChartTemplates()

        fig = templates.time_series(
            df, 'x', 'y', height=600, width=1000
        )

        assert fig.layout.height == 600
        assert fig.layout.width == 1000

    def test_custom_labels(self) -> None:
        """Test charts with custom axis labels."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        templates = ChartTemplates()

        fig = templates.time_series(
            df, 'x', 'y',
            x_label='Custom X',
            y_label='Custom Y'
        )

        assert fig.layout.xaxis.title.text == 'Custom X'
        assert fig.layout.yaxis.title.text == 'Custom Y'


def test_chart_templates_function() -> None:
    """Test chart_templates() convenience function."""
    templates = chart_templates()
    assert isinstance(templates, ChartTemplates)
