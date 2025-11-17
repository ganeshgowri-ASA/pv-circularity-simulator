"""
Basic usage examples for PV Circularity Visualization Library.

This script demonstrates the fundamental usage patterns of the visualization
library including creating charts, applying themes, and exporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pv_circularity.visualization import VisualizationLib


def main() -> None:
    """Run basic usage examples."""
    print("PV Circularity Visualization Library - Basic Usage Examples")
    print("=" * 70)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Initialize visualization library with solar theme
    viz = VisualizationLib(default_theme='solar')
    print(f"\n✓ Initialized VisualizationLib with 'solar' theme")

    # Example 1: Time Series Chart
    print("\n1. Creating Time Series Chart...")
    df_timeseries = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=365, freq='D'),
        'power_output': 800 + 200 * np.sin(np.linspace(0, 4*np.pi, 365)) + np.random.randn(365) * 50,
        'irradiance': 600 + 400 * np.sin(np.linspace(0, 4*np.pi, 365)) + np.random.randn(365) * 30,
    })

    fig1 = viz.templates.time_series(
        df_timeseries,
        x_col='timestamp',
        y_cols=['power_output', 'irradiance'],
        title='Annual PV System Performance',
        x_label='Date',
        y_label='Power (W) / Irradiance (W/m²)'
    )

    viz.export(fig1, output_dir / 'timeseries_chart.png', width=1600, height=900)
    print(f"   ✓ Saved: {output_dir / 'timeseries_chart.png'}")

    # Example 2: Bar Chart
    print("\n2. Creating Bar Chart...")
    df_modules = pd.DataFrame({
        'module_type': ['Monocrystalline', 'Polycrystalline', 'Thin Film', 'Perovskite'],
        'efficiency': [22.5, 18.2, 14.8, 25.2],
        'technology': ['Silicon', 'Silicon', 'CdTe', 'Hybrid']
    })

    fig2 = viz.templates.bar_chart(
        df_modules,
        x_col='module_type',
        y_col='efficiency',
        title='PV Module Technology Efficiency Comparison',
        x_label='Module Type',
        y_label='Efficiency (%)',
        color_col='technology'
    )

    viz.export(fig2, output_dir / 'bar_chart.png')
    print(f"   ✓ Saved: {output_dir / 'bar_chart.png'}")

    # Example 3: Scatter Plot with Trendline
    print("\n3. Creating Scatter Plot...")
    df_scatter = pd.DataFrame({
        'irradiance': np.random.rand(100) * 1000,
        'power': np.random.rand(100) * 500 + 200,
    })

    fig3 = viz.templates.scatter_plot(
        df_scatter,
        x_col='irradiance',
        y_col='power',
        title='Power vs Irradiance',
        trendline=True
    )

    viz.export(fig3, output_dir / 'scatter_plot.png')
    print(f"   ✓ Saved: {output_dir / 'scatter_plot.png'}")

    # Example 4: Pie Chart - Circularity Distribution
    print("\n4. Creating Pie Chart...")
    df_circularity = pd.DataFrame({
        'category': ['Recycled', 'Reused', 'Repurposed', 'Disposed'],
        'percentage': [45, 30, 15, 10]
    })

    fig4 = viz.templates.pie_chart(
        df_circularity,
        values_col='percentage',
        names_col='category',
        title='End-of-Life Material Distribution',
        hole=0.4  # Donut chart
    )

    viz.export(fig4, output_dir / 'pie_chart.png')
    print(f"   ✓ Saved: {output_dir / 'pie_chart.png'}")

    # Example 5: Heatmap
    print("\n5. Creating Heatmap...")
    # Simulated degradation data: efficiency over years and temperature
    years = list(range(1, 26))  # 25 years
    temps = [20, 30, 40, 50, 60]  # Temperatures in °C
    efficiency_data = np.zeros((len(temps), len(years)))

    for i, temp in enumerate(temps):
        base_degradation = 0.5  # Base degradation rate %/year
        temp_factor = 1 + (temp - 25) * 0.01  # Temperature acceleration
        for j, year in enumerate(years):
            efficiency_data[i, j] = 100 - (base_degradation * temp_factor * year)

    df_heatmap = pd.DataFrame(
        efficiency_data,
        index=[f'{t}°C' for t in temps],
        columns=[f'Year {y}' for y in years]
    )

    fig5 = viz.templates.heatmap(
        df_heatmap,
        title='PV Module Efficiency Degradation (Temperature vs Time)',
        colorscale='RdYlGn',
        show_values=False
    )

    viz.export(fig5, output_dir / 'heatmap.png')
    print(f"   ✓ Saved: {output_dir / 'heatmap.png'}")

    # Example 6: Quick Plot (Auto-detection)
    print("\n6. Creating Quick Plot...")
    df_quick = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=90, freq='D'),
        'power': np.random.rand(90) * 1000 + 500
    })

    fig6 = viz.quick_plot(df_quick, x='date', y='power', title='Quick Plot Example')
    viz.export(fig6, output_dir / 'quick_plot.png')
    print(f"   ✓ Saved: {output_dir / 'quick_plot.png'}")

    # Example 7: Theme Switching
    print("\n7. Demonstrating Theme Switching...")
    print(f"   Available themes: {viz.list_themes()}")

    for theme_name in ['circularity', 'performance', 'dark']:
        viz.set_theme(theme_name)
        fig_theme = viz.templates.bar_chart(
            df_modules, 'module_type', 'efficiency',
            title=f'Chart with {theme_name.title()} Theme'
        )
        viz.export(fig_theme, output_dir / f'theme_{theme_name}.png')
        print(f"   ✓ Saved: {output_dir / f'theme_{theme_name}.png'}")

    # Get library info
    print("\n8. Library Information:")
    info = viz.get_info()
    print(f"   Version: {info['version']}")
    print(f"   Current Theme: {info['current_theme']}")
    print(f"   Available Chart Types: {len(info['capabilities']['chart_templates'])}")

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print(f"✓ Output files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
