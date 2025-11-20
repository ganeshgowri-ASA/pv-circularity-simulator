"""
Advanced usage examples for PV Circularity Visualization Library.

This script demonstrates advanced features including interactive plots,
custom components, dashboards, and specialized PV visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pv_circularity.visualization import VisualizationLib
from pv_circularity.visualization.components import (
    IVCurveVisualizer,
    EfficiencyHeatmap,
    DegradationAnalyzer,
    SankeyFlowDiagram,
)


def main() -> None:
    """Run advanced usage examples."""
    print("PV Circularity Visualization Library - Advanced Examples")
    print("=" * 70)

    output_dir = Path("output/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = VisualizationLib(default_theme='solar')

    # Example 1: Interactive Time Series with Range Selector
    print("\n1. Creating Interactive Time Series...")
    df_interactive = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=365, freq='H'),
        'power': 500 + 300 * np.sin(np.linspace(0, 50*np.pi, 365)) + np.random.randn(365) * 50,
    })

    fig_interactive = viz.interactive.create_interactive_timeseries(
        df_interactive,
        x='timestamp',
        y='power',
        title='Interactive PV Power Output (Hourly)',
        enable_rangeslider=True,
        enable_rangeselector=True
    )

    viz.export(fig_interactive, output_dir / 'interactive_timeseries.html')
    print(f"   ✓ Saved: {output_dir / 'interactive_timeseries.html'}")

    # Example 2: I-V Curve Visualization
    print("\n2. Creating I-V Curve Visualization...")
    iv_viz = IVCurveVisualizer()

    # Generate sample I-V data
    voltage = np.linspace(0, 40, 100)
    isc = 8.5  # Short-circuit current
    voc = 40.0  # Open-circuit voltage
    vmp = 32.5  # Voltage at max power
    imp = 7.8  # Current at max power

    # Simplified I-V curve model
    current = isc * (1 - np.exp((voltage - voc) / 5))

    fig_iv = iv_viz.plot_iv_curve(
        voltage=voltage,
        current=current,
        title='PV Module I-V Characteristic Curve',
        voc=voc,
        isc=isc,
        vmp=vmp,
        imp=imp,
        include_power=True
    )

    viz.export(fig_iv, output_dir / 'iv_curve.png', width=1200, height=800)
    print(f"   ✓ Saved: {output_dir / 'iv_curve.png'}")

    # Example 3: I-V Curve Comparison
    print("\n3. Creating I-V Curve Comparison...")
    curves = {
        'Standard (25°C, 1000 W/m²)': (
            voltage,
            isc * (1 - np.exp((voltage - voc) / 5))
        ),
        'High Temp (45°C, 1000 W/m²)': (
            voltage,
            (isc - 0.5) * (1 - np.exp((voltage - (voc - 4)) / 5))
        ),
        'Low Irradiance (25°C, 500 W/m²)': (
            voltage,
            (isc * 0.5) * (1 - np.exp((voltage - (voc - 2)) / 5))
        ),
    }

    fig_iv_compare = iv_viz.compare_iv_curves(
        curves,
        title='I-V Curves Under Different Conditions'
    )

    viz.export(fig_iv_compare, output_dir / 'iv_comparison.png')
    print(f"   ✓ Saved: {output_dir / 'iv_comparison.png'}")

    # Example 4: Efficiency Heatmap
    print("\n4. Creating Efficiency Heatmap...")
    eff_viz = EfficiencyHeatmap()

    temperatures = [10, 20, 30, 40, 50, 60, 70]
    irradiances = [200, 400, 600, 800, 1000]

    # Simulate efficiency data
    efficiency_matrix = np.zeros((len(irradiances), len(temperatures)))
    for i, irr in enumerate(irradiances):
        for j, temp in enumerate(temperatures):
            base_eff = 20.0
            temp_coeff = -0.4  # %/°C
            irr_factor = 1 - (1000 - irr) * 0.0001
            efficiency_matrix[i, j] = base_eff + temp_coeff * (temp - 25) * irr_factor

    fig_eff_heatmap = eff_viz.create_2d_efficiency_map(
        x_values=temperatures,
        y_values=irradiances,
        efficiency_matrix=efficiency_matrix,
        x_label='Temperature (°C)',
        y_label='Irradiance (W/m²)',
        title='PV Module Efficiency Map',
        show_values=True
    )

    viz.export(fig_eff_heatmap, output_dir / 'efficiency_heatmap.png')
    print(f"   ✓ Saved: {output_dir / 'efficiency_heatmap.png'}")

    # Example 5: Degradation Analysis
    print("\n5. Creating Degradation Analysis...")
    deg_analyzer = DegradationAnalyzer()

    years = np.arange(0, 25)
    # Simulate degradation with some noise
    pr = 100 - 0.5 * years + np.random.randn(25) * 0.5

    fig_degradation = deg_analyzer.plot_degradation_trend(
        time_years=years,
        performance_ratio=pr,
        title='PV System 25-Year Degradation Analysis',
        calculate_rate=True,
        add_forecast=True,
        forecast_years=10
    )

    viz.export(fig_degradation, output_dir / 'degradation_analysis.png')
    print(f"   ✓ Saved: {output_dir / 'degradation_analysis.png'}")

    # Example 6: Material Flow Sankey Diagram
    print("\n6. Creating Material Flow Sankey Diagram...")
    sankey = SankeyFlowDiagram()

    sources = [
        'Manufacturing', 'Manufacturing', 'Manufacturing',
        'Installation', 'Installation',
        'Operation', 'Operation',
        'Collection', 'Collection', 'Collection',
        'Recycling'
    ]
    targets = [
        'Installation', 'Production Loss', 'Quality Control Reject',
        'Operation', 'Installation Loss',
        'Collection', 'Operating Loss',
        'Recycling', 'Refurbishment', 'Disposal',
        'New Materials'
    ]
    values = [
        950, 30, 20,  # Manufacturing
        900, 50,      # Installation
        850, 50,      # Operation
        700, 100, 50, # Collection
        600           # Recycling
    ]

    fig_sankey = sankey.create_material_flow(
        sources=sources,
        targets=targets,
        values=values,
        title='PV Module Material Flow (kg)'
    )

    viz.export(fig_sankey, output_dir / 'material_flow.html')
    print(f"   ✓ Saved: {output_dir / 'material_flow.html'}")

    # Example 7: Energy Balance Diagram
    print("\n7. Creating Energy Balance Diagram...")
    losses = {
        'Thermal Loss': 150,
        'Optical Loss': 50,
        'Electrical Loss': 80,
        'Soiling Loss': 20,
    }

    fig_energy = sankey.create_energy_balance(
        input_energy=1000,
        output_useful=700,
        losses=losses,
        title='PV System Energy Balance (W)'
    )

    viz.export(fig_energy, output_dir / 'energy_balance.html')
    print(f"   ✓ Saved: {output_dir / 'energy_balance.html'}")

    # Example 8: PV Performance Dashboard
    print("\n8. Creating PV Performance Dashboard...")
    df_dashboard = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
        'power': 500 + 300 * np.sin(np.linspace(0, 20*np.pi, 200)) + np.random.randn(200) * 50,
        'irradiance': 400 + 500 * np.sin(np.linspace(0, 20*np.pi, 200)) + np.random.randn(200) * 30,
        'temperature': 20 + 15 * np.sin(np.linspace(0, 20*np.pi, 200)) + np.random.randn(200) * 3,
    })

    fig_dashboard = viz.create_pv_performance_dashboard(
        df_dashboard,
        timestamp_col='timestamp',
        power_col='power',
        irradiance_col='irradiance',
        temperature_col='temperature',
        title='Comprehensive PV System Monitoring Dashboard'
    )

    viz.export(fig_dashboard, output_dir / 'performance_dashboard.png', width=1600, height=1200)
    print(f"   ✓ Saved: {output_dir / 'performance_dashboard.png'}")

    # Example 9: Circularity Analysis
    print("\n9. Creating Circularity Analysis...")
    df_circularity = pd.DataFrame({
        'stage': ['Raw Materials', 'Manufacturing', 'Use Phase', 'Collection', 'Recycling'],
        'material_flow': [1000, 950, 900, 800, 700]
    })

    fig_circularity = viz.create_circularity_analysis(
        df_circularity,
        categories='stage',
        values='material_flow',
        title='PV Lifecycle Material Flow Analysis'
    )

    viz.export(fig_circularity, output_dir / 'circularity_analysis.png')
    print(f"   ✓ Saved: {output_dir / 'circularity_analysis.png'}")

    # Example 10: Multi-format Export
    print("\n10. Demonstrating Multi-format Export...")
    df_export = pd.DataFrame({
        'category': ['A', 'B', 'C'],
        'value': [10, 20, 15]
    })

    fig_export = viz.templates.bar_chart(df_export, 'category', 'value')

    # Export to multiple formats
    for format in ['png', 'svg', 'html', 'json']:
        viz.export(fig_export, output_dir / f'multi_export.{format}')
        print(f"   ✓ Saved: {output_dir / f'multi_export.{format}'}")

    print("\n" + "=" * 70)
    print("✓ All advanced examples completed successfully!")
    print(f"✓ Output files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
