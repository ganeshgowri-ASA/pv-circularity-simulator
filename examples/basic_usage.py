"""
Basic usage examples for SCAPS wrapper.

This script demonstrates:
1. Creating standard cell templates
2. Running simulations
3. Analyzing results
4. Exporting data
"""

from pathlib import Path

from src.modules import CellTemplates, SCAPSInterface


def main():
    """Run basic SCAPS wrapper examples."""
    print("=" * 70)
    print("SCAPS-1D Python Wrapper - Basic Usage Examples")
    print("=" * 70)

    # Create SCAPS interface
    print("\n1. Initializing SCAPS interface...")
    scaps = SCAPSInterface(
        working_directory=Path("./example_simulations"),
        cache_directory=Path("./.example_cache"),
        enable_cache=True
    )
    print("   ✓ Interface initialized")

    # Example 1: PERC cell
    print("\n2. Simulating PERC cell...")
    perc = CellTemplates.create_perc_cell()
    perc_results = scaps.run_simulation(perc)

    print(f"\n   PERC Results:")
    print(f"   - Voc:        {perc_results.voc:.4f} V")
    print(f"   - Jsc:        {perc_results.jsc:.4f} mA/cm²")
    print(f"   - Fill Factor: {perc_results.ff:.4f}")
    print(f"   - Efficiency:  {perc_results.efficiency*100:.2f}%")
    print(f"   - Pmax:       {perc_results.pmax:.4f} mW/cm²")

    # Example 2: TOPCon cell
    print("\n3. Simulating TOPCon cell...")
    topcon = CellTemplates.create_topcon_cell()
    topcon_results = scaps.run_simulation(topcon)

    print(f"\n   TOPCon Results:")
    print(f"   - Voc:        {topcon_results.voc:.4f} V")
    print(f"   - Jsc:        {topcon_results.jsc:.4f} mA/cm²")
    print(f"   - Fill Factor: {topcon_results.ff:.4f}")
    print(f"   - Efficiency:  {topcon_results.efficiency*100:.2f}%")
    print(f"   - Pmax:       {topcon_results.pmax:.4f} mW/cm²")

    # Example 3: HJT cell
    print("\n4. Simulating HJT cell...")
    hjt = CellTemplates.create_hjt_cell()
    hjt_results = scaps.run_simulation(hjt)

    print(f"\n   HJT Results:")
    print(f"   - Voc:        {hjt_results.voc:.4f} V")
    print(f"   - Jsc:        {hjt_results.jsc:.4f} mA/cm²")
    print(f"   - Fill Factor: {hjt_results.ff:.4f}")
    print(f"   - Efficiency:  {hjt_results.efficiency*100:.2f}%")
    print(f"   - Pmax:       {hjt_results.pmax:.4f} mW/cm²")

    # Example 4: Architecture comparison
    print("\n5. Architecture Comparison:")
    print(f"\n   {'Architecture':<15} {'Voc (V)':<10} {'Jsc (mA/cm²)':<15} {'FF':<8} {'Eff (%)':<10}")
    print(f"   {'-'*68}")
    print(f"   {'PERC':<15} {perc_results.voc:<10.4f} {perc_results.jsc:<15.4f} "
          f"{perc_results.ff:<8.4f} {perc_results.efficiency*100:<10.2f}")
    print(f"   {'TOPCon':<15} {topcon_results.voc:<10.4f} {topcon_results.jsc:<15.4f} "
          f"{topcon_results.ff:<8.4f} {topcon_results.efficiency*100:<10.2f}")
    print(f"   {'HJT':<15} {hjt_results.voc:<10.4f} {hjt_results.jsc:<15.4f} "
          f"{hjt_results.ff:<8.4f} {hjt_results.efficiency*100:<10.2f}")

    # Example 5: Export results
    print("\n6. Exporting results...")
    output_dir = Path("./example_outputs")
    output_dir.mkdir(exist_ok=True)

    scaps.export_results(perc_results, output_dir / "perc_results.json", format="json")
    scaps.export_results(perc_results, output_dir / "perc_results.csv", format="csv")

    print(f"   ✓ Results exported to {output_dir}")

    # Example 6: Parametric sweep
    print("\n7. Running parametric sweep (emitter doping)...")
    doping_levels = [1e18, 5e18, 1e19, 5e19, 1e20]
    sweep_results = []

    for doping in doping_levels:
        params = CellTemplates.create_perc_cell(emitter_doping=doping)
        result = scaps.run_simulation(params)
        sweep_results.append((doping, result))

    print(f"\n   {'Doping (cm⁻³)':<20} {'Efficiency (%)':<15} {'Voc (V)':<10} {'Jsc (mA/cm²)'}")
    print(f"   {'-'*68}")
    for doping, result in sweep_results:
        print(f"   {doping:<20.2e} {result.efficiency*100:<15.2f} "
              f"{result.voc:<10.4f} {result.jsc:.4f}")

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
