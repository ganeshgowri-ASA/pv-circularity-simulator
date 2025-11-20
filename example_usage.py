"""
Example usage of the PV Circularity Simulator core infrastructure.

This script demonstrates how to use the SessionManager and Pydantic data models.
"""

from pathlib import Path
from src.core import (
    SessionManager,
    SimulationModule,
    create_default_monocrystalline_cell,
    create_example_silicon_material,
    CellTechnology,
    MaterialType,
    Module,
    ModuleLayout,
    CuttingPattern,
    PVSystem,
    Location,
    MountingType,
    FinancialModel,
)


def main():
    """Run example workflow."""
    print("=" * 70)
    print("PV Circularity Simulator - Example Usage")
    print("=" * 70)
    print()

    # 1. Initialize Session Manager
    print("1. Initializing Session Manager...")
    manager = SessionManager(projects_directory=Path("./example_projects"))
    print(f"   Projects directory: {manager.projects_directory}")
    print()

    # 2. Create a new project
    print("2. Creating new project...")
    manager.create_new_project(
        project_name="Example PV System Design",
        description="Demonstration of 330W monocrystalline module design",
        author="Example User"
    )
    print(f"   Project created: {manager.current_session.metadata.project_name}")
    print(f"   Project ID: {manager.current_session.metadata.project_id}")
    print()

    # 3. Material Selection Module
    print("3. Material Selection Module")
    silicon = create_example_silicon_material()
    print(f"   Material: {silicon.name}")
    print(f"   Type: {silicon.material_type}")
    print(f"   Cost per module: ${silicon.total_cost:.2f}")
    print(f"   Carbon footprint: {silicon.total_carbon_footprint:.2f} kg CO2-eq")

    manager.update_module_status(
        module=SimulationModule.MATERIAL_SELECTION.value,
        completion_percentage=100,
        completed=True,
        data={
            'materials': [silicon.name],
            'total_cost': silicon.total_cost
        }
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.MATERIAL_SELECTION.value):.0f}%")
    print()

    # 4. Cell Design Module
    print("4. Cell Design Module")
    cell = create_default_monocrystalline_cell()
    print(f"   Technology: {cell.technology}")
    print(f"   Efficiency: {cell.efficiency * 100:.2f}%")
    print(f"   Power output: {cell.power_output:.2f}W")
    print(f"   Vmp: {cell.voltage_at_max_power:.3f}V, Imp: {cell.current_at_max_power:.3f}A")
    print(f"   Fill factor: {cell.fill_factor:.3f}")

    manager.update_module_status(
        module=SimulationModule.CELL_DESIGN.value,
        completion_percentage=100,
        completed=True,
        data={
            'technology': cell.technology,
            'efficiency': cell.efficiency,
            'power': cell.power_output
        }
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.CELL_DESIGN.value):.0f}%")
    print()

    # 5. Cutting Pattern Module
    print("5. Cutting Pattern Module")
    cutting_pattern = CuttingPattern(
        pattern_type="half-cut",
        segments_per_cell=2,
        cutting_loss=0.01,
        efficiency_gain=0.02,
        cost_increase=0.05
    )
    print(f"   Pattern: {cutting_pattern.pattern_type}")
    print(f"   Segments per cell: {cutting_pattern.segments_per_cell}")
    print(f"   Efficiency gain: {cutting_pattern.efficiency_gain * 100:.1f}%")

    manager.update_module_status(
        module=SimulationModule.CUTTING_PATTERN.value,
        completion_percentage=100,
        completed=True,
        data={'pattern': cutting_pattern.pattern_type}
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.CUTTING_PATTERN.value):.0f}%")
    print()

    # 6. Module Engineering
    print("6. Module Engineering Module")
    layout = ModuleLayout(
        cells_in_series=60,
        cells_in_parallel=1,
        bypass_diodes=3,
        rows=10,
        columns=6
    )

    module = Module(
        model_name="Example 330W Mono",
        manufacturer="Example Solar Co.",
        cell=cell,
        layout=layout,
        cutting_pattern=cutting_pattern,
        materials=[silicon],
        rated_power=330.0,
        length=1.65,
        width=0.992,
        thickness=0.035,
        weight=18.5,
        efficiency=0.201,
        warranty_years=25,
        performance_warranty_years=25
    )

    print(f"   Module: {module.model_name}")
    print(f"   Power: {module.rated_power:.0f}W")
    print(f"   Dimensions: {module.length:.2f}m × {module.width:.2f}m")
    print(f"   Area: {module.area:.3f} m²")
    print(f"   Total cells: {module.total_cells}")
    print(f"   Power density: {module.power_density:.1f} W/m²")

    manager.update_module_status(
        module=SimulationModule.MODULE_ENGINEERING.value,
        completion_percentage=100,
        completed=True,
        data={
            'module_name': module.model_name,
            'rated_power': module.rated_power,
            'efficiency': module.efficiency
        }
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.MODULE_ENGINEERING.value):.0f}%")
    print()

    # 7. System Design Module
    print("7. System Design Module")
    location = Location(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=10.0,
        timezone="America/New_York",
        city="New York",
        country="USA"
    )

    system = PVSystem(
        system_name="Example 33kW Rooftop System",
        location=location,
        modules=[module],
        module_quantity=100,
        mounting_type=MountingType.ROOF_MOUNTED,
        tilt_angle=30.0,
        azimuth_angle=180.0,  # South-facing
        dc_capacity=33.0,
        ac_capacity=30.0,
        inverter_efficiency=0.96,
        system_losses=0.14
    )

    print(f"   System: {system.system_name}")
    print(f"   Location: {system.location.city}, {system.location.country}")
    print(f"   Modules: {system.module_quantity} × {module.rated_power:.0f}W")
    print(f"   DC capacity: {system.dc_capacity:.1f} kW")
    print(f"   AC capacity: {system.ac_capacity:.1f} kW")
    print(f"   DC/AC ratio: {system.dc_ac_ratio:.2f}")
    print(f"   Total area: {system.total_module_area:.1f} m²")

    manager.update_module_status(
        module=SimulationModule.SYSTEM_DESIGN.value,
        completion_percentage=100,
        completed=True,
        data={
            'system_name': system.system_name,
            'dc_capacity': system.dc_capacity,
            'module_count': system.module_quantity
        }
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.SYSTEM_DESIGN.value):.0f}%")
    print()

    # 8. Financial Analysis Module
    print("8. Financial Analysis Module")
    financial = FinancialModel(
        system_cost=50000.0,
        module_cost=25000.0,
        inverter_cost=8000.0,
        balance_of_system_cost=12000.0,
        installation_cost=5000.0,
        annual_om_cost=500.0,
        electricity_rate=0.12,
        electricity_rate_escalation=0.02,
        discount_rate=0.06,
        incentives=15000.0,  # 30% ITC
        system_lifetime=25
    )

    annual_energy = 45000.0  # kWh/year estimate
    lcoe = financial.calculate_lcoe(annual_energy)
    npv = financial.calculate_npv(annual_energy)
    payback = financial.calculate_simple_payback(annual_energy)

    print(f"   System cost: ${financial.system_cost:,.0f}")
    print(f"   Incentives: ${financial.incentives:,.0f}")
    print(f"   Net cost: ${financial.system_cost - financial.incentives:,.0f}")
    print(f"   Annual energy: {annual_energy:,.0f} kWh")
    print(f"   LCOE: ${lcoe:.4f}/kWh")
    print(f"   NPV: ${npv:,.0f}")
    print(f"   Simple payback: {payback:.1f} years")

    manager.update_module_status(
        module=SimulationModule.FINANCIAL_ANALYSIS.value,
        completion_percentage=100,
        completed=True,
        data={
            'lcoe': lcoe,
            'npv': npv,
            'payback': payback
        }
    )
    print(f"   ✓ Module completed: {manager.get_completion_percentage(SimulationModule.FINANCIAL_ANALYSIS.value):.0f}%")
    print()

    # 9. Overall Progress
    print("9. Overall Project Progress")
    overall_completion = manager.get_completion_percentage()
    print(f"   Overall completion: {overall_completion:.1f}%")

    summary = manager.export_session_summary()
    print(f"   Completed modules: {sum(1 for m in summary['modules'].values() if m['completed'])}/11")
    print(f"   Total activities logged: {summary['total_activities']}")
    print()

    # 10. Save Project
    print("10. Saving project...")
    project_path = manager.save_project()
    print(f"   Saved to: {project_path}")
    print()

    # 11. List all projects
    print("11. Available projects:")
    projects = manager.list_projects()
    for i, proj in enumerate(projects, 1):
        print(f"   {i}. {proj['project_name']} ({proj['completion_percentage']:.0f}% complete)")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
