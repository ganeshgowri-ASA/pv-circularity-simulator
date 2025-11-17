"""Example usage of the RepairOptimizer class.

This script demonstrates a complete workflow for PV system maintenance:
1. Setting up the optimizer and inventory
2. Diagnosing faults in components
3. Estimating repair costs
4. Creating an optimized maintenance schedule
5. Managing spare parts inventory
"""

from datetime import datetime, timedelta

from pv_simulator.managers.repair_optimizer import RepairOptimizer
from pv_simulator.models.maintenance import (
    ComponentHealth,
    ComponentStatus,
    MaintenancePriority,
    MaintenanceType,
    RepairTask,
    SparePart,
)


def main():
    """Demonstrate RepairOptimizer usage with a realistic scenario."""
    print("=" * 80)
    print("PV System Repair Optimization - Example Workflow")
    print("=" * 80)
    print()

    # ==================== Step 1: Initialize Optimizer ====================
    print("Step 1: Initializing RepairOptimizer")
    print("-" * 80)

    optimizer = RepairOptimizer(
        labor_rate=85.0,  # $85/hour for skilled technicians
        overhead_rate=0.18,  # 18% overhead
        fault_detection_threshold=0.6,  # 60% confidence threshold
    )

    print(f"✓ Labor rate: ${optimizer.labor_rate}/hour")
    print(f"✓ Overhead rate: {optimizer.overhead_rate * 100}%")
    print(f"✓ Fault detection threshold: {optimizer.fault_detection_threshold}")
    print()

    # ==================== Step 2: Set Up Spare Parts Inventory ====================
    print("Step 2: Setting up spare parts inventory")
    print("-" * 80)

    spare_parts = [
        SparePart(
            part_id="BP-001",
            part_name="Bypass Diode",
            part_number="BD-12V-10A",
            category="electrical",
            quantity_available=75,
            unit_cost=18.50,
            lead_time_days=5,
            reorder_point=30,
            reorder_quantity=100,
            supplier="SolarTech Components",
        ),
        SparePart(
            part_id="JB-001",
            part_name="Junction Box",
            part_number="JB-PV-STD",
            category="electrical",
            quantity_available=40,
            unit_cost=32.00,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=50,
            supplier="SolarTech Components",
        ),
        SparePart(
            part_id="INV-CAP-001",
            part_name="Inverter Capacitor",
            part_number="CAP-450V-150uF",
            category="electrical",
            quantity_available=25,
            unit_cost=65.00,
            lead_time_days=10,
            reorder_point=15,
            reorder_quantity=30,
            supplier="PowerElectronics Inc",
        ),
        SparePart(
            part_id="FAN-001",
            part_name="Cooling Fan 120mm",
            part_number="CF-120-12V-HIGH",
            category="thermal",
            quantity_available=18,
            unit_cost=45.00,
            lead_time_days=7,
            reorder_point=20,
            reorder_quantity=40,
            supplier="ThermalSolutions Ltd",
        ),
        SparePart(
            part_id="PANEL-001",
            part_name="Replacement Panel 300W",
            part_number="PANEL-MONO-300W",
            category="panel",
            quantity_available=8,
            unit_cost=285.00,
            lead_time_days=14,
            reorder_point=5,
            reorder_quantity=10,
            supplier="SunPower Manufacturing",
        ),
    ]

    for part in spare_parts:
        optimizer.add_spare_part(part)
        print(f"✓ Added: {part.part_name} - Qty: {part.quantity_available}")

    print(f"\nTotal parts in inventory: {len(optimizer.spare_parts)}")
    print()

    # ==================== Step 3: Diagnose Faults ====================
    print("Step 3: Diagnosing faults in PV system components")
    print("-" * 80)

    # Scenario: Solar farm with multiple component issues
    diagnosis_results = []

    # Fault 1: Panel with electrical issue
    print("\n[Component: PANEL-A23]")
    fault_1 = optimizer.fault_diagnosis(
        component_id="PANEL-A23",
        component_type="panel",
        performance_data={
            "voltage": 24.5,
            "current": 6.8,
            "efficiency": 0.145,
            "temperature": 47.0,
        },
        baseline_data={
            "voltage": 30.0,
            "current": 8.0,
            "efficiency": 0.18,
            "temperature": 45.0,
        },
    )

    if fault_1:
        print(f"✗ FAULT DETECTED: {fault_1.fault_type.value}")
        print(f"  Severity: {fault_1.severity.value}")
        print(f"  Confidence: {fault_1.diagnosis_confidence:.1%}")
        print(f"  Symptoms: {list(fault_1.symptoms.keys())}")
        print(f"  Root cause: {fault_1.root_cause}")
        diagnosis_results.append(fault_1)

    # Fault 2: Inverter with thermal issue
    print("\n[Component: INV-CENTRAL-01]")
    fault_2 = optimizer.fault_diagnosis(
        component_id="INV-CENTRAL-01",
        component_type="inverter",
        performance_data={
            "efficiency": 0.89,
            "temperature": 82.0,
            "voltage_in": 595.0,
            "voltage_out": 238.0,
        },
        baseline_data={
            "efficiency": 0.96,
            "temperature": 50.0,
            "voltage_in": 600.0,
            "voltage_out": 240.0,
        },
    )

    if fault_2:
        print(f"✗ FAULT DETECTED: {fault_2.fault_type.value}")
        print(f"  Severity: {fault_2.severity.value}")
        print(f"  Confidence: {fault_2.diagnosis_confidence:.1%}")
        print(f"  Symptoms: {list(fault_2.symptoms.keys())}")
        print(f"  Root cause: {fault_2.root_cause}")
        diagnosis_results.append(fault_2)

    # Fault 3: Panel with degradation (using historical data)
    print("\n[Component: PANEL-B17]")
    historical_performance = [
        {"efficiency": 0.180},
        {"efficiency": 0.178},
        {"efficiency": 0.175},
        {"efficiency": 0.172},
        {"efficiency": 0.168},
        {"efficiency": 0.164},
    ]

    fault_3 = optimizer.fault_diagnosis(
        component_id="PANEL-B17",
        component_type="panel",
        performance_data={"efficiency": 0.160},
        baseline_data={"efficiency": 0.18},
        historical_data=historical_performance,
    )

    if fault_3:
        print(f"✗ FAULT DETECTED: {fault_3.fault_type.value}")
        print(f"  Severity: {fault_3.severity.value}")
        print(f"  Confidence: {fault_3.diagnosis_confidence:.1%}")
        print(f"  Symptoms: {list(fault_3.symptoms.keys())}")
        print(f"  Root cause: {fault_3.root_cause}")
        diagnosis_results.append(fault_3)

    print(f"\nTotal faults detected: {len(diagnosis_results)}")
    print()

    # ==================== Step 4: Estimate Repair Costs ====================
    print("Step 4: Estimating repair costs")
    print("-" * 80)

    repair_estimates = []
    for fault in diagnosis_results:
        estimate = optimizer.repair_cost_estimation(fault, include_parts=True)
        repair_estimates.append(estimate)

        print(f"\n[Repair Estimate for {fault.component_id}]")
        print(f"  Labor: {estimate.labor_hours:.1f} hours @ ${estimate.labor_rate}/hr = ${estimate.labor_cost:.2f}")
        print(f"  Parts: ${estimate.parts_cost:.2f}")
        if estimate.parts_breakdown:
            for part_name, cost in estimate.parts_breakdown.items():
                print(f"    - {part_name}: ${cost:.2f}")
        print(f"  Overhead: ${estimate.overhead_cost:.2f}")
        print(f"  TOTAL: ${estimate.total_cost:.2f}")
        print(f"  Confidence: {estimate.confidence_level:.1%}")

    total_cost = sum(e.total_cost for e in repair_estimates)
    print(f"\nTotal estimated repair cost: ${total_cost:.2f}")
    print()

    # ==================== Step 5: Create Repair Tasks ====================
    print("Step 5: Creating repair tasks")
    print("-" * 80)

    repair_tasks = []

    # Map faults to repair tasks with appropriate priorities
    task_configs = [
        (fault_1, MaintenancePriority.HIGH, ["electrical", "pv_systems"]),
        (fault_2, MaintenancePriority.EMERGENCY, ["electrical", "inverters", "thermal"]),
        (fault_3, MaintenancePriority.MEDIUM, ["electrical", "pv_systems"]),
    ]

    for i, (fault, priority, skills) in enumerate(task_configs):
        if fault:
            estimate = repair_estimates[i]
            task = RepairTask(
                fault_id=fault.fault_id,
                component_id=fault.component_id,
                task_type=MaintenanceType.CORRECTIVE,
                priority=priority,
                estimated_duration_hours=estimate.labor_hours,
                required_skills=skills,
                estimated_cost=estimate.total_cost,
                notes=f"Repair {fault.fault_type.value} fault",
            )
            repair_tasks.append(task)
            print(f"✓ Created task for {fault.component_id} - Priority: {priority.name}")

    print(f"\nTotal tasks created: {len(repair_tasks)}")
    print()

    # ==================== Step 6: Optimize Maintenance Schedule ====================
    print("Step 6: Creating optimized maintenance schedule")
    print("-" * 80)

    start_date = datetime.now()
    end_date = start_date + timedelta(days=21)

    schedule = optimizer.maintenance_scheduling(
        tasks=repair_tasks,
        start_date=start_date,
        end_date=end_date,
        max_daily_hours=8.0,
        optimization_objective="maximize_priority",
    )

    print(f"\nSchedule created: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Optimization objective: {schedule.optimization_objective}")
    print(f"Constraints satisfied: {schedule.constraints_satisfied}")
    print(f"Total estimated cost: ${schedule.total_estimated_cost:.2f}")
    print(f"Total estimated hours: {schedule.total_estimated_hours:.1f}")
    print(f"\nScheduled tasks:")

    for i, task in enumerate(schedule.tasks, 1):
        print(f"\n  Task {i}: {task.component_id}")
        print(f"    Priority: {task.priority.name}")
        print(f"    Duration: {task.estimated_duration_hours:.1f} hours")
        print(f"    Cost: ${task.estimated_cost:.2f}")
        if task.scheduled_start:
            print(f"    Scheduled: {task.scheduled_start.strftime('%Y-%m-%d %H:%M')}")
        print(f"    Skills required: {', '.join(task.required_skills)}")

    print()

    # ==================== Step 7: Spare Parts Management ====================
    print("Step 7: Managing spare parts inventory")
    print("-" * 80)

    parts_report = optimizer.spare_parts_management(
        check_inventory=True, generate_reorder_list=True, forecast_days=90
    )

    print("\nInventory Status:")
    for part_id, status in parts_report["inventory_status"].items():
        part = optimizer.spare_parts[part_id]
        status_symbol = "⚠" if status["needs_reorder"] else "✓"
        print(
            f"  {status_symbol} {status['part_name']}: "
            f"{status['quantity_on_hand']} available "
            f"(Reorder point: {status['reorder_point']})"
        )

    if parts_report["critical_shortages"]:
        print("\n⚠ CRITICAL SHORTAGES:")
        for shortage in parts_report["critical_shortages"]:
            print(f"  - {shortage['part_name']}: OUT OF STOCK")
            print(f"    Lead time: {shortage['lead_time_days']} days")

    if parts_report["reorder_recommendations"]:
        print("\nReorder Recommendations:")
        for rec in parts_report["reorder_recommendations"]:
            print(f"  • {rec['part_name']}")
            print(f"    Part #: {rec['part_number']}")
            print(f"    Current qty: {rec['current_quantity']}")
            print(f"    Reorder qty: {rec['reorder_quantity']}")
            print(f"    Cost: ${rec['total_cost']:.2f}")
            print(f"    Supplier: {rec['supplier']}")
            print(f"    Reason: {rec['reason']}")
            print()

        print(f"Total reorder cost: ${parts_report['total_reorder_cost']:.2f}")

    print()

    # ==================== Summary ====================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Faults diagnosed: {len(diagnosis_results)}")
    print(f"Active faults: {len(optimizer.get_active_faults())}")
    print(f"Repairs scheduled: {len(schedule.tasks)}")
    print(f"Total repair cost: ${schedule.total_estimated_cost:.2f}")
    print(f"Total repair time: {schedule.total_estimated_hours:.1f} hours")
    print(f"Spare parts requiring reorder: {len(parts_report['reorder_recommendations'])}")
    print(f"Total reorder cost: ${parts_report['total_reorder_cost']:.2f}")
    print()
    print("✓ Workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
