"""
Demo script for B08-S04 and B08-S05 components.

This script demonstrates the usage of fault reporting, maintenance scheduling,
work order management, defect database, and diagnostics UI.
"""

from datetime import datetime, date, timedelta
from pathlib import Path

from src.pv_circularity.models import (
    Defect,
    DefectType,
    DefectSeverity,
    Coordinates,
    Technician,
    SparePart,
)
from src.pv_circularity.b08_diagnostics.b08_s04 import (
    FaultReportGenerator,
    MaintenanceScheduler,
    WorkOrderManagement,
)
from src.pv_circularity.b08_diagnostics.b08_s05 import (
    DefectDatabase,
    DiagnosticsUI,
)


def create_sample_defects():
    """Create sample defects for demonstration."""
    defects = []

    defect_configs = [
        (DefectType.CRACK, DefectSeverity.HIGH, 100, 150, 8.0, "Significant crack in cell"),
        (DefectType.HOTSPOT, DefectSeverity.CRITICAL, 200, 250, 15.0, "Critical hotspot detected"),
        (DefectType.DELAMINATION, DefectSeverity.MEDIUM, 150, 200, 6.0, "Edge delamination"),
        (DefectType.SOILING, DefectSeverity.LOW, 300, 100, 2.0, "Heavy soiling layer"),
        (DefectType.PID, DefectSeverity.HIGH, 250, 300, 12.0, "Potential induced degradation"),
    ]

    for idx, (d_type, severity, x, y, power_loss, desc) in enumerate(defect_configs):
        defect = Defect(
            type=d_type,
            severity=severity,
            location=Coordinates(x=x, y=y, width=50, height=50),
            confidence=0.85,
            panel_id=f"PANEL-{idx+1:03d}",
            module_id=f"MODULE-{(idx // 10)+1:02d}",
            string_id=f"STRING-{(idx // 50)+1:02d}",
            description=desc,
            estimated_power_loss=power_loss,
        )
        defects.append(defect)

    return defects


def demo_fault_report_generation():
    """Demonstrate fault report generation."""
    print("=" * 80)
    print("DEMO 1: Fault Report Generation")
    print("=" * 80)

    # Create sample defects
    defects = create_sample_defects()
    print(f"Created {len(defects)} sample defects")

    # Initialize fault report generator
    generator = FaultReportGenerator()

    # Generate automated fault report
    report = generator.automated_report_generation(
        site_id="SITE-DEMO-001",
        defects=defects,
        report_title="Demonstration Fault Report"
    )

    print(f"\nGenerated Report: {report.report_title}")
    print(f"Total Defects: {report.total_defects}")
    print(f"Critical Defects: {report.critical_defects}")
    print(f"Estimated Total Cost: ${report.estimated_total_cost:,.2f}")
    print(f"Estimated Power Loss: {report.estimated_power_loss:.2f}%")

    print("\nRecommendations:")
    for idx, rec in enumerate(report.recommendations, 1):
        print(f"  {idx}. {rec}")

    print("\nDiagnostic Summary:")
    for diag in report.diagnostics[:3]:  # Show first 3
        print(f"  - {diag.defect.type.value}: {diag.recommended_action.value}")
        print(f"    Cost: ${diag.estimated_cost:.2f}, Priority: {diag.priority}")

    return report, defects


def demo_maintenance_scheduling():
    """Demonstrate maintenance scheduling."""
    print("\n" + "=" * 80)
    print("DEMO 2: Maintenance Scheduling")
    print("=" * 80)

    # Initialize scheduler
    scheduler = MaintenanceScheduler()

    # Generate preventive maintenance plan
    schedules = scheduler.preventive_maintenance_planning(
        site_id="SITE-DEMO-001",
        planning_horizon_days=365,
        panel_count=500,
    )

    print(f"\nGenerated {len(schedules)} maintenance schedules for the next year")

    # Show schedule summary by type
    from collections import Counter
    schedule_types = Counter(s.maintenance_type.value for s in schedules)

    print("\nSchedule Breakdown:")
    for sched_type, count in schedule_types.items():
        print(f"  {sched_type}: {count} scheduled activities")

    # Show next 5 upcoming schedules
    upcoming = sorted(schedules, key=lambda s: s.scheduled_date)[:5]
    print("\nNext 5 Upcoming Maintenance Activities:")
    for schedule in upcoming:
        print(f"  - {schedule.scheduled_date}: {schedule.schedule_name}")
        print(f"    Type: {schedule.maintenance_type.value}, Duration: {schedule.estimated_duration_hours}h")

    # Add spare parts
    parts = [
        SparePart(
            part_number="PANEL-300W",
            part_name="300W Solar Panel",
            category="panel",
            quantity_available=10,
            reorder_level=5,
            unit_cost=300.0,
        ),
        SparePart(
            part_number="CONNECTOR-MC4",
            part_name="MC4 Connector",
            category="electrical",
            quantity_available=50,
            reorder_level=20,
            unit_cost=5.0,
        ),
    ]

    for part in parts:
        scheduler.add_spare_part(part)

    # Analyze spare parts requirements
    parts_analysis = scheduler.spare_parts_management(schedules)

    print("\nSpare Parts Analysis:")
    print(f"  Parts Required: {len(parts_analysis['required'])} types")
    if parts_analysis['shortages']:
        print(f"  Shortages Detected: {list(parts_analysis['shortages'].keys())}")
    if parts_analysis['reorder']:
        print(f"  Reorder Needed: {list(parts_analysis['reorder'].keys())}")

    return scheduler, schedules


def demo_work_order_management():
    """Demonstrate work order management."""
    print("\n" + "=" * 80)
    print("DEMO 3: Work Order Management")
    print("=" * 80)

    # Initialize work order management
    wom = WorkOrderManagement()

    # Create sample technicians
    technicians = [
        Technician(
            technician_id="TECH-001",
            name="John Smith",
            skills=["electrical_maintenance", "safety_certified"],
            availability=True,
            hourly_rate=75.0,
        ),
        Technician(
            technician_id="TECH-002",
            name="Jane Doe",
            skills=["mechanical_maintenance", "electrical_maintenance", "safety_certified"],
            availability=True,
            hourly_rate=85.0,
        ),
    ]

    for tech in technicians:
        wom.add_technician(tech)

    print(f"Added {len(technicians)} technicians to the system")

    # Create maintenance schedules from previous demo
    from src.pv_circularity.models import MaintenanceSchedule, MaintenanceType, MaintenancePriority

    schedule = MaintenanceSchedule(
        schedule_name="Emergency Panel Replacement",
        site_id="SITE-DEMO-001",
        maintenance_type=MaintenanceType.CORRECTIVE,
        priority=MaintenancePriority.CRITICAL,
        scheduled_date=date.today() + timedelta(days=1),
        estimated_duration_hours=4.0,
        required_skills=["electrical_maintenance", "safety_certified"],
    )

    # Create work order
    work_order = wom.create_work_order(schedule, auto_assign=True)

    print(f"\nCreated Work Order: {work_order.work_order_number}")
    print(f"  Status: {work_order.status.value}")
    print(f"  Assigned to: {work_order.assigned_technician_id}")
    print(f"  Estimated Cost: ${work_order.estimated_cost:.2f}")

    # Start work
    wom.task_tracking(
        work_order_id=work_order.id,
        status=work_order.status.__class__.IN_PROGRESS,
        actual_start=datetime.utcnow(),
        notes="Started panel replacement"
    )

    print("\n  Work started...")

    # Complete work
    wom.task_tracking(
        work_order_id=work_order.id,
        status=work_order.status.__class__.COMPLETED,
        actual_end=datetime.utcnow() + timedelta(hours=3.5),
        actual_cost=425.50,
        notes="Panel successfully replaced and tested"
    )

    print("  Work completed")

    # Verify completion
    wom.completion_verification(
        work_order_id=work_order.id,
        verified_by="supervisor@example.com",
        verification_passed=True,
        verification_notes="All work completed to standard"
    )

    print("  Work verified")

    # Get work order status
    status = wom.get_work_order_status(work_order.id)
    print(f"\nFinal Status:")
    print(f"  Status: {status['status']}")
    print(f"  Duration: {status['duration_actual_hours']:.2f}h")
    print(f"  Actual Cost: ${status['actual_cost']:.2f}")
    print(f"  Verified: {status['verification_status']}")

    return wom


def demo_defect_database():
    """Demonstrate defect database."""
    print("\n" + "=" * 80)
    print("DEMO 4: Defect Database & Analytics")
    print("=" * 80)

    # Initialize database
    db = DefectDatabase()

    # Add sample defects
    defects = create_sample_defects()
    for defect in defects:
        db.add_defect(defect)

    print(f"Added {len(defects)} defects to database")

    # Get statistics
    stats = db.get_statistics()

    print("\nDatabase Statistics:")
    print(f"  Total Defects: {stats['total_defects']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2%}")
    print(f"  Average Power Loss: {stats['average_power_loss']:.2f}%")

    print("\n  Defects by Type:")
    for d_type, count in stats['by_type'].items():
        print(f"    {d_type}: {count}")

    print("\n  Defects by Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"    {severity}: {count}")

    # Pattern recognition
    patterns = db.pattern_recognition()

    print(f"\nIdentified {len(patterns)} defect patterns")
    for pattern in patterns[:3]:  # Show first 3
        print(f"  - {pattern.pattern_name}")
        print(f"    Frequency: {pattern.frequency}, Correlation: {pattern.correlation_score:.2f}")

    # Fleet-wide analysis
    analysis = db.fleet_wide_analysis(
        fleet_id="FLEET-DEMO",
        site_ids=["SITE-DEMO-001", "SITE-DEMO-002"],
    )

    print(f"\nFleet Analysis:")
    print(f"  Fleet Health Score: {analysis.fleet_health_score:.1f}/100")
    print(f"  Total Defects: {analysis.total_defects}")

    return db


def main():
    """Run all demos."""
    print("\n")
    print("#" * 80)
    print("# PV CIRCULARITY SIMULATOR - B08 DIAGNOSTICS & MAINTENANCE DEMO")
    print("#" * 80)
    print("\nThis demo showcases the B08-S04 and B08-S05 components:")
    print("  - B08-S04: Fault Reports & Maintenance Recommendations")
    print("  - B08-S05: Diagnostics UI & Defect Management Dashboard")

    # Run demos
    report, defects = demo_fault_report_generation()
    scheduler, schedules = demo_maintenance_scheduling()
    wom = demo_work_order_management()
    db = demo_defect_database()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nAll B08-S04 and B08-S05 components demonstrated successfully!")
    print("\nNext Steps:")
    print("  1. Run tests: pytest tests/")
    print("  2. Launch Streamlit dashboard: streamlit run [dashboard_script.py]")
    print("  3. Explore the API documentation in docs/")


if __name__ == "__main__":
    main()
