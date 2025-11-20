# B08: Diagnostics & Maintenance Management System

## Overview

The B08 module provides comprehensive diagnostics and maintenance management for PV (Photovoltaic) systems. It consists of two main sub-modules:

- **B08-S04**: Fault Reports & Maintenance Recommendations
- **B08-S05**: Diagnostics UI & Defect Management Dashboard

## B08-S04: Fault Reports & Maintenance Recommendations

### Components

#### 1. FaultReportGenerator

Automated fault report generation with intelligent defect analysis.

**Key Features:**
- Automated report generation from defect data
- Defect categorization (structural, electrical, environmental, performance)
- Severity assessment based on power loss and defect characteristics
- Repair cost estimation with customizable rates
- Actionable recommendations

**Example Usage:**

```python
from pv_circularity.b08_diagnostics.b08_s04 import FaultReportGenerator

# Initialize generator
generator = FaultReportGenerator()

# Generate report
report = generator.automated_report_generation(
    site_id="SITE-001",
    defects=detected_defects,
    report_title="Q1 2024 Fault Report"
)

print(f"Total defects: {report.total_defects}")
print(f"Estimated cost: ${report.estimated_total_cost:,.2f}")
print(f"Recommendations: {report.recommendations}")
```

**Methods:**

- `automated_report_generation()`: Generate complete fault report
- `defect_categorization()`: Categorize defects by type
- `severity_assessment()`: Assess defect severity
- `repair_cost_estimation()`: Estimate repair costs

#### 2. MaintenanceScheduler

Intelligent maintenance planning and scheduling system.

**Key Features:**
- Preventive maintenance planning
- Corrective action tracking
- Spare parts management
- Automatic reorder detection
- Seasonal maintenance scheduling

**Example Usage:**

```python
from pv_circularity.b08_diagnostics.b08_s04 import MaintenanceScheduler

# Initialize scheduler
scheduler = MaintenanceScheduler()

# Generate maintenance plan
schedules = scheduler.preventive_maintenance_planning(
    site_id="SITE-001",
    planning_horizon_days=365,
    panel_count=500
)

# Manage spare parts
parts_analysis = scheduler.spare_parts_management(schedules)
print(f"Shortages: {parts_analysis['shortages']}")
print(f"Reorder needed: {parts_analysis['reorder']}")
```

**Methods:**

- `preventive_maintenance_planning()`: Generate maintenance schedules
- `corrective_action_tracking()`: Track corrective actions
- `spare_parts_management()`: Manage spare parts inventory

#### 3. WorkOrderManagement

Complete work order lifecycle management.

**Key Features:**
- Automated work order creation
- Intelligent technician assignment
- Task tracking and status updates
- Completion verification
- Workload balancing

**Example Usage:**

```python
from pv_circularity.b08_diagnostics.b08_s04 import WorkOrderManagement

# Initialize system
wom = WorkOrderManagement()

# Create and assign work order
work_order = wom.create_work_order(maintenance_schedule)

# Track progress
wom.task_tracking(
    work_order_id=work_order.id,
    status=WorkOrderStatus.IN_PROGRESS,
    actual_start=datetime.now()
)

# Verify completion
wom.completion_verification(
    work_order_id=work_order.id,
    verified_by="supervisor@example.com",
    verification_passed=True
)
```

**Methods:**

- `create_work_order()`: Create work order from schedule
- `technician_assignment()`: Assign technician (auto or manual)
- `task_tracking()`: Update work order progress
- `completion_verification()`: Verify completed work

## B08-S05: Diagnostics UI & Defect Management Dashboard

### Components

#### 1. DefectDatabase

Comprehensive defect data management with analytics.

**Key Features:**
- Defect storage and retrieval
- Historical tracking with snapshots
- Pattern recognition using ML (DBSCAN clustering)
- Fleet-wide analysis
- Advanced querying and filtering

**Example Usage:**

```python
from pv_circularity.b08_diagnostics.b08_s05 import DefectDatabase

# Initialize database
db = DefectDatabase()

# Add defects
for defect in detected_defects:
    db.add_defect(defect)

# Query defects
critical_defects = db.query_defects(
    filters={"severity": DefectSeverity.CRITICAL}
)

# Pattern recognition
patterns = db.pattern_recognition(site_id="SITE-001")

# Fleet analysis
analysis = db.fleet_wide_analysis(
    fleet_id="FLEET-NORTHEAST",
    site_ids=["SITE-001", "SITE-002", "SITE-003"]
)

print(f"Fleet health score: {analysis.fleet_health_score}")
```

**Methods:**

- `add_defect()`: Add defect to database
- `defect_history()`: Retrieve historical data
- `pattern_recognition()`: Identify recurring patterns
- `fleet_wide_analysis()`: Analyze multiple sites
- `query_defects()`: Advanced defect queries
- `get_statistics()`: Get comprehensive statistics

#### 2. DiagnosticsUI

Interactive Streamlit dashboard for visualization.

**Key Features:**
- Defect gallery with image visualization
- Interactive heatmaps (spatial, temporal, type vs severity)
- Repair tracking with timelines
- Cost analysis dashboard
- Real-time metrics and KPIs

**Example Usage:**

```python
from pv_circularity.b08_diagnostics.b08_s05 import DiagnosticsUI

# Initialize UI
ui = DiagnosticsUI(defect_database=db)

# Render complete dashboard
ui.render_dashboard(
    defects=defects,
    work_orders=work_orders,
    fault_reports=reports
)

# Or render individual components
ui.defect_gallery(defects=recent_defects)
ui.severity_heatmaps(site_id="SITE-001")
ui.repair_tracking(work_orders=active_orders)
ui.cost_analysis_dashboard(fault_reports=reports)
```

**Methods:**

- `defect_gallery()`: Display defect image gallery
- `severity_heatmaps()`: Show severity heatmaps
- `repair_tracking()`: Track repair progress
- `cost_analysis_dashboard()`: Analyze costs
- `render_dashboard()`: Complete integrated dashboard

## Data Models

### Core Models

All models are based on Pydantic for validation and serialization.

#### Defect
```python
Defect(
    type: DefectType,          # CRACK, HOTSPOT, DELAMINATION, etc.
    severity: DefectSeverity,  # LOW, MEDIUM, HIGH, CRITICAL
    location: Coordinates,      # Spatial coordinates
    confidence: float,          # 0.0-1.0
    panel_id: str,
    estimated_power_loss: float,
    description: Optional[str]
)
```

#### MaintenanceSchedule
```python
MaintenanceSchedule(
    schedule_name: str,
    site_id: str,
    maintenance_type: MaintenanceType,
    priority: MaintenancePriority,
    scheduled_date: date,
    estimated_duration_hours: float,
    required_parts: List[str],
    required_skills: List[str]
)
```

#### WorkOrder
```python
WorkOrder(
    work_order_number: str,
    title: str,
    site_id: str,
    maintenance_type: MaintenanceType,
    status: WorkOrderStatus,
    assigned_technician_id: Optional[str],
    estimated_cost: float,
    actual_cost: float
)
```

## Architecture

```
B08 Diagnostics & Maintenance
├── B08-S04: Fault Reports & Maintenance
│   ├── FaultReportGenerator
│   │   ├── automated_report_generation()
│   │   ├── defect_categorization()
│   │   ├── severity_assessment()
│   │   └── repair_cost_estimation()
│   ├── MaintenanceScheduler
│   │   ├── preventive_maintenance_planning()
│   │   ├── corrective_action_tracking()
│   │   └── spare_parts_management()
│   └── WorkOrderManagement
│       ├── create_work_order()
│       ├── technician_assignment()
│       ├── task_tracking()
│       └── completion_verification()
└── B08-S05: Diagnostics UI & Dashboard
    ├── DefectDatabase
    │   ├── defect_history()
    │   ├── pattern_recognition()
    │   └── fleet_wide_analysis()
    └── DiagnosticsUI
        ├── defect_gallery()
        ├── severity_heatmaps()
        ├── repair_tracking()
        └── cost_analysis_dashboard()
```

## Installation

```bash
# Clone repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_circularity --cov-report=html

# Run specific test file
pytest tests/unit/test_b08/test_fault_report_generator.py
```

## Running the Demo

```bash
# Run the comprehensive demo
python docs/examples/b08_demo.py
```

## Launching the Dashboard

```bash
# Launch Streamlit dashboard (requires separate dashboard script)
streamlit run your_dashboard_script.py
```

## Configuration

### Cost Estimation Configuration

```python
from pv_circularity.b08_diagnostics.b08_s04 import CostEstimationConfig

config = CostEstimationConfig(
    base_labor_rate=75.0,
    panel_replacement_cost=300.0,
    cleaning_cost_per_panel=25.0,
    emergency_multiplier=1.5
)

generator = FaultReportGenerator(cost_config=config)
```

### Maintenance Policy Configuration

```python
from pv_circularity.b08_diagnostics.b08_s04 import MaintenancePolicy

policy = MaintenancePolicy(
    preventive_interval_days=180,
    inspection_interval_days=90,
    cleaning_interval_days=60,
    critical_response_hours=24
)

scheduler = MaintenanceScheduler(policy=policy)
```

### Database Configuration

```python
from pv_circularity.b08_diagnostics.b08_s05 import DatabaseConfig

config = DatabaseConfig(
    storage_backend="memory",
    enable_caching=True,
    pattern_recognition_threshold=0.7,
    clustering_epsilon=0.5
)

db = DefectDatabase(config=config)
```

## API Reference

For detailed API documentation, see:
- [FaultReportGenerator API](./api/fault_report_generator.md)
- [MaintenanceScheduler API](./api/maintenance_scheduler.md)
- [WorkOrderManagement API](./api/work_order_management.md)
- [DefectDatabase API](./api/defect_database.md)
- [DiagnosticsUI API](./api/diagnostics_ui.md)

## Performance Considerations

### DefectDatabase
- Uses in-memory storage by default (fast but not persistent)
- Caching enabled for improved query performance
- Pattern recognition uses DBSCAN (O(n log n) complexity)

### Pattern Recognition
- Performs spatial clustering using DBSCAN
- Temporal pattern detection with time windows
- Characteristic pattern analysis

### Recommendations
- For large datasets (>10,000 defects), consider:
  - Using external database (SQL/NoSQL)
  - Implementing pagination for queries
  - Batch processing for pattern recognition

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **Streamlit Issues**
   ```bash
   # Reinstall Streamlit
   pip install --upgrade streamlit
   ```

3. **NumPy/scikit-learn Errors**
   ```bash
   # Reinstall scientific packages
   pip install --upgrade numpy scikit-learn
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues
- Documentation: https://github.com/ganeshgowri-ASA/pv-circularity-simulator/docs

## Changelog

### Version 0.1.0 (Initial Release)
- B08-S04: Fault Reports & Maintenance Recommendations
  - FaultReportGenerator with automated reporting
  - MaintenanceScheduler with preventive planning
  - WorkOrderManagement with technician assignment
- B08-S05: Diagnostics UI & Defect Management
  - DefectDatabase with pattern recognition
  - DiagnosticsUI with interactive visualizations
- Comprehensive test suite
- Full documentation and examples
