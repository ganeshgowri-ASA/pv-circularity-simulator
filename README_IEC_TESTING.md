# IEC Testing Results & Reporting Dashboard (BATCH4-B04-S04)

## Overview

Comprehensive IEC testing results management and interactive reporting dashboard for all IEC test standards including IEC 61215 (Module Qualification), IEC 61730 (Safety), IEC 63202 (CTM Power Loss), IEC 63209, and IEC 63279.

## Features

### 1. IECTestResultsManager
- **load_test_results()**: Import test results from B04-S01, B04-S02, B04-S03
- **aggregate_compliance_status()**: Determine overall compliance across all IEC standards
- **compare_to_standards()**: Benchmark test results against IEC requirements
- **generate_compliance_matrix()**: Create pass/fail matrix for all test sequences
- **export_test_package()**: Export complete test documentation package (JSON, Excel, XML)
- **track_test_history()**: Maintain historical test data for trend analysis

### 2. TestReportGenerator
- **generate_iec61215_report()**: Comprehensive IEC 61215 qualification report with all MQT test results
- **generate_iec61730_report()**: Safety qualification report with electrical/fire/mechanical safety results
- **generate_iec63202_report()**: CTM power loss report with detailed loss breakdown
- **generate_combined_report()**: Multi-standard combined certification report
- **add_test_photos()**: Integrate test photos and visual evidence
- **add_certification_signatures()**: Digital signature fields for certification bodies

### 3. ComplianceVisualization
- **test_results_dashboard()**: Interactive Streamlit dashboard with all test results
- **pass_fail_summary()**: Visual summary of compliance status across all tests
- **degradation_timeline_chart()**: Power degradation over test sequences
- **iv_curve_comparison()**: Before/after IV curves for each test sequence
- **failure_mode_analysis()**: Visual analysis of any test failures
- **ctm_loss_waterfall()**: Waterfall chart of CTM power losses

### 4. CertificationWorkflow
- **prepare_certification_package()**: Package all documentation for TÜV, UL, IEC CB, etc.
- **track_certification_status()**: Monitor certification application status
- **manage_certification_costs()**: Track certification fees and timeline
- **handle_recertification()**: Manage periodic recertification requirements
- **international_certification_mapping()**: Map IEC to local standards (UL, CSA, JET, CQC)

### 5. Streamlit Dashboard (IECTestingUI)
- Multi-tab interface: IEC 61215 | IEC 61730 | IEC 63202 | Combined View | Certification | Export
- Interactive test result explorer with filtering and search
- Real-time compliance status indicators (pass/fail badges)
- Test result comparison tools (multiple modules/designs)
- PDF report generation with custom branding
- Export functionality (Excel, JSON, XML)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using the package
pip install -e .
```

## Quick Start

### Running IEC Tests

```python
from src.iec_testing.B04_S01_iec61215 import IEC61215Tester
from src.iec_testing.B04_S02_iec61730 import IEC61730Tester
from src.iec_testing.B04_S03_iec63202 import IEC63202Tester

# Run IEC 61215 qualification
tester_61215 = IEC61215Tester()
result_61215 = tester_61215.run_full_qualification(
    module_id="MODULE_001",
    module_type="PV-400W-PERC",
    manufacturer="Solar Inc.",
    test_campaign_id="TC-2025-001"
)

# Run IEC 61730 safety
tester_61730 = IEC61730Tester()
result_61730 = tester_61730.run_full_safety_qualification(
    module_id="MODULE_001",
    module_type="PV-400W-PERC",
    manufacturer="Solar Inc.",
    test_campaign_id="TC-2025-001"
)

# Run IEC 63202 CTM analysis
tester_63202 = IEC63202Tester()
result_63202 = tester_63202.run_full_ctm_analysis(
    module_id="MODULE_001",
    module_type="PV-400W-PERC",
    manufacturer="Solar Inc.",
    test_campaign_id="TC-2025-001"
)
```

### Managing Test Results

```python
from src.iec_testing.B04_S04_reporting_dashboard import IECTestResultsManager

# Initialize manager
manager = IECTestResultsManager()

# Load test results
manager.load_test_results(
    result_61215=result_61215,
    result_61730=result_61730,
    result_63202=result_63202
)

# Generate compliance matrix
matrix = manager.generate_compliance_matrix()
print(f"Overall compliance: {matrix.overall_compliance}")
print(f"Compliance rate: {matrix.compliance_rate:.1f}%")

# Export test package
from pathlib import Path
exported = manager.export_test_package(
    output_dir=Path("./exports"),
    format="excel"
)
```

### Generating Reports

```python
from src.iec_testing.B04_S04_reporting_dashboard import TestReportGenerator
from pathlib import Path

# Initialize generator
generator = TestReportGenerator(company_name="My Test Lab")

# Generate IEC 61215 report
generator.generate_iec61215_report(
    result=result_61215,
    output_path=Path("./reports/iec_61215_report.pdf")
)

# Generate combined report
generator.generate_combined_report(
    result_61215=result_61215,
    result_61730=result_61730,
    result_63202=result_63202,
    output_path=Path("./reports/combined_report.pdf")
)
```

### Creating Visualizations

```python
from src.iec_testing.B04_S04_reporting_dashboard import ComplianceVisualization

# Initialize visualization
viz = ComplianceVisualization()

# Create pass/fail summary
fig = viz.pass_fail_summary(matrix)
fig.show()

# Create IV curve comparison
fig = viz.iv_curve_comparison(
    iv_initial=result_61215.test_sequence.iv_curve_initial,
    iv_final=result_61215.test_sequence.iv_curve_final
)
fig.show()

# Create CTM loss waterfall
fig = viz.ctm_loss_waterfall(result_63202.ctm_loss_breakdown)
fig.show()
```

### Running Streamlit Dashboard

```bash
# Start the dashboard
streamlit run src/ui/streamlit_app.py
```

Then navigate to http://localhost:8501 in your browser.

## Data Models

All data validation is handled through Pydantic models:

- **IEC61215Result**: Complete IEC 61215 qualification results
- **IEC61730Result**: Complete IEC 61730 safety qualification results
- **IEC63202Result**: Complete IEC 63202 CTM power loss results
- **ComplianceMatrix**: Pass/fail matrix for all test sequences
- **ComplianceReport**: Comprehensive compliance report across all standards
- **CertificationPackage**: Complete certification package for submission
- **TestHistory**: Historical test data for trend analysis

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_B04_S04.py -v

# Run with coverage
pytest tests/test_B04_S04.py --cov=src/iec_testing --cov-report=html
```

## IEC Standards Coverage

### IEC 61215 (Module Qualification Testing)
- Visual inspection (initial and final)
- Performance at STC
- Wet leakage current test
- Thermal cycling (200 cycles, -40°C to +85°C)
- Humidity-freeze test (10 cycles)
- Damp heat test (1000 hours at 85°C/85%RH)
- UV preconditioning
- Mechanical load test (2400 Pa)
- Hail impact test (25mm ice balls)
- Hot spot endurance test

### IEC 61730 (Safety Qualification)
- Insulation resistance test
- Dielectric withstand test
- Ground continuity test
- Fire resistance test
- Mechanical stress test
- Impact resistance test
- UV exposure test
- Corrosion resistance test

### IEC 63202 (CTM Power Loss)
- Optical loss analysis
- Electrical loss analysis
- Thermal loss characterization
- Cell mismatch analysis
- Interconnection loss measurement
- Inactive area loss calculation

## Certification Bodies Supported

- TÜV Rheinland
- TÜV SÜD
- UL (Underwriters Laboratories)
- IEC CB Scheme
- CSA (Canadian Standards Association)
- JET (Japan Electrical Safety & Environment Technology Laboratories)
- CQC (China Quality Certification Centre)
- VDE
- Intertek
- SGS

## Export Formats

- **JSON**: Complete test data in JSON format
- **Excel**: Formatted spreadsheets with charts
- **PDF**: Professional certification reports
- **XML**: For certification portal integration

## Architecture

```
src/iec_testing/
├── models/
│   └── test_models.py          # Pydantic data models
├── B04_S01_iec61215.py         # IEC 61215 testing
├── B04_S02_iec61730.py         # IEC 61730 testing
├── B04_S03_iec63202.py         # IEC 63202 testing
└── B04_S04_reporting_dashboard.py  # Main reporting module

src/ui/
└── streamlit_app.py            # Streamlit dashboard

tests/
└── test_B04_S04.py            # Comprehensive test suite
```

## Contributing

This module follows strict production-ready standards:
- Full docstrings on every function
- Complete type hints for all methods
- Pydantic models for all data validation
- Comprehensive error handling and logging
- Unit test coverage for all functionality

## License

MIT License - See LICENSE file for details

## Contact

For support or questions, please open an issue on the GitHub repository.
