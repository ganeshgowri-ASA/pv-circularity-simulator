# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, circular economy modeling, **AI-powered defect detection**, and **intelligent alerting system**.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### 🔬 AI-Powered Defect Detection (BATCH6-B08-S01)

Production-ready defect detection system using Roboflow AI and computer vision:

- **RoboflowIntegrator**: AI-powered defect detection
  - `model_deployment()`: Deploy and manage Roboflow models
  - `image_preprocessing()`: Advanced image preprocessing pipeline
  - `batch_inference()`: High-throughput batch processing
  - `real_time_detection()`: Low-latency real-time detection

- **DefectClassifier**: Specialized defect classification
  - `crack_detection()`: Micro-crack and cell breakage detection
  - `hotspot_identification()`: Thermal hotspot analysis
  - `delamination_detection()`: Layer separation detection
  - `soiling_analysis()`: Dirt and debris detection
  - `pid_detection()`: Potential-Induced Degradation (PID) analysis

- **ImageProcessing**: Multi-modal image analysis
  - `el_image_analysis()`: Electroluminescence imaging
  - `thermal_image_processing()`: Thermal/infrared analysis
  - `rgb_analysis()`: Visual inspection and soiling detection

- **DefectReporting**: Comprehensive reporting system
  - `severity_classification()`: AI-powered severity assessment
  - `location_mapping()`: GPS and grid-based location tracking
  - `repair_recommendations()`: Automated repair planning with cost estimates

**Technologies**: Roboflow API, OpenCV, scikit-image, Pydantic, asyncio

### 🔔 Alerts & Notifications System (BATCH5-B07-S04)

Production-ready intelligent alerting with multi-channel notifications:

- **AlertManager**: Intelligent alert orchestration
  - `rule_engine()`: Custom rule-based alert triggering
  - `threshold_monitoring()`: Real-time threshold monitoring
  - `anomaly_detection()`: Statistical anomaly detection (Z-score, IQR)
  - `escalation_workflows()`: Multi-level escalation automation

- **NotificationChannels**: Multi-channel delivery
  - `email_alerts()`: SMTP and SendGrid integration
  - `sms_notifications()`: Twilio SMS delivery
  - `slack_webhooks()`: Slack workspace integration
  - `mobile_push()`: Push notification support (placeholder)

- **AlertHistory**: Comprehensive tracking
  - `alert_log()`: Persistent alert logging (SQLite)
  - `acknowledgment_tracking()`: User acknowledgment tracking
  - `resolution_tracking()`: Issue resolution workflow

**Technologies**: Twilio, SendGrid, Slack SDK, SQLite, asyncio

## 📦 Installation

### Requirements

- Python 3.9+
- pip or poetry

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Or install with optional features
pip install -e ".[all]"

# Copy environment configuration
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### Optional Dependencies

```bash
# Defect detection only
pip install -e ".[roboflow]"

# Notifications only
pip install -e ".[notifications]"

# Database support
pip install -e ".[database]"

# Development tools
pip install -r requirements-dev.txt
```

## 🔧 Configuration

Configuration is managed through environment variables and `.env` file:

```bash
# Roboflow AI Configuration
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=pv-defect-detection
ROBOFLOW_MODEL_VERSION=1

# Email Alerts (SMTP)
SMTP_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=alerts@pv-simulator.com

# Twilio SMS
TWILIO_ENABLED=false
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1234567890

# Slack Webhooks
SLACK_ENABLED=false
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Alert Configuration
ALERT_ENABLED=true
ALERT_MIN_SEVERITY=WARNING
MONITORING_ENABLED=true
```

## 💻 Usage Examples

### Defect Detection

```python
from pv_simulator.monitoring import (
    RoboflowIntegrator,
    DefectClassifier,
    ImageProcessing,
    DefectReporting
)

# Initialize components
roboflow = RoboflowIntegrator()
classifier = DefectClassifier(roboflow)
processor = ImageProcessing()
reporter = DefectReporting()

# Analyze EL image for defects
el_results = processor.el_image_analysis("panel_el.jpg", segment_cells=True)
print(f"Quality score: {el_results['metadata'].quality_score:.2f}")
print(f"Found {len(el_results['cells'])} cells")

# Detect cracks using AI
cracks = classifier.crack_detection("panel.jpg", use_ai=True)
for crack in cracks:
    print(f"Crack detected: severity={crack.severity.value}, "
          f"confidence={crack.confidence:.2f}")

# Detect hotspots in thermal image
hotspots = classifier.hotspot_identification("thermal.jpg", threshold_celsius=85)
print(f"Found {len(hotspots)} hotspots")

# Generate comprehensive defect report
if cracks:
    report = reporter.generate_report(
        cracks[0],
        panel_id="PNL-001",
        array_position=(5, 10),
        gps_coords=(37.7749, -122.4194),
        context={"panel_age_years": 10, "high_humidity": True}
    )
    print(f"Report ID: {report.report_id}")
    print(f"Repair priority: {report.recommendation.priority.value}")
    print(f"Estimated cost: ${report.recommendation.estimated_cost:.2f}")
    print(f"Estimated time: {report.recommendation.estimated_time:.1f} hours")
```

### Thermal Analysis

```python
from pv_simulator.monitoring import ImageProcessing

processor = ImageProcessing()

# Process thermal image
thermal_result = processor.thermal_image_processing(
    "thermal_panel.jpg",
    temp_min=0,
    temp_max=120
)

print(f"Temperature range: {thermal_result.min_temp:.1f}°C - {thermal_result.max_temp:.1f}°C")
print(f"Average temperature: {thermal_result.mean_temp:.1f}°C")
print(f"Hotspots detected: {thermal_result.hotspot_count}")
```

### Alert Management

```python
from pv_simulator.alerts import (
    AlertManager,
    AlertRule,
    NotificationChannels,
    AlertHistory
)
from pv_simulator.config import AlertSeverity

# Initialize alert system
manager = AlertManager()
notifier = NotificationChannels()
history = AlertHistory()

# Register alert handler
def handle_alert(alert):
    """Custom alert handler."""
    print(f"Alert: {alert.title} - {alert.severity.value}")

    # Log to history
    history.alert_log(alert)

    # Send notifications
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
        notifier.send_multi_channel(
            alert.title,
            alert.message,
            channels=["email", "sms", "slack"],
            severity=alert.severity.value,
            metadata=alert.context
        )

manager.register_handler(handle_alert)

# Create threshold rule
manager.threshold_monitoring(
    "panel_efficiency",
    "<",
    0.15,
    severity=AlertSeverity.HIGH,
    window_size=3
)

# Create custom rule
rule = AlertRule(
    name="Critical Temperature",
    description="Panel temperature exceeds safe limits",
    condition=lambda ctx: ctx.get("temperature", 0) > 90,
    severity=AlertSeverity.CRITICAL
)
manager.rule_engine(rule)

# Enable anomaly detection
manager.anomaly_detection(
    "power_output",
    method="zscore",
    sensitivity=0.95
)

# Check metrics
metrics = {
    "panel_efficiency": 0.14,
    "temperature": 92,
    "power_output": 245
}
alerts = manager.check_thresholds(metrics)
alerts += manager.check_anomalies(metrics)
alerts += manager.evaluate_rules(metrics)

print(f"Triggered {len(alerts)} alerts")
```

### Escalation Workflows

```python
from pv_simulator.alerts import AlertManager, EscalationLevel
from pv_simulator.config import AlertSeverity

manager = AlertManager()

# Configure multi-level escalation
manager.escalation_workflows([
    EscalationLevel(
        level=1,
        delay_seconds=300,  # 5 minutes
        severity_threshold=AlertSeverity.WARNING,
        notification_channels=["email"]
    ),
    EscalationLevel(
        level=2,
        delay_seconds=600,  # 10 minutes
        severity_threshold=AlertSeverity.HIGH,
        notification_channels=["email", "sms"]
    ),
    EscalationLevel(
        level=3,
        delay_seconds=900,  # 15 minutes
        severity_threshold=AlertSeverity.CRITICAL,
        notification_channels=["email", "sms", "slack"]
    )
])

# Trigger critical alert
alert = manager.trigger_alert(
    "Panel Failure Detected",
    "Critical crack detected in panel PNL-123",
    AlertSeverity.CRITICAL,
    source="defect_detector",
    context={"panel_id": "PNL-123", "defect_type": "crack"}
)
# Escalation will automatically occur if not acknowledged
```

### Alert History and Analytics

```python
from pv_simulator.alerts import AlertHistory
from datetime import datetime, timedelta

history = AlertHistory()

# Acknowledge alert
history.acknowledgment_tracking(
    "alert-123",
    acknowledged_by="operator@example.com",
    notes="Investigating panel"
)

# Resolve alert
history.resolution_tracking(
    "alert-123",
    resolved_by="technician@example.com",
    resolution_method="Panel replaced",
    resolution_notes="Replaced panel PNL-123 with new unit"
)

# Get statistics
stats = history.get_statistics(
    start_time=datetime.utcnow() - timedelta(days=30)
)

print(f"Total alerts (30 days): {stats.total_alerts}")
print(f"Critical alerts: {stats.alerts_by_severity.get('CRITICAL', 0)}")
print(f"Resolved: {stats.resolved_count}")
print(f"Average time to acknowledge: {stats.average_time_to_acknowledge:.1f}s")
print(f"Average time to resolve: {stats.average_time_to_resolve:.1f}s")

# Get unresolved critical alerts
critical_alerts = history.get_alerts(
    severity=AlertSeverity.CRITICAL,
    resolved=False,
    limit=50
)
print(f"Unresolved critical alerts: {len(critical_alerts)}")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pv_simulator --cov-report=html

# Run specific test file
pytest tests/unit/test_alert_manager.py

# Run with verbose output
pytest -v
```

## 📊 Architecture

```
pv-circularity-simulator/
├── src/pv_simulator/
│   ├── config/              # Configuration management
│   │   └── settings.py      # Pydantic settings
│   ├── monitoring/          # Defect detection
│   │   └── detectors/
│   │       ├── roboflow_integrator.py    # AI integration
│   │       ├── defect_classifier.py      # Defect classification
│   │       ├── image_processing.py       # Image analysis
│   │       └── defect_reporting.py       # Reporting system
│   └── alerts/              # Alert & notification system
│       ├── manager.py       # Alert orchestration
│       ├── history.py       # Tracking & analytics
│       └── handlers/
│           └── notification_channels.py  # Multi-channel delivery
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
└── docs/                   # Documentation

```

## 🔑 Key Technologies

- **AI/ML**: Roboflow API, OpenCV, scikit-image
- **Configuration**: Pydantic v2, pydantic-settings
- **Notifications**: Twilio (SMS), SendGrid (Email), Slack SDK
- **Database**: SQLite (expandable to PostgreSQL)
- **Async**: asyncio, aiohttp
- **Testing**: pytest, pytest-cov, pytest-asyncio

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Roboflow for AI-powered defect detection
- OpenCV community for computer vision tools
- Twilio, SendGrid, and Slack for notification services

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Status**: Production-ready ✅
**Version**: 0.1.0
**Last Updated**: 2025-11-17
