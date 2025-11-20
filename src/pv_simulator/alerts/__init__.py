"""Alerts and notifications module for PV Circularity Simulator."""

from pv_simulator.alerts.handlers.notification_channels import (
    EmailAlerts,
    MobilePush,
    NotificationChannels,
    NotificationResult,
    SlackWebhooks,
    SMSNotifications,
)
from pv_simulator.alerts.history import (
    AcknowledgmentRecord,
    AlertHistory,
    AlertStatistics,
    ResolutionRecord,
)
from pv_simulator.alerts.manager import (
    Alert,
    AlertManager,
    AlertRule,
    AnomalyDetectionConfig,
    EscalationLevel,
    ThresholdRule,
)

__all__ = [
    # Alert Manager
    "AlertManager",
    "Alert",
    "AlertRule",
    "ThresholdRule",
    "AnomalyDetectionConfig",
    "EscalationLevel",
    # Notification Channels
    "NotificationChannels",
    "EmailAlerts",
    "SMSNotifications",
    "SlackWebhooks",
    "MobilePush",
    "NotificationResult",
    # Alert History
    "AlertHistory",
    "AcknowledgmentRecord",
    "ResolutionRecord",
    "AlertStatistics",
]
