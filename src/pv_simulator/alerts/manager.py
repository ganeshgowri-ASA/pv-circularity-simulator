"""
Alert Manager for PV Circularity Simulator.

This module provides centralized alert management including rule-based triggering,
threshold monitoring, anomaly detection, and escalation workflows.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from pv_simulator.config import AlertSeverity, get_settings

logger = logging.getLogger(__name__)


class AlertRule(BaseModel):
    """
    Alert rule definition.

    Attributes:
        rule_id: Unique rule identifier
        name: Human-readable rule name
        description: Rule description
        condition: Condition expression or callable
        severity: Alert severity when triggered
        enabled: Whether rule is active
        cooldown_seconds: Minimum time between alerts
        metadata: Additional rule metadata
    """

    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    condition: Any  # Can be string expression or callable
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_seconds: int = Field(default=300, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """
    Alert instance.

    Attributes:
        alert_id: Unique alert identifier
        rule_id: ID of rule that triggered this alert
        severity: Alert severity level
        title: Alert title
        message: Detailed alert message
        timestamp: Alert creation timestamp
        source: Alert source (e.g., module name)
        context: Additional context data
        acknowledged: Whether alert has been acknowledged
        resolved: Whether issue has been resolved
    """

    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    rule_id: Optional[str] = None
    severity: AlertSeverity
    title: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "system"
    context: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


class ThresholdRule(BaseModel):
    """
    Threshold-based monitoring rule.

    Attributes:
        metric_name: Name of metric to monitor
        operator: Comparison operator (>, <, >=, <=, ==, !=)
        threshold_value: Threshold value
        severity: Alert severity
        window_size: Number of consecutive violations required
        enabled: Whether rule is active
    """

    metric_name: str
    operator: str = Field(..., pattern="^(>|<|>=|<=|==|!=)$")
    threshold_value: float
    severity: AlertSeverity = AlertSeverity.WARNING
    window_size: int = Field(default=1, ge=1)
    enabled: bool = True


class AnomalyDetectionConfig(BaseModel):
    """
    Anomaly detection configuration.

    Attributes:
        metric_name: Metric to monitor
        method: Detection method (zscore, iqr, isolation_forest)
        sensitivity: Detection sensitivity (0-1)
        window_size: Historical window size
        min_samples: Minimum samples required
    """

    metric_name: str
    method: str = Field(default="zscore", pattern="^(zscore|iqr|isolation_forest)$")
    sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)
    window_size: int = Field(default=100, ge=10)
    min_samples: int = Field(default=30, ge=10)


class EscalationLevel(BaseModel):
    """
    Escalation level definition.

    Attributes:
        level: Escalation level (1, 2, 3, etc.)
        delay_seconds: Time before escalation
        severity_threshold: Minimum severity for this level
        notification_channels: Channels to use at this level
    """

    level: int = Field(..., ge=1)
    delay_seconds: int = Field(..., ge=0)
    severity_threshold: AlertSeverity
    notification_channels: List[str] = Field(default_factory=list)


class AlertManager:
    """
    Centralized alert management system.

    Provides:
    - Rule-based alert triggering
    - Threshold monitoring
    - Anomaly detection
    - Alert deduplication
    - Escalation workflows
    - Alert history and tracking
    """

    def __init__(self):
        """Initialize alert manager."""
        self.settings = get_settings()
        self.rules: Dict[str, AlertRule] = {}
        self.threshold_rules: Dict[str, ThresholdRule] = {}
        self.anomaly_configs: Dict[str, AnomalyDetectionConfig] = {}
        self.escalation_levels: List[EscalationLevel] = []

        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.rule_last_triggered: Dict[str, datetime] = {}

        # Metric history for anomaly detection
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Threshold violation tracking
        self.threshold_violations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Handlers
        self.alert_handlers: List[Callable] = []

        # Escalation tracking
        self.escalation_timers: Dict[str, asyncio.Task] = {}

        logger.info("Initialized AlertManager")

    def rule_engine(self, rule: AlertRule) -> str:
        """
        Register a custom alert rule.

        Args:
            rule: AlertRule to register

        Returns:
            Rule ID

        Example:
            >>> manager = AlertManager()
            >>> rule = AlertRule(
            ...     name="High Temperature",
            ...     description="Alert when temperature exceeds 90°C",
            ...     condition=lambda ctx: ctx.get('temperature', 0) > 90,
            ...     severity=AlertSeverity.CRITICAL
            ... )
            >>> rule_id = manager.rule_engine(rule)
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Registered rule: {rule.name} (ID: {rule.rule_id})")
        return rule.rule_id

    def threshold_monitoring(
        self,
        metric_name: str,
        operator: str,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        window_size: int = 1,
    ) -> str:
        """
        Set up threshold-based monitoring for a metric.

        Args:
            metric_name: Name of metric to monitor
            operator: Comparison operator (>, <, >=, <=, ==, !=)
            threshold_value: Threshold value
            severity: Alert severity when triggered
            window_size: Number of consecutive violations required

        Returns:
            Rule ID

        Example:
            >>> manager = AlertManager()
            >>> rule_id = manager.threshold_monitoring(
            ...     "panel_efficiency",
            ...     "<",
            ...     0.15,
            ...     severity=AlertSeverity.HIGH,
            ...     window_size=3
            ... )
        """
        rule = ThresholdRule(
            metric_name=metric_name,
            operator=operator,
            threshold_value=threshold_value,
            severity=severity,
            window_size=window_size,
        )

        rule_id = f"threshold_{metric_name}_{operator}_{threshold_value}"
        self.threshold_rules[rule_id] = rule

        logger.info(
            f"Registered threshold monitor: {metric_name} {operator} {threshold_value} "
            f"(severity={severity.value}, window={window_size})"
        )

        return rule_id

    def anomaly_detection(
        self,
        metric_name: str,
        method: str = "zscore",
        sensitivity: float = 0.95,
        window_size: int = 100,
        min_samples: int = 30,
    ) -> str:
        """
        Enable anomaly detection for a metric.

        Detects statistical anomalies using various methods:
        - zscore: Z-score based detection
        - iqr: Interquartile range method
        - isolation_forest: ML-based isolation forest

        Args:
            metric_name: Metric to monitor
            method: Detection method
            sensitivity: Detection sensitivity (0-1)
            window_size: Historical window size
            min_samples: Minimum samples required

        Returns:
            Configuration ID

        Example:
            >>> manager = AlertManager()
            >>> config_id = manager.anomaly_detection(
            ...     "power_output",
            ...     method="zscore",
            ...     sensitivity=0.95
            ... )
        """
        config = AnomalyDetectionConfig(
            metric_name=metric_name,
            method=method,
            sensitivity=sensitivity,
            window_size=window_size,
            min_samples=min_samples,
        )

        self.anomaly_configs[metric_name] = config

        logger.info(
            f"Enabled anomaly detection: {metric_name} (method={method}, "
            f"sensitivity={sensitivity})"
        )

        return metric_name

    def escalation_workflows(
        self,
        levels: List[EscalationLevel],
    ) -> None:
        """
        Configure multi-level escalation workflow.

        Escalation automatically increases alert priority and notification
        scope if alerts are not acknowledged within specified timeframes.

        Args:
            levels: List of escalation levels

        Example:
            >>> manager = AlertManager()
            >>> manager.escalation_workflows([
            ...     EscalationLevel(
            ...         level=1,
            ...         delay_seconds=300,
            ...         severity_threshold=AlertSeverity.WARNING,
            ...         notification_channels=["email"]
            ...     ),
            ...     EscalationLevel(
            ...         level=2,
            ...         delay_seconds=600,
            ...         severity_threshold=AlertSeverity.HIGH,
            ...         notification_channels=["email", "sms"]
            ...     ),
            ...     EscalationLevel(
            ...         level=3,
            ...         delay_seconds=900,
            ...         severity_threshold=AlertSeverity.CRITICAL,
            ...         notification_channels=["email", "sms", "slack"]
            ...     ),
            ... ])
        """
        self.escalation_levels = sorted(levels, key=lambda x: x.level)
        logger.info(f"Configured {len(levels)} escalation levels")

    def trigger_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str = "system",
        context: Optional[Dict[str, Any]] = None,
        rule_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Manually trigger an alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            source: Alert source
            context: Additional context
            rule_id: Associated rule ID

        Returns:
            Created Alert or None if suppressed

        Example:
            >>> manager = AlertManager()
            >>> alert = manager.trigger_alert(
            ...     "Panel Defect Detected",
            ...     "Critical crack detected in panel PNL-123",
            ...     AlertSeverity.CRITICAL,
            ...     source="defect_detector",
            ...     context={"panel_id": "PNL-123", "defect_type": "crack"}
            ... )
        """
        # Check if alert meets minimum severity
        if not self._check_severity_threshold(severity):
            logger.debug(f"Alert suppressed: severity {severity.value} below threshold")
            return None

        # Check rule cooldown
        if rule_id and not self._check_cooldown(rule_id):
            logger.debug(f"Alert suppressed: rule {rule_id} in cooldown period")
            return None

        # Create alert
        alert = Alert(
            rule_id=rule_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            context=context or {},
        )

        # Check for duplicates
        if self._is_duplicate(alert):
            logger.debug(f"Alert suppressed: duplicate detected")
            return None

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Update rule trigger time
        if rule_id:
            self.rule_last_triggered[rule_id] = datetime.utcnow()

        # Dispatch to handlers
        self._dispatch_alert(alert)

        # Start escalation workflow if configured
        if self.escalation_levels:
            self._start_escalation(alert)

        logger.info(
            f"Alert triggered: {alert.title} (severity={severity.value}, "
            f"id={alert.alert_id})"
        )

        return alert

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against current context.

        Args:
            context: Current system context/state

        Returns:
            List of triggered alerts

        Example:
            >>> manager = AlertManager()
            >>> context = {"temperature": 95, "efficiency": 0.12}
            >>> alerts = manager.evaluate_rules(context)
        """
        triggered_alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            try:
                # Evaluate condition
                if callable(rule.condition):
                    triggered = rule.condition(context)
                else:
                    # Simple string evaluation (be careful with security!)
                    triggered = eval(rule.condition, {"__builtins__": {}}, context)

                if triggered:
                    alert = self.trigger_alert(
                        title=rule.name,
                        message=rule.description,
                        severity=rule.severity,
                        source="rule_engine",
                        context=context,
                        rule_id=rule.rule_id,
                    )
                    if alert:
                        triggered_alerts.append(alert)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        return triggered_alerts

    def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check all threshold rules against current metrics.

        Args:
            metrics: Current metric values

        Returns:
            List of triggered alerts

        Example:
            >>> manager = AlertManager()
            >>> metrics = {"temperature": 92, "efficiency": 0.14}
            >>> alerts = manager.check_thresholds(metrics)
        """
        triggered_alerts = []

        for rule_id, rule in self.threshold_rules.items():
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue

            # Evaluate threshold
            violated = self._evaluate_threshold(
                metric_value, rule.operator, rule.threshold_value
            )

            # Track violations
            self.threshold_violations[rule_id].append(violated)

            # Check if window size is met
            recent_violations = list(self.threshold_violations[rule_id])[-rule.window_size :]
            if len(recent_violations) == rule.window_size and all(recent_violations):
                alert = self.trigger_alert(
                    title=f"Threshold Exceeded: {rule.metric_name}",
                    message=f"{rule.metric_name} {rule.operator} {rule.threshold_value} "
                    f"(current: {metric_value})",
                    severity=rule.severity,
                    source="threshold_monitor",
                    context={"metric": rule.metric_name, "value": metric_value},
                    rule_id=rule_id,
                )
                if alert:
                    triggered_alerts.append(alert)

        return triggered_alerts

    def check_anomalies(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check for anomalies in metrics.

        Args:
            metrics: Current metric values

        Returns:
            List of anomaly alerts

        Example:
            >>> manager = AlertManager()
            >>> metrics = {"power_output": 120}  # Anomalously low
            >>> alerts = manager.check_anomalies(metrics)
        """
        triggered_alerts = []

        for metric_name, config in self.anomaly_configs.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]

            # Store in history
            self.metric_history[metric_name].append(value)

            # Need minimum samples
            if len(self.metric_history[metric_name]) < config.min_samples:
                continue

            # Detect anomaly
            is_anomaly = self._detect_anomaly(metric_name, value, config)

            if is_anomaly:
                alert = self.trigger_alert(
                    title=f"Anomaly Detected: {metric_name}",
                    message=f"Anomalous value detected for {metric_name}: {value}",
                    severity=AlertSeverity.WARNING,
                    source="anomaly_detector",
                    context={
                        "metric": metric_name,
                        "value": value,
                        "method": config.method,
                    },
                    rule_id=f"anomaly_{metric_name}",
                )
                if alert:
                    triggered_alerts.append(alert)

        return triggered_alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User or system acknowledging

        Returns:
            True if successful

        Example:
            >>> manager = AlertManager()
            >>> manager.acknowledge_alert("alert-123", "user@example.com")
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

            # Cancel escalation
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]

            return True

        return False

    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """
        Mark alert as resolved.

        Args:
            alert_id: Alert ID to resolve
            resolved_by: User or system resolving

        Returns:
            True if successful

        Example:
            >>> manager = AlertManager()
            >>> manager.resolve_alert("alert-123", "technician@example.com")
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")

            # Remove from active alerts
            del self.active_alerts[alert_id]

            # Cancel escalation
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]

            return True

        return False

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Register an alert handler.

        Args:
            handler: Callable that receives Alert objects

        Example:
            >>> manager = AlertManager()
            >>> def log_alert(alert: Alert):
            ...     print(f"Alert: {alert.title}")
            >>> manager.register_handler(log_alert)
        """
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None, source: Optional[str] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            source: Filter by source

        Returns:
            List of active alerts

        Example:
            >>> manager = AlertManager()
            >>> critical_alerts = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts

    # Helper methods

    def _check_severity_threshold(self, severity: AlertSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.EMERGENCY: 4,
        }

        return severity_order.get(severity, 0) >= severity_order.get(
            self.settings.alert.min_severity, 0
        )

    def _check_cooldown(self, rule_id: str) -> bool:
        """Check if rule is past cooldown period."""
        if rule_id not in self.rule_last_triggered:
            return True

        rule = self.rules.get(rule_id) or self.threshold_rules.get(rule_id)
        if not rule:
            return True

        last_triggered = self.rule_last_triggered[rule_id]
        cooldown = getattr(rule, "cooldown_seconds", 300)
        elapsed = (datetime.utcnow() - last_triggered).total_seconds()

        return elapsed >= cooldown

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within deduplication window."""
        window_seconds = self.settings.alert.deduplicate_window_seconds
        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)

        for existing in self.active_alerts.values():
            if (
                existing.title == alert.title
                and existing.severity == alert.severity
                and existing.source == alert.source
            ):
                existing_time = datetime.fromisoformat(existing.timestamp)
                if existing_time > cutoff_time:
                    return True

        return False

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to all registered handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")

    def _start_escalation(self, alert: Alert) -> None:
        """Start escalation workflow for alert."""
        if not self.escalation_levels:
            return

        async def escalate():
            for level in self.escalation_levels:
                if alert.severity.value < level.severity_threshold.value:
                    continue

                await asyncio.sleep(level.delay_seconds)

                if alert.acknowledged or alert.resolved:
                    break

                logger.warning(
                    f"Escalating alert {alert.alert_id} to level {level.level}"
                )
                # Escalation would trigger additional notifications here

        # Create escalation task
        task = asyncio.create_task(escalate())
        self.escalation_timers[alert.alert_id] = task

    def _evaluate_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate threshold condition."""
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t,
        }
        return operators[operator](value, threshold)

    def _detect_anomaly(
        self, metric_name: str, value: float, config: AnomalyDetectionConfig
    ) -> bool:
        """Detect if value is anomalous."""
        history = list(self.metric_history[metric_name])

        if config.method == "zscore":
            return self._zscore_anomaly(value, history, config.sensitivity)
        elif config.method == "iqr":
            return self._iqr_anomaly(value, history)
        else:
            # Isolation forest would require sklearn - simplified here
            return False

    def _zscore_anomaly(self, value: float, history: List[float], sensitivity: float) -> bool:
        """Z-score based anomaly detection."""
        import statistics

        if len(history) < 2:
            return False

        mean = statistics.mean(history)
        stdev = statistics.stdev(history)

        if stdev == 0:
            return False

        z_score = abs((value - mean) / stdev)
        threshold = 2.0 + (1.0 - sensitivity) * 2.0  # 2-4 sigma

        return z_score > threshold

    def _iqr_anomaly(self, value: float, history: List[float]) -> bool:
        """IQR-based anomaly detection."""
        if len(history) < 4:
            return False

        sorted_history = sorted(history)
        q1_idx = len(sorted_history) // 4
        q3_idx = 3 * len(sorted_history) // 4

        q1 = sorted_history[q1_idx]
        q3 = sorted_history[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return value < lower_bound or value > upper_bound
