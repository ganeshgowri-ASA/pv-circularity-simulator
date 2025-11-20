"""
Data models for PV circularity metrics, KPIs, and dashboard components.

This module defines the core data structures used throughout the dashboard
components to represent various metrics, performance indicators, and notifications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class TrendDirection(Enum):
    """Enum representing the direction of a metric's trend."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class NotificationLevel(Enum):
    """Enum representing the severity level of a notification."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricStatus(Enum):
    """Enum representing the status of a metric or KPI."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class MetricCard:
    """
    Represents a metric card for dashboard display.

    A metric card displays a single key metric with optional trend information,
    comparison values, and visual styling.

    Attributes:
        title: The display title of the metric
        value: The current value of the metric
        unit: The unit of measurement (e.g., 'kWh', '%', 'kg CO2')
        description: Optional detailed description of the metric
        trend_direction: Direction of the trend (up, down, flat)
        trend_value: Percentage or absolute change in the metric
        comparison_label: Label for the comparison period (e.g., 'vs. last month')
        icon: Icon identifier for visual representation
        color: Color scheme for the card (e.g., 'blue', 'green', 'red')
        status: Current status of the metric
        metadata: Additional metadata for custom use cases
    """
    title: str
    value: float
    unit: str
    description: Optional[str] = None
    trend_direction: TrendDirection = TrendDirection.FLAT
    trend_value: Optional[float] = None
    comparison_label: Optional[str] = "vs. previous period"
    icon: Optional[str] = None
    color: str = "blue"
    status: MetricStatus = MetricStatus.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_trend_emoji(self) -> str:
        """Get emoji representation of trend direction."""
        return {
            TrendDirection.UP: "📈",
            TrendDirection.DOWN: "📉",
            TrendDirection.FLAT: "➡️",
        }.get(self.trend_direction, "➡️")

    def is_positive_trend(self) -> bool:
        """
        Determine if the trend is positive based on the metric type.

        Returns:
            True if trend is considered positive, False otherwise
        """
        # This can be customized based on metric type
        # For most metrics, upward trend is positive
        return self.trend_direction == TrendDirection.UP


@dataclass
class KPI:
    """
    Represents a Key Performance Indicator for dashboard display.

    KPIs are critical metrics that measure performance against strategic goals.
    They typically include targets, thresholds, and historical data.

    Attributes:
        name: The name of the KPI
        current_value: Current value of the KPI
        target_value: Target or goal value
        unit: Unit of measurement
        description: Detailed description of what the KPI measures
        threshold_excellent: Value above which performance is excellent
        threshold_good: Value above which performance is good
        threshold_fair: Value above which performance is fair
        historical_values: List of historical values for trend analysis
        category: Category of the KPI (e.g., 'efficiency', 'circularity')
        weight: Importance weight (0-1) for aggregate calculations
        is_higher_better: Whether higher values indicate better performance
        last_updated: Timestamp of last update
        metadata: Additional metadata for custom use cases
    """
    name: str
    current_value: float
    target_value: float
    unit: str
    description: Optional[str] = None
    threshold_excellent: Optional[float] = None
    threshold_good: Optional[float] = None
    threshold_fair: Optional[float] = None
    historical_values: List[float] = field(default_factory=list)
    category: str = "general"
    weight: float = 1.0
    is_higher_better: bool = True
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_performance_percentage(self) -> float:
        """
        Calculate performance as percentage of target.

        Returns:
            Performance percentage (0-100+)
        """
        if self.target_value == 0:
            return 0.0
        return (self.current_value / self.target_value) * 100

    def get_status(self) -> MetricStatus:
        """
        Determine the status based on thresholds.

        Returns:
            MetricStatus enum value
        """
        value = self.current_value

        if not self.is_higher_better:
            # For metrics where lower is better, invert the logic
            if self.threshold_excellent and value <= self.threshold_excellent:
                return MetricStatus.EXCELLENT
            elif self.threshold_good and value <= self.threshold_good:
                return MetricStatus.GOOD
            elif self.threshold_fair and value <= self.threshold_fair:
                return MetricStatus.FAIR
            else:
                return MetricStatus.POOR
        else:
            # For metrics where higher is better
            if self.threshold_excellent and value >= self.threshold_excellent:
                return MetricStatus.EXCELLENT
            elif self.threshold_good and value >= self.threshold_good:
                return MetricStatus.GOOD
            elif self.threshold_fair and value >= self.threshold_fair:
                return MetricStatus.FAIR
            else:
                return MetricStatus.POOR

    def get_gap_to_target(self) -> float:
        """
        Calculate the gap between current value and target.

        Returns:
            Absolute difference between current and target value
        """
        return self.target_value - self.current_value


@dataclass
class ProgressMetric:
    """
    Represents a progress metric for tracking towards goals.

    Progress metrics show advancement towards specific objectives with
    multiple stages or milestones.

    Attributes:
        name: Name of the progress metric
        current_value: Current progress value
        target_value: Target or completion value
        unit: Unit of measurement
        description: Description of what is being tracked
        milestones: List of milestone values to display
        start_value: Starting value (default 0)
        color: Color scheme for the progress bar
        show_percentage: Whether to display as percentage
        stages: Named stages with their threshold values
        completion_date: Expected or actual completion date
        is_overdue: Whether the metric is behind schedule
        metadata: Additional metadata for custom use cases
    """
    name: str
    current_value: float
    target_value: float
    unit: str
    description: Optional[str] = None
    milestones: List[float] = field(default_factory=list)
    start_value: float = 0.0
    color: str = "blue"
    show_percentage: bool = True
    stages: Dict[str, float] = field(default_factory=dict)
    completion_date: Optional[datetime] = None
    is_overdue: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_progress_percentage(self) -> float:
        """
        Calculate progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        if self.target_value == self.start_value:
            return 0.0

        progress = ((self.current_value - self.start_value) /
                   (self.target_value - self.start_value)) * 100
        return max(0.0, min(100.0, progress))

    def get_remaining(self) -> float:
        """
        Calculate remaining value to reach target.

        Returns:
            Remaining value
        """
        return max(0.0, self.target_value - self.current_value)

    def is_completed(self) -> bool:
        """
        Check if the progress metric has reached its target.

        Returns:
            True if current value meets or exceeds target
        """
        return self.current_value >= self.target_value

    def get_current_stage(self) -> Optional[str]:
        """
        Get the current stage name based on progress.

        Returns:
            Name of the current stage or None
        """
        if not self.stages:
            return None

        current_stage = None
        for stage_name, threshold in sorted(self.stages.items(),
                                           key=lambda x: x[1]):
            if self.current_value >= threshold:
                current_stage = stage_name
            else:
                break

        return current_stage


@dataclass
class Notification:
    """
    Represents a notification or alert for the dashboard.

    Notifications inform users about important events, alerts, or status changes
    in the PV system.

    Attributes:
        title: Notification title
        message: Detailed notification message
        level: Severity level of the notification
        timestamp: When the notification was created
        category: Category of the notification (e.g., 'system', 'performance')
        is_read: Whether the notification has been read
        is_dismissible: Whether the notification can be dismissed
        action_label: Label for optional action button
        action_url: URL or identifier for the action
        source: Source system or component that generated the notification
        priority: Priority ranking (1-10, higher is more important)
        expires_at: Optional expiration timestamp
        metadata: Additional metadata for custom use cases
    """
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"
    is_read: bool = False
    is_dismissible: bool = True
    action_label: Optional[str] = None
    action_url: Optional[str] = None
    source: Optional[str] = None
    priority: int = 5
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_icon(self) -> str:
        """
        Get emoji icon based on notification level.

        Returns:
            Emoji string representing the notification level
        """
        return {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }.get(self.level, "ℹ️")

    def is_expired(self) -> bool:
        """
        Check if the notification has expired.

        Returns:
            True if notification is expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def get_age_seconds(self) -> float:
        """
        Get the age of the notification in seconds.

        Returns:
            Number of seconds since notification was created
        """
        return (datetime.now() - self.timestamp).total_seconds()
