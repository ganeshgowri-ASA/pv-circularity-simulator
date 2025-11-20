"""
Alert History and Tracking Module.

This module provides comprehensive alert history management including
logging, acknowledgment tracking, resolution tracking, and analytics.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pv_simulator.alerts.manager import Alert
from pv_simulator.config import AlertSeverity, get_settings

logger = logging.getLogger(__name__)


class AcknowledgmentRecord(BaseModel):
    """
    Alert acknowledgment record.

    Attributes:
        alert_id: Associated alert ID
        acknowledged_by: User or system that acknowledged
        acknowledged_at: Acknowledgment timestamp
        notes: Optional acknowledgment notes
    """

    alert_id: str
    acknowledged_by: str
    acknowledged_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: Optional[str] = None


class ResolutionRecord(BaseModel):
    """
    Alert resolution record.

    Attributes:
        alert_id: Associated alert ID
        resolved_by: User or system that resolved
        resolved_at: Resolution timestamp
        resolution_method: How the issue was resolved
        resolution_notes: Detailed resolution notes
        time_to_resolve_seconds: Time from creation to resolution
    """

    alert_id: str
    resolved_by: str
    resolved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None
    time_to_resolve_seconds: Optional[float] = None


class AlertStatistics(BaseModel):
    """
    Alert statistics and metrics.

    Attributes:
        total_alerts: Total number of alerts
        alerts_by_severity: Count by severity level
        alerts_by_source: Count by source
        acknowledged_count: Number of acknowledged alerts
        resolved_count: Number of resolved alerts
        average_time_to_acknowledge: Average acknowledgment time (seconds)
        average_time_to_resolve: Average resolution time (seconds)
        active_alerts: Number of currently active alerts
    """

    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    alerts_by_source: Dict[str, int] = Field(default_factory=dict)
    acknowledged_count: int = 0
    resolved_count: int = 0
    average_time_to_acknowledge: Optional[float] = None
    average_time_to_resolve: Optional[float] = None
    active_alerts: int = 0


class AlertHistory:
    """
    Alert history and tracking system.

    Provides comprehensive alert logging, acknowledgment tracking,
    resolution tracking, and analytics capabilities.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize alert history.

        Args:
            db_path: Path to SQLite database file (default: ./alert_history.db)
        """
        self.settings = get_settings()
        self.db_path = db_path or str(Path("./alert_history.db"))

        # Initialize database
        self._init_database()

        logger.info(f"Initialized AlertHistory (database: {self.db_path})")

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Alerts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    context TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Acknowledgments table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS acknowledgments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    acknowledged_by TEXT NOT NULL,
                    acknowledged_at TEXT NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """
            )

            # Resolutions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resolutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    resolved_by TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    resolution_method TEXT,
                    resolution_notes TEXT,
                    time_to_resolve_seconds REAL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """
            )

            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)"
            )

            conn.commit()

        logger.info("Database schema initialized")

    def alert_log(self, alert: Alert) -> bool:
        """
        Log an alert to history.

        Args:
            alert: Alert to log

        Returns:
            True if successful

        Example:
            >>> history = AlertHistory()
            >>> alert = Alert(
            ...     severity=AlertSeverity.CRITICAL,
            ...     title="Panel Defect",
            ...     message="Critical crack detected",
            ...     source="defect_detector"
            ... )
            >>> history.alert_log(alert)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, rule_id, severity, title, message,
                        timestamp, source, context, acknowledged, resolved
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.rule_id,
                        alert.severity.value,
                        alert.title,
                        alert.message,
                        alert.timestamp,
                        alert.source,
                        json.dumps(alert.context),
                        1 if alert.acknowledged else 0,
                        1 if alert.resolved else 0,
                    ),
                )

                conn.commit()

            logger.debug(f"Logged alert: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False

    def acknowledgment_tracking(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Track alert acknowledgment.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User or system acknowledging
            notes: Optional acknowledgment notes

        Returns:
            True if successful

        Example:
            >>> history = AlertHistory()
            >>> history.acknowledgment_tracking(
            ...     "alert-123",
            ...     "operator@example.com",
            ...     notes="Investigating issue"
            ... )
        """
        try:
            record = AcknowledgmentRecord(
                alert_id=alert_id, acknowledged_by=acknowledged_by, notes=notes
            )

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert acknowledgment record
                cursor.execute(
                    """
                    INSERT INTO acknowledgments (
                        alert_id, acknowledged_by, acknowledged_at, notes
                    ) VALUES (?, ?, ?, ?)
                """,
                    (
                        record.alert_id,
                        record.acknowledged_by,
                        record.acknowledged_at,
                        record.notes,
                    ),
                )

                # Update alert status
                cursor.execute(
                    """
                    UPDATE alerts
                    SET acknowledged = 1
                    WHERE alert_id = ?
                """,
                    (alert_id,),
                )

                conn.commit()

            logger.info(f"Acknowledged alert {alert_id} by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to track acknowledgment: {e}")
            return False

    def resolution_tracking(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_method: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Track alert resolution.

        Args:
            alert_id: Alert ID to resolve
            resolved_by: User or system resolving
            resolution_method: How the issue was resolved
            resolution_notes: Detailed resolution notes

        Returns:
            True if successful

        Example:
            >>> history = AlertHistory()
            >>> history.resolution_tracking(
            ...     "alert-123",
            ...     "technician@example.com",
            ...     resolution_method="Panel replaced",
            ...     resolution_notes="Replaced defective panel PNL-123 with new unit"
            ... )
        """
        try:
            # Calculate time to resolve
            time_to_resolve = self._calculate_time_to_resolve(alert_id)

            record = ResolutionRecord(
                alert_id=alert_id,
                resolved_by=resolved_by,
                resolution_method=resolution_method,
                resolution_notes=resolution_notes,
                time_to_resolve_seconds=time_to_resolve,
            )

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert resolution record
                cursor.execute(
                    """
                    INSERT INTO resolutions (
                        alert_id, resolved_by, resolved_at, resolution_method,
                        resolution_notes, time_to_resolve_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.alert_id,
                        record.resolved_by,
                        record.resolved_at,
                        record.resolution_method,
                        record.resolution_notes,
                        record.time_to_resolve_seconds,
                    ),
                )

                # Update alert status
                cursor.execute(
                    """
                    UPDATE alerts
                    SET resolved = 1, acknowledged = 1
                    WHERE alert_id = ?
                """,
                    (alert_id,),
                )

                conn.commit()

            logger.info(
                f"Resolved alert {alert_id} by {resolved_by} "
                f"(time to resolve: {time_to_resolve:.1f}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to track resolution: {e}")
            return False

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Get alert by ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
                row = cursor.fetchone()

                if row:
                    alert_data = dict(row)
                    alert_data["context"] = json.loads(alert_data["context"])
                    alert_data["acknowledged"] = bool(alert_data["acknowledged"])
                    alert_data["resolved"] = bool(alert_data["resolved"])
                    return alert_data

                return None

        except Exception as e:
            logger.error(f"Failed to get alert: {e}")
            return None

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts with filtering.

        Args:
            severity: Filter by severity
            source: Filter by source
            acknowledged: Filter by acknowledgment status
            resolved: Filter by resolution status
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of alert data

        Example:
            >>> history = AlertHistory()
            >>> critical_alerts = history.get_alerts(
            ...     severity=AlertSeverity.CRITICAL,
            ...     resolved=False
            ... )
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM alerts WHERE 1=1"
                params = []

                if severity:
                    query += " AND severity = ?"
                    params.append(severity.value)

                if source:
                    query += " AND source = ?"
                    params.append(source)

                if acknowledged is not None:
                    query += " AND acknowledged = ?"
                    params.append(1 if acknowledged else 0)

                if resolved is not None:
                    query += " AND resolved = ?"
                    params.append(1 if resolved else 0)

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                alerts = []
                for row in rows:
                    alert_data = dict(row)
                    alert_data["context"] = json.loads(alert_data["context"])
                    alert_data["acknowledged"] = bool(alert_data["acknowledged"])
                    alert_data["resolved"] = bool(alert_data["resolved"])
                    alerts.append(alert_data)

                return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AlertStatistics:
        """
        Get alert statistics.

        Args:
            start_time: Start time for statistics
            end_time: End time for statistics

        Returns:
            AlertStatistics

        Example:
            >>> history = AlertHistory()
            >>> stats = history.get_statistics()
            >>> print(f"Total alerts: {stats.total_alerts}")
            >>> print(f"Critical alerts: {stats.alerts_by_severity.get('CRITICAL', 0)}")
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build time filter
                time_filter = ""
                params = []
                if start_time:
                    time_filter += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                if end_time:
                    time_filter += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                # Total alerts
                cursor.execute(f"SELECT COUNT(*) FROM alerts WHERE 1=1{time_filter}", params)
                total_alerts = cursor.fetchone()[0]

                # Alerts by severity
                cursor.execute(
                    f"SELECT severity, COUNT(*) FROM alerts WHERE 1=1{time_filter} GROUP BY severity",
                    params,
                )
                alerts_by_severity = dict(cursor.fetchall())

                # Alerts by source
                cursor.execute(
                    f"SELECT source, COUNT(*) FROM alerts WHERE 1=1{time_filter} GROUP BY source",
                    params,
                )
                alerts_by_source = dict(cursor.fetchall())

                # Acknowledged count
                cursor.execute(
                    f"SELECT COUNT(*) FROM alerts WHERE acknowledged = 1{time_filter}", params
                )
                acknowledged_count = cursor.fetchone()[0]

                # Resolved count
                cursor.execute(
                    f"SELECT COUNT(*) FROM alerts WHERE resolved = 1{time_filter}", params
                )
                resolved_count = cursor.fetchone()[0]

                # Active alerts
                cursor.execute(
                    f"SELECT COUNT(*) FROM alerts WHERE resolved = 0{time_filter}", params
                )
                active_alerts = cursor.fetchone()[0]

                # Average time to acknowledge
                cursor.execute(
                    f"""
                    SELECT AVG(
                        (julianday(ack.acknowledged_at) - julianday(a.timestamp)) * 86400
                    )
                    FROM alerts a
                    JOIN acknowledgments ack ON a.alert_id = ack.alert_id
                    WHERE 1=1{time_filter}
                """,
                    params,
                )
                avg_ack_time = cursor.fetchone()[0]

                # Average time to resolve
                cursor.execute(
                    f"""
                    SELECT AVG(time_to_resolve_seconds)
                    FROM resolutions r
                    JOIN alerts a ON r.alert_id = a.alert_id
                    WHERE 1=1{time_filter}
                """,
                    params,
                )
                avg_resolve_time = cursor.fetchone()[0]

                stats = AlertStatistics(
                    total_alerts=total_alerts,
                    alerts_by_severity=alerts_by_severity,
                    alerts_by_source=alerts_by_source,
                    acknowledged_count=acknowledged_count,
                    resolved_count=resolved_count,
                    average_time_to_acknowledge=avg_ack_time,
                    average_time_to_resolve=avg_resolve_time,
                    active_alerts=active_alerts,
                )

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return AlertStatistics()

    def get_acknowledgment_history(self, alert_id: str) -> List[AcknowledgmentRecord]:
        """
        Get acknowledgment history for an alert.

        Args:
            alert_id: Alert ID

        Returns:
            List of acknowledgment records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM acknowledgments
                    WHERE alert_id = ?
                    ORDER BY acknowledged_at
                """,
                    (alert_id,),
                )

                records = []
                for row in cursor.fetchall():
                    record = AcknowledgmentRecord(**dict(row))
                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Failed to get acknowledgment history: {e}")
            return []

    def get_resolution_history(self, alert_id: str) -> List[ResolutionRecord]:
        """
        Get resolution history for an alert.

        Args:
            alert_id: Alert ID

        Returns:
            List of resolution records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM resolutions
                    WHERE alert_id = ?
                    ORDER BY resolved_at
                """,
                    (alert_id,),
                )

                records = []
                for row in cursor.fetchall():
                    record = ResolutionRecord(**dict(row))
                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Failed to get resolution history: {e}")
            return []

    def cleanup_old_alerts(self, days: int = 90) -> int:
        """
        Clean up old resolved alerts.

        Args:
            days: Delete alerts older than this many days

        Returns:
            Number of alerts deleted

        Example:
            >>> history = AlertHistory()
            >>> deleted = history.cleanup_old_alerts(days=90)
            >>> print(f"Deleted {deleted} old alerts")
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete old resolved alerts
                cursor.execute(
                    """
                    DELETE FROM alerts
                    WHERE resolved = 1
                    AND timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                deleted_count = cursor.rowcount
                conn.commit()

            logger.info(f"Cleaned up {deleted_count} old alerts (older than {days} days)")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
            return 0

    def _calculate_time_to_resolve(self, alert_id: str) -> Optional[float]:
        """Calculate time from alert creation to resolution."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT timestamp FROM alerts WHERE alert_id = ?", (alert_id,))
                row = cursor.fetchone()

                if row:
                    created_at = datetime.fromisoformat(row[0])
                    resolved_at = datetime.utcnow()
                    delta = (resolved_at - created_at).total_seconds()
                    return delta

                return None

        except Exception as e:
            logger.error(f"Failed to calculate time to resolve: {e}")
            return None
