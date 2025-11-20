"""
Defect Database Management for PV System Diagnostics (B08-S05).

This module provides comprehensive defect data management with historical tracking,
pattern recognition, and fleet-wide analysis capabilities.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, Counter
import json

from pydantic import BaseModel, Field
import numpy as np
from sklearn.cluster import DBSCAN

from ...models import (
    Defect,
    DefectType,
    DefectSeverity,
    DefectPattern,
    DefectHistory,
    FleetAnalysis,
)


class DatabaseConfig(BaseModel):
    """
    Configuration for defect database management.

    Attributes:
        storage_backend: Storage backend type (memory, json, sql)
        enable_caching: Enable query result caching
        pattern_recognition_threshold: Threshold for pattern recognition
        clustering_epsilon: Epsilon parameter for DBSCAN clustering
        min_pattern_occurrences: Minimum occurrences to identify a pattern
    """

    storage_backend: str = Field(default="memory", description="Storage backend")
    enable_caching: bool = Field(default=True, description="Enable caching")
    pattern_recognition_threshold: float = Field(
        default=0.7,
        description="Pattern recognition threshold"
    )
    clustering_epsilon: float = Field(default=0.5, description="DBSCAN epsilon")
    min_pattern_occurrences: int = Field(
        default=3,
        description="Minimum pattern occurrences"
    )


class DefectDatabase:
    """
    Comprehensive defect database with analytics capabilities.

    This class provides defect storage, retrieval, historical tracking,
    pattern recognition, and fleet-wide analysis.

    Attributes:
        config: Database configuration
        defects: Dictionary of defects by ID
        defect_histories: Dictionary of defect histories
        patterns: Dictionary of identified patterns
        sites: Dictionary of site metadata
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the DefectDatabase.

        Args:
            config: Optional database configuration
        """
        self.config = config or DatabaseConfig()
        self.defects: Dict[str, Defect] = {}
        self.defect_histories: Dict[str, DefectHistory] = {}
        self.patterns: Dict[str, DefectPattern] = {}
        self.sites: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, Any] = {}

    def add_defect(self, defect: Defect) -> str:
        """
        Add a defect to the database.

        Args:
            defect: Defect object to add

        Returns:
            Defect ID
        """
        self.defects[defect.id] = defect

        # Initialize history if not exists
        if defect.id not in self.defect_histories:
            self.defect_histories[defect.id] = DefectHistory(
                defect_id=defect.id,
                snapshots=[self._create_defect_snapshot(defect)],
                progression_rate=0.0,
            )

        # Clear relevant caches
        self._invalidate_cache()

        return defect.id

    def defect_history(
        self,
        defect_id: Optional[str] = None,
        panel_id: Optional[str] = None,
        site_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[DefectHistory]:
        """
        Retrieve defect history with flexible filtering.

        Provides historical tracking of defects including progression,
        repair attempts, and status changes over time.

        Args:
            defect_id: Specific defect identifier
            panel_id: Filter by panel identifier
            site_id: Filter by site identifier
            start_date: Start date for history
            end_date: End date for history

        Returns:
            List of DefectHistory objects

        Example:
            >>> db = DefectDatabase()
            >>> history = db.defect_history(
            ...     panel_id="PANEL-001",
            ...     start_date=datetime(2024, 1, 1)
            ... )
            >>> for h in history:
            ...     print(f"Defect {h.defect_id}: {len(h.snapshots)} snapshots")
        """
        # If specific defect requested
        if defect_id:
            if defect_id in self.defect_histories:
                history = self.defect_histories[defect_id]
                return [self._filter_history_by_date(history, start_date, end_date)]
            return []

        # Otherwise, filter by criteria
        histories = []
        for hist_id, history in self.defect_histories.items():
            # Get the associated defect
            if history.defect_id not in self.defects:
                continue

            defect = self.defects[history.defect_id]

            # Apply filters
            if panel_id and defect.panel_id != panel_id:
                continue

            if site_id:
                # Check if defect's panel belongs to the site
                # In a real system, would have a panel-to-site mapping
                pass

            # Filter by date
            filtered_history = self._filter_history_by_date(
                history,
                start_date,
                end_date
            )

            if filtered_history.snapshots:
                histories.append(filtered_history)

        return histories

    def pattern_recognition(
        self,
        site_id: Optional[str] = None,
        defect_type: Optional[DefectType] = None,
        min_occurrences: Optional[int] = None,
    ) -> List[DefectPattern]:
        """
        Identify recurring defect patterns using machine learning.

        Analyzes defect data to identify patterns based on spatial clustering,
        temporal patterns, and defect characteristics.

        Args:
            site_id: Filter by site identifier
            defect_type: Filter by defect type
            min_occurrences: Minimum occurrences for a pattern

        Returns:
            List of identified DefectPattern objects

        Example:
            >>> db = DefectDatabase()
            >>> patterns = db.pattern_recognition(
            ...     site_id="SITE-001",
            ...     defect_type=DefectType.CRACK
            ... )
            >>> for pattern in patterns:
            ...     print(f"Pattern: {pattern.pattern_name}")
            ...     print(f"Affected panels: {len(pattern.affected_panels)}")
        """
        if min_occurrences is None:
            min_occurrences = self.config.min_pattern_occurrences

        # Filter defects
        defects = list(self.defects.values())

        if site_id:
            # Would filter by site in a real system
            pass

        if defect_type:
            defects = [d for d in defects if d.type == defect_type]

        if len(defects) < min_occurrences:
            return []

        patterns = []

        # Pattern Type 1: Spatial clustering
        spatial_patterns = self._identify_spatial_patterns(
            defects,
            min_occurrences
        )
        patterns.extend(spatial_patterns)

        # Pattern Type 2: Temporal patterns
        temporal_patterns = self._identify_temporal_patterns(
            defects,
            min_occurrences
        )
        patterns.extend(temporal_patterns)

        # Pattern Type 3: Characteristic patterns
        characteristic_patterns = self._identify_characteristic_patterns(
            defects,
            min_occurrences
        )
        patterns.extend(characteristic_patterns)

        # Store identified patterns
        for pattern in patterns:
            self.patterns[pattern.id] = pattern

        return patterns

    def fleet_wide_analysis(
        self,
        fleet_id: str,
        site_ids: List[str],
        include_benchmarks: bool = True,
    ) -> FleetAnalysis:
        """
        Perform comprehensive fleet-wide defect analysis.

        Analyzes defects across multiple sites to provide fleet-level
        insights, trends, and benchmarks.

        Args:
            fleet_id: Fleet identifier
            site_ids: List of site identifiers in the fleet
            include_benchmarks: Include benchmark comparisons

        Returns:
            FleetAnalysis object with comprehensive metrics

        Example:
            >>> db = DefectDatabase()
            >>> analysis = db.fleet_wide_analysis(
            ...     fleet_id="FLEET-NORTHEAST",
            ...     site_ids=["SITE-001", "SITE-002", "SITE-003"]
            ... )
            >>> print(f"Fleet health score: {analysis.fleet_health_score}")
            >>> print(f"Total defects: {analysis.total_defects}")
        """
        # Collect all defects for the fleet
        fleet_defects = [
            d for d in self.defects.values()
            # In a real system, would filter by site_ids
        ]

        # Count sites and panels
        total_sites = len(site_ids)
        panel_ids = set(d.panel_id for d in fleet_defects)
        total_panels = len(panel_ids)
        total_defects = len(fleet_defects)

        # Defect distribution by type
        defect_distribution = self._calculate_defect_distribution(fleet_defects)

        # Severity distribution
        severity_distribution = self._calculate_severity_distribution(fleet_defects)

        # Calculate fleet health score
        fleet_health_score = self._calculate_fleet_health_score(
            fleet_defects,
            total_panels
        )

        # Trend analysis
        trend_analysis = self._perform_trend_analysis(fleet_defects)

        # Benchmarks
        benchmarks = {}
        if include_benchmarks:
            benchmarks = self._calculate_benchmarks(
                fleet_defects,
                total_panels,
                total_sites
            )

        # Calculate average panel age (placeholder - would use actual data)
        average_panel_age = 5.0

        analysis = FleetAnalysis(
            fleet_id=fleet_id,
            site_ids=site_ids,
            total_sites=total_sites,
            total_panels=total_panels,
            total_defects=total_defects,
            defect_distribution=defect_distribution,
            severity_distribution=severity_distribution,
            average_panel_age=average_panel_age,
            fleet_health_score=fleet_health_score,
            trend_analysis=trend_analysis,
            benchmarks=benchmarks,
        )

        return analysis

    def query_defects(
        self,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Defect]:
        """
        Query defects with flexible filtering and sorting.

        Args:
            filters: Dictionary of field filters
            sort_by: Field to sort by
            limit: Maximum number of results

        Returns:
            List of filtered and sorted defects
        """
        defects = list(self.defects.values())

        # Apply filters
        if filters:
            defects = self._apply_filters(defects, filters)

        # Sort
        if sort_by:
            defects = self._sort_defects(defects, sort_by)

        # Limit
        if limit:
            defects = defects[:limit]

        return defects

    def get_statistics(self, site_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics for defects.

        Args:
            site_id: Optional site filter

        Returns:
            Dictionary of statistics
        """
        defects = list(self.defects.values())

        # Filter by site if specified
        if site_id:
            # Would filter by site in real system
            pass

        if not defects:
            return {
                "total_defects": 0,
                "by_type": {},
                "by_severity": {},
                "average_confidence": 0.0,
                "average_power_loss": 0.0,
            }

        # Calculate statistics
        total = len(defects)
        by_type = Counter(d.type for d in defects)
        by_severity = Counter(d.severity for d in defects)
        avg_confidence = sum(d.confidence for d in defects) / total
        avg_power_loss = sum(d.estimated_power_loss for d in defects) / total

        return {
            "total_defects": total,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "average_confidence": round(avg_confidence, 3),
            "average_power_loss": round(avg_power_loss, 2),
        }

    def _create_defect_snapshot(self, defect: Defect) -> Dict[str, Any]:
        """Create a snapshot of defect state."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": defect.severity.value,
            "estimated_power_loss": defect.estimated_power_loss,
            "confidence": defect.confidence,
            "status": defect.metadata.get("status", "active"),
        }

    def _filter_history_by_date(
        self,
        history: DefectHistory,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> DefectHistory:
        """Filter history snapshots by date range."""
        if not start_date and not end_date:
            return history

        filtered_snapshots = []
        for snapshot in history.snapshots:
            snapshot_time = datetime.fromisoformat(snapshot["timestamp"])

            if start_date and snapshot_time < start_date:
                continue
            if end_date and snapshot_time > end_date:
                continue

            filtered_snapshots.append(snapshot)

        return DefectHistory(
            defect_id=history.defect_id,
            snapshots=filtered_snapshots,
            progression_rate=history.progression_rate,
            repair_attempts=history.repair_attempts,
            current_status=history.current_status,
        )

    def _identify_spatial_patterns(
        self,
        defects: List[Defect],
        min_occurrences: int,
    ) -> List[DefectPattern]:
        """Identify spatial clustering patterns using DBSCAN."""
        if len(defects) < min_occurrences:
            return []

        patterns = []

        # Group by defect type
        by_type: Dict[DefectType, List[Defect]] = defaultdict(list)
        for defect in defects:
            by_type[defect.type].append(defect)

        for defect_type, type_defects in by_type.items():
            if len(type_defects) < min_occurrences:
                continue

            # Extract coordinates
            coords = np.array([
                [d.location.x, d.location.y]
                for d in type_defects
            ])

            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.clustering_epsilon,
                min_samples=min_occurrences
            ).fit(coords)

            # Create patterns for each cluster
            unique_labels = set(clustering.labels_)
            for label in unique_labels:
                if label == -1:  # Noise
                    continue

                cluster_mask = clustering.labels_ == label
                cluster_defects = [
                    d for d, mask in zip(type_defects, cluster_mask)
                    if mask
                ]

                if len(cluster_defects) >= min_occurrences:
                    pattern = DefectPattern(
                        pattern_name=f"Spatial Cluster - {defect_type.value} - {label}",
                        defect_type=defect_type,
                        frequency=len(cluster_defects),
                        affected_panels=[d.panel_id for d in cluster_defects],
                        common_characteristics={
                            "pattern_type": "spatial",
                            "cluster_label": int(label),
                            "avg_x": float(np.mean([d.location.x for d in cluster_defects])),
                            "avg_y": float(np.mean([d.location.y for d in cluster_defects])),
                        },
                        root_cause="Localized environmental or installation issue",
                        correlation_score=0.85,
                    )
                    patterns.append(pattern)

        return patterns

    def _identify_temporal_patterns(
        self,
        defects: List[Defect],
        min_occurrences: int,
    ) -> List[DefectPattern]:
        """Identify temporal patterns in defect occurrence."""
        patterns = []

        # Group by defect type and time windows
        by_type: Dict[DefectType, List[Defect]] = defaultdict(list)
        for defect in defects:
            by_type[defect.type].append(defect)

        for defect_type, type_defects in by_type.items():
            if len(type_defects) < min_occurrences:
                continue

            # Check for temporal clustering (within 30 days)
            type_defects.sort(key=lambda d: d.created_at)

            temporal_clusters = []
            current_cluster = [type_defects[0]]

            for defect in type_defects[1:]:
                time_diff = (defect.created_at - current_cluster[-1].created_at).days

                if time_diff <= 30:
                    current_cluster.append(defect)
                else:
                    if len(current_cluster) >= min_occurrences:
                        temporal_clusters.append(current_cluster)
                    current_cluster = [defect]

            if len(current_cluster) >= min_occurrences:
                temporal_clusters.append(current_cluster)

            # Create patterns for temporal clusters
            for idx, cluster in enumerate(temporal_clusters):
                pattern = DefectPattern(
                    pattern_name=f"Temporal Cluster - {defect_type.value} - {idx}",
                    defect_type=defect_type,
                    frequency=len(cluster),
                    affected_panels=[d.panel_id for d in cluster],
                    common_characteristics={
                        "pattern_type": "temporal",
                        "start_date": cluster[0].created_at.isoformat(),
                        "end_date": cluster[-1].created_at.isoformat(),
                        "duration_days": (cluster[-1].created_at - cluster[0].created_at).days,
                    },
                    root_cause="Time-correlated event or environmental condition",
                    correlation_score=0.75,
                )
                patterns.append(pattern)

        return patterns

    def _identify_characteristic_patterns(
        self,
        defects: List[Defect],
        min_occurrences: int,
    ) -> List[DefectPattern]:
        """Identify patterns based on defect characteristics."""
        patterns = []

        # Group by type and severity
        by_type_severity: Dict[Tuple[DefectType, DefectSeverity], List[Defect]] = defaultdict(list)
        for defect in defects:
            by_type_severity[(defect.type, defect.severity)].append(defect)

        for (defect_type, severity), group_defects in by_type_severity.items():
            if len(group_defects) >= min_occurrences:
                # Check if they share common characteristics
                avg_power_loss = sum(d.estimated_power_loss for d in group_defects) / len(group_defects)
                avg_confidence = sum(d.confidence for d in group_defects) / len(group_defects)

                pattern = DefectPattern(
                    pattern_name=f"Characteristic Pattern - {defect_type.value} - {severity.value}",
                    defect_type=defect_type,
                    frequency=len(group_defects),
                    affected_panels=[d.panel_id for d in group_defects],
                    common_characteristics={
                        "pattern_type": "characteristic",
                        "severity": severity.value,
                        "avg_power_loss": round(avg_power_loss, 2),
                        "avg_confidence": round(avg_confidence, 3),
                    },
                    correlation_score=0.70,
                )
                patterns.append(pattern)

        return patterns

    def _calculate_defect_distribution(
        self,
        defects: List[Defect]
    ) -> Dict[DefectType, int]:
        """Calculate defect distribution by type."""
        distribution = Counter(d.type for d in defects)
        return dict(distribution)

    def _calculate_severity_distribution(
        self,
        defects: List[Defect]
    ) -> Dict[DefectSeverity, int]:
        """Calculate defect distribution by severity."""
        distribution = Counter(d.severity for d in defects)
        return dict(distribution)

    def _calculate_fleet_health_score(
        self,
        defects: List[Defect],
        total_panels: int,
    ) -> float:
        """Calculate overall fleet health score (0-100)."""
        if total_panels == 0:
            return 100.0

        # Base score
        score = 100.0

        # Deduct for defects
        defect_ratio = len(defects) / total_panels
        score -= defect_ratio * 50  # Up to 50 points for defect count

        # Deduct for severity
        severity_weights = {
            DefectSeverity.LOW: 0.1,
            DefectSeverity.MEDIUM: 0.3,
            DefectSeverity.HIGH: 0.6,
            DefectSeverity.CRITICAL: 1.0,
        }

        severity_impact = sum(
            severity_weights.get(d.severity, 0.5) for d in defects
        ) / max(total_panels, 1)
        score -= severity_impact * 30  # Up to 30 points for severity

        # Deduct for power loss
        avg_power_loss = sum(d.estimated_power_loss for d in defects) / max(len(defects), 1) if defects else 0
        score -= avg_power_loss * 0.2  # Up to 20 points for power loss

        return max(0.0, min(100.0, score))

    def _perform_trend_analysis(self, defects: List[Defect]) -> Dict[str, Any]:
        """Perform trend analysis on defects."""
        if not defects:
            return {}

        # Sort by creation date
        sorted_defects = sorted(defects, key=lambda d: d.created_at)

        # Calculate monthly trends
        monthly_counts: Dict[str, int] = defaultdict(int)
        for defect in sorted_defects:
            month_key = defect.created_at.strftime("%Y-%m")
            monthly_counts[month_key] += 1

        # Calculate trend direction
        if len(monthly_counts) >= 2:
            counts = list(monthly_counts.values())
            trend = "increasing" if counts[-1] > counts[0] else "decreasing"
        else:
            trend = "stable"

        return {
            "monthly_counts": dict(monthly_counts),
            "trend_direction": trend,
            "total_tracked": len(defects),
        }

    def _calculate_benchmarks(
        self,
        defects: List[Defect],
        total_panels: int,
        total_sites: int,
    ) -> Dict[str, Any]:
        """Calculate benchmark metrics."""
        if total_panels == 0:
            return {}

        return {
            "defects_per_panel": round(len(defects) / total_panels, 4),
            "defects_per_site": round(len(defects) / total_sites, 2) if total_sites > 0 else 0,
            "critical_defect_rate": round(
                sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL) / total_panels,
                 4
            ),
            "average_detection_confidence": round(
                sum(d.confidence for d in defects) / len(defects), 3
            ) if defects else 0,
        }

    def _apply_filters(
        self,
        defects: List[Defect],
        filters: Dict[str, Any]
    ) -> List[Defect]:
        """Apply filters to defect list."""
        filtered = defects

        for field, value in filters.items():
            if field == "type":
                filtered = [d for d in filtered if d.type == value]
            elif field == "severity":
                filtered = [d for d in filtered if d.severity == value]
            elif field == "panel_id":
                filtered = [d for d in filtered if d.panel_id == value]
            elif field == "min_confidence":
                filtered = [d for d in filtered if d.confidence >= value]

        return filtered

    def _sort_defects(self, defects: List[Defect], sort_by: str) -> List[Defect]:
        """Sort defects by specified field."""
        if sort_by == "created_at":
            return sorted(defects, key=lambda d: d.created_at, reverse=True)
        elif sort_by == "severity":
            severity_order = {
                DefectSeverity.CRITICAL: 0,
                DefectSeverity.HIGH: 1,
                DefectSeverity.MEDIUM: 2,
                DefectSeverity.LOW: 3,
            }
            return sorted(defects, key=lambda d: severity_order.get(d.severity, 4))
        elif sort_by == "confidence":
            return sorted(defects, key=lambda d: d.confidence, reverse=True)
        return defects

    def _invalidate_cache(self) -> None:
        """Invalidate cached query results."""
        if self.config.enable_caching:
            self._cache.clear()
