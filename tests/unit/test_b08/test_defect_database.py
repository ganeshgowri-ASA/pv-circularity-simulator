"""
Tests for DefectDatabase (B08-S05).
"""

from datetime import datetime, timedelta
from typing import List
import pytest

from src.pv_circularity.models import (
    Defect,
    DefectType,
    DefectSeverity,
)
from src.pv_circularity.b08_diagnostics.b08_s05 import (
    DefectDatabase,
    DatabaseConfig,
)


class TestDefectDatabase:
    """Test suite for DefectDatabase."""

    def test_initialization(self):
        """Test that DefectDatabase initializes correctly."""
        db = DefectDatabase()
        assert db is not None
        assert isinstance(db.config, DatabaseConfig)
        assert len(db.defects) == 0

    def test_add_defect(self, sample_defect: Defect):
        """Test adding a defect to the database."""
        db = DefectDatabase()
        defect_id = db.add_defect(sample_defect)

        assert defect_id == sample_defect.id
        assert defect_id in db.defects
        assert db.defects[defect_id] == sample_defect

    def test_add_multiple_defects(self, sample_defects: List[Defect]):
        """Test adding multiple defects."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        assert len(db.defects) == len(sample_defects)

    def test_defect_history(self, sample_defect: Defect):
        """Test retrieving defect history."""
        db = DefectDatabase()
        db.add_defect(sample_defect)

        history = db.defect_history(defect_id=sample_defect.id)

        assert len(history) == 1
        assert history[0].defect_id == sample_defect.id
        assert len(history[0].snapshots) > 0

    def test_defect_history_with_date_filter(self, sample_defect: Defect):
        """Test defect history with date filtering."""
        db = DefectDatabase()
        db.add_defect(sample_defect)

        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)

        history = db.defect_history(
            defect_id=sample_defect.id,
            start_date=start_date,
            end_date=end_date,
        )

        assert len(history) > 0

    def test_pattern_recognition(self, sample_defects: List[Defect]):
        """Test pattern recognition."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        patterns = db.pattern_recognition()

        assert isinstance(patterns, list)
        # May or may not find patterns depending on defect distribution

    def test_pattern_recognition_by_type(self, sample_defects: List[Defect]):
        """Test pattern recognition filtered by type."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        patterns = db.pattern_recognition(defect_type=DefectType.CRACK)

        assert isinstance(patterns, list)

    def test_fleet_wide_analysis(self, sample_defects: List[Defect]):
        """Test fleet-wide analysis."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        analysis = db.fleet_wide_analysis(
            fleet_id="FLEET-001",
            site_ids=["SITE-001", "SITE-002"],
        )

        assert analysis.fleet_id == "FLEET-001"
        assert analysis.total_defects >= 0
        assert 0 <= analysis.fleet_health_score <= 100

    def test_query_defects_no_filters(self, sample_defects: List[Defect]):
        """Test querying defects without filters."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        results = db.query_defects()

        assert len(results) == len(sample_defects)

    def test_query_defects_with_type_filter(self, sample_defects: List[Defect]):
        """Test querying defects with type filter."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        results = db.query_defects(filters={"type": DefectType.CRACK})

        assert all(d.type == DefectType.CRACK for d in results)

    def test_query_defects_with_severity_filter(self, sample_defects: List[Defect]):
        """Test querying defects with severity filter."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        results = db.query_defects(filters={"severity": DefectSeverity.HIGH})

        assert all(d.severity == DefectSeverity.HIGH for d in results)

    def test_query_defects_with_limit(self, sample_defects: List[Defect]):
        """Test querying defects with limit."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        limit = 2
        results = db.query_defects(limit=limit)

        assert len(results) <= limit

    def test_get_statistics(self, sample_defects: List[Defect]):
        """Test getting defect statistics."""
        db = DefectDatabase()

        for defect in sample_defects:
            db.add_defect(defect)

        stats = db.get_statistics()

        assert stats["total_defects"] == len(sample_defects)
        assert "by_type" in stats
        assert "by_severity" in stats
        assert "average_confidence" in stats
        assert stats["average_confidence"] > 0

    def test_get_statistics_empty_database(self):
        """Test getting statistics from empty database."""
        db = DefectDatabase()
        stats = db.get_statistics()

        assert stats["total_defects"] == 0
        assert stats["average_confidence"] == 0.0
