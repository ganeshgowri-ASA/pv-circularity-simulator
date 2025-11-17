"""
Tests for weather cache functionality.
"""

import pytest

from pv_simulator.config import Settings
from pv_simulator.weather.cache import (
    CacheManager,
    MemoryCacheBackend,
    SQLiteCacheBackend,
)


class TestMemoryCacheBackend:
    """Tests for in-memory cache backend."""

    def test_set_and_get(self):
        """Test setting and getting cache values."""
        cache = MemoryCacheBackend()

        cache.set("test_key", "test_value", ttl=60)
        assert cache.get("test_key") == "test_value"

    def test_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        cache = MemoryCacheBackend()
        assert cache.get("nonexistent") is None

    def test_expiration(self):
        """Test cache expiration."""
        cache = MemoryCacheBackend()

        # Set with very short TTL
        cache.set("test_key", "test_value", ttl=0)

        # Should be expired immediately
        import time
        time.sleep(0.1)
        assert cache.get("test_key") is None

    def test_delete(self):
        """Test deleting cache entries."""
        cache = MemoryCacheBackend()

        cache.set("test_key", "test_value", ttl=60)
        cache.delete("test_key")
        assert cache.get("test_key") is None

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = MemoryCacheBackend()

        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_exists(self):
        """Test checking if key exists."""
        cache = MemoryCacheBackend()

        cache.set("test_key", "test_value", ttl=60)
        assert cache.exists("test_key") is True
        assert cache.exists("nonexistent") is False


class TestSQLiteCacheBackend:
    """Tests for SQLite cache backend."""

    def test_set_and_get(self, tmp_path):
        """Test setting and getting cache values."""
        db_path = tmp_path / "test_cache.db"
        cache = SQLiteCacheBackend(str(db_path))

        cache.set("test_key", "test_value", ttl=60)
        assert cache.get("test_key") == "test_value"

    def test_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        db_path = tmp_path / "test_cache.db"

        # Create and populate cache
        cache1 = SQLiteCacheBackend(str(db_path))
        cache1.set("test_key", "test_value", ttl=60)

        # Create new instance and verify data persists
        cache2 = SQLiteCacheBackend(str(db_path))
        assert cache2.get("test_key") == "test_value"

    def test_cleanup_expired(self, tmp_path):
        """Test cleanup of expired entries."""
        db_path = tmp_path / "test_cache.db"
        cache = SQLiteCacheBackend(str(db_path))

        # Set entry with short TTL
        cache.set("test_key", "test_value", ttl=0)

        import time
        time.sleep(0.1)

        # Cleanup should remove expired entries
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("test_key") is None


class TestCacheManager:
    """Tests for high-level cache manager."""

    def test_cache_json_serialization(self, cache_manager):
        """Test JSON serialization of cached values."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        cache_manager.set("test", data)
        retrieved = cache_manager.get("test")

        assert retrieved == data

    def test_generate_key(self, cache_manager):
        """Test cache key generation."""
        key = cache_manager.generate_key("provider", "current", 40.7128, -74.0060)
        assert "provider" in key
        assert "current" in key

    def test_cache_miss(self, cache_manager):
        """Test cache miss returns None."""
        assert cache_manager.get("nonexistent") is None

    def test_default_ttl(self, test_settings, cache_manager):
        """Test using default TTL."""
        cache_manager.set("test", "value")
        # Should use default TTL from settings
        assert cache_manager.exists("test")
