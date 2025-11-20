"""
Cache manager for weather API data.

This module provides a flexible caching system supporting multiple backends
(SQLite, Redis, in-memory) to reduce API calls and improve performance.
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import redis

from pv_simulator.config import Settings

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value as string, or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int) -> None:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (as string)
            ttl: Time-to-live in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using a dictionary."""

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self._cache: dict[str, tuple[str, datetime]] = {}
        logger.info("Initialized in-memory cache backend")

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from memory cache."""
        if key not in self._cache:
            return None

        value, expires_at = self._cache[key]

        if datetime.utcnow() > expires_at:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: str, ttl: int) -> None:
        """Store value in memory cache."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Delete value from memory cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        logger.info("Cleared in-memory cache")

    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        return self.get(key) is not None

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.utcnow()
        expired_keys = [key for key, (_, expires_at) in self._cache.items() if now > expires_at]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


class SQLiteCacheBackend(CacheBackend):
    """SQLite-based persistent cache backend."""

    def __init__(self, db_path: str = "./cache.db") -> None:
        """
        Initialize SQLite cache backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Initialized SQLite cache backend at {db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database with cache table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)"
            )
            conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from SQLite cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT value FROM cache
                WHERE key = ? AND expires_at > datetime('now')
                """,
                (key,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str, ttl: int) -> None:
        """Store value in SQLite cache."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
                """,
                (key, value, expires_at),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        """Delete value from SQLite cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        """Clear all cached values."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cleared SQLite cache")

    def exists(self, key: str) -> bool:
        """Check if key exists in SQLite cache."""
        return self.get(key) is not None

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at <= datetime('now')"
            )
            conn.commit()
            removed = cursor.rowcount

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired cache entries")

        return removed


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend."""

    def __init__(self, redis_url: str) -> None:
        """
        Initialize Redis cache backend.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self._test_connection()
        logger.info(f"Initialized Redis cache backend at {redis_url}")

    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from Redis cache."""
        try:
            return self.redis_client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: int) -> None:
        """Store value in Redis cache with TTL."""
        try:
            self.redis_client.setex(key, ttl, value)
        except redis.RedisError as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> None:
        """Delete value from Redis cache."""
        try:
            self.redis_client.delete(key)
        except redis.RedisError as e:
            logger.error(f"Redis delete error: {e}")

    def clear(self) -> None:
        """Clear all cached values in current database."""
        try:
            self.redis_client.flushdb()
            logger.info("Cleared Redis cache")
        except redis.RedisError as e:
            logger.error(f"Redis clear error: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return bool(self.redis_client.exists(key))
        except redis.RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return False


class CacheManager:
    """
    High-level cache manager for weather API data.

    Provides a unified interface for caching weather data with automatic
    serialization/deserialization and configurable backends.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize cache manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.backend = self._create_backend()
        self.default_ttl = settings.cache_ttl

    def _create_backend(self) -> CacheBackend:
        """
        Create appropriate cache backend based on settings.

        Returns:
            Cache backend instance
        """
        cache_type = self.settings.cache_type

        if cache_type == "memory":
            return MemoryCacheBackend()
        elif cache_type == "sqlite":
            return SQLiteCacheBackend()
        elif cache_type == "redis":
            return RedisCacheBackend(self.settings.redis_url)
        else:
            logger.warning(f"Unknown cache type {cache_type}, using memory cache")
            return MemoryCacheBackend()

    def _serialize(self, value: Any) -> str:
        """
        Serialize value for caching.

        Args:
            value: Value to serialize

        Returns:
            Serialized value as JSON string
        """
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        """
        Deserialize cached value.

        Args:
            value: Serialized value

        Returns:
            Deserialized value
        """
        return json.loads(value)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve and deserialize value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found
        """
        cached = self.backend.get(key)
        if cached is None:
            return None

        try:
            return self._deserialize(cached)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize cached value for key {key}: {e}")
            self.backend.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Serialize and store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            serialized = self._serialize(value)
            self.backend.set(key, serialized, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize value for key {key}: {e}")

    def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        self.backend.delete(key)

    def clear(self) -> None:
        """Clear all cached values."""
        self.backend.clear()

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        return self.backend.exists(key)

    def generate_key(self, *parts: Any) -> str:
        """
        Generate cache key from parts.

        Args:
            *parts: Key components

        Returns:
            Cache key string
        """
        return ":".join(str(part) for part in parts)

    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        Only works with memory and SQLite backends (Redis handles expiration automatically).

        Returns:
            Number of entries removed
        """
        if isinstance(self.backend, (MemoryCacheBackend, SQLiteCacheBackend)):
            return self.backend.cleanup_expired()
        return 0


def create_cache_manager(settings: Optional[Settings] = None) -> CacheManager:
    """
    Factory function to create a cache manager instance.

    Args:
        settings: Application settings (loads from environment if not provided)

    Returns:
        CacheManager instance
    """
    if settings is None:
        from pv_simulator import get_settings

        settings = get_settings()

    return CacheManager(settings)
