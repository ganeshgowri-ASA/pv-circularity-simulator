"""
Tests for rate limiter functionality.
"""

import asyncio
import time
import pytest

from pv_circularity_simulator.integration.models import RateLimitConfig
from pv_circularity_simulator.integration.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_disabled(self):
        """Test that disabled rate limiter allows all requests."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        # Should allow multiple requests instantly
        for _ in range(1000):
            assert limiter.acquire(blocking=False) is True

    def test_rate_limiter_acquire(self):
        """Test basic token acquisition."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        # Should be able to acquire up to max_requests tokens
        for _ in range(10):
            assert limiter.acquire(blocking=False) is True

        # Next request should fail (non-blocking)
        assert limiter.acquire(blocking=False) is False

    def test_rate_limiter_refill(self):
        """Test token refill over time."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        # Wait for refill
        time.sleep(0.5)

        # Should have some tokens available
        assert limiter.get_available_tokens() > 0

    def test_rate_limiter_blocking(self):
        """Test blocking token acquisition."""
        config = RateLimitConfig(
            max_requests=5,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        # Exhaust tokens
        for _ in range(5):
            limiter.acquire()

        # This should block briefly then succeed
        start = time.time()
        limiter.acquire(blocking=True)
        elapsed = time.time() - start

        # Should have waited for refill
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_async(self):
        """Test async token acquisition."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        # Should be able to acquire tokens asynchronously
        for _ in range(10):
            assert await limiter.acquire_async(blocking=False) is True

        # Next request should fail
        assert await limiter.acquire_async(blocking=False) is False

    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        assert limiter.get_available_tokens() == 0

        # Reset should restore full capacity
        limiter.reset()
        assert limiter.get_available_tokens() == 10

    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        config = RateLimitConfig(
            max_requests=100,
            time_window=60.0
        )
        limiter = RateLimiter(config)

        stats = limiter.get_stats()
        assert stats["enabled"] is True
        assert stats["max_tokens"] == 100
        assert stats["available_tokens"] == 100

    def test_rate_limiter_context_manager(self):
        """Test rate limiter as context manager."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        initial_tokens = limiter.get_available_tokens()

        with limiter:
            # Token should be consumed
            assert limiter.get_available_tokens() < initial_tokens

    @pytest.mark.asyncio
    async def test_rate_limiter_async_context_manager(self):
        """Test rate limiter as async context manager."""
        config = RateLimitConfig(
            max_requests=10,
            time_window=1.0
        )
        limiter = RateLimiter(config)

        initial_tokens = limiter.get_available_tokens()

        async with limiter:
            # Token should be consumed
            assert limiter.get_available_tokens() < initial_tokens
