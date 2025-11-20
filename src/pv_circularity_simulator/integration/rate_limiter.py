"""
Rate limiting implementation using the token bucket algorithm.

This module provides a thread-safe and async-safe rate limiter that controls
the rate of API requests to prevent exceeding API rate limits.
"""

import asyncio
import time
from threading import Lock
from typing import Optional

from .models import RateLimitConfig


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request rates.

    The token bucket algorithm allows for bursty traffic while maintaining
    an average rate limit. Tokens are added to the bucket at a constant rate,
    and each request consumes one token. If no tokens are available, the
    request must wait.

    This implementation is thread-safe and supports both synchronous and
    asynchronous usage patterns.

    Attributes:
        config: Rate limiting configuration
        tokens: Current number of available tokens
        max_tokens: Maximum number of tokens (burst size)
        refill_rate: Rate at which tokens are added (tokens per second)
        last_refill: Timestamp of last token refill
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.enabled = config.enabled

        if self.enabled:
            # Calculate tokens per second
            self.refill_rate = config.max_requests / config.time_window
            self.max_tokens = config.burst_size or config.max_requests
            self.tokens = float(self.max_tokens)
            self.last_refill = time.time()

            # Locks for thread safety
            self._sync_lock = Lock()
            self._async_lock = asyncio.Lock()

    def _refill_tokens(self) -> None:
        """
        Refill tokens based on time elapsed since last refill.

        This method calculates how many tokens should be added based on
        the time elapsed and the refill rate, ensuring we never exceed
        the maximum token count.
        """
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now

    def _calculate_wait_time(self, tokens_needed: int = 1) -> float:
        """
        Calculate how long to wait for tokens to become available.

        Args:
            tokens_needed: Number of tokens needed (default: 1)

        Returns:
            Wait time in seconds (0 if tokens are available)
        """
        if self.tokens >= tokens_needed:
            return 0.0

        # Calculate how long until we have enough tokens
        tokens_deficit = tokens_needed - self.tokens
        wait_time = tokens_deficit / self.refill_rate
        return wait_time

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Acquire tokens from the bucket (synchronous).

        Args:
            tokens: Number of tokens to acquire (default: 1)
            blocking: Whether to wait for tokens to become available

        Returns:
            True if tokens were acquired, False if not available and non-blocking

        Raises:
            ValueError: If tokens requested exceeds max_tokens
        """
        if not self.enabled:
            return True

        if tokens > self.max_tokens:
            raise ValueError(
                f"Cannot acquire {tokens} tokens; max is {self.max_tokens}"
            )

        with self._sync_lock:
            self._refill_tokens()

            if not blocking and self.tokens < tokens:
                return False

            wait_time = self._calculate_wait_time(tokens)
            if wait_time > 0:
                time.sleep(wait_time)
                self._refill_tokens()

            self.tokens -= tokens
            return True

    async def acquire_async(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Acquire tokens from the bucket asynchronously.

        Args:
            tokens: Number of tokens to acquire (default: 1)
            blocking: Whether to wait for tokens to become available

        Returns:
            True if tokens were acquired, False if not available and non-blocking

        Raises:
            ValueError: If tokens requested exceeds max_tokens
        """
        if not self.enabled:
            return True

        if tokens > self.max_tokens:
            raise ValueError(
                f"Cannot acquire {tokens} tokens; max is {self.max_tokens}"
            )

        async with self._async_lock:
            self._refill_tokens()

            if not blocking and self.tokens < tokens:
                return False

            wait_time = self._calculate_wait_time(tokens)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._refill_tokens()

            self.tokens -= tokens
            return True

    def get_available_tokens(self) -> float:
        """
        Get the current number of available tokens.

        Returns:
            Number of available tokens
        """
        if not self.enabled:
            return float('inf')

        with self._sync_lock:
            self._refill_tokens()
            return self.tokens

    def reset(self) -> None:
        """
        Reset the rate limiter to full capacity.

        This method is useful for testing or when you want to reset
        the rate limiter state.
        """
        if not self.enabled:
            return

        with self._sync_lock:
            self.tokens = float(self.max_tokens)
            self.last_refill = time.time()

    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.

        Args:
            config: New rate limiting configuration
        """
        with self._sync_lock:
            self.config = config
            self.enabled = config.enabled

            if self.enabled:
                self.refill_rate = config.max_requests / config.time_window
                old_max = self.max_tokens
                self.max_tokens = config.burst_size or config.max_requests

                # Adjust current tokens proportionally
                if old_max > 0:
                    self.tokens = min(
                        self.max_tokens,
                        self.tokens * (self.max_tokens / old_max)
                    )
                else:
                    self.tokens = float(self.max_tokens)

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary containing rate limiter statistics
        """
        if not self.enabled:
            return {
                "enabled": False,
                "available_tokens": None,
                "max_tokens": None,
                "refill_rate": None,
            }

        with self._sync_lock:
            self._refill_tokens()
            return {
                "enabled": True,
                "available_tokens": self.tokens,
                "max_tokens": self.max_tokens,
                "refill_rate": self.refill_rate,
                "utilization": 1.0 - (self.tokens / self.max_tokens),
            }

    def __enter__(self):
        """Context manager entry - acquire a token."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    async def __aenter__(self):
        """Async context manager entry - acquire a token."""
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
