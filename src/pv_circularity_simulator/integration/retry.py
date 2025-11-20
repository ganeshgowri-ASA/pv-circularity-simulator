"""
Retry logic with exponential backoff for handling transient failures.

This module provides a robust retry mechanism for API requests, implementing
exponential backoff with optional jitter to handle rate limits and transient errors.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, List, Optional, Set, TypeVar, Union

from .models import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryExhausted(Exception):
    """Exception raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        """
        Initialize RetryExhausted exception.

        Args:
            message: Error message
            last_exception: The last exception that caused the retry to fail
        """
        super().__init__(message)
        self.last_exception = last_exception


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    This class implements a sophisticated retry mechanism with:
    - Exponential backoff with configurable base
    - Optional jitter to prevent thundering herd
    - Configurable retry conditions based on status codes
    - Maximum retry attempts and delays
    - Support for both sync and async operations

    Attributes:
        config: Retry configuration
        attempt_count: Number of retry attempts made in current operation
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize the retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config
        self.attempt_count = 0

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry attempt.

        Uses exponential backoff: delay = initial_delay * (base ^ attempt)
        Optionally adds jitter to prevent synchronized retries.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        # Calculate exponential backoff
        delay = self.config.initial_delay * (
            self.config.exponential_base ** attempt
        )

        # Cap at maximum delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled (±25% of delay)
        if self.config.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)  # Ensure non-negative

    def should_retry(
        self,
        attempt: int,
        status_code: Optional[int] = None,
        exception: Optional[Exception] = None,
    ) -> bool:
        """
        Determine if a retry should be attempted.

        Args:
            attempt: Current attempt number (0-indexed)
            status_code: HTTP status code from failed request
            exception: Exception raised during request

        Returns:
            True if retry should be attempted, False otherwise
        """
        if not self.config.enabled:
            return False

        # Check if we've exceeded max retries
        if attempt >= self.config.max_retries:
            return False

        # If we have a status code, check if it's in retry list
        if status_code is not None:
            return status_code in self.config.retry_on_status_codes

        # If we have an exception, retry on network errors
        if exception is not None:
            # Retry on common network exceptions
            retryable_exceptions = (
                ConnectionError,
                TimeoutError,
                OSError,
            )
            return isinstance(exception, retryable_exceptions)

        return False

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with retry logic (synchronous).

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from successful function execution

        Raises:
            RetryExhausted: If all retry attempts are exhausted
        """
        if not self.config.enabled:
            return func(*args, **kwargs)

        last_exception = None
        self.attempt_count = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"Request succeeded after {attempt} retry attempt(s)"
                    )
                return result

            except Exception as e:
                last_exception = e
                self.attempt_count = attempt + 1

                # Check if we should retry
                status_code = getattr(e, 'status_code', None)
                if not self.should_retry(attempt, status_code, e):
                    logger.error(
                        f"Request failed with non-retryable error: {e}"
                    )
                    raise

                # Calculate delay and log
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

        # All retries exhausted
        raise RetryExhausted(
            f"Request failed after {self.config.max_retries + 1} attempts",
            last_exception=last_exception
        )

    async def execute_with_retry_async(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a coroutine with retry logic (asynchronous).

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from successful function execution

        Raises:
            RetryExhausted: If all retry attempts are exhausted
        """
        if not self.config.enabled:
            return await func(*args, **kwargs)

        last_exception = None
        self.attempt_count = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"Request succeeded after {attempt} retry attempt(s)"
                    )
                return result

            except Exception as e:
                last_exception = e
                self.attempt_count = attempt + 1

                # Check if we should retry
                status_code = getattr(e, 'status_code', None)
                if not self.should_retry(attempt, status_code, e):
                    logger.error(
                        f"Request failed with non-retryable error: {e}"
                    )
                    raise

                # Calculate delay and log
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise RetryExhausted(
            f"Request failed after {self.config.max_retries + 1} attempts",
            last_exception=last_exception
        )

    def reset(self) -> None:
        """Reset the retry handler state."""
        self.attempt_count = 0

    def get_stats(self) -> dict:
        """
        Get retry handler statistics.

        Returns:
            Dictionary containing retry statistics
        """
        return {
            "enabled": self.config.enabled,
            "max_retries": self.config.max_retries,
            "current_attempt": self.attempt_count,
            "initial_delay": self.config.initial_delay,
            "max_delay": self.config.max_delay,
            "exponential_base": self.config.exponential_base,
            "jitter_enabled": self.config.jitter,
            "retry_status_codes": self.config.retry_on_status_codes,
        }


def retry_on_exception(
    config: RetryConfig,
    exceptions: Union[type, tuple] = Exception
):
    """
    Decorator for retrying functions on specific exceptions.

    Args:
        config: Retry configuration
        exceptions: Exception type(s) to catch and retry

    Returns:
        Decorator function

    Example:
        @retry_on_exception(retry_config, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            # ... code that might fail
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = RetryHandler(config)

            def _func():
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Re-raise so retry handler can catch it
                    raise e

            return handler.execute_with_retry(_func)

        return wrapper
    return decorator


def async_retry_on_exception(
    config: RetryConfig,
    exceptions: Union[type, tuple] = Exception
):
    """
    Decorator for retrying async functions on specific exceptions.

    Args:
        config: Retry configuration
        exceptions: Exception type(s) to catch and retry

    Returns:
        Decorator function

    Example:
        @async_retry_on_exception(retry_config, exceptions=(ConnectionError,))
        async def fetch_data():
            # ... async code that might fail
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(config)

            async def _func():
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    # Re-raise so retry handler can catch it
                    raise e

            return await handler.execute_with_retry_async(_func)

        return wrapper
    return decorator
