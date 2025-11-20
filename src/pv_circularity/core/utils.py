"""
Common utility functions used across the application.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

import pytz

from .exceptions import PVCircularityError
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def get_utc_now() -> datetime:
    """
    Get current UTC datetime with timezone awareness.

    Returns:
        Current UTC datetime

    Example:
        >>> now = get_utc_now()
        >>> print(now.tzinfo)
        UTC
    """
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC.

    Args:
        dt: Datetime to convert (can be naive or timezone-aware)

    Returns:
        Timezone-aware UTC datetime

    Example:
        >>> local_time = datetime.now()
        >>> utc_time = to_utc(local_time)
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def from_timestamp(timestamp: Union[int, float], tz: Optional[str] = None) -> datetime:
    """
    Convert Unix timestamp to datetime.

    Args:
        timestamp: Unix timestamp (seconds since epoch)
        tz: Timezone name (e.g., 'UTC', 'US/Pacific'). If None, returns UTC.

    Returns:
        Timezone-aware datetime

    Example:
        >>> dt = from_timestamp(1234567890)
        >>> dt_pacific = from_timestamp(1234567890, tz='US/Pacific')
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if tz and tz != "UTC":
        target_tz = pytz.timezone(tz)
        dt = dt.astimezone(target_tz)
    return dt


def to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp (seconds since epoch)

    Example:
        >>> dt = get_utc_now()
        >>> timestamp = to_timestamp(dt)
    """
    return dt.timestamp()


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator to retry a function on exception with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay (exponential backoff)
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function

    Example:
        >>> @retry_on_exception(max_retries=3, delay=1.0, backoff=2.0)
        ... async def fetch_data():
        ...     # Code that might fail
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Function call failed, retrying",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=current_delay,
                            error=str(e),
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "Function call failed after all retries",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e),
                        )

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Function call failed, retrying",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=current_delay,
                            error=str(e),
                        )
                        import time

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "Function call failed after all retries",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e),
                        )

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def chunks(data: list[T], chunk_size: int) -> list[list[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        data: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> list(chunks(data, 2))
        [[1, 2], [3, 4], [5]]
    """
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero

    Returns:
        Result of division or default value

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=None)
        None
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.

    Args:
        value: Value to clamp
        min_value: Minimum bound
        max_value: Maximum bound

    Returns:
        Clamped value

    Example:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_value, min(value, max_value))


def percentage(part: float, whole: float, precision: int = 2) -> float:
    """
    Calculate percentage with safe division.

    Args:
        part: Part value
        whole: Whole value
        precision: Number of decimal places

    Returns:
        Percentage value

    Example:
        >>> percentage(25, 100)
        25.0
        >>> percentage(1, 3)
        33.33
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, precision)


async def run_periodic(
    func: Callable,
    interval_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Run an async function periodically at specified interval.

    Args:
        func: Async function to run
        interval_seconds: Interval between executions in seconds
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Example:
        >>> async def monitor():
        ...     print("Monitoring...")
        >>> await run_periodic(monitor, interval_seconds=5)
    """
    while True:
        try:
            await func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Error in periodic task",
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
        await asyncio.sleep(interval_seconds)


def validate_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> None:
    """
    Validate that a value is within specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value for error messages

    Raises:
        ValueError: If value is outside the specified range

    Example:
        >>> validate_range(5, min_value=0, max_value=10)  # OK
        >>> validate_range(-5, min_value=0, max_value=10)  # Raises ValueError
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
