"""
Base weather API client with rate limiting and error handling.

This module provides the foundation for all weather API client implementations,
including automatic retries, rate limiting, and standardized error handling.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pv_simulator.config import Settings
from pv_simulator.models.weather import (
    CurrentWeather,
    ForecastWeather,
    GeoLocation,
    HistoricalWeather,
    WeatherProvider,
)
from pv_simulator.weather.cache import CacheManager

logger = logging.getLogger(__name__)


class WeatherAPIException(Exception):
    """Base exception for weather API errors."""

    pass


class RateLimitExceeded(WeatherAPIException):
    """Exception raised when API rate limit is exceeded."""

    pass


class APIAuthenticationError(WeatherAPIException):
    """Exception raised when API authentication fails."""

    pass


class APIRequestError(WeatherAPIException):
    """Exception raised when API request fails."""

    pass


class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Implements a thread-safe token bucket algorithm to enforce rate limits
    on API requests.
    """

    def __init__(self, requests_per_minute: int) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_update = time.time()
        self.max_tokens = float(requests_per_minute)
        self.refill_rate = requests_per_minute / 60.0  # tokens per second

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for a request.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_for_token(self, timeout: float = 60.0) -> bool:
        """
        Wait for a token to become available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if token was acquired, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.acquire():
                return True
            time.sleep(0.1)

        return False

    def get_wait_time(self) -> float:
        """
        Get estimated wait time for next token.

        Returns:
            Wait time in seconds
        """
        self._refill_tokens()

        if self.tokens >= 1:
            return 0.0

        return (1.0 - self.tokens) / self.refill_rate


class BaseWeatherClient(ABC):
    """
    Abstract base class for weather API clients.

    Provides common functionality including rate limiting, caching,
    retries, and error handling.
    """

    def __init__(
        self,
        settings: Settings,
        cache_manager: CacheManager,
        provider: WeatherProvider,
    ) -> None:
        """
        Initialize base weather client.

        Args:
            settings: Application settings
            cache_manager: Cache manager instance
            provider: Weather provider identifier
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.provider = provider
        self.rate_limiter = RateLimiter(settings.get_rate_limit(provider.value))
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

        logger.info(f"Initialized {provider.value} weather client")

    def __del__(self) -> None:
        """Clean up HTTP clients."""
        try:
            self.client.close()
        except Exception:
            pass

    async def __aenter__(self) -> "BaseWeatherClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.async_client.aclose()

    def _check_rate_limit(self) -> None:
        """
        Check rate limit before making request.

        Raises:
            RateLimitExceeded: If rate limit is exceeded and token cannot be acquired
        """
        if not self.rate_limiter.wait_for_token(timeout=5.0):
            wait_time = self.rate_limiter.get_wait_time()
            raise RateLimitExceeded(
                f"Rate limit exceeded for {self.provider.value}. "
                f"Wait {wait_time:.1f}s before retry."
            )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    def _make_request(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Make HTTP GET request with retries.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            Response JSON data

        Raises:
            APIAuthenticationError: If authentication fails
            APIRequestError: If request fails
        """
        self._check_rate_limit()

        try:
            response = self.client.get(url, params=params, headers=headers)

            if response.status_code == 401:
                raise APIAuthenticationError(
                    f"Authentication failed for {self.provider.value}"
                )
            elif response.status_code == 429:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.provider.value}"
                )
            elif response.status_code >= 400:
                raise APIRequestError(
                    f"Request failed with status {response.status_code}: {response.text}"
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {self.provider.value}: {e}")
            raise APIRequestError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {self.provider.value}: {e}")
            raise APIRequestError(f"Request error: {e}") from e

    def _get_cache_key(self, *parts: Any) -> str:
        """
        Generate cache key for request.

        Args:
            *parts: Key components

        Returns:
            Cache key string
        """
        return self.cache_manager.generate_key(self.provider.value, *parts)

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None
        """
        cached = self.cache_manager.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {self.provider.value}: {cache_key}")
        return cached

    def _save_to_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Save data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds
        """
        self.cache_manager.set(cache_key, data, ttl)
        logger.debug(f"Cached data for {self.provider.value}: {cache_key}")

    @abstractmethod
    def get_current_weather(self, location: GeoLocation) -> CurrentWeather:
        """
        Get current weather conditions for a location.

        Args:
            location: Geographic location

        Returns:
            Current weather data

        Raises:
            WeatherAPIException: If request fails
        """
        pass

    @abstractmethod
    def get_forecast(
        self, location: GeoLocation, days: int = 7
    ) -> ForecastWeather:
        """
        Get weather forecast for a location.

        Args:
            location: Geographic location
            days: Number of days to forecast (default: 7)

        Returns:
            Weather forecast data

        Raises:
            WeatherAPIException: If request fails
        """
        pass

    @abstractmethod
    def get_historical(
        self,
        location: GeoLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalWeather:
        """
        Get historical weather data for a location.

        Args:
            location: Geographic location
            start_date: Start of historical period
            end_date: End of historical period

        Returns:
            Historical weather data

        Raises:
            WeatherAPIException: If request fails
        """
        pass

    def check_availability(self) -> bool:
        """
        Check if API is available and credentials are valid.

        Returns:
            True if API is available, False otherwise
        """
        try:
            # Try a simple request to verify API availability
            test_location = GeoLocation(latitude=0.0, longitude=0.0)
            self.get_current_weather(test_location)
            return True
        except Exception as e:
            logger.warning(f"API availability check failed for {self.provider.value}: {e}")
            return False
