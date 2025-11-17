"""
Base API client for weather data services.

Provides common functionality for all API clients including
retry logic, error handling, and rate limiting.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pv_simulator.config.settings import settings

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """
    Base class for API clients with retry logic and error handling.

    Attributes:
        base_url: Base URL for the API
        api_key: API key for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """
        Initialize base API client.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: from settings)
            max_retries: Maximum retry attempts (default: from settings)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout or settings.api_timeout_seconds
        self.max_retries = max_retries or settings.max_api_retries

        # Create session with retry strategy
        self.session = self._create_session()

        logger.info(f"{self.__class__.__name__} initialized with base_url: {base_url}")

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry strategy.

        Returns:
            Configured requests session
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _build_headers(self) -> Dict[str, str]:
        """
        Build request headers.

        Returns:
            Dictionary of headers
        """
        headers = {
            "User-Agent": "pv-circularity-simulator/0.1.0",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """
        Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (appended to base_url)
            params: Query parameters
            data: Request body data
            headers: Additional headers

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Merge headers
        request_headers = self._build_headers()
        if headers:
            request_headers.update(headers)

        logger.debug(f"{method} {url} with params={params}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> requests.Response:
        """
        Make GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for _make_request

        Returns:
            Response object
        """
        return self._make_request("GET", endpoint, params=params, **kwargs)

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Make POST request.

        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            **kwargs: Additional arguments for _make_request

        Returns:
            Response object
        """
        return self._make_request("POST", endpoint, params=params, data=data, **kwargs)

    @abstractmethod
    def get_data(self, **params: Any) -> Dict[str, Any]:
        """
        Abstract method to fetch data from the API.

        Must be implemented by subclasses.

        Args:
            **params: API-specific parameters

        Returns:
            Dictionary containing fetched data
        """
        pass

    def close(self) -> None:
        """Close the session."""
        self.session.close()
        logger.info(f"{self.__class__.__name__} session closed")

    def __enter__(self) -> "BaseAPIClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
