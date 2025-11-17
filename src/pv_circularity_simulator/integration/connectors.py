"""
API connectors for making HTTP/REST requests with authentication and retry logic.

This module provides flexible API connectors that support various HTTP methods,
authentication schemes, rate limiting, and automatic retry with exponential backoff.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from .auth import AuthenticationHandler, create_auth_handler
from .models import (
    APIRequest,
    APIResponse,
    HTTPMethod,
    IntegrationConfig,
)
from .rate_limiter import RateLimiter
from .retry import RetryHandler, RetryExhausted

logger = logging.getLogger(__name__)


class APIConnector(ABC):
    """
    Abstract base class for API connectors.

    This class defines the interface for all API connectors and provides
    common functionality for making HTTP requests with authentication,
    rate limiting, and retry logic.

    Attributes:
        config: Integration configuration
        auth_handler: Authentication handler
        rate_limiter: Rate limiter instance
        retry_handler: Retry handler instance
    """

    def __init__(self, config: IntegrationConfig):
        """
        Initialize the API connector.

        Args:
            config: Integration configuration
        """
        self.config = config
        self.auth_handler = create_auth_handler(config.auth)
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.retry_handler = RetryHandler(config.retry)

    @abstractmethod
    def request(self, request: APIRequest) -> APIResponse:
        """
        Make a synchronous API request.

        Args:
            request: API request parameters

        Returns:
            API response
        """
        pass

    @abstractmethod
    async def request_async(self, request: APIRequest) -> APIResponse:
        """
        Make an asynchronous API request.

        Args:
            request: API request parameters

        Returns:
            API response
        """
        pass

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from base URL and endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL
        """
        # Ensure base_url ends with / and endpoint doesn't start with /
        base = str(self.config.base_url).rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base}/{endpoint}"

    def _prepare_headers(
        self,
        request: APIRequest,
        auth_headers: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Prepare request headers by merging default, auth, and request-specific headers.

        Args:
            request: API request parameters
            auth_headers: Headers from authentication handler

        Returns:
            Merged headers dictionary
        """
        headers = {}

        # Start with default headers from config
        headers.update(self.config.default_headers)

        # Add authentication headers
        headers.update(auth_headers)

        # Add request-specific headers (highest priority)
        if request.headers:
            headers.update(request.headers)

        return headers

    def _create_response(
        self,
        response: httpx.Response,
        start_time: float,
        retry_count: int = 0
    ) -> APIResponse:
        """
        Create an APIResponse object from httpx.Response.

        Args:
            response: httpx Response object
            start_time: Request start time
            retry_count: Number of retries attempted

        Returns:
            APIResponse object
        """
        elapsed_time = time.time() - start_time
        success = 200 <= response.status_code < 300

        # Try to parse JSON response
        json_data = None
        try:
            json_data = response.json()
        except Exception:
            pass

        return APIResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            data=response.text,
            json_data=json_data,
            success=success,
            error_message=None if success else response.text,
            elapsed_time=elapsed_time,
            timestamp=datetime.utcnow(),
            retry_count=retry_count,
        )

    def _create_error_response(
        self,
        exception: Exception,
        start_time: float,
        retry_count: int = 0
    ) -> APIResponse:
        """
        Create an APIResponse object from an exception.

        Args:
            exception: Exception that occurred
            start_time: Request start time
            retry_count: Number of retries attempted

        Returns:
            APIResponse object representing the error
        """
        elapsed_time = time.time() - start_time

        return APIResponse(
            status_code=getattr(exception, 'status_code', 0),
            headers={},
            data=None,
            json_data=None,
            success=False,
            error_message=str(exception),
            elapsed_time=elapsed_time,
            timestamp=datetime.utcnow(),
            retry_count=retry_count,
        )


class RESTConnector(APIConnector):
    """
    REST API connector with full HTTP method support.

    This connector provides a complete implementation for making REST API
    requests with support for all standard HTTP methods, authentication,
    rate limiting, and automatic retry with exponential backoff.
    """

    def request(self, request: APIRequest) -> APIResponse:
        """
        Make a synchronous REST API request.

        This method handles the complete request lifecycle including:
        - Rate limiting
        - Authentication
        - Request execution with retry logic
        - Response processing

        Args:
            request: API request parameters

        Returns:
            API response

        Raises:
            Exception: If request fails after all retries
        """
        start_time = time.time()

        # Acquire rate limit token
        self.rate_limiter.acquire()

        try:
            # Execute request with retry logic
            def _execute_request():
                return self._execute_sync_request(request)

            response = self.retry_handler.execute_with_retry(_execute_request)
            return self._create_response(
                response,
                start_time,
                retry_count=self.retry_handler.attempt_count
            )

        except RetryExhausted as e:
            logger.error(f"Request failed after retries: {e}")
            if e.last_exception:
                if isinstance(e.last_exception, httpx.HTTPStatusError):
                    return self._create_response(
                        e.last_exception.response,
                        start_time,
                        retry_count=self.retry_handler.attempt_count
                    )
                return self._create_error_response(
                    e.last_exception,
                    start_time,
                    retry_count=self.retry_handler.attempt_count
                )
            return self._create_error_response(
                e,
                start_time,
                retry_count=self.retry_handler.attempt_count
            )

        except Exception as e:
            logger.error(f"Unexpected error during request: {e}")
            return self._create_error_response(start_time=start_time, exception=e)

    async def request_async(self, request: APIRequest) -> APIResponse:
        """
        Make an asynchronous REST API request.

        This method handles the complete request lifecycle asynchronously:
        - Rate limiting
        - Authentication
        - Request execution with retry logic
        - Response processing

        Args:
            request: API request parameters

        Returns:
            API response

        Raises:
            Exception: If request fails after all retries
        """
        start_time = time.time()

        # Acquire rate limit token
        await self.rate_limiter.acquire_async()

        try:
            # Execute request with retry logic
            async def _execute_request():
                return await self._execute_async_request(request)

            response = await self.retry_handler.execute_with_retry_async(
                _execute_request
            )
            return self._create_response(
                response,
                start_time,
                retry_count=self.retry_handler.attempt_count
            )

        except RetryExhausted as e:
            logger.error(f"Request failed after retries: {e}")
            if e.last_exception:
                if isinstance(e.last_exception, httpx.HTTPStatusError):
                    return self._create_response(
                        e.last_exception.response,
                        start_time,
                        retry_count=self.retry_handler.attempt_count
                    )
                return self._create_error_response(
                    e.last_exception,
                    start_time,
                    retry_count=self.retry_handler.attempt_count
                )
            return self._create_error_response(
                e,
                start_time,
                retry_count=self.retry_handler.attempt_count
            )

        except Exception as e:
            logger.error(f"Unexpected error during request: {e}")
            return self._create_error_response(e, start_time)

    def _execute_sync_request(self, request: APIRequest) -> httpx.Response:
        """
        Execute a synchronous HTTP request.

        Args:
            request: API request parameters

        Returns:
            httpx Response object

        Raises:
            httpx.HTTPStatusError: If response status indicates error
        """
        # Apply authentication
        auth_headers = self.auth_handler.apply_auth({})
        headers = self._prepare_headers(request, auth_headers)

        # Build URL
        url = self._build_url(request.endpoint)

        # Determine timeout
        timeout = request.timeout or self.config.timeout

        # Make request
        with httpx.Client(verify=self.config.verify_ssl) as client:
            response = client.request(
                method=request.method.value,
                url=url,
                params=request.params,
                data=request.data,
                json=request.json,
                headers=headers,
                timeout=timeout,
            )

            # Check for HTTP errors and raise if retry should be attempted
            if response.status_code in self.config.retry.retry_on_status_codes:
                response.raise_for_status()

            return response

    async def _execute_async_request(self, request: APIRequest) -> httpx.Response:
        """
        Execute an asynchronous HTTP request.

        Args:
            request: API request parameters

        Returns:
            httpx Response object

        Raises:
            httpx.HTTPStatusError: If response status indicates error
        """
        # Apply authentication
        auth_headers = await self.auth_handler.apply_auth_async({})
        headers = self._prepare_headers(request, auth_headers)

        # Build URL
        url = self._build_url(request.endpoint)

        # Determine timeout
        timeout = request.timeout or self.config.timeout

        # Make request
        async with httpx.AsyncClient(verify=self.config.verify_ssl) as client:
            response = await client.request(
                method=request.method.value,
                url=url,
                params=request.params,
                data=request.data,
                json=request.json,
                headers=headers,
                timeout=timeout,
            )

            # Check for HTTP errors and raise if retry should be attempted
            if response.status_code in self.config.retry.retry_on_status_codes:
                response.raise_for_status()

            return response

    # Convenience methods for common HTTP methods

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.GET,
            endpoint=endpoint,
            params=params,
            **kwargs
        )
        return self.request(request)

    async def get_async(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an async GET request.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.GET,
            endpoint=endpoint,
            params=params,
            **kwargs
        )
        return await self.request_async(request)

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.POST,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return self.request(request)

    async def post_async(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an async POST request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.POST,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return await self.request_async(request)

    def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PUT,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return self.request(request)

    async def put_async(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an async PUT request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PUT,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return await self.request_async(request)

    def delete(
        self,
        endpoint: str,
        **kwargs
    ) -> APIResponse:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.DELETE,
            endpoint=endpoint,
            **kwargs
        )
        return self.request(request)

    async def delete_async(
        self,
        endpoint: str,
        **kwargs
    ) -> APIResponse:
        """
        Make an async DELETE request.

        Args:
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.DELETE,
            endpoint=endpoint,
            **kwargs
        )
        return await self.request_async(request)

    def patch(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make a PATCH request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PATCH,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return self.request(request)

    async def patch_async(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an async PATCH request.

        Args:
            endpoint: API endpoint path
            json: JSON request body
            data: Request body data
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PATCH,
            endpoint=endpoint,
            json=json,
            data=data,
            **kwargs
        )
        return await self.request_async(request)
