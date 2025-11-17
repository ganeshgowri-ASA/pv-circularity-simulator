"""
Authentication handlers for various API authentication schemes.

This module provides a flexible authentication system supporting multiple
authentication types including API keys, Bearer tokens, OAuth2, and Basic auth.
"""

import asyncio
import base64
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlencode

import httpx

from .models import AuthConfig, AuthType


class AuthenticationHandler(ABC):
    """
    Abstract base class for authentication handlers.

    This class defines the interface that all authentication handlers must implement.
    Each handler is responsible for adding the appropriate authentication credentials
    to HTTP requests.
    """

    def __init__(self, config: AuthConfig):
        """
        Initialize the authentication handler.

        Args:
            config: Authentication configuration
        """
        self.config = config

    @abstractmethod
    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply authentication to request headers.

        Args:
            headers: Existing request headers

        Returns:
            Updated headers with authentication credentials
        """
        pass

    @abstractmethod
    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply authentication to request headers asynchronously.

        Args:
            headers: Existing request headers

        Returns:
            Updated headers with authentication credentials
        """
        pass

    def get_additional_headers(self) -> Dict[str, str]:
        """
        Get additional headers configured in auth config.

        Returns:
            Dictionary of additional headers
        """
        return self.config.additional_headers.copy()


class NoAuth(AuthenticationHandler):
    """
    No authentication handler.

    This handler does not add any authentication credentials to requests.
    """

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply no authentication (return headers unchanged).

        Args:
            headers: Existing request headers

        Returns:
            Headers with additional configured headers
        """
        headers.update(self.get_additional_headers())
        return headers

    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply no authentication asynchronously.

        Args:
            headers: Existing request headers

        Returns:
            Headers with additional configured headers
        """
        return self.apply_auth(headers)


class APIKeyAuth(AuthenticationHandler):
    """
    API Key authentication handler.

    This handler adds an API key to request headers using a configurable header name.
    """

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply API key authentication to request headers.

        Args:
            headers: Existing request headers

        Returns:
            Headers with API key added

        Raises:
            ValueError: If API key is not configured
        """
        if not self.config.api_key:
            raise ValueError("API key not configured")

        headers[self.config.api_key_header] = self.config.api_key.get_secret_value()
        headers.update(self.get_additional_headers())
        return headers

    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply API key authentication asynchronously.

        Args:
            headers: Existing request headers

        Returns:
            Headers with API key added
        """
        return self.apply_auth(headers)


class BearerTokenAuth(AuthenticationHandler):
    """
    Bearer token authentication handler.

    This handler adds a Bearer token to the Authorization header.
    """

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply Bearer token authentication to request headers.

        Args:
            headers: Existing request headers

        Returns:
            Headers with Bearer token added

        Raises:
            ValueError: If token is not configured
        """
        if not self.config.token:
            raise ValueError("Bearer token not configured")

        headers["Authorization"] = f"Bearer {self.config.token.get_secret_value()}"
        headers.update(self.get_additional_headers())
        return headers

    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply Bearer token authentication asynchronously.

        Args:
            headers: Existing request headers

        Returns:
            Headers with Bearer token added
        """
        return self.apply_auth(headers)


class BasicAuth(AuthenticationHandler):
    """
    Basic authentication handler.

    This handler adds Basic authentication credentials to the Authorization header.
    """

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply Basic authentication to request headers.

        Args:
            headers: Existing request headers

        Returns:
            Headers with Basic auth credentials added

        Raises:
            ValueError: If username or password is not configured
        """
        if not self.config.username or not self.config.password:
            raise ValueError("Username and password required for Basic auth")

        credentials = f"{self.config.username}:{self.config.password.get_secret_value()}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        headers.update(self.get_additional_headers())
        return headers

    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply Basic authentication asynchronously.

        Args:
            headers: Existing request headers

        Returns:
            Headers with Basic auth credentials added
        """
        return self.apply_auth(headers)


class OAuth2Auth(AuthenticationHandler):
    """
    OAuth2 authentication handler with automatic token refresh.

    This handler manages OAuth2 tokens, automatically refreshing them when expired.
    It uses the client credentials flow to obtain access tokens.
    """

    def __init__(self, config: AuthConfig):
        """
        Initialize OAuth2 authentication handler.

        Args:
            config: Authentication configuration

        Raises:
            ValueError: If required OAuth2 configuration is missing
        """
        super().__init__(config)
        if not config.client_id or not config.client_secret or not config.token_url:
            raise ValueError(
                "OAuth2 requires client_id, client_secret, and token_url"
            )

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._lock = asyncio.Lock()

    def _is_token_valid(self) -> bool:
        """
        Check if the current access token is valid.

        Returns:
            True if token exists and is not expired
        """
        if not self._access_token or not self._token_expiry:
            return False
        # Add 60 second buffer to avoid using token right before expiry
        return datetime.utcnow() < (self._token_expiry - timedelta(seconds=60))

    def _request_token(self) -> Dict[str, any]:
        """
        Request a new access token from the OAuth2 token endpoint.

        Returns:
            Token response data

        Raises:
            Exception: If token request fails
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret.get_secret_value(),
        }

        with httpx.Client() as client:
            response = client.post(
                str(self.config.token_url),
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return response.json()

    async def _request_token_async(self) -> Dict[str, any]:
        """
        Request a new access token asynchronously.

        Returns:
            Token response data

        Raises:
            Exception: If token request fails
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret.get_secret_value(),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                str(self.config.token_url),
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return response.json()

    def _update_token(self, token_data: Dict[str, any]) -> None:
        """
        Update the stored access token and expiry time.

        Args:
            token_data: Token response data containing access_token and expires_in
        """
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply OAuth2 authentication to request headers.

        This method will automatically request a new token if needed.

        Args:
            headers: Existing request headers

        Returns:
            Headers with OAuth2 Bearer token added

        Raises:
            Exception: If token request fails
        """
        if not self._is_token_valid():
            token_data = self._request_token()
            self._update_token(token_data)

        headers["Authorization"] = f"Bearer {self._access_token}"
        headers.update(self.get_additional_headers())
        return headers

    async def apply_auth_async(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Apply OAuth2 authentication asynchronously with automatic token refresh.

        This method uses a lock to prevent multiple concurrent token refresh requests.

        Args:
            headers: Existing request headers

        Returns:
            Headers with OAuth2 Bearer token added

        Raises:
            Exception: If token request fails
        """
        async with self._lock:
            if not self._is_token_valid():
                token_data = await self._request_token_async()
                self._update_token(token_data)

        headers["Authorization"] = f"Bearer {self._access_token}"
        headers.update(self.get_additional_headers())
        return headers


def create_auth_handler(config: AuthConfig) -> AuthenticationHandler:
    """
    Factory function to create the appropriate authentication handler.

    Args:
        config: Authentication configuration

    Returns:
        Authentication handler instance

    Raises:
        ValueError: If auth_type is not supported
    """
    handlers = {
        AuthType.NONE: NoAuth,
        AuthType.API_KEY: APIKeyAuth,
        AuthType.BEARER_TOKEN: BearerTokenAuth,
        AuthType.BASIC: BasicAuth,
        AuthType.OAUTH2: OAuth2Auth,
    }

    handler_class = handlers.get(config.auth_type)
    if not handler_class:
        raise ValueError(f"Unsupported auth type: {config.auth_type}")

    return handler_class(config)
