"""
JWT (JSON Web Token) handling for authentication tokens.

This module provides comprehensive JWT token generation, validation, and
management for secure authentication and session handling.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets

from .exceptions import InvalidTokenError, SessionExpiredError


class JWTHandler:
    """
    Handles JWT token generation, validation, and decoding.

    Provides secure token-based authentication with configurable expiration,
    refresh tokens, and token revocation support.
    """

    # Default token expiration times
    DEFAULT_ACCESS_TOKEN_EXPIRY = timedelta(hours=1)
    DEFAULT_REFRESH_TOKEN_EXPIRY = timedelta(days=7)

    # JWT algorithms
    ALGORITHM = "HS256"

    def __init__(
        self,
        secret_key: str,
        algorithm: str = ALGORITHM,
        access_token_expiry: Optional[timedelta] = None,
        refresh_token_expiry: Optional[timedelta] = None,
    ):
        """
        Initialize JWT handler.

        Args:
            secret_key: Secret key for signing tokens. Should be cryptographically secure.
            algorithm: JWT signing algorithm (default: HS256)
            access_token_expiry: Access token expiration time (default: 1 hour)
            refresh_token_expiry: Refresh token expiration time (default: 7 days)

        Raises:
            ValueError: If secret key is too short or empty
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters long")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expiry = access_token_expiry or self.DEFAULT_ACCESS_TOKEN_EXPIRY
        self.refresh_token_expiry = refresh_token_expiry or self.DEFAULT_REFRESH_TOKEN_EXPIRY

    def generate_access_token(
        self,
        user_id: str,
        username: str,
        roles: list[str],
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate an access token for authenticated user.

        Args:
            user_id: Unique user identifier
            username: Username
            roles: List of role names assigned to user
            additional_claims: Optional additional claims to include in token

        Returns:
            Encoded JWT access token as string

        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not username:
            raise ValueError("user_id and username are required")

        now = datetime.utcnow()
        expiry = now + self.access_token_expiry

        # Standard JWT claims
        payload = {
            "sub": user_id,  # Subject (user ID)
            "username": username,
            "roles": roles,
            "iat": now,  # Issued at
            "exp": expiry,  # Expiration
            "nbf": now,  # Not before
            "jti": secrets.token_urlsafe(16),  # JWT ID (unique token identifier)
            "type": "access",
        }

        # Add additional claims if provided
        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def generate_refresh_token(
        self,
        user_id: str,
        username: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a refresh token for token renewal.

        Args:
            user_id: Unique user identifier
            username: Username
            additional_claims: Optional additional claims to include in token

        Returns:
            Encoded JWT refresh token as string

        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not username:
            raise ValueError("user_id and username are required")

        now = datetime.utcnow()
        expiry = now + self.refresh_token_expiry

        payload = {
            "sub": user_id,
            "username": username,
            "iat": now,
            "exp": expiry,
            "nbf": now,
            "jti": secrets.token_urlsafe(16),
            "type": "refresh",
        }

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str, verify: bool = True) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token to decode
            verify: Whether to verify signature and expiration (default: True)

        Returns:
            Dictionary containing decoded token payload

        Raises:
            InvalidTokenError: If token is invalid or malformed
            SessionExpiredError: If token has expired
        """
        if not token:
            raise InvalidTokenError("Token cannot be empty")

        try:
            options = {"verify_signature": verify, "verify_exp": verify}

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options=options,
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise SessionExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            raise InvalidTokenError(f"Token decode error: {str(e)}")

    def validate_token(self, token: str) -> tuple[bool, Optional[str]]:
        """
        Validate a token and return status.

        Args:
            token: JWT token to validate

        Returns:
            Tuple of (is_valid, error_message)
            is_valid: True if token is valid, False otherwise
            error_message: Error description if invalid, None if valid
        """
        try:
            self.decode_token(token, verify=True)
            return True, None
        except SessionExpiredError:
            return False, "Token has expired"
        except InvalidTokenError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def extract_user_id(self, token: str) -> str:
        """
        Extract user ID from token without full validation.

        Useful for logging or initial checks before full validation.

        Args:
            token: JWT token

        Returns:
            User ID from token

        Raises:
            InvalidTokenError: If token is invalid or missing user ID
        """
        try:
            # Decode without verification for quick extraction
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            user_id = payload.get("sub")
            if not user_id:
                raise InvalidTokenError("Token missing user ID (sub claim)")
            return user_id
        except Exception as e:
            raise InvalidTokenError(f"Cannot extract user ID: {str(e)}")

    def extract_username(self, token: str) -> str:
        """
        Extract username from token without full validation.

        Args:
            token: JWT token

        Returns:
            Username from token

        Raises:
            InvalidTokenError: If token is invalid or missing username
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            username = payload.get("username")
            if not username:
                raise InvalidTokenError("Token missing username")
            return username
        except Exception as e:
            raise InvalidTokenError(f"Cannot extract username: {str(e)}")

    def get_token_expiry(self, token: str) -> datetime:
        """
        Get expiration time from token.

        Args:
            token: JWT token

        Returns:
            Expiration datetime

        Raises:
            InvalidTokenError: If token is invalid or missing expiry
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            exp = payload.get("exp")
            if not exp:
                raise InvalidTokenError("Token missing expiration (exp claim)")
            return datetime.fromtimestamp(exp)
        except InvalidTokenError:
            raise
        except Exception as e:
            raise InvalidTokenError(f"Cannot extract expiry: {str(e)}")

    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired.

        Args:
            token: JWT token

        Returns:
            True if token is expired, False otherwise
        """
        try:
            expiry = self.get_token_expiry(token)
            return datetime.utcnow() > expiry
        except InvalidTokenError:
            return True

    def get_remaining_time(self, token: str) -> timedelta:
        """
        Get remaining time until token expiration.

        Args:
            token: JWT token

        Returns:
            Timedelta representing remaining time (negative if expired)

        Raises:
            InvalidTokenError: If token is invalid
        """
        expiry = self.get_token_expiry(token)
        remaining = expiry - datetime.utcnow()
        return remaining

    @staticmethod
    def generate_secret_key(length: int = 64) -> str:
        """
        Generate a cryptographically secure secret key.

        Args:
            length: Length of the key in bytes (default: 64)

        Returns:
            URL-safe base64-encoded secret key
        """
        if length < 32:
            raise ValueError("Secret key must be at least 32 bytes")
        return secrets.token_urlsafe(length)


# Token blacklist for revoked tokens (in-memory storage)
# In production, use Redis or database for distributed systems
class TokenBlacklist:
    """
    Token blacklist for managing revoked tokens.

    In production environments, this should be backed by Redis, database,
    or other persistent storage for distributed systems.
    """

    def __init__(self):
        """Initialize token blacklist."""
        self._blacklist: set[str] = set()

    def add_token(self, token: str) -> None:
        """
        Add a token to the blacklist.

        Args:
            token: Token to blacklist
        """
        self._blacklist.add(token)

    def is_blacklisted(self, token: str) -> bool:
        """
        Check if token is blacklisted.

        Args:
            token: Token to check

        Returns:
            True if token is blacklisted, False otherwise
        """
        return token in self._blacklist

    def remove_token(self, token: str) -> None:
        """
        Remove a token from blacklist.

        Args:
            token: Token to remove
        """
        self._blacklist.discard(token)

    def clear(self) -> None:
        """Clear all blacklisted tokens."""
        self._blacklist.clear()

    def size(self) -> int:
        """
        Get number of blacklisted tokens.

        Returns:
            Number of blacklisted tokens
        """
        return len(self._blacklist)
