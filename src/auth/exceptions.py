"""
Custom exceptions for authentication and authorization system.
"""

from typing import Optional


class AuthenticationError(Exception):
    """Base exception for authentication-related errors."""

    def __init__(self, message: str = "Authentication failed", details: Optional[str] = None):
        """
        Initialize authentication error.

        Args:
            message: Main error message
            details: Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
        user_permissions: Optional[list[str]] = None,
    ):
        """
        Initialize authorization error.

        Args:
            message: Main error message
            required_permission: The permission that was required
            user_permissions: The permissions the user has
        """
        self.message = message
        self.required_permission = required_permission
        self.user_permissions = user_permissions or []
        super().__init__(self.message)


class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid or malformed."""

    def __init__(self, message: str = "Invalid or malformed token"):
        """
        Initialize invalid token error.

        Args:
            message: Error message
        """
        super().__init__(message)


class SessionExpiredError(AuthenticationError):
    """Raised when user session has expired."""

    def __init__(self, message: str = "Session has expired"):
        """
        Initialize session expired error.

        Args:
            message: Error message
        """
        super().__init__(message)


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""

    def __init__(self, message: str = "Invalid username or password"):
        """
        Initialize invalid credentials error.

        Args:
            message: Error message
        """
        super().__init__(message)


class UserNotFoundError(AuthenticationError):
    """Raised when user cannot be found in the system."""

    def __init__(self, username: str):
        """
        Initialize user not found error.

        Args:
            username: Username that was not found
        """
        super().__init__(f"User '{username}' not found")
        self.username = username


class DuplicateUserError(AuthenticationError):
    """Raised when attempting to create a user that already exists."""

    def __init__(self, username: str):
        """
        Initialize duplicate user error.

        Args:
            username: Username that already exists
        """
        super().__init__(f"User '{username}' already exists")
        self.username = username
