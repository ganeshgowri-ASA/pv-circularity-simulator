"""
Configuration settings for Authentication & Access Control System.

This module provides configuration management with environment variable
support and secure defaults.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AuthConfig:
    """Configuration class for authentication system."""

    # JWT Secret Key - MUST be set in production via environment variable
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"
    SECRET_KEY = os.getenv(
        "AUTH_SECRET_KEY",
        # Default key for development ONLY - DO NOT use in production!
        "dev-secret-key-change-in-production-" + "x" * 32,
    )

    # Password hashing settings
    BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))  # 12-14 recommended for production

    # Token expiration settings
    ACCESS_TOKEN_EXPIRY_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRY_HOURS", "1"))
    REFRESH_TOKEN_EXPIRY_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "7"))

    # Session settings
    SESSION_DURATION_HOURS = int(os.getenv("SESSION_DURATION_HOURS", "1"))
    MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "5"))

    # Account security settings
    MAX_FAILED_LOGIN_ATTEMPTS = int(os.getenv("MAX_FAILED_LOGIN_ATTEMPTS", "5"))
    ACCOUNT_LOCKOUT_MINUTES = int(os.getenv("ACCOUNT_LOCKOUT_MINUTES", "30"))

    # Password validation settings
    PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
    PASSWORD_REQUIRE_UPPERCASE = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_LOWERCASE = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_DIGITS = os.getenv("PASSWORD_REQUIRE_DIGITS", "true").lower() == "true"
    PASSWORD_REQUIRE_SPECIAL = os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_AUTH_EVENTS = os.getenv("LOG_AUTH_EVENTS", "true").lower() == "true"

    # Database (for future implementation)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///auth.db")

    @classmethod
    def get_access_token_expiry(cls) -> timedelta:
        """Get access token expiry as timedelta."""
        return timedelta(hours=cls.ACCESS_TOKEN_EXPIRY_HOURS)

    @classmethod
    def get_refresh_token_expiry(cls) -> timedelta:
        """Get refresh token expiry as timedelta."""
        return timedelta(days=cls.REFRESH_TOKEN_EXPIRY_DAYS)

    @classmethod
    def get_session_duration(cls) -> timedelta:
        """Get session duration as timedelta."""
        return timedelta(hours=cls.SESSION_DURATION_HOURS)

    @classmethod
    def get_lockout_duration(cls) -> timedelta:
        """Get account lockout duration as timedelta."""
        return timedelta(minutes=cls.ACCOUNT_LOCKOUT_MINUTES)

    @classmethod
    def validate_config(cls) -> list[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check secret key
        if len(cls.SECRET_KEY) < 32:
            errors.append("SECRET_KEY must be at least 32 characters")

        if "change-in-production" in cls.SECRET_KEY:
            errors.append("WARNING: Using default SECRET_KEY - set AUTH_SECRET_KEY environment variable!")

        # Check bcrypt rounds
        if not 4 <= cls.BCRYPT_ROUNDS <= 31:
            errors.append("BCRYPT_ROUNDS must be between 4 and 31")

        # Check password settings
        if cls.PASSWORD_MIN_LENGTH < 4:
            errors.append("PASSWORD_MIN_LENGTH must be at least 4")

        return errors


# Create default instance
config = AuthConfig()
