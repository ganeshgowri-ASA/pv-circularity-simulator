"""Tests for JWT token handling."""

import pytest
import time
from datetime import timedelta

from src.auth.jwt_handler import JWTHandler, TokenBlacklist
from src.auth.exceptions import InvalidTokenError, SessionExpiredError


class TestJWTHandler:
    """Tests for JWTHandler class."""

    @pytest.fixture
    def handler(self):
        """Create JWTHandler instance."""
        secret_key = "test-secret-key-" + "x" * 32
        return JWTHandler(
            secret_key=secret_key,
            access_token_expiry=timedelta(seconds=10),
            refresh_token_expiry=timedelta(seconds=20),
        )

    def test_generate_access_token(self, handler):
        """Test access token generation."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin", "user"],
        )

        assert token is not None
        assert isinstance(token, str)

    def test_generate_refresh_token(self, handler):
        """Test refresh token generation."""
        token = handler.generate_refresh_token(
            user_id="user123",
            username="testuser",
        )

        assert token is not None
        assert isinstance(token, str)

    def test_decode_token(self, handler):
        """Test token decoding."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        payload = handler.decode_token(token)

        assert payload["sub"] == "user123"
        assert payload["username"] == "testuser"
        assert payload["roles"] == ["admin"]
        assert payload["type"] == "access"

    def test_decode_expired_token(self, handler):
        """Test decoding expired token."""
        # Create handler with very short expiry
        short_handler = JWTHandler(
            secret_key="test-secret-key-" + "x" * 32,
            access_token_expiry=timedelta(seconds=1),
        )

        token = short_handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        # Wait for token to expire
        time.sleep(2)

        with pytest.raises(SessionExpiredError):
            short_handler.decode_token(token)

    def test_validate_token(self, handler):
        """Test token validation."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        is_valid, error = handler.validate_token(token)

        assert is_valid is True
        assert error is None

    def test_validate_invalid_token(self, handler):
        """Test validating invalid token."""
        is_valid, error = handler.validate_token("invalid.token.here")

        assert is_valid is False
        assert error is not None

    def test_extract_user_id(self, handler):
        """Test extracting user ID from token."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        user_id = handler.extract_user_id(token)
        assert user_id == "user123"

    def test_extract_username(self, handler):
        """Test extracting username from token."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        username = handler.extract_username(token)
        assert username == "testuser"

    def test_get_token_expiry(self, handler):
        """Test getting token expiry time."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        expiry = handler.get_token_expiry(token)
        assert expiry is not None

    def test_is_token_expired(self, handler):
        """Test checking if token is expired."""
        token = handler.generate_access_token(
            user_id="user123",
            username="testuser",
            roles=["admin"],
        )

        assert handler.is_token_expired(token) is False

    def test_generate_secret_key(self):
        """Test secret key generation."""
        key = JWTHandler.generate_secret_key(64)

        assert key is not None
        assert len(key) >= 64


class TestTokenBlacklist:
    """Tests for TokenBlacklist class."""

    @pytest.fixture
    def blacklist(self):
        """Create TokenBlacklist instance."""
        return TokenBlacklist()

    def test_add_token(self, blacklist):
        """Test adding token to blacklist."""
        token = "test.token.here"
        blacklist.add_token(token)

        assert blacklist.is_blacklisted(token) is True

    def test_is_blacklisted(self, blacklist):
        """Test checking if token is blacklisted."""
        token = "test.token.here"

        assert blacklist.is_blacklisted(token) is False

        blacklist.add_token(token)

        assert blacklist.is_blacklisted(token) is True

    def test_remove_token(self, blacklist):
        """Test removing token from blacklist."""
        token = "test.token.here"
        blacklist.add_token(token)

        assert blacklist.is_blacklisted(token) is True

        blacklist.remove_token(token)

        assert blacklist.is_blacklisted(token) is False

    def test_clear(self, blacklist):
        """Test clearing blacklist."""
        blacklist.add_token("token1")
        blacklist.add_token("token2")

        assert blacklist.size() == 2

        blacklist.clear()

        assert blacklist.size() == 0

    def test_size(self, blacklist):
        """Test getting blacklist size."""
        assert blacklist.size() == 0

        blacklist.add_token("token1")
        assert blacklist.size() == 1

        blacklist.add_token("token2")
        assert blacklist.size() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
