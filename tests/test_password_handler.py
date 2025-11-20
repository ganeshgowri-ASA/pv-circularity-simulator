"""Tests for password handling."""

import pytest
from src.auth.password_handler import PasswordHandler


class TestPasswordHandler:
    """Tests for PasswordHandler class."""

    @pytest.fixture
    def handler(self):
        """Create PasswordHandler instance."""
        return PasswordHandler(rounds=4)  # Low rounds for faster tests

    def test_hash_password(self, handler):
        """Test password hashing."""
        password = "SecurePass123!"
        hashed = handler.hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self, handler):
        """Test password verification with correct password."""
        password = "SecurePass123!"
        hashed = handler.hash_password(password)

        assert handler.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self, handler):
        """Test password verification with incorrect password."""
        password = "SecurePass123!"
        hashed = handler.hash_password(password)

        assert handler.verify_password("WrongPassword!", hashed) is False

    def test_validate_password_strength_valid(self, handler):
        """Test password strength validation with valid password."""
        is_valid, errors = handler.validate_password_strength("SecurePass123!")

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_password_strength_too_short(self, handler):
        """Test password validation with too short password."""
        is_valid, errors = handler.validate_password_strength("Short1!")

        assert is_valid is False
        assert any("at least" in err for err in errors)

    def test_validate_password_strength_no_uppercase(self, handler):
        """Test password validation without uppercase."""
        is_valid, errors = handler.validate_password_strength("securepass123!")

        assert is_valid is False
        assert any("uppercase" in err for err in errors)

    def test_validate_password_strength_no_lowercase(self, handler):
        """Test password validation without lowercase."""
        is_valid, errors = handler.validate_password_strength("SECUREPASS123!")

        assert is_valid is False
        assert any("lowercase" in err for err in errors)

    def test_validate_password_strength_no_digits(self, handler):
        """Test password validation without digits."""
        is_valid, errors = handler.validate_password_strength("SecurePassword!")

        assert is_valid is False
        assert any("digit" in err for err in errors)

    def test_validate_password_strength_no_special(self, handler):
        """Test password validation without special characters."""
        is_valid, errors = handler.validate_password_strength("SecurePass123")

        assert is_valid is False
        assert any("special" in err for err in errors)

    def test_generate_temporary_password(self):
        """Test temporary password generation."""
        password = PasswordHandler.generate_temporary_password(16)

        assert len(password) == 16
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*" for c in password)

    def test_needs_rehash(self, handler):
        """Test checking if password needs rehashing."""
        password = "SecurePass123!"
        hashed = handler.hash_password(password)

        # Should not need rehash with same rounds
        assert handler.needs_rehash(hashed) is False

        # Create handler with higher rounds
        stronger_handler = PasswordHandler(rounds=14)
        assert stronger_handler.needs_rehash(hashed) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
