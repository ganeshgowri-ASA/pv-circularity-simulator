"""
Password hashing and validation using bcrypt.

This module provides secure password hashing, verification, and validation
capabilities using industry-standard bcrypt algorithm with configurable
work factor for protection against brute-force attacks.
"""

import re
import bcrypt
from typing import Optional


class PasswordHandler:
    """
    Handles password hashing, verification, and validation.

    Uses bcrypt for secure password hashing with configurable work factor.
    Implements password strength validation with customizable requirements.
    """

    # Default bcrypt work factor (cost factor)
    # Higher values = more secure but slower (4-31, default 12)
    DEFAULT_ROUNDS = 12

    # Password validation regex patterns
    MIN_LENGTH = 8
    MAX_LENGTH = 128

    def __init__(self, rounds: int = DEFAULT_ROUNDS):
        """
        Initialize password handler.

        Args:
            rounds: Bcrypt work factor (4-31). Higher is more secure but slower.
                   Recommended: 12-14 for production systems.
        """
        if not 4 <= rounds <= 31:
            raise ValueError("Bcrypt rounds must be between 4 and 31")
        self.rounds = rounds

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password to hash

        Returns:
            Bcrypt hashed password as a string

        Raises:
            ValueError: If password is empty or too long
        """
        if not password:
            raise ValueError("Password cannot be empty")

        if len(password) > self.MAX_LENGTH:
            raise ValueError(f"Password cannot exceed {self.MAX_LENGTH} characters")

        # Convert password to bytes and hash
        password_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)

        # Return as string for storage
        return hashed.decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain text password to verify
            password_hash: Bcrypt hash to verify against

        Returns:
            True if password matches hash, False otherwise

        Raises:
            ValueError: If password or hash is empty
        """
        if not password:
            raise ValueError("Password cannot be empty")

        if not password_hash:
            raise ValueError("Password hash cannot be empty")

        try:
            password_bytes = password.encode("utf-8")
            hash_bytes = password_hash.encode("utf-8")
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except Exception:
            # If any error occurs during verification, return False
            # This prevents timing attacks and handles corrupted hashes
            return False

    def validate_password_strength(
        self,
        password: str,
        min_length: int = MIN_LENGTH,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digits: bool = True,
        require_special: bool = True,
    ) -> tuple[bool, list[str]]:
        """
        Validate password strength against security requirements.

        Args:
            password: Password to validate
            min_length: Minimum password length (default: 8)
            require_uppercase: Require at least one uppercase letter
            require_lowercase: Require at least one lowercase letter
            require_digits: Require at least one digit
            require_special: Require at least one special character

        Returns:
            Tuple of (is_valid, list_of_errors)
            is_valid: True if password meets all requirements
            list_of_errors: List of error messages for failed requirements
        """
        errors = []

        if not password:
            errors.append("Password cannot be empty")
            return False, errors

        # Check length
        if len(password) < min_length:
            errors.append(f"Password must be at least {min_length} characters long")

        if len(password) > self.MAX_LENGTH:
            errors.append(f"Password cannot exceed {self.MAX_LENGTH} characters")

        # Check for uppercase letters
        if require_uppercase and not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")

        # Check for lowercase letters
        if require_lowercase and not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")

        # Check for digits
        if require_digits and not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")

        # Check for special characters
        if require_special and not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if a password hash needs to be rehashed with current settings.

        This is useful when you've increased the work factor and want to
        upgrade existing password hashes on next login.

        Args:
            password_hash: Existing password hash to check

        Returns:
            True if hash should be regenerated, False otherwise
        """
        try:
            hash_bytes = password_hash.encode("utf-8")
            # Extract the cost factor from the hash
            # Bcrypt hash format: $2b$[cost]$[salt][hash]
            cost_str = hash_bytes.split(b"$")[2]
            current_cost = int(cost_str)
            return current_cost < self.rounds
        except (IndexError, ValueError):
            # If we can't parse the hash, assume it needs rehashing
            return True

    @staticmethod
    def generate_temporary_password(length: int = 16) -> str:
        """
        Generate a secure temporary password.

        Args:
            length: Length of the password to generate (default: 16)

        Returns:
            Randomly generated password meeting strength requirements
        """
        import secrets
        import string

        if length < 8:
            raise ValueError("Temporary password must be at least 8 characters")

        # Ensure password contains all required character types
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"

        while True:
            password = "".join(secrets.choice(alphabet) for _ in range(length))

            # Verify it meets requirements
            if (
                re.search(r"[A-Z]", password)
                and re.search(r"[a-z]", password)
                and re.search(r"\d", password)
                and re.search(r"[!@#$%^&*]", password)
            ):
                return password


# Singleton instance for convenience
default_password_handler = PasswordHandler()


def hash_password(password: str) -> str:
    """
    Convenience function to hash a password using default settings.

    Args:
        password: Plain text password to hash

    Returns:
        Bcrypt hashed password
    """
    return default_password_handler.hash_password(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Convenience function to verify a password using default settings.

    Args:
        password: Plain text password to verify
        password_hash: Bcrypt hash to verify against

    Returns:
        True if password matches hash, False otherwise
    """
    return default_password_handler.verify_password(password, password_hash)


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Convenience function to validate password strength using default settings.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    return default_password_handler.validate_password_strength(password)
