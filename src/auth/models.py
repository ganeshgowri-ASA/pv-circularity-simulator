"""
Data models for authentication and authorization system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Set
from enum import Enum
import uuid


class UserStatus(Enum):
    """User account status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"
    SUSPENDED = "suspended"


@dataclass
class Permission:
    """
    Represents a specific permission in the system.

    Permissions are granular access rights that can be assigned to roles.
    Examples: 'read:data', 'write:data', 'delete:data', 'admin:system'
    """

    name: str
    description: str
    resource: str  # e.g., 'simulation', 'report', 'user'
    action: str  # e.g., 'read', 'write', 'delete', 'execute'

    def __str__(self) -> str:
        """Return string representation of permission."""
        return f"{self.action}:{self.resource}"

    def __hash__(self) -> int:
        """Make permission hashable for use in sets."""
        return hash((self.name, self.resource, self.action))

    def __eq__(self, other: object) -> bool:
        """Compare permissions for equality."""
        if not isinstance(other, Permission):
            return False
        return (
            self.name == other.name
            and self.resource == other.resource
            and self.action == other.action
        )


@dataclass
class Role:
    """
    Represents a role with associated permissions.

    Roles are collections of permissions that define what a user can do.
    Examples: 'admin', 'researcher', 'operator', 'viewer'
    """

    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False  # System roles cannot be deleted
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_permission(self, permission: Permission) -> None:
        """
        Add a permission to this role.

        Args:
            permission: Permission to add
        """
        self.permissions.add(permission)
        self.updated_at = datetime.utcnow()

    def remove_permission(self, permission: Permission) -> None:
        """
        Remove a permission from this role.

        Args:
            permission: Permission to remove
        """
        self.permissions.discard(permission)
        self.updated_at = datetime.utcnow()

    def has_permission(self, permission: Permission) -> bool:
        """
        Check if this role has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if role has the permission, False otherwise
        """
        return permission in self.permissions

    def __hash__(self) -> int:
        """Make role hashable for use in sets."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare roles for equality."""
        if not isinstance(other, Role):
            return False
        return self.name == other.name


@dataclass
class User:
    """
    Represents a user in the authentication system.

    This class stores user credentials, roles, and metadata for authentication
    and authorization purposes.
    """

    username: str
    email: str
    password_hash: str
    roles: Set[Role] = field(default_factory=set)
    status: UserStatus = UserStatus.ACTIVE
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    metadata: dict = field(default_factory=dict)

    def add_role(self, role: Role) -> None:
        """
        Add a role to this user.

        Args:
            role: Role to add
        """
        self.roles.add(role)
        self.updated_at = datetime.utcnow()

    def remove_role(self, role: Role) -> None:
        """
        Remove a role from this user.

        Args:
            role: Role to remove
        """
        self.roles.discard(role)
        self.updated_at = datetime.utcnow()

    def has_role(self, role_name: str) -> bool:
        """
        Check if user has a specific role.

        Args:
            role_name: Name of the role to check

        Returns:
            True if user has the role, False otherwise
        """
        return any(role.name == role_name for role in self.roles)

    def get_all_permissions(self) -> Set[Permission]:
        """
        Get all permissions from all roles assigned to this user.

        Returns:
            Set of all permissions the user has
        """
        all_permissions: Set[Permission] = set()
        for role in self.roles:
            all_permissions.update(role.permissions)
        return all_permissions

    def has_permission(self, permission: Permission) -> bool:
        """
        Check if user has a specific permission through any of their roles.

        Args:
            permission: Permission to check

        Returns:
            True if user has the permission, False otherwise
        """
        return permission in self.get_all_permissions()

    def is_active(self) -> bool:
        """
        Check if user account is active.

        Returns:
            True if user is active, False otherwise
        """
        return self.status == UserStatus.ACTIVE

    def lock_account(self) -> None:
        """Lock the user account (usually after failed login attempts)."""
        self.status = UserStatus.LOCKED
        self.updated_at = datetime.utcnow()

    def unlock_account(self) -> None:
        """Unlock the user account."""
        self.status = UserStatus.ACTIVE
        self.failed_login_attempts = 0
        self.updated_at = datetime.utcnow()

    def increment_failed_login(self) -> None:
        """Increment failed login attempt counter."""
        self.failed_login_attempts += 1
        self.updated_at = datetime.utcnow()

    def reset_failed_login(self) -> None:
        """Reset failed login attempt counter."""
        self.failed_login_attempts = 0
        self.updated_at = datetime.utcnow()

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert user object to dictionary.

        Args:
            include_sensitive: Whether to include sensitive data like password_hash

        Returns:
            Dictionary representation of the user
        """
        user_dict = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "status": self.status.value,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "roles": [role.name for role in self.roles],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata,
        }

        if include_sensitive:
            user_dict["password_hash"] = self.password_hash
            user_dict["failed_login_attempts"] = self.failed_login_attempts

        return user_dict


@dataclass
class Session:
    """
    Represents an active user session.

    Sessions track authenticated users and their activity in the system.
    """

    session_id: str
    user_id: str
    username: str
    token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """
        Check if session has expired.

        Returns:
            True if session has expired, False otherwise
        """
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """
        Check if session is valid (active and not expired).

        Returns:
            True if session is valid, False otherwise
        """
        return self.is_active and not self.is_expired()

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def invalidate(self) -> None:
        """Invalidate the session."""
        self.is_active = False

    def to_dict(self) -> dict:
        """
        Convert session object to dictionary.

        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
            "is_expired": self.is_expired(),
            "is_valid": self.is_valid(),
            "metadata": self.metadata,
        }
