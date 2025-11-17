"""
Role-Based Access Control (RBAC) implementation.

This module provides comprehensive role and permission management with
hierarchical roles, permission inheritance, and fine-grained access control.
"""

from typing import Optional, Set, Dict, Callable, Any
from functools import wraps
import logging

from .models import User, Role, Permission
from .exceptions import AuthorizationError

logger = logging.getLogger(__name__)


class RBACManager:
    """
    Manages role-based access control with permission validation.

    Provides methods to check permissions, validate access, and manage
    role-permission relationships.
    """

    def __init__(self):
        """Initialize RBAC manager."""
        self._role_hierarchy: Dict[str, Set[str]] = {}  # parent -> children roles
        self._permission_cache: Dict[str, Set[Permission]] = {}  # role -> permissions

    def set_role_hierarchy(self, parent_role: str, child_roles: Set[str]) -> None:
        """
        Define role hierarchy for permission inheritance.

        Child roles inherit all permissions from parent roles.

        Args:
            parent_role: Parent role name
            child_roles: Set of child role names
        """
        self._role_hierarchy[parent_role] = child_roles
        self._clear_permission_cache()

    def get_inherited_roles(self, role_name: str) -> Set[str]:
        """
        Get all roles inherited by a given role (including itself).

        Args:
            role_name: Role name to get inheritance for

        Returns:
            Set of all role names including parent roles
        """
        inherited = {role_name}

        # Check if this role is a child of any parent
        for parent, children in self._role_hierarchy.items():
            if role_name in children:
                inherited.add(parent)
                # Recursively get parent's inherited roles
                inherited.update(self.get_inherited_roles(parent))

        return inherited

    def get_effective_permissions(self, user: User) -> Set[Permission]:
        """
        Get all effective permissions for a user including inherited permissions.

        Args:
            user: User object

        Returns:
            Set of all permissions (including inherited ones)
        """
        all_permissions: Set[Permission] = set()

        for role in user.roles:
            # Get permissions from this role
            all_permissions.update(role.permissions)

            # Get permissions from inherited roles
            inherited_role_names = self.get_inherited_roles(role.name)
            for inherited_role_name in inherited_role_names:
                # Find the role object and add its permissions
                for user_role in user.roles:
                    if user_role.name == inherited_role_name:
                        all_permissions.update(user_role.permissions)

        return all_permissions

    def check_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has a specific permission.

        Args:
            user: User to check permission for
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        if not user.is_active():
            logger.warning(f"Permission check failed: User {user.username} is not active")
            return False

        effective_permissions = self.get_effective_permissions(user)
        return permission in effective_permissions

    def check_permission_by_action(
        self, user: User, resource: str, action: str
    ) -> bool:
        """
        Check if user can perform an action on a resource.

        Args:
            user: User to check permission for
            resource: Resource name (e.g., 'simulation', 'report')
            action: Action name (e.g., 'read', 'write', 'delete')

        Returns:
            True if user has permission, False otherwise
        """
        if not user.is_active():
            return False

        effective_permissions = self.get_effective_permissions(user)

        # Check for exact match
        for perm in effective_permissions:
            if perm.resource == resource and perm.action == action:
                return True

        # Check for wildcard permissions
        for perm in effective_permissions:
            # Wildcard resource: action:*
            if perm.resource == "*" and perm.action == action:
                return True
            # Wildcard action: *:resource
            if perm.resource == resource and perm.action == "*":
                return True
            # Super admin: *:*
            if perm.resource == "*" and perm.action == "*":
                return True

        return False

    def require_permission(
        self, user: User, permission: Permission, raise_exception: bool = True
    ) -> bool:
        """
        Require user to have a specific permission.

        Args:
            user: User to check permission for
            permission: Permission required
            raise_exception: Whether to raise exception if permission denied

        Returns:
            True if user has permission

        Raises:
            AuthorizationError: If user lacks permission and raise_exception=True
        """
        has_permission = self.check_permission(user, permission)

        if not has_permission and raise_exception:
            user_permissions = [str(p) for p in self.get_effective_permissions(user)]
            raise AuthorizationError(
                message=f"User '{user.username}' lacks required permission",
                required_permission=str(permission),
                user_permissions=user_permissions,
            )

        return has_permission

    def require_role(self, user: User, role_name: str, raise_exception: bool = True) -> bool:
        """
        Require user to have a specific role.

        Args:
            user: User to check role for
            role_name: Role name required
            raise_exception: Whether to raise exception if role not found

        Returns:
            True if user has role

        Raises:
            AuthorizationError: If user lacks role and raise_exception=True
        """
        has_role = user.has_role(role_name)

        if not has_role and raise_exception:
            user_roles = [role.name for role in user.roles]
            raise AuthorizationError(
                message=f"User '{user.username}' lacks required role '{role_name}'",
                required_permission=f"role:{role_name}",
                user_permissions=[f"role:{r}" for r in user_roles],
            )

        return has_role

    def require_any_role(
        self, user: User, role_names: Set[str], raise_exception: bool = True
    ) -> bool:
        """
        Require user to have at least one of the specified roles.

        Args:
            user: User to check roles for
            role_names: Set of acceptable role names
            raise_exception: Whether to raise exception if no role found

        Returns:
            True if user has at least one of the roles

        Raises:
            AuthorizationError: If user lacks all roles and raise_exception=True
        """
        for role_name in role_names:
            if user.has_role(role_name):
                return True

        if raise_exception:
            user_roles = [role.name for role in user.roles]
            raise AuthorizationError(
                message=f"User '{user.username}' lacks any of required roles: {role_names}",
                required_permission=f"any_role:{','.join(role_names)}",
                user_permissions=[f"role:{r}" for r in user_roles],
            )

        return False

    def require_all_roles(
        self, user: User, role_names: Set[str], raise_exception: bool = True
    ) -> bool:
        """
        Require user to have all of the specified roles.

        Args:
            user: User to check roles for
            role_names: Set of required role names
            raise_exception: Whether to raise exception if any role missing

        Returns:
            True if user has all the roles

        Raises:
            AuthorizationError: If user lacks any role and raise_exception=True
        """
        for role_name in role_names:
            if not user.has_role(role_name):
                if raise_exception:
                    user_roles = [role.name for role in user.roles]
                    raise AuthorizationError(
                        message=f"User '{user.username}' lacks required role '{role_name}'",
                        required_permission=f"all_roles:{','.join(role_names)}",
                        user_permissions=[f"role:{r}" for r in user_roles],
                    )
                return False

        return True

    def _clear_permission_cache(self) -> None:
        """Clear the permission cache (used when hierarchy changes)."""
        self._permission_cache.clear()


# Decorator for permission checking
def require_permission(permission: Permission):
    """
    Decorator to require permission for a function.

    Args:
        permission: Permission required to execute function

    Usage:
        @require_permission(Permission("manage_users", "User management", "user", "write"))
        def update_user(user: User, target_user: User):
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract user from arguments
            user = None
            if args and isinstance(args[0], User):
                user = args[0]
            elif "user" in kwargs and isinstance(kwargs["user"], User):
                user = kwargs["user"]

            if user is None:
                raise AuthorizationError("User context required for permission check")

            # Check permission
            rbac = RBACManager()
            rbac.require_permission(user, permission, raise_exception=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role_name: str):
    """
    Decorator to require role for a function.

    Args:
        role_name: Role name required to execute function

    Usage:
        @require_role("admin")
        def admin_function(user: User):
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract user from arguments
            user = None
            if args and isinstance(args[0], User):
                user = args[0]
            elif "user" in kwargs and isinstance(kwargs["user"], User):
                user = kwargs["user"]

            if user is None:
                raise AuthorizationError("User context required for role check")

            # Check role
            rbac = RBACManager()
            rbac.require_role(user, role_name, raise_exception=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_role(role_names: Set[str]):
    """
    Decorator to require at least one of specified roles for a function.

    Args:
        role_names: Set of acceptable role names

    Usage:
        @require_any_role({"admin", "moderator"})
        def moderate_content(user: User):
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            user = None
            if args and isinstance(args[0], User):
                user = args[0]
            elif "user" in kwargs and isinstance(kwargs["user"], User):
                user = kwargs["user"]

            if user is None:
                raise AuthorizationError("User context required for role check")

            rbac = RBACManager()
            rbac.require_any_role(user, role_names, raise_exception=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Predefined system permissions
class SystemPermissions:
    """Predefined system permissions for common operations."""

    # User management
    USER_READ = Permission("user:read", "View user information", "user", "read")
    USER_WRITE = Permission("user:write", "Create and update users", "user", "write")
    USER_DELETE = Permission("user:delete", "Delete users", "user", "delete")

    # Simulation management
    SIMULATION_READ = Permission("simulation:read", "View simulations", "simulation", "read")
    SIMULATION_WRITE = Permission(
        "simulation:write", "Create and update simulations", "simulation", "write"
    )
    SIMULATION_DELETE = Permission("simulation:delete", "Delete simulations", "simulation", "delete")
    SIMULATION_EXECUTE = Permission(
        "simulation:execute", "Execute simulations", "simulation", "execute"
    )

    # Report management
    REPORT_READ = Permission("report:read", "View reports", "report", "read")
    REPORT_WRITE = Permission("report:write", "Create and update reports", "report", "write")
    REPORT_DELETE = Permission("report:delete", "Delete reports", "report", "delete")

    # System administration
    SYSTEM_ADMIN = Permission("system:admin", "Full system administration", "*", "*")
    SYSTEM_CONFIG = Permission("system:config", "Configure system settings", "system", "write")
