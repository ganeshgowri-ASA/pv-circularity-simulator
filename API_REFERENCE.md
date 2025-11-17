# API Reference - Authentication & Access Control System

Complete API documentation for the Authentication & Access Control System.

## Table of Contents

- [AuthenticationManager](#authenticationmanager)
- [User Management](#user-management)
- [Authentication Methods](#authentication-methods)
- [Role-Based Access Control](#role-based-access-control)
- [Permission Validation](#permission-validation)
- [Session Management](#session-management)
- [Role Management](#role-management)
- [Models](#models)
- [Exceptions](#exceptions)

---

## AuthenticationManager

### Class: `AuthenticationManager`

Core authentication and authorization manager.

#### Constructor

```python
AuthenticationManager(
    secret_key: str,
    password_rounds: int = 12,
    session_duration: Optional[timedelta] = None,
)
```

**Parameters:**
- `secret_key` (str): Secret key for JWT token signing (minimum 32 characters)
- `password_rounds` (int, optional): Bcrypt work factor for password hashing (4-31). Default: 12
- `session_duration` (timedelta, optional): Default session duration. Default: 1 hour

**Raises:**
- `ValueError`: If secret_key is too short or password_rounds invalid

**Example:**
```python
from datetime import timedelta
auth_manager = AuthenticationManager(
    secret_key="your-secret-key-here-at-least-32-chars",
    password_rounds=12,
    session_duration=timedelta(hours=2),
)
```

---

## User Management

### `create_user()`

Create a new user account.

```python
create_user(
    username: str,
    email: str,
    password: str,
    roles: Optional[Set[Role]] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    validate_password: bool = True,
) -> User
```

**Parameters:**
- `username` (str): Unique username
- `email` (str): User email address
- `password` (str): Plain text password (will be hashed)
- `roles` (Set[Role], optional): Set of roles to assign to user
- `first_name` (str, optional): User's first name
- `last_name` (str, optional): User's last name
- `validate_password` (bool): Whether to validate password strength. Default: True

**Returns:** `User` object

**Raises:**
- `DuplicateUserError`: If username already exists
- `ValueError`: If password validation fails

**Example:**
```python
user = auth_manager.create_user(
    username="jsmith",
    email="jsmith@example.com",
    password="SecurePass123!",
    roles={researcher_role},
    first_name="John",
    last_name="Smith",
)
```

### `get_user()`

Get user by username.

```python
get_user(username: str) -> Optional[User]
```

**Parameters:**
- `username` (str): Username to look up

**Returns:** `User` object if found, `None` otherwise

### `get_user_by_id()`

Get user by user ID.

```python
get_user_by_id(user_id: str) -> Optional[User]
```

**Parameters:**
- `user_id` (str): User ID to look up

**Returns:** `User` object if found, `None` otherwise

### `delete_user()`

Delete a user account.

```python
delete_user(username: str) -> bool
```

**Parameters:**
- `username` (str): Username to delete

**Returns:** `True` if user was deleted, `False` if not found

---

## Authentication Methods

### `user_login()` ⭐

Authenticate user and create session.

```python
user_login(
    username: str,
    password: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]
```

**Parameters:**
- `username` (str): Username
- `password` (str): Plain text password
- `ip_address` (str, optional): Client IP address
- `user_agent` (str, optional): Client user agent string

**Returns:** Dictionary containing:
- `access_token` (str): JWT access token
- `refresh_token` (str): JWT refresh token
- `session_id` (str): Session identifier
- `user` (dict): User information
- `expires_at` (str): Token expiration time (ISO format)

**Raises:**
- `InvalidCredentialsError`: If credentials are invalid
- `AuthenticationError`: If account is locked or inactive

**Example:**
```python
result = auth_manager.user_login(
    username="jsmith",
    password="SecurePass123!",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
)

print(f"Access Token: {result['access_token']}")
print(f"Session ID: {result['session_id']}")
```

### `logout()`

Log out user and invalidate session.

```python
logout(session_id: str) -> bool
```

**Parameters:**
- `session_id` (str): Session ID to logout

**Returns:** `True` if logout successful, `False` if session not found

**Example:**
```python
success = auth_manager.logout(session_id)
```

### `refresh_token()`

Refresh access token using refresh token.

```python
refresh_token(refresh_token: str) -> Dict[str, str]
```

**Parameters:**
- `refresh_token` (str): Valid refresh token

**Returns:** Dictionary with new `access_token` and `refresh_token`

**Raises:**
- `InvalidTokenError`: If refresh token is invalid
- `SessionExpiredError`: If refresh token has expired

**Example:**
```python
new_tokens = auth_manager.refresh_token(old_refresh_token)
print(f"New Access Token: {new_tokens['access_token']}")
```

---

## Role-Based Access Control

### `role_based_access_control()` ⭐

Check if user has required role(s) for access control.

```python
role_based_access_control(
    user: User,
    required_role: Optional[str] = None,
    required_roles: Optional[Set[str]] = None,
    require_all: bool = False,
) -> bool
```

**Parameters:**
- `user` (User): User to check
- `required_role` (str, optional): Single required role name
- `required_roles` (Set[str], optional): Set of required role names
- `require_all` (bool): If True, user must have all roles; if False, any role. Default: False

**Returns:** `True` if user has required role(s)

**Raises:**
- `ValueError`: If neither required_role nor required_roles provided
- `AuthorizationError`: If user lacks required roles

**Examples:**

Single role check:
```python
# Check if user has admin role
auth_manager.role_based_access_control(
    user=user,
    required_role="admin",
)
```

Any of multiple roles:
```python
# User must have at least one of these roles
auth_manager.role_based_access_control(
    user=user,
    required_roles={"admin", "moderator", "supervisor"},
    require_all=False,
)
```

All roles required:
```python
# User must have all of these roles
auth_manager.role_based_access_control(
    user=user,
    required_roles={"researcher", "certified"},
    require_all=True,
)
```

### `check_user_role()`

Check if user has a specific role (non-throwing version).

```python
check_user_role(user: User, role_name: str) -> bool
```

**Parameters:**
- `user` (User): User to check
- `role_name` (str): Role name to check

**Returns:** `True` if user has role, `False` otherwise

---

## Permission Validation

### `permission_validator()` ⭐

Validate if user has required permission.

```python
permission_validator(
    user: User,
    permission: Optional[Permission] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
) -> bool
```

**Parameters:**
- `user` (User): User to validate
- `permission` (Permission, optional): Permission object to check
- `resource` (str, optional): Resource name (alternative to permission object)
- `action` (str, optional): Action name (alternative to permission object)

**Returns:** `True` if user has permission

**Raises:**
- `ValueError`: If neither permission nor (resource, action) provided
- `AuthorizationError`: If user lacks required permission

**Examples:**

Using Permission object:
```python
from src.auth.rbac import SystemPermissions

auth_manager.permission_validator(
    user=user,
    permission=SystemPermissions.SIMULATION_EXECUTE,
)
```

Using resource and action:
```python
auth_manager.permission_validator(
    user=user,
    resource="simulation",
    action="execute",
)
```

### `check_user_permission()`

Check if user has permission (non-throwing version).

```python
check_user_permission(
    user: User,
    permission: Optional[Permission] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
) -> bool
```

**Parameters:**
- `user` (User): User to check
- `permission` (Permission, optional): Permission object to check
- `resource` (str, optional): Resource name
- `action` (str, optional): Action name

**Returns:** `True` if user has permission, `False` otherwise

---

## Session Management

### `session_management()` ⭐

Manage user sessions with various operations.

```python
session_management(
    action: str,
    session_id: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Any
```

**Parameters:**
- `action` (str): Action to perform (see below)
- `session_id` (str, optional): Session ID
- `token` (str, optional): Access token
- `user_id` (str, optional): User ID

**Actions:**
- `"validate"`: Validate a session (requires `session_id` or `token`)
- `"get"`: Get session details (requires `session_id`)
- `"list"`: List all sessions for a user (requires `user_id`)
- `"cleanup"`: Remove expired sessions
- `"invalidate"`: Invalidate a session (requires `session_id`)

**Returns:** Depends on action:
- validate: `Session` object if valid
- get: `Session` object
- list: List of session dictionaries
- cleanup: Number of sessions cleaned up (int)
- invalidate: `True` if successful

**Raises:**
- `ValueError`: If required parameters missing
- `SessionExpiredError`: If session is expired
- `InvalidTokenError`: If token is invalid

**Examples:**

Validate session:
```python
session = auth_manager.session_management(
    action="validate",
    session_id="session-id-here",
)
print(f"Session valid for user: {session.username}")
```

Get session details:
```python
session = auth_manager.session_management(
    action="get",
    session_id="session-id-here",
)
print(f"Created: {session.created_at}")
print(f"IP: {session.ip_address}")
```

List user sessions:
```python
sessions = auth_manager.session_management(
    action="list",
    user_id="user-id-here",
)
print(f"User has {len(sessions)} active sessions")
```

Cleanup expired sessions:
```python
count = auth_manager.session_management(action="cleanup")
print(f"Cleaned up {count} expired sessions")
```

Invalidate session:
```python
success = auth_manager.session_management(
    action="invalidate",
    session_id="session-id-here",
)
```

---

## Role Management

### `create_role()`

Create a new role.

```python
create_role(
    name: str,
    description: str,
    permissions: Optional[Set[Permission]] = None,
    is_system_role: bool = False,
) -> Role
```

**Parameters:**
- `name` (str): Unique role name
- `description` (str): Role description
- `permissions` (Set[Permission], optional): Set of permissions for this role
- `is_system_role` (bool): Whether this is a system role (cannot be deleted). Default: False

**Returns:** `Role` object

**Raises:**
- `ValueError`: If role already exists

**Example:**
```python
from src.auth.rbac import SystemPermissions

admin_role = auth_manager.create_role(
    name="admin",
    description="System Administrator",
    permissions={
        SystemPermissions.SYSTEM_ADMIN,
        SystemPermissions.USER_READ,
        SystemPermissions.USER_WRITE,
        SystemPermissions.USER_DELETE,
    },
    is_system_role=True,
)
```

### `get_role()`

Get role by name.

```python
get_role(name: str) -> Optional[Role]
```

**Parameters:**
- `name` (str): Role name

**Returns:** `Role` object if found, `None` otherwise

### `assign_role_to_user()`

Assign a role to a user.

```python
assign_role_to_user(user: User, role: Role) -> None
```

**Parameters:**
- `user` (User): User to assign role to
- `role` (Role): Role to assign

**Example:**
```python
auth_manager.assign_role_to_user(user, admin_role)
```

### `remove_role_from_user()`

Remove a role from a user.

```python
remove_role_from_user(user: User, role: Role) -> None
```

**Parameters:**
- `user` (User): User to remove role from
- `role` (Role): Role to remove

---

## Models

### User

Represents a user in the authentication system.

**Attributes:**
- `user_id` (str): Unique user identifier
- `username` (str): Username
- `email` (str): Email address
- `password_hash` (str): Hashed password
- `roles` (Set[Role]): User roles
- `status` (UserStatus): Account status (ACTIVE, INACTIVE, LOCKED, SUSPENDED)
- `first_name` (str, optional): First name
- `last_name` (str, optional): Last name
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp
- `last_login` (datetime, optional): Last login timestamp
- `failed_login_attempts` (int): Failed login counter
- `metadata` (dict): Additional metadata

**Methods:**
- `add_role(role: Role)`: Add a role
- `remove_role(role: Role)`: Remove a role
- `has_role(role_name: str) -> bool`: Check if user has role
- `get_all_permissions() -> Set[Permission]`: Get all permissions
- `has_permission(permission: Permission) -> bool`: Check permission
- `is_active() -> bool`: Check if account is active
- `to_dict(include_sensitive: bool = False) -> dict`: Convert to dictionary

### Role

Represents a role with associated permissions.

**Attributes:**
- `name` (str): Role name
- `description` (str): Role description
- `permissions` (Set[Permission]): Role permissions
- `is_system_role` (bool): Whether this is a system role
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp

**Methods:**
- `add_permission(permission: Permission)`: Add permission
- `remove_permission(permission: Permission)`: Remove permission
- `has_permission(permission: Permission) -> bool`: Check permission

### Permission

Represents a specific permission.

**Attributes:**
- `name` (str): Permission name
- `description` (str): Permission description
- `resource` (str): Resource name (e.g., 'simulation', 'report')
- `action` (str): Action name (e.g., 'read', 'write', 'delete')

**String Format:** `{action}:{resource}` (e.g., "read:simulation")

### Session

Represents an active user session.

**Attributes:**
- `session_id` (str): Session identifier
- `user_id` (str): User ID
- `username` (str): Username
- `token` (str): Access token
- `created_at` (datetime): Creation timestamp
- `expires_at` (datetime): Expiration timestamp
- `last_activity` (datetime): Last activity timestamp
- `ip_address` (str, optional): Client IP address
- `user_agent` (str, optional): Client user agent
- `is_active` (bool): Whether session is active
- `metadata` (dict): Additional metadata

**Methods:**
- `is_expired() -> bool`: Check if expired
- `is_valid() -> bool`: Check if valid (active and not expired)
- `update_activity()`: Update last activity timestamp
- `invalidate()`: Invalidate the session
- `to_dict() -> dict`: Convert to dictionary

---

## Exceptions

### AuthenticationError

Base exception for authentication-related errors.

**Attributes:**
- `message` (str): Error message
- `details` (str, optional): Additional details

### AuthorizationError

Raised when user lacks required permissions.

**Attributes:**
- `message` (str): Error message
- `required_permission` (str, optional): Required permission
- `user_permissions` (list[str]): User's permissions

### InvalidCredentialsError

Raised when login credentials are invalid.

### SessionExpiredError

Raised when user session has expired.

### InvalidTokenError

Raised when JWT token is invalid or malformed.

### UserNotFoundError

Raised when user cannot be found in the system.

**Attributes:**
- `username` (str): Username that was not found

### DuplicateUserError

Raised when attempting to create a user that already exists.

**Attributes:**
- `username` (str): Username that already exists

---

## System Permissions

Pre-defined permissions available in `src.auth.rbac.SystemPermissions`:

**User Management:**
- `USER_READ`: View user information
- `USER_WRITE`: Create and update users
- `USER_DELETE`: Delete users

**Simulation Management:**
- `SIMULATION_READ`: View simulations
- `SIMULATION_WRITE`: Create and update simulations
- `SIMULATION_DELETE`: Delete simulations
- `SIMULATION_EXECUTE`: Execute simulations

**Report Management:**
- `REPORT_READ`: View reports
- `REPORT_WRITE`: Create and update reports
- `REPORT_DELETE`: Delete reports

**System Administration:**
- `SYSTEM_ADMIN`: Full system administration
- `SYSTEM_CONFIG`: Configure system settings

---

## Complete Example

```python
from datetime import timedelta
from src.auth import AuthenticationManager, Role, Permission
from src.auth.rbac import SystemPermissions
from config import AuthConfig

# Initialize
auth_manager = AuthenticationManager(
    secret_key=AuthConfig.SECRET_KEY,
    password_rounds=12,
    session_duration=timedelta(hours=2),
)

# Create role
researcher_role = auth_manager.create_role(
    name="researcher",
    description="Research Scientist",
    permissions={
        SystemPermissions.SIMULATION_READ,
        SystemPermissions.SIMULATION_WRITE,
        SystemPermissions.SIMULATION_EXECUTE,
        SystemPermissions.REPORT_READ,
        SystemPermissions.REPORT_WRITE,
    },
)

# Create user
user = auth_manager.create_user(
    username="jsmith",
    email="jsmith@example.com",
    password="SecurePass123!",
    roles={researcher_role},
    first_name="John",
    last_name="Smith",
)

# Login
result = auth_manager.user_login(
    username="jsmith",
    password="SecurePass123!",
    ip_address="192.168.1.100",
)

# Check role
auth_manager.role_based_access_control(user, required_role="researcher")

# Check permission
auth_manager.permission_validator(
    user=user,
    resource="simulation",
    action="execute",
)

# Validate session
session = auth_manager.session_management(
    action="validate",
    session_id=result["session_id"],
)

# Logout
auth_manager.logout(result["session_id"])
```

---

For more examples, see `examples/basic_usage.py`.
