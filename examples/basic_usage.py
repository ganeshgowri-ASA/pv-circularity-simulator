"""
Basic usage examples for Authentication & Access Control System.

This demonstrates the core functionality of the authentication system.
"""

from src.auth import (
    AuthenticationManager,
    Role,
    Permission,
    AuthenticationError,
    AuthorizationError,
)
from src.auth.rbac import SystemPermissions
from config import AuthConfig


def main():
    """Demonstrate basic authentication system usage."""
    print("=" * 80)
    print("Authentication & Access Control System - Basic Usage")
    print("=" * 80)

    # Initialize Authentication Manager
    print("\n1. Initializing Authentication Manager...")
    auth_manager = AuthenticationManager(
        secret_key=AuthConfig.SECRET_KEY,
        password_rounds=AuthConfig.BCRYPT_ROUNDS,
        session_duration=AuthConfig.get_session_duration(),
    )
    print("   ✓ Authentication Manager initialized")

    # Create roles
    print("\n2. Creating roles...")
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
    print("   ✓ Admin role created")

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
    print("   ✓ Researcher role created")

    viewer_role = auth_manager.create_role(
        name="viewer",
        description="Read-only Viewer",
        permissions={
            SystemPermissions.SIMULATION_READ,
            SystemPermissions.REPORT_READ,
        },
    )
    print("   ✓ Viewer role created")

    # Create users
    print("\n3. Creating users...")
    admin_user = auth_manager.create_user(
        username="admin",
        email="admin@pv-simulator.com",
        password="AdminPass123!",
        roles={admin_role},
        first_name="System",
        last_name="Administrator",
    )
    print(f"   ✓ Admin user created: {admin_user.username}")

    researcher_user = auth_manager.create_user(
        username="researcher1",
        email="researcher@pv-simulator.com",
        password="ResearchPass123!",
        roles={researcher_role},
        first_name="Jane",
        last_name="Smith",
    )
    print(f"   ✓ Researcher user created: {researcher_user.username}")

    viewer_user = auth_manager.create_user(
        username="viewer1",
        email="viewer@pv-simulator.com",
        password="ViewerPass123!",
        roles={viewer_role},
        first_name="John",
        last_name="Doe",
    )
    print(f"   ✓ Viewer user created: {viewer_user.username}")

    # User login
    print("\n4. User Login - user_login()")
    print("   Attempting login for researcher...")
    try:
        login_result = auth_manager.user_login(
            username="researcher1",
            password="ResearchPass123!",
            ip_address="192.168.1.100",
            user_agent="Python/Example",
        )
        print(f"   ✓ Login successful!")
        print(f"     - Session ID: {login_result['session_id']}")
        print(f"     - User: {login_result['user']['username']}")
        print(f"     - Roles: {', '.join(login_result['user']['roles'])}")
        print(f"     - Token expires: {login_result['expires_at']}")
    except AuthenticationError as e:
        print(f"   ✗ Login failed: {e}")

    # Role-based access control
    print("\n5. Role-Based Access Control - role_based_access_control()")

    # Check if researcher has researcher role
    print("   Checking if researcher has 'researcher' role...")
    try:
        result = auth_manager.role_based_access_control(
            user=researcher_user,
            required_role="researcher",
        )
        print(f"   ✓ Access granted: {result}")
    except AuthorizationError as e:
        print(f"   ✗ Access denied: {e}")

    # Check if researcher has admin role (should fail)
    print("   Checking if researcher has 'admin' role...")
    try:
        result = auth_manager.role_based_access_control(
            user=researcher_user,
            required_role="admin",
        )
        print(f"   ✓ Access granted: {result}")
    except AuthorizationError as e:
        print(f"   ✗ Access denied: {e.message}")

    # Permission validation
    print("\n6. Permission Validation - permission_validator()")

    # Check if researcher can execute simulations
    print("   Checking if researcher can execute simulations...")
    try:
        result = auth_manager.permission_validator(
            user=researcher_user,
            permission=SystemPermissions.SIMULATION_EXECUTE,
        )
        print(f"   ✓ Permission granted: {result}")
    except AuthorizationError as e:
        print(f"   ✗ Permission denied: {e}")

    # Check if researcher can delete users (should fail)
    print("   Checking if researcher can delete users...")
    try:
        result = auth_manager.permission_validator(
            user=researcher_user,
            permission=SystemPermissions.USER_DELETE,
        )
        print(f"   ✓ Permission granted: {result}")
    except AuthorizationError as e:
        print(f"   ✗ Permission denied: {e.message}")

    # Alternative permission check using resource and action
    print("   Checking if researcher can write reports...")
    try:
        result = auth_manager.permission_validator(
            user=researcher_user,
            resource="report",
            action="write",
        )
        print(f"   ✓ Permission granted: {result}")
    except AuthorizationError as e:
        print(f"   ✗ Permission denied: {e}")

    # Session management
    print("\n7. Session Management - session_management()")

    # Validate session
    print(f"   Validating session {login_result['session_id'][:8]}...")
    try:
        session = auth_manager.session_management(
            action="validate",
            session_id=login_result["session_id"],
        )
        print(f"   ✓ Session valid: {session.username}")
    except Exception as e:
        print(f"   ✗ Session invalid: {e}")

    # List user sessions
    print(f"   Listing sessions for user {researcher_user.user_id[:8]}...")
    sessions = auth_manager.session_management(
        action="list",
        user_id=researcher_user.user_id,
    )
    print(f"   ✓ Found {len(sessions)} active session(s)")

    # Get session details
    print(f"   Getting session details...")
    session_detail = auth_manager.session_management(
        action="get",
        session_id=login_result["session_id"],
    )
    if session_detail:
        print(f"   ✓ Session details retrieved")
        print(f"     - Username: {session_detail.username}")
        print(f"     - IP Address: {session_detail.ip_address}")
        print(f"     - Created: {session_detail.created_at}")

    # Token refresh
    print("\n8. Token Refresh")
    print("   Refreshing access token...")
    try:
        new_tokens = auth_manager.refresh_token(login_result["refresh_token"])
        print(f"   ✓ Token refreshed successfully")
    except Exception as e:
        print(f"   ✗ Token refresh failed: {e}")

    # Logout
    print("\n9. Logout")
    print(f"   Logging out session {login_result['session_id'][:8]}...")
    result = auth_manager.logout(login_result["session_id"])
    if result:
        print(f"   ✓ Logout successful")
    else:
        print(f"   ✗ Logout failed")

    # Verify session is invalidated
    print("   Verifying session is invalidated...")
    try:
        auth_manager.session_management(
            action="validate",
            session_id=login_result["session_id"],
        )
        print("   ✗ Session still valid (unexpected)")
    except Exception:
        print("   ✓ Session invalidated successfully")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
