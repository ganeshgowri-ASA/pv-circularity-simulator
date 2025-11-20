"""
Comprehensive tests for AuthenticationManager.
"""

import pytest
from datetime import datetime, timedelta

from src.auth import (
    AuthenticationManager,
    User,
    Role,
    Permission,
    AuthenticationError,
    AuthorizationError,
    InvalidCredentialsError,
    SessionExpiredError,
    DuplicateUserError,
)
from src.auth.models import UserStatus
from src.auth.rbac import SystemPermissions


@pytest.fixture
def auth_manager():
    """Create AuthenticationManager instance for testing."""
    secret_key = "test-secret-key-" + "x" * 32
    return AuthenticationManager(secret_key=secret_key, password_rounds=4)


@pytest.fixture
def test_roles():
    """Create test roles with permissions."""
    admin_role = Role(
        name="admin",
        description="Administrator",
        permissions={
            SystemPermissions.SYSTEM_ADMIN,
            SystemPermissions.USER_READ,
            SystemPermissions.USER_WRITE,
            SystemPermissions.USER_DELETE,
        },
        is_system_role=True,
    )

    researcher_role = Role(
        name="researcher",
        description="Researcher",
        permissions={
            SystemPermissions.SIMULATION_READ,
            SystemPermissions.SIMULATION_WRITE,
            SystemPermissions.SIMULATION_EXECUTE,
            SystemPermissions.REPORT_READ,
        },
    )

    viewer_role = Role(
        name="viewer",
        description="Read-only viewer",
        permissions={
            SystemPermissions.SIMULATION_READ,
            SystemPermissions.REPORT_READ,
        },
    )

    return {
        "admin": admin_role,
        "researcher": researcher_role,
        "viewer": viewer_role,
    }


class TestUserManagement:
    """Tests for user management operations."""

    def test_create_user(self, auth_manager, test_roles):
        """Test user creation."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
            first_name="Test",
            last_name="User",
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.has_role("researcher")
        assert user.status == UserStatus.ACTIVE

    def test_create_duplicate_user(self, auth_manager):
        """Test that duplicate username raises error."""
        auth_manager.create_user(
            username="duplicate",
            email="test@example.com",
            password="SecurePass123!",
        )

        with pytest.raises(DuplicateUserError):
            auth_manager.create_user(
                username="duplicate",
                email="other@example.com",
                password="SecurePass123!",
            )

    def test_create_user_weak_password(self, auth_manager):
        """Test that weak password raises error."""
        with pytest.raises(ValueError, match="Password validation failed"):
            auth_manager.create_user(
                username="testuser",
                email="test@example.com",
                password="weak",
                validate_password=True,
            )

    def test_get_user(self, auth_manager):
        """Test retrieving user by username."""
        created_user = auth_manager.create_user(
            username="findme",
            email="find@example.com",
            password="SecurePass123!",
        )

        found_user = auth_manager.get_user("findme")
        assert found_user is not None
        assert found_user.username == "findme"
        assert found_user.user_id == created_user.user_id

    def test_get_user_by_id(self, auth_manager):
        """Test retrieving user by ID."""
        created_user = auth_manager.create_user(
            username="findme",
            email="find@example.com",
            password="SecurePass123!",
        )

        found_user = auth_manager.get_user_by_id(created_user.user_id)
        assert found_user is not None
        assert found_user.username == "findme"

    def test_delete_user(self, auth_manager):
        """Test user deletion."""
        auth_manager.create_user(
            username="deleteme",
            email="delete@example.com",
            password="SecurePass123!",
        )

        result = auth_manager.delete_user("deleteme")
        assert result is True

        user = auth_manager.get_user("deleteme")
        assert user is None


class TestAuthentication:
    """Tests for authentication operations."""

    def test_successful_login(self, auth_manager, test_roles):
        """Test successful user login."""
        auth_manager.create_user(
            username="loginuser",
            email="login@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        result = auth_manager.user_login(
            username="loginuser",
            password="SecurePass123!",
            ip_address="127.0.0.1",
        )

        assert "access_token" in result
        assert "refresh_token" in result
        assert "session_id" in result
        assert "user" in result
        assert result["user"]["username"] == "loginuser"

    def test_login_invalid_username(self, auth_manager):
        """Test login with invalid username."""
        with pytest.raises(InvalidCredentialsError):
            auth_manager.user_login(
                username="nonexistent",
                password="SecurePass123!",
            )

    def test_login_invalid_password(self, auth_manager):
        """Test login with invalid password."""
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )

        with pytest.raises(InvalidCredentialsError):
            auth_manager.user_login(
                username="testuser",
                password="WrongPassword!",
            )

    def test_account_lockout(self, auth_manager):
        """Test account lockout after failed login attempts."""
        auth_manager.create_user(
            username="locktest",
            email="lock@example.com",
            password="SecurePass123!",
        )

        # Attempt failed logins
        for _ in range(auth_manager.MAX_FAILED_LOGIN_ATTEMPTS):
            try:
                auth_manager.user_login(username="locktest", password="wrong")
            except InvalidCredentialsError:
                pass

        # Account should be locked now
        with pytest.raises(AuthenticationError, match="locked"):
            auth_manager.user_login(username="locktest", password="SecurePass123!")

    def test_logout(self, auth_manager):
        """Test user logout."""
        auth_manager.create_user(
            username="logoutuser",
            email="logout@example.com",
            password="SecurePass123!",
        )

        login_result = auth_manager.user_login(
            username="logoutuser",
            password="SecurePass123!",
        )

        session_id = login_result["session_id"]
        result = auth_manager.logout(session_id)
        assert result is True

    def test_refresh_token(self, auth_manager):
        """Test token refresh."""
        auth_manager.create_user(
            username="refreshuser",
            email="refresh@example.com",
            password="SecurePass123!",
        )

        login_result = auth_manager.user_login(
            username="refreshuser",
            password="SecurePass123!",
        )

        refresh_token = login_result["refresh_token"]
        new_tokens = auth_manager.refresh_token(refresh_token)

        assert "access_token" in new_tokens
        assert "refresh_token" in new_tokens
        assert new_tokens["access_token"] != login_result["access_token"]


class TestRoleBasedAccessControl:
    """Tests for role-based access control."""

    def test_role_based_access_control_single_role(self, auth_manager, test_roles):
        """Test RBAC with single required role."""
        user = auth_manager.create_user(
            username="researcher",
            email="researcher@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        # Should pass
        result = auth_manager.role_based_access_control(
            user=user,
            required_role="researcher",
        )
        assert result is True

        # Should fail
        with pytest.raises(AuthorizationError):
            auth_manager.role_based_access_control(
                user=user,
                required_role="admin",
            )

    def test_role_based_access_control_any_role(self, auth_manager, test_roles):
        """Test RBAC with any of multiple roles."""
        user = auth_manager.create_user(
            username="viewer",
            email="viewer@example.com",
            password="SecurePass123!",
            roles={test_roles["viewer"]},
        )

        # Should pass (user has viewer role)
        result = auth_manager.role_based_access_control(
            user=user,
            required_roles={"admin", "researcher", "viewer"},
            require_all=False,
        )
        assert result is True

    def test_role_based_access_control_all_roles(self, auth_manager, test_roles):
        """Test RBAC requiring all roles."""
        user = auth_manager.create_user(
            username="multiuser",
            email="multi@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"], test_roles["viewer"]},
        )

        # Should pass
        result = auth_manager.role_based_access_control(
            user=user,
            required_roles={"researcher", "viewer"},
            require_all=True,
        )
        assert result is True

        # Should fail (user doesn't have admin)
        with pytest.raises(AuthorizationError):
            auth_manager.role_based_access_control(
                user=user,
                required_roles={"researcher", "viewer", "admin"},
                require_all=True,
            )


class TestPermissionValidation:
    """Tests for permission validation."""

    def test_permission_validator_with_permission_object(self, auth_manager, test_roles):
        """Test permission validation with Permission object."""
        user = auth_manager.create_user(
            username="researcher",
            email="researcher@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        # Should pass
        result = auth_manager.permission_validator(
            user=user,
            permission=SystemPermissions.SIMULATION_READ,
        )
        assert result is True

        # Should fail
        with pytest.raises(AuthorizationError):
            auth_manager.permission_validator(
                user=user,
                permission=SystemPermissions.USER_DELETE,
            )

    def test_permission_validator_with_resource_action(self, auth_manager, test_roles):
        """Test permission validation with resource and action."""
        user = auth_manager.create_user(
            username="researcher",
            email="researcher@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        # Should pass
        result = auth_manager.permission_validator(
            user=user,
            resource="simulation",
            action="read",
        )
        assert result is True

        # Should fail
        with pytest.raises(AuthorizationError):
            auth_manager.permission_validator(
                user=user,
                resource="user",
                action="delete",
            )

    def test_check_user_permission_non_throwing(self, auth_manager, test_roles):
        """Test non-throwing permission check."""
        user = auth_manager.create_user(
            username="researcher",
            email="researcher@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        assert auth_manager.check_user_permission(
            user=user,
            permission=SystemPermissions.SIMULATION_READ,
        ) is True

        assert auth_manager.check_user_permission(
            user=user,
            permission=SystemPermissions.USER_DELETE,
        ) is False


class TestSessionManagement:
    """Tests for session management."""

    def test_session_validate(self, auth_manager):
        """Test session validation."""
        auth_manager.create_user(
            username="sessionuser",
            email="session@example.com",
            password="SecurePass123!",
        )

        login_result = auth_manager.user_login(
            username="sessionuser",
            password="SecurePass123!",
        )

        session = auth_manager.session_management(
            action="validate",
            session_id=login_result["session_id"],
        )

        assert session is not None
        assert session.username == "sessionuser"

    def test_session_get(self, auth_manager):
        """Test getting session details."""
        auth_manager.create_user(
            username="sessionuser",
            email="session@example.com",
            password="SecurePass123!",
        )

        login_result = auth_manager.user_login(
            username="sessionuser",
            password="SecurePass123!",
        )

        session = auth_manager.session_management(
            action="get",
            session_id=login_result["session_id"],
        )

        assert session is not None
        assert session.username == "sessionuser"

    def test_session_list(self, auth_manager):
        """Test listing user sessions."""
        user = auth_manager.create_user(
            username="sessionuser",
            email="session@example.com",
            password="SecurePass123!",
        )

        # Create multiple sessions
        for _ in range(3):
            auth_manager.user_login(
                username="sessionuser",
                password="SecurePass123!",
            )

        sessions = auth_manager.session_management(
            action="list",
            user_id=user.user_id,
        )

        assert len(sessions) == 3

    def test_session_cleanup(self, auth_manager):
        """Test cleaning up expired sessions."""
        # This is a basic test - in practice you'd need to manipulate time
        count = auth_manager.session_management(action="cleanup")
        assert count >= 0

    def test_session_invalidate(self, auth_manager):
        """Test session invalidation."""
        auth_manager.create_user(
            username="sessionuser",
            email="session@example.com",
            password="SecurePass123!",
        )

        login_result = auth_manager.user_login(
            username="sessionuser",
            password="SecurePass123!",
        )

        result = auth_manager.session_management(
            action="invalidate",
            session_id=login_result["session_id"],
        )

        assert result is True


class TestRoleManagement:
    """Tests for role management."""

    def test_create_role(self, auth_manager):
        """Test role creation."""
        role = auth_manager.create_role(
            name="tester",
            description="Test role",
            permissions={SystemPermissions.SIMULATION_READ},
        )

        assert role.name == "tester"
        assert role.description == "Test role"
        assert len(role.permissions) == 1

    def test_get_role(self, auth_manager):
        """Test role retrieval."""
        auth_manager.create_role(
            name="tester",
            description="Test role",
        )

        role = auth_manager.get_role("tester")
        assert role is not None
        assert role.name == "tester"

    def test_assign_role_to_user(self, auth_manager, test_roles):
        """Test assigning role to user."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )

        auth_manager.assign_role_to_user(user, test_roles["researcher"])

        assert user.has_role("researcher")

    def test_remove_role_from_user(self, auth_manager, test_roles):
        """Test removing role from user."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            roles={test_roles["researcher"]},
        )

        auth_manager.remove_role_from_user(user, test_roles["researcher"])

        assert not user.has_role("researcher")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
