"""
Core Authentication Manager for user authentication and session management.

This module provides the main AuthenticationManager class that coordinates
all authentication, authorization, and session management operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
import uuid

from .models import User, Role, Permission, Session, UserStatus
from .password_handler import PasswordHandler
from .jwt_handler import JWTHandler, TokenBlacklist
from .rbac import RBACManager
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidCredentialsError,
    SessionExpiredError,
    UserNotFoundError,
    DuplicateUserError,
    InvalidTokenError,
)

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """
    Core authentication and authorization manager.

    Provides comprehensive user authentication, role-based access control,
    permission validation, and session management functionality.

    This is the main entry point for all authentication operations in the system.
    """

    # Account lockout settings
    MAX_FAILED_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION = timedelta(minutes=30)

    # Session settings
    DEFAULT_SESSION_DURATION = timedelta(hours=1)
    MAX_CONCURRENT_SESSIONS = 5

    def __init__(
        self,
        secret_key: str,
        password_rounds: int = 12,
        session_duration: Optional[timedelta] = None,
    ):
        """
        Initialize Authentication Manager.

        Args:
            secret_key: Secret key for JWT token signing (min 32 characters)
            password_rounds: Bcrypt work factor for password hashing (4-31)
            session_duration: Default session duration (default: 1 hour)

        Raises:
            ValueError: If secret_key is too short or password_rounds invalid
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")

        self.password_handler = PasswordHandler(rounds=password_rounds)
        self.jwt_handler = JWTHandler(secret_key=secret_key)
        self.rbac_manager = RBACManager()
        self.token_blacklist = TokenBlacklist()

        self.session_duration = session_duration or self.DEFAULT_SESSION_DURATION

        # In-memory storage (replace with database in production)
        self._users: Dict[str, User] = {}  # username -> User
        self._users_by_id: Dict[str, User] = {}  # user_id -> User
        self._sessions: Dict[str, Session] = {}  # session_id -> Session
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self._roles: Dict[str, Role] = {}  # role_name -> Role

        logger.info("AuthenticationManager initialized successfully")

    # ============================================================================
    # User Management
    # ============================================================================

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[Set[Role]] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        validate_password: bool = True,
    ) -> User:
        """
        Create a new user account.

        Args:
            username: Unique username
            email: User email address
            password: Plain text password (will be hashed)
            roles: Set of roles to assign to user
            first_name: User's first name
            last_name: User's last name
            validate_password: Whether to validate password strength

        Returns:
            Created User object

        Raises:
            DuplicateUserError: If username already exists
            ValueError: If password validation fails
        """
        if username in self._users:
            raise DuplicateUserError(username)

        # Validate password strength if requested
        if validate_password:
            is_valid, errors = self.password_handler.validate_password_strength(password)
            if not is_valid:
                raise ValueError(f"Password validation failed: {', '.join(errors)}")

        # Hash password
        password_hash = self.password_handler.hash_password(password)

        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or set(),
            first_name=first_name,
            last_name=last_name,
        )

        # Store user
        self._users[username] = user
        self._users_by_id[user.user_id] = user

        logger.info(f"User created: {username} (ID: {user.user_id})")
        return user

    def get_user(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username to look up

        Returns:
            User object if found, None otherwise
        """
        return self._users.get(username)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by user ID.

        Args:
            user_id: User ID to look up

        Returns:
            User object if found, None otherwise
        """
        return self._users_by_id.get(user_id)

    def delete_user(self, username: str) -> bool:
        """
        Delete a user account.

        Args:
            username: Username to delete

        Returns:
            True if user was deleted, False if not found
        """
        user = self._users.pop(username, None)
        if user:
            self._users_by_id.pop(user.user_id, None)
            # Invalidate all user sessions
            self._invalidate_user_sessions(user.user_id)
            logger.info(f"User deleted: {username}")
            return True
        return False

    # ============================================================================
    # Authentication - user_login()
    # ============================================================================

    def user_login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate user and create session.

        This is the main login method that validates credentials, creates
        a session, and returns authentication tokens.

        Args:
            username: Username
            password: Plain text password
            ip_address: Optional client IP address
            user_agent: Optional client user agent string

        Returns:
            Dictionary containing:
                - access_token: JWT access token
                - refresh_token: JWT refresh token
                - session_id: Session identifier
                - user: User information
                - expires_at: Token expiration time

        Raises:
            InvalidCredentialsError: If credentials are invalid
            AuthenticationError: If account is locked or inactive
        """
        user = self.get_user(username)

        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            raise InvalidCredentialsError()

        # Check if account is locked
        if user.status == UserStatus.LOCKED:
            logger.warning(f"Login attempt for locked account: {username}")
            raise AuthenticationError(
                "Account is locked due to multiple failed login attempts. "
                "Please contact administrator."
            )

        # Check if account is active
        if not user.is_active():
            logger.warning(f"Login attempt for inactive account: {username} (status: {user.status})")
            raise AuthenticationError(f"Account is {user.status.value}")

        # Verify password
        if not self.password_handler.verify_password(password, user.password_hash):
            user.increment_failed_login()

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.MAX_FAILED_LOGIN_ATTEMPTS:
                user.lock_account()
                logger.warning(
                    f"Account locked due to failed login attempts: {username} "
                    f"({user.failed_login_attempts} attempts)"
                )
                raise AuthenticationError(
                    "Account has been locked due to multiple failed login attempts"
                )

            logger.warning(
                f"Failed login attempt for user: {username} "
                f"({user.failed_login_attempts}/{self.MAX_FAILED_LOGIN_ATTEMPTS})"
            )
            raise InvalidCredentialsError()

        # Successful authentication
        user.reset_failed_login()
        user.update_last_login()

        # Generate tokens
        roles = [role.name for role in user.roles]
        access_token = self.jwt_handler.generate_access_token(
            user_id=user.user_id,
            username=user.username,
            roles=roles,
        )
        refresh_token = self.jwt_handler.generate_refresh_token(
            user_id=user.user_id,
            username=user.username,
        )

        # Create session
        session = self._create_session(
            user=user,
            token=access_token,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        logger.info(
            f"User logged in successfully: {username} "
            f"(Session: {session.session_id}, IP: {ip_address})"
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session.session_id,
            "user": user.to_dict(include_sensitive=False),
            "expires_at": session.expires_at.isoformat(),
        }

    def logout(self, session_id: str) -> bool:
        """
        Log out user and invalidate session.

        Args:
            session_id: Session ID to logout

        Returns:
            True if logout successful, False if session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Invalidate session
        session.invalidate()

        # Add token to blacklist
        self.token_blacklist.add_token(session.token)

        # Remove from active sessions
        if session.user_id in self._user_sessions:
            self._user_sessions[session.user_id].discard(session_id)

        logger.info(f"User logged out: {session.username} (Session: {session_id})")
        return True

    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Dictionary with new access_token and refresh_token

        Raises:
            InvalidTokenError: If refresh token is invalid
            SessionExpiredError: If refresh token has expired
        """
        # Validate refresh token
        payload = self.jwt_handler.decode_token(refresh_token)

        if payload.get("type") != "refresh":
            raise InvalidTokenError("Invalid token type")

        user_id = payload.get("sub")
        username = payload.get("username")

        if not user_id or not username:
            raise InvalidTokenError("Invalid token payload")

        # Get user
        user = self.get_user_by_id(user_id)
        if not user or not user.is_active():
            raise AuthenticationError("User not found or inactive")

        # Generate new tokens
        roles = [role.name for role in user.roles]
        new_access_token = self.jwt_handler.generate_access_token(
            user_id=user.user_id,
            username=user.username,
            roles=roles,
        )
        new_refresh_token = self.jwt_handler.generate_refresh_token(
            user_id=user.user_id,
            username=user.username,
        )

        # Blacklist old refresh token
        self.token_blacklist.add_token(refresh_token)

        logger.info(f"Token refreshed for user: {username}")

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
        }

    # ============================================================================
    # Role-Based Access Control - role_based_access_control()
    # ============================================================================

    def role_based_access_control(
        self,
        user: User,
        required_role: Optional[str] = None,
        required_roles: Optional[Set[str]] = None,
        require_all: bool = False,
    ) -> bool:
        """
        Check if user has required role(s) for access control.

        This method implements role-based access control by validating
        user roles against requirements.

        Args:
            user: User to check
            required_role: Single required role name
            required_roles: Set of required role names
            require_all: If True, user must have all roles; if False, any role

        Returns:
            True if user has required role(s), False otherwise

        Raises:
            ValueError: If neither required_role nor required_roles provided
            AuthorizationError: If user lacks required roles
        """
        if not user.is_active():
            raise AuthorizationError("User account is not active")

        if required_role:
            return self.rbac_manager.require_role(user, required_role, raise_exception=True)

        if required_roles:
            if require_all:
                return self.rbac_manager.require_all_roles(
                    user, required_roles, raise_exception=True
                )
            else:
                return self.rbac_manager.require_any_role(
                    user, required_roles, raise_exception=True
                )

        raise ValueError("Either required_role or required_roles must be provided")

    def check_user_role(self, user: User, role_name: str) -> bool:
        """
        Check if user has a specific role (non-throwing version).

        Args:
            user: User to check
            role_name: Role name to check

        Returns:
            True if user has role, False otherwise
        """
        return user.has_role(role_name)

    # ============================================================================
    # Permission Validation - permission_validator()
    # ============================================================================

    def permission_validator(
        self,
        user: User,
        permission: Optional[Permission] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """
        Validate if user has required permission.

        This method implements fine-grained permission validation for
        access control decisions.

        Args:
            user: User to validate
            permission: Permission object to check
            resource: Resource name (alternative to permission object)
            action: Action name (alternative to permission object)

        Returns:
            True if user has permission, False otherwise

        Raises:
            ValueError: If neither permission nor (resource, action) provided
            AuthorizationError: If user lacks required permission
        """
        if not user.is_active():
            raise AuthorizationError("User account is not active")

        if permission:
            return self.rbac_manager.require_permission(
                user, permission, raise_exception=True
            )

        if resource and action:
            has_permission = self.rbac_manager.check_permission_by_action(
                user, resource, action
            )
            if not has_permission:
                raise AuthorizationError(
                    f"User '{user.username}' lacks permission: {action}:{resource}"
                )
            return True

        raise ValueError("Either permission or (resource, action) must be provided")

    def check_user_permission(
        self,
        user: User,
        permission: Optional[Permission] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """
        Check if user has permission (non-throwing version).

        Args:
            user: User to check
            permission: Permission object to check
            resource: Resource name (alternative to permission object)
            action: Action name (alternative to permission object)

        Returns:
            True if user has permission, False otherwise
        """
        if permission:
            return self.rbac_manager.check_permission(user, permission)

        if resource and action:
            return self.rbac_manager.check_permission_by_action(user, resource, action)

        return False

    # ============================================================================
    # Session Management - session_management()
    # ============================================================================

    def session_management(
        self,
        action: str,
        session_id: Optional[str] = None,
        token: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """
        Manage user sessions with various operations.

        This method provides comprehensive session management including
        validation, retrieval, listing, and cleanup.

        Args:
            action: Action to perform:
                - "validate": Validate a session
                - "get": Get session details
                - "list": List all sessions for a user
                - "cleanup": Remove expired sessions
                - "invalidate": Invalidate a session
            session_id: Session ID (required for validate, get, invalidate)
            token: Access token (alternative to session_id for validate)
            user_id: User ID (required for list)

        Returns:
            Depends on action:
                - validate: Session object if valid
                - get: Session object
                - list: List of session dictionaries
                - cleanup: Number of sessions cleaned up
                - invalidate: True if successful

        Raises:
            ValueError: If required parameters missing
            SessionExpiredError: If session is expired
            InvalidTokenError: If token is invalid
        """
        if action == "validate":
            if session_id:
                return self._validate_session(session_id)
            elif token:
                return self._validate_token(token)
            else:
                raise ValueError("Either session_id or token required for validate")

        elif action == "get":
            if not session_id:
                raise ValueError("session_id required for get")
            return self._get_session(session_id)

        elif action == "list":
            if not user_id:
                raise ValueError("user_id required for list")
            return self._list_user_sessions(user_id)

        elif action == "cleanup":
            return self._cleanup_expired_sessions()

        elif action == "invalidate":
            if not session_id:
                raise ValueError("session_id required for invalidate")
            return self.logout(session_id)

        else:
            raise ValueError(
                f"Invalid action: {action}. Must be one of: "
                "validate, get, list, cleanup, invalidate"
            )

    def _create_session(
        self,
        user: User,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Create a new session for authenticated user."""
        now = datetime.utcnow()
        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=user.user_id,
            username=user.username,
            token=token,
            created_at=now,
            expires_at=now + self.session_duration,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store session
        self._sessions[session.session_id] = session

        # Track user sessions
        if user.user_id not in self._user_sessions:
            self._user_sessions[user.user_id] = set()
        self._user_sessions[user.user_id].add(session.session_id)

        # Enforce max concurrent sessions
        self._enforce_session_limit(user.user_id)

        return session

    def _validate_session(self, session_id: str) -> Session:
        """Validate session by ID."""
        session = self._sessions.get(session_id)

        if not session:
            raise SessionExpiredError("Session not found")

        if not session.is_valid():
            raise SessionExpiredError("Session has expired or been invalidated")

        # Update activity
        session.update_activity()

        return session

    def _validate_token(self, token: str) -> Session:
        """Validate session by token."""
        # Check if token is blacklisted
        if self.token_blacklist.is_blacklisted(token):
            raise InvalidTokenError("Token has been revoked")

        # Validate token
        payload = self.jwt_handler.decode_token(token)

        # Find session with this token
        for session in self._sessions.values():
            if session.token == token:
                if not session.is_valid():
                    raise SessionExpiredError("Session has expired")
                session.update_activity()
                return session

        raise SessionExpiredError("Session not found")

    def _get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def _list_user_sessions(self, user_id: str) -> list[Dict[str, Any]]:
        """List all active sessions for a user."""
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []

        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session and session.is_valid():
                sessions.append(session.to_dict())

        return sessions

    def _cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        expired = []

        for session_id, session in self._sessions.items():
            if not session.is_valid():
                expired.append(session_id)

        for session_id in expired:
            session = self._sessions.pop(session_id)
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def _invalidate_user_sessions(self, user_id: str) -> None:
        """Invalidate all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, set()).copy()

        for session_id in session_ids:
            self.logout(session_id)

    def _enforce_session_limit(self, user_id: str) -> None:
        """Enforce maximum concurrent sessions per user."""
        if user_id not in self._user_sessions:
            return

        session_ids = list(self._user_sessions[user_id])

        if len(session_ids) > self.MAX_CONCURRENT_SESSIONS:
            # Remove oldest sessions
            sessions = [
                (sid, self._sessions[sid])
                for sid in session_ids
                if sid in self._sessions
            ]
            sessions.sort(key=lambda x: x[1].created_at)

            to_remove = len(sessions) - self.MAX_CONCURRENT_SESSIONS
            for session_id, _ in sessions[:to_remove]:
                self.logout(session_id)

            logger.info(
                f"Enforced session limit for user {user_id}: "
                f"removed {to_remove} oldest sessions"
            )

    # ============================================================================
    # Role Management
    # ============================================================================

    def create_role(
        self,
        name: str,
        description: str,
        permissions: Optional[Set[Permission]] = None,
        is_system_role: bool = False,
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Unique role name
            description: Role description
            permissions: Set of permissions for this role
            is_system_role: Whether this is a system role (cannot be deleted)

        Returns:
            Created Role object

        Raises:
            ValueError: If role already exists
        """
        if name in self._roles:
            raise ValueError(f"Role '{name}' already exists")

        role = Role(
            name=name,
            description=description,
            permissions=permissions or set(),
            is_system_role=is_system_role,
        )

        self._roles[name] = role
        logger.info(f"Role created: {name}")
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """
        Get role by name.

        Args:
            name: Role name

        Returns:
            Role object if found, None otherwise
        """
        return self._roles.get(name)

    def assign_role_to_user(self, user: User, role: Role) -> None:
        """
        Assign a role to a user.

        Args:
            user: User to assign role to
            role: Role to assign
        """
        user.add_role(role)
        logger.info(f"Role '{role.name}' assigned to user: {user.username}")

    def remove_role_from_user(self, user: User, role: Role) -> None:
        """
        Remove a role from a user.

        Args:
            user: User to remove role from
            role: Role to remove
        """
        user.remove_role(role)
        logger.info(f"Role '{role.name}' removed from user: {user.username}")
