# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🔐 Authentication & Access Control System

Production-ready authentication and authorization system with JWT tokens, password hashing, and role-based access control.

### Features

- **User Authentication**: Secure login with JWT tokens and session management
- **Password Security**: Bcrypt hashing with configurable work factor
- **Role-Based Access Control (RBAC)**: Hierarchical roles with permission inheritance
- **Permission System**: Fine-grained permissions for resource-level access control
- **Session Management**: Token-based sessions with expiration and invalidation
- **Account Security**: Failed login tracking, account lockout, and password validation
- **Production-Ready**: Full type hints, comprehensive docstrings, and extensive test coverage

### Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy and edit .env file)
cp .env.example .env
# Edit .env and set your AUTH_SECRET_KEY
```

#### Basic Usage

```python
from src.auth import AuthenticationManager, Role, Permission
from config import AuthConfig

# Initialize authentication manager
auth_manager = AuthenticationManager(
    secret_key=AuthConfig.SECRET_KEY,
    password_rounds=12,
)

# Create a role with permissions
admin_role = auth_manager.create_role(
    name="admin",
    description="Administrator",
    permissions={Permission("system:admin", "Full access", "*", "*")},
)

# Create a user
user = auth_manager.create_user(
    username="admin",
    email="admin@example.com",
    password="SecurePass123!",
    roles={admin_role},
)

# User login
result = auth_manager.user_login(
    username="admin",
    password="SecurePass123!",
)
print(f"Access Token: {result['access_token']}")

# Check role-based access
auth_manager.role_based_access_control(user, required_role="admin")

# Validate permission
auth_manager.permission_validator(
    user=user,
    resource="simulation",
    action="execute",
)

# Manage session
session = auth_manager.session_management(
    action="validate",
    session_id=result["session_id"],
)
```

### Core API Methods

#### `user_login(username, password, ip_address=None, user_agent=None)`
Authenticate user and create session with JWT tokens.

**Returns**: Dictionary with access_token, refresh_token, session_id, user info, and expiration.

#### `role_based_access_control(user, required_role=None, required_roles=None, require_all=False)`
Check if user has required role(s) for access control.

**Returns**: True if user has required role(s).

**Raises**: `AuthorizationError` if user lacks required roles.

#### `permission_validator(user, permission=None, resource=None, action=None)`
Validate if user has required permission.

**Returns**: True if user has permission.

**Raises**: `AuthorizationError` if user lacks required permission.

#### `session_management(action, session_id=None, token=None, user_id=None)`
Manage sessions with actions: validate, get, list, cleanup, invalidate.

**Returns**: Depends on action (Session object, list of sessions, cleanup count, or success boolean).

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/auth --cov-report=html

# Run specific test file
pytest tests/test_authentication_manager.py -v
```

### Running Examples

```bash
# Run basic usage example
python examples/basic_usage.py
```

### Architecture

```
src/auth/
├── authentication_manager.py  # Core AuthenticationManager class
├── models.py                   # User, Role, Permission, Session models
├── password_handler.py         # Password hashing and validation
├── jwt_handler.py              # JWT token generation and validation
├── rbac.py                     # Role-based access control
└── exceptions.py               # Custom exceptions

tests/
├── test_authentication_manager.py
├── test_password_handler.py
└── test_jwt_handler.py

examples/
└── basic_usage.py              # Usage examples
```

### Security Features

- **Password Hashing**: Bcrypt with configurable work factor (default: 12 rounds)
- **JWT Tokens**: HS256 algorithm with configurable expiration
- **Account Lockout**: Automatic lockout after failed login attempts
- **Session Management**: Token blacklist and concurrent session limits
- **Password Validation**: Configurable strength requirements
- **Token Refresh**: Secure token renewal without re-authentication

### Configuration

Configuration is managed via environment variables (see `.env.example`):

- `AUTH_SECRET_KEY`: JWT signing key (min 32 characters)
- `BCRYPT_ROUNDS`: Password hashing work factor (4-31, default: 12)
- `ACCESS_TOKEN_EXPIRY_HOURS`: Access token lifetime (default: 1)
- `REFRESH_TOKEN_EXPIRY_DAYS`: Refresh token lifetime (default: 7)
- `MAX_FAILED_LOGIN_ATTEMPTS`: Failed login threshold (default: 5)
- `MAX_CONCURRENT_SESSIONS`: Per-user session limit (default: 5)

### Type Hints & Documentation

All code includes:
- Full type hints for all function parameters and return values
- Comprehensive docstrings with Args, Returns, and Raises sections
- Detailed inline comments for complex logic
- Production-ready error handling

### License

MIT License - see LICENSE file for details.
