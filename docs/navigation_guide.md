# Navigation & Routing System Guide

## Overview

The PV Circularity Simulator features a comprehensive, production-ready navigation and routing system built specifically for Streamlit applications. This guide covers all aspects of the NavigationManager system.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Page Registry](#page-registry)
4. [Route Handler](#route-handler)
5. [Breadcrumbs](#breadcrumbs)
6. [Deep Linking](#deep-linking)
7. [Advanced Features](#advanced-features)
8. [API Reference](#api-reference)

## Architecture

The navigation system is designed around four main pillars:

```
┌─────────────────────────────────────────────────────┐
│             NavigationManager                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ Page Registry│  │ Route Handler│                │
│  └──────────────┘  └──────────────┘                │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  Breadcrumbs │  │ Deep Linking │                │
│  └──────────────┘  └──────────────┘                │
│                                                      │
└─────────────────────────────────────────────────────┘
           ↓                    ↓
    Streamlit                URL
  Session State           Parameters
```

## Core Components

### NavigationManager

The central class that manages all navigation functionality.

```python
from navigation import NavigationManager

nav = NavigationManager(
    default_page="home",           # Default page to show
    auth_callback=check_auth,      # Optional auth function
    error_handler=handle_error     # Optional error handler
)
```

### PageConfig

Configuration dataclass for registered pages.

```python
from navigation import PageConfig, AccessLevel

config = PageConfig(
    name="dashboard",              # Unique page identifier
    title="Dashboard",             # Display title
    icon="📊",                     # Icon (emoji or name)
    render_func=render_dashboard,  # Rendering function
    description="Main dashboard",  # Optional description
    access_level=AccessLevel.PUBLIC,  # Access level
    parent="home",                 # Parent page for breadcrumbs
    show_in_sidebar=True,          # Show in navigation
    url_params={"view": "default"}, # Default URL params
    keywords=["dashboard", "overview"],  # Search keywords
    order=1,                       # Display order
    custom_auth_check=None         # Custom auth function
)
```

### Route

Represents a navigation route with parameters.

```python
from navigation import Route

route = Route(
    page_name="dashboard",
    params={"filter": "active", "sort": "date"},
    query_string="page=dashboard&filter=active&sort=date"
)

# Convert to URL
url = route.to_url()
```

### AccessLevel

Authorization levels for pages.

```python
from navigation import AccessLevel

AccessLevel.PUBLIC        # No authentication required
AccessLevel.AUTHENTICATED # Requires authentication
AccessLevel.ADMIN         # Requires admin privileges
AccessLevel.CUSTOM        # Custom authorization logic
```

## Page Registry

The page registry system allows you to register pages with comprehensive metadata.

### Decorator-Based Registration

```python
@nav.page_registry(
    name="dashboard",
    title="Dashboard",
    icon="📊",
    description="Main application dashboard",
    parent="home",
    order=1,
    keywords=["dashboard", "overview", "stats"]
)
def dashboard_page():
    st.title("Dashboard")
    st.write("Dashboard content here")
```

### Programmatic Registration

```python
def my_page():
    st.write("Page content")

config = PageConfig(
    name="mypage",
    title="My Page",
    icon="🎯",
    render_func=my_page
)

nav.register_page(config)
```

### Hierarchical Pages

Create parent-child relationships for breadcrumb navigation:

```python
# Parent page
@nav.page_registry(name="settings", title="Settings", icon="⚙️")
def settings():
    st.write("Settings")

# Child page
@nav.page_registry(
    name="user_settings",
    title="User Settings",
    icon="👤",
    parent="settings"  # References parent
)
def user_settings():
    st.write("User settings")
```

### Access Control

Control page access with authorization levels:

```python
# Public page (no auth required)
@nav.page_registry(
    name="home",
    title="Home",
    icon="🏠",
    access_level=AccessLevel.PUBLIC
)
def home():
    st.write("Welcome!")

# Authenticated page
@nav.page_registry(
    name="profile",
    title="Profile",
    icon="👤",
    access_level=AccessLevel.AUTHENTICATED
)
def profile():
    st.write("User profile")

# Custom authorization
def check_admin():
    return st.session_state.get("is_admin", False)

@nav.page_registry(
    name="admin",
    title="Admin Panel",
    icon="🔧",
    custom_auth_check=check_admin
)
def admin():
    st.write("Admin panel")
```

## Route Handler

The route handler processes navigation and renders pages.

### Basic Usage

```python
# In your main Streamlit app
def main():
    nav = NavigationManager()

    # Register pages...

    # Handle routing
    nav.route_handler()

if __name__ == "__main__":
    main()
```

### Navigation Methods

```python
# Navigate to a page
nav.navigate("dashboard")

# Navigate with parameters
nav.navigate("user_profile", params={"user_id": "123"})

# Navigate without adding to history
nav.navigate("popup", add_to_history=False)

# Go back to previous page
nav.go_back()

# Get current page
current = nav.get_current_page()  # Returns: "dashboard"

# Get current parameters
params = nav.get_current_params()  # Returns: {"user_id": "123"}

# Get previous page
previous = nav.get_previous_page()  # Returns: "home"
```

### Navigation History

```python
# Get navigation history
history = nav.get_navigation_history()
# Returns: [
#     {"page": "home", "params": {}},
#     {"page": "dashboard", "params": {"view": "overview"}},
#     {"page": "settings", "params": {}}
# ]

# Clear history
nav.clear_history()
```

## Breadcrumbs

Display hierarchical navigation breadcrumbs.

### Basic Breadcrumbs

```python
# Display breadcrumbs
nav.breadcrumbs()
# Renders: 🏠 Home / 📊 Dashboard / ⚙️ Settings
```

### Customized Breadcrumbs

```python
# Custom separator
nav.breadcrumbs(separator=" → ")
# Renders: 🏠 Home → 📊 Dashboard → ⚙️ Settings

# Without icons
nav.breadcrumbs(show_icons=False)
# Renders: Home / Dashboard / Settings

# Non-clickable breadcrumbs
nav.breadcrumbs(clickable=False)

# Render in specific container
with st.sidebar:
    nav.breadcrumbs(container=st.sidebar)
```

### Programmatic Access

```python
# Get breadcrumb trail without rendering
trail = nav.breadcrumbs(container=None)
# Returns: [
#     {"name": "home", "title": "Home"},
#     {"name": "dashboard", "title": "Dashboard"},
#     {"name": "settings", "title": "Settings"}
# ]

# Use breadcrumb data
for crumb in trail:
    st.write(f"Page: {crumb['title']}")
```

## Deep Linking

Enable URL-based navigation and shareable links.

### Automatic Deep Linking

The route handler automatically processes URL parameters:

```
http://localhost:8501/?page=dashboard&filter=active&sort=date
```

This will:
1. Navigate to the "dashboard" page
2. Pass `filter=active` and `sort=date` as parameters

### Manual Deep Linking

```python
# Extract route from URL
route = nav.deep_linking()

# Access route information
page_name = route.page_name      # "dashboard"
params = route.params            # {"filter": "active", "sort": "date"}
query_string = route.query_string  # "page=dashboard&filter=active..."
```

### Setting URL Parameters

```python
# Set URL parameters for current page
nav.set_query_params({
    "filter": "active",
    "sort": "date",
    "view": "grid"
})

# URL becomes: ?page=current_page&filter=active&sort=date&view=grid
```

### Creating Shareable Links

```python
# Get deep link for current page
current_page = nav.get_current_page()
params = nav.get_current_params()

if params:
    from urllib.parse import urlencode
    query_string = urlencode({"page": current_page, **params})
    deep_link = f"{base_url}?{query_string}"
else:
    deep_link = f"{base_url}?page={current_page}"

st.code(deep_link)
# Renders: http://localhost:8501/?page=dashboard&filter=active
```

## Advanced Features

### Sidebar Navigation

Render automatic navigation menu in sidebar:

```python
with st.sidebar:
    nav.render_sidebar_navigation(
        show_icons=True,        # Show page icons
        group_by_parent=True,   # Group child pages under parents
        show_search=True        # Show search box for pages
    )
```

### Search Functionality

Pages are searchable by title, description, and keywords:

```python
@nav.page_registry(
    name="analytics",
    title="Analytics Dashboard",
    description="Advanced analytics and reporting",
    keywords=["analytics", "reports", "charts", "data"]
)
def analytics():
    st.write("Analytics")
```

Users can search for pages using the sidebar search box.

### Error Handling

Provide custom error handling:

```python
def handle_nav_error(error: Exception, page_name: str):
    st.error(f"Error on page '{page_name}': {str(error)}")
    # Log error, send notification, etc.

nav = NavigationManager(error_handler=handle_nav_error)
```

### Authentication Integration

Implement global authentication:

```python
def check_authentication():
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)

nav = NavigationManager(
    default_page="login",
    auth_callback=check_authentication
)
```

### Page-Specific Parameters

Set default parameters for pages:

```python
@nav.page_registry(
    name="dashboard",
    title="Dashboard",
    icon="📊",
    url_params={"view": "overview", "period": "month"}
)
def dashboard():
    params = nav.get_current_params()
    view = params.get("view", "overview")
    period = params.get("period", "month")

    st.write(f"View: {view}, Period: {period}")
```

## API Reference

### NavigationManager Methods

#### `__init__(default_page, auth_callback, error_handler)`
Initialize the NavigationManager.

#### `page_registry(**kwargs) -> Callable`
Decorator for registering pages.

#### `register_page(page_config: PageConfig) -> None`
Programmatically register a page.

#### `navigate(page_name: str, params: Dict, add_to_history: bool) -> bool`
Navigate to a page.

#### `route_handler() -> None`
Handle routing and render current page.

#### `breadcrumbs(separator, show_icons, clickable, container) -> List[Dict]`
Display and return breadcrumb trail.

#### `deep_linking(use_query_params: bool) -> Route`
Extract route from URL parameters.

#### `get_current_page() -> str`
Get current page name.

#### `get_current_params() -> Dict`
Get current URL parameters.

#### `get_previous_page() -> Optional[str]`
Get previous page name.

#### `get_navigation_history() -> List[Dict]`
Get navigation history.

#### `clear_history() -> None`
Clear navigation history.

#### `go_back() -> bool`
Navigate to previous page.

#### `set_query_params(params: Dict) -> None`
Set URL query parameters.

#### `render_sidebar_navigation(show_icons, group_by_parent, show_search) -> None`
Render navigation menu in sidebar.

### Session State Keys

The NavigationManager uses these session state keys:

- `nav_current_page`: Current page name
- `nav_previous_page`: Previous page name
- `nav_params`: Current URL parameters
- `nav_history`: Navigation history
- `nav_breadcrumbs`: Breadcrumb trail

## Best Practices

### 1. Consistent Naming

Use consistent, descriptive page names:

```python
# Good
@nav.page_registry(name="user_profile", title="User Profile")
@nav.page_registry(name="user_settings", title="User Settings")

# Avoid
@nav.page_registry(name="page1", title="Profile")
@nav.page_registry(name="pg2", title="Settings")
```

### 2. Hierarchical Organization

Use parent-child relationships for related pages:

```python
@nav.page_registry(name="admin", title="Admin")
@nav.page_registry(name="admin_users", title="Users", parent="admin")
@nav.page_registry(name="admin_settings", title="Settings", parent="admin")
```

### 3. Meaningful Keywords

Add search keywords for better discoverability:

```python
@nav.page_registry(
    name="data_export",
    title="Data Export",
    keywords=["export", "download", "csv", "excel", "data"]
)
```

### 4. Access Control

Always specify access levels explicitly:

```python
@nav.page_registry(
    name="public_page",
    access_level=AccessLevel.PUBLIC
)

@nav.page_registry(
    name="admin_page",
    access_level=AccessLevel.ADMIN
)
```

### 5. Error Handling

Implement robust error handling:

```python
nav = NavigationManager(
    error_handler=lambda e, page: st.error(f"Error: {e}")
)
```

## Examples

### Complete Application Example

```python
import streamlit as st
from navigation import NavigationManager, AccessLevel

st.set_page_config(page_title="My App", layout="wide")

def main():
    nav = NavigationManager(default_page="home")

    @nav.page_registry(
        name="home",
        title="Home",
        icon="🏠",
        order=0
    )
    def home():
        st.title("Home")
        st.write("Welcome!")

        if st.button("Go to Dashboard"):
            nav.navigate("dashboard")
            st.rerun()

    @nav.page_registry(
        name="dashboard",
        title="Dashboard",
        icon="📊",
        parent="home",
        order=1
    )
    def dashboard():
        st.title("Dashboard")

        # Use URL parameters
        params = nav.get_current_params()
        view = params.get("view", "default")

        st.write(f"View: {view}")

    # Sidebar navigation
    with st.sidebar:
        nav.render_sidebar_navigation()

    # Breadcrumbs
    nav.breadcrumbs(separator=" → ")

    st.markdown("---")

    # Handle routing
    nav.route_handler()

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Pages Not Appearing in Sidebar

Ensure `show_in_sidebar=True` (default):

```python
@nav.page_registry(name="page", title="Page", show_in_sidebar=True)
```

### Deep Links Not Working

Check that URL parameters are properly formatted:

```
Correct: ?page=dashboard&filter=active
Wrong:   ?dashboard&filter=active
```

### Authorization Issues

Verify auth callback returns boolean:

```python
def check_auth():
    return bool(st.session_state.get("user"))  # Must return True/False
```

### Navigation Not Updating

Call `st.rerun()` after navigation:

```python
if st.button("Navigate"):
    nav.navigate("other_page")
    st.rerun()  # Force Streamlit to rerun
```

## Performance Tips

1. **Lazy Load Pages**: Only import page modules when needed
2. **Cache Data**: Use `@st.cache_data` for expensive operations
3. **Minimal Reruns**: Avoid unnecessary `st.rerun()` calls
4. **Efficient Rendering**: Keep page render functions lightweight

## Conclusion

The Navigation & Routing System provides a comprehensive, production-ready solution for Streamlit applications. With features like page registry, route handling, breadcrumbs, and deep linking, you can build complex multi-page applications with ease.

For more examples and use cases, see the `src/main.py` file in the repository.
