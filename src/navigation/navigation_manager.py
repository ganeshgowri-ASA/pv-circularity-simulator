"""Navigation Manager - Core routing and navigation system for Streamlit applications.

This module provides a comprehensive navigation system with support for:
- Page registration with metadata
- Route handling with URL parameters
- Breadcrumb navigation
- Deep linking support
- Session state management
- Authentication and authorization hooks
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is required for NavigationManager. "
        "Install it with: pip install streamlit"
    )


class AccessLevel(Enum):
    """Access levels for page authorization."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"
    CUSTOM = "custom"


@dataclass
class PageConfig:
    """Configuration for a registered page.

    Attributes:
        name: Unique identifier for the page (used in routing).
        title: Display title for the page.
        icon: Icon to display (emoji or icon name).
        render_func: Function to render the page content.
        description: Optional description of the page.
        access_level: Required access level to view the page.
        parent: Parent page name for breadcrumb navigation.
        show_in_sidebar: Whether to show this page in navigation sidebar.
        url_params: Default URL parameters for this page.
        keywords: Search keywords for the page.
        order: Display order in navigation (lower numbers first).
        custom_auth_check: Optional custom authorization function.
    """

    name: str
    title: str
    icon: str
    render_func: Callable[[], None]
    description: str = ""
    access_level: AccessLevel = AccessLevel.PUBLIC
    parent: Optional[str] = None
    show_in_sidebar: bool = True
    url_params: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    order: int = 0
    custom_auth_check: Optional[Callable[[], bool]] = None

    def __post_init__(self):
        """Validate page configuration after initialization."""
        if not self.name:
            raise ValueError("Page name cannot be empty")
        if not callable(self.render_func):
            raise ValueError(f"render_func must be callable for page '{self.name}'")


@dataclass
class Route:
    """Represents a navigation route.

    Attributes:
        page_name: Name of the target page.
        params: URL parameters for the route.
        query_string: Raw query string.
    """

    page_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    query_string: str = ""

    def to_url(self) -> str:
        """Convert route to URL query string.

        Returns:
            URL query string representation of the route.
        """
        params = {"page": self.page_name, **self.params}
        return urlencode(params)


class NavigationManager:
    """Production-ready navigation and routing manager for Streamlit applications.

    This class provides a comprehensive navigation system with page registration,
    routing, breadcrumbs, and deep linking capabilities. It manages navigation
    state through Streamlit's session state and supports URL parameter handling.

    Example:
        >>> nav = NavigationManager()
        >>>
        >>> # Register a page
        >>> @nav.page_registry(name="home", title="Home", icon="🏠")
        >>> def home_page():
        >>>     st.write("Welcome to the home page!")
        >>>
        >>> # Navigate to a page
        >>> nav.navigate("home")
        >>>
        >>> # Render the current page
        >>> nav.route_handler()

    Attributes:
        _pages: Dictionary of registered pages.
        _navigation_history: List of navigation history.
        _auth_callback: Optional authentication check callback.
        _error_handler: Optional error handler callback.
    """

    # Session state keys
    _SESSION_CURRENT_PAGE = "nav_current_page"
    _SESSION_PREVIOUS_PAGE = "nav_previous_page"
    _SESSION_PARAMS = "nav_params"
    _SESSION_HISTORY = "nav_history"
    _SESSION_BREADCRUMBS = "nav_breadcrumbs"

    def __init__(
        self,
        default_page: str = "home",
        auth_callback: Optional[Callable[[], bool]] = None,
        error_handler: Optional[Callable[[Exception, str], None]] = None,
    ):
        """Initialize the NavigationManager.

        Args:
            default_page: Default page to display when no route is specified.
            auth_callback: Optional global authentication check function.
            error_handler: Optional error handler for navigation errors.
        """
        self._pages: Dict[str, PageConfig] = {}
        self._default_page = default_page
        self._auth_callback = auth_callback
        self._error_handler = error_handler
        self._initialized = False

        # Initialize session state
        self._init_session_state()

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state for navigation."""
        if self._SESSION_CURRENT_PAGE not in st.session_state:
            st.session_state[self._SESSION_CURRENT_PAGE] = self._default_page
        if self._SESSION_PREVIOUS_PAGE not in st.session_state:
            st.session_state[self._SESSION_PREVIOUS_PAGE] = None
        if self._SESSION_PARAMS not in st.session_state:
            st.session_state[self._SESSION_PARAMS] = {}
        if self._SESSION_HISTORY not in st.session_state:
            st.session_state[self._SESSION_HISTORY] = []
        if self._SESSION_BREADCRUMBS not in st.session_state:
            st.session_state[self._SESSION_BREADCRUMBS] = []

    def page_registry(
        self,
        name: str,
        title: str,
        icon: str = "📄",
        description: str = "",
        access_level: AccessLevel = AccessLevel.PUBLIC,
        parent: Optional[str] = None,
        show_in_sidebar: bool = True,
        url_params: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        order: int = 0,
        custom_auth_check: Optional[Callable[[], bool]] = None,
    ) -> Callable:
        """Register a page with the navigation system (decorator).

        This method can be used as a decorator to register page rendering functions
        with comprehensive metadata for routing, display, and authorization.

        Args:
            name: Unique identifier for the page.
            title: Display title for the page.
            icon: Icon to display (emoji or icon name).
            description: Optional description of the page.
            access_level: Required access level to view the page.
            parent: Parent page name for breadcrumb navigation.
            show_in_sidebar: Whether to show this page in navigation sidebar.
            url_params: Default URL parameters for this page.
            keywords: Search keywords for the page.
            order: Display order in navigation (lower numbers first).
            custom_auth_check: Optional custom authorization function.

        Returns:
            Decorator function for registering the page.

        Raises:
            ValueError: If page name is empty or already registered.

        Example:
            >>> @nav.page_registry(
            >>>     name="dashboard",
            >>>     title="Dashboard",
            >>>     icon="📊",
            >>>     parent="home"
            >>> )
            >>> def dashboard_page():
            >>>     st.write("Dashboard content")
        """

        def decorator(func: Callable[[], None]) -> Callable[[], None]:
            """Inner decorator function."""
            if name in self._pages:
                raise ValueError(f"Page '{name}' is already registered")

            page_config = PageConfig(
                name=name,
                title=title,
                icon=icon,
                render_func=func,
                description=description,
                access_level=access_level,
                parent=parent,
                show_in_sidebar=show_in_sidebar,
                url_params=url_params or {},
                keywords=keywords or [],
                order=order,
                custom_auth_check=custom_auth_check,
            )

            self._pages[name] = page_config
            return func

        return decorator

    def register_page(self, page_config: PageConfig) -> None:
        """Register a page programmatically (non-decorator approach).

        Args:
            page_config: PageConfig instance with page configuration.

        Raises:
            ValueError: If page name is already registered.

        Example:
            >>> def my_page():
            >>>     st.write("My page content")
            >>>
            >>> config = PageConfig(
            >>>     name="mypage",
            >>>     title="My Page",
            >>>     icon="🎯",
            >>>     render_func=my_page
            >>> )
            >>> nav.register_page(config)
        """
        if page_config.name in self._pages:
            raise ValueError(f"Page '{page_config.name}' is already registered")
        self._pages[page_config.name] = page_config

    def get_registered_pages(self) -> Dict[str, PageConfig]:
        """Get all registered pages.

        Returns:
            Dictionary of page name to PageConfig mappings.
        """
        return self._pages.copy()

    def get_page_config(self, page_name: str) -> Optional[PageConfig]:
        """Get configuration for a specific page.

        Args:
            page_name: Name of the page.

        Returns:
            PageConfig if page exists, None otherwise.
        """
        return self._pages.get(page_name)

    def navigate(
        self,
        page_name: str,
        params: Optional[Dict[str, Any]] = None,
        add_to_history: bool = True,
    ) -> bool:
        """Navigate to a specified page.

        Args:
            page_name: Name of the page to navigate to.
            params: Optional URL parameters to pass to the page.
            add_to_history: Whether to add this navigation to history.

        Returns:
            True if navigation successful, False otherwise.

        Raises:
            ValueError: If page does not exist.

        Example:
            >>> # Simple navigation
            >>> nav.navigate("dashboard")
            >>>
            >>> # Navigation with parameters
            >>> nav.navigate("user_profile", params={"user_id": "123"})
        """
        if page_name not in self._pages:
            raise ValueError(f"Page '{page_name}' is not registered")

        # Store previous page
        current_page = st.session_state[self._SESSION_CURRENT_PAGE]
        st.session_state[self._SESSION_PREVIOUS_PAGE] = current_page

        # Update current page and params
        st.session_state[self._SESSION_CURRENT_PAGE] = page_name
        st.session_state[self._SESSION_PARAMS] = params or {}

        # Add to navigation history
        if add_to_history:
            history = st.session_state[self._SESSION_HISTORY]
            history.append({"page": page_name, "params": params or {}})
            st.session_state[self._SESSION_HISTORY] = history

        # Update breadcrumbs
        self._update_breadcrumbs(page_name)

        return True

    def _update_breadcrumbs(self, page_name: str) -> None:
        """Update breadcrumb trail for current navigation.

        Args:
            page_name: Current page name.
        """
        breadcrumbs = []
        current = page_name

        # Build breadcrumb trail by following parent links
        while current:
            page_config = self._pages.get(current)
            if not page_config:
                break

            breadcrumbs.insert(0, {"name": current, "title": page_config.title})
            current = page_config.parent

        st.session_state[self._SESSION_BREADCRUMBS] = breadcrumbs

    def route_handler(self) -> None:
        """Handle routing and render the current page.

        This is the main routing function that should be called in the Streamlit
        app to handle navigation and render the appropriate page content.

        The function:
        1. Checks URL parameters for deep linking
        2. Validates user authorization
        3. Renders the current page content
        4. Handles errors gracefully

        Example:
            >>> # In your main Streamlit app
            >>> nav = NavigationManager()
            >>>
            >>> # Register pages...
            >>>
            >>> # Handle routing
            >>> nav.route_handler()
        """
        # Check for URL parameters (deep linking)
        self._handle_deep_linking()

        # Get current page
        current_page_name = st.session_state.get(
            self._SESSION_CURRENT_PAGE, self._default_page
        )

        # Ensure page exists
        if current_page_name not in self._pages:
            current_page_name = self._default_page
            st.session_state[self._SESSION_CURRENT_PAGE] = current_page_name

        page_config = self._pages.get(current_page_name)
        if not page_config:
            st.error(f"Page '{current_page_name}' not found")
            return

        # Check authorization
        if not self._check_authorization(page_config):
            st.warning("🔒 You do not have permission to access this page")
            if self._default_page != current_page_name:
                self.navigate(self._default_page)
            return

        # Render page content
        try:
            page_config.render_func()
        except Exception as e:
            if self._error_handler:
                self._error_handler(e, current_page_name)
            else:
                st.error(f"Error rendering page '{page_config.title}': {str(e)}")
                st.exception(e)

    def _check_authorization(self, page_config: PageConfig) -> bool:
        """Check if user is authorized to access a page.

        Args:
            page_config: Configuration of the page to check.

        Returns:
            True if authorized, False otherwise.
        """
        # Custom authorization check takes precedence
        if page_config.custom_auth_check:
            return page_config.custom_auth_check()

        # Public pages are always accessible
        if page_config.access_level == AccessLevel.PUBLIC:
            return True

        # Use global auth callback if provided
        if self._auth_callback:
            return self._auth_callback()

        # Default: allow access (can be changed to deny by default)
        return True

    def breadcrumbs(
        self,
        separator: str = " / ",
        show_icons: bool = True,
        clickable: bool = True,
        container: Optional[Any] = None,
    ) -> List[Dict[str, str]]:
        """Display and return navigation breadcrumbs.

        Args:
            separator: Separator string between breadcrumb items.
            show_icons: Whether to show page icons in breadcrumbs.
            clickable: Whether breadcrumbs should be clickable links.
            container: Optional Streamlit container to render breadcrumbs in.

        Returns:
            List of breadcrumb dictionaries with 'name' and 'title' keys.

        Example:
            >>> # Display breadcrumbs at top of page
            >>> nav.breadcrumbs(separator=" > ", clickable=True)
            >>>
            >>> # Get breadcrumbs without displaying
            >>> trail = nav.breadcrumbs(container=None)
        """
        breadcrumbs = st.session_state.get(self._SESSION_BREADCRUMBS, [])

        # Render breadcrumbs if container provided (or use default)
        if container is not None or container is None:
            # Use provided container or create default
            ctx = container if container is not None else st

            if breadcrumbs:
                breadcrumb_parts = []

                for i, crumb in enumerate(breadcrumbs):
                    page_config = self._pages.get(crumb["name"])
                    if not page_config:
                        continue

                    # Build breadcrumb text
                    icon = f"{page_config.icon} " if show_icons else ""
                    text = f"{icon}{crumb['title']}"

                    # Create clickable link or plain text
                    if clickable and i < len(breadcrumbs) - 1:
                        breadcrumb_parts.append(f"[{text}](#{crumb['name']})")
                    else:
                        breadcrumb_parts.append(text)

                # Display breadcrumbs
                breadcrumb_html = separator.join(breadcrumb_parts)

                if hasattr(ctx, 'markdown'):
                    ctx.markdown(f"**Navigation:** {breadcrumb_html}")

        return breadcrumbs

    def deep_linking(self, use_query_params: bool = True) -> Route:
        """Enable deep linking support through URL parameters.

        This method extracts routing information from URL query parameters,
        allowing users to bookmark or share direct links to specific pages
        and states within the application.

        Args:
            use_query_params: Whether to use Streamlit's query_params API.

        Returns:
            Route object with extracted page and parameters.

        Example:
            >>> # URL: http://localhost:8501/?page=dashboard&user_id=123
            >>> route = nav.deep_linking()
            >>> # route.page_name = "dashboard"
            >>> # route.params = {"user_id": "123"}
        """
        route = Route(page_name=self._default_page)

        try:
            # Get query parameters from Streamlit
            query_params = st.query_params

            # Extract page name
            if "page" in query_params:
                page_name = query_params["page"]
                if isinstance(page_name, list):
                    page_name = page_name[0]
                route.page_name = page_name

            # Extract other parameters
            params = {}
            for key, value in query_params.items():
                if key != "page":
                    # Handle list values
                    if isinstance(value, list):
                        params[key] = value[0] if len(value) == 1 else value
                    else:
                        params[key] = value

            route.params = params
            route.query_string = urlencode(query_params)

        except Exception as e:
            # Fallback: use default page
            if self._error_handler:
                self._error_handler(e, "deep_linking")

        return route

    def _handle_deep_linking(self) -> None:
        """Handle deep linking by checking URL parameters."""
        route = self.deep_linking()

        # Navigate if URL specifies a different page
        if route.page_name != st.session_state[self._SESSION_CURRENT_PAGE]:
            if route.page_name in self._pages:
                self.navigate(route.page_name, route.params, add_to_history=False)

    def set_query_params(self, params: Dict[str, Any]) -> None:
        """Set URL query parameters for current page.

        Args:
            params: Dictionary of parameters to set in URL.

        Example:
            >>> nav.set_query_params({"filter": "active", "sort": "date"})
        """
        current_page = st.session_state[self._SESSION_CURRENT_PAGE]
        all_params = {"page": current_page, **params}
        st.query_params.update(all_params)

    def get_current_page(self) -> str:
        """Get the name of the current page.

        Returns:
            Current page name.
        """
        return st.session_state.get(self._SESSION_CURRENT_PAGE, self._default_page)

    def get_current_params(self) -> Dict[str, Any]:
        """Get URL parameters for the current page.

        Returns:
            Dictionary of current URL parameters.
        """
        return st.session_state.get(self._SESSION_PARAMS, {}).copy()

    def get_previous_page(self) -> Optional[str]:
        """Get the name of the previous page.

        Returns:
            Previous page name or None if no previous page.
        """
        return st.session_state.get(self._SESSION_PREVIOUS_PAGE)

    def get_navigation_history(self) -> List[Dict[str, Any]]:
        """Get navigation history.

        Returns:
            List of navigation history entries with page and params.
        """
        return st.session_state.get(self._SESSION_HISTORY, []).copy()

    def clear_history(self) -> None:
        """Clear navigation history."""
        st.session_state[self._SESSION_HISTORY] = []

    def go_back(self) -> bool:
        """Navigate to the previous page in history.

        Returns:
            True if navigation successful, False if no history.
        """
        history = st.session_state.get(self._SESSION_HISTORY, [])
        if len(history) > 1:
            # Remove current page
            history.pop()
            # Get previous page
            previous = history[-1]
            # Navigate without adding to history
            return self.navigate(
                previous["page"], previous.get("params"), add_to_history=False
            )
        return False

    def render_sidebar_navigation(
        self,
        show_icons: bool = True,
        group_by_parent: bool = True,
        show_search: bool = True,
    ) -> None:
        """Render navigation menu in Streamlit sidebar.

        Args:
            show_icons: Whether to show page icons.
            group_by_parent: Whether to group pages by parent.
            show_search: Whether to show page search box.

        Example:
            >>> # In your Streamlit app
            >>> with st.sidebar:
            >>>     nav.render_sidebar_navigation()
        """
        st.sidebar.title("🧭 Navigation")

        # Optional search
        if show_search and len(self._pages) > 5:
            search_term = st.sidebar.text_input("🔍 Search pages", "")
        else:
            search_term = ""

        # Get pages for sidebar
        sidebar_pages = [
            p for p in self._pages.values() if p.show_in_sidebar
        ]

        # Filter by search term
        if search_term:
            search_term_lower = search_term.lower()
            sidebar_pages = [
                p for p in sidebar_pages
                if search_term_lower in p.title.lower()
                or search_term_lower in p.description.lower()
                or any(search_term_lower in kw.lower() for kw in p.keywords)
            ]

        # Sort pages
        sidebar_pages.sort(key=lambda p: (p.order, p.title))

        # Group by parent if requested
        if group_by_parent:
            # Root pages (no parent)
            root_pages = [p for p in sidebar_pages if not p.parent]
            child_pages = [p for p in sidebar_pages if p.parent]

            for page in root_pages:
                self._render_sidebar_page(page, show_icons)

                # Render children
                children = [p for p in child_pages if p.parent == page.name]
                for child in children:
                    st.sidebar.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ ", unsafe_allow_html=True)
                    self._render_sidebar_page(child, show_icons)
        else:
            for page in sidebar_pages:
                self._render_sidebar_page(page, show_icons)

    def _render_sidebar_page(self, page: PageConfig, show_icons: bool) -> None:
        """Render a single page button in sidebar.

        Args:
            page: Page configuration to render.
            show_icons: Whether to show the page icon.
        """
        icon = f"{page.icon} " if show_icons else ""
        label = f"{icon}{page.title}"

        current_page = st.session_state[self._SESSION_CURRENT_PAGE]
        is_current = current_page == page.name

        if st.sidebar.button(
            label,
            key=f"nav_btn_{page.name}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            self.navigate(page.name)
            st.rerun()
