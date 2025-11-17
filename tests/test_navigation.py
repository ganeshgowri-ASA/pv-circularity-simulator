"""Comprehensive tests for NavigationManager.

This module contains unit and integration tests for the navigation and routing system.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Mock Streamlit before importing navigation
import sys
sys.modules['streamlit'] = MagicMock()

from src.navigation.navigation_manager import (
    NavigationManager,
    PageConfig,
    Route,
    AccessLevel,
)


class TestPageConfig:
    """Tests for PageConfig dataclass."""

    def test_page_config_creation(self):
        """Test creating a basic PageConfig."""
        def dummy_func():
            pass

        config = PageConfig(
            name="test_page",
            title="Test Page",
            icon="🧪",
            render_func=dummy_func,
        )

        assert config.name == "test_page"
        assert config.title == "Test Page"
        assert config.icon == "🧪"
        assert config.render_func == dummy_func
        assert config.access_level == AccessLevel.PUBLIC
        assert config.show_in_sidebar is True

    def test_page_config_with_all_params(self):
        """Test PageConfig with all parameters."""
        def dummy_func():
            pass

        def auth_check():
            return True

        config = PageConfig(
            name="test",
            title="Test",
            icon="🧪",
            render_func=dummy_func,
            description="Test description",
            access_level=AccessLevel.AUTHENTICATED,
            parent="home",
            show_in_sidebar=False,
            url_params={"id": "123"},
            keywords=["test", "sample"],
            order=5,
            custom_auth_check=auth_check,
        )

        assert config.description == "Test description"
        assert config.access_level == AccessLevel.AUTHENTICATED
        assert config.parent == "home"
        assert config.show_in_sidebar is False
        assert config.url_params == {"id": "123"}
        assert config.keywords == ["test", "sample"]
        assert config.order == 5
        assert config.custom_auth_check == auth_check

    def test_page_config_validation_empty_name(self):
        """Test that empty page name raises ValueError."""
        def dummy_func():
            pass

        with pytest.raises(ValueError, match="Page name cannot be empty"):
            PageConfig(name="", title="Test", icon="🧪", render_func=dummy_func)

    def test_page_config_validation_invalid_render_func(self):
        """Test that non-callable render_func raises ValueError."""
        with pytest.raises(ValueError, match="render_func must be callable"):
            PageConfig(name="test", title="Test", icon="🧪", render_func="not_callable")


class TestRoute:
    """Tests for Route dataclass."""

    def test_route_creation(self):
        """Test creating a basic Route."""
        route = Route(page_name="test_page")

        assert route.page_name == "test_page"
        assert route.params == {}
        assert route.query_string == ""

    def test_route_with_params(self):
        """Test Route with parameters."""
        route = Route(
            page_name="test_page",
            params={"id": "123", "filter": "active"},
            query_string="id=123&filter=active",
        )

        assert route.page_name == "test_page"
        assert route.params == {"id": "123", "filter": "active"}
        assert route.query_string == "id=123&filter=active"

    def test_route_to_url(self):
        """Test converting Route to URL query string."""
        route = Route(page_name="test_page", params={"id": "123", "filter": "active"})

        url = route.to_url()

        # URL encoding order might vary, so check components
        assert "page=test_page" in url
        assert "id=123" in url
        assert "filter=active" in url


@patch('src.navigation.navigation_manager.st')
class TestNavigationManager:
    """Tests for NavigationManager class."""

    def test_initialization(self, mock_st):
        """Test NavigationManager initialization."""
        mock_st.session_state = {}
        nav = NavigationManager(default_page="home")

        assert nav._default_page == "home"
        assert nav._auth_callback is None
        assert nav._error_handler is None
        assert len(nav._pages) == 0

    def test_initialization_with_callbacks(self, mock_st):
        """Test NavigationManager with auth and error callbacks."""
        mock_st.session_state = {}

        def auth_callback():
            return True

        def error_handler(e, page):
            pass

        nav = NavigationManager(
            default_page="home",
            auth_callback=auth_callback,
            error_handler=error_handler,
        )

        assert nav._auth_callback == auth_callback
        assert nav._error_handler == error_handler

    def test_page_registration_decorator(self, mock_st):
        """Test page registration using decorator."""
        mock_st.session_state = {}
        nav = NavigationManager()

        @nav.page_registry(name="test", title="Test Page", icon="🧪")
        def test_page():
            pass

        assert "test" in nav._pages
        config = nav._pages["test"]
        assert config.name == "test"
        assert config.title == "Test Page"
        assert config.icon == "🧪"
        assert config.render_func == test_page

    def test_page_registration_programmatic(self, mock_st):
        """Test programmatic page registration."""
        mock_st.session_state = {}
        nav = NavigationManager()

        def test_page():
            pass

        config = PageConfig(
            name="test",
            title="Test Page",
            icon="🧪",
            render_func=test_page,
        )

        nav.register_page(config)

        assert "test" in nav._pages
        assert nav._pages["test"] == config

    def test_duplicate_page_registration(self, mock_st):
        """Test that registering duplicate page raises error."""
        mock_st.session_state = {}
        nav = NavigationManager()

        @nav.page_registry(name="test", title="Test", icon="🧪")
        def test_page1():
            pass

        with pytest.raises(ValueError, match="already registered"):

            @nav.page_registry(name="test", title="Test 2", icon="🧪")
            def test_page2():
                pass

    def test_get_registered_pages(self, mock_st):
        """Test getting all registered pages."""
        mock_st.session_state = {}
        nav = NavigationManager()

        @nav.page_registry(name="page1", title="Page 1", icon="1️⃣")
        def page1():
            pass

        @nav.page_registry(name="page2", title="Page 2", icon="2️⃣")
        def page2():
            pass

        pages = nav.get_registered_pages()

        assert len(pages) == 2
        assert "page1" in pages
        assert "page2" in pages

    def test_get_page_config(self, mock_st):
        """Test getting specific page config."""
        mock_st.session_state = {}
        nav = NavigationManager()

        @nav.page_registry(name="test", title="Test", icon="🧪")
        def test_page():
            pass

        config = nav.get_page_config("test")
        assert config is not None
        assert config.name == "test"

        missing_config = nav.get_page_config("nonexistent")
        assert missing_config is None

    def test_navigate(self, mock_st):
        """Test navigation between pages."""
        mock_st.session_state = {
            "nav_current_page": "home",
            "nav_previous_page": None,
            "nav_params": {},
            "nav_history": [],
            "nav_breadcrumbs": [],
        }

        nav = NavigationManager(default_page="home")

        @nav.page_registry(name="home", title="Home", icon="🏠")
        def home():
            pass

        @nav.page_registry(name="test", title="Test", icon="🧪")
        def test():
            pass

        # Navigate to test page
        result = nav.navigate("test", params={"id": "123"})

        assert result is True
        assert mock_st.session_state["nav_current_page"] == "test"
        assert mock_st.session_state["nav_previous_page"] == "home"
        assert mock_st.session_state["nav_params"] == {"id": "123"}

    def test_navigate_invalid_page(self, mock_st):
        """Test navigation to non-existent page raises error."""
        mock_st.session_state = {}
        nav = NavigationManager()

        with pytest.raises(ValueError, match="not registered"):
            nav.navigate("nonexistent")

    def test_get_current_page(self, mock_st):
        """Test getting current page."""
        mock_st.session_state = {"nav_current_page": "test"}
        nav = NavigationManager(default_page="home")

        assert nav.get_current_page() == "test"

    def test_get_current_params(self, mock_st):
        """Test getting current parameters."""
        mock_st.session_state = {"nav_params": {"id": "123"}}
        nav = NavigationManager()

        params = nav.get_current_params()
        assert params == {"id": "123"}

    def test_get_previous_page(self, mock_st):
        """Test getting previous page."""
        mock_st.session_state = {"nav_previous_page": "home"}
        nav = NavigationManager()

        assert nav.get_previous_page() == "home"

    def test_navigation_history(self, mock_st):
        """Test navigation history tracking."""
        mock_st.session_state = {
            "nav_current_page": "home",
            "nav_previous_page": None,
            "nav_params": {},
            "nav_history": [],
            "nav_breadcrumbs": [],
        }

        nav = NavigationManager()

        @nav.page_registry(name="home", title="Home", icon="🏠")
        def home():
            pass

        @nav.page_registry(name="page1", title="Page 1", icon="1️⃣")
        def page1():
            pass

        @nav.page_registry(name="page2", title="Page 2", icon="2️⃣")
        def page2():
            pass

        # Navigate through pages
        nav.navigate("page1")
        nav.navigate("page2")

        history = nav.get_navigation_history()
        assert len(history) == 2
        assert history[0]["page"] == "page1"
        assert history[1]["page"] == "page2"

    def test_clear_history(self, mock_st):
        """Test clearing navigation history."""
        mock_st.session_state = {
            "nav_history": [{"page": "page1"}, {"page": "page2"}]
        }

        nav = NavigationManager()
        nav.clear_history()

        assert mock_st.session_state["nav_history"] == []

    def test_go_back(self, mock_st):
        """Test going back in navigation history."""
        mock_st.session_state = {
            "nav_current_page": "page2",
            "nav_previous_page": "page1",
            "nav_params": {},
            "nav_history": [
                {"page": "home", "params": {}},
                {"page": "page1", "params": {}},
                {"page": "page2", "params": {}},
            ],
            "nav_breadcrumbs": [],
        }

        nav = NavigationManager()

        @nav.page_registry(name="home", title="Home", icon="🏠")
        def home():
            pass

        @nav.page_registry(name="page1", title="Page 1", icon="1️⃣")
        def page1():
            pass

        @nav.page_registry(name="page2", title="Page 2", icon="2️⃣")
        def page2():
            pass

        result = nav.go_back()

        assert result is True
        assert mock_st.session_state["nav_current_page"] == "page1"

    def test_breadcrumbs_generation(self, mock_st):
        """Test breadcrumb generation."""
        mock_st.session_state = {
            "nav_current_page": "child",
            "nav_breadcrumbs": [
                {"name": "home", "title": "Home"},
                {"name": "parent", "title": "Parent"},
                {"name": "child", "title": "Child"},
            ],
        }

        nav = NavigationManager()

        @nav.page_registry(name="home", title="Home", icon="🏠")
        def home():
            pass

        @nav.page_registry(name="parent", title="Parent", icon="📁", parent="home")
        def parent():
            pass

        @nav.page_registry(name="child", title="Child", icon="📄", parent="parent")
        def child():
            pass

        breadcrumbs = nav.breadcrumbs(container=None)

        assert len(breadcrumbs) == 3
        assert breadcrumbs[0]["name"] == "home"
        assert breadcrumbs[1]["name"] == "parent"
        assert breadcrumbs[2]["name"] == "child"

    def test_authorization_public_page(self, mock_st):
        """Test authorization for public pages."""
        mock_st.session_state = {}
        nav = NavigationManager()

        def test_page():
            pass

        config = PageConfig(
            name="test",
            title="Test",
            icon="🧪",
            render_func=test_page,
            access_level=AccessLevel.PUBLIC,
        )

        assert nav._check_authorization(config) is True

    def test_authorization_custom_check(self, mock_st):
        """Test authorization with custom check function."""
        mock_st.session_state = {}
        nav = NavigationManager()

        def custom_auth():
            return False

        def test_page():
            pass

        config = PageConfig(
            name="test",
            title="Test",
            icon="🧪",
            render_func=test_page,
            custom_auth_check=custom_auth,
        )

        assert nav._check_authorization(config) is False

    def test_deep_linking_route_extraction(self, mock_st):
        """Test deep linking route extraction from URL params."""
        mock_st.session_state = {}
        mock_st.query_params = {
            "page": "test_page",
            "id": "123",
            "filter": "active",
        }

        nav = NavigationManager()
        route = nav.deep_linking()

        assert route.page_name == "test_page"
        assert route.params.get("id") == "123"
        assert route.params.get("filter") == "active"

    def test_set_query_params(self, mock_st):
        """Test setting URL query parameters."""
        mock_st.session_state = {"nav_current_page": "test"}
        mock_st.query_params = MagicMock()

        nav = NavigationManager()
        nav.set_query_params({"id": "123", "filter": "active"})

        mock_st.query_params.update.assert_called_once_with(
            {"page": "test", "id": "123", "filter": "active"}
        )


class TestAccessLevel:
    """Tests for AccessLevel enum."""

    def test_access_levels(self):
        """Test all access level values."""
        assert AccessLevel.PUBLIC.value == "public"
        assert AccessLevel.AUTHENTICATED.value == "authenticated"
        assert AccessLevel.ADMIN.value == "admin"
        assert AccessLevel.CUSTOM.value == "custom"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
