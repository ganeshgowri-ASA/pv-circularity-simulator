"""
B15-S03: Navigation & Routing
Production-ready Streamlit navigation system with multi-page routing.
"""

import streamlit as st
from typing import Dict, List, Optional, Callable
from pathlib import Path

from ..core.data_models import NavigationItem


class NavigationSystem:
    """
    Comprehensive navigation and routing system for Streamlit apps.
    """

    def __init__(self):
        """Initialize navigation system."""
        self.pages: Dict[str, NavigationItem] = {}
        self.page_callbacks: Dict[str, Callable] = {}
        self.current_page = None

    def multi_page_routing(self,
                          pages: List[NavigationItem],
                          callbacks: Dict[str, Callable]) -> None:
        """
        Set up multi-page routing.

        Args:
            pages: List of navigation items
            callbacks: Dictionary mapping page_id to render function
        """
        # Register pages
        for page in pages:
            self.pages[page.page_id] = page

        # Register callbacks
        self.page_callbacks = callbacks

        # Render navigation
        self._render_navigation()

    def menu_structure(self, style: str = "sidebar") -> None:
        """
        Render menu structure.

        Args:
            style: Menu style ('sidebar', 'horizontal', 'tabs')
        """
        if style == "sidebar":
            self._render_sidebar_menu()
        elif style == "horizontal":
            self._render_horizontal_menu()
        elif style == "tabs":
            self._render_tabs_menu()
        else:
            st.error(f"Unknown menu style: {style}")

    def _render_sidebar_menu(self) -> None:
        """Render sidebar menu."""
        st.sidebar.title("🧭 Navigation")

        # Group pages by parent
        root_pages = [p for p in self.pages.values() if p.parent_id is None]
        root_pages.sort(key=lambda x: x.order)

        for page in root_pages:
            if page.enabled:
                icon = page.icon or "📄"
                if st.sidebar.button(f"{icon} {page.label}", key=page.page_id):
                    self.current_page = page.page_id
                    st.session_state['current_page'] = page.page_id

                # Render children if any
                children = [p for p in self.pages.values() if p.parent_id == page.page_id]
                if children:
                    with st.sidebar.expander(f"└─ {page.label} submenu"):
                        for child in sorted(children, key=lambda x: x.order):
                            if child.enabled:
                                child_icon = child.icon or "  •"
                                if st.button(f"{child_icon} {child.label}", key=child.page_id):
                                    self.current_page = child.page_id
                                    st.session_state['current_page'] = child.page_id

    def _render_horizontal_menu(self) -> None:
        """Render horizontal menu."""
        root_pages = [p for p in self.pages.values() if p.parent_id is None and p.enabled]
        root_pages.sort(key=lambda x: x.order)

        cols = st.columns(len(root_pages))

        for idx, page in enumerate(root_pages):
            with cols[idx]:
                icon = page.icon or "📄"
                if st.button(f"{icon} {page.label}", key=f"nav_{page.page_id}"):
                    self.current_page = page.page_id
                    st.session_state['current_page'] = page.page_id

    def _render_tabs_menu(self) -> None:
        """Render tabs menu."""
        root_pages = [p for p in self.pages.values() if p.parent_id is None and p.enabled]
        root_pages.sort(key=lambda x: x.order)

        tab_labels = [f"{p.icon or '📄'} {p.label}" for p in root_pages]
        tabs = st.tabs(tab_labels)

        for idx, page in enumerate(root_pages):
            with tabs[idx]:
                if page.page_id in self.page_callbacks:
                    self.page_callbacks[page.page_id]()

    def _render_navigation(self) -> None:
        """Render main navigation."""
        # Get current page from session state
        if 'current_page' not in st.session_state:
            # Default to first page
            root_pages = [p for p in self.pages.values() if p.parent_id is None and p.enabled]
            if root_pages:
                st.session_state['current_page'] = sorted(root_pages, key=lambda x: x.order)[0].page_id

        current_page_id = st.session_state.get('current_page')

        # Render current page
        if current_page_id and current_page_id in self.page_callbacks:
            self.page_callbacks[current_page_id]()

    def breadcrumb_navigation(self, page_id: str) -> str:
        """
        Generate breadcrumb navigation for a page.

        Args:
            page_id: Current page ID

        Returns:
            Breadcrumb HTML string
        """
        if page_id not in self.pages:
            return ""

        breadcrumbs = []
        current_page = self.pages[page_id]

        # Build breadcrumb trail
        while current_page is not None:
            breadcrumbs.insert(0, current_page.label)
            if current_page.parent_id:
                current_page = self.pages.get(current_page.parent_id)
            else:
                break

        return " / ".join(breadcrumbs)


def create_default_navigation() -> NavigationSystem:
    """
    Create default navigation structure for PV circularity simulator.

    Returns:
        Configured NavigationSystem
    """
    nav = NavigationSystem()

    # Define pages
    pages = [
        NavigationItem(
            label="Home",
            page_id="home",
            icon="🏠",
            order=0,
            enabled=True
        ),
        NavigationItem(
            label="Hybrid Energy",
            page_id="hybrid_energy",
            icon="⚡",
            order=1,
            enabled=True
        ),
        NavigationItem(
            label="Battery Systems",
            page_id="battery",
            icon="🔋",
            parent_id="hybrid_energy",
            order=0,
            enabled=True
        ),
        NavigationItem(
            label="Wind-Solar Hybrid",
            page_id="wind_solar",
            icon="💨",
            parent_id="hybrid_energy",
            order=1,
            enabled=True
        ),
        NavigationItem(
            label="Hydrogen Systems",
            page_id="hydrogen",
            icon="💧",
            parent_id="hybrid_energy",
            order=2,
            enabled=True
        ),
        NavigationItem(
            label="Grid Integration",
            page_id="grid",
            icon="🔌",
            parent_id="hybrid_energy",
            order=3,
            enabled=True
        ),
        NavigationItem(
            label="Financial Analysis",
            page_id="financial",
            icon="💰",
            order=2,
            enabled=True
        ),
        NavigationItem(
            label="LCOE Calculator",
            page_id="lcoe",
            icon="📊",
            parent_id="financial",
            order=0,
            enabled=True
        ),
        NavigationItem(
            label="NPV Analysis",
            page_id="npv",
            icon="📈",
            parent_id="financial",
            order=1,
            enabled=True
        ),
        NavigationItem(
            label="IRR Modeling",
            page_id="irr",
            icon="💹",
            parent_id="financial",
            order=2,
            enabled=True
        ),
        NavigationItem(
            label="Data Export",
            page_id="export",
            icon="💾",
            order=3,
            enabled=True
        ),
    ]

    for page in pages:
        nav.pages[page.page_id] = page

    return nav


__all__ = ["NavigationSystem", "create_default_navigation"]
