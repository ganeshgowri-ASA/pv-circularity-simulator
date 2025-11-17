"""Navigation and Routing System for PV Circularity Simulator.

This module provides a comprehensive navigation and routing system for Streamlit
applications, including page registration, route handling, breadcrumbs, and deep linking.
"""

from .navigation_manager import NavigationManager, PageConfig, Route

__all__ = ["NavigationManager", "PageConfig", "Route"]
