"""
Color utilities for dashboard components.

This module provides helper functions for generating colors based on status,
values, and gradients for visual representations in dashboard components.
"""

from typing import Tuple, Optional
from ..models.metrics import MetricStatus, NotificationLevel


def get_status_color(status: MetricStatus) -> str:
    """
    Get color hex code based on metric status.

    Args:
        status: MetricStatus enum value

    Returns:
        Hex color code string

    Examples:
        >>> get_status_color(MetricStatus.EXCELLENT)
        '#10b981'
    """
    color_map = {
        MetricStatus.EXCELLENT: "#10b981",  # Green
        MetricStatus.GOOD: "#3b82f6",      # Blue
        MetricStatus.FAIR: "#f59e0b",      # Amber
        MetricStatus.POOR: "#ef4444",      # Red
        MetricStatus.CRITICAL: "#991b1b",  # Dark Red
    }
    return color_map.get(status, "#6b7280")  # Default gray


def get_notification_color(level: NotificationLevel) -> str:
    """
    Get color hex code based on notification level.

    Args:
        level: NotificationLevel enum value

    Returns:
        Hex color code string

    Examples:
        >>> get_notification_color(NotificationLevel.SUCCESS)
        '#10b981'
    """
    color_map = {
        NotificationLevel.INFO: "#3b82f6",      # Blue
        NotificationLevel.SUCCESS: "#10b981",   # Green
        NotificationLevel.WARNING: "#f59e0b",   # Amber
        NotificationLevel.ERROR: "#ef4444",     # Red
        NotificationLevel.CRITICAL: "#991b1b",  # Dark Red
    }
    return color_map.get(level, "#6b7280")  # Default gray


def get_gradient_color(
    value: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
    color_start: str = "#ef4444",
    color_end: str = "#10b981"
) -> str:
    """
    Get interpolated color based on value within a range.

    Args:
        value: Current value
        min_value: Minimum value in range
        max_value: Maximum value in range
        color_start: Hex color for minimum value (default red)
        color_end: Hex color for maximum value (default green)

    Returns:
        Interpolated hex color code

    Examples:
        >>> get_gradient_color(50, 0, 100)
        '#7fa950'
    """
    # Normalize value to 0-1 range
    normalized = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))

    # Parse hex colors
    start_rgb = hex_to_rgb(color_start)
    end_rgb = hex_to_rgb(color_end)

    # Interpolate
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * normalized)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * normalized)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * normalized)

    return rgb_to_hex(r, g, b)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color code to RGB tuple.

    Args:
        hex_color: Hex color code (e.g., '#ff0000' or 'ff0000')

    Returns:
        Tuple of (r, g, b) values (0-255)

    Examples:
        >>> hex_to_rgb('#ff0000')
        (255, 0, 0)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color code.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        Hex color code string

    Examples:
        >>> rgb_to_hex(255, 0, 0)
        '#ff0000'
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def get_color_palette(palette_name: str = "default") -> dict:
    """
    Get predefined color palette for dashboard theming.

    Args:
        palette_name: Name of the color palette

    Returns:
        Dictionary of color names to hex codes

    Available palettes:
        - default: Standard dashboard colors
        - circularity: Colors themed for circular economy
        - performance: Colors for performance metrics
    """
    palettes = {
        "default": {
            "primary": "#3b82f6",
            "secondary": "#8b5cf6",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "info": "#06b6d4",
            "light": "#f3f4f6",
            "dark": "#1f2937",
        },
        "circularity": {
            "reduce": "#10b981",    # Green
            "reuse": "#3b82f6",     # Blue
            "recycle": "#8b5cf6",   # Purple
            "primary": "#059669",   # Dark green
            "secondary": "#0d9488", # Teal
            "accent": "#84cc16",    # Lime
        },
        "performance": {
            "excellent": "#10b981",
            "good": "#3b82f6",
            "fair": "#f59e0b",
            "poor": "#ef4444",
            "critical": "#991b1b",
        },
    }
    return palettes.get(palette_name, palettes["default"])


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Lighten a hex color by a factor.

    Args:
        hex_color: Hex color code
        factor: Lightening factor (0-1, higher = lighter)

    Returns:
        Lightened hex color code

    Examples:
        >>> lighten_color('#ff0000', 0.5)
        '#ff7f7f'
    """
    r, g, b = hex_to_rgb(hex_color)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return rgb_to_hex(r, g, b)


def darken_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Darken a hex color by a factor.

    Args:
        hex_color: Hex color code
        factor: Darkening factor (0-1, higher = darker)

    Returns:
        Darkened hex color code

    Examples:
        >>> darken_color('#ff0000', 0.5)
        '#7f0000'
    """
    r, g, b = hex_to_rgb(hex_color)

    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))

    return rgb_to_hex(r, g, b)
