"""
Custom theming system for PV Circularity visualizations.

This module provides a comprehensive theming system with predefined color schemes,
custom themes for different visualization contexts, and utilities for theme
customization and application.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class ColorPalette:
    """
    Represents a color palette for visualizations.

    Attributes:
        primary: Primary color for main data series
        secondary: Secondary color for supporting data
        accent: Accent color for highlights and important elements
        background: Background color for plots
        text: Text color for labels and annotations
        grid: Grid line color
        categorical: List of colors for categorical data
        sequential: List of colors for sequential data (gradients)
        diverging: List of colors for diverging data (centered at zero)
    """
    primary: str
    secondary: str
    accent: str
    background: str = "#FFFFFF"
    text: str = "#2C3E50"
    grid: str = "#E0E0E0"
    categorical: List[str] = field(default_factory=list)
    sequential: List[str] = field(default_factory=list)
    diverging: List[str] = field(default_factory=list)


class ThemeManager:
    """
    Manages visualization themes and provides theme application utilities.

    This class handles theme creation, customization, and application to both
    Plotly and Altair visualizations. It provides predefined themes optimized
    for PV circularity data visualization.

    Examples:
        >>> theme_mgr = ThemeManager()
        >>> theme_mgr.set_theme('solar')
        >>> fig = theme_mgr.apply_theme(fig, 'solar')
    """

    def __init__(self) -> None:
        """Initialize the theme manager with predefined themes."""
        self._themes: Dict[str, ColorPalette] = {}
        self._current_theme: str = "default"
        self._initialize_default_themes()

    def _initialize_default_themes(self) -> None:
        """Initialize all predefined themes."""
        # Solar Energy Theme - Optimized for PV data
        self._themes["solar"] = ColorPalette(
            primary="#FF6B35",  # Solar orange
            secondary="#004E89",  # Deep blue
            accent="#F7B801",  # Golden yellow
            background="#FFFFFF",
            text="#2C3E50",
            grid="#E8F4F8",
            categorical=[
                "#FF6B35", "#004E89", "#F7B801", "#1B998B",
                "#C73E1D", "#6A4C93", "#2D00F7", "#06A77D"
            ],
            sequential=[
                "#FFF4E6", "#FFE0B2", "#FFCC80", "#FFB74D",
                "#FFA726", "#FF9800", "#FB8C00", "#F57C00"
            ],
            diverging=[
                "#004E89", "#0077BE", "#4DA8DA", "#B8E0F6",
                "#FFFFFF", "#FFD8B8", "#FFB074", "#FF6B35"
            ]
        )

        # Circularity Theme - For lifecycle and circular economy
        self._themes["circularity"] = ColorPalette(
            primary="#2ECC71",  # Green for sustainability
            secondary="#3498DB",  # Blue for efficiency
            accent="#E74C3C",  # Red for waste/loss
            background="#F8F9FA",
            text="#34495E",
            grid="#DEE2E6",
            categorical=[
                "#2ECC71", "#3498DB", "#E74C3C", "#F39C12",
                "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6"
            ],
            sequential=[
                "#D5F4E6", "#A8E6CE", "#7ED7B5", "#52C99D",
                "#2ECC71", "#27AE60", "#229954", "#1E8449"
            ],
            diverging=[
                "#E74C3C", "#EC7063", "#F1948A", "#F5B7B1",
                "#FFFFFF", "#A9DFBF", "#7DCEA0", "#52BE80", "#2ECC71"
            ]
        )

        # Performance Theme - For monitoring and analytics
        self._themes["performance"] = ColorPalette(
            primary="#6C5CE7",  # Purple for premium feel
            secondary="#00B894",  # Teal for positive metrics
            accent="#FDCB6E",  # Yellow for warnings
            background="#FAFAFA",
            text="#2D3436",
            grid="#DFE6E9",
            categorical=[
                "#6C5CE7", "#00B894", "#FDCB6E", "#FF7675",
                "#74B9FF", "#A29BFE", "#FD79A8", "#55EFC4"
            ],
            sequential=[
                "#E8E4F3", "#D2CDE7", "#BCB5DB", "#A69ECF",
                "#9086C3", "#7A6FB7", "#6C5CE7", "#5F52D1"
            ],
            diverging=[
                "#FF7675", "#FF9999", "#FFBBBB", "#FFDDDD",
                "#FFFFFF", "#C7E8DD", "#8EDCC0", "#55D0A3", "#00B894"
            ]
        )

        # Technical Theme - For engineering and technical analysis
        self._themes["technical"] = ColorPalette(
            primary="#1E272E",  # Dark gray
            secondary="#0ABDE3",  # Bright cyan
            accent="#FD79A8",  # Pink accent
            background="#FFFFFF",
            text="#2F3640",
            grid="#E1E8ED",
            categorical=[
                "#0ABDE3", "#10AC84", "#F79F1F", "#EE5A6F",
                "#5F27CD", "#00D2D3", "#FF9FF3", "#54A0FF"
            ],
            sequential=[
                "#E3F8FC", "#C7F1F9", "#ABEAF6", "#8FE3F3",
                "#73DCF0", "#57D5ED", "#3BCEEA", "#1FC8E7"
            ],
            diverging=[
                "#1E272E", "#485460", "#808E9B", "#D2DAE2",
                "#FFFFFF", "#C7F1F9", "#8FE3F3", "#57D5ED", "#0ABDE3"
            ]
        )

        # Default Theme - Clean and professional
        self._themes["default"] = ColorPalette(
            primary="#1F77B4",  # Matplotlib blue
            secondary="#FF7F0E",  # Matplotlib orange
            accent="#2CA02C",  # Matplotlib green
            background="#FFFFFF",
            text="#333333",
            grid="#CCCCCC",
            categorical=[
                "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
                "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"
            ],
            sequential=[
                "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
                "#4292C6", "#2171B5", "#08519C", "#08306B"
            ],
            diverging=[
                "#D62728", "#F28585", "#FFCCCC", "#FFE6E6",
                "#FFFFFF", "#CCE5FF", "#99CBFF", "#66B2FF", "#1F77B4"
            ]
        )

        # Dark Theme - For dark mode interfaces
        self._themes["dark"] = ColorPalette(
            primary="#00D9FF",  # Cyan
            secondary="#FF6B9D",  # Pink
            accent="#FFD93D",  # Yellow
            background="#1E1E1E",
            text="#E0E0E0",
            grid="#3A3A3A",
            categorical=[
                "#00D9FF", "#FF6B9D", "#6BCF7F", "#FFD93D",
                "#A78BFA", "#FF8C42", "#60E1CB", "#FB88B4"
            ],
            sequential=[
                "#003540", "#00576A", "#007994", "#009BBE",
                "#00BDE8", "#00D9FF", "#33E0FF", "#66E7FF"
            ],
            diverging=[
                "#FF6B9D", "#FF8BAE", "#FFABBF", "#FFCBD0",
                "#2A2A2A", "#5EC9E8", "#3DB9D9", "#1CA9CA", "#00D9FF"
            ]
        )

    def get_theme(self, name: str) -> ColorPalette:
        """
        Get a theme by name.

        Args:
            name: Name of the theme to retrieve

        Returns:
            ColorPalette object for the requested theme

        Raises:
            ValueError: If theme name doesn't exist

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> solar_theme = theme_mgr.get_theme('solar')
        """
        if name not in self._themes:
            available = ", ".join(self._themes.keys())
            raise ValueError(
                f"Theme '{name}' not found. Available themes: {available}"
            )
        return self._themes[name]

    def list_themes(self) -> List[str]:
        """
        List all available theme names.

        Returns:
            List of available theme names

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> themes = theme_mgr.list_themes()
            >>> print(themes)
            ['solar', 'circularity', 'performance', 'technical', 'default', 'dark']
        """
        return list(self._themes.keys())

    def set_theme(self, name: str) -> None:
        """
        Set the current active theme.

        Args:
            name: Name of the theme to activate

        Raises:
            ValueError: If theme name doesn't exist

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> theme_mgr.set_theme('solar')
        """
        if name not in self._themes:
            available = ", ".join(self._themes.keys())
            raise ValueError(
                f"Theme '{name}' not found. Available themes: {available}"
            )
        self._current_theme = name
        self._register_plotly_theme(name)

    def get_current_theme(self) -> ColorPalette:
        """
        Get the currently active theme.

        Returns:
            ColorPalette object for the current theme

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> theme_mgr.set_theme('solar')
            >>> current = theme_mgr.get_current_theme()
        """
        return self._themes[self._current_theme]

    def create_custom_theme(
        self,
        name: str,
        primary: str,
        secondary: str,
        accent: str,
        background: str = "#FFFFFF",
        text: str = "#333333",
        grid: str = "#CCCCCC",
        categorical: Optional[List[str]] = None,
        sequential: Optional[List[str]] = None,
        diverging: Optional[List[str]] = None,
    ) -> ColorPalette:
        """
        Create and register a custom theme.

        Args:
            name: Unique name for the custom theme
            primary: Primary color (hex code)
            secondary: Secondary color (hex code)
            accent: Accent color (hex code)
            background: Background color (default: white)
            text: Text color (default: dark gray)
            grid: Grid line color (default: light gray)
            categorical: List of categorical colors (optional)
            sequential: List of sequential colors (optional)
            diverging: List of diverging colors (optional)

        Returns:
            The created ColorPalette object

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> custom = theme_mgr.create_custom_theme(
            ...     name='corporate',
            ...     primary='#003366',
            ...     secondary='#66B2FF',
            ...     accent='#FFD700'
            ... )
        """
        theme = ColorPalette(
            primary=primary,
            secondary=secondary,
            accent=accent,
            background=background,
            text=text,
            grid=grid,
            categorical=categorical or [primary, secondary, accent],
            sequential=sequential or [primary],
            diverging=diverging or [secondary, background, primary],
        )
        self._themes[name] = theme
        return theme

    def _register_plotly_theme(self, name: str) -> None:
        """
        Register a theme as a Plotly template.

        Args:
            name: Name of the theme to register
        """
        theme = self._themes[name]

        # Create Plotly template
        template = go.layout.Template()

        # Set layout defaults
        template.layout = go.Layout(
            plot_bgcolor=theme.background,
            paper_bgcolor=theme.background,
            font=dict(color=theme.text, family="Arial, sans-serif", size=12),
            title=dict(font=dict(size=16, color=theme.text)),
            xaxis=dict(
                gridcolor=theme.grid,
                linecolor=theme.text,
                tickcolor=theme.text,
                title=dict(font=dict(size=14)),
            ),
            yaxis=dict(
                gridcolor=theme.grid,
                linecolor=theme.text,
                tickcolor=theme.text,
                title=dict(font=dict(size=14)),
            ),
            colorway=theme.categorical,
            colorscale=dict(
                sequential=self._generate_plotly_colorscale(theme.sequential),
                sequentialminus=self._generate_plotly_colorscale(theme.sequential[::-1]),
                diverging=self._generate_plotly_colorscale(theme.diverging),
            ),
        )

        # Register template
        pio.templates[f"pv_{name}"] = template
        pio.templates.default = f"pv_{name}"

    def _generate_plotly_colorscale(self, colors: List[str]) -> List[List[Union[float, str]]]:
        """
        Generate a Plotly-compatible colorscale from a list of colors.

        Args:
            colors: List of hex color codes

        Returns:
            Plotly colorscale format [[0, color1], [1, color2], ...]
        """
        if not colors:
            return [[0, "#FFFFFF"], [1, "#000000"]]

        if len(colors) == 1:
            return [[0, colors[0]], [1, colors[0]]]

        step = 1.0 / (len(colors) - 1)
        return [[i * step, color] for i, color in enumerate(colors)]

    def apply_theme_to_plotly(
        self,
        fig: go.Figure,
        theme_name: Optional[str] = None
    ) -> go.Figure:
        """
        Apply a theme to a Plotly figure.

        Args:
            fig: Plotly figure to style
            theme_name: Name of theme to apply (uses current theme if None)

        Returns:
            Styled Plotly figure

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[4,5,6])])
            >>> styled_fig = theme_mgr.apply_theme_to_plotly(fig, 'solar')
        """
        theme_name = theme_name or self._current_theme
        theme = self.get_theme(theme_name)

        fig.update_layout(
            template=f"pv_{theme_name}",
            plot_bgcolor=theme.background,
            paper_bgcolor=theme.background,
            font=dict(color=theme.text),
        )

        return fig

    def get_altair_theme(self, theme_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get theme configuration for Altair visualizations.

        Args:
            theme_name: Name of theme to use (uses current theme if None)

        Returns:
            Dictionary with Altair theme configuration

        Examples:
            >>> theme_mgr = ThemeManager()
            >>> altair_config = theme_mgr.get_altair_theme('solar')
            >>> import altair as alt
            >>> alt.themes.register('solar', lambda: altair_config)
            >>> alt.themes.enable('solar')
        """
        theme_name = theme_name or self._current_theme
        theme = self.get_theme(theme_name)

        return {
            "config": {
                "background": theme.background,
                "view": {
                    "strokeWidth": 0,
                    "fill": theme.background,
                },
                "axis": {
                    "gridColor": theme.grid,
                    "domainColor": theme.text,
                    "tickColor": theme.text,
                    "labelColor": theme.text,
                    "titleColor": theme.text,
                    "titleFontSize": 14,
                    "labelFontSize": 12,
                },
                "legend": {
                    "labelColor": theme.text,
                    "titleColor": theme.text,
                    "titleFontSize": 13,
                    "labelFontSize": 11,
                },
                "title": {
                    "color": theme.text,
                    "fontSize": 16,
                    "fontWeight": "bold",
                },
                "range": {
                    "category": theme.categorical,
                    "diverging": theme.diverging,
                    "heatmap": theme.sequential,
                },
                "mark": {
                    "color": theme.primary,
                },
            }
        }


# Global theme manager instance
_global_theme_manager = ThemeManager()


def get_theme_manager() -> ThemeManager:
    """
    Get the global theme manager instance.

    Returns:
        Global ThemeManager instance

    Examples:
        >>> from pv_circularity.visualization.themes import get_theme_manager
        >>> theme_mgr = get_theme_manager()
        >>> theme_mgr.set_theme('solar')
    """
    return _global_theme_manager


def custom_themes() -> ThemeManager:
    """
    Access the custom theming system.

    This is a convenience function that returns the global theme manager,
    providing access to all theming capabilities.

    Returns:
        ThemeManager instance for theme management

    Examples:
        >>> from pv_circularity.visualization import custom_themes
        >>> themes = custom_themes()
        >>> themes.set_theme('solar')
        >>> print(themes.list_themes())
    """
    return get_theme_manager()
