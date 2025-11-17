"""
Tests for theme management functionality.
"""

import pytest
import plotly.graph_objects as go
from pv_circularity.visualization.themes import (
    ThemeManager,
    ColorPalette,
    custom_themes,
    get_theme_manager,
)


class TestColorPalette:
    """Test suite for ColorPalette dataclass."""

    def test_color_palette_creation(self) -> None:
        """Test creating a color palette."""
        palette = ColorPalette(
            primary="#FF0000",
            secondary="#00FF00",
            accent="#0000FF"
        )

        assert palette.primary == "#FF0000"
        assert palette.secondary == "#00FF00"
        assert palette.accent == "#0000FF"
        assert palette.background == "#FFFFFF"  # Default

    def test_color_palette_with_lists(self) -> None:
        """Test creating a palette with color lists."""
        categorical = ["#FF0000", "#00FF00", "#0000FF"]
        sequential = ["#FFCCCC", "#FF9999", "#FF0000"]

        palette = ColorPalette(
            primary="#FF0000",
            secondary="#00FF00",
            accent="#0000FF",
            categorical=categorical,
            sequential=sequential
        )

        assert palette.categorical == categorical
        assert palette.sequential == sequential


class TestThemeManager:
    """Test suite for ThemeManager class."""

    def test_initialization(self) -> None:
        """Test theme manager initialization."""
        manager = ThemeManager()
        assert manager._current_theme == "default"
        assert len(manager._themes) > 0

    def test_predefined_themes_exist(self) -> None:
        """Test that all predefined themes are available."""
        manager = ThemeManager()
        expected_themes = [
            'default', 'solar', 'circularity',
            'performance', 'technical', 'dark'
        ]

        for theme in expected_themes:
            assert theme in manager._themes

    def test_get_theme(self) -> None:
        """Test retrieving a theme."""
        manager = ThemeManager()
        solar_theme = manager.get_theme('solar')

        assert isinstance(solar_theme, ColorPalette)
        assert solar_theme.primary == "#FF6B35"

    def test_get_theme_invalid(self) -> None:
        """Test getting non-existent theme raises error."""
        manager = ThemeManager()

        with pytest.raises(ValueError, match="Theme 'invalid' not found"):
            manager.get_theme('invalid')

    def test_list_themes(self) -> None:
        """Test listing all themes."""
        manager = ThemeManager()
        themes = manager.list_themes()

        assert isinstance(themes, list)
        assert len(themes) >= 6
        assert 'solar' in themes

    def test_set_theme(self) -> None:
        """Test setting active theme."""
        manager = ThemeManager()
        manager.set_theme('solar')

        assert manager._current_theme == 'solar'

    def test_set_theme_invalid(self) -> None:
        """Test setting invalid theme raises error."""
        manager = ThemeManager()

        with pytest.raises(ValueError, match="Theme 'invalid' not found"):
            manager.set_theme('invalid')

    def test_get_current_theme(self) -> None:
        """Test getting current theme."""
        manager = ThemeManager()
        manager.set_theme('circularity')

        current = manager.get_current_theme()

        assert isinstance(current, ColorPalette)
        assert current.primary == "#2ECC71"

    def test_create_custom_theme(self) -> None:
        """Test creating a custom theme."""
        manager = ThemeManager()

        custom = manager.create_custom_theme(
            name='test_theme',
            primary='#123456',
            secondary='#654321',
            accent='#ABCDEF'
        )

        assert isinstance(custom, ColorPalette)
        assert custom.primary == '#123456'
        assert 'test_theme' in manager._themes

    def test_create_custom_theme_with_all_params(self) -> None:
        """Test creating custom theme with all parameters."""
        manager = ThemeManager()

        categorical = ["#111111", "#222222", "#333333"]
        sequential = ["#AAAAAA", "#BBBBBB", "#CCCCCC"]
        diverging = ["#111111", "#FFFFFF", "#999999"]

        custom = manager.create_custom_theme(
            name='full_custom',
            primary='#FF0000',
            secondary='#00FF00',
            accent='#0000FF',
            background='#F0F0F0',
            text='#333333',
            grid='#DDDDDD',
            categorical=categorical,
            sequential=sequential,
            diverging=diverging
        )

        assert custom.background == '#F0F0F0'
        assert custom.categorical == categorical
        assert custom.sequential == sequential

    def test_apply_theme_to_plotly(self) -> None:
        """Test applying theme to Plotly figure."""
        manager = ThemeManager()
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])

        styled_fig = manager.apply_theme_to_plotly(fig, 'solar')

        assert isinstance(styled_fig, go.Figure)
        assert styled_fig.layout.template is not None

    def test_get_altair_theme(self) -> None:
        """Test getting Altair theme configuration."""
        manager = ThemeManager()
        altair_config = manager.get_altair_theme('solar')

        assert isinstance(altair_config, dict)
        assert 'config' in altair_config
        assert 'background' in altair_config['config']

    def test_theme_colors_correctness(self) -> None:
        """Test that theme colors are correctly defined."""
        manager = ThemeManager()

        # Solar theme colors
        solar = manager.get_theme('solar')
        assert solar.primary == "#FF6B35"  # Solar orange
        assert solar.secondary == "#004E89"  # Deep blue

        # Circularity theme colors
        circ = manager.get_theme('circularity')
        assert circ.primary == "#2ECC71"  # Green for sustainability

        # Dark theme colors
        dark = manager.get_theme('dark')
        assert dark.background == "#1E1E1E"  # Dark background


class TestGlobalFunctions:
    """Test suite for global helper functions."""

    def test_get_theme_manager(self) -> None:
        """Test getting global theme manager."""
        manager = get_theme_manager()
        assert isinstance(manager, ThemeManager)

    def test_custom_themes_function(self) -> None:
        """Test custom_themes() convenience function."""
        themes = custom_themes()
        assert isinstance(themes, ThemeManager)

    def test_global_instance_consistency(self) -> None:
        """Test that global instance is consistent."""
        manager1 = get_theme_manager()
        manager2 = custom_themes()

        # Both should reference the same instance
        assert manager1 is manager2


class TestThemeIntegration:
    """Integration tests for theme functionality."""

    def test_theme_switching(self) -> None:
        """Test switching between themes."""
        manager = ThemeManager()

        # Start with default
        assert manager._current_theme == 'default'

        # Switch to solar
        manager.set_theme('solar')
        assert manager._current_theme == 'solar'

        # Switch to dark
        manager.set_theme('dark')
        assert manager._current_theme == 'dark'

        # Back to default
        manager.set_theme('default')
        assert manager._current_theme == 'default'

    def test_theme_application_workflow(self) -> None:
        """Test complete theme application workflow."""
        manager = ThemeManager()

        # Create a figure
        fig = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3])])

        # Apply different themes
        for theme_name in ['solar', 'circularity', 'performance']:
            styled_fig = manager.apply_theme_to_plotly(fig, theme_name)
            assert isinstance(styled_fig, go.Figure)
            assert styled_fig.layout.template is not None
