"""
Formatting utilities for dashboard components.

This module provides helper functions for formatting numbers, percentages,
currencies, and other values for display in dashboard components.
"""

from typing import Optional


def format_number(
    value: float,
    decimals: int = 2,
    compact: bool = False,
    suffix: str = ""
) -> str:
    """
    Format a number for display with optional compact notation.

    Args:
        value: The number to format
        decimals: Number of decimal places (default 2)
        compact: Use compact notation (K, M, B) for large numbers
        suffix: Optional suffix to append (e.g., ' units')

    Returns:
        Formatted string representation of the number

    Examples:
        >>> format_number(1234.56)
        '1,234.56'
        >>> format_number(1234567, compact=True)
        '1.23M'
        >>> format_number(1500, compact=True, suffix=' kWh')
        '1.50K kWh'
    """
    if compact:
        if abs(value) >= 1_000_000_000:
            formatted = f"{value / 1_000_000_000:.{decimals}f}B"
        elif abs(value) >= 1_000_000:
            formatted = f"{value / 1_000_000:.{decimals}f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value / 1_000:.{decimals}f}K"
        else:
            formatted = f"{value:.{decimals}f}"
    else:
        formatted = f"{value:,.{decimals}f}"

    return f"{formatted}{suffix}"


def format_percentage(
    value: float,
    decimals: int = 1,
    include_sign: bool = False
) -> str:
    """
    Format a percentage value for display.

    Args:
        value: The percentage value (e.g., 45.5 for 45.5%)
        decimals: Number of decimal places (default 1)
        include_sign: Include + sign for positive values

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(45.5)
        '45.5%'
        >>> format_percentage(12.345, decimals=2)
        '12.35%'
        >>> format_percentage(5.5, include_sign=True)
        '+5.5%'
    """
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_currency(
    value: float,
    currency: str = "$",
    decimals: int = 2,
    compact: bool = False
) -> str:
    """
    Format a currency value for display.

    Args:
        value: The currency value
        currency: Currency symbol (default '$')
        decimals: Number of decimal places (default 2)
        compact: Use compact notation for large values

    Returns:
        Formatted currency string

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1500000, compact=True)
        '$1.50M'
        >>> format_currency(99.99, currency='€')
        '€99.99'
    """
    formatted_value = format_number(value, decimals=decimals, compact=compact)
    return f"{currency}{formatted_value}"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(125)
        '2m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = [f"{hours}h"]
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0:
            parts.append(f"{secs}s")
        return " ".join(parts)


def format_file_size(bytes: float) -> str:
    """
    Format a file size in bytes to human-readable format.

    Args:
        bytes: Size in bytes

    Returns:
        Formatted file size string

    Examples:
        >>> format_file_size(1024)
        '1.00 KB'
        >>> format_file_size(1048576)
        '1.00 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated (default '...')

    Returns:
        Truncated text string

    Examples:
        >>> truncate_text("This is a very long text", max_length=10)
        'This is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
