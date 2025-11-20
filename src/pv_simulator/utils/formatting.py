"""
Formatting Functions for PV Circularity Simulator.

This module provides comprehensive formatting utilities for numbers, dates,
currencies, and report generation commonly used in photovoltaic system analysis.
"""

from typing import Union, Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal

Number = Union[int, float, Decimal]


# ============================================================================
# Number Formatting
# ============================================================================


def format_number(
    value: Number,
    decimal_places: int = 2,
    thousands_separator: str = ",",
    decimal_separator: str = ".",
) -> str:
    """
    Format a number with specified decimal places and separators.

    Args:
        value: Number to format
        decimal_places: Number of decimal places (default: 2)
        thousands_separator: Separator for thousands (default: ',')
        decimal_separator: Separator for decimals (default: '.')

    Returns:
        Formatted number string

    Examples:
        >>> format_number(1234.5678)
        '1,234.57'
        >>> format_number(1234567.89, 1)
        '1,234,567.9'
        >>> format_number(1234.5, thousands_separator=' ', decimal_separator=',')
        '1 234,50'
    """
    # Round to specified decimal places
    rounded = round(float(value), decimal_places)

    # Split into integer and decimal parts
    if decimal_places > 0:
        integer_part, decimal_part = f"{rounded:.{decimal_places}f}".split(".")
    else:
        integer_part = str(int(rounded))
        decimal_part = ""

    # Add thousands separator
    if thousands_separator:
        integer_part = f"{int(integer_part):,}".replace(",", thousands_separator)

    # Combine parts
    if decimal_part:
        return f"{integer_part}{decimal_separator}{decimal_part}"
    else:
        return integer_part


def format_percentage(
    value: Number,
    decimal_places: int = 1,
    multiply_by_100: bool = True,
) -> str:
    """
    Format a number as a percentage.

    Args:
        value: Number to format
        decimal_places: Number of decimal places (default: 1)
        multiply_by_100: If True, multiply by 100 (for decimal inputs like 0.15)

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(0.1567)
        '15.7%'
        >>> format_percentage(85.5, multiply_by_100=False)
        '85.5%'
        >>> format_percentage(0.12345, decimal_places=2)
        '12.35%'
    """
    if multiply_by_100:
        value = float(value) * 100

    return f"{value:.{decimal_places}f}%"


def format_currency(
    value: Number,
    currency_symbol: str = "$",
    decimal_places: int = 2,
    symbol_position: str = "prefix",
) -> str:
    """
    Format a number as currency.

    Args:
        value: Amount to format
        currency_symbol: Currency symbol (default: '$')
        decimal_places: Number of decimal places (default: 2)
        symbol_position: 'prefix' or 'suffix' (default: 'prefix')

    Returns:
        Formatted currency string

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.5, currency_symbol='€', symbol_position='suffix')
        '1,234.50€'
        >>> format_currency(1000000, decimal_places=0)
        '$1,000,000'
    """
    formatted_number = format_number(value, decimal_places)

    if symbol_position == "prefix":
        return f"{currency_symbol}{formatted_number}"
    else:
        return f"{formatted_number}{currency_symbol}"


def format_scientific(value: Number, decimal_places: int = 2) -> str:
    """
    Format a number in scientific notation.

    Args:
        value: Number to format
        decimal_places: Number of decimal places (default: 2)

    Returns:
        Formatted scientific notation string

    Examples:
        >>> format_scientific(1234567.89)
        '1.23e+06'
        >>> format_scientific(0.00012345, decimal_places=3)
        '1.235e-04'
    """
    return f"{float(value):.{decimal_places}e}"


def format_engineering(value: Number, decimal_places: int = 2) -> str:
    """
    Format a number with engineering notation (powers of 1000).

    Args:
        value: Number to format
        decimal_places: Number of decimal places (default: 2)

    Returns:
        Formatted engineering notation string

    Examples:
        >>> format_engineering(1234)
        '1.23k'
        >>> format_engineering(1234567)
        '1.23M'
        >>> format_engineering(0.0123)
        '12.30m'
    """
    prefixes = [
        (1e12, "T"),  # Tera
        (1e9, "G"),  # Giga
        (1e6, "M"),  # Mega
        (1e3, "k"),  # kilo
        (1, ""),  # base
        (1e-3, "m"),  # milli
        (1e-6, "μ"),  # micro
        (1e-9, "n"),  # nano
        (1e-12, "p"),  # pico
    ]

    abs_value = abs(float(value))

    for scale, prefix in prefixes:
        if abs_value >= scale:
            scaled_value = value / scale
            return f"{scaled_value:.{decimal_places}f}{prefix}"

    return f"{value:.{decimal_places}f}"


def format_si_unit(
    value: Number,
    unit: str,
    decimal_places: int = 2,
) -> str:
    """
    Format a number with SI unit prefix.

    Args:
        value: Number to format
        unit: Base unit (e.g., 'W', 'Wh', 'kg')
        decimal_places: Number of decimal places (default: 2)

    Returns:
        Formatted string with SI prefix and unit

    Examples:
        >>> format_si_unit(1234, "W")
        '1.23 kW'
        >>> format_si_unit(5000000, "Wh")
        '5.00 MWh'
        >>> format_si_unit(0.025, "kg")
        '25.00 g'
    """
    formatted_value = format_engineering(value, decimal_places)
    return f"{formatted_value}{unit}"


def format_compact(value: Number) -> str:
    """
    Format a number in compact notation (e.g., 1.2K, 3.4M).

    Args:
        value: Number to format

    Returns:
        Compact formatted string

    Examples:
        >>> format_compact(1234)
        '1.2K'
        >>> format_compact(1234567)
        '1.2M'
        >>> format_compact(123)
        '123'
    """
    abs_value = abs(float(value))

    if abs_value >= 1e9:
        return f"{value / 1e9:.1f}B"
    elif abs_value >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        return str(int(value))


# ============================================================================
# Date and Time Formatting
# ============================================================================


def format_date(
    date_value: Union[datetime, date],
    format_string: str = "%Y-%m-%d",
) -> str:
    """
    Format a date or datetime object.

    Args:
        date_value: Date or datetime to format
        format_string: strftime format string (default: ISO format)

    Returns:
        Formatted date string

    Examples:
        >>> from datetime import date
        >>> format_date(date(2024, 3, 15))
        '2024-03-15'
        >>> format_date(date(2024, 3, 15), "%B %d, %Y")
        'March 15, 2024'
    """
    return date_value.strftime(format_string)


def format_datetime(
    datetime_value: datetime,
    format_string: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Format a datetime object.

    Args:
        datetime_value: Datetime to format
        format_string: strftime format string (default: ISO-like format)

    Returns:
        Formatted datetime string

    Examples:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 3, 15, 14, 30, 45)
        >>> format_datetime(dt)
        '2024-03-15 14:30:45'
        >>> format_datetime(dt, "%B %d, %Y at %I:%M %p")
        'March 15, 2024 at 02:30 PM'
    """
    return datetime_value.strftime(format_string)


def format_duration(seconds: Number) -> str:
    """
    Format a duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(3665)
        '1h 1m 5s'
        >>> format_duration(90)
        '1m 30s'
        >>> format_duration(45)
        '45s'
    """
    total_seconds = int(seconds)

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_timestamp(
    timestamp: Union[int, float, datetime],
    format_string: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Format a Unix timestamp or datetime to string.

    Args:
        timestamp: Unix timestamp (seconds since epoch) or datetime object
        format_string: strftime format string

    Returns:
        Formatted timestamp string

    Examples:
        >>> format_timestamp(1710000000)
        '2024-03-09 15:20:00'
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp

    return dt.strftime(format_string)


# ============================================================================
# Data Structure Formatting
# ============================================================================


def format_table(
    data: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
    column_widths: Optional[Dict[str, int]] = None,
) -> str:
    """
    Format data as an ASCII table.

    Args:
        data: List of dictionaries representing rows
        headers: Optional list of column headers (if None, use dict keys)
        column_widths: Optional dict mapping column names to widths

    Returns:
        Formatted table string

    Examples:
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> print(format_table(data))
        name  | age
        ------|----
        Alice | 30
        Bob   | 25
    """
    if not data:
        return ""

    # Determine headers
    if headers is None:
        headers = list(data[0].keys())

    # Calculate column widths
    if column_widths is None:
        column_widths = {}
        for header in headers:
            max_width = len(header)
            for row in data:
                value_width = len(str(row.get(header, "")))
                max_width = max(max_width, value_width)
            column_widths[header] = max_width

    # Format header row
    header_row = " | ".join(
        header.ljust(column_widths[header]) for header in headers
    )

    # Format separator row
    separator = "-|-".join("-" * column_widths[header] for header in headers)

    # Format data rows
    data_rows = []
    for row in data:
        formatted_row = " | ".join(
            str(row.get(header, "")).ljust(column_widths[header])
            for header in headers
        )
        data_rows.append(formatted_row)

    # Combine all parts
    return "\n".join([header_row, separator] + data_rows)


def format_list(
    items: List[Any],
    bullet: str = "•",
    indent: int = 2,
) -> str:
    """
    Format a list with bullets.

    Args:
        items: List of items to format
        bullet: Bullet character (default: '•')
        indent: Indentation spaces (default: 2)

    Returns:
        Formatted list string

    Examples:
        >>> items = ["First item", "Second item", "Third item"]
        >>> print(format_list(items))
        • First item
        • Second item
        • Third item
    """
    indent_str = " " * indent
    return "\n".join(f"{indent_str}{bullet} {item}" for item in items)


def format_key_value_pairs(
    data: Dict[str, Any],
    separator: str = ": ",
    key_width: Optional[int] = None,
) -> str:
    """
    Format dictionary as key-value pairs.

    Args:
        data: Dictionary to format
        separator: Separator between key and value (default: ': ')
        key_width: Optional fixed width for keys (for alignment)

    Returns:
        Formatted key-value string

    Examples:
        >>> data = {"Name": "Solar Panel", "Power": "300W", "Efficiency": "20%"}
        >>> print(format_key_value_pairs(data))
        Name: Solar Panel
        Power: 300W
        Efficiency: 20%
    """
    if key_width is None:
        key_width = max(len(str(k)) for k in data.keys()) if data else 0

    lines = []
    for key, value in data.items():
        formatted_key = str(key).ljust(key_width)
        lines.append(f"{formatted_key}{separator}{value}")

    return "\n".join(lines)


# ============================================================================
# Report Formatting
# ============================================================================


def format_report_header(
    title: str,
    width: int = 80,
    border_char: str = "=",
) -> str:
    """
    Format a report header with title.

    Args:
        title: Report title
        width: Total width of header (default: 80)
        border_char: Character for border (default: '=')

    Returns:
        Formatted header string

    Examples:
        >>> print(format_report_header("PV System Analysis"))
        ================================================================================
                                    PV System Analysis
        ================================================================================
    """
    border = border_char * width
    centered_title = title.center(width)
    return f"{border}\n{centered_title}\n{border}"


def format_report_section(
    title: str,
    content: str,
    width: int = 80,
) -> str:
    """
    Format a report section with title and content.

    Args:
        title: Section title
        content: Section content
        width: Total width (default: 80)

    Returns:
        Formatted section string

    Examples:
        >>> section = format_report_section("Summary", "This is the summary.")
        >>> print(section)
        Summary
        ───────────────────────────────────────────────────────────────────────────────
        This is the summary.
    """
    separator = "─" * width
    return f"{title}\n{separator}\n{content}"


def format_summary_box(
    title: str,
    items: Dict[str, Any],
    width: int = 60,
) -> str:
    """
    Format a summary box with title and key metrics.

    Args:
        title: Box title
        items: Dictionary of metric names and values
        width: Box width (default: 60)

    Returns:
        Formatted summary box

    Examples:
        >>> metrics = {"Total Energy": "1000 kWh", "Efficiency": "18.5%"}
        >>> print(format_summary_box("System Metrics", metrics))
        ┌────────────────────────────────────────────────────────────┐
        │                      System Metrics                        │
        ├────────────────────────────────────────────────────────────┤
        │ Total Energy : 1000 kWh                                    │
        │ Efficiency   : 18.5%                                       │
        └────────────────────────────────────────────────────────────┘
    """
    lines = []

    # Top border
    lines.append("┌" + "─" * (width - 2) + "┐")

    # Title
    lines.append("│" + title.center(width - 2) + "│")

    # Separator
    lines.append("├" + "─" * (width - 2) + "┤")

    # Items
    if items:
        max_key_length = max(len(str(k)) for k in items.keys())
        for key, value in items.items():
            padded_key = str(key).ljust(max_key_length)
            content = f"{padded_key} : {value}"
            # Ensure content fits within box
            if len(content) > width - 4:
                content = content[: width - 7] + "..."
            lines.append("│ " + content.ljust(width - 3) + "│")

    # Bottom border
    lines.append("└" + "─" * (width - 2) + "┘")

    return "\n".join(lines)


def format_progress_bar(
    current: Number,
    total: Number,
    width: int = 50,
    fill_char: str = "█",
    empty_char: str = "░",
    show_percentage: bool = True,
) -> str:
    """
    Format a progress bar.

    Args:
        current: Current progress value
        total: Total value (100%)
        width: Bar width in characters (default: 50)
        fill_char: Character for filled portion (default: '█')
        empty_char: Character for empty portion (default: '░')
        show_percentage: If True, show percentage (default: True)

    Returns:
        Formatted progress bar string

    Examples:
        >>> format_progress_bar(75, 100)
        '█████████████████████████████████████░░░░░░░░░░░░░ 75.0%'
        >>> format_progress_bar(50, 200, width=20)
        '█████░░░░░░░░░░░░░░░ 25.0%'
    """
    if total == 0:
        percentage = 0
    else:
        percentage = (float(current) / float(total)) * 100

    filled_width = int((percentage / 100) * width)
    empty_width = width - filled_width

    bar = fill_char * filled_width + empty_char * empty_width

    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    else:
        return bar


def truncate_string(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """
    Truncate a string to a maximum length.

    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append to truncated string (default: '...')

    Returns:
        Truncated string

    Examples:
        >>> truncate_string("This is a long string", 10)
        'This is...'
        >>> truncate_string("Short", 10)
        'Short'
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix
