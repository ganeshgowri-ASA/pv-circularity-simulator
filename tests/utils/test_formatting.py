"""
Tests for formatting utilities.
"""

import pytest
from datetime import datetime, date

from src.pv_simulator.utils.formatting import (
    format_number,
    format_percentage,
    format_currency,
    format_scientific,
    format_engineering,
    format_si_unit,
    format_compact,
    format_date,
    format_datetime,
    format_duration,
    format_timestamp,
    format_table,
    format_list,
    format_key_value_pairs,
    format_report_header,
    format_report_section,
    format_summary_box,
    format_progress_bar,
    truncate_string,
)


class TestNumberFormatting:
    """Tests for number formatting functions."""

    def test_format_number_basic(self):
        """Test basic number formatting."""
        assert format_number(1234.5678) == "1,234.57"
        assert format_number(1234567.89, 1) == "1,234,567.9"

    def test_format_number_no_decimals(self):
        """Test formatting without decimals."""
        assert format_number(1234, decimal_places=0) == "1,234"

    def test_format_number_custom_separators(self):
        """Test custom separators."""
        result = format_number(1234.5, thousands_separator=" ", decimal_separator=",")
        assert result == "1 234,50"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.1567) == "15.7%"
        assert format_percentage(0.12345, decimal_places=2) == "12.35%"

    def test_format_percentage_no_multiply(self):
        """Test percentage without multiplication."""
        assert format_percentage(85.5, multiply_by_100=False) == "85.5%"

    def test_format_currency_prefix(self):
        """Test currency formatting with prefix."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1000000, decimal_places=0) == "$1,000,000"

    def test_format_currency_suffix(self):
        """Test currency formatting with suffix."""
        result = format_currency(1234.5, currency_symbol="€", symbol_position="suffix")
        assert result == "1,234.50€"

    def test_format_scientific(self):
        """Test scientific notation formatting."""
        result = format_scientific(1234567.89)
        assert "1.23e+06" in result

        result = format_scientific(0.00012345, decimal_places=3)
        assert "1.234e-04" in result

    def test_format_engineering(self):
        """Test engineering notation formatting."""
        assert format_engineering(1234) == "1.23k"
        assert format_engineering(1234567) == "1.23M"
        assert format_engineering(1234567890) == "1.23G"

    def test_format_engineering_small(self):
        """Test engineering notation for small numbers."""
        result = format_engineering(0.0123)
        assert "12." in result
        assert "m" in result  # milli

    def test_format_si_unit(self):
        """Test SI unit formatting."""
        assert format_si_unit(1234, "W") == "1.23kW"
        assert format_si_unit(5000000, "Wh") == "5.00MWh"

    def test_format_compact(self):
        """Test compact number formatting."""
        assert format_compact(1234) == "1.2K"
        assert format_compact(1234567) == "1.2M"
        assert format_compact(1234567890) == "1.2B"
        assert format_compact(123) == "123"


class TestDateTimeFormatting:
    """Tests for date and time formatting functions."""

    def test_format_date(self):
        """Test date formatting."""
        test_date = date(2024, 3, 15)
        assert format_date(test_date) == "2024-03-15"

    def test_format_date_custom_format(self):
        """Test date with custom format."""
        test_date = date(2024, 3, 15)
        result = format_date(test_date, "%B %d, %Y")
        assert result == "March 15, 2024"

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 3, 15, 14, 30, 45)
        assert format_datetime(dt) == "2024-03-15 14:30:45"

    def test_format_datetime_custom_format(self):
        """Test datetime with custom format."""
        dt = datetime(2024, 3, 15, 14, 30, 45)
        result = format_datetime(dt, "%B %d, %Y at %I:%M %p")
        assert result == "March 15, 2024 at 02:30 PM"

    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(45) == "45s"
        assert format_duration(86400) == "1d"

    def test_format_duration_complex(self):
        """Test complex duration."""
        # 2 days, 3 hours, 4 minutes, 5 seconds
        seconds = 2 * 86400 + 3 * 3600 + 4 * 60 + 5
        result = format_duration(seconds)
        assert "2d" in result
        assert "3h" in result
        assert "4m" in result
        assert "5s" in result

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        dt = datetime(2024, 3, 9, 15, 20, 0)
        result = format_timestamp(dt)
        assert "2024-03-09" in result
        assert "15:20:00" in result


class TestDataStructureFormatting:
    """Tests for data structure formatting functions."""

    def test_format_table(self):
        """Test table formatting."""
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = format_table(data)

        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "|" in result  # Table separator
        assert "-" in result  # Header separator

    def test_format_table_empty(self):
        """Test empty table."""
        result = format_table([])
        assert result == ""

    def test_format_table_custom_headers(self):
        """Test table with custom headers."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = format_table(data, headers=["a"])

        assert "a" in result
        assert "b" not in result

    def test_format_list(self):
        """Test list formatting."""
        items = ["First item", "Second item", "Third item"]
        result = format_list(items)

        assert "First item" in result
        assert "Second item" in result
        assert "Third item" in result
        assert "•" in result

    def test_format_list_custom_bullet(self):
        """Test list with custom bullet."""
        items = ["Item 1", "Item 2"]
        result = format_list(items, bullet="-")

        assert "-" in result
        assert "•" not in result

    def test_format_key_value_pairs(self):
        """Test key-value pair formatting."""
        data = {"Name": "Solar Panel", "Power": "300W", "Efficiency": "20%"}
        result = format_key_value_pairs(data)

        assert "Name" in result
        assert "Solar Panel" in result
        assert "Power" in result
        assert "300W" in result
        assert ":" in result

    def test_format_key_value_pairs_custom_separator(self):
        """Test key-value with custom separator."""
        data = {"a": 1, "b": 2}
        result = format_key_value_pairs(data, separator=" = ")

        assert " = " in result
        assert ":" not in result


class TestReportFormatting:
    """Tests for report formatting functions."""

    def test_format_report_header(self):
        """Test report header formatting."""
        result = format_report_header("PV System Analysis")

        assert "PV System Analysis" in result
        assert "=" in result
        lines = result.split("\n")
        assert len(lines) == 3  # Border, title, border

    def test_format_report_section(self):
        """Test report section formatting."""
        result = format_report_section("Summary", "This is the summary.")

        assert "Summary" in result
        assert "This is the summary." in result
        assert "─" in result  # Separator

    def test_format_summary_box(self):
        """Test summary box formatting."""
        metrics = {"Total Energy": "1000 kWh", "Efficiency": "18.5%"}
        result = format_summary_box("System Metrics", metrics)

        assert "System Metrics" in result
        assert "Total Energy" in result
        assert "1000 kWh" in result
        assert "Efficiency" in result
        assert "18.5%" in result
        assert "┌" in result  # Box corners
        assert "└" in result

    def test_format_summary_box_empty(self):
        """Test summary box with no items."""
        result = format_summary_box("Empty", {})

        assert "Empty" in result
        assert "┌" in result
        assert "└" in result

    def test_format_progress_bar(self):
        """Test progress bar formatting."""
        result = format_progress_bar(75, 100)

        assert "75.0%" in result
        assert "█" in result  # Fill character
        assert "░" in result  # Empty character

    def test_format_progress_bar_full(self):
        """Test full progress bar."""
        result = format_progress_bar(100, 100, width=20)

        assert "100.0%" in result
        assert result.count("█") == 20  # All filled

    def test_format_progress_bar_no_percentage(self):
        """Test progress bar without percentage."""
        result = format_progress_bar(50, 100, show_percentage=False)

        assert "%" not in result
        assert "█" in result

    def test_format_progress_bar_zero_total(self):
        """Test progress bar with zero total."""
        result = format_progress_bar(0, 0)
        assert "0.0%" in result


class TestStringUtilities:
    """Tests for string utility functions."""

    def test_truncate_string_no_truncation(self):
        """Test truncate with short string."""
        result = truncate_string("Short", 10)
        assert result == "Short"

    def test_truncate_string_with_truncation(self):
        """Test truncate with long string."""
        result = truncate_string("This is a long string", 10)
        assert result == "This is..."
        assert len(result) == 10

    def test_truncate_string_custom_suffix(self):
        """Test truncate with custom suffix."""
        result = truncate_string("Long text here", 10, suffix="…")
        assert result == "Long text…"
        assert len(result) == 10
