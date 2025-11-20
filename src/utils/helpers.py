"""
Utility functions and helpers for the PV Circularity Simulator.

This module provides common utility functions used throughout the application.
"""

from typing import Dict, Any, Optional
import re


def format_power(power_kw: float, precision: int = 2) -> str:
    """
    Format power value with appropriate units.

    Args:
        power_kw: Power in kilowatts
        precision: Decimal precision

    Returns:
        Formatted string with units (W, kW, or MW)
    """
    if abs(power_kw) < 0.001:
        return f"{power_kw * 1000000:.{precision}f} W"
    elif abs(power_kw) < 1.0:
        return f"{power_kw * 1000:.{precision}f} W"
    elif abs(power_kw) < 1000.0:
        return f"{power_kw:.{precision}f} kW"
    else:
        return f"{power_kw / 1000:.{precision}f} MW"


def format_energy(energy_kwh: float, precision: int = 2) -> str:
    """
    Format energy value with appropriate units.

    Args:
        energy_kwh: Energy in kilowatt-hours
        precision: Decimal precision

    Returns:
        Formatted string with units (Wh, kWh, or MWh)
    """
    if abs(energy_kwh) < 0.001:
        return f"{energy_kwh * 1000000:.{precision}f} Wh"
    elif abs(energy_kwh) < 1.0:
        return f"{energy_kwh * 1000:.{precision}f} Wh"
    elif abs(energy_kwh) < 1000.0:
        return f"{energy_kwh:.{precision}f} kWh"
    else:
        return f"{energy_kwh / 1000:.{precision}f} MWh"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a ratio as a percentage.

    Args:
        value: Ratio value (0.0-1.0)
        precision: Decimal precision

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    required_fields = ["system_name"]
    for field in required_fields:
        if field not in config:
            return False

    # Validate components if present
    if "components" in config:
        if not isinstance(config["components"], list):
            return False

        for component in config["components"]:
            if not isinstance(component, dict):
                return False
            if "component_id" not in component or "component_type" not in component:
                return False

    return True


def sanitize_component_id(component_id: str) -> str:
    """
    Sanitize component ID to ensure it's valid.

    Args:
        component_id: Raw component ID

    Returns:
        Sanitized component ID
    """
    # Remove special characters, keep alphanumeric and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', component_id)

    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'c_' + sanitized

    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = sanitized + '_001'

    return sanitized


def calculate_time_steps(duration_hours: int, step_minutes: int) -> int:
    """
    Calculate number of time steps for simulation.

    Args:
        duration_hours: Simulation duration in hours
        step_minutes: Time step in minutes

    Returns:
        Number of time steps
    """
    total_minutes = duration_hours * 60
    return int(total_minutes / step_minutes)


def generate_default_load_profile(num_steps: int) -> list:
    """
    Generate a default load profile for simulation.

    Args:
        num_steps: Number of time steps

    Returns:
        List of load values in kW
    """
    import numpy as np

    # Create a simple daily load profile
    # Higher during day, lower at night
    hours = np.linspace(0, 24, num_steps)

    # Base load + variable component
    base_load = 2.0  # kW
    variable_load = 3.0 * (np.sin((hours - 6) * np.pi / 12) + 1) / 2

    load_profile = base_load + variable_load
    return load_profile.tolist()


def generate_default_irradiance_profile(num_steps: int) -> list:
    """
    Generate a default solar irradiance profile.

    Args:
        num_steps: Number of time steps

    Returns:
        List of irradiance values in W/m²
    """
    import numpy as np

    # Create a simple daily irradiance profile
    hours = np.linspace(0, 24, num_steps)

    # Solar irradiance (higher during day)
    irradiance = np.maximum(
        0,
        1000 * np.sin((hours - 6) * np.pi / 12)
    )

    return irradiance.tolist()


def color_by_status(status: str) -> str:
    """
    Get color code for component status.

    Args:
        status: Component status string

    Returns:
        Color code for Streamlit or plotting
    """
    status_colors = {
        "operating": "#28a745",  # Green
        "idle": "#6c757d",  # Gray
        "charging": "#007bff",  # Blue
        "discharging": "#ffc107",  # Yellow
        "error": "#dc3545",  # Red
        "maintenance": "#fd7e14",  # Orange
    }

    return status_colors.get(status.lower(), "#6c757d")


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
