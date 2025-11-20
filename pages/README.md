# Pages Directory

This directory contains the individual page modules for the PV Circularity Simulator multi-page Streamlit application.

## Structure

Each page module should follow this structure:

```python
"""
Page Name - Brief Description

Detailed description of the page functionality.
"""

import streamlit as st
from typing import Any, Dict, Optional


def render() -> None:
    """
    Render the page content.

    This function is called by the main app when this page is active.
    All page-specific UI and logic should be contained here.
    """
    st.header("Page Title")

    # Page content goes here
    pass


def initialize_page_state() -> None:
    """
    Initialize page-specific session state variables.

    Called before render() to ensure all required state exists.
    """
    if "page_specific_var" not in st.session_state:
        st.session_state.page_specific_var = default_value


# Additional helper functions as needed
```

## Planned Pages

1. **home.py** - Application overview and dashboard
2. **cell_design.py** - PV cell design and SCAPS integration
3. **module_engineering.py** - Module design and configuration
4. **ctm_loss_analysis.py** - Cell-to-module loss analysis
5. **system_planning.py** - System architecture and planning
6. **performance_monitoring.py** - Real-time performance tracking
7. **energy_forecasting.py** - Energy production forecasting
8. **reliability_testing.py** - Reliability and durability testing
9. **circularity_analysis.py** - 3R analysis (Reduce, Reuse, Recycle)
10. **circular_economy.py** - Circular economy modeling and metrics
11. **settings.py** - Application settings and preferences

## Development Guidelines

- Each page should be self-contained and independent
- Use `st.session_state` for sharing data between pages
- Follow the established docstring format (Google style)
- Include type hints for all functions
- Handle errors gracefully with appropriate user feedback
- Use the session_state.simulation_data structure for storing results
