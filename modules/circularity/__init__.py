"""
Circularity Module Package - Group 4.

This package provides comprehensive circular economy and advanced system design tools:
- Revamp & Repower Planning (B10)
- Circularity Assessment - 3R Framework (B11)
- Hybrid Energy System Design (B12)
"""

from modules.circularity.revamp_repower import (
    RevampRepowerPlanner,
    render_revamp_repower
)

from modules.circularity.circularity_3r import (
    CircularityAnalyzer,
    render_circularity_3r
)

from modules.circularity.hybrid_systems import (
    HybridSystemDesigner,
    render_hybrid_systems
)

__all__ = [
    # Revamp & Repower
    'RevampRepowerPlanner',
    'render_revamp_repower',

    # Circularity 3R
    'CircularityAnalyzer',
    'render_circularity_3r',

    # Hybrid Systems
    'HybridSystemDesigner',
    'render_hybrid_systems',
]

__version__ = '1.0.0'
__author__ = 'PV Circularity Simulator Team'
