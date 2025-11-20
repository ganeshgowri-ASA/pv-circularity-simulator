"""
PV Circularity Simulator.

End-to-end photovoltaic lifecycle simulation platform including
cell design, module engineering, system planning, performance monitoring,
circularity modeling, and grid integration.
"""

__version__ = "0.1.0"

from pv_circularity_simulator.grid import GridInteraction, GridState

__all__ = ["GridInteraction", "GridState", "__version__"]
