"""
Design module for PV Circularity Simulator.

This package contains modules for PV cell and module design:
- Materials database and selection
- Cell design and SCAPS-1D simulation
- Module design and CTM loss analysis
"""

from modules.design.materials_database import MaterialsDatabase, render_materials_database
from modules.design.cell_design import CellDesignSimulator, render_cell_design
from modules.design.module_design import ModuleDesigner, render_module_design

__all__ = [
    "MaterialsDatabase",
    "render_materials_database",
    "CellDesignSimulator",
    "render_cell_design",
    "ModuleDesigner",
    "render_module_design"
]
