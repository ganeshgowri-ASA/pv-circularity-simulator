"""
Modules package for PV Circularity Simulator.

Contains specialized modules for different aspects of PV simulation:
- scaps_wrapper: SCAPS-1D integration for device physics simulation
"""

from .scaps_wrapper import (
    CellArchitecture,
    CellTemplates,
    Contact,
    ContactType,
    DefectDistribution,
    DefectType,
    DeviceParams,
    DopingProfile,
    DopingType,
    InterfaceProperties,
    Layer,
    MaterialProperties,
    MaterialType,
    OpticalProperties,
    SCAPSInterface,
    SimulationResults,
    SimulationSettings,
)

__all__ = [
    # Main interface
    "SCAPSInterface",
    "CellTemplates",
    # Models
    "DeviceParams",
    "SimulationResults",
    "SimulationSettings",
    # Materials
    "MaterialType",
    "MaterialProperties",
    # Layers and structure
    "Layer",
    "DopingProfile",
    "DopingType",
    "InterfaceProperties",
    # Defects
    "DefectDistribution",
    "DefectType",
    # Contacts
    "Contact",
    "ContactType",
    # Optics
    "OpticalProperties",
    # Architecture
    "CellArchitecture",
]
