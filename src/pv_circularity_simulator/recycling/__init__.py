"""
Recycling Economics & Material Recovery Module

This module provides comprehensive economic modeling for PV panel recycling,
including material extraction costs, recovery rates, revenue calculations,
and environmental credits.
"""

from .economics import (
    RecyclingEconomics,
    MaterialExtractionCosts,
    RecoveryRates,
    RecyclingRevenue,
    EnvironmentalCredits,
    PVMaterialType,
    RecyclingTechnology,
)

__all__ = [
    "RecyclingEconomics",
    "MaterialExtractionCosts",
    "RecoveryRates",
    "RecyclingRevenue",
    "EnvironmentalCredits",
    "PVMaterialType",
    "RecyclingTechnology",
]
