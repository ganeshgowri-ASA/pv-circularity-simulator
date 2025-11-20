"""Enumerations for PV system components and analysis."""

from enum import Enum


class ComponentType(str, Enum):
    """Types of PV system components."""

    MODULE = "module"
    INVERTER = "inverter"
    RACKING = "racking"
    WIRING = "wiring"
    MONITORING = "monitoring"
    COMBINER_BOX = "combiner_box"
    TRANSFORMER = "transformer"
    DISCONNECT = "disconnect"


class HealthStatus(str, Enum):
    """Health status of components."""

    EXCELLENT = "excellent"  # 95-100% of rated performance
    GOOD = "good"  # 85-95% of rated performance
    FAIR = "fair"  # 70-85% of rated performance
    POOR = "poor"  # 50-70% of rated performance
    CRITICAL = "critical"  # <50% of rated performance
    FAILED = "failed"  # Non-functional


class ModuleTechnology(str, Enum):
    """PV module technology types."""

    MONO_SI = "monocrystalline_silicon"
    POLY_SI = "polycrystalline_silicon"
    THIN_FILM_CDTE = "thin_film_cdte"
    THIN_FILM_CIGS = "thin_film_cigs"
    PERC = "perc"
    TOPCON = "topcon"
    HJT = "heterojunction"
    BIFACIAL = "bifacial"
    TANDEM = "tandem"


class ClimateZone(str, Enum):
    """Climate zones affecting PV system performance."""

    TROPICAL = "tropical"
    ARID = "arid"
    TEMPERATE = "temperate"
    CONTINENTAL = "continental"
    POLAR = "polar"
    COASTAL = "coastal"
    MOUNTAIN = "mountain"


class RepowerStrategy(str, Enum):
    """Strategies for repowering PV systems."""

    FULL_REPLACEMENT = "full_replacement"  # Replace all components
    MODULE_ONLY = "module_only"  # Replace only modules
    SELECTIVE = "selective"  # Replace specific failing components
    UPGRADE = "upgrade"  # Replace and upgrade capacity
    HYBRID = "hybrid"  # Add new capacity alongside existing
