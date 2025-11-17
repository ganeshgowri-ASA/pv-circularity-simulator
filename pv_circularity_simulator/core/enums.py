"""Enumerations for PV module assessment and grading."""

from enum import Enum


class ModuleCondition(str, Enum):
    """Physical condition grades for PV modules.

    Attributes:
        EXCELLENT: Module in like-new condition, minimal wear
        GOOD: Module with minor cosmetic defects, fully functional
        FAIR: Module with visible wear but operational
        POOR: Module with significant damage or degradation
        FAILED: Module non-operational or critically damaged
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


class PerformanceLevel(str, Enum):
    """Performance classification based on testing results.

    Attributes:
        HIGH: Performance ≥90% of nameplate rating
        MEDIUM: Performance 70-90% of nameplate rating
        LOW: Performance 50-70% of nameplate rating
        CRITICAL: Performance <50% of nameplate rating
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class ReusePotential(str, Enum):
    """Overall reuse potential classification.

    Attributes:
        DIRECT_REUSE: Suitable for direct reuse in similar applications
        SECONDARY_MARKET: Suitable for secondary market applications
        COMPONENT_RECOVERY: Only suitable for component recovery
        RECYCLE_ONLY: Module should be recycled
        DISPOSE: Module requires proper disposal
    """
    DIRECT_REUSE = "direct_reuse"
    SECONDARY_MARKET = "secondary_market"
    COMPONENT_RECOVERY = "component_recovery"
    RECYCLE_ONLY = "recycle_only"
    DISPOSE = "dispose"


class DegradationType(str, Enum):
    """Types of module degradation.

    Attributes:
        POWER_LOSS: General power output degradation
        HOT_SPOT: Hot spot formation
        DELAMINATION: Layer delamination
        CELL_CRACK: Cell cracking
        DISCOLORATION: Module discoloration
        CORROSION: Corrosion of contacts/frames
        JUNCTION_BOX: Junction box issues
        BYPASS_DIODE: Bypass diode failure
        ENCAPSULANT: Encapsulant degradation
    """
    POWER_LOSS = "power_loss"
    HOT_SPOT = "hot_spot"
    DELAMINATION = "delamination"
    CELL_CRACK = "cell_crack"
    DISCOLORATION = "discoloration"
    CORROSION = "corrosion"
    JUNCTION_BOX = "junction_box"
    BYPASS_DIODE = "bypass_diode"
    ENCAPSULANT = "encapsulant"


class MarketSegment(str, Enum):
    """Target market segments for reused modules.

    Attributes:
        PREMIUM: Premium applications requiring high performance
        STANDARD: Standard residential/commercial applications
        OFF_GRID: Off-grid and remote applications
        DEVELOPING: Developing market applications
        INDUSTRIAL: Industrial and utility-scale applications
    """
    PREMIUM = "premium"
    STANDARD = "standard"
    OFF_GRID = "off_grid"
    DEVELOPING = "developing"
    INDUSTRIAL = "industrial"
