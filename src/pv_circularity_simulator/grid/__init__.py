"""
Grid interaction and smart grid integration module.

Provides comprehensive grid interaction capabilities for PV systems including
grid code compliance, reactive power control, frequency regulation, and
SCADA communication.
"""

from pv_circularity_simulator.grid.grid_interaction import (
    ComplianceCheckResult,
    FrequencyRegulationConfig,
    GridCodeLimits,
    GridCodeStandard,
    GridConnectionState,
    GridInteraction,
    GridState,
    PowerQualityMetric,
    ReactivePowerControlConfig,
    SCADAConfig,
    SCADAProtocol,
)

__all__ = [
    "GridInteraction",
    "GridState",
    "GridCodeStandard",
    "GridCodeLimits",
    "GridConnectionState",
    "PowerQualityMetric",
    "ReactivePowerControlConfig",
    "FrequencyRegulationConfig",
    "SCADAConfig",
    "SCADAProtocol",
    "ComplianceCheckResult",
]
