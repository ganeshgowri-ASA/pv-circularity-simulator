"""
IEC 63202 CTM Testing Module.

This module provides comprehensive Cell-to-Module (CTM) testing and validation
according to IEC 63202 standard.
"""

from pv_circularity_simulator.core.iec63202.models import (
    CTMTestConfig,
    CTMTestResult,
    CTMCertificate,
    CellProperties,
    ModuleConfiguration,
    ReferenceDeviceData,
    FlashSimulatorData,
    IVCurveData,
    CTMLossComponents,
    CellTechnology,
    FlashSimulatorType,
    TestStatus,
)
from pv_circularity_simulator.core.iec63202.tester import IEC63202CTMTester
from pv_circularity_simulator.core.iec63202.loss_analyzer import CTMPowerLossAnalyzer
from pv_circularity_simulator.core.iec63202.calibration import ReferenceDeviceCalibration
from pv_circularity_simulator.core.iec63202.report import CTMTestReport

__all__ = [
    # Models
    "CTMTestConfig",
    "CTMTestResult",
    "CTMCertificate",
    "CellProperties",
    "ModuleConfiguration",
    "ReferenceDeviceData",
    "FlashSimulatorData",
    "IVCurveData",
    "CTMLossComponents",
    "CellTechnology",
    "FlashSimulatorType",
    "TestStatus",
    # Classes
    "IEC63202CTMTester",
    "CTMPowerLossAnalyzer",
    "ReferenceDeviceCalibration",
    "CTMTestReport",
]
