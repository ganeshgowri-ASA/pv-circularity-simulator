"""Monitoring module for PV panel analysis."""

from pv_simulator.monitoring.detectors.defect_classifier import (
    ClassifiedDefect,
    DefectClassifier,
    DefectLocation,
    DefectSeverity,
    DefectType,
)
from pv_simulator.monitoring.detectors.defect_reporting import (
    DefectReport,
    DefectReporting,
    LocationMap,
    RepairMethod,
    RepairPriority,
    RepairRecommendation,
)
from pv_simulator.monitoring.detectors.image_processing import (
    CellSegmentation,
    ImageMetadata,
    ImageProcessing,
    ThermalAnalysisResult,
)
from pv_simulator.monitoring.detectors.roboflow_integrator import (
    BoundingBox,
    Detection,
    InferenceResult,
    RoboflowIntegrator,
)

__all__ = [
    # Roboflow Integration
    "RoboflowIntegrator",
    "Detection",
    "BoundingBox",
    "InferenceResult",
    # Defect Classification
    "DefectClassifier",
    "ClassifiedDefect",
    "DefectType",
    "DefectSeverity",
    "DefectLocation",
    # Image Processing
    "ImageProcessing",
    "ImageMetadata",
    "CellSegmentation",
    "ThermalAnalysisResult",
    # Defect Reporting
    "DefectReporting",
    "DefectReport",
    "LocationMap",
    "RepairRecommendation",
    "RepairMethod",
    "RepairPriority",
]
