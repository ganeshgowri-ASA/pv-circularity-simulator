"""
Defect Classification Module for PV Panels.

This module provides specialized classification methods for different types
of photovoltaic panel defects including cracks, hotspots, delamination,
soiling, and potential-induced degradation (PID).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from scipy import ndimage
from skimage import feature, filters, measure, morphology

from pv_simulator.config import get_settings
from pv_simulator.monitoring.detectors.roboflow_integrator import (
    Detection,
    InferenceResult,
    RoboflowIntegrator,
)

logger = logging.getLogger(__name__)


class DefectType(str, Enum):
    """Types of PV panel defects."""

    CRACK = "crack"
    HOTSPOT = "hotspot"
    DELAMINATION = "delamination"
    SOILING = "soiling"
    PID = "pid"
    CORROSION = "corrosion"
    BURN_MARK = "burn_mark"
    CELL_BREAKAGE = "cell_breakage"
    UNKNOWN = "unknown"


class DefectSeverity(str, Enum):
    """Defect severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefectLocation(BaseModel):
    """
    Defect location information.

    Attributes:
        x: X coordinate (normalized 0-1)
        y: Y coordinate (normalized 0-1)
        width: Width (normalized 0-1)
        height: Height (normalized 0-1)
        cell_row: Cell row index (if applicable)
        cell_col: Cell column index (if applicable)
    """

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)
    cell_row: Optional[int] = None
    cell_col: Optional[int] = None


class ClassifiedDefect(BaseModel):
    """
    Classified defect with detailed analysis.

    Attributes:
        defect_type: Type of defect
        severity: Severity level
        confidence: Detection confidence (0-1)
        location: Defect location
        area_percentage: Percentage of panel area affected
        characteristics: Additional characteristics
        timestamp: Detection timestamp
    """

    defect_type: DefectType
    severity: DefectSeverity
    confidence: float = Field(..., ge=0.0, le=1.0)
    location: DefectLocation
    area_percentage: float = Field(..., ge=0.0, le=100.0)
    characteristics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


class DefectClassifier:
    """
    PV panel defect classifier using computer vision and AI.

    This class provides specialized methods for detecting and classifying
    different types of defects in photovoltaic panels.

    Attributes:
        roboflow: Roboflow API integrator
        settings: Application settings
    """

    def __init__(self, roboflow_integrator: Optional[RoboflowIntegrator] = None):
        """
        Initialize defect classifier.

        Args:
            roboflow_integrator: Optional RoboflowIntegrator instance
        """
        self.settings = get_settings()
        self.roboflow = roboflow_integrator or RoboflowIntegrator()
        logger.info("Initialized DefectClassifier")

    def crack_detection(
        self, image: Union[str, np.ndarray, Image.Image], use_ai: bool = True
    ) -> List[ClassifiedDefect]:
        """
        Detect cracks in PV panels.

        Uses both AI-based detection (Roboflow) and traditional computer vision
        techniques to identify micro-cracks and cell breakage.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            use_ai: Whether to use AI-based detection

        Returns:
            List of detected crack defects

        Example:
            >>> classifier = DefectClassifier()
            >>> defects = classifier.crack_detection("panel.jpg")
            >>> for defect in defects:
            ...     print(f"Crack severity: {defect.severity}")
        """
        defects = []

        if use_ai:
            # AI-based detection using Roboflow
            try:
                result = self.roboflow.real_time_detection(image, preprocess=True)
                for detection in result.detections:
                    if detection.class_name.lower() in ["crack", "microcrack", "cell_breakage"]:
                        # Classify severity based on size and confidence
                        severity = self._classify_crack_severity(
                            detection.bbox.width,
                            detection.bbox.height,
                            detection.confidence,
                        )

                        area_pct = (
                            detection.bbox.width
                            * detection.bbox.height
                            / (detection.image_width * detection.image_height)
                            * 100
                        )

                        defect = ClassifiedDefect(
                            defect_type=DefectType.CRACK,
                            severity=severity,
                            confidence=detection.confidence,
                            location=DefectLocation(
                                x=detection.bbox.x / detection.image_width,
                                y=detection.bbox.y / detection.image_height,
                                width=detection.bbox.width / detection.image_width,
                                height=detection.bbox.height / detection.image_height,
                            ),
                            area_percentage=area_pct,
                            characteristics={
                                "detection_method": "ai",
                                "class_name": detection.class_name,
                            },
                        )
                        defects.append(defect)

            except Exception as e:
                logger.error(f"AI crack detection failed: {e}")

        # Computer vision-based crack detection
        try:
            cv_defects = self._cv_crack_detection(image)
            defects.extend(cv_defects)
        except Exception as e:
            logger.error(f"CV crack detection failed: {e}")

        logger.info(f"Crack detection: found {len(defects)} cracks")
        return defects

    def hotspot_identification(
        self, thermal_image: Union[str, np.ndarray, Image.Image], threshold_celsius: float = 85.0
    ) -> List[ClassifiedDefect]:
        """
        Identify hotspots in thermal images of PV panels.

        Hotspots indicate areas of high electrical resistance or cell damage
        that generate excessive heat.

        Args:
            thermal_image: Thermal/infrared image
            threshold_celsius: Temperature threshold for hotspot detection

        Returns:
            List of detected hotspot defects

        Example:
            >>> classifier = DefectClassifier()
            >>> hotspots = classifier.hotspot_identification("thermal.jpg", threshold_celsius=90)
            >>> print(f"Found {len(hotspots)} hotspots")
        """
        defects = []

        # Load thermal image
        if isinstance(thermal_image, str):
            img = cv2.imread(thermal_image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(thermal_image, Image.Image):
            img = np.array(thermal_image.convert("L"))
        else:
            img = thermal_image

        if img is None:
            raise ValueError("Failed to load thermal image")

        # Normalize to temperature range (assuming 0-255 maps to 0-150°C)
        temp_img = img.astype(float) * 150.0 / 255.0

        # Threshold for hotspots
        hotspot_mask = temp_img > threshold_celsius

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hotspot_mask = cv2.morphologyEx(hotspot_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Find connected components
        labeled = measure.label(hotspot_mask)
        regions = measure.regionprops(labeled, intensity_image=temp_img)

        for region in regions:
            if region.area < 10:  # Filter small noise
                continue

            # Calculate hotspot characteristics
            max_temp = region.max_intensity
            mean_temp = region.mean_intensity
            area_pct = region.area / (img.shape[0] * img.shape[1]) * 100

            # Classify severity based on temperature
            if max_temp > 100:
                severity = DefectSeverity.CRITICAL
            elif max_temp > 90:
                severity = DefectSeverity.HIGH
            elif max_temp > threshold_celsius:
                severity = DefectSeverity.MEDIUM
            else:
                severity = DefectSeverity.LOW

            # Get bounding box
            minr, minc, maxr, maxc = region.bbox
            center_y = (minr + maxr) / 2 / img.shape[0]
            center_x = (minc + maxc) / 2 / img.shape[1]
            width = (maxc - minc) / img.shape[1]
            height = (maxr - minr) / img.shape[0]

            defect = ClassifiedDefect(
                defect_type=DefectType.HOTSPOT,
                severity=severity,
                confidence=min(1.0, (max_temp - threshold_celsius) / threshold_celsius),
                location=DefectLocation(x=center_x, y=center_y, width=width, height=height),
                area_percentage=area_pct,
                characteristics={
                    "max_temperature": float(max_temp),
                    "mean_temperature": float(mean_temp),
                    "detection_method": "thermal",
                },
            )
            defects.append(defect)

        logger.info(f"Hotspot detection: found {len(defects)} hotspots")
        return defects

    def delamination_detection(
        self, image: Union[str, np.ndarray, Image.Image], use_ai: bool = True
    ) -> List[ClassifiedDefect]:
        """
        Detect delamination in PV panels.

        Delamination is the separation of panel layers, visible as bubbles
        or discoloration in EL/visual images.

        Args:
            image: Input image (EL or visual)
            use_ai: Whether to use AI-based detection

        Returns:
            List of detected delamination defects

        Example:
            >>> classifier = DefectClassifier()
            >>> defects = classifier.delamination_detection("el_image.jpg")
        """
        defects = []

        if use_ai:
            # AI-based detection
            try:
                result = self.roboflow.real_time_detection(image, preprocess=True)
                for detection in result.detections:
                    if detection.class_name.lower() in ["delamination", "bubble", "separation"]:
                        area_pct = (
                            detection.bbox.width
                            * detection.bbox.height
                            / (detection.image_width * detection.image_height)
                            * 100
                        )

                        # Delamination severity based on area
                        if area_pct > 5:
                            severity = DefectSeverity.CRITICAL
                        elif area_pct > 2:
                            severity = DefectSeverity.HIGH
                        elif area_pct > 0.5:
                            severity = DefectSeverity.MEDIUM
                        else:
                            severity = DefectSeverity.LOW

                        defect = ClassifiedDefect(
                            defect_type=DefectType.DELAMINATION,
                            severity=severity,
                            confidence=detection.confidence,
                            location=DefectLocation(
                                x=detection.bbox.x / detection.image_width,
                                y=detection.bbox.y / detection.image_height,
                                width=detection.bbox.width / detection.image_width,
                                height=detection.bbox.height / detection.image_height,
                            ),
                            area_percentage=area_pct,
                            characteristics={
                                "detection_method": "ai",
                                "class_name": detection.class_name,
                            },
                        )
                        defects.append(defect)

            except Exception as e:
                logger.error(f"AI delamination detection failed: {e}")

        # Computer vision-based detection for bubbles/discoloration
        try:
            cv_defects = self._cv_delamination_detection(image)
            defects.extend(cv_defects)
        except Exception as e:
            logger.error(f"CV delamination detection failed: {e}")

        logger.info(f"Delamination detection: found {len(defects)} defects")
        return defects

    def soiling_analysis(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> List[ClassifiedDefect]:
        """
        Analyze soiling (dirt, dust, debris) on PV panels.

        Soiling reduces panel efficiency by blocking sunlight.

        Args:
            image: Visual/RGB image of panel

        Returns:
            List of detected soiling defects

        Example:
            >>> classifier = DefectClassifier()
            >>> soiling = classifier.soiling_analysis("panel_rgb.jpg")
            >>> total_soiling = sum(d.area_percentage for d in soiling)
            >>> print(f"Total soiling coverage: {total_soiling:.1f}%")
        """
        defects = []

        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect dark regions (soiling)
        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
        )

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = img.shape[0] * img.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small noise
                continue

            area_pct = area / total_area * 100

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Classify severity based on coverage
            if area_pct > 10:
                severity = DefectSeverity.HIGH
            elif area_pct > 5:
                severity = DefectSeverity.MEDIUM
            else:
                severity = DefectSeverity.LOW

            defect = ClassifiedDefect(
                defect_type=DefectType.SOILING,
                severity=severity,
                confidence=0.8,  # Computer vision confidence
                location=DefectLocation(
                    x=(x + w / 2) / img.shape[1],
                    y=(y + h / 2) / img.shape[0],
                    width=w / img.shape[1],
                    height=h / img.shape[0],
                ),
                area_percentage=area_pct,
                characteristics={
                    "detection_method": "cv",
                    "contour_area": int(area),
                },
            )
            defects.append(defect)

        # Calculate total soiling coverage
        total_soiling = sum(d.area_percentage for d in defects)
        logger.info(f"Soiling analysis: {total_soiling:.1f}% coverage, {len(defects)} regions")

        return defects

    def pid_detection(
        self, el_image: Union[str, np.ndarray, Image.Image]
    ) -> List[ClassifiedDefect]:
        """
        Detect Potential-Induced Degradation (PID) in EL images.

        PID appears as darkened cells or regions in electroluminescence images,
        indicating performance degradation.

        Args:
            el_image: Electroluminescence image

        Returns:
            List of detected PID defects

        Example:
            >>> classifier = DefectClassifier()
            >>> pid_defects = classifier.pid_detection("el_image.jpg")
            >>> critical_defects = [d for d in pid_defects if d.severity == DefectSeverity.CRITICAL]
        """
        defects = []

        # Load EL image
        if isinstance(el_image, str):
            img = cv2.imread(el_image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(el_image, Image.Image):
            img = np.array(el_image.convert("L"))
        else:
            img = el_image

        if img is None:
            raise ValueError("Failed to load EL image")

        # Normalize
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Detect dark cells (PID indicator)
        # Calculate mean intensity
        mean_intensity = np.mean(img_normalized)

        # Threshold for dark regions (PID)
        threshold = mean_intensity * 0.6  # 60% of mean
        pid_mask = img_normalized < threshold

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        pid_mask = cv2.morphologyEx(pid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find connected components
        labeled = measure.label(pid_mask)
        regions = measure.regionprops(labeled, intensity_image=img_normalized)

        for region in regions:
            if region.area < 50:  # Filter noise
                continue

            # Calculate PID characteristics
            mean_intensity_region = region.mean_intensity
            intensity_ratio = mean_intensity_region / mean_intensity
            area_pct = region.area / (img.shape[0] * img.shape[1]) * 100

            # Classify severity based on intensity reduction
            if intensity_ratio < 0.3:
                severity = DefectSeverity.CRITICAL
            elif intensity_ratio < 0.5:
                severity = DefectSeverity.HIGH
            elif intensity_ratio < 0.7:
                severity = DefectSeverity.MEDIUM
            else:
                severity = DefectSeverity.LOW

            # Get bounding box
            minr, minc, maxr, maxc = region.bbox
            center_y = (minr + maxr) / 2 / img.shape[0]
            center_x = (minc + maxc) / 2 / img.shape[1]
            width = (maxc - minc) / img.shape[1]
            height = (maxr - minr) / img.shape[0]

            defect = ClassifiedDefect(
                defect_type=DefectType.PID,
                severity=severity,
                confidence=1.0 - intensity_ratio,  # Lower intensity = higher confidence
                location=DefectLocation(x=center_x, y=center_y, width=width, height=height),
                area_percentage=area_pct,
                characteristics={
                    "detection_method": "el_analysis",
                    "intensity_ratio": float(intensity_ratio),
                    "mean_intensity": float(mean_intensity_region),
                },
            )
            defects.append(defect)

        logger.info(f"PID detection: found {len(defects)} affected regions")
        return defects

    def _classify_crack_severity(
        self, width: float, height: float, confidence: float
    ) -> DefectSeverity:
        """Classify crack severity based on size and confidence."""
        area = width * height

        if area > 10000 or confidence > 0.9:
            return DefectSeverity.CRITICAL
        elif area > 5000 or confidence > 0.7:
            return DefectSeverity.HIGH
        elif area > 1000 or confidence > 0.5:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW

    def _cv_crack_detection(self, image: Union[str, np.ndarray, Image.Image]) -> List[ClassifiedDefect]:
        """Computer vision-based crack detection using edge detection."""
        defects = []

        # Load image
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("L"))
        else:
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = image

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Detect edges (cracks appear as lines)
        edges = cv2.Canny(img, 50, 150)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Cracks typically have high aspect ratio (long and thin)
            if aspect_ratio > 3 or aspect_ratio < 0.3:
                area_pct = area / (img.shape[0] * img.shape[1]) * 100

                defect = ClassifiedDefect(
                    defect_type=DefectType.CRACK,
                    severity=DefectSeverity.MEDIUM,
                    confidence=0.6,
                    location=DefectLocation(
                        x=(x + w / 2) / img.shape[1],
                        y=(y + h / 2) / img.shape[0],
                        width=w / img.shape[1],
                        height=h / img.shape[0],
                    ),
                    area_percentage=area_pct,
                    characteristics={
                        "detection_method": "cv_edge",
                        "aspect_ratio": float(aspect_ratio),
                    },
                )
                defects.append(defect)

        return defects

    def _cv_delamination_detection(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> List[ClassifiedDefect]:
        """Computer vision-based delamination detection."""
        defects = []

        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Convert to LAB color space (better for color analysis)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Detect bright spots (bubbles/delamination often appear brighter)
        threshold = np.percentile(l, 90)
        bubble_mask = l > threshold

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bubble_mask = cv2.morphologyEx(bubble_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            area_pct = area / (img.shape[0] * img.shape[1]) * 100

            defect = ClassifiedDefect(
                defect_type=DefectType.DELAMINATION,
                severity=DefectSeverity.MEDIUM if area_pct > 1 else DefectSeverity.LOW,
                confidence=0.7,
                location=DefectLocation(
                    x=(x + w / 2) / img.shape[1],
                    y=(y + h / 2) / img.shape[0],
                    width=w / img.shape[1],
                    height=h / img.shape[0],
                ),
                area_percentage=area_pct,
                characteristics={
                    "detection_method": "cv_color",
                    "contour_area": int(area),
                },
            )
            defects.append(defect)

        return defects
