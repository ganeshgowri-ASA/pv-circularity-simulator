"""
Image Processing Module for PV Panel Analysis.

This module provides specialized image processing functions for different
imaging modalities: Electroluminescence (EL), thermal/infrared, and RGB/visual.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from scipy import ndimage, signal
from skimage import exposure, filters, measure, morphology, restoration

logger = logging.getLogger(__name__)


class ImageMetadata(BaseModel):
    """
    Image metadata and analysis results.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels
        dtype: Data type of pixels
        mean_intensity: Mean pixel intensity
        std_intensity: Standard deviation of intensity
        contrast: Image contrast measure
        quality_score: Overall quality score (0-1)
    """

    width: int
    height: int
    channels: int
    dtype: str
    mean_intensity: float
    std_intensity: float
    contrast: float
    quality_score: float = Field(ge=0.0, le=1.0)


class CellSegmentation(BaseModel):
    """
    Individual cell segmentation result.

    Attributes:
        cell_id: Unique cell identifier
        row: Row index
        col: Column index
        bbox: Bounding box (x, y, width, height)
        mean_intensity: Mean cell intensity
        uniformity: Intensity uniformity (0-1)
        defects: List of defects in this cell
    """

    cell_id: int
    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    mean_intensity: float
    uniformity: float = Field(ge=0.0, le=1.0)
    defects: List[str] = Field(default_factory=list)


class ThermalAnalysisResult(BaseModel):
    """
    Thermal image analysis result.

    Attributes:
        min_temp: Minimum temperature (Celsius)
        max_temp: Maximum temperature (Celsius)
        mean_temp: Mean temperature (Celsius)
        std_temp: Temperature standard deviation
        hotspot_count: Number of hotspots detected
        temperature_map: Temperature distribution map
    """

    min_temp: float
    max_temp: float
    mean_temp: float
    std_temp: float
    hotspot_count: int
    temperature_map: Optional[List[List[float]]] = None


class ImageProcessing:
    """
    Advanced image processing for PV panel analysis.

    Provides specialized processing methods for different imaging types:
    - EL (Electroluminescence) imaging for cell-level defects
    - Thermal imaging for hotspot and temperature analysis
    - RGB imaging for visual inspection and soiling detection
    """

    def __init__(self):
        """Initialize image processing module."""
        logger.info("Initialized ImageProcessing module")

    def el_image_analysis(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        segment_cells: bool = True,
        enhance: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze electroluminescence (EL) images for cell-level defects.

        EL imaging reveals electrical characteristics of solar cells,
        showing defects invisible to the naked eye.

        Args:
            image: Input EL image
            segment_cells: Whether to segment individual cells
            enhance: Whether to apply image enhancement

        Returns:
            Dictionary containing analysis results including:
            - metadata: Image metadata
            - cells: Cell segmentation results (if enabled)
            - defect_map: Binary defect map
            - quality_metrics: Quality assessment metrics

        Example:
            >>> processor = ImageProcessing()
            >>> results = processor.el_image_analysis("el_panel.jpg")
            >>> print(f"Detected {len(results['cells'])} cells")
            >>> print(f"Quality score: {results['metadata'].quality_score:.2f}")
        """
        logger.info("Starting EL image analysis")

        # Load image
        img = self._load_grayscale_image(image)

        # Enhance image if requested
        if enhance:
            img = self._enhance_el_image(img)

        # Calculate metadata
        metadata = self._calculate_image_metadata(img)

        results = {
            "metadata": metadata,
            "enhanced_image": img,
        }

        # Segment cells if requested
        if segment_cells:
            cells = self._segment_cells(img)
            results["cells"] = cells
            logger.info(f"Segmented {len(cells)} cells")

        # Create defect map
        defect_map = self._create_el_defect_map(img)
        results["defect_map"] = defect_map

        # Calculate quality metrics
        quality_metrics = self._calculate_el_quality_metrics(img, defect_map)
        results["quality_metrics"] = quality_metrics

        logger.info(f"EL analysis complete: quality_score={metadata.quality_score:.2f}")
        return results

    def thermal_image_processing(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        temp_min: float = 0.0,
        temp_max: float = 150.0,
        calibration: Optional[Dict[str, float]] = None,
    ) -> ThermalAnalysisResult:
        """
        Process thermal/infrared images for temperature analysis.

        Thermal imaging detects hotspots and temperature anomalies
        in operating PV panels.

        Args:
            image: Input thermal image
            temp_min: Minimum temperature for scaling (Celsius)
            temp_max: Maximum temperature for scaling (Celsius)
            calibration: Optional calibration parameters

        Returns:
            ThermalAnalysisResult with temperature analysis

        Example:
            >>> processor = ImageProcessing()
            >>> result = processor.thermal_image_processing("thermal.jpg", temp_max=120)
            >>> print(f"Max temp: {result.max_temp:.1f}°C")
            >>> print(f"Hotspots: {result.hotspot_count}")
        """
        logger.info("Starting thermal image processing")

        # Load thermal image
        img = self._load_grayscale_image(image)

        # Apply calibration if provided
        if calibration:
            img = self._apply_thermal_calibration(img, calibration)

        # Convert pixel values to temperature
        temp_map = self._pixel_to_temperature(img, temp_min, temp_max)

        # Calculate statistics
        min_temp = float(np.min(temp_map))
        max_temp = float(np.max(temp_map))
        mean_temp = float(np.mean(temp_map))
        std_temp = float(np.std(temp_map))

        # Detect hotspots (temperatures above threshold)
        threshold = mean_temp + 2 * std_temp
        hotspot_mask = temp_map > threshold
        hotspot_count = measure.label(hotspot_mask, return_num=True)[1]

        result = ThermalAnalysisResult(
            min_temp=min_temp,
            max_temp=max_temp,
            mean_temp=mean_temp,
            std_temp=std_temp,
            hotspot_count=hotspot_count,
            temperature_map=temp_map.tolist() if temp_map.size < 1000000 else None,
        )

        logger.info(
            f"Thermal analysis: temp_range={min_temp:.1f}-{max_temp:.1f}°C, "
            f"hotspots={hotspot_count}"
        )
        return result

    def rgb_analysis(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        detect_soiling: bool = True,
        color_calibration: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze RGB/visual images for surface defects and soiling.

        RGB imaging provides visual inspection data for detecting
        soiling, discoloration, and visible damage.

        Args:
            image: Input RGB image
            detect_soiling: Whether to perform soiling detection
            color_calibration: Whether to apply color calibration

        Returns:
            Dictionary containing:
            - metadata: Image metadata
            - soiling_map: Soiling detection map (if enabled)
            - color_stats: Color channel statistics
            - surface_quality: Surface quality metrics

        Example:
            >>> processor = ImageProcessing()
            >>> results = processor.rgb_analysis("panel_visual.jpg")
            >>> print(f"Soiling coverage: {results['soiling_coverage']:.1f}%")
            >>> print(f"Surface quality: {results['surface_quality']:.2f}")
        """
        logger.info("Starting RGB image analysis")

        # Load RGB image
        img = self._load_rgb_image(image)

        # Apply color calibration if requested
        if color_calibration:
            img = self._calibrate_rgb_image(img)

        # Calculate metadata
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        metadata = self._calculate_image_metadata(gray)

        results = {
            "metadata": metadata,
            "calibrated_image": img,
        }

        # Calculate color statistics
        color_stats = self._calculate_color_statistics(img)
        results["color_stats"] = color_stats

        # Detect soiling if requested
        if detect_soiling:
            soiling_map = self._detect_soiling_rgb(img)
            soiling_coverage = np.sum(soiling_map) / soiling_map.size * 100
            results["soiling_map"] = soiling_map
            results["soiling_coverage"] = float(soiling_coverage)
            logger.info(f"Soiling coverage: {soiling_coverage:.1f}%")

        # Calculate surface quality
        surface_quality = self._calculate_surface_quality(img)
        results["surface_quality"] = surface_quality

        logger.info(f"RGB analysis complete: surface_quality={surface_quality:.2f}")
        return results

    # Helper methods

    def _load_grayscale_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Load and convert image to grayscale."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("L"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img

    def _load_rgb_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Load RGB image."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img

    def _enhance_el_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance EL image using CLAHE and denoising."""
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        return enhanced

    def _calculate_image_metadata(self, img: np.ndarray) -> ImageMetadata:
        """Calculate image metadata and quality metrics."""
        # Calculate statistics
        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        # Calculate contrast (Michelson contrast)
        max_i = float(np.max(img))
        min_i = float(np.min(img))
        contrast = (max_i - min_i) / (max_i + min_i + 1e-10)

        # Quality score based on contrast and sharpness
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()

        # Normalize quality score (0-1)
        contrast_score = min(1.0, contrast * 2)
        sharpness_score = min(1.0, sharpness / 1000)
        quality_score = (contrast_score + sharpness_score) / 2

        return ImageMetadata(
            width=img.shape[1],
            height=img.shape[0],
            channels=1 if len(img.shape) == 2 else img.shape[2],
            dtype=str(img.dtype),
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            contrast=contrast,
            quality_score=quality_score,
        )

    def _segment_cells(self, img: np.ndarray, grid_size: Tuple[int, int] = (6, 10)) -> List[CellSegmentation]:
        """
        Segment individual solar cells from panel image.

        Args:
            img: Grayscale panel image
            grid_size: Expected grid size (rows, cols)

        Returns:
            List of segmented cells
        """
        cells = []
        rows, cols = grid_size
        h, w = img.shape

        cell_h = h // rows
        cell_w = w // cols

        cell_id = 0
        for i in range(rows):
            for j in range(cols):
                # Extract cell region
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < rows - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < cols - 1 else w

                cell_img = img[y1:y2, x1:x2]

                # Calculate cell statistics
                mean_intensity = float(np.mean(cell_img))
                std_intensity = float(np.std(cell_img))
                uniformity = 1.0 - min(1.0, std_intensity / (mean_intensity + 1e-10))

                # Detect defects in cell
                defects = []
                if mean_intensity < np.mean(img) * 0.7:
                    defects.append("low_intensity")
                if std_intensity > np.std(img) * 1.5:
                    defects.append("non_uniform")

                cell = CellSegmentation(
                    cell_id=cell_id,
                    row=i,
                    col=j,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    mean_intensity=mean_intensity,
                    uniformity=uniformity,
                    defects=defects,
                )
                cells.append(cell)
                cell_id += 1

        return cells

    def _create_el_defect_map(self, img: np.ndarray) -> np.ndarray:
        """Create binary defect map from EL image."""
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
        )

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        defect_map = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return defect_map

    def _calculate_el_quality_metrics(self, img: np.ndarray, defect_map: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for EL image."""
        defect_ratio = np.sum(defect_map > 0) / defect_map.size

        metrics = {
            "defect_ratio": float(defect_ratio),
            "mean_intensity": float(np.mean(img)),
            "intensity_uniformity": float(1.0 - np.std(img) / (np.mean(img) + 1e-10)),
            "overall_quality": float(1.0 - defect_ratio),
        }

        return metrics

    def _pixel_to_temperature(
        self, img: np.ndarray, temp_min: float, temp_max: float
    ) -> np.ndarray:
        """Convert pixel values to temperature."""
        # Normalize to 0-1
        normalized = img.astype(float) / 255.0

        # Scale to temperature range
        temp_map = normalized * (temp_max - temp_min) + temp_min

        return temp_map

    def _apply_thermal_calibration(
        self, img: np.ndarray, calibration: Dict[str, float]
    ) -> np.ndarray:
        """Apply thermal calibration parameters."""
        # Simple linear calibration
        offset = calibration.get("offset", 0.0)
        scale = calibration.get("scale", 1.0)

        calibrated = img.astype(float) * scale + offset
        calibrated = np.clip(calibrated, 0, 255).astype(np.uint8)

        return calibrated

    def _calibrate_rgb_image(self, img: np.ndarray) -> np.ndarray:
        """Apply white balance and color calibration to RGB image."""
        # Simple gray world white balance
        avg_r = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 2])

        avg_gray = (avg_r + avg_g + avg_b) / 3

        img_calibrated = img.copy().astype(float)
        img_calibrated[:, :, 0] *= avg_gray / (avg_r + 1e-10)
        img_calibrated[:, :, 1] *= avg_gray / (avg_g + 1e-10)
        img_calibrated[:, :, 2] *= avg_gray / (avg_b + 1e-10)

        img_calibrated = np.clip(img_calibrated, 0, 255).astype(np.uint8)

        return img_calibrated

    def _calculate_color_statistics(self, img: np.ndarray) -> Dict[str, Any]:
        """Calculate color channel statistics."""
        stats = {
            "red": {
                "mean": float(np.mean(img[:, :, 0])),
                "std": float(np.std(img[:, :, 0])),
            },
            "green": {
                "mean": float(np.mean(img[:, :, 1])),
                "std": float(np.std(img[:, :, 1])),
            },
            "blue": {
                "mean": float(np.mean(img[:, :, 2])),
                "std": float(np.std(img[:, :, 2])),
            },
        }
        return stats

    def _detect_soiling_rgb(self, img: np.ndarray) -> np.ndarray:
        """Detect soiling in RGB image."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect dark regions (soiling)
        threshold = np.percentile(gray, 30)
        soiling_mask = gray < threshold

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        soiling_mask = cv2.morphologyEx(soiling_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        return soiling_mask

    def _calculate_surface_quality(self, img: np.ndarray) -> float:
        """Calculate overall surface quality score."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate uniformity
        uniformity = 1.0 - min(1.0, np.std(gray) / (np.mean(gray) + 1e-10))

        # Calculate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(1.0, laplacian.var() / 1000)

        # Combined quality score
        quality = (uniformity + sharpness) / 2

        return float(quality)
