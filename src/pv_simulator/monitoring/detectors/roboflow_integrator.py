"""
Roboflow AI Integration for PV Defect Detection.

This module provides integration with Roboflow's computer vision API for
detecting defects in photovoltaic panels through image analysis.
"""

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, Field

from pv_simulator.config import get_settings

logger = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for detected objects.

    Attributes:
        x: Center X coordinate
        y: Center Y coordinate
        width: Box width
        height: Box height
    """

    x: float = Field(..., description="Center X coordinate")
    y: float = Field(..., description="Center Y coordinate")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """
        Convert to (x1, y1, x2, y2) format.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x1 = self.x - self.width / 2
        y1 = self.y - self.height / 2
        x2 = self.x + self.width / 2
        y2 = self.y + self.height / 2
        return (x1, y1, x2, y2)


class Detection(BaseModel):
    """
    Single detection result from Roboflow.

    Attributes:
        class_name: Detected defect class
        confidence: Detection confidence score (0-1)
        bbox: Bounding box coordinates
        image_width: Original image width
        image_height: Original image height
    """

    class_name: str = Field(..., description="Detected class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box")
    image_width: int = Field(..., description="Image width")
    image_height: int = Field(..., description="Image height")


class InferenceResult(BaseModel):
    """
    Complete inference result for a single image.

    Attributes:
        image_id: Unique identifier for the image
        detections: List of detections found
        inference_time_ms: Inference time in milliseconds
        model_version: Model version used
    """

    image_id: str = Field(..., description="Image identifier")
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(..., description="Inference time")
    model_version: str = Field(..., description="Model version")


class RoboflowIntegrator:
    """
    Roboflow API integration for defect detection.

    This class handles communication with Roboflow's inference API,
    including image preprocessing, batch processing, and real-time detection.

    Attributes:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        model_version: Model version number
        confidence_threshold: Minimum confidence for detections
        overlap_threshold: IoU threshold for NMS
        max_batch_size: Maximum batch size
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        model_version: Optional[int] = None,
    ):
        """
        Initialize Roboflow integrator.

        Args:
            api_key: Roboflow API key (defaults to settings)
            workspace: Workspace name (defaults to settings)
            project: Project name (defaults to settings)
            model_version: Model version (defaults to settings)
        """
        settings = get_settings()
        self.api_key = api_key or settings.roboflow.api_key.get_secret_value()
        self.workspace = workspace or settings.roboflow.workspace
        self.project = project or settings.roboflow.project
        self.model_version = model_version or settings.roboflow.model_version
        self.confidence_threshold = settings.roboflow.confidence_threshold
        self.overlap_threshold = settings.roboflow.overlap_threshold
        self.max_batch_size = settings.roboflow.max_batch_size

        self.base_url = "https://detect.roboflow.com"
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"Initialized RoboflowIntegrator for {self.workspace}/{self.project}/"
            f"{self.model_version}"
        )

    def model_deployment(self) -> Dict[str, Any]:
        """
        Get model deployment information.

        This method retrieves metadata about the deployed model including
        version, classes, and deployment status.

        Returns:
            Dictionary containing model deployment information

        Example:
            >>> integrator = RoboflowIntegrator()
            >>> info = integrator.model_deployment()
            >>> print(info['classes'])
            ['crack', 'hotspot', 'delamination', 'soiling']
        """
        try:
            url = f"{self.base_url}/{self.workspace}/{self.project}/{self.model_version}"
            params = {"api_key": self.api_key, "format": "json"}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            deployment_info = {
                "workspace": self.workspace,
                "project": self.project,
                "version": self.model_version,
                "classes": data.get("model", {}).get("classes", []),
                "status": "active",
                "endpoint": url,
            }

            logger.info(f"Model deployment info: {deployment_info}")
            return deployment_info

        except Exception as e:
            logger.error(f"Failed to get model deployment info: {e}")
            return {
                "workspace": self.workspace,
                "project": self.project,
                "version": self.model_version,
                "status": "error",
                "error": str(e),
            }

    def image_preprocessing(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        denoise: bool = False,
        enhance_contrast: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image for defect detection.

        Applies various preprocessing techniques to improve detection accuracy
        including resizing, normalization, denoising, and contrast enhancement.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            resize: Target size as (width, height), None to keep original
            normalize: Whether to normalize pixel values
            denoise: Whether to apply denoising
            enhance_contrast: Whether to enhance contrast using CLAHE

        Returns:
            Tuple of (preprocessed image array, metadata dict)

        Example:
            >>> integrator = RoboflowIntegrator()
            >>> img, metadata = integrator.image_preprocessing("panel.jpg", resize=(640, 640))
            >>> print(metadata['original_size'])
            (1920, 1080)
        """
        # Load image
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Failed to load image from {image}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        metadata = {
            "original_size": img_array.shape[:2],
            "original_dtype": str(img_array.dtype),
        }

        # Resize if requested
        if resize:
            img_array = cv2.resize(img_array, resize, interpolation=cv2.INTER_AREA)
            metadata["resized"] = True
            metadata["target_size"] = resize
        else:
            metadata["resized"] = False

        # Denoise
        if denoise:
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            metadata["denoised"] = True

        # Enhance contrast using CLAHE
        if enhance_contrast:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_array = cv2.merge([l, a, b])
            img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
            metadata["contrast_enhanced"] = True

        # Normalize
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
            metadata["normalized"] = True

        metadata["final_size"] = img_array.shape[:2]
        metadata["final_dtype"] = str(img_array.dtype)

        logger.debug(f"Preprocessed image: {metadata}")
        return img_array, metadata

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string for API transmission.

        Args:
            image: Image array (RGB, values 0-255 or 0-1)

        Returns:
            Base64 encoded image string
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Encode to JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        # Base64 encode
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return encoded

    def real_time_detection(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        preprocess: bool = True,
    ) -> InferenceResult:
        """
        Perform real-time defect detection on a single image.

        This method provides low-latency detection for real-time applications.

        Args:
            image: Input image
            preprocess: Whether to apply preprocessing

        Returns:
            InferenceResult containing detections

        Example:
            >>> integrator = RoboflowIntegrator()
            >>> result = integrator.real_time_detection("panel.jpg")
            >>> for detection in result.detections:
            ...     print(f"{detection.class_name}: {detection.confidence:.2f}")
        """
        import time

        start_time = time.time()

        # Preprocess if requested
        if preprocess:
            image_array, _ = self.image_preprocessing(image, resize=(640, 640))
        else:
            if isinstance(image, (str, Path)):
                image_array = cv2.imread(str(image))
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

        # Encode image
        encoded_image = self._encode_image(image_array)

        # Make API request
        url = f"{self.base_url}/{self.workspace}/{self.project}/{self.model_version}"
        params = {
            "api_key": self.api_key,
            "confidence": self.confidence_threshold,
            "overlap": self.overlap_threshold,
        }

        try:
            response = requests.post(
                url,
                params=params,
                data=encoded_image,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            response.raise_for_status()
            result_data = response.json()

            # Parse detections
            detections = []
            for pred in result_data.get("predictions", []):
                detection = Detection(
                    class_name=pred["class"],
                    confidence=pred["confidence"],
                    bbox=BoundingBox(
                        x=pred["x"], y=pred["y"], width=pred["width"], height=pred["height"]
                    ),
                    image_width=result_data.get("image", {}).get("width", 0),
                    image_height=result_data.get("image", {}).get("height", 0),
                )
                detections.append(detection)

            inference_time = (time.time() - start_time) * 1000

            result = InferenceResult(
                image_id=str(image) if isinstance(image, (str, Path)) else "unknown",
                detections=detections,
                inference_time_ms=inference_time,
                model_version=str(self.model_version),
            )

            logger.info(
                f"Real-time detection: {len(detections)} detections in {inference_time:.2f}ms"
            )
            return result

        except Exception as e:
            logger.error(f"Real-time detection failed: {e}")
            raise

    def batch_inference(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        preprocess: bool = True,
    ) -> List[InferenceResult]:
        """
        Perform batch inference on multiple images.

        This method processes multiple images efficiently using batching
        to reduce API calls and improve throughput.

        Args:
            images: List of input images
            preprocess: Whether to apply preprocessing

        Returns:
            List of InferenceResult objects

        Example:
            >>> integrator = RoboflowIntegrator()
            >>> images = ["panel1.jpg", "panel2.jpg", "panel3.jpg"]
            >>> results = integrator.batch_inference(images)
            >>> print(f"Processed {len(results)} images")
        """
        results = []

        # Process in batches
        for i in range(0, len(images), self.max_batch_size):
            batch = images[i : i + self.max_batch_size]
            logger.info(f"Processing batch {i // self.max_batch_size + 1}: {len(batch)} images")

            for image in batch:
                try:
                    result = self.real_time_detection(image, preprocess=preprocess)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image {image}: {e}")
                    # Create empty result for failed image
                    results.append(
                        InferenceResult(
                            image_id=str(image),
                            detections=[],
                            inference_time_ms=0.0,
                            model_version=str(self.model_version),
                        )
                    )

        logger.info(f"Batch inference complete: {len(results)} results")
        return results

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def async_real_time_detection(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        preprocess: bool = True,
    ) -> InferenceResult:
        """
        Async version of real-time detection.

        Args:
            image: Input image
            preprocess: Whether to apply preprocessing

        Returns:
            InferenceResult containing detections
        """
        import time

        start_time = time.time()

        # Preprocess if requested
        if preprocess:
            image_array, _ = self.image_preprocessing(image, resize=(640, 640))
        else:
            if isinstance(image, (str, Path)):
                image_array = cv2.imread(str(image))
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

        # Encode image
        encoded_image = self._encode_image(image_array)

        # Make async API request
        url = f"{self.base_url}/{self.workspace}/{self.project}/{self.model_version}"
        params = {
            "api_key": self.api_key,
            "confidence": self.confidence_threshold,
            "overlap": self.overlap_threshold,
        }

        session = await self._get_session()

        try:
            async with session.post(
                url,
                params=params,
                data=encoded_image,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                result_data = await response.json()

                # Parse detections
                detections = []
                for pred in result_data.get("predictions", []):
                    detection = Detection(
                        class_name=pred["class"],
                        confidence=pred["confidence"],
                        bbox=BoundingBox(
                            x=pred["x"], y=pred["y"], width=pred["width"], height=pred["height"]
                        ),
                        image_width=result_data.get("image", {}).get("width", 0),
                        image_height=result_data.get("image", {}).get("height", 0),
                    )
                    detections.append(detection)

                inference_time = (time.time() - start_time) * 1000

                return InferenceResult(
                    image_id=str(image) if isinstance(image, (str, Path)) else "unknown",
                    detections=detections,
                    inference_time_ms=inference_time,
                    model_version=str(self.model_version),
                )

        except Exception as e:
            logger.error(f"Async detection failed: {e}")
            raise

    async def async_batch_inference(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        preprocess: bool = True,
    ) -> List[InferenceResult]:
        """
        Async batch inference for improved performance.

        Args:
            images: List of input images
            preprocess: Whether to apply preprocessing

        Returns:
            List of InferenceResult objects
        """
        tasks = [self.async_real_time_detection(img, preprocess) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process image {images[i]}: {result}")
                processed_results.append(
                    InferenceResult(
                        image_id=str(images[i]),
                        detections=[],
                        inference_time_ms=0.0,
                        model_version=str(self.model_version),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def close(self) -> None:
        """Close async session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __enter__(self) -> "RoboflowIntegrator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    async def __aenter__(self) -> "RoboflowIntegrator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
