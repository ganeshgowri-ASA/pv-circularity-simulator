"""
Pytest configuration and fixtures.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple 640x640 RGB image
    img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    return img


@pytest.fixture
def sample_el_image():
    """Create a sample EL (grayscale) image."""
    # Create a simple 640x640 grayscale image with some variations
    img = Image.fromarray(np.random.randint(50, 200, (640, 640), dtype=np.uint8), mode="L")
    return img


@pytest.fixture
def sample_thermal_image():
    """Create a sample thermal image."""
    # Create a thermal image with hotspots
    img_array = np.random.randint(100, 150, (480, 640), dtype=np.uint8)
    # Add some hotspots
    img_array[100:150, 200:250] = 200  # Hotspot area
    img = Image.fromarray(img_array, mode="L")
    return img


@pytest.fixture
def mock_roboflow_response():
    """Mock Roboflow API response."""
    return {
        "predictions": [
            {
                "class": "crack",
                "confidence": 0.85,
                "x": 320,
                "y": 240,
                "width": 50,
                "height": 100,
            },
            {
                "class": "hotspot",
                "confidence": 0.92,
                "x": 100,
                "y": 100,
                "width": 30,
                "height": 30,
            },
        ],
        "image": {"width": 640, "height": 480},
    }


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for testing."""
    os.environ.setdefault("ROBOFLOW_API_KEY", "test_api_key")
    os.environ.setdefault("ROBOFLOW_WORKSPACE", "test_workspace")
    os.environ.setdefault("ROBOFLOW_PROJECT", "test_project")
    os.environ.setdefault("ALERT_ENABLED", "true")
    os.environ.setdefault("MONITORING_ENABLED", "true")
    yield
