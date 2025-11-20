"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_dates():
    """Generate sample date range for testing."""
    start_date = datetime(2020, 1, 1)
    return [start_date + timedelta(days=i) for i in range(100)]


@pytest.fixture(scope="session")
def sample_values():
    """Generate sample values for testing."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(100)) + 50


@pytest.fixture
def tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf

        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line(
        "markers", "requires_tensorflow: mark test as requiring TensorFlow"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    skip_tensorflow = pytest.mark.skip(reason="TensorFlow not available")

    for item in items:
        if "requires_tensorflow" in item.keywords:
            try:
                import tensorflow as tf
            except ImportError:
                item.add_marker(skip_tensorflow)
