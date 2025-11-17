"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


@pytest.fixture
def random_state():
    """Fixed random state for reproducibility."""
    return 42


@pytest.fixture
def sample_regression_data(random_state):
    """
    Generate sample regression data for testing.

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10.0,
        random_state=random_state,
    )
    return X, y


@pytest.fixture
def sample_time_series_data(random_state):
    """
    Generate sample time series data for forecasting tests.

    Returns:
        Tuple of (X, y) representing time series features and targets
    """
    np.random.seed(random_state)

    # Create time-based features
    n_samples = 200
    time_index = np.arange(n_samples)

    # Create features with trend and seasonality
    trend = 0.5 * time_index
    seasonality = 10 * np.sin(2 * np.pi * time_index / 50)
    noise = np.random.normal(0, 5, n_samples)

    y = trend + seasonality + noise

    # Create lagged features
    X = np.column_stack([
        time_index,
        np.sin(2 * np.pi * time_index / 50),
        np.cos(2 * np.pi * time_index / 50),
        np.sin(2 * np.pi * time_index / 25),
        np.cos(2 * np.pi * time_index / 25),
    ])

    return X, y


@pytest.fixture
def sample_base_models(random_state):
    """
    Create a list of sample base models for ensemble testing.

    Returns:
        List of scikit-learn estimators
    """
    return [
        LinearRegression(),
        Ridge(alpha=1.0, random_state=random_state),
        DecisionTreeRegressor(max_depth=5, random_state=random_state),
    ]


@pytest.fixture
def sample_dataframe_data(sample_regression_data):
    """
    Convert regression data to pandas DataFrame format.

    Returns:
        Tuple of (X_df, y_series) as pandas objects
    """
    X, y = sample_regression_data

    X_df = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    )

    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def train_test_split_data(sample_regression_data):
    """
    Split sample data into train and test sets.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X, y = sample_regression_data

    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test
