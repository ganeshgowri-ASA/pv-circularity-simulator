"""
Base Forecaster Classes
=======================

Abstract base classes for forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    This class defines the interface that all forecasting models must implement,
    ensuring consistency across different forecasting approaches.

    Attributes:
        name: Name of the forecaster
        fitted: Whether the model has been fitted
        metadata: Additional metadata about the forecaster
    """

    def __init__(self, name: str = "BaseForecaster"):
        """
        Initialize the base forecaster.

        Args:
            name: Name identifier for the forecaster
        """
        self.name = name
        self.fitted = False
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> "BaseForecaster":
        """
        Fit the forecasting model to training data.

        Args:
            X: Training features
            y: Training target values
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Args:
            X: Features for prediction
            **kwargs: Additional keyword arguments

        Returns:
            Predicted values
        """
        pass

    def fit_predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and generate predictions in one step.

        Args:
            X: Training features
            y: Training target values
            **kwargs: Additional keyword arguments

        Returns:
            Predicted values for X
        """
        self.fit(X, y, **kwargs)
        return self.predict(X, **kwargs)

    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            True if the model is fitted, False otherwise
        """
        return self.fitted

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary containing model metadata
        """
        return self.metadata.copy()

    def __repr__(self) -> str:
        """String representation of the forecaster."""
        fitted_status = "fitted" if self.fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status='{fitted_status}')"
