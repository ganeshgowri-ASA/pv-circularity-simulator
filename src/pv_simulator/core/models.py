"""
Core domain models for the PV simulator.

This module defines the base classes and domain models for PV systems,
forecasting, and time series analysis.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from pv_simulator.core.schemas import (
    ForecastConfig,
    ForecastResult,
    ModelMetrics,
    ModelType,
    TimeSeriesData,
)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    This class defines the interface that all forecasting models must implement,
    ensuring consistency across different model types.
    """

    def __init__(self, model_type: ModelType, **kwargs: Any) -> None:
        """
        Initialize the base forecaster.

        Args:
            model_type: Type of forecasting model
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.model: Optional[Any] = None
        self.is_fitted = False
        self.training_history: List[Dict[str, Any]] = []
        self.params = kwargs

    @abstractmethod
    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "BaseForecaster":
        """
        Fit the forecasting model to the data.

        Args:
            data: Time series data to fit
            exogenous: Optional exogenous variables
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate forecasts for the specified horizon.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables for forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional prediction parameters

        Returns:
            ForecastResult containing predictions and confidence intervals
        """
        pass

    @abstractmethod
    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """
        Evaluate model performance.

        Args:
            actual: Actual observed values
            predicted: Predicted values

        Returns:
            ModelMetrics containing evaluation metrics
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseForecaster":
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        self.params.update(params)
        return self

    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk.

        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("Model saving not implemented for this forecaster")

    def load_model(self, filepath: str) -> None:
        """
        Load a fitted model from disk.

        Args:
            filepath: Path to load the model from
        """
        raise NotImplementedError("Model loading not implemented for this forecaster")


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecasting model that combines multiple forecasters.

    This class provides methods to combine predictions from multiple models
    using various ensemble strategies (averaging, weighted averaging, stacking, etc.).
    """

    def __init__(
        self,
        forecasters: List[BaseForecaster],
        weights: Optional[List[float]] = None,
        ensemble_method: str = "average",
    ) -> None:
        """
        Initialize the ensemble forecaster.

        Args:
            forecasters: List of base forecasters to ensemble
            weights: Optional weights for weighted averaging
            ensemble_method: Method to combine forecasts ('average', 'weighted', 'median')
        """
        super().__init__(ModelType.ENSEMBLE)
        self.forecasters = forecasters
        self.weights = weights
        self.ensemble_method = ensemble_method

        if weights is not None and len(weights) != len(forecasters):
            raise ValueError("Number of weights must match number of forecasters")

    def fit(
        self,
        data: TimeSeriesData,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "EnsembleForecaster":
        """
        Fit all base forecasters.

        Args:
            data: Time series data to fit
            exogenous: Optional exogenous variables
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        for forecaster in self.forecasters:
            forecaster.fit(data, exogenous, **kwargs)

        self.is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        exogenous: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> ForecastResult:
        """
        Generate ensemble forecasts.

        Args:
            horizon: Number of periods to forecast
            exogenous: Optional exogenous variables
            confidence_level: Confidence level for intervals
            **kwargs: Additional parameters

        Returns:
            ForecastResult with ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get predictions from all forecasters
        predictions = [
            f.predict(horizon, exogenous, confidence_level, **kwargs)
            for f in self.forecasters
        ]

        # Combine predictions based on ensemble method
        if self.ensemble_method == "average":
            combined_preds = np.mean(
                [p.predictions for p in predictions], axis=0
            ).tolist()
        elif self.ensemble_method == "weighted":
            if self.weights is None:
                raise ValueError("Weights required for weighted ensemble")
            combined_preds = np.average(
                [p.predictions for p in predictions], axis=0, weights=self.weights
            ).tolist()
        elif self.ensemble_method == "median":
            combined_preds = np.median(
                [p.predictions for p in predictions], axis=0
            ).tolist()
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        # Use the first prediction's timestamps
        timestamps = predictions[0].timestamps

        # Combine confidence intervals
        lower_bounds = [p.lower_bound for p in predictions if p.lower_bound is not None]
        upper_bounds = [p.upper_bound for p in predictions if p.upper_bound is not None]

        lower_bound = (
            np.min(lower_bounds, axis=0).tolist() if lower_bounds else None
        )
        upper_bound = (
            np.max(upper_bounds, axis=0).tolist() if upper_bounds else None
        )

        return ForecastResult(
            model_type=self.model_type,
            timestamps=timestamps,
            predictions=combined_preds,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={"ensemble_method": self.ensemble_method},
        )

    def evaluate(
        self, actual: TimeSeriesData, predicted: ForecastResult
    ) -> ModelMetrics:
        """
        Evaluate ensemble performance.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            ModelMetrics with evaluation results
        """
        # Use the first forecaster's evaluate method
        return self.forecasters[0].evaluate(actual, predicted)
