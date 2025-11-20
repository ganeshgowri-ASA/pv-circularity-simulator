"""
Model training utilities with hyperparameter tuning and cross-validation.

This module provides tools for training forecasting models with automated
hyperparameter tuning, cross-validation, and model selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from optuna import Trial, create_study
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit

from pv_simulator.core.models import BaseForecaster
from pv_simulator.core.schemas import (
    ModelMetrics,
    ModelType,
    TimeSeriesData,
    TrainingConfig,
)
from pv_simulator.forecasting.metrics import MetricsCalculator
from pv_simulator.forecasting.statistical import ARIMAModel, SARIMAModel
from pv_simulator.forecasting.ml_forecaster import ProphetForecaster, XGBoostForecaster


class ModelTraining:
    """
    Comprehensive model training with hyperparameter optimization.

    Provides methods for cross-validation, hyperparameter tuning using Optuna,
    and automated model selection.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Initialize model training.

        Args:
            config: Training configuration (uses defaults if None)
        """
        self.config = config or TrainingConfig()
        self.metrics_calculator = MetricsCalculator()

    def cross_validation(
        self,
        forecaster: BaseForecaster,
        data: TimeSeriesData,
        horizon: int,
        n_splits: Optional[int] = None,
    ) -> List[ModelMetrics]:
        """
        Perform time series cross-validation.

        Uses expanding window cross-validation to evaluate model performance
        on multiple train/test splits.

        Args:
            forecaster: Forecaster instance to evaluate
            data: Time series data
            horizon: Forecast horizon
            n_splits: Number of CV splits (uses config if None)

        Returns:
            List of ModelMetrics for each split

        Example:
            >>> trainer = ModelTraining()
            >>> metrics_list = trainer.cross_validation(
            ...     ARIMAModel(), data, horizon=12, n_splits=5
            ... )
        """
        n_splits = n_splits or self.config.cross_validation_folds

        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics_list = []
        values = np.array(data.values)
        timestamps = data.timestamps

        for train_idx, test_idx in tscv.split(values):
            # Create train/test splits
            train_data = TimeSeriesData(
                timestamps=[timestamps[i] for i in train_idx],
                values=values[train_idx].tolist(),
                frequency=data.frequency,
            )

            test_data = TimeSeriesData(
                timestamps=[timestamps[i] for i in test_idx],
                values=values[test_idx].tolist(),
                frequency=data.frequency,
            )

            # Fit and predict
            try:
                forecaster.fit(train_data)
                forecast = forecaster.predict(horizon=len(test_idx))

                # Evaluate
                metrics = self.metrics_calculator.calculate_metrics(
                    test_data.values, forecast.predictions
                )
                metrics_list.append(metrics)

            except Exception as e:
                print(f"Warning: CV fold failed: {str(e)}")
                continue

        return metrics_list

    def hyperparameter_tuning(
        self,
        forecaster_class: Type[BaseForecaster],
        data: TimeSeriesData,
        horizon: int,
        n_trials: Optional[int] = None,
        metric: str = "rmse",
    ) -> Tuple[Dict[str, Any], ModelMetrics]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            forecaster_class: Forecaster class to tune
            data: Time series data
            horizon: Forecast horizon
            n_trials: Number of tuning trials (uses config if None)
            metric: Metric to optimize ('rmse', 'mae', 'mape')

        Returns:
            Tuple of (best_params, best_metrics)

        Example:
            >>> trainer = ModelTraining()
            >>> best_params, metrics = trainer.hyperparameter_tuning(
            ...     ARIMAModel, data, horizon=12, n_trials=100
            ... )
        """
        n_trials = n_trials or self.config.tuning_trials

        def objective(trial: Trial) -> float:
            """Optuna objective function."""
            # Define hyperparameter search space based on model type
            if forecaster_class == ARIMAModel:
                params = {
                    "order": (
                        trial.suggest_int("p", 0, 5),
                        trial.suggest_int("d", 0, 2),
                        trial.suggest_int("q", 0, 5),
                    )
                }
            elif forecaster_class == SARIMAModel:
                params = {
                    "order": (
                        trial.suggest_int("p", 0, 3),
                        trial.suggest_int("d", 0, 2),
                        trial.suggest_int("q", 0, 3),
                    ),
                    "seasonal_order": (
                        trial.suggest_int("P", 0, 2),
                        trial.suggest_int("D", 0, 1),
                        trial.suggest_int("Q", 0, 2),
                        trial.suggest_int("s", 4, 24),
                    ),
                }
            elif forecaster_class == XGBoostForecaster:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
            elif forecaster_class == ProphetForecaster:
                params = {
                    "seasonality_mode": trial.suggest_categorical(
                        "seasonality_mode", ["additive", "multiplicative"]
                    ),
                    "changepoint_prior_scale": trial.suggest_float(
                        "changepoint_prior_scale", 0.001, 0.5
                    ),
                    "seasonality_prior_scale": trial.suggest_float(
                        "seasonality_prior_scale", 0.01, 10.0
                    ),
                }
            else:
                params = {}

            # Create and evaluate forecaster
            try:
                forecaster = forecaster_class(**params)

                # Perform cross-validation
                metrics_list = self.cross_validation(forecaster, data, horizon, n_splits=3)

                if not metrics_list:
                    return float("inf")

                # Average the selected metric across folds
                avg_metric = np.mean([getattr(m, metric) for m in metrics_list])
                return avg_metric

            except Exception as e:
                # Return a large value if training fails
                print(f"Trial failed: {str(e)}")
                return float("inf")

        # Create and run study
        study = create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.config.random_state),
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params

        # Convert ARIMA/SARIMA params back to tuples
        if forecaster_class == ARIMAModel:
            best_params = {
                "order": (best_params["p"], best_params["d"], best_params["q"])
            }
        elif forecaster_class == SARIMAModel:
            best_params = {
                "order": (best_params["p"], best_params["d"], best_params["q"]),
                "seasonal_order": (
                    best_params["P"],
                    best_params["D"],
                    best_params["Q"],
                    best_params["s"],
                ),
            }

        # Train final model with best params
        best_forecaster = forecaster_class(**best_params)

        # Split data for final evaluation
        split_idx = int(len(data.values) * (1 - self.config.test_size))
        train_data = TimeSeriesData(
            timestamps=data.timestamps[:split_idx],
            values=data.values[:split_idx],
            frequency=data.frequency,
        )
        test_data = TimeSeriesData(
            timestamps=data.timestamps[split_idx:],
            values=data.values[split_idx:],
            frequency=data.frequency,
        )

        best_forecaster.fit(train_data)
        forecast = best_forecaster.predict(horizon=len(test_data.values))
        best_metrics = self.metrics_calculator.calculate_metrics(
            test_data.values, forecast.predictions
        )

        return best_params, best_metrics

    def model_selection(
        self,
        data: TimeSeriesData,
        horizon: int,
        models: Optional[List[Type[BaseForecaster]]] = None,
        metric: str = "rmse",
    ) -> Tuple[Type[BaseForecaster], Dict[str, Any], ModelMetrics]:
        """
        Select best model from multiple candidates.

        Compares different model types and selects the best performing one
        based on cross-validation.

        Args:
            data: Time series data
            horizon: Forecast horizon
            models: List of model classes to compare (uses defaults if None)
            metric: Metric to use for selection ('rmse', 'mae', 'mape')

        Returns:
            Tuple of (best_model_class, best_params, best_metrics)

        Example:
            >>> trainer = ModelTraining()
            >>> best_model, params, metrics = trainer.model_selection(
            ...     data, horizon=12
            ... )
        """
        # Default models to compare
        if models is None:
            models = [ARIMAModel, SARIMAModel, ProphetForecaster, XGBoostForecaster]

        results = []

        for model_class in models:
            print(f"Evaluating {model_class.__name__}...")

            try:
                if self.config.hyperparameter_tuning:
                    # Tune hyperparameters
                    best_params, metrics = self.hyperparameter_tuning(
                        model_class, data, horizon, n_trials=50, metric=metric
                    )
                else:
                    # Use default parameters
                    forecaster = model_class()
                    metrics_list = self.cross_validation(forecaster, data, horizon)

                    if not metrics_list:
                        continue

                    # Average metrics
                    metrics = ModelMetrics(
                        mae=np.mean([m.mae for m in metrics_list]),
                        rmse=np.mean([m.rmse for m in metrics_list]),
                        mape=np.mean([m.mape for m in metrics_list]),
                        r2=np.mean([m.r2 for m in metrics_list]),
                    )
                    best_params = {}

                results.append(
                    {
                        "model_class": model_class,
                        "params": best_params,
                        "metrics": metrics,
                        "score": getattr(metrics, metric),
                    }
                )

            except Exception as e:
                print(f"Failed to evaluate {model_class.__name__}: {str(e)}")
                continue

        if not results:
            raise ValueError("No models could be successfully evaluated")

        # Select best model (lowest error)
        best_result = min(results, key=lambda x: x["score"])

        return (
            best_result["model_class"],
            best_result["params"],
            best_result["metrics"],
        )

    def train_test_split(
        self, data: TimeSeriesData, test_size: Optional[float] = None
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        """
        Split time series data into train and test sets.

        Args:
            data: Time series data
            test_size: Fraction of data for testing (uses config if None)

        Returns:
            Tuple of (train_data, test_data)
        """
        test_size = test_size or self.config.test_size
        split_idx = int(len(data.values) * (1 - test_size))

        train_data = TimeSeriesData(
            timestamps=data.timestamps[:split_idx],
            values=data.values[:split_idx],
            frequency=data.frequency,
        )

        test_data = TimeSeriesData(
            timestamps=data.timestamps[split_idx:],
            values=data.values[split_idx:],
            frequency=data.frequency,
        )

        return train_data, test_data

    def evaluate_on_test_set(
        self,
        forecaster: BaseForecaster,
        train_data: TimeSeriesData,
        test_data: TimeSeriesData,
    ) -> ModelMetrics:
        """
        Train on training set and evaluate on test set.

        Args:
            forecaster: Forecaster to train
            train_data: Training data
            test_data: Test data

        Returns:
            ModelMetrics on test set
        """
        # Train model
        forecaster.fit(train_data)

        # Predict on test set
        forecast = forecaster.predict(horizon=len(test_data.values))

        # Evaluate
        metrics = self.metrics_calculator.calculate_metrics(
            test_data.values, forecast.predictions
        )

        return metrics

    def get_cv_summary(self, metrics_list: List[ModelMetrics]) -> Dict[str, float]:
        """
        Compute summary statistics from cross-validation results.

        Args:
            metrics_list: List of metrics from CV folds

        Returns:
            Dictionary with mean and std for each metric

        Example:
            >>> summary = trainer.get_cv_summary(metrics_list)
            >>> print(f"RMSE: {summary['rmse_mean']} ± {summary['rmse_std']}")
        """
        if not metrics_list:
            raise ValueError("Empty metrics list")

        summary = {}

        for metric in ["mae", "rmse", "mape", "r2"]:
            values = [getattr(m, metric) for m in metrics_list]
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)

        return summary
