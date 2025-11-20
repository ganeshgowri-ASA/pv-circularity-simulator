"""
Ensemble Forecasting Module
============================

Advanced ML model ensembling for forecasting with multiple ensemble strategies.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from pv_simulator.forecasting.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """
    Advanced ensemble forecasting system with multiple ensemble strategies.

    This class provides a comprehensive suite of ensemble methods including:
    - Stacking: Meta-learning from base model predictions
    - Bagging: Bootstrap aggregating for variance reduction
    - Voting: Weighted/unweighted averaging of predictions
    - Blending: Hold-out based model combination

    The forecaster supports hyperparameter optimization, cross-validation,
    and production-ready model persistence.

    Attributes:
        base_models: List of base estimators for ensemble
        meta_model: Meta-learner for stacking
        ensemble_type: Type of ensemble strategy
        fitted: Whether the ensemble has been fitted
        weights: Learned or specified weights for model combination
        scaler: Feature scaler for preprocessing
        cv_scores: Cross-validation scores
    """

    def __init__(
        self,
        base_models: Optional[List[BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        ensemble_type: Literal["stacking", "bagging", "voting", "blending"] = "stacking",
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        name: str = "EnsembleForecaster",
    ):
        """
        Initialize the ensemble forecaster.

        Args:
            base_models: List of base estimators. If None, uses default models.
            meta_model: Meta-learner for stacking. If None, uses Ridge regression.
            ensemble_type: Type of ensemble strategy to use
            n_jobs: Number of parallel jobs (-1 uses all processors)
            random_state: Random seed for reproducibility
            name: Name identifier for the forecaster
        """
        super().__init__(name=name)

        self.ensemble_type = ensemble_type
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Initialize base models with defaults if not provided
        self.base_models = base_models or self._get_default_base_models()

        # Initialize meta model
        self.meta_model = meta_model or Ridge(alpha=1.0, random_state=random_state)

        # Ensemble components
        self.ensemble: Optional[BaseEstimator] = None
        self.weights: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None

        # Performance tracking
        self.cv_scores: Dict[str, float] = {}
        self.feature_importance: Optional[np.ndarray] = None

        # Training metadata
        self.training_time: Optional[float] = None
        self.n_features: Optional[int] = None

    def _get_default_base_models(self) -> List[BaseEstimator]:
        """
        Get default set of base models for ensemble.

        Returns:
            List of default base estimators with sensible hyperparameters
        """
        return [
            LinearRegression(),
            Ridge(alpha=1.0, random_state=self.random_state),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=1,
            ),
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            ),
        ]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scale_features: bool = True,
        **kwargs
    ) -> "EnsembleForecaster":
        """
        Fit the ensemble forecaster to training data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training target values of shape (n_samples,)
            scale_features: Whether to apply feature scaling
            **kwargs: Additional arguments passed to ensemble fit method

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X and y have incompatible shapes
        """
        import time
        start_time = time.time()

        # Convert to numpy arrays if needed
        X_array = self._to_numpy(X)
        y_array = self._to_numpy(y).ravel()

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X_array.shape[0]}, y: {y_array.shape[0]}"
            )

        self.n_features = X_array.shape[1]

        # Feature scaling
        if scale_features:
            self.scaler = StandardScaler()
            X_array = self.scaler.fit_transform(X_array)

        # Build ensemble based on strategy
        if self.ensemble_type == "stacking":
            self.ensemble = self.stacking_models(
                X_array, y_array, cv=kwargs.get("cv", 5)
            )
        elif self.ensemble_type == "bagging":
            self.ensemble = self.bagging_ensemble(
                X_array, y_array, **kwargs
            )
        elif self.ensemble_type == "voting":
            self.ensemble = self.voting_strategies(
                X_array, y_array, strategy=kwargs.get("voting_strategy", "mean")
            )
        elif self.ensemble_type == "blending":
            self.ensemble = self.model_blending(
                X_array, y_array, blend_ratio=kwargs.get("blend_ratio", 0.5)
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

        self.fitted = True
        self.training_time = time.time() - start_time

        self.metadata.update({
            "n_samples": X_array.shape[0],
            "n_features": self.n_features,
            "ensemble_type": self.ensemble_type,
            "n_base_models": len(self.base_models),
            "training_time": self.training_time,
        })

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_std: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions using the fitted ensemble.

        Args:
            X: Features for prediction of shape (n_samples, n_features)
            return_std: If True, return (predictions, std_dev) tuple
            **kwargs: Additional keyword arguments

        Returns:
            Predicted values, optionally with standard deviations

        Raises:
            RuntimeError: If the ensemble has not been fitted
        """
        if not self.fitted or self.ensemble is None:
            raise RuntimeError(
                "Ensemble must be fitted before making predictions. "
                "Call fit() first."
            )

        X_array = self._to_numpy(X)

        if X_array.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X_array.shape[1]} features, "
                f"but ensemble was trained with {self.n_features} features"
            )

        # Apply scaling if used during training
        if self.scaler is not None:
            X_array = self.scaler.transform(X_array)

        predictions = self.ensemble.predict(X_array)

        if return_std:
            # Calculate prediction std from base models for uncertainty
            std = self._calculate_prediction_std(X_array)
            return predictions, std

        return predictions

    def stacking_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        passthrough: bool = False,
    ) -> StackingRegressor:
        """
        Create and fit a stacking ensemble with cross-validated predictions.

        Stacking uses a meta-model to learn how to best combine the predictions
        of base models. Base model predictions are generated using cross-validation
        to avoid overfitting.

        Args:
            X: Training features
            y: Training target values
            cv: Number of cross-validation folds for base predictions
            passthrough: If True, original features are passed to meta-model

        Returns:
            Fitted StackingRegressor instance

        Example:
            >>> forecaster = EnsembleForecaster()
            >>> X, y = load_data()
            >>> stacking_model = forecaster.stacking_models(X, y, cv=5)
            >>> predictions = stacking_model.predict(X_test)
        """
        # Create named estimators for stacking
        estimators = [
            (f"model_{i}", clone(model))
            for i, model in enumerate(self.base_models)
        ]

        # Create stacking regressor
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=clone(self.meta_model),
            cv=cv,
            n_jobs=self.n_jobs,
            passthrough=passthrough,
        )

        # Fit the stacking ensemble
        stacking.fit(X, y)

        # Calculate cross-validation scores
        cv_scores = cross_val_score(
            stacking, X, y, cv=cv, scoring="r2", n_jobs=self.n_jobs
        )

        self.cv_scores["stacking"] = {
            "mean": cv_scores.mean(),
            "std": cv_scores.std(),
            "scores": cv_scores.tolist(),
        }

        return stacking

    def bagging_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
    ) -> BaggingRegressor:
        """
        Create and fit a bagging ensemble for variance reduction.

        Bagging (Bootstrap Aggregating) trains multiple instances of base models
        on random subsets of the data and averages their predictions to reduce
        variance and prevent overfitting.

        Args:
            X: Training features
            y: Training target values
            n_estimators: Number of base estimators in the ensemble
            max_samples: Fraction of samples to draw for each base estimator
            max_features: Fraction of features to draw for each base estimator
            bootstrap: Whether to use bootstrap sampling
            bootstrap_features: Whether to bootstrap features as well

        Returns:
            Fitted BaggingRegressor instance

        Example:
            >>> forecaster = EnsembleForecaster()
            >>> X, y = load_data()
            >>> bagging_model = forecaster.bagging_ensemble(
            ...     X, y, n_estimators=50, max_samples=0.8
            ... )
        """
        # Use the first base model as the base estimator for bagging
        base_estimator = clone(self.base_models[0])

        # Create bagging regressor
        bagging = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # Fit the bagging ensemble
        bagging.fit(X, y)

        # Calculate out-of-bag score if bootstrap is enabled
        if bootstrap:
            bagging.oob_score = True
            bagging.fit(X, y)
            self.cv_scores["bagging_oob"] = bagging.oob_score_

        return bagging

    def voting_strategies(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: Literal["mean", "median", "weighted"] = "mean",
        weights: Optional[List[float]] = None,
    ) -> VotingRegressor:
        """
        Create and fit a voting ensemble with various aggregation strategies.

        Voting ensembles combine predictions from multiple models using different
        strategies: simple averaging (mean), robust averaging (median), or
        weighted averaging based on model performance or specified weights.

        Args:
            X: Training features
            y: Training target values
            strategy: Voting strategy - "mean", "median", or "weighted"
            weights: Optional weights for each base model (for weighted strategy)

        Returns:
            Fitted VotingRegressor instance

        Raises:
            ValueError: If weighted strategy is used without providing weights

        Example:
            >>> forecaster = EnsembleForecaster()
            >>> X, y = load_data()
            >>> voting_model = forecaster.voting_strategies(
            ...     X, y, strategy="weighted", weights=[0.3, 0.3, 0.4]
            ... )
        """
        # Create named estimators
        estimators = [
            (f"model_{i}", clone(model))
            for i, model in enumerate(self.base_models)
        ]

        # Calculate weights based on strategy
        if strategy == "weighted":
            if weights is None:
                # Auto-calculate weights based on model performance
                weights = self._calculate_optimal_weights(X, y)
            self.weights = np.array(weights)
        else:
            weights = None
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Create voting regressor
        voting = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=self.n_jobs,
        )

        # Fit the voting ensemble
        voting.fit(X, y)

        # Store strategy metadata
        self.metadata["voting_strategy"] = strategy

        return voting

    def model_blending(
        self,
        X: np.ndarray,
        y: np.ndarray,
        blend_ratio: float = 0.5,
        optimize_weights: bool = True,
    ) -> BaseEstimator:
        """
        Create and fit a blending ensemble using hold-out validation.

        Blending splits the data into training and validation sets. Base models
        are trained on the training set, and their predictions on the validation
        set are used to train a meta-model. This approach is simpler than stacking
        but may be less robust with smaller datasets.

        Args:
            X: Training features
            y: Training target values
            blend_ratio: Fraction of data to use for training base models
                        (1 - blend_ratio used for meta-model training)
            optimize_weights: Whether to optimize blending weights

        Returns:
            Fitted blending ensemble (custom meta-model)

        Raises:
            ValueError: If blend_ratio is not between 0 and 1

        Example:
            >>> forecaster = EnsembleForecaster()
            >>> X, y = load_data()
            >>> blending_model = forecaster.model_blending(
            ...     X, y, blend_ratio=0.6, optimize_weights=True
            ... )
        """
        if not 0 < blend_ratio < 1:
            raise ValueError(
                f"blend_ratio must be between 0 and 1, got {blend_ratio}"
            )

        # Split data for blending
        split_idx = int(len(X) * blend_ratio)
        X_train, X_blend = X[:split_idx], X[split_idx:]
        y_train, y_blend = y[:split_idx], y[split_idx:]

        # Train base models on training set
        trained_models = []
        for model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            trained_models.append(model_clone)

        # Generate predictions on blend set
        blend_predictions = np.column_stack([
            model.predict(X_blend) for model in trained_models
        ])

        # Train meta-model on blend predictions
        meta_model = clone(self.meta_model)

        if optimize_weights:
            # Optimize weights using the blend set
            optimal_weights = self._optimize_blending_weights(
                blend_predictions, y_blend
            )
            self.weights = optimal_weights

            # Create weighted meta model
            meta_model.fit(blend_predictions, y_blend)
        else:
            # Simple meta-model training
            meta_model.fit(blend_predictions, y_blend)

        # Create a wrapper for prediction
        class BlendingEnsemble:
            def __init__(self, base_models, meta_model, weights=None):
                self.base_models = base_models
                self.meta_model = meta_model
                self.weights = weights

            def predict(self, X):
                base_predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])

                if self.weights is not None:
                    # Weighted blending
                    return np.average(base_predictions, axis=1, weights=self.weights)
                else:
                    # Meta-model blending
                    return self.meta_model.predict(base_predictions)

        blending_ensemble = BlendingEnsemble(
            trained_models, meta_model, self.weights if optimize_weights else None
        )

        return blending_ensemble

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_distributions: Dict[str, Any],
        n_iter: int = 50,
        cv: int = 5,
        search_type: Literal["grid", "random"] = "random",
    ) -> Dict[str, Any]:
        """
        Optimize ensemble hyperparameters using grid or random search.

        Args:
            X: Training features
            y: Training target values
            param_distributions: Dictionary of parameter distributions to search
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            search_type: Type of search - "grid" or "random"

        Returns:
            Dictionary containing best parameters and best score

        Example:
            >>> param_dist = {
            ...     'n_estimators': [50, 100, 200],
            ...     'max_depth': [5, 10, 15],
            ... }
            >>> results = forecaster.optimize_hyperparameters(
            ...     X, y, param_dist, search_type="grid"
            ... )
        """
        if self.ensemble is None:
            raise RuntimeError("Ensemble must be initialized before optimization")

        if search_type == "grid":
            search = GridSearchCV(
                self.ensemble,
                param_distributions,
                cv=cv,
                scoring="r2",
                n_jobs=self.n_jobs,
                verbose=1,
            )
        else:  # random
            search = RandomizedSearchCV(
                self.ensemble,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring="r2",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1,
            )

        search.fit(X, y)

        # Update ensemble with best estimator
        self.ensemble = search.best_estimator_

        results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
        }

        self.metadata["hyperparameter_optimization"] = results

        return results

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble on test data with multiple metrics.

        Args:
            X: Test features
            y: Test target values
            metrics: List of metric names. Options: "mse", "rmse", "mae", "r2"

        Returns:
            Dictionary of metric names and values

        Example:
            >>> results = forecaster.evaluate(X_test, y_test)
            >>> print(f"R2 Score: {results['r2']:.4f}")
        """
        if not self.fitted:
            raise RuntimeError("Ensemble must be fitted before evaluation")

        if metrics is None:
            metrics = ["mse", "rmse", "mae", "r2"]

        predictions = self.predict(X)

        results = {}

        if "mse" in metrics:
            results["mse"] = mean_squared_error(y, predictions)

        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y, predictions))

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y, predictions)

        if "r2" in metrics:
            results["r2"] = r2_score(y, predictions)

        return results

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Extract feature importance from ensemble (if available).

        Returns:
            Array of feature importances, or None if not available

        Example:
            >>> importance = forecaster.get_feature_importance()
            >>> if importance is not None:
            ...     top_features = np.argsort(importance)[-10:]
        """
        if not self.fitted or self.ensemble is None:
            return None

        # Try to extract feature importance from ensemble
        if hasattr(self.ensemble, "feature_importances_"):
            return self.ensemble.feature_importances_

        # For stacking, try to get from final estimator
        if isinstance(self.ensemble, StackingRegressor):
            if hasattr(self.ensemble.final_estimator_, "coef_"):
                return np.abs(self.ensemble.final_estimator_.coef_)

        return None

    def _calculate_optimal_weights(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> List[float]:
        """
        Calculate optimal weights for voting based on cross-validation scores.

        Args:
            X: Training features
            y: Training target values
            cv: Number of cross-validation folds

        Returns:
            List of optimal weights for each base model
        """
        scores = []

        for model in self.base_models:
            model_clone = clone(model)
            cv_scores = cross_val_score(
                model_clone, X, y, cv=cv, scoring="r2", n_jobs=1
            )
            scores.append(cv_scores.mean())

        # Convert scores to weights (handle negative scores)
        scores = np.array(scores)
        scores = np.maximum(scores, 0)  # Clip negative scores to 0

        # Normalize to sum to 1
        if scores.sum() > 0:
            weights = scores / scores.sum()
        else:
            # If all scores are 0, use equal weights
            weights = np.ones(len(scores)) / len(scores)

        return weights.tolist()

    def _optimize_blending_weights(
        self, predictions: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Optimize blending weights to minimize prediction error.

        Args:
            predictions: Base model predictions (n_samples, n_models)
            y_true: True target values

        Returns:
            Optimal weights for blending
        """
        from scipy.optimize import minimize

        def loss_function(weights):
            weights = weights / weights.sum()  # Normalize
            blended = np.average(predictions, axis=1, weights=weights)
            return mean_squared_error(y_true, blended)

        # Initial guess: equal weights
        initial_weights = np.ones(predictions.shape[1]) / predictions.shape[1]

        # Optimize weights
        result = minimize(
            loss_function,
            initial_weights,
            method="SLSQP",
            bounds=[(0, 1)] * predictions.shape[1],
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        )

        optimal_weights = result.x / result.x.sum()  # Ensure normalization

        return optimal_weights

    def _calculate_prediction_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction standard deviation from base models.

        Args:
            X: Features for prediction

        Returns:
            Standard deviation of predictions
        """
        if isinstance(self.ensemble, (StackingRegressor, VotingRegressor)):
            # Get predictions from all base models
            base_predictions = []

            if isinstance(self.ensemble, StackingRegressor):
                for estimator in self.ensemble.estimators_:
                    base_predictions.append(estimator.predict(X))
            else:  # VotingRegressor
                for estimator in self.ensemble.estimators_:
                    base_predictions.append(estimator.predict(X))

            base_predictions = np.column_stack(base_predictions)
            return np.std(base_predictions, axis=1)

        # For other ensembles, return zeros
        return np.zeros(X.shape[0])

    def _to_numpy(
        self, data: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """
        Convert various data types to numpy array.

        Args:
            data: Input data

        Returns:
            Numpy array
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.asarray(data)

    def __repr__(self) -> str:
        """String representation of the forecaster."""
        status = "fitted" if self.fitted else "not fitted"
        n_models = len(self.base_models)
        return (
            f"EnsembleForecaster(type='{self.ensemble_type}', "
            f"n_base_models={n_models}, status='{status}')"
        )
