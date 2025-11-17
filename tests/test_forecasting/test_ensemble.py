"""
Tests for Ensemble Forecasting Module
======================================

Comprehensive tests for the EnsembleForecaster class.
"""

import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from pv_simulator.forecasting.ensemble import EnsembleForecaster


class TestEnsembleForecasterInitialization:
    """Test ensemble forecaster initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        forecaster = EnsembleForecaster()

        assert forecaster.name == "EnsembleForecaster"
        assert forecaster.ensemble_type == "stacking"
        assert forecaster.fitted is False
        assert len(forecaster.base_models) == 4  # Default models
        assert forecaster.meta_model is not None

    def test_custom_initialization(self, sample_base_models):
        """Test initialization with custom parameters."""
        meta_model = Lasso(alpha=0.1)

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            meta_model=meta_model,
            ensemble_type="voting",
            name="CustomForecaster",
        )

        assert forecaster.name == "CustomForecaster"
        assert forecaster.ensemble_type == "voting"
        assert len(forecaster.base_models) == 3

    def test_ensemble_type_options(self):
        """Test different ensemble type options."""
        for ensemble_type in ["stacking", "bagging", "voting", "blending"]:
            forecaster = EnsembleForecaster(ensemble_type=ensemble_type)
            assert forecaster.ensemble_type == ensemble_type


class TestEnsembleForecasterFitting:
    """Test ensemble forecaster fitting methods."""

    def test_fit_stacking(self, sample_regression_data, sample_base_models):
        """Test fitting with stacking ensemble."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        result = forecaster.fit(X, y)

        assert result is forecaster  # Method chaining
        assert forecaster.fitted is True
        assert isinstance(forecaster.ensemble, StackingRegressor)
        assert "stacking" in forecaster.cv_scores

    def test_fit_bagging(self, sample_regression_data, sample_base_models):
        """Test fitting with bagging ensemble."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="bagging",
        )

        forecaster.fit(X, y, n_estimators=20, max_samples=0.8)

        assert forecaster.fitted is True
        assert isinstance(forecaster.ensemble, BaggingRegressor)

    def test_fit_voting(self, sample_regression_data, sample_base_models):
        """Test fitting with voting ensemble."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="voting",
        )

        forecaster.fit(X, y, voting_strategy="mean")

        assert forecaster.fitted is True
        assert isinstance(forecaster.ensemble, VotingRegressor)

    def test_fit_blending(self, sample_regression_data, sample_base_models):
        """Test fitting with blending ensemble."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="blending",
        )

        forecaster.fit(X, y, blend_ratio=0.6)

        assert forecaster.fitted is True
        assert forecaster.ensemble is not None

    def test_fit_with_dataframe(self, sample_dataframe_data, sample_base_models):
        """Test fitting with pandas DataFrame input."""
        X_df, y_series = sample_dataframe_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_df, y_series)

        assert forecaster.fitted is True

    def test_fit_with_scaling(self, sample_regression_data, sample_base_models):
        """Test fitting with feature scaling."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X, y, scale_features=True)

        assert forecaster.scaler is not None
        assert forecaster.fitted is True

    def test_fit_without_scaling(self, sample_regression_data, sample_base_models):
        """Test fitting without feature scaling."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X, y, scale_features=False)

        assert forecaster.scaler is None
        assert forecaster.fitted is True

    def test_fit_updates_metadata(self, sample_regression_data, sample_base_models):
        """Test that fitting updates metadata correctly."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X, y)

        assert "n_samples" in forecaster.metadata
        assert "n_features" in forecaster.metadata
        assert "ensemble_type" in forecaster.metadata
        assert forecaster.metadata["n_samples"] == X.shape[0]
        assert forecaster.metadata["n_features"] == X.shape[1]

    def test_fit_invalid_shapes(self, sample_base_models):
        """Test fitting with mismatched X and y shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(80)  # Wrong size

        forecaster = EnsembleForecaster(base_models=sample_base_models)

        with pytest.raises(ValueError, match="same number of samples"):
            forecaster.fit(X, y)


class TestEnsembleForecasterPrediction:
    """Test ensemble forecaster prediction methods."""

    def test_predict_basic(self, train_test_split_data, sample_base_models):
        """Test basic prediction functionality."""
        X_train, X_test, y_train, y_test = train_test_split_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)
        predictions = forecaster.predict(X_test)

        assert predictions.shape == y_test.shape
        assert isinstance(predictions, np.ndarray)

    def test_predict_with_std(self, train_test_split_data, sample_base_models):
        """Test prediction with standard deviation."""
        X_train, X_test, y_train, y_test = train_test_split_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)
        predictions, std = forecaster.predict(X_test, return_std=True)

        assert predictions.shape == y_test.shape
        assert std.shape == y_test.shape
        assert np.all(std >= 0)  # Standard deviation should be non-negative

    def test_predict_without_fit(self, sample_regression_data):
        """Test that prediction fails without fitting."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster()

        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(X)

    def test_predict_wrong_features(self, train_test_split_data, sample_base_models):
        """Test prediction with wrong number of features."""
        X_train, X_test, y_train, y_test = train_test_split_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)

        # Create test data with wrong number of features
        X_wrong = np.random.randn(10, X_test.shape[1] + 5)

        with pytest.raises(ValueError, match="features"):
            forecaster.predict(X_wrong)

    def test_fit_predict(self, sample_regression_data, sample_base_models):
        """Test fit_predict convenience method."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        predictions = forecaster.fit_predict(X, y)

        assert forecaster.fitted is True
        assert predictions.shape == y.shape


class TestStackingModels:
    """Test stacking_models method specifically."""

    def test_stacking_basic(self, sample_regression_data, sample_base_models):
        """Test basic stacking functionality."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        stacking_model = forecaster.stacking_models(X, y, cv=3)

        assert isinstance(stacking_model, StackingRegressor)
        assert len(stacking_model.estimators) == len(sample_base_models)

    def test_stacking_with_passthrough(self, sample_regression_data, sample_base_models):
        """Test stacking with passthrough enabled."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        stacking_model = forecaster.stacking_models(X, y, cv=3, passthrough=True)

        assert stacking_model.passthrough is True

    def test_stacking_cv_scores(self, sample_regression_data, sample_base_models):
        """Test that stacking calculates CV scores."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        forecaster.stacking_models(X, y, cv=5)

        assert "stacking" in forecaster.cv_scores
        assert "mean" in forecaster.cv_scores["stacking"]
        assert "std" in forecaster.cv_scores["stacking"]


class TestBaggingEnsemble:
    """Test bagging_ensemble method specifically."""

    def test_bagging_basic(self, sample_regression_data, sample_base_models):
        """Test basic bagging functionality."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        bagging_model = forecaster.bagging_ensemble(X, y, n_estimators=10)

        assert isinstance(bagging_model, BaggingRegressor)
        assert bagging_model.n_estimators == 10

    def test_bagging_with_bootstrap(self, sample_regression_data, sample_base_models):
        """Test bagging with bootstrap enabled."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        bagging_model = forecaster.bagging_ensemble(
            X, y, n_estimators=20, bootstrap=True
        )

        assert bagging_model.bootstrap is True

    def test_bagging_max_samples(self, sample_regression_data, sample_base_models):
        """Test bagging with custom max_samples."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        bagging_model = forecaster.bagging_ensemble(
            X, y, n_estimators=15, max_samples=0.7
        )

        assert bagging_model.max_samples == 0.7


class TestVotingStrategies:
    """Test voting_strategies method specifically."""

    def test_voting_mean_strategy(self, sample_regression_data, sample_base_models):
        """Test voting with mean strategy."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        voting_model = forecaster.voting_strategies(X, y, strategy="mean")

        assert isinstance(voting_model, VotingRegressor)
        assert voting_model.weights is None  # Mean uses no weights

    def test_voting_weighted_strategy(self, sample_regression_data, sample_base_models):
        """Test voting with weighted strategy."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        weights = [0.5, 0.3, 0.2]

        voting_model = forecaster.voting_strategies(
            X, y, strategy="weighted", weights=weights
        )

        assert isinstance(voting_model, VotingRegressor)
        assert voting_model.weights is not None
        assert np.allclose(voting_model.weights, weights)

    def test_voting_auto_weights(self, sample_regression_data, sample_base_models):
        """Test voting with automatically calculated weights."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        voting_model = forecaster.voting_strategies(X, y, strategy="weighted")

        assert isinstance(voting_model, VotingRegressor)
        assert forecaster.weights is not None
        assert len(forecaster.weights) == len(sample_base_models)
        # Weights should sum to approximately 1
        assert np.isclose(forecaster.weights.sum(), 1.0)


class TestModelBlending:
    """Test model_blending method specifically."""

    def test_blending_basic(self, sample_regression_data, sample_base_models):
        """Test basic blending functionality."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        blending_model = forecaster.model_blending(X, y, blend_ratio=0.6)

        assert blending_model is not None
        assert hasattr(blending_model, "predict")

    def test_blending_with_optimization(self, sample_regression_data, sample_base_models):
        """Test blending with weight optimization."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        blending_model = forecaster.model_blending(
            X, y, blend_ratio=0.5, optimize_weights=True
        )

        assert forecaster.weights is not None
        assert len(forecaster.weights) == len(sample_base_models)

    def test_blending_invalid_ratio(self, sample_regression_data, sample_base_models):
        """Test blending with invalid blend ratio."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)

        with pytest.raises(ValueError, match="blend_ratio"):
            forecaster.model_blending(X, y, blend_ratio=1.5)

        with pytest.raises(ValueError, match="blend_ratio"):
            forecaster.model_blending(X, y, blend_ratio=0.0)


class TestEnsembleEvaluation:
    """Test ensemble evaluation methods."""

    def test_evaluate_basic(self, train_test_split_data, sample_base_models):
        """Test basic evaluation functionality."""
        X_train, X_test, y_train, y_test = train_test_split_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)
        results = forecaster.evaluate(X_test, y_test)

        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

        # Check that metrics are reasonable
        assert results["mse"] >= 0
        assert results["rmse"] >= 0
        assert results["mae"] >= 0

    def test_evaluate_custom_metrics(self, train_test_split_data, sample_base_models):
        """Test evaluation with custom metric selection."""
        X_train, X_test, y_train, y_test = train_test_split_data

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)
        results = forecaster.evaluate(X_test, y_test, metrics=["r2", "mae"])

        assert "r2" in results
        assert "mae" in results
        assert "mse" not in results
        assert "rmse" not in results

    def test_evaluate_without_fit(self, sample_regression_data):
        """Test that evaluation fails without fitting."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster()

        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.evaluate(X, y)


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_is_fitted(self, sample_regression_data, sample_base_models):
        """Test is_fitted method."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)

        assert forecaster.is_fitted() is False

        forecaster.fit(X, y)

        assert forecaster.is_fitted() is True

    def test_get_metadata(self, sample_regression_data, sample_base_models):
        """Test get_metadata method."""
        X, y = sample_regression_data

        forecaster = EnsembleForecaster(base_models=sample_base_models)
        forecaster.fit(X, y)

        metadata = forecaster.get_metadata()

        assert isinstance(metadata, dict)
        assert "n_samples" in metadata
        assert "n_features" in metadata

    def test_repr(self, sample_base_models):
        """Test string representation."""
        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="voting",
        )

        repr_str = repr(forecaster)

        assert "EnsembleForecaster" in repr_str
        assert "voting" in repr_str
        assert "not fitted" in repr_str

    def test_get_feature_importance(self, sample_regression_data):
        """Test feature importance extraction."""
        X, y = sample_regression_data

        # Use ensemble type that supports feature importance
        forecaster = EnsembleForecaster(ensemble_type="stacking")
        forecaster.fit(X, y)

        importance = forecaster.get_feature_importance()

        # May be None for some ensemble types
        if importance is not None:
            assert len(importance) > 0


class TestTimeSeriesForecasting:
    """Test ensemble with time series data."""

    def test_time_series_stacking(self, sample_time_series_data, sample_base_models):
        """Test stacking with time series data."""
        X, y = sample_time_series_data

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        forecaster.fit(X_train, y_train)
        predictions = forecaster.predict(X_test)

        assert predictions.shape == y_test.shape

        # Evaluate performance
        results = forecaster.evaluate(X_test, y_test)
        assert results["r2"] > 0  # Should have some predictive power


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self, sample_base_models):
        """Test with very small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.randn(20)

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="stacking",
        )

        # Should work but might have warnings
        forecaster.fit(X, y, cv=3)
        predictions = forecaster.predict(X)

        assert predictions.shape == y.shape

    def test_single_feature(self, sample_base_models):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = 2 * X.ravel() + np.random.randn(100) * 0.1

        forecaster = EnsembleForecaster(
            base_models=sample_base_models,
            ensemble_type="voting",
        )

        forecaster.fit(X, y)
        predictions = forecaster.predict(X)

        assert predictions.shape == y.shape

    def test_perfect_predictions(self, sample_base_models):
        """Test with perfectly predictable data."""
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3  # Perfect linear relationship

        forecaster = EnsembleForecaster(
            base_models=[LinearRegression()],
            ensemble_type="stacking",
        )

        forecaster.fit(X, y)
        results = forecaster.evaluate(X, y)

        # Should achieve very high R2 score
        assert results["r2"] > 0.99
