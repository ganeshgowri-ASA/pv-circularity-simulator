"""
Ensemble Forecasting Example
=============================

Comprehensive examples demonstrating the EnsembleForecaster class capabilities.

This script shows:
1. Basic usage with different ensemble strategies
2. Custom model configuration
3. Hyperparameter optimization
4. Performance evaluation
5. Time series forecasting application
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from pv_simulator.forecasting.ensemble import EnsembleForecaster


def example_1_basic_stacking():
    """Example 1: Basic stacking ensemble."""
    print("=" * 80)
    print("Example 1: Basic Stacking Ensemble")
    print("=" * 80)

    # Generate sample data
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and fit ensemble forecaster
    forecaster = EnsembleForecaster(
        ensemble_type="stacking",
        random_state=42,
        name="StackingExample",
    )

    print("\nFitting stacking ensemble...")
    forecaster.fit(X_train, y_train, scale_features=True)

    print(f"Ensemble: {forecaster}")
    print(f"Training time: {forecaster.training_time:.4f} seconds")

    # Make predictions
    predictions = forecaster.predict(X_test)

    # Evaluate
    results = forecaster.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    print(f"  R² Score: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")

    # Get predictions with uncertainty
    pred_with_std, std = forecaster.predict(X_test, return_std=True)
    print(f"\nPrediction uncertainty (mean std): {std.mean():.4f}")

    print("\n")


def example_2_custom_models():
    """Example 2: Ensemble with custom base models."""
    print("=" * 80)
    print("Example 2: Custom Base Models")
    print("=" * 80)

    # Generate data
    X, y = make_regression(
        n_samples=400,
        n_features=15,
        n_informative=10,
        noise=5.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Define custom base models
    custom_models = [
        Ridge(alpha=0.5),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
        GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    ]

    # Create ensemble with custom models
    forecaster = EnsembleForecaster(
        base_models=custom_models,
        ensemble_type="stacking",
        random_state=42,
    )

    print(f"\nUsing {len(custom_models)} custom base models:")
    for i, model in enumerate(custom_models, 1):
        print(f"  {i}. {model.__class__.__name__}")

    print("\nFitting ensemble...")
    forecaster.fit(X_train, y_train)

    # Evaluate
    results = forecaster.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    for metric, value in results.items():
        print(f"  {metric.upper()}: {value:.4f}")

    print("\n")


def example_3_voting_strategies():
    """Example 3: Different voting strategies."""
    print("=" * 80)
    print("Example 3: Voting Strategies Comparison")
    print("=" * 80)

    # Generate data
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=8,
        noise=8.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    strategies = ["mean", "weighted"]

    for strategy in strategies:
        print(f"\n{'=' * 40}")
        print(f"Strategy: {strategy.upper()}")
        print('=' * 40)

        forecaster = EnsembleForecaster(
            ensemble_type="voting",
            random_state=42,
        )

        forecaster.fit(X_train, y_train, voting_strategy=strategy)

        if forecaster.weights is not None:
            print(f"\nModel weights: {forecaster.weights}")

        results = forecaster.evaluate(X_test, y_test)
        print(f"R² Score: {results['r2']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")

    print("\n")


def example_4_bagging_ensemble():
    """Example 4: Bagging ensemble."""
    print("=" * 80)
    print("Example 4: Bagging Ensemble")
    print("=" * 80)

    # Generate data
    X, y = make_regression(
        n_samples=400,
        n_features=12,
        n_informative=10,
        noise=7.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create bagging ensemble
    forecaster = EnsembleForecaster(
        ensemble_type="bagging",
        random_state=42,
    )

    print("\nFitting bagging ensemble with 100 estimators...")
    forecaster.fit(
        X_train,
        y_train,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.9,
        bootstrap=True,
    )

    # Evaluate
    results = forecaster.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    print(f"  R² Score: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")

    print("\n")


def example_5_blending_ensemble():
    """Example 5: Model blending."""
    print("=" * 80)
    print("Example 5: Model Blending")
    print("=" * 80)

    # Generate data
    X, y = make_regression(
        n_samples=500,
        n_features=15,
        n_informative=12,
        noise=6.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Test different blend ratios
    blend_ratios = [0.5, 0.6, 0.7]

    best_r2 = -float('inf')
    best_ratio = None

    for ratio in blend_ratios:
        print(f"\n{'=' * 40}")
        print(f"Blend Ratio: {ratio}")
        print('=' * 40)

        forecaster = EnsembleForecaster(
            ensemble_type="blending",
            random_state=42,
        )

        forecaster.fit(X_train, y_train, blend_ratio=ratio, optimize_weights=True)

        if forecaster.weights is not None:
            print(f"Optimized weights: {forecaster.weights}")

        results = forecaster.evaluate(X_test, y_test)
        print(f"R² Score: {results['r2']:.4f}")

        if results['r2'] > best_r2:
            best_r2 = results['r2']
            best_ratio = ratio

    print(f"\nBest blend ratio: {best_ratio} (R² = {best_r2:.4f})")
    print("\n")


def example_6_time_series_forecasting():
    """Example 6: Time series forecasting application."""
    print("=" * 80)
    print("Example 6: Time Series Forecasting")
    print("=" * 80)

    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 500

    # Time index
    time = np.arange(n_samples)

    # Create complex time series with trend, seasonality, and noise
    trend = 0.5 * time
    seasonality_1 = 20 * np.sin(2 * np.pi * time / 50)
    seasonality_2 = 10 * np.sin(2 * np.pi * time / 25)
    noise = np.random.normal(0, 5, n_samples)

    y = trend + seasonality_1 + seasonality_2 + noise

    # Create lagged features
    lags = [1, 2, 3, 5, 7, 10, 14]
    X_features = []

    for lag in lags:
        X_features.append(np.concatenate([np.zeros(lag), y[:-lag]]))

    # Add time-based features
    X_features.append(np.sin(2 * np.pi * time / 50))
    X_features.append(np.cos(2 * np.pi * time / 50))
    X_features.append(np.sin(2 * np.pi * time / 25))
    X_features.append(np.cos(2 * np.pi * time / 25))

    X = np.column_stack(X_features)

    # Split: use first 80% for training, last 20% for testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTime series length: {n_samples}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")

    # Create ensemble forecaster
    forecaster = EnsembleForecaster(
        ensemble_type="stacking",
        random_state=42,
    )

    print("\nFitting ensemble for time series forecasting...")
    forecaster.fit(X_train, y_train, scale_features=True, cv=5)

    # Make predictions
    predictions = forecaster.predict(X_test)

    # Evaluate
    results = forecaster.evaluate(X_test, y_test)

    print("\nForecasting Performance:")
    print(f"  R² Score: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")

    # Show sample predictions
    print("\nSample Predictions (first 5 test points):")
    print("  Actual    Predicted    Error")
    for i in range(min(5, len(predictions))):
        error = abs(y_test[i] - predictions[i])
        print(f"  {y_test[i]:7.2f}    {predictions[i]:7.2f}    {error:7.2f}")

    print("\n")


def example_7_ensemble_comparison():
    """Example 7: Compare all ensemble strategies."""
    print("=" * 80)
    print("Example 7: Ensemble Strategy Comparison")
    print("=" * 80)

    # Generate data
    X, y = make_regression(
        n_samples=600,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ensemble_types = ["stacking", "bagging", "voting", "blending"]
    results_comparison = {}

    for ensemble_type in ensemble_types:
        print(f"\n{'=' * 40}")
        print(f"Ensemble Type: {ensemble_type.upper()}")
        print('=' * 40)

        forecaster = EnsembleForecaster(
            ensemble_type=ensemble_type,
            random_state=42,
        )

        # Fit with appropriate parameters
        if ensemble_type == "voting":
            forecaster.fit(X_train, y_train, voting_strategy="weighted")
        elif ensemble_type == "bagging":
            forecaster.fit(X_train, y_train, n_estimators=50)
        elif ensemble_type == "blending":
            forecaster.fit(X_train, y_train, blend_ratio=0.6)
        else:
            forecaster.fit(X_train, y_train)

        # Evaluate
        results = forecaster.evaluate(X_test, y_test)
        results_comparison[ensemble_type] = results

        print(f"R² Score: {results['r2']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n{'Ensemble Type':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 45)

    for ensemble_type, results in results_comparison.items():
        print(
            f"{ensemble_type:<15} "
            f"{results['r2']:<10.4f} "
            f"{results['rmse']:<10.4f} "
            f"{results['mae']:<10.4f}"
        )

    # Find best performer
    best_ensemble = max(results_comparison.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest performer: {best_ensemble[0].upper()} "
          f"(R² = {best_ensemble[1]['r2']:.4f})")

    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# ENSEMBLE FORECASTING EXAMPLES")
    print("#" * 80)
    print("\n")

    try:
        example_1_basic_stacking()
        example_2_custom_models()
        example_3_voting_strategies()
        example_4_bagging_ensemble()
        example_5_blending_ensemble()
        example_6_time_series_forecasting()
        example_7_ensemble_comparison()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
