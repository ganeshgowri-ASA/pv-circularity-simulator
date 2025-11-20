# PV Circularity Simulator

A comprehensive platform for photovoltaic lifecycle simulation, energy forecasting, and circular economy modeling.

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## Features

### Advanced ML Ensemble Forecasting

The PV Circularity Simulator includes a production-ready ensemble forecasting system with:

- **Stacking**: Meta-learning from base model predictions with cross-validation
- **Bagging**: Bootstrap aggregating for variance reduction
- **Voting**: Weighted/unweighted averaging strategies
- **Blending**: Hold-out based model combination

### Key Capabilities

- ✅ Multiple ensemble strategies (stacking, bagging, voting, blending)
- ✅ Hyperparameter optimization (grid search, random search)
- ✅ Automatic weight optimization for model combination
- ✅ Feature scaling and preprocessing
- ✅ Cross-validation and performance metrics
- ✅ Prediction uncertainty estimation
- ✅ Time series forecasting support
- ✅ Comprehensive evaluation metrics (R², RMSE, MAE, MSE)
- ✅ Full type hints and docstrings
- ✅ Production-ready error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (for testing and development)
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Usage

```python
from pv_simulator.forecasting.ensemble import EnsembleForecaster
from sklearn.model_selection import train_test_split

# Load your data
X, y = load_your_data()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create ensemble forecaster
forecaster = EnsembleForecaster(
    ensemble_type="stacking",
    random_state=42
)

# Fit the model
forecaster.fit(X_train, y_train)

# Make predictions
predictions = forecaster.predict(X_test)

# Evaluate performance
results = forecaster.evaluate(X_test, y_test)
print(f"R² Score: {results['r2']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
```

### Custom Base Models

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

# Define custom base models
custom_models = [
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
]

# Create ensemble with custom models
forecaster = EnsembleForecaster(
    base_models=custom_models,
    ensemble_type="stacking"
)

forecaster.fit(X_train, y_train)
```

### Voting Strategies

```python
# Weighted voting with automatic weight optimization
forecaster = EnsembleForecaster(ensemble_type="voting")
forecaster.fit(X_train, y_train, voting_strategy="weighted")

# Custom weights
forecaster.fit(X_train, y_train, voting_strategy="weighted", weights=[0.3, 0.3, 0.4])
```

### Bagging Ensemble

```python
# Bagging with custom parameters
forecaster = EnsembleForecaster(ensemble_type="bagging")
forecaster.fit(
    X_train, y_train,
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True
)
```

### Model Blending

```python
# Blending with weight optimization
forecaster = EnsembleForecaster(ensemble_type="blending")
forecaster.fit(
    X_train, y_train,
    blend_ratio=0.6,
    optimize_weights=True
)
```

### Prediction with Uncertainty

```python
# Get predictions with standard deviation
predictions, std = forecaster.predict(X_test, return_std=True)

print(f"Mean prediction uncertainty: {std.mean():.4f}")
```

## Examples

Comprehensive examples are available in the `examples/` directory:

```bash
# Run ensemble forecasting examples
python examples/ensemble_forecasting_example.py
```

The examples demonstrate:
1. Basic stacking ensemble
2. Custom model configuration
3. Voting strategies comparison
4. Bagging ensemble
5. Model blending
6. Time series forecasting
7. Complete ensemble strategy comparison

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pv_simulator --cov-report=html

# Run specific test file
pytest tests/test_forecasting/test_ensemble.py

# Run with verbose output
pytest -v
```

## API Reference

### EnsembleForecaster

Main class for ensemble forecasting.

**Parameters:**
- `base_models` (List[BaseEstimator], optional): List of base estimators
- `meta_model` (BaseEstimator, optional): Meta-learner for stacking
- `ensemble_type` (str): Type of ensemble - "stacking", "bagging", "voting", or "blending"
- `n_jobs` (int): Number of parallel jobs (-1 uses all processors)
- `random_state` (int, optional): Random seed for reproducibility

**Methods:**

#### fit(X, y, **kwargs)
Fit the ensemble forecaster to training data.

#### predict(X, return_std=False)
Generate predictions using the fitted ensemble.

#### stacking_models(X, y, cv=5, passthrough=False)
Create and fit a stacking ensemble with cross-validated predictions.

#### bagging_ensemble(X, y, n_estimators=10, max_samples=1.0, ...)
Create and fit a bagging ensemble for variance reduction.

#### voting_strategies(X, y, strategy='mean', weights=None)
Create and fit a voting ensemble with various aggregation strategies.

#### model_blending(X, y, blend_ratio=0.5, optimize_weights=True)
Create and fit a blending ensemble using hold-out validation.

#### evaluate(X, y, metrics=None)
Evaluate the ensemble on test data with multiple metrics.

#### optimize_hyperparameters(X, y, param_distributions, ...)
Optimize ensemble hyperparameters using grid or random search.

## Project Structure

```
pv-circularity-simulator/
├── pv_simulator/              # Main package
│   ├── forecasting/           # Forecasting modules
│   │   ├── base.py           # Base forecaster classes
│   │   ├── ensemble.py       # EnsembleForecaster
│   │   └── models/           # Individual ML models
│   └── utils/                # Utility functions
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   └── test_forecasting/    # Forecasting tests
│       └── test_ensemble.py # Ensemble tests
├── examples/                 # Example scripts
│   └── ensemble_forecasting_example.py
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Dependencies

### Core Dependencies
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0

### Development Dependencies
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- mypy >= 0.950

## Performance Metrics

The ensemble forecaster supports the following metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code is properly formatted: `black pv_simulator/`
3. Type hints are included
4. Docstrings follow NumPy/Google style
5. New features include tests

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pv_circularity_simulator,
  title = {PV Circularity Simulator},
  author = {PV Circularity Team},
  year = {2025},
  url = {https://github.com/yourusername/pv-circularity-simulator}
}
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Note**: This is a production-ready implementation with comprehensive testing, full documentation, and type hints throughout. All ensemble methods are based on scikit-learn's robust implementations and include proper error handling and validation.
