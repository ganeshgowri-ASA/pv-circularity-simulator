"""Tests for metrics calculation."""

import numpy as np
import pytest

from pv_simulator.forecasting.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for metrics calculator."""

    @pytest.fixture
    def sample_data(self):
        """Sample actual and predicted values."""
        actual = [100, 110, 105, 115, 120, 125, 130]
        predicted = [98, 112, 103, 117, 118, 127, 132]
        return actual, predicted

    def test_calculate_mae(self, sample_data):
        """Test MAE calculation."""
        actual, predicted = sample_data
        calculator = MetricsCalculator()

        mae = calculator.calculate_mae(np.array(actual), np.array(predicted))
        assert mae > 0
        assert isinstance(mae, float)

    def test_calculate_rmse(self, sample_data):
        """Test RMSE calculation."""
        actual, predicted = sample_data
        calculator = MetricsCalculator()

        rmse = calculator.calculate_rmse(np.array(actual), np.array(predicted))
        assert rmse > 0
        assert isinstance(rmse, float)

    def test_calculate_mape(self, sample_data):
        """Test MAPE calculation."""
        actual, predicted = sample_data
        calculator = MetricsCalculator()

        mape = calculator.calculate_mape(np.array(actual), np.array(predicted))
        assert mape >= 0
        assert isinstance(mape, float)

    def test_calculate_metrics(self, sample_data):
        """Test comprehensive metrics calculation."""
        actual, predicted = sample_data
        calculator = MetricsCalculator()

        metrics = calculator.calculate_metrics(actual, predicted)

        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0
        assert -1 <= metrics.r2 <= 1
