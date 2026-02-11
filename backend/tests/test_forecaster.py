"""
Tests for adaptive forecaster models.
"""

import numpy as np
import pandas as pd
import pytest

from ..models.adaptive_forecaster import AdaptiveForecaster


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    # Simple trend + noise
    values = 100 + np.cumsum(np.random.randn(200) * 0.5) + np.arange(200) * 0.1
    return pd.Series(values, index=dates)


def test_adaptive_forecaster_initialization():
    """Test forecaster initialization."""
    forecaster = AdaptiveForecaster("TEST", lag_window=10, horizon=1)
    assert forecaster.symbol == "TEST"
    assert forecaster.lag_window == 10
    assert forecaster.horizon == 1
    assert not forecaster.is_fitted


def test_adaptive_forecaster_fit(sample_data):
    """Test initial model training."""
    forecaster = AdaptiveForecaster("TEST", lag_window=20, horizon=1)
    forecaster.fit(sample_data, initial=True)
    assert forecaster.is_fitted
    assert forecaster.training_count == 1


def test_adaptive_forecaster_predict(sample_data):
    """Test prediction generation."""
    forecaster = AdaptiveForecaster("TEST", lag_window=20, horizon=1)
    forecaster.fit(sample_data, initial=True)

    last_prices = sample_data.iloc[-30:]
    predictions = forecaster.predict(last_prices, steps=5)

    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_adaptive_forecaster_partial_fit(sample_data):
    """Test incremental learning."""
    forecaster = AdaptiveForecaster("TEST", lag_window=20, horizon=1)
    forecaster.fit(sample_data.iloc[:150], initial=True)

    initial_count = forecaster.training_count

    # Partial fit on new data
    new_data = sample_data.iloc[150:]
    forecaster.partial_fit(new_data)

    assert forecaster.training_count > initial_count


def test_adaptive_forecaster_evaluate(sample_data):
    """Test model evaluation."""
    forecaster = AdaptiveForecaster("TEST", lag_window=20, horizon=1)
    forecaster.fit(sample_data.iloc[:150], initial=True)

    test_data = sample_data.iloc[150:]
    metrics = forecaster.evaluate(test_data)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_adaptive_forecaster_save_load(sample_data, tmp_path):
    """Test model saving and loading."""
    forecaster = AdaptiveForecaster("TEST", lag_window=20, horizon=1)
    forecaster.fit(sample_data, initial=True)

    # Save
    version_tag = "test_v1"
    artifact_path = forecaster.save(version_tag)

    # Load
    loaded = AdaptiveForecaster.load(artifact_path)

    assert loaded.symbol == forecaster.symbol
    assert loaded.is_fitted == forecaster.is_fitted
    assert loaded.lag_window == forecaster.lag_window

