"""
Comprehensive unit tests for all forecasting models.
"""

import numpy as np
import pandas as pd
import pytest

from ..models.forecasting_models import (
    BaseForecaster,
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    VARForecaster,
    MovingAverageForecaster,
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
)


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    # Simple trend + noise
    values = 100 + np.cumsum(np.random.randn(200) * 0.5) + np.arange(200) * 0.1
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for VAR model."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    
    base_price = 100
    data = []
    for i in range(100):
        change = np.random.randn() * 2
        open_price = base_price + change
        high_price = open_price + abs(np.random.randn() * 1)
        low_price = open_price - abs(np.random.randn() * 1)
        close_price = open_price + np.random.randn() * 0.5
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        base_price = close_price
    
    return pd.DataFrame(data, index=dates)


def test_arima_forecaster_initialization():
    """Test ARIMA forecaster initialization."""
    forecaster = ARIMAForecaster("TEST", horizon=5)
    assert forecaster.symbol == "TEST"
    assert forecaster.horizon == 5
    assert not forecaster.is_fitted


def test_arima_forecaster_fit_predict(sample_data):
    """Test ARIMA forecaster fit and predict."""
    forecaster = ARIMAForecaster("TEST", horizon=5)
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_arima_forecaster_evaluate(sample_data):
    """Test ARIMA forecaster evaluation."""
    forecaster = ARIMAForecaster("TEST", horizon=5)
    forecaster.fit(sample_data.iloc[:150])
    
    test_data = sample_data.iloc[150:]
    predictions = forecaster.predict(steps=len(test_data))
    
    metrics = forecaster.evaluate(test_data, predictions)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_exponential_smoothing_forecaster(sample_data):
    """Test Exponential Smoothing forecaster."""
    forecaster = ExponentialSmoothingForecaster("TEST", horizon=5)
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_moving_average_forecaster_sma(sample_data):
    """Test Simple Moving Average forecaster."""
    forecaster = MovingAverageForecaster("TEST", horizon=5, window=20, ma_type="SMA")
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    assert forecaster.last_ma_value is not None
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))
    # SMA forecast should be constant
    assert len(set(predictions)) == 1


def test_moving_average_forecaster_ema(sample_data):
    """Test Exponential Moving Average forecaster."""
    forecaster = MovingAverageForecaster("TEST", horizon=5, window=20, ma_type="EMA")
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_moving_average_forecaster_invalid_type():
    """Test Moving Average forecaster with invalid type."""
    with pytest.raises(ValueError, match="ma_type must be 'SMA' or 'EMA'"):
        MovingAverageForecaster("TEST", horizon=5, ma_type="INVALID")


def test_moving_average_forecaster_insufficient_data():
    """Test Moving Average forecaster with insufficient data."""
    forecaster = MovingAverageForecaster("TEST", horizon=5, window=50)
    short_data = pd.Series([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(short_data)


def test_var_forecaster(sample_ohlc_data):
    """Test VAR forecaster."""
    forecaster = VARForecaster("TEST", horizon=5)
    forecaster.fit(sample_ohlc_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_var_forecaster_insufficient_data():
    """Test VAR forecaster with insufficient data."""
    forecaster = VARForecaster("TEST", horizon=5)
    short_data = pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [2, 3, 4],
        'Low': [0.5, 1.5, 2.5],
        'Close': [1.5, 2.5, 3.5]
    })
    
    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(short_data)


def test_lstm_forecaster(sample_data):
    """Test LSTM forecaster."""
    forecaster = LSTMForecaster("TEST", horizon=5, lag_window=30)
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(sample_data, steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_lstm_forecaster_insufficient_data():
    """Test LSTM forecaster with insufficient data."""
    forecaster = LSTMForecaster("TEST", horizon=5, lag_window=30)
    short_data = pd.Series([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(short_data)


def test_gru_forecaster(sample_data):
    """Test GRU forecaster."""
    forecaster = GRUForecaster("TEST", horizon=5, lag_window=30)
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(sample_data, steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_transformer_forecaster(sample_data):
    """Test Transformer forecaster."""
    forecaster = TransformerForecaster("TEST", horizon=5, lag_window=30)
    forecaster.fit(sample_data)
    
    assert forecaster.is_fitted
    predictions = forecaster.predict(sample_data, steps=5)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_base_forecaster_evaluate():
    """Test BaseForecaster evaluate method."""
    actual = pd.Series([100, 101, 102, 103, 104])
    predicted = np.array([100.5, 101.2, 101.8, 103.1, 103.9])
    
    # Create a dummy forecaster
    class DummyForecaster(BaseForecaster):
        def fit(self, data):
            return self
        def predict(self, data=None, steps=None):
            return predicted
    
    forecaster = DummyForecaster("TEST", horizon=5)
    metrics = forecaster.evaluate(actual, predicted)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_base_forecaster_evaluate_mismatched_lengths():
    """Test BaseForecaster evaluate with mismatched lengths."""
    actual = pd.Series([100, 101, 102, 103, 104, 105])
    predicted = np.array([100.5, 101.2, 101.8, 103.1, 103.9])
    
    class DummyForecaster(BaseForecaster):
        def fit(self, data):
            return self
        def predict(self, data=None, steps=None):
            return predicted
    
    forecaster = DummyForecaster("TEST", horizon=5)
    metrics = forecaster.evaluate(actual, predicted)
    
    # Should handle length mismatch gracefully
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics


def test_all_forecasters_require_fit():
    """Test that all forecasters require fitting before prediction."""
    forecasters = [
        ARIMAForecaster("TEST", horizon=5),
        ExponentialSmoothingForecaster("TEST", horizon=5),
        MovingAverageForecaster("TEST", horizon=5),
        LSTMForecaster("TEST", horizon=5),
        GRUForecaster("TEST", horizon=5),
        TransformerForecaster("TEST", horizon=5),
    ]
    
    sample_data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 10)
    
    for forecaster in forecasters:
        with pytest.raises((ValueError, AttributeError)):
            forecaster.predict(sample_data, steps=5)

