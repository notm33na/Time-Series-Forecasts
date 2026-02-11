"""
Training script for Exponential Smoothing model.
Handles data splitting, model training, evaluation, and saving.
"""

import sys
from pathlib import Path

# Add project root to path (so backend can be imported as a package)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService

settings = get_settings()


def train_exponential_smoothing(
    symbol: str,
    trend: str = 'add',
    seasonal: int = None,
    test_size: float = 0.2,
    save_path: Path = None
):
    """
    Train Exponential Smoothing model with data splitting, evaluation, and saving.
    
    Args:
        symbol: Stock symbol
        trend: Trend type ('add' or 'mul')
        seasonal: Seasonal period (None for no seasonality)
        test_size: Proportion of data for testing
        save_path: Path to save model (.pkl file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"Training Exponential Smoothing for {symbol} (trend={trend}, seasonal={seasonal})")
    
    # Load data
    data_service = DataIngestionService()
    df = data_service.get_prices(symbol)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: need at least 100 points, got {len(df)}")
    
    # Normalize column names (handle both MongoDB lowercase and yfinance capitalized)
    if 'Close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'Close': 'close'})
    if 'close' in df.columns:
        prices = df["close"].sort_index()
    elif 'Close' in df.columns:
        prices = df["Close"].sort_index()
    else:
        raise ValueError(f"Missing 'close' or 'Close' column. Available: {list(df.columns)}")
    
    # Data splitting
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Train model
    print("Fitting Exponential Smoothing model...")
    model = ExponentialSmoothing(
        train_data,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal if seasonal else None,
    )
    fitted_model = model.fit()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_forecast = fitted_model.forecast(steps=len(test_data))
    
    mae = mean_absolute_error(test_data, test_forecast)
    rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
    mape = mean_absolute_percentage_error(test_data, test_forecast) * 100
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }
    
    print(f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_exponential_smoothing_{timestamp}.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_state = {
        "model": fitted_model,
        "symbol": symbol,
        "trend": trend,
        "seasonal": seasonal,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "metrics": metrics
    }
    
    joblib.dump(model_state, save_path)
    print(f"Model saved to {save_path}")
    
    return {
        "model": fitted_model,
        "metrics": metrics,
        "save_path": save_path,
        "train_size": len(train_data),
        "test_size": len(test_data)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Exponential Smoothing model")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--trend", type=str, default="add", choices=["add", "mul"], help="Trend type")
    parser.add_argument("--seasonal", type=int, default=None, help="Seasonal period")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_exponential_smoothing(
        symbol=args.symbol,
        trend=args.trend,
        seasonal=args.seasonal,
        test_size=args.test_size,
        save_path=Path(args.save_path) if args.save_path else None
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {result['save_path']}")
    print(f"Metrics: {result['metrics']}")

