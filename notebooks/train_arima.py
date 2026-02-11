"""
Training script for ARIMA model.
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
from statsmodels.tsa.arima.model import ARIMA  # type: ignore[reportMissingImports]
from datetime import datetime

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService

settings = get_settings()


def train_arima(
    symbol: str,
    order: tuple = (5, 1, 2),  # Updated to match i221500_A02_nlp
    test_size: float = 0.3,  # 70/30 split like i221500_A02_nlp
    save_path: Path = None
):
    """
    Train ARIMA model on Close_diff (difference of Close prices).
    Matches i221500_A02_nlp logic: predicts Close_diff with order (5,1,2).
    
    Args:
        symbol: Stock symbol
        order: ARIMA order (p, d, q) - default (5,1,2) matches i221500_A02_nlp
        test_size: Proportion of data for testing (default 0.3 for 70/30 split)
        save_path: Path to save model (.pkl file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"Training ARIMA({order}) for {symbol} on Close_diff (matching i221500_A02_nlp)")
    
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
    
    # Calculate Close_diff (difference of Close prices) - matches i221500_A02_nlp
    close_diff = prices.diff().dropna()
    
    if len(close_diff) < 50:
        raise ValueError(f"Insufficient data after differencing: need at least 50 points, got {len(close_diff)}")
    
    # Data splitting (70/30 like i221500_A02_nlp)
    train_size = int(len(close_diff) * (1 - test_size))
    train_data = close_diff[:train_size]
    test_data = close_diff[train_size:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"Training on Close_diff (price differences)")
    
    # Train model on Close_diff
    print("Fitting ARIMA model on Close_diff...")
    model = ARIMA(train_data, order=order)
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
        save_path = settings.models_dir / f"{symbol}_arima_{timestamp}.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state (fitted ARIMA model)
    model_state = {
        "model": fitted_model,
        "symbol": symbol,
        "horizon": 1,  # Default horizon
        "order": order,
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
    
    parser = argparse.ArgumentParser(description="Train ARIMA model")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--order", type=int, nargs=3, default=[5, 1, 2], 
                       help="ARIMA order (p d q) - default (5,1,2) matches i221500_A02_nlp")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set proportion (0.3 = 70/30 split like i221500_A02_nlp)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_arima(
        symbol=args.symbol,
        order=tuple(args.order),
        test_size=args.test_size,
        save_path=Path(args.save_path) if args.save_path else None
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {result['save_path']}")
    print(f"Metrics: {result['metrics']}")

