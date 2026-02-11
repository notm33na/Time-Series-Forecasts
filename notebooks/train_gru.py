"""
Training script for GRU model.
Handles data splitting, model training, evaluation, and saving.
"""

import sys
from pathlib import Path

# Add project root to path (so backend can be imported as a package)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf  # type: ignore
from datetime import datetime

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService

settings = get_settings()


def create_sequences(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for GRU."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_gru(
    symbol: str,
    lag_window: int = 60,
    units: int = 50,
    epochs: int = 20,
    batch_size: int = 32,
    test_size: float = 0.2,
    save_path: Path = None
):
    """
    Train GRU model with data splitting, evaluation, and saving.
    
    Args:
        symbol: Stock symbol
        lag_window: Sequence length
        units: GRU units per layer
        epochs: Training epochs
        batch_size: Batch size
        test_size: Proportion of data for testing
        save_path: Path to save model (.h5 file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"Training GRU for {symbol} (window={lag_window}, units={units})")
    
    # Load data - try MongoDB first, fallback to yfinance if empty
    df = None
    try:
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol)
        if df is None or len(df) == 0:
            print(f"⚠️  No data found in MongoDB for {symbol}, fetching from yfinance...")
            df = None
        else:
            print(f"✓ Found {len(df)} records in MongoDB")
    except Exception as e:
        print(f"⚠️  Could not fetch from MongoDB: {e}")
        print(f"   Falling back to yfinance...")
        df = None
    
    # If MongoDB is empty or unavailable, fetch directly from yfinance
    if df is None or len(df) == 0:
        try:
            import yfinance as yf
            print(f"📥 Fetching data from yfinance for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y")
            
            if df is None or len(df) == 0:
                raise ValueError(f"No data available for {symbol} from yfinance")
            
            # Convert to expected format
            df = df.reset_index()
            if 'Date' in df.columns:
                df['date'] = df['Date']
            else:
                df['date'] = df.index
            
            # Ensure we have Close column
            if 'Close' not in df.columns:
                raise ValueError(f"Missing 'Close' column in fetched data")
            
            print(f"✓ Fetched {len(df)} records from yfinance")
            
            # Optionally save to MongoDB for future use
            try:
                data_service = DataIngestionService()
                data_service.ingest_from_dataframe(symbol, df)
                print(f"✓ Saved {len(df)} records to MongoDB for future use")
            except Exception as e:
                print(f"⚠️  Could not save to MongoDB (continuing anyway): {e}")
        except ImportError:
            raise ImportError(
                "yfinance is required when MongoDB is empty. "
                "Install with: pip install yfinance"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to fetch data for {symbol}: {e}\n"
                "Please ensure:\n"
                "  1. MongoDB has data, OR\n"
                "  2. yfinance is installed and can fetch data for this symbol"
            )
    
    if len(df) < lag_window + 100:
        raise ValueError(f"Insufficient data: need at least {lag_window + 100} points, got {len(df)}")
    
    # Set date as index if it exists (for proper sorting)
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    elif df.index.name != 'date' and not isinstance(df.index, pd.DatetimeIndex):
        # If no date column but index is not datetime, try to convert
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
    
    # Normalize column names (handle both MongoDB lowercase and yfinance capitalized)
    if 'Close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'Close': 'close'})
    if 'close' in df.columns:
        prices = df["close"].sort_index().values
    elif 'Close' in df.columns:
        prices = df["Close"].sort_index().values
    else:
        raise ValueError(f"Missing 'close' or 'Close' column. Available: {list(df.columns)}")
    
    # Data splitting
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lag_window)
    X_test, y_test = create_sequences(test_scaled, lag_window)
    
    if len(X_train) == 0:
        raise ValueError(f"Insufficient data after sequence creation")
    
    # Check if test set has enough sequences
    if len(X_test) == 0:
        print(f"⚠️  Warning: Test set too small (need at least {lag_window + 1} points, got {len(test_data)}). Skipping test evaluation.")
        # Use last portion of training data for validation instead
        val_split = int(len(X_train) * 0.2)
        X_test = X_train[-val_split:]
        y_test = y_train[-val_split:]
        X_train = X_train[:-val_split]
        y_train = y_train[:-val_split]
        print(f"Using last {len(X_test)} sequences from training set for testing")
    
    # Reshape for GRU
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    print("Building GRU model...")
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(units, return_sequences=True, input_shape=(lag_window, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(units, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    print(f"Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    mape = mean_absolute_percentage_error(y_test_actual, test_predictions) * 100
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }
    
    print(f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_gru_{timestamp}.weights.h5"
    else:
        # Ensure .weights.h5 extension for Keras
        save_path = Path(save_path)
        if not str(save_path).endswith('.weights.h5'):
            save_path = save_path.with_suffix('.weights.h5')
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model.save_weights(str(save_path))
    
    # Save scaler and metadata
    # Metadata file: replace .weights.h5 with .pkl
    metadata_path = save_path.with_name(save_path.stem.replace('.weights', '') + '.pkl')
    import joblib
    metadata = {
        "scaler": scaler,
        "symbol": symbol,
        "lag_window": lag_window,
        "units": units,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "metrics": metrics,
        "model_config": model.get_config()
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"Model weights saved to {save_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "save_path": save_path,
        "metadata_path": metadata_path,
        "train_size": len(train_data),
        "test_size": len(test_data)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GRU model")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--lag-window", type=int, default=60, help="Sequence length")
    parser.add_argument("--units", type=int, default=50, help="GRU units")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_gru(
        symbol=args.symbol,
        lag_window=args.lag_window,
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        save_path=Path(args.save_path) if args.save_path else None
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {result['save_path']}")
    print(f"Metadata saved to: {result['metadata_path']}")
    print(f"Metrics: {result['metrics']}")

