"""
Training script for LSTM model.
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


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        SMAPE value (percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Where denominator is zero, use a small epsilon
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService

settings = get_settings()


# Removed create_sequences - not used in i221500_A02_nlp style LSTM


def train_lstm(
    symbol: str,
    units: int = 50,
    epochs: int = 5,  # Updated to match i221500_A02_nlp
    batch_size: int = 16,  # Updated to match i221500_A02_nlp
    test_size: float = 0.3,  # 70/30 split like i221500_A02_nlp
    save_path: Path = None
):
    """
    Train LSTM model on multiple features to predict Close_diff.
    Matches i221500_A02_nlp logic:
    - Uses multiple features: ['Open','High','Low','Close','Adj Close','Volume'] (normalized)
    - Predicts Close_diff (difference of Close prices)
    - Architecture: LSTM(50, return_sequences=True) -> LSTM(50) -> Dense(1)
    - Input shape: (samples, 1, features)
    
    Args:
        symbol: Stock symbol
        units: LSTM units per layer (default 50 matches i221500_A02_nlp)
        epochs: Training epochs (default 5 matches i221500_A02_nlp)
        batch_size: Batch size (default 16 matches i221500_A02_nlp)
        test_size: Proportion of data for testing (default 0.3 for 70/30 split)
        save_path: Path to save model (.h5 file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"Training LSTM for {symbol} on Close_diff with multiple features (matching i221500_A02_nlp)")
    print(f"  Architecture: LSTM({units}, return_sequences=True) -> LSTM({units}) -> Dense(1)")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    
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
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: need at least 100 points, got {len(df)}")
    
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
    # Map to expected column names
    col_mapping = {}
    for target_col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if target_col in df.columns:
            continue
        # Try lowercase
        lower_col = target_col.lower()
        if lower_col in df.columns:
            col_mapping[lower_col] = target_col
        # Try with space
        elif ' ' in target_col and target_col.replace(' ', '').lower() in df.columns:
            col_mapping[target_col.replace(' ', '').lower()] = target_col
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    # Get available numeric columns (matching i221500_A02_nlp: Open, High, Low, Close, Adj Close, Volume)
    num_cols = []
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            num_cols.append(col)
        elif col == 'Adj Close' and 'Adj Close' not in df.columns:
            # Adj Close might not be available, that's okay
            continue
    
    if len(num_cols) < 4:
        raise ValueError(f"Need at least 4 features (Open, High, Low, Close). Found: {num_cols}")
    
    print(f"Using features: {num_cols}")
    
    # Normalize features (matching i221500_A02_nlp)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[num_cols])
    df_scaled = pd.DataFrame(scaled, columns=num_cols, index=df.index)
    
    # Calculate Close_diff on normalized Close (matching i221500_A02_nlp)
    close_col = [c for c in num_cols if 'Close' in c and 'Adj' not in c][0]
    df_scaled['Close_diff'] = df_scaled[close_col].diff()
    df_scaled = df_scaled.dropna()
    
    if len(df_scaled) < 50:
        raise ValueError(f"Insufficient data after differencing: need at least 50 points, got {len(df_scaled)}")
    
    # Features: all columns except Close and Close_diff (matching i221500_A02_nlp)
    feature_cols = [c for c in num_cols if c != close_col]
    X = df_scaled[feature_cols].values
    y = df_scaled['Close_diff'].values
    
    # Data splitting (70/30 like i221500_A02_nlp)
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Training on Close_diff (price differences) with {len(feature_cols)} features")
    
    # Reshape for LSTM: (samples, 1, features) - matches i221500_A02_nlp
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Build model architecture matching i221500_A02_nlp
    print("Building LSTM model...")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train with parameters from i221500_A02_nlp: epochs=5, batch_size=16
    print(f"Training for {epochs} epochs...")
    history = model.fit(
        X_train_lstm, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions = model.predict(X_test_lstm, verbose=0).flatten()
    
    # Predictions are already Close_diff, no inverse transform needed
    # (i221500_A02_nlp doesn't inverse transform Close_diff)
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mape = mean_absolute_percentage_error(y_test, test_predictions) * 100
    smape = symmetric_mean_absolute_percentage_error(y_test, test_predictions)
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape)
    }
    
    print(f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_lstm_{timestamp}.weights.h5"
    else:
        # Ensure .weights.h5 extension for Keras
        save_path = Path(save_path)
        if not str(save_path).endswith('.weights.h5'):
            save_path = save_path.with_suffix('.weights.h5')
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights and architecture
    model.save_weights(str(save_path))
    
    # Save scaler and metadata separately
    # Metadata file: replace .weights.h5 with .pkl
    metadata_path = save_path.with_name(save_path.stem.replace('.weights', '') + '.pkl')
    import joblib
    metadata = {
        "scaler": scaler,
        "symbol": symbol,
        "horizon": 1,  # Default horizon
        "lag_window": 1,  # For i221500_A02_nlp style, lag_window is 1 (single timestep)
        "units": units,
        "feature_cols": feature_cols,  # Save feature columns used
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "model_config": model.get_config()
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"Model weights saved to {save_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "save_path": save_path,
        "metadata_path": metadata_path,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LSTM model (matching i221500_A02_nlp logic)")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--units", type=int, default=50, help="LSTM units (default 50 matches i221500_A02_nlp)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default 5 matches i221500_A02_nlp)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16 matches i221500_A02_nlp)")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set proportion (0.3 = 70/30 split like i221500_A02_nlp)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_lstm(
        symbol=args.symbol,
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

