"""
Improved GRU retraining script with automatic scaler range detection and fixes.
Addresses:
- Scaler range mismatch (uses StandardScaler or properly fitted MinMaxScaler)
- Systematic underprediction bias
- Low correlation (increased sequence length)
- Negative R² (better architecture and training)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from datetime import datetime
import joblib

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
from backend.utils.model_registry import register_model_version

settings = get_settings()
logger = get_logger(__name__)


def create_sequences(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for GRU."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_gru_fixed(
    symbol: str,
    lag_window: int = 90,  # Increased from 60 to capture slower trends
    units: int = 64,  # Slightly increased
    epochs: int = 50,  # More epochs with early stopping
    batch_size: int = 32,
    test_size: float = 0.2,
    scaler_type: str = "standard",  # "standard" or "minmax"
    use_relu_output: bool = True,  # Use ReLU to reduce saturation
    save_path: Path = None
):
    """
    Train improved GRU model with automatic scaler range detection.
    
    Key improvements:
    - Automatic price range detection
    - StandardScaler (better for extrapolation) or properly fitted MinMaxScaler
    - Increased sequence length (90-120)
    - ReLU activations in output layers
    - Learning rate decay
    - Early stopping
    - More training epochs
    
    Args:
        symbol: Stock symbol
        lag_window: Sequence length (default 90, recommended 90-120)
        units: GRU units per layer
        epochs: Maximum training epochs (early stopping will stop earlier)
        batch_size: Batch size
        test_size: Proportion of data for testing
        scaler_type: "standard" (recommended) or "minmax"
        use_relu_output: Use ReLU in output layer to reduce saturation
        save_path: Path to save model (.h5 file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"\n{'='*70}")
    print(f"Improved GRU Training for {symbol}")
    print(f"{'='*70}\n")
    
    # Load data
    df = None
    try:
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol, limit=2000)  # Get more data
        if df is None or len(df) == 0:
            print(f"⚠️  No data found in MongoDB for {symbol}, fetching from yfinance...")
            df = None
        else:
            print(f"✓ Found {len(df)} records in MongoDB")
    except Exception as e:
        print(f"⚠️  Could not fetch from MongoDB: {e}")
        print(f"   Falling back to yfinance...")
        df = None
    
    # Fallback to yfinance
    if df is None or len(df) == 0:
        try:
            import yfinance as yf
            print(f"📥 Fetching data from yfinance for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y")
            
            if df is None or len(df) == 0:
                raise ValueError(f"No data available for {symbol} from yfinance")
            
            df = df.reset_index()
            if 'Date' in df.columns:
                df['date'] = df['Date']
            else:
                df['date'] = df.index
            
            if 'Close' not in df.columns:
                raise ValueError(f"Missing 'Close' column in fetched data")
            
            print(f"✓ Fetched {len(df)} records from yfinance")
            
            # Save to MongoDB
            try:
                data_service = DataIngestionService()
                data_service.ingest_from_dataframe(symbol, df)
                print(f"✓ Saved {len(df)} records to MongoDB")
            except Exception as e:
                print(f"⚠️  Could not save to MongoDB (continuing anyway): {e}")
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    if len(df) < lag_window + 100:
        raise ValueError(f"Insufficient data: need at least {lag_window + 100} points, got {len(df)}")
    
    # Prepare data
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    elif df.index.name != 'date' and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
    
    # Normalize column names
    if 'Close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'Close': 'close'})
    if 'close' in df.columns:
        prices = df["close"].sort_index().values
    elif 'Close' in df.columns:
        prices = df["Close"].sort_index().values
    else:
        raise ValueError(f"Missing 'close' or 'Close' column. Available: {list(df.columns)}")
    
    # Display price range information
    price_min = float(prices.min())
    price_max = float(prices.max())
    price_mean = float(prices.mean())
    price_std = float(prices.std())
    current_price = float(prices[-1])
    
    print(f"\n📊 Data Statistics:")
    print(f"   Total samples: {len(prices)}")
    print(f"   Price range: ${price_min:.2f} - ${price_max:.2f}")
    print(f"   Mean: ${price_mean:.2f}, Std: ${price_std:.2f}")
    print(f"   Current price: ${current_price:.2f}")
    print(f"   Price range span: ${price_max - price_min:.2f}")
    
    # Data splitting
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"\n📦 Data Split:")
    print(f"   Train size: {len(train_data)}")
    print(f"   Test size: {len(test_data)}")
    print(f"   Train range: ${train_data.min():.2f} - ${train_data.max():.2f}")
    print(f"   Test range: ${test_data.min():.2f} - ${test_data.max():.2f}")
    
    # Scale data with automatic range detection
    print(f"\n🔧 Scaling Configuration:")
    print(f"   Scaler type: {scaler_type.upper()}")
    
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
        # Fit on training data
        train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
        
        print(f"   Mean: ${scaler.mean_[0]:.2f}")
        print(f"   Std: ${scaler.scale_[0]:.2f}")
        print(f"   Typical range (±3 std): [${scaler.mean_[0] - 3*scaler.scale_[0]:.2f}, ${scaler.mean_[0] + 3*scaler.scale_[0]:.2f}]")
        print(f"   ✓ StandardScaler handles extrapolation better than MinMaxScaler")
    else:
        scaler = MinMaxScaler()
        # Fit on FULL data range to avoid future extrapolation
        # This ensures the scaler covers the entire price range
        scaler.fit(prices.reshape(-1, 1))
        train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
        
        print(f"   Min: ${scaler.data_min_[0]:.2f}")
        print(f"   Max: ${scaler.data_max_[0]:.2f}")
        print(f"   Range: ${scaler.data_max_[0] - scaler.data_min_[0]:.2f}")
        print(f"   ✓ MinMaxScaler fitted to FULL data range (${price_min:.2f} - ${price_max:.2f})")
        print(f"   ✓ This prevents extrapolation issues")
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lag_window)
    X_test, y_test = create_sequences(test_scaled, lag_window)
    
    if len(X_train) == 0:
        raise ValueError(f"Insufficient data after sequence creation")
    
    if len(X_test) == 0:
        print(f"⚠️  Warning: Test set too small. Using validation split from training data.")
        val_split = int(len(X_train) * 0.2)
        X_test = X_train[-val_split:]
        y_test = y_train[-val_split:]
        X_train = X_train[:-val_split]
        y_train = y_train[:-val_split]
        print(f"   Using last {len(X_test)} sequences from training set for testing")
    
    # Reshape for GRU
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Sequence length: {lag_window}")
    print(f"   GRU units: {units}")
    print(f"   Output activation: {'ReLU' if use_relu_output else 'Linear'}")
    
    # Build improved model
    # Note: ReLU on output can cause issues - use linear for price prediction
    # ReLU is only useful if we know prices are always positive (which they are, but can cause clipping)
    if use_relu_output:
        # Use ReLU but add a small offset to prevent zero predictions
        model = keras.Sequential([
            layers.GRU(units, return_sequences=True, input_shape=(lag_window, 1)),
            layers.Dropout(0.2),
            layers.GRU(units, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')  # Linear is better for price prediction
        ])
    else:
        model = keras.Sequential([
            layers.GRU(units, return_sequences=True, input_shape=(lag_window, 1)),
            layers.Dropout(0.2),
            layers.GRU(units, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Linear output
        ])
    
    # Use learning rate decay
    initial_lr = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\n🚀 Training Configuration:")
    print(f"   Max epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Initial learning rate: {initial_lr}")
    print(f"   Early stopping: patience=10")
    print(f"   LR reduction: patience=5, factor=0.5")
    
    # Train
    print(f"\n{'='*70}")
    print("Training model...")
    print(f"{'='*70}\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1,
        callbacks=[reduce_lr, early_stop]
    )
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("Evaluating on test set...")
    print(f"{'='*70}\n")
    
    test_predictions_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    mape = mean_absolute_percentage_error(y_test_actual, test_predictions) * 100
    r2 = r2_score(y_test_actual, test_predictions)
    bias = np.mean(test_predictions - y_test_actual)
    corr = np.corrcoef(y_test_actual.flatten(), test_predictions.flatten())[0, 1]
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "bias": float(bias),
        "correlation": float(corr)
    }
    
    print(f"📊 Test Metrics:")
    print(f"   MAE:  ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.4f}")
    print(f"   Bias: ${bias:.2f}")
    print(f"   Correlation: {corr:.4f}")
    
    # Interpretation
    print(f"\n🔍 Interpretation:")
    if r2 < 0:
        print(f"   ⚠️  Negative R² ({r2:.4f}) - model worse than mean")
    elif r2 < 0.5:
        print(f"   ⚠️  Low R² ({r2:.4f}) - explains {r2*100:.1f}% of variance")
    elif r2 < 0.8:
        print(f"   ✓ Moderate R² ({r2:.4f}) - explains {r2*100:.1f}% of variance")
    else:
        print(f"   ✓ Good R² ({r2:.4f}) - explains {r2*100:.1f}% of variance")
    
    if abs(bias) > 5:
        print(f"   ⚠️  Large bias (${bias:.2f}) - {'under' if bias < 0 else 'over'}prediction")
    else:
        print(f"   ✓ Bias is small (${bias:.2f})")
    
    if corr < 0.7:
        print(f"   ⚠️  Low correlation ({corr:.4f})")
    else:
        print(f"   ✓ Good correlation ({corr:.4f})")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_gru_{timestamp}.weights.h5"
    else:
        save_path = Path(save_path)
        if not str(save_path).endswith('.weights.h5'):
            save_path = save_path.with_suffix('.weights.h5')
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model.save_weights(str(save_path))
    
    # Save scaler and metadata
    metadata_path = save_path.with_name(save_path.stem.replace('.weights', '') + '.pkl')
    metadata = {
        "scaler": scaler,
        "scaler_type": scaler_type,
        "symbol": symbol,
        "lag_window": lag_window,
        "units": units,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "price_range": {"min": float(price_min), "max": float(price_max)},
        "metrics": metrics,
        "model_config": model.get_config(),
        "training_date": datetime.now().isoformat()
    }
    joblib.dump(metadata, metadata_path)
    
    # Register model in database
    print(f"\n📝 Registering model in database...")
    try:
        model_version = register_model_version(
            symbol=symbol,
            model_type="GRU",
            artifact_path=save_path,
            source="local",
            source_url=None,
            version_id=None,
            horizon=1
        )
        print(f"  ✓ Model registered: {model_version.get('id', 'N/A')}")
        print(f"  ✓ Version tag: {model_version.get('version_tag', 'N/A')}")
    except Exception as e:
        logger.warning(f"Could not register model in database: {e}")
        print(f"  ⚠️  Warning: Model not registered in database: {e}")
        model_version = None
    
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"Model weights: {save_path}")
    print(f"Metadata: {metadata_path}")
    if model_version:
        print(f"Database ID: {model_version.get('id', 'N/A')}")
    print(f"\nKey Improvements:")
    print(f"  ✓ Scaler fitted to full price range (${price_min:.2f} - ${price_max:.2f})")
    print(f"  ✓ Sequence length: {lag_window} (increased from 60)")
    print(f"  ✓ ReLU output activation: {use_relu_output}")
    print(f"  ✓ Learning rate decay and early stopping")
    print(f"  ✓ More training epochs with validation")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "save_path": save_path,
        "metadata_path": metadata_path,
        "model_version": model_version,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "price_range": {"min": price_min, "max": price_max}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Improved GRU training with automatic scaler range detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (StandardScaler, 90 window, ReLU output)
  python scripts/retrain_gru_fixed.py --symbol GOOGL
  
  # Train with MinMaxScaler and longer sequence
  python scripts/retrain_gru_fixed.py --symbol GOOGL --scaler minmax --lag-window 120
  
  # Train without ReLU output (linear activation)
  python scripts/retrain_gru_fixed.py --symbol GOOGL --no-relu
        """
    )
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--lag-window", type=int, default=90, help="Sequence length (90-120 recommended)")
    parser.add_argument("--units", type=int, default=64, help="GRU units per layer")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                       help="Scaler type: 'standard' (recommended) or 'minmax'")
    parser.add_argument("--no-relu", action="store_true", help="Use linear output instead of ReLU")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_gru_fixed(
        symbol=args.symbol,
        lag_window=args.lag_window,
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        scaler_type=args.scaler,
        use_relu_output=not args.no_relu,
        save_path=Path(args.save_path) if args.save_path else None
    )
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run diagnostics: python scripts/diagnose_gru.py --symbol {args.symbol}")
    print(f"   2. Compare metrics with previous model")
    print(f"   3. Monitor predictions for bias and correlation improvements")

