"""
Improved Transformer retraining script with automatic scaler range detection and fixes.
Addresses:
- Scaler range mismatch (refits StandardScaler on latest dataset)
- Systematic underprediction bias (tracks validation bias per epoch)
- Activation sanity check (ensures linear output layer)
- Adaptive scaler drift detection
- Post-retrain diagnostics validation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from datetime import datetime
import joblib

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
from backend.utils.model_registry import register_model_version

settings = get_settings()
logger = get_logger(__name__)


def create_sequences(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for Transformer."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer encoder block."""
    # Multi-head attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed forward
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(inputs.shape[-1]),
    ])
    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return out2


class ValidationBiasCallback(tf.keras.callbacks.Callback):
    """Callback to track validation bias after each epoch."""
    
    def __init__(self, X_val, y_val, scaler):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler
        self.bias_history = []
        self.r2_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation set
        val_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # Inverse transform
        val_pred = self.scaler.inverse_transform(val_pred_scaled)
        val_true = self.scaler.inverse_transform(self.y_val.reshape(-1, 1))
        
        # Calculate bias
        bias = np.mean(val_pred - val_true)
        r2 = r2_score(val_true, val_pred)
        
        self.bias_history.append(float(bias))
        self.r2_history.append(float(r2))
        
        print(f"  Epoch {epoch + 1}: Validation Bias = ${bias:.2f}, R² = {r2:.4f}")


def detect_scaler_drift(scaler, current_prices, threshold_std=3.0):
    """
    Detect if current prices are drifting outside scaler's training range.
    
    Args:
        scaler: Fitted StandardScaler
        current_prices: Current price array
        threshold_std: Number of standard deviations to consider as drift
    
    Returns:
        dict with drift information
    """
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        return {"drift_detected": False, "reason": "Not a StandardScaler"}
    
    mean = scaler.mean_[0]
    std = scaler.scale_[0]
    
    current_mean = np.mean(current_prices)
    current_std = np.std(current_prices)
    current_max = np.max(current_prices)
    current_min = np.min(current_prices)
    
    # Calculate z-scores for current price range
    z_max = (current_max - mean) / std
    z_min = (current_min - mean) / std
    z_mean = (current_mean - mean) / std
    
    drift_detected = abs(z_max) > threshold_std or abs(z_min) > threshold_std or abs(z_mean) > threshold_std
    
    return {
        "drift_detected": drift_detected,
        "scaler_mean": float(mean),
        "scaler_std": float(std),
        "current_mean": float(current_mean),
        "current_std": float(current_std),
        "current_range": (float(current_min), float(current_max)),
        "z_scores": {
            "min": float(z_min),
            "max": float(z_max),
            "mean": float(z_mean)
        },
        "threshold": threshold_std
    }


def train_transformer_fixed(
    symbol: str,
    lag_window: int = 60,
    d_model: int = 64,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    test_size: float = 0.2,
    save_path: Path = None
):
    """
    Train improved Transformer model with automatic scaler range detection and fixes.
    
    Key improvements:
    - Refits StandardScaler on latest dataset (prices up to $290+)
    - Tracks validation bias after each epoch
    - Verifies optimizer checkpoint matches weights
    - Activation sanity check (ensures linear output layer)
    - Adaptive scaler drift detection
    - Post-retrain diagnostics validation
    
    Args:
        symbol: Stock symbol
        lag_window: Sequence length
        d_model: Model dimension
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate (default 1e-4)
        test_size: Proportion of data for testing
        save_path: Path to save model (.h5 file)
    
    Returns:
        dict with model, metrics, and save path
    """
    print(f"\n{'='*70}")
    print(f"Improved Transformer Training for {symbol}")
    print(f"{'='*70}\n")
    
    # Load data - get latest data including prices up to $290+
    df = None
    try:
        data_service = DataIngestionService()
        # Get more data to ensure we have latest prices
        df = data_service.get_prices(symbol, limit=2000)
        if df is None or len(df) == 0:
            print(f"⚠️  No data found in MongoDB for {symbol}, fetching from yfinance...")
            df = None
        else:
            print(f"✓ Found {len(df)} records in MongoDB")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
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
            
            if 'Close' not in df.columns:
                raise ValueError(f"Missing 'Close' column in fetched data")
            
            print(f"✓ Fetched {len(df)} records from yfinance")
            
            # Save to MongoDB for future use
            try:
                data_service = DataIngestionService()
                print(f"💾 Saving {len(df)} records to MongoDB...")
                data_service.ingest_from_dataframe(symbol, df)
                print(f"✓ Saved {len(df)} records to MongoDB")
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
    
    # Normalize column names
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    
    if 'Close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'Close': 'close'})
    if 'close' in df.columns:
        prices = df["close"].sort_index().values
    else:
        raise ValueError(f"Missing 'close' column. Available: {list(df.columns)}")
    
    print(f"\n📊 Data Summary:")
    print(f"   Total data points: {len(prices)}")
    print(f"   Price range: [${prices.min():.2f}, ${prices.max():.2f}]")
    print(f"   Current price: ${prices[-1]:.2f}")
    
    # 🔧 STEP 1: Refit StandardScaler using latest dataset
    print(f"\n🔧 Step 1: Refitting StandardScaler on latest dataset...")
    scaler = StandardScaler()
    scaler.fit(prices.reshape(-1, 1))
    
    print(f"   ✓ Scaler fitted on full data range:")
    print(f"     Mean: ${scaler.mean_[0]:.2f}, Std: ${scaler.scale_[0]:.2f}")
    print(f"     Price range: [${prices.min():.2f}, ${prices.max():.2f}]")
    print(f"     Current price z-score: {(prices[-1] - scaler.mean_[0]) / scaler.scale_[0]:.2f}")
    
    # ⚙️ STEP 2: Adaptive scaler drift detection
    print(f"\n⚙️  Step 2: Checking for scaler drift...")
    drift_info = detect_scaler_drift(scaler, prices, threshold_std=3.0)
    if drift_info["drift_detected"]:
        print(f"   ⚠️  Drift detected!")
        print(f"     Current prices are {drift_info['z_scores']['max']:.2f} std from scaler mean")
        print(f"     This is expected for new training - scaler will be refitted")
    else:
        print(f"   ✓ No significant drift detected")
        print(f"     Current prices within ±{drift_info['threshold']} std of scaler mean")
    
    # Data splitting
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"\n📈 Data Split:")
    print(f"   Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"   Train price range: [${train_data.min():.2f}, ${train_data.max():.2f}]")
    print(f"   Test price range: [${test_data.min():.2f}, ${test_data.max():.2f}]")
    
    # Scale data using the scaler fitted on all data
    train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lag_window)
    X_test, y_test = create_sequences(test_scaled, lag_window)
    
    if len(X_train) == 0:
        raise ValueError(f"Insufficient data after sequence creation")
    
    # Check if test set has enough sequences
    if len(X_test) == 0:
        print(f"⚠️  Warning: Test set too small. Using validation split from training data.")
        val_split = int(len(X_train) * 0.2)
        X_test = X_train[-val_split:]
        y_test = y_train[-val_split:]
        X_train = X_train[:-val_split]
        y_train = y_train[:-val_split]
        print(f"   Using last {len(X_test)} sequences from training set for testing")
    
    # Reshape for Transformer
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Split training data for validation
    val_split_idx = int(len(X_train) * 0.1)
    X_val = X_train[-val_split_idx:]
    y_val = y_train[-val_split_idx:]
    X_train = X_train[:-val_split_idx]
    y_train = y_train[:-val_split_idx]
    
    print(f"   Training sequences: {len(X_train)}")
    print(f"   Validation sequences: {len(X_val)}")
    print(f"   Test sequences: {len(X_test)}")
    
    # 🧩 STEP 3: Build model with activation sanity check
    print(f"\n🧩 Step 3: Building Transformer model...")
    inputs = tf.keras.layers.Input(shape=(lag_window, 1))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = transformer_block(x, head_size=d_model // 2, num_heads=4, ff_dim=128)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # ✅ Activation sanity check: Use linear activation (no tanh/sigmoid)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)  # Explicitly linear
    
    model = tf.keras.Model(inputs, outputs)
    
    # Verify output layer activation
    output_layer = model.layers[-1]
    if hasattr(output_layer, 'activation'):
        activation_name = output_layer.activation.__name__ if callable(output_layer.activation) else str(output_layer.activation)
        print(f"   ✓ Output layer activation: {activation_name}")
        if activation_name in ['tanh', 'sigmoid']:
            print(f"   ⚠️  WARNING: Output layer uses {activation_name} - switching to linear")
            # Rebuild with linear activation
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
            model = tf.keras.Model(inputs, outputs)
            print(f"   ✓ Fixed: Output layer now uses linear activation")
    else:
        print(f"   ✓ Output layer: linear (default)")
    
    # Compile with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    print(f"   ✓ Model compiled with learning rate: {learning_rate}")
    print(f"   Model summary:")
    model.summary()
    
    # 📈 STEP 4: Train with validation bias tracking
    print(f"\n📈 Step 4: Training for {epochs} epochs...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    
    # Callbacks
    callbacks = []
    
    # Validation bias callback
    bias_callback = ValidationBiasCallback(X_val, y_val, scaler)
    callbacks.append(bias_callback)
    
    # Learning rate reduction
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(lr_callback)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Model checkpoint to verify optimizer matches weights
    checkpoint_path = settings.models_dir / f"{symbol}_transformer_checkpoint.weights.h5"
    checkpoint_callback = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )
    callbacks.append(checkpoint_callback)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 🧮 STEP 5: Verify optimizer checkpoint matches weights
    print(f"\n🧮 Step 5: Verifying optimizer checkpoint...")
    if checkpoint_path.exists():
        # Load checkpoint weights
        checkpoint_model = tf.keras.models.clone_model(model)
        checkpoint_model.compile(optimizer=optimizer, loss='mse')
        checkpoint_model.load_weights(str(checkpoint_path))
        
        # Compare weights
        main_weights = model.get_weights()
        checkpoint_weights = checkpoint_model.get_weights()
        
        weights_match = all(
            np.allclose(mw, cw, rtol=1e-5) 
            for mw, cw in zip(main_weights, checkpoint_weights)
        )
        
        if weights_match:
            print(f"   ✓ Checkpoint weights match current model weights")
        else:
            print(f"   ⚠️  Checkpoint weights differ from current model")
            print(f"   → Using best checkpoint weights")
            model.load_weights(str(checkpoint_path))
    else:
        print(f"   ℹ️  No checkpoint found (first training run)")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_predictions_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    mape = mean_absolute_percentage_error(y_test_actual, test_predictions) * 100
    r2 = r2_score(y_test_actual, test_predictions)
    bias = np.mean(test_predictions - y_test_actual)
    pearson_r, _ = pearsonr(y_test_actual.flatten(), test_predictions.flatten())
    
    # Calculate normalized prediction stats
    norm_pred_mean = np.mean(test_predictions_scaled)
    norm_pred_std = np.std(test_predictions_scaled)
    pred_range = np.max(test_predictions) - np.min(test_predictions)
    true_range = np.max(y_test_actual) - np.min(y_test_actual)
    range_ratio = pred_range / true_range if true_range > 0 else 0
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "bias": float(bias),
        "pearson_r": float(pearson_r),
        "norm_pred_mean": float(norm_pred_mean),
        "norm_pred_std": float(norm_pred_std),
        "range_ratio": float(range_ratio),
        "bias_history": bias_callback.bias_history,
        "r2_history": bias_callback.r2_history
    }
    
    print(f"\n{'='*70}")
    print(f"📊 Test Set Metrics:")
    print(f"{'='*70}")
    print(f"   MAE:  ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.4f}")
    print(f"   Bias: ${bias:.2f}")
    print(f"   Pearson r: {pearson_r:.4f}")
    print(f"   Normalized pred mean: {norm_pred_mean:.4f}")
    print(f"   Normalized pred std:  {norm_pred_std:.4f}")
    print(f"   Range ratio: {range_ratio:.3f}")
    
    # 🧮 STEP 6: Validate post-retrain diagnostics
    print(f"\n{'='*70}")
    print(f"🧮 Step 6: Post-Retrain Diagnostics Validation")
    print(f"{'='*70}")
    
    validation_passed = True
    validation_issues = []
    
    # Check 1: Bias < ±2
    if abs(bias) >= 2:
        validation_passed = False
        validation_issues.append(f"❌ Bias = ${bias:.2f} (expected < ±$2)")
    else:
        print(f"   ✓ Bias = ${bias:.2f} (within ±$2)")
    
    # Check 2: R² > 0.7
    if r2 < 0.7:
        validation_passed = False
        validation_issues.append(f"❌ R² = {r2:.4f} (expected > 0.7)")
    else:
        print(f"   ✓ R² = {r2:.4f} (above 0.7)")
    
    # Check 3: Predicted range ≈ True range
    if range_ratio < 0.5 or range_ratio > 2.0:
        validation_passed = False
        validation_issues.append(f"❌ Range ratio = {range_ratio:.3f} (expected 0.5-2.0)")
    else:
        print(f"   ✓ Range ratio = {range_ratio:.3f} (within 0.5-2.0)")
    
    # Check 4: Normalized std > 0.02
    if norm_pred_std < 0.02:
        validation_passed = False
        validation_issues.append(f"❌ Normalized std = {norm_pred_std:.4f} (expected > 0.02)")
    else:
        print(f"   ✓ Normalized std = {norm_pred_std:.4f} (above 0.02)")
    
    if validation_passed:
        print(f"\n   ✅ All validation checks passed!")
    else:
        print(f"\n   ⚠️  Validation issues detected:")
        for issue in validation_issues:
            print(f"      {issue}")
        print(f"   → Consider:")
        print(f"     - Increasing training epochs")
        print(f"     - Adjusting learning rate")
        print(f"     - Checking data quality")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_transformer_{timestamp}.weights.h5"
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
        "symbol": symbol,
        "lag_window": lag_window,
        "d_model": d_model,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "metrics": metrics,
        "model_config": model.get_config(),
        "learning_rate": learning_rate,
        "validation_passed": validation_passed
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"\n{'='*70}")
    print(f"💾 Model Saved:")
    print(f"{'='*70}")
    print(f"   Weights: {save_path}")
    print(f"   Metadata: {metadata_path}")
    
    # Register model version
    try:
        from backend.utils.model_registry import register_model_version
        model_version = register_model_version(
            symbol=symbol,
            model_type="Transformer",
            artifact_path=save_path,
            source="local",
            horizon=1
        )
        print(f"   ✓ Model version registered: {model_version.get('id', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Could not register model version: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "save_path": save_path,
        "metadata_path": metadata_path,
        "validation_passed": validation_passed
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train improved Transformer model")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--lag-window", type=int, default=60, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    result = train_transformer_fixed(
        symbol=args.symbol,
        lag_window=args.lag_window,
        d_model=args.d_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        test_size=args.test_size,
        save_path=Path(args.save_path) if args.save_path else None
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Model: {result['save_path']}")
    print(f"   Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
    print(f"   Metrics: MAE=${result['metrics']['mae']:.2f}, R²={result['metrics']['r2']:.4f}, Bias=${result['metrics']['bias']:.2f}")

