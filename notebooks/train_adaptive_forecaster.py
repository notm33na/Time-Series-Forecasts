"""
Training script for Adaptive Forecaster (SGD-based).
Handles data splitting, model training, evaluation, and saving.

⚠️ Local training is disabled. This script must be run on Hugging Face Spaces or Notebooks.
For local execution, use --remote-train to trigger a remote training job.
"""

import sys
import os
from pathlib import Path

# Backend imports will be loaded conditionally after environment check
# This allows the script to show "local training disabled" message
# even if backend imports fail


def is_remote_environment() -> bool:
    """
    Check if running in a remote Hugging Face environment.
    
    Detects Hugging Face Spaces or Notebooks by checking for:
    - SPACE_ID (set automatically in Hugging Face Spaces)
    - HF_TOKEN (Hugging Face authentication token)
    
    Note: CLOUD_TRAINING is a permission flag, not an environment detection flag.
    
    Returns:
        True if running on Hugging Face Spaces/Notebooks, False otherwise
    """
    # Primary detection: SPACE_ID is automatically set in HF Spaces
    if os.getenv("SPACE_ID"):
        return True
    
    # Secondary detection: HF_TOKEN indicates HF environment
    if os.getenv("HF_TOKEN"):
        return True
    
    return False


def load_backend_imports():
    """
    Load backend modules conditionally.
    This is called only when we're actually going to train or use backend functionality.
    """
    # Add project root to path (so backend can be imported as a package)
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import backend modules
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from datetime import datetime
    
    from backend.config import get_settings
    from backend.services.data_ingestion import DataIngestionService
    
    settings = get_settings()
    
    return {
        'pd': pd,
        'np': np,
        'joblib': joblib,
        'SGDRegressor': SGDRegressor,
        'StandardScaler': StandardScaler,
        'mean_absolute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'datetime': datetime,
        'settings': settings,
        'DataIngestionService': DataIngestionService
    }


def authenticate_huggingface() -> None:
    """
    Authenticate with Hugging Face Hub using HF_TOKEN.
    
    This function automatically logs in when running in a remote environment.
    Raises an error if HF_TOKEN is not found when authentication is required.
    
    Raises:
        RuntimeError: If HF_TOKEN is not set in the environment
    """
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise RuntimeError(
            "❌ HF_TOKEN not found in environment variables.\n"
            "   To train on Hugging Face Spaces/Notebooks, you must set HF_TOKEN.\n"
            "   Get your token from: https://huggingface.co/settings/tokens\n"
            "   Then set it as an environment variable or in your Space secrets."
        )
    
    try:
        from huggingface_hub import login
        print("🔐 Authenticating with Hugging Face Hub...")
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Successfully authenticated with Hugging Face Hub")
    except ImportError:
        print("⚠️  Warning: 'huggingface_hub' not installed. Install with: pip install huggingface_hub")
        print("   Continuing without explicit login (token may be used automatically)...")
    except Exception as e:
        print(f"⚠️  Warning: Failed to authenticate with Hugging Face Hub: {e}")
        print("   Continuing anyway (token may be used automatically)...")


def trigger_remote_training(
    symbol: str,
    lag_window: int = 24,
    horizon: int = 1,
    learning_rate: str = "adaptive",
    test_size: float = 0.2,
    hf_space_url: str = None
) -> None:
    """
    Trigger a remote training job on Hugging Face Spaces.
    
    Args:
        symbol: Stock symbol
        lag_window: Number of lag features
        horizon: Forecast horizon
        learning_rate: Learning rate type
        test_size: Proportion of data for testing
        hf_space_url: Hugging Face Space URL (e.g., 'https://huggingface.co/spaces/username/space-name')
    """
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library is required for remote training.")
        print("   Install it with: pip install requests")
        raise ImportError(
            "The 'requests' library is required for remote training. "
            "Install it with: pip install requests"
        )
    
    # Default space URL if not provided
    if not hf_space_url:
        hf_username = os.getenv("HF_USERNAME", "Zarm33na")
        space_name = os.getenv("HF_SPACE_NAME", "NLP")
        hf_space_url = f"https://huggingface.co/spaces/{hf_username}/{space_name}"
    
    # Construct training endpoint
    # Note: Your Space needs to have a training API endpoint configured
    # Common options: /api/predict (Gradio), /api/train, or custom endpoint
    custom_endpoint = os.getenv("HF_TRAIN_ENDPOINT")
    if custom_endpoint:
        # Use custom endpoint if provided
        if custom_endpoint.startswith("http"):
            train_endpoint = custom_endpoint
        else:
            train_endpoint = f"{hf_space_url}{custom_endpoint}" if custom_endpoint.startswith("/") else f"{hf_space_url}/{custom_endpoint}"
    else:
        # Default: try Gradio API endpoint
        train_endpoint = f"{hf_space_url}/api/predict"
    
    # Prepare training parameters
    params = {
        "symbol": symbol,
        "lag_window": lag_window,
        "horizon": horizon,
        "learning_rate": learning_rate,
        "test_size": test_size,
        "model_type": "adaptive_forecaster"
    }
    
    print(f"🚀 Triggering remote training job on Hugging Face...")
    print(f"   Space: {hf_space_url}")
    print(f"   Endpoint: {train_endpoint}")
    print(f"   Symbol: {symbol}")
    print(f"   Parameters: {params}")
    print()
    print(f"💡 Note: If this fails, you may need to:")
    print(f"   1. Set up a training API endpoint in your Space")
    print(f"   2. Use --hf-space-url with the full endpoint URL")
    print(f"   3. Or set HF_TRAIN_ENDPOINT environment variable")
    print()
    
    try:
        # Send POST request to trigger training
        response = requests.post(
            train_endpoint,
            json=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code in [200, 202]:
            print(f"✅ Remote training job triggered successfully!")
            print(f"   Job URL: {hf_space_url}")
            print(f"   Check the Space logs for training progress.")
            
            # Try to extract job ID if available
            try:
                job_data = response.json()
                if "job_id" in job_data:
                    print(f"   Job ID: {job_data['job_id']}")
            except:
                pass
        else:
            print(f"⚠️  Warning: Received status code {response.status_code}")
            print(f"   Response: {response.text}")
            print(f"   You may need to check the Space configuration.")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to trigger remote training: {e}")
        print(f"\n💡 Alternative: Run this script directly on Hugging Face Spaces/Notebooks:")
        print(f"   python train_adaptive_forecaster.py --symbol {symbol} \\")
        print(f"       --lag-window {lag_window} --horizon {horizon}")
        raise


def create_features(series, lag_window: int, horizon: int, np):
    """Create lag features from time series."""
    if len(series) < lag_window + horizon:
        raise ValueError(f"Series too short: need at least {lag_window + horizon} points")

    X, y = [], []
    for i in range(lag_window, len(series) - horizon + 1):
        X.append(series.iloc[i - lag_window : i].values)
        y.append(series.iloc[i + horizon - 1])

    return np.array(X), np.array(y)


def train_adaptive_forecaster(
    symbol: str,
    lag_window: int = 24,
    horizon: int = 1,
    learning_rate: str = "adaptive",
    test_size: float = 0.2,
    save_path: Path = None,
    skip_db: bool = False,
    save_to_db: bool = False,
    allow_local: bool = False
):
    """
    Train Adaptive Forecaster with data splitting, evaluation, and saving.
    
    ⚠️ By default, this function only runs in remote Hugging Face environments.
    Use allow_local=True to enable local training.
    
    Args:
        symbol: Stock symbol
        lag_window: Number of lag features
        horizon: Forecast horizon
        learning_rate: Learning rate type
        test_size: Proportion of data for testing
        save_path: Path to save model (.pkl file)
        skip_db: Skip MongoDB and fetch from yfinance
        save_to_db: Force save data to MongoDB
        allow_local: Allow training on local hardware (bypasses remote checks)
    
    Returns:
        dict with model, metrics, and save path
    
    Raises:
        RuntimeError: If called in a local environment without allow_local=True
    """
    # Step 1: Check if running in remote environment (unless local training is allowed)
    space_id = os.getenv("SPACE_ID")
    hf_token = os.getenv("HF_TOKEN")
    
    if allow_local:
        print("⚠️  Local training enabled (--allow-local-training flag)")
        print("   Training will proceed on local hardware.")
        print()
    else:
        # Remote environment checks
        if not space_id and not hf_token:
            raise RuntimeError(
                "❌ Not running in a remote Hugging Face environment.\n"
                "   Required: SPACE_ID (automatically set in HF Spaces) or HF_TOKEN\n"
                "   Local training is disabled by default.\n"
                "   Options:\n"
                "   1. Use --allow-local-training to train locally\n"
                "   2. Use --remote-train to trigger a remote training job\n"
                "   3. Run on Hugging Face Spaces/Notebooks"
            )
        
        # Step 2: Check for CLOUD_TRAINING flag (required to allow training)
        if os.getenv("CLOUD_TRAINING") != "true":
            raise RuntimeError(
                "⚠️ Training is disabled. Set CLOUD_TRAINING=true to enable training.\n"
                "   In Hugging Face Spaces, add CLOUD_TRAINING=true to your environment variables/secrets."
            )
        
        # Step 3: Verify HF_TOKEN is present for authentication
        if not hf_token:
            raise RuntimeError(
                "❌ HF_TOKEN not found. Authentication is required for training.\n"
                "   Set HF_TOKEN in your environment variables or Space secrets.\n"
                "   Get your token from: https://huggingface.co/settings/tokens"
            )
        
        # Authenticate with Hugging Face Hub
        authenticate_huggingface()
    
    # Load backend imports (only when actually training)
    imports = load_backend_imports()
    pd = imports['pd']
    np = imports['np']
    joblib = imports['joblib']
    SGDRegressor = imports['SGDRegressor']
    StandardScaler = imports['StandardScaler']
    mean_absolute_error = imports['mean_absolute_error']
    mean_squared_error = imports['mean_squared_error']
    mean_absolute_percentage_error = imports['mean_absolute_percentage_error']
    datetime = imports['datetime']
    settings = imports['settings']
    DataIngestionService = imports['DataIngestionService']
    
    print(f"Training Adaptive Forecaster for {symbol} (lag={lag_window}, horizon={horizon})")
    if allow_local:
        print(f"✓ Running locally (--allow-local-training enabled)")
    else:
        print(f"✓ Running in remote environment (Hugging Face)")
        print(f"✓ CLOUD_TRAINING enabled")
    
    # Load data - try MongoDB first, fallback to yfinance if empty
    df = None
    use_db = not skip_db  # Skip DB if flag is set
    
    if skip_db:
        print(f"⏭️  Skipping MongoDB (--skip-db flag set), fetching from yfinance...")
    else:
        try:
            data_service = DataIngestionService()
            df = data_service.get_prices(symbol)
            if df is None or len(df) == 0:
                print(f"⚠️  No data found in MongoDB for {symbol}, fetching from yfinance...")
                use_db = False
            else:
                print(f"✓ Found {len(df)} records in MongoDB")
        except Exception as e:
            print(f"⚠️  Could not fetch from MongoDB: {e}")
            print(f"   Falling back to yfinance...")
            use_db = False
    
    # If MongoDB is empty or unavailable, fetch directly from yfinance
    if not use_db or df is None or len(df) == 0:
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
            
            # Save to MongoDB for future use (required)
            print(f"💾 Saving {len(df)} records to MongoDB...")
            try:
                data_service = DataIngestionService()
                data_service.ingest_from_dataframe(symbol, df)
                print(f"✅ Successfully saved {len(df)} records to MongoDB")
            except Exception as e:
                print(f"❌ WARNING: Could not save to MongoDB: {e}")
                print(f"   Training will continue, but data won't be available for future runs.")
                print(f"   Please check MongoDB connection and try again.")
                # Don't fail training, but warn the user
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
    
    # Validate data size
    if len(df) < lag_window + horizon + 100:
        raise ValueError(
            f"Insufficient data: need at least {lag_window + horizon + 100} points, "
            f"but got {len(df)}"
        )
    
    # Save to MongoDB if requested (even if data came from MongoDB - useful for refreshing)
    if save_to_db:
        print(f"💾 Force saving {len(df)} records to MongoDB (--save-to-db flag)...")
        try:
            data_service = DataIngestionService()
            data_service.ingest_from_dataframe(symbol, df)
            print(f"✅ Successfully saved {len(df)} records to MongoDB")
        except Exception as e:
            print(f"❌ WARNING: Could not save to MongoDB: {e}")
            print(f"   Training will continue, but data may not be updated in MongoDB.")
    
    # Normalize column names (handle both MongoDB lowercase and yfinance capitalized)
    # MongoDB uses: close, open, high, low, volume
    # yfinance uses: Close, Open, High, Low, Volume
    column_mapping = {
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume',
        'Date': 'date'
    }
    
    # Rename columns to lowercase if they exist in capitalized form
    for cap_col, low_col in column_mapping.items():
        if cap_col in df.columns and low_col not in df.columns:
            df = df.rename(columns={cap_col: low_col})
    
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
    
    # Extract prices - try both 'close' and 'Close'
    if 'close' in df.columns:
        prices = df["close"].sort_index()
    elif 'Close' in df.columns:
        prices = df["Close"].sort_index()
    else:
        raise ValueError(
            f"Missing 'close' or 'Close' column in data. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Data splitting
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create features
    X_train, y_train = create_features(train_data, lag_window, horizon, np)
    X_test, y_test = create_features(test_data, lag_window, horizon, np)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (only runs in remote environment)
    print("Training SGD model...")
    model = SGDRegressor(
        learning_rate=learning_rate,
        random_state=42,
        warm_start=True,
        max_iter=1000,
        tol=1e-3,
    )
    # This fit() call only executes in remote environments due to the check above
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mape = mean_absolute_percentage_error(y_test, test_predictions) * 100
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }
    
    print(f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = settings.models_dir / f"{symbol}_adaptive_{timestamp}.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_state = {
        "model": model,
        "scaler": scaler,
        "symbol": symbol,
        "lag_window": lag_window,
        "horizon": horizon,
        "learning_rate": learning_rate,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "metrics": metrics
    }
    
    joblib.dump(model_state, save_path)
    
    print(f"Model saved to {save_path}")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "save_path": save_path,
        "train_size": len(train_data),
        "test_size": len(test_data)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Adaptive Forecaster",
        epilog="""
⚠️  Local training is disabled by default. Use --allow-local-training to train locally.

Examples:
  # Local execution (shows message to train remotely)
  python train_adaptive_forecaster.py --symbol AAPL

  # Enable local training
  python train_adaptive_forecaster.py --allow-local-training --symbol AAPL

  # Trigger remote training job (uses Zarm33na/NLP by default)
  python train_adaptive_forecaster.py --remote-train --symbol AAPL
  
  # Trigger with custom Space URL
  python train_adaptive_forecaster.py --remote-train --symbol AAPL --hf-space-url https://huggingface.co/spaces/Zarm33na/NLP

  # On Hugging Face Spaces/Notebooks (trains normally)
  python train_adaptive_forecaster.py --symbol AAPL --lag-window 24 --horizon 1
  
  # Skip MongoDB and fetch directly from yfinance
  python train_adaptive_forecaster.py --symbol AAPL --skip-db
  
  # Force save data to MongoDB (even if data exists)
  python train_adaptive_forecaster.py --symbol AAPL --save-to-db
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--lag-window", type=int, default=24, help="Lag window size")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--learning-rate", type=str, default="adaptive", help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    parser.add_argument(
        "--remote-train",
        action="store_true",
        help="Trigger a remote training job on Hugging Face Spaces instead of training locally"
    )
    parser.add_argument(
        "--hf-space-url",
        type=str,
        default=None,
        help="Hugging Face Space URL (e.g., 'https://huggingface.co/spaces/username/space-name')"
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip MongoDB and fetch data directly from yfinance (useful when MongoDB is empty)"
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Force save fetched data to MongoDB (useful for refreshing/updating data)"
    )
    parser.add_argument(
        "--allow-local-training",
        action="store_true",
        help="Allow training on local hardware (bypasses remote environment checks)"
    )
    
    args = parser.parse_args()
    
    # Check if user wants to trigger remote training
    if args.remote_train:
        print("=" * 70)
        print("🚀 REMOTE TRAINING MODE")
        print("=" * 70)
        trigger_remote_training(
            symbol=args.symbol,
            lag_window=args.lag_window,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            test_size=args.test_size,
            hf_space_url=args.hf_space_url
        )
        sys.exit(0)
    
    # Check if running locally (not in remote environment)
    if not is_remote_environment() and not args.allow_local_training:
        print("=" * 70)
        print("⚠️  LOCAL TRAINING IS DISABLED")
        print("=" * 70)
        print()
        print("This script is configured to prevent training on local hardware by default.")
        print("Please run this training job on Hugging Face Spaces or Notebooks.")
        print()
        print("Options:")
        print("  1. Enable local training:")
        print(f"     python train_adaptive_forecaster.py --allow-local-training --symbol {args.symbol}")
        print()
        print("  2. Trigger a remote training job:")
        print(f"     python train_adaptive_forecaster.py --remote-train --symbol {args.symbol}")
        print()
        print("  3. Run directly on Hugging Face Spaces/Notebooks:")
        print("     - Set environment variables:")
        print("       * SPACE_ID (automatically set in HF Spaces)")
        print("       * HF_TOKEN (required for authentication)")
        print("       * CLOUD_TRAINING=true (required to enable training)")
        print("     - Then run: python train_adaptive_forecaster.py --symbol <SYMBOL>")
        print()
        print("  4. For more information, see:")
        print("     notebooks/HUGGINGFACE_TRAINING_GUIDE.md")
        print()
        print("=" * 70)
        sys.exit(1)
    
    # Proceed with training (either remote or local if allowed)
    if args.allow_local_training:
        print("=" * 70)
        print("⚠️  LOCAL TRAINING MODE")
        print("=" * 70)
        print()
    else:
        print("=" * 70)
        print("✓ Running in remote environment - Training enabled")
        print("=" * 70)
        print()
    
    try:
        result = train_adaptive_forecaster(
            symbol=args.symbol,
            lag_window=args.lag_window,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            test_size=args.test_size,
            save_path=Path(args.save_path) if args.save_path else None,
            skip_db=args.skip_db,
            save_to_db=args.save_to_db,
            allow_local=args.allow_local_training
        )
        
        print()
        print("=" * 70)
        print("✅ Training complete!")
        print("=" * 70)
        print(f"Model saved to: {result['save_path']}")
        print(f"Metrics: {result['metrics']}")
        print("=" * 70)
        
    except RuntimeError as e:
        # This should not happen if is_remote_environment() check passed,
        # but handle it gracefully just in case
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

