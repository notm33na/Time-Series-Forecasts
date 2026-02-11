"""
Training script for Ensemble Model.

The ensemble combines multiple pre-trained models:
- ARIMA
- Exponential Smoothing
- LSTM
- GRU
- Transformer

IMPORTANT: You must train all individual models first before using the ensemble.
The ensemble loads pre-trained models and combines their predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import joblib
import warnings

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService
from backend.models.adaptive_ensemble import AdaptiveEnsemble, create_default_ensemble
from backend.services.model_loader import ModelLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Suppress statsmodels index warnings (they're informational, not errors)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

settings = get_settings()


def train_ensemble(
    symbol: str,
    horizon: int = 24,
    test_size: float = 0.2,
    save_path: Path = None
):
    """
    Train an ensemble model by loading pre-trained individual models.
    
    Args:
        symbol: Stock symbol
        horizon: Forecast horizon (number of steps)
        test_size: Proportion of data for testing
        save_path: Path to save ensemble metadata
    
    Returns:
        dict with ensemble, metrics, and save path
    """
    print(f"Training Ensemble for {symbol} (horizon={horizon})")
    
    # Load data
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
    
    # Set date as index if it exists
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    elif df.index.name != 'date' and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except:
            pass
    
    # Ensure index is DatetimeIndex and sorted for time series models
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except:
            # If conversion fails, create a simple range index
            # This will cause warnings but won't break functionality
            pass
    
    # Normalize column names
    if 'Close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'Close': 'close'})
    if 'close' in df.columns:
        prices = df["close"].sort_index()
    elif 'Close' in df.columns:
        prices = df["Close"].sort_index()
    else:
        raise ValueError(f"Missing 'close' or 'Close' column. Available: {list(df.columns)}")
    
    if len(prices) < horizon + 100:
        raise ValueError(f"Insufficient data: need at least {horizon + 100} points, got {len(prices)}")
    
    # Split data
    split_idx = int(len(prices) * (1 - test_size))
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Try to load pre-trained models first
    print("\n📦 Attempting to load pre-trained models...")
    model_loader = ModelLoader()
    loaded_models = {}
    
    # First try loading from database
    try:
        loaded_models = model_loader.load_ensemble_models(symbol, horizon)
        print(f"✓ Loaded {len(loaded_models)} pre-trained models from database")
    except Exception as e:
        print(f"⚠️  Could not load from database: {e}")
        print(f"   Searching filesystem for trained models...")
        
        # Try loading from filesystem
        from backend.models.forecasting_models import (
            ARIMAForecaster,
            ExponentialSmoothingForecaster,
            LSTMForecaster,
            GRUForecaster,
            TransformerForecaster,
        )
        
        model_type_map = {
            "ARIMA": (ARIMAForecaster, ["*.pkl"]),
            "ExponentialSmoothing": (ExponentialSmoothingForecaster, ["*.pkl"]),
            "LSTM": (LSTMForecaster, ["*.weights.h5", "*.h5"]),
            "GRU": (GRUForecaster, ["*.weights.h5", "*.h5"]),
            "Transformer": (TransformerForecaster, ["*.weights.h5", "*.h5"]),
        }
        
        # Check multiple possible locations
        settings = get_settings()  # Get settings in this scope
        possible_dirs = [
            settings.models_dir,
            Path(__file__).parent.parent / "backend" / "artifacts",
            Path(__file__).parent.parent / "artifacts",
        ]
        
        models_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                models_dir = dir_path
                print(f"   Searching in: {models_dir}")
                break
        
        if not models_dir:
            print(f"   ⚠️  Models directory not found in any of: {possible_dirs}")
        else:
            # Search for model files matching the symbol
            for model_type, (model_class, patterns) in model_type_map.items():
                found = False
                
                # For neural models (LSTM, GRU, Transformer), search for .pkl files first
                # because the naming convention is: {base}.pkl and {base}.weights.h5
                if model_type in ["LSTM", "GRU", "Transformer"]:
                    # Search for .pkl metadata files
                    pkl_pattern = f"{symbol}_{model_type.lower()}_*.pkl"
                    pkl_files = list(models_dir.glob(pkl_pattern))
                    
                    # Filter out .weights.pkl files (we want the base .pkl)
                    pkl_files = [f for f in pkl_files if '.weights.pkl' not in f.name]
                    
                    if pkl_files:
                        # Sort by modification time (newest first)
                        pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        pkl_path = pkl_files[0]
                        
                        # Find corresponding .weights.h5 file
                        # The .pkl is named: AAPL_lstm_20251110_184158.pkl
                        # The .h5 should be: AAPL_lstm_20251110_184158.weights.h5
                        base_name = pkl_path.stem
                        h5_path = pkl_path.parent / f"{base_name}.weights.h5"
                        
                        if h5_path.exists():
                            try:
                                print(f"   Found {model_type}: {h5_path.name} (with {pkl_path.name})")
                                # Load using the .h5 file (load_model will find the .pkl)
                                model = model_class.load_model(h5_path)
                                found = True
                            except Exception as load_error:
                                print(f"   ⚠️  Could not load {model_type}: {load_error}")
                                continue
                
                # For non-neural models or if neural model search failed, use original pattern search
                if not found:
                    for pattern in patterns:
                        # Skip .h5 patterns for neural models (already tried above)
                        if model_type in ["LSTM", "GRU", "Transformer"] and pattern.endswith('.h5'):
                            continue
                        
                        # Search for files like: {symbol}_{model_type_lower}_*.{ext}
                        search_pattern = f"{symbol}_{model_type.lower()}_*{pattern}"
                        matching_files = list(models_dir.glob(search_pattern))
                        
                        if not matching_files:
                            # Try alternative pattern: {symbol}_{model_type}_*.{ext}
                            search_pattern = f"{symbol}_{model_type}_*{pattern}"
                            matching_files = list(models_dir.glob(search_pattern))
                        
                        # Sort by modification time (newest first)
                        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        
                        if matching_files:
                            try:
                                model_path = matching_files[0]
                                print(f"   Found {model_type}: {model_path.name}")
                                model = model_class.load_model(model_path)
                                found = True
                                break
                            except Exception as load_error:
                                print(f"   ⚠️  Could not load {model_type} from {matching_files[0]}: {load_error}")
                                continue
                
                if found:
                    # Create a dummy model_version dict for compatibility (MongoDB format)
                    # Use the appropriate path (h5_path for neural models, model_path for others)
                    if model_type in ["LSTM", "GRU", "Transformer"] and 'h5_path' in locals():
                        artifact_path = h5_path
                    else:
                        artifact_path = model_path if 'model_path' in locals() else matching_files[0] if 'matching_files' in locals() and matching_files else None
                    
                    if artifact_path:
                        dummy_version = {
                            "id": "0",
                            "symbol": symbol,
                            "horizon": horizon,
                            "version_tag": f"{symbol}_{model_type}_filesystem",
                            "trained_at": datetime.now(timezone.utc),
                            "artifact_path": str(artifact_path),
                            "training_info": {"model_type": model_type}
                        }
                        loaded_models[model_type] = (model, dummy_version)
                
                if not found:
                    print(f"   ⚠️  No {model_type} model found for {symbol}")
    
    # Create ensemble from loaded models
    if loaded_models:
        print(f"\n✓ Found {len(loaded_models)} pre-trained models: {', '.join(loaded_models.keys())}")
        ensemble = model_loader.create_ensemble_from_loaded_models(
            symbol, horizon, loaded_models
        )
        print(f"✓ Created ensemble from {len(loaded_models)} pre-trained models")
    else:
        raise ValueError(
            f"No trained models found for {symbol}.\n"
            f"Please train individual models first:\n"
            f"  python train_arima.py --symbol {symbol}\n"
            f"  python train_lstm.py --symbol {symbol} --epochs 50\n"
            f"  python train_gru.py --symbol {symbol} --epochs 50\n"
            f"  python train_transformer.py --symbol {symbol} --epochs 50\n"
            f"  python train_exponential_smoothing.py --symbol {symbol}"
        )
    
    # Check if ensemble has any fitted models
    fitted_count = sum(1 for f in ensemble.forecasters if hasattr(f, 'is_fitted') and f.is_fitted)
    print(f"\n📊 Ensemble has {fitted_count}/{len(ensemble.forecasters)} fitted models")
    
    if fitted_count == 0:
        print("⚠️  No models are fitted. Attempting to fit ensemble again...")
        ensemble.fit(train_data)
        fitted_count = sum(1 for f in ensemble.forecasters if hasattr(f, 'is_fitted') and f.is_fitted)
        print(f"   After refitting: {fitted_count}/{len(ensemble.forecasters)} models fitted")
    
    if fitted_count == 0:
        raise ValueError(
            "No models could be fitted in the ensemble. "
            "Please ensure individual models are trained first, or check model configuration."
        )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating ensemble on test set...")
    test_predictions = []
    test_actuals = []
    
    # Make predictions for each point in test set (simplified: just use last part of test set)
    # Use a simpler evaluation: predict on the last portion of test data
    eval_window = min(20, len(test_data) - horizon)  # Evaluate on last 20 points or available
    
    for i in range(eval_window):
        # Use data up to current point in test set
        historical_data = pd.concat([train_data, test_data[:i+1]])
        
        # Ensure proper DatetimeIndex for time series models
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            try:
                historical_data.index = pd.to_datetime(historical_data.index)
            except:
                # If index can't be converted, create a simple range index
                # This will cause warnings but won't break functionality
                pass
        
        try:
            pred = ensemble.predict(historical_data, steps=horizon)
            actual = test_data.iloc[i+1:i+1+horizon].values if i+1+horizon <= len(test_data) else test_data.iloc[i+1:].values
            
            # Handle case where we don't have enough future data
            if len(actual) < horizon:
                # Use what we have
                pred = pred[:len(actual)]
            
            if len(pred) == len(actual) and len(pred) > 0:
                test_predictions.append(pred)
                test_actuals.append(actual)
        except Exception as e:
            # Only print first few errors to avoid spam
            if i < 3:
                print(f"⚠️  Prediction failed at index {i}: {e}")
            continue
    
    if not test_predictions:
        # Try a simpler prediction: just predict from the end of training data
        print("⚠️  Rolling window predictions failed. Trying single prediction from training data...")
        try:
            # Ensure proper DatetimeIndex for time series models
            train_data_indexed = train_data.copy()
            if not isinstance(train_data_indexed.index, pd.DatetimeIndex):
                try:
                    train_data_indexed.index = pd.to_datetime(train_data_indexed.index)
                except:
                    pass
            
            pred = ensemble.predict(train_data_indexed, steps=min(horizon, len(test_data)))
            actual = test_data.iloc[:len(pred)].values
            if len(pred) == len(actual) and len(pred) > 0:
                test_predictions.append(pred)
                test_actuals.append(actual)
                print(f"✓ Got {len(pred)} predictions from single forecast")
        except Exception as e:
            raise ValueError(
                f"Could not generate any predictions: {e}\n"
                f"Ensemble has {fitted_count} fitted models. "
                f"Please check that models are properly trained and can make predictions."
            )
    
    if not test_predictions:
        raise ValueError("No valid predictions generated for evaluation")
    
    # Flatten predictions and actuals for metrics
    all_preds = np.concatenate(test_predictions)
    all_actuals = np.concatenate(test_actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    mape = mean_absolute_percentage_error(all_actuals, all_preds) * 100
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }
    
    print(f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Get ensemble performance summary
    performance = ensemble.get_model_performance()
    print(f"\n📈 Ensemble Model Performance:")
    for model_name, perf in performance.items():
        weight = perf['weight']
        mae_val = perf['recent_mae']
        if mae_val:
            print(f"   {model_name}: weight={weight:.3f}, recent_MAE={mae_val:.4f}")
        else:
            print(f"   {model_name}: weight={weight:.3f}, recent_MAE=N/A")
    
    # Save ensemble metadata
    settings = get_settings()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_path is None:
        save_path = settings.models_dir / f"{symbol}_ensemble_{timestamp}.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble metadata (not the full ensemble, as it contains references to other models)
    metadata = {
        "symbol": symbol,
        "horizon": horizon,
        "metrics": metrics,
        "model_names": ensemble.model_names,
        "weights": ensemble.weights.tolist(),
        "performance": performance,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "timestamp": timestamp
    }
    
    joblib.dump(metadata, save_path)
    print(f"\n✅ Ensemble metadata saved to {save_path}")
    print(f"   Note: Individual models must be saved separately using their training scripts.")
    
    return {
        "ensemble": ensemble,
        "metrics": metrics,
        "performance": performance,
        "save_path": save_path,
        "train_size": len(train_data),
        "test_size": len(test_data)
    }


if __name__ == "__main__":
    import argparse
    
    # Default symbols (same as train_all_models.py)
    DEFAULT_SYMBOLS = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Google
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        "META",   # Meta
        "NVDA",   # NVIDIA
        "JPM",    # JPMorgan
        "V",      # Visa
        "JNJ",    # Johnson & Johnson
    ]
    
    parser = argparse.ArgumentParser(
        description="Train Ensemble model (combines ARIMA, LSTM, GRU, Transformer, Exponential Smoothing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ensemble for AAPL (will load pre-trained models if available)
  python train_ensemble.py --symbol AAPL

  # Train ensemble for ALL default symbols
  python train_ensemble.py --all-symbols

  # Train ensemble with custom horizon
  python train_ensemble.py --symbol AAPL --horizon 48

  # Train ensemble for multiple symbols
  python train_ensemble.py --symbol AAPL,MSFT,GOOGL

Note: Before training the ensemble, make sure you have trained the individual models:
  python train_arima.py --symbol AAPL
  python train_lstm.py --symbol AAPL --epochs 50
  python train_gru.py --symbol AAPL --epochs 50
  python train_transformer.py --symbol AAPL --epochs 50
  python train_exponential_smoothing.py --symbol AAPL
        """
    )
    
    parser.add_argument("--symbol", type=str, default=None, help="Stock symbol (comma-separated for multiple)")
    parser.add_argument("--all-symbols", action="store_true", help="Train ensemble for all default symbols")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion (default: 0.2)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save ensemble metadata")
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.all_symbols:
        symbols = DEFAULT_SYMBOLS
        print(f"📋 Training ensemble for ALL default symbols: {', '.join(symbols)}")
    elif args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]
    else:
        parser.error("Either --symbol or --all-symbols must be specified")
    
    results = {}
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"Training Ensemble for {symbol}")
        print(f"{'='*70}\n")
        
        try:
            result = train_ensemble(
                symbol=symbol,
                horizon=args.horizon,
                test_size=args.test_size,
                save_path=Path(args.save_path) if args.save_path else None
            )
            results[symbol] = result
            print(f"\n✅ Ensemble training completed for {symbol}")
            print(f"   Metrics: MAE={result['metrics']['mae']:.4f}, RMSE={result['metrics']['rmse']:.4f}, MAPE={result['metrics']['mape']:.2f}%")
        except Exception as e:
            print(f"\n❌ Ensemble training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = None
    
    print(f"\n{'='*70}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*70}")
    for symbol, result in results.items():
        if result:
            print(f"✅ {symbol}: MAE={result['metrics']['mae']:.4f}, RMSE={result['metrics']['rmse']:.4f}")
        else:
            print(f"❌ {symbol}: Failed")

