"""
Inference/Prediction script for Ensemble Model.

This script loads a pre-trained ensemble model and generates forecasts.
The ensemble combines multiple pre-trained models:
- ARIMA
- Exponential Smoothing
- LSTM
- GRU
- Transformer

IMPORTANT: You must train all individual models first before using this script.
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
import warnings
import argparse

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService
from backend.services.model_loader import ModelLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Suppress statsmodels index warnings (they're informational, not errors)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

settings = get_settings()


def predict_ensemble(
    symbol: str,
    horizon: int = 24,
    model_version_id: str = None,
    output_format: str = "console",  # "console", "json", "csv"
    output_path: Path = None,
    include_individual: bool = False,
):
    """
    Generate predictions using a pre-trained ensemble model.
    
    Args:
        symbol: Stock symbol
        horizon: Forecast horizon (number of steps)
        model_version_id: Optional specific model version ID to use
        output_format: Output format ("console", "json", "csv")
        output_path: Path to save output (if not console)
        include_individual: Whether to include individual model predictions
    
    Returns:
        dict with predictions, metrics (if historical data available), and metadata
    """
    print(f"🔮 Generating Ensemble Forecast for {symbol} (horizon={horizon})")
    print(f"{'='*70}\n")
    
    # Load data
    print("📥 Loading recent data...")
    data_service = DataIngestionService()
    df = None
    
    try:
        df = data_service.get_prices(symbol, limit=200)
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
    
    if len(prices) < 60:
        raise ValueError(f"Insufficient data: need at least 60 points, got {len(prices)}")
    
    print(f"✓ Data loaded: {len(prices)} points")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   Last close price: ${prices.iloc[-1]:.2f}\n")
    
    # Load ensemble
    print("📦 Loading ensemble model...")
    model_loader = ModelLoader()
    
    if model_version_id:
        # Load specific ensemble version (if ensembles are stored as versions)
        # For now, we'll create ensemble from individual models
        print(f"   Loading specific model version: {model_version_id}")
        # This would require ensemble to be stored as a model version
        # For now, fall through to loading individual models
        pass
    
    # Load individual models and create ensemble
    try:
        loaded_models = model_loader.load_ensemble_models(symbol, horizon)
        print(f"✓ Loaded {len(loaded_models)} pre-trained models: {', '.join(loaded_models.keys())}")
    except Exception as e:
        raise ValueError(
            f"Could not load ensemble models: {e}\n"
            f"Please ensure individual models are trained first:\n"
            f"  python train_arima.py --symbol {symbol}\n"
            f"  python train_lstm.py --symbol {symbol} --epochs 50\n"
            f"  python train_gru.py --symbol {symbol} --epochs 50\n"
            f"  python train_transformer.py --symbol {symbol} --epochs 50\n"
            f"  python train_exponential_smoothing.py --symbol {symbol}"
        )
    
    # Create ensemble from loaded models
    ensemble = model_loader.create_ensemble_from_loaded_models(
        symbol, horizon, loaded_models
    )
    print(f"✓ Created ensemble from {len(loaded_models)} models\n")
    
    # Get ensemble performance summary
    performance = ensemble.get_model_performance()
    print("📊 Ensemble Model Weights:")
    for model_name, perf in performance.items():
        weight = perf['weight']
        mae_val = perf['recent_mae']
        if mae_val:
            print(f"   {model_name:20s}: weight={weight:.3f}, recent_MAE={mae_val:.4f}")
        else:
            print(f"   {model_name:20s}: weight={weight:.3f}, recent_MAE=N/A")
    print()
    
    # Generate predictions
    print(f"🔮 Generating {horizon}-step forecast...")
    
    # Prepare data for prediction
    # LSTM needs DataFrame, others use Series
    data_for_prediction = prices
    
    # Check if we have DataFrame with all columns for LSTM
    if 'LSTM' in loaded_models:
        # Try to use full DataFrame if available
        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            data_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        else:
            # Create minimal DataFrame from prices
            data_df = pd.DataFrame({
                'Close': prices.values,
                'Open': prices.values,
                'High': prices.values,
                'Low': prices.values,
                'Volume': np.ones(len(prices)) * 1000000
            }, index=prices.index)
    else:
        data_df = None
    
    try:
        # Make ensemble prediction
        if data_df is not None and 'LSTM' in loaded_models:
            predictions = ensemble.predict(data=prices, data_df=data_df.tail(1), steps=horizon)
        else:
            predictions = ensemble.predict(data=prices, steps=horizon)
        
        print(f"✓ Generated {len(predictions)} predictions\n")
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")
    
    # Get individual model predictions if requested
    individual_predictions = {}
    if include_individual:
        print("📈 Generating individual model predictions...")
        for model_type, (model, model_version) in loaded_models.items():
            try:
                if model_type == "LSTM" and data_df is not None:
                    ind_pred = model.predict(data_df.tail(1), steps=horizon)
                    # LSTM predicts Close_diff, convert to absolute
                    last_close = float(prices.iloc[-1])
                    ind_pred_abs = last_close + np.cumsum(ind_pred)
                    individual_predictions[model_type] = ind_pred_abs
                else:
                    ind_pred = model.predict(prices, steps=horizon)
                    # Check if model predicts diff or absolute
                    if model_type in ["ARIMA", "LSTM"]:
                        last_close = float(prices.iloc[-1])
                        ind_pred_abs = last_close + np.cumsum(ind_pred)
                        individual_predictions[model_type] = ind_pred_abs
                    else:
                        individual_predictions[model_type] = ind_pred
                print(f"   ✓ {model_type}")
            except Exception as e:
                print(f"   ⚠️  {model_type} failed: {e}")
        print()
    
    # Create forecast dates
    last_date = prices.index[-1]
    if isinstance(last_date, pd.Timestamp):
        # Generate future dates (assuming daily data)
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
    else:
        # If we can't determine frequency, use simple index
        forecast_dates = range(1, horizon + 1)
    
    # Create results dictionary
    results = {
        "symbol": symbol,
        "horizon": horizon,
        "last_close": float(prices.iloc[-1]),
        "last_date": str(prices.index[-1]),
        "predictions": predictions.tolist(),
        "forecast_dates": [str(d) for d in forecast_dates],
        "model_count": len(loaded_models),
        "model_types": list(loaded_models.keys()),
        "performance": performance,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if include_individual:
        results["individual_predictions"] = {
            k: v.tolist() for k, v in individual_predictions.items()
        }
    
    # Calculate some basic statistics
    results["forecast_stats"] = {
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "first_prediction": float(predictions[0]),
        "last_prediction": float(predictions[-1]),
        "change_from_last": float(predictions[-1] - prices.iloc[-1]),
        "change_pct": float((predictions[-1] - prices.iloc[-1]) / prices.iloc[-1] * 100),
    }
    
    # Display results
    if output_format == "console":
        print("="*70)
        print(f"ENSEMBLE FORECAST RESULTS for {symbol}")
        print("="*70)
        print(f"\nLast Close Price: ${results['last_close']:.2f}")
        print(f"Last Date: {results['last_date']}")
        print(f"\nForecast Statistics:")
        print(f"  Min:  ${results['forecast_stats']['min']:.2f}")
        print(f"  Max:  ${results['forecast_stats']['max']:.2f}")
        print(f"  Mean: ${results['forecast_stats']['mean']:.2f}")
        print(f"  Std:  ${results['forecast_stats']['std']:.2f}")
        print(f"\nForecast Range:")
        print(f"  First:  ${results['forecast_stats']['first_prediction']:.2f}")
        print(f"  Last:   ${results['forecast_stats']['last_prediction']:.2f}")
        print(f"  Change: ${results['forecast_stats']['change_from_last']:.2f} ({results['forecast_stats']['change_pct']:.2f}%)")
        
        print(f"\nForecast Values (first 10):")
        for i in range(min(10, len(predictions))):
            date_str = str(forecast_dates[i]) if i < len(forecast_dates) else f"Step {i+1}"
            print(f"  {date_str}: ${predictions[i]:.2f}")
        
        if len(predictions) > 10:
            print(f"  ... ({len(predictions) - 10} more)")
        
        if include_individual and individual_predictions:
            print(f"\nIndividual Model Predictions (last step):")
            for model_type, preds in individual_predictions.items():
                print(f"  {model_type:20s}: ${preds[-1]:.2f}")
        
        print("\n" + "="*70)
    
    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "json":
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to {output_path}")
        elif output_format == "csv":
            # Create DataFrame with predictions
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'prediction': predictions
            })
            forecast_df.to_csv(output_path, index=False)
            print(f"✓ Predictions saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions using pre-trained Ensemble model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate forecast for AAPL
  python predict_ensemble.py --symbol AAPL

  # Generate forecast with custom horizon
  python predict_ensemble.py --symbol AAPL --horizon 48

  # Save predictions to JSON file
  python predict_ensemble.py --symbol AAPL --output-format json --output-path predictions.json

  # Save predictions to CSV
  python predict_ensemble.py --symbol AAPL --output-format csv --output-path predictions.csv

  # Include individual model predictions
  python predict_ensemble.py --symbol AAPL --include-individual

Note: Before using this script, ensure individual models are trained:
  python train_arima.py --symbol AAPL
  python train_lstm.py --symbol AAPL --epochs 50
  python train_gru.py --symbol AAPL --epochs 50
  python train_transformer.py --symbol AAPL --epochs 50
  python train_exponential_smoothing.py --symbol AAPL
        """
    )
    
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24)")
    parser.add_argument("--model-version-id", type=str, default=None, help="Specific model version ID (optional)")
    parser.add_argument("--output-format", type=str, choices=["console", "json", "csv"], default="console", help="Output format (default: console)")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save output (required for json/csv)")
    parser.add_argument("--include-individual", action="store_true", help="Include individual model predictions")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.output_format in ["json", "csv"] and not args.output_path:
        parser.error(f"--output-path is required when --output-format is {args.output_format}")
    
    try:
        results = predict_ensemble(
            symbol=args.symbol,
            horizon=args.horizon,
            model_version_id=args.model_version_id,
            output_format=args.output_format,
            output_path=Path(args.output_path) if args.output_path else None,
            include_individual=args.include_individual,
        )
        
        print("\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

