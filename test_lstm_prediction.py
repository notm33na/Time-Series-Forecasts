"""
Test script to verify LSTM is producing predictions.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.services.model_loader import ModelLoader
from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def test_lstm_prediction(symbol: str = "GOOGL", horizon_steps: int = 24):
    """Test if LSTM can generate predictions."""
    print(f"\n{'='*70}")
    print(f"Testing LSTM Prediction for {symbol}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Load LSTM model
        print("1. Loading LSTM model...")
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, "LSTM", horizon=horizon_steps)
            print(f"   ✓ LSTM model loaded successfully")
            print(f"   Model version: {model_version.get('id', 'N/A')}")
            print(f"   Model horizon: {model_version.get('horizon', 'N/A')}")
        except ValueError as e:
            print(f"   ❌ Failed to load LSTM model: {e}")
            print(f"   Try training LSTM first: python scripts/train_lstm.py --symbol {symbol}")
            return False
        
        # 2. Check model attributes
        print("\n2. Checking model attributes...")
        has_feature_cols = hasattr(model, 'feature_cols')
        has_scaler = hasattr(model, 'scaler')
        is_fitted = getattr(model, 'is_fitted', False)
        has_close_range = hasattr(model, 'close_range')
        
        print(f"   feature_cols: {'✓' if has_feature_cols else '❌'} {getattr(model, 'feature_cols', None)}")
        print(f"   scaler: {'✓' if has_scaler else '❌'}")
        print(f"   is_fitted: {'✓' if is_fitted else '❌'}")
        print(f"   close_range: {'✓' if has_close_range else '❌'} {getattr(model, 'close_range', None)}")
        
        if not is_fitted:
            print("   ❌ Model is not fitted!")
            return False
        
        # 3. Get data
        print("\n3. Fetching data...")
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol, limit=200)
        if df.empty:
            print(f"   ❌ No data found for {symbol}")
            return False
        
        print(f"   ✓ Fetched {len(df)} data points")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Last close: ${df['close'].iloc[-1]:.2f}")
        
        # 4. Prepare input DataFrame
        print("\n4. Preparing input DataFrame...")
        last_row = df.iloc[[-1]].copy()
        
        # Get expected features
        expected_features = getattr(model, 'feature_cols', None)
        if expected_features is None:
            print("   ⚠️  Model missing feature_cols, will infer from data")
            expected_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"   Expected features: {expected_features}")
        print(f"   Available columns: {list(last_row.columns)}")
        
        # Create input DataFrame with expected features + Close (scaler needs Close even if not in feature_cols)
        input_df = pd.DataFrame()
        all_needed_cols = list(expected_features)
        # Add Close if not already in expected_features (scaler was fit on all columns including Close)
        if 'Close' not in all_needed_cols and 'close' not in [c.lower() for c in all_needed_cols]:
            all_needed_cols.append('Close')
        
        for feat_col in all_needed_cols:
            if feat_col in last_row.columns:
                input_df[feat_col] = last_row[feat_col].values
            else:
                col_lower = feat_col.lower()
                matching = [c for c in last_row.columns if c.lower() == col_lower]
                if matching:
                    input_df[feat_col] = last_row[matching[0]].values
                else:
                    # Use close as fallback
                    close_val = last_row.get('close', df['close'].iloc[-1])
                    if isinstance(close_val, pd.Series):
                        close_val = close_val.iloc[0] if len(close_val) > 0 else df['close'].iloc[-1]
                    input_df[feat_col] = close_val
                    print(f"   ⚠️  Created missing feature '{feat_col}' using Close: {close_val}")
        
        print(f"   ✓ Input DataFrame shape: {input_df.shape}")
        print(f"   Input DataFrame columns: {list(input_df.columns)}")
        print(f"   Input values: {input_df.iloc[0].to_dict()}")
        
        # 5. Generate prediction
        print(f"\n5. Generating {horizon_steps}-step prediction...")
        try:
            pred = model.predict(input_df, steps=horizon_steps)
            print(f"   ✓ Prediction generated successfully!")
            print(f"   Prediction shape: {pred.shape}")
            print(f"   Prediction type: {type(pred)}")
            print(f"   First 5 values: {pred[:5] if len(pred) >= 5 else pred}")
            print(f"   Min: {np.min(pred):.4f}, Max: {np.max(pred):.4f}, Mean: {np.mean(pred):.4f}")
            
            # Check if prediction is valid
            if len(pred) == 0:
                print("   ❌ Prediction is empty!")
                return False
            
            if not all(np.isfinite(pred)):
                print("   ❌ Prediction contains NaN or Inf values!")
                return False
            
            # 6. Convert to absolute prices
            print("\n6. Converting Close_diff to absolute prices...")
            last_close = float(df['close'].iloc[-1])
            pred_absolute = last_close + np.cumsum(pred)
            
            print(f"   Last close: ${last_close:.2f}")
            print(f"   First prediction: ${pred_absolute[0]:.2f}")
            print(f"   Change: ${pred_absolute[0] - last_close:.2f} ({((pred_absolute[0] - last_close) / last_close * 100):.2f}%)")
            print(f"   First 5 absolute prices: {pred_absolute[:5]}")
            
            if pred_absolute[0] < 0:
                print("   ❌ First prediction is negative!")
                return False
            
            print(f"\n{'='*70}")
            print(f"✅ LSTM PREDICTION TEST PASSED")
            print(f"{'='*70}\n")
            return True
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LSTM prediction")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Symbol to test")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon steps")
    
    args = parser.parse_args()
    
    success = test_lstm_prediction(args.symbol, args.horizon)
    sys.exit(0 if success else 1)

