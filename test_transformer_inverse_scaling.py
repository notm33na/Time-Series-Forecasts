"""
Test script to verify Transformer inverse scaling is working correctly.
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

def test_transformer_inverse_scaling(symbol: str = "GOOGL", horizon_steps: int = 1):
    """Test if Transformer inverse scaling is working correctly."""
    print(f"\n{'='*70}")
    print(f"Testing Transformer Inverse Scaling for {symbol}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Load Transformer model
        print("1. Loading Transformer model...")
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, "Transformer", horizon=horizon_steps)
            print(f"   ✓ Transformer model loaded")
        except ValueError as e:
            print(f"   ❌ Failed to load Transformer model: {e}")
            return False
        
        # 2. Check scaler
        print("\n2. Checking scaler...")
        if not hasattr(model, 'scaler'):
            print("   ❌ Model missing scaler!")
            return False
        
        scaler = model.scaler
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            scaler_min = float(scaler.data_min_[0])
            scaler_max = float(scaler.data_max_[0])
            print(f"   Scaler min: ${scaler_min:.2f}")
            print(f"   Scaler max: ${scaler_max:.2f}")
            print(f"   Scaler range: ${scaler_max - scaler_min:.2f}")
        else:
            print("   ❌ Scaler missing data_min_/data_max_ attributes")
            return False
        
        # 3. Get data
        print("\n3. Fetching data...")
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol, limit=200)
        if df.empty:
            print(f"   ❌ No data found for {symbol}")
            return False
        
        prices = df['close']
        last_close = float(prices.iloc[-1])
        print(f"   ✓ Fetched {len(prices)} data points")
        print(f"   Last close: ${last_close:.2f}")
        
        # 4. Test preprocessing
        print("\n4. Testing preprocessing (scaling)...")
        try:
            scaled = model.preprocess_input(prices)
            print(f"   ✓ Preprocessing successful")
            print(f"   Scaled input shape: {scaled.shape}")
            print(f"   Scaled input range: [{np.min(scaled):.6f}, {np.max(scaled):.6f}]")
            print(f"   Scaled input mean: {np.mean(scaled):.6f}")
            
            # Verify scaling: last_close should map to a value between 0 and 1
            last_scaled = scaled[-1]
            print(f"   Last close ${last_close:.2f} -> scaled: {last_scaled:.6f}")
            
            # Check if scaled value is reasonable (should be ~0-1 for MinMaxScaler)
            if last_scaled < 0 or last_scaled > 1.5:  # Allow some margin for extrapolation
                print(f"   ⚠️  WARNING: Scaled value {last_scaled:.6f} is outside [0, 1] range!")
                print(f"      This indicates extrapolation (price outside training range)")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Test model prediction (normalized)
        print("\n5. Testing model prediction (normalized output)...")
        try:
            # Get the first prediction in normalized space
            current_window = scaled.copy()
            X = current_window.reshape((1, model.lag_window, 1))
            pred_normalized = model.model.predict(X, verbose=0)[0, 0]
            
            print(f"   ✓ Model prediction (normalized): {pred_normalized:.6f}")
            print(f"   Normalized prediction range: [0, 1] expected for MinMaxScaler")
            
            if pred_normalized < 0 or pred_normalized > 1.5:
                print(f"   ⚠️  WARNING: Normalized prediction {pred_normalized:.6f} is outside [0, 1] range!")
        except Exception as e:
            print(f"   ❌ Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. Test inverse transform manually
        print("\n6. Testing inverse transform manually...")
        try:
            pred_normalized_array = np.array([[pred_normalized]])
            pred_denormalized_manual = scaler.inverse_transform(pred_normalized_array)[0, 0]
            
            print(f"   Normalized value: {pred_normalized:.6f}")
            print(f"   Inverse transform formula: min + (max - min) * normalized")
            print(f"   Manual calculation: ${scaler_min:.2f} + (${scaler_max - scaler_min:.2f} * {pred_normalized:.6f})")
            print(f"   Manual result: ${pred_denormalized_manual:.2f}")
        except Exception as e:
            print(f"   ❌ Manual inverse transform failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 7. Test full predict() method
        print("\n7. Testing full predict() method...")
        try:
            pred_full = model.predict(prices, steps=horizon_steps)
            first_pred = float(pred_full[0])
            
            print(f"   ✓ Full predict() successful")
            print(f"   First prediction: ${first_pred:.2f}")
            print(f"   Change from close: {((first_pred - last_close) / last_close * 100):+.2f}%")
            
            # Compare with manual calculation
            print(f"\n   Comparison:")
            print(f"   Manual inverse: ${pred_denormalized_manual:.2f}")
            print(f"   Full predict(): ${first_pred:.2f}")
            diff = abs(first_pred - pred_denormalized_manual)
            if diff < 0.01:
                print(f"   ✓ Results match (difference: ${diff:.4f})")
            else:
                print(f"   ⚠️  Results differ by ${diff:.4f}")
                print(f"      This may be due to iterative prediction vs single step")
        except Exception as e:
            print(f"   ❌ Full predict() failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 8. Verify inverse transform is being called
        print("\n8. Verifying inverse transform is called...")
        # Check the predict method code
        import inspect
        source = inspect.getsource(model.predict)
        if 'inverse_transform' in source:
            print(f"   ✓ predict() method contains 'inverse_transform'")
        else:
            print(f"   ❌ predict() method does NOT contain 'inverse_transform'!")
            print(f"   This means inverse scaling is NOT happening!")
            return False
        
        # 9. Check if prediction is in normalized or absolute space
        print("\n9. Checking prediction space...")
        if first_pred < 0 or first_pred > 1.5:
            if first_pred < 10:
                print(f"   ⚠️  Prediction ${first_pred:.2f} looks like it might be in normalized space!")
                print(f"      (Should be ~${last_close:.2f} if correctly inverse-transformed)")
            else:
                print(f"   ✓ Prediction ${first_pred:.2f} appears to be in absolute price space")
        else:
            print(f"   ❌ CRITICAL: Prediction ${first_pred:.2f} is in normalized space [0, 1]!")
            print(f"      Inverse transform is NOT working!")
            return False
        
        # 10. Final validation
        print("\n10. Final validation...")
        if abs(first_pred - last_close) > last_close * 0.5:  # More than 50% change
            print(f"   ⚠️  Prediction is very different from current price")
            print(f"      This could be due to:")
            print(f"      - Extrapolation (price outside scaler range)")
            print(f"      - Model predicting significant movement")
            print(f"      - Model needs retraining")
        else:
            print(f"   ✓ Prediction is within reasonable range")
        
        print(f"\n{'='*70}")
        print(f"✅ TRANSFORMER INVERSE SCALING TEST COMPLETE")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Transformer inverse scaling")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Symbol to test")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon steps")
    
    args = parser.parse_args()
    
    success = test_transformer_inverse_scaling(args.symbol, args.horizon)
    sys.exit(0 if success else 1)

