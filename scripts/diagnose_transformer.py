"""
Diagnostic script to check Transformer model scaler range vs current prices.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.services.model_loader import ModelLoader
from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def diagnose_transformer(symbol: str = "GOOGL"):
    """Diagnose Transformer model scaler range issues."""
    print(f"\n{'='*70}")
    print(f"Transformer Model Diagnosis for {symbol}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Load Transformer model
        print("1. Loading Transformer model...")
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, "Transformer")
            print(f"   ✓ Transformer model loaded")
            print(f"   Model version: {model_version.get('id', 'N/A')}")
        except ValueError as e:
            print(f"   ❌ Failed to load Transformer model: {e}")
            return False
        
        # 2. Check scaler range
        print("\n2. Checking scaler range...")
        if not hasattr(model, 'scaler'):
            print("   ❌ Model missing scaler!")
            return False
        
        scaler = model.scaler
        is_minmax = hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_')
        is_standard = hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')
        
        if is_minmax:
            scaler_min = float(scaler.data_min_[0])
            scaler_max = float(scaler.data_max_[0])
            scaler_range = scaler_max - scaler_min
            print(f"   Scaler type: MinMaxScaler")
            print(f"   Scaler min: ${scaler_min:.2f}")
            print(f"   Scaler max: ${scaler_max:.2f}")
            print(f"   Scaler range: ${scaler_range:.2f}")
        elif is_standard:
            scaler_mean = float(scaler.mean_[0])
            scaler_std = float(scaler.scale_[0])
            print(f"   Scaler type: StandardScaler")
            print(f"   Mean: ${scaler_mean:.2f}")
            print(f"   Std: ${scaler_std:.2f}")
            print(f"   Range (±3 std): [${scaler_mean - 3*scaler_std:.2f}, ${scaler_mean + 3*scaler_std:.2f}]")
            # For comparison, calculate approximate min/max from training data
            scaler_min = scaler_mean - 3 * scaler_std
            scaler_max = scaler_mean + 3 * scaler_std
        else:
            print("   ⚠️  Unknown scaler type (neither MinMax nor Standard)")
            return False
        
        # 3. Get current prices
        print("\n3. Checking current prices...")
        data_service = DataIngestionService()
        df = data_service.get_prices(symbol, limit=100)
        if df.empty:
            print(f"   ❌ No data found for {symbol}")
            return False
        
        prices = df['close']
        current_price = float(prices.iloc[-1])
        price_min = float(prices.min())
        price_max = float(prices.max())
        price_range = price_max - price_min
        
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Recent price range: [${price_min:.2f}, ${price_max:.2f}]")
        print(f"   Recent price range span: ${price_range:.2f}")
        
        # 4. Compare scaler range with current prices
        print("\n4. Comparing scaler range with current prices...")
        if is_minmax:
            if current_price < scaler_min:
                print(f"   ⚠️  WARNING: Current price ${current_price:.2f} is BELOW scaler min ${scaler_min:.2f}")
                print(f"      Difference: ${scaler_min - current_price:.2f} ({((scaler_min - current_price) / current_price * 100):.2f}%)")
                print(f"      This causes EXTRAPOLATION - model is predicting outside training range!")
            elif current_price > scaler_max:
                print(f"   ⚠️  WARNING: Current price ${current_price:.2f} is ABOVE scaler max ${scaler_max:.2f}")
                print(f"      Difference: ${current_price - scaler_max:.2f} ({((current_price - scaler_max) / scaler_max * 100):.2f}%)")
                print(f"      This causes EXTRAPOLATION - model is predicting outside training range!")
            else:
                print(f"   ✓ Current price is within scaler range")
                scaler_range = scaler_max - scaler_min
                overlap_pct = ((min(price_max, scaler_max) - max(price_min, scaler_min)) / scaler_range * 100)
                print(f"   Price range overlap: {overlap_pct:.1f}%")
        elif is_standard:
            z_score = (current_price - scaler_mean) / scaler_std
            print(f"   Current price z-score: {z_score:.2f} (mean=${scaler_mean:.2f}, std=${scaler_std:.2f})")
            if abs(z_score) > 3:
                print(f"   ⚠️  WARNING: Current price is {abs(z_score):.2f} std from mean")
                print(f"      This may indicate extrapolation - model is predicting outside typical range!")
            else:
                print(f"   ✓ Current price is within ±3 std of mean (typical range)")
        
        # 5. Test prediction
        print("\n5. Testing prediction...")
        change_pct = 0.0
        try:
            # Get normalized prediction to diagnose saturation
            scaled_input = model.preprocess_input(prices)
            if hasattr(model, 'lag_window') and len(scaled_input) >= model.lag_window:
                import tensorflow as tf
                window = scaled_input[-model.lag_window:]
                X = window.reshape((1, model.lag_window, 1))
                pred_normalized = model.model.predict(X, verbose=0)[0, 0]
                
                if is_minmax:
                    # Calculate what price this normalized value represents
                    pred_price_from_norm = scaler_min + pred_normalized * (scaler_max - scaler_min)
                    
                    print(f"   Normalized prediction: {pred_normalized:.6f} (range [0, 1])")
                    print(f"   This normalized value maps to: ${pred_price_from_norm:.2f}")
                    if pred_normalized < 0.95:
                        print(f"   ⚠️  Model output is below 0.95 - possible saturation near upper bound")
                        print(f"      (0.90 would map to ${scaler_min + 0.90 * (scaler_max - scaler_min):.2f})")
                elif is_standard:
                    # For StandardScaler, inverse transform the normalized value
                    pred_price_from_norm = scaler_mean + pred_normalized * scaler_std
                    
                    print(f"   Normalized prediction: {pred_normalized:.6f} (standardized, mean=0, std=1)")
                    print(f"   This normalized value maps to: ${pred_price_from_norm:.2f}")
                    if abs(pred_normalized) > 2:
                        print(f"   ⚠️  Model output is >2 std from mean - may indicate extrapolation")
            
            pred = model.predict(prices, steps=1)
            first_pred = float(pred[0])
            change_pct = ((first_pred - current_price) / current_price * 100) if current_price > 0 else 0
            
            print(f"   Final prediction: ${first_pred:.2f}")
            print(f"   Change: {change_pct:+.2f}%")
            
            if abs(change_pct) > 20:
                print(f"   ⚠️  EXTREME PREDICTION (>20% change)")
                if current_price < scaler_min or current_price > scaler_max:
                    pct_outside = ((current_price - scaler_max) / scaler_max * 100) if current_price > scaler_max else ((scaler_min - current_price) / scaler_min * 100)
                    if pct_outside <= 1.0:
                        print(f"   ℹ️  Price is only {pct_outside:.2f}% outside range - clamp should NOT trigger")
                    else:
                        print(f"   🔴 ROOT CAUSE: Price {pct_outside:.2f}% outside scaler range causes extrapolation!")
                        print(f"   💡 SOLUTION: Retrain Transformer model on current data range")
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. Recommendations
        print("\n6. Recommendations:")
        if is_minmax:
            if current_price < scaler_min or current_price > scaler_max:
                pct_outside = ((current_price - scaler_max) / scaler_max * 100) if current_price > scaler_max else ((scaler_min - current_price) / scaler_min * 100)
                if pct_outside <= 1.0:
                    print("   ℹ️  Price is slightly outside scaler range (<1%) - clamp disabled")
                    print("   - Model should work normally, but monitor for saturation")
                else:
                    print("   🔴 CRITICAL: Model needs retraining!")
                    print("   - Current price is significantly outside scaler training range")
                    print("   - Model is extrapolating, causing unreliable predictions")
                    print("   - Run: python notebooks/train_transformer.py --symbol GOOGL")
            elif abs(change_pct) > 20:
                print("   ⚠️  Model producing extreme predictions")
                print("   - Consider retraining with more recent data")
                print("   - Check if model is overfitting or saturating")
            else:
                print("   ✓ Model appears to be working correctly")
        elif is_standard:
            if abs(z_score) > 3.5:
                print("   🔴 CRITICAL: Model needs retraining!")
                print(f"   - Current price is {z_score:.2f} std from mean (very extreme)")
                print("   - Model is extrapolating, causing unreliable predictions")
                print("   - Run: python notebooks/train_transformer.py --symbol GOOGL")
            elif abs(z_score) > 3:
                print("   ⚠️  Price is >3 std from mean - consider retraining")
                print("   - Model may be extrapolating")
                print("   - Monitor predictions closely")
            elif abs(change_pct) > 20:
                print("   ⚠️  Model producing extreme predictions")
                print("   - Consider retraining with more recent data")
                print("   - Check if model is overfitting")
            else:
                print("   ✓ Model appears to be working correctly")
        
        print(f"\n{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose Transformer model")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Symbol to diagnose")
    
    args = parser.parse_args()
    
    success = diagnose_transformer(args.symbol)
    sys.exit(0 if success else 1)

