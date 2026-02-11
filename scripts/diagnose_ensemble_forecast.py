"""
Diagnostic script to investigate ensemble forecast drop issue.

This script systematically checks:
1. Model configurations and scaling
2. Data validation
3. Scaling consistency
4. Ensemble combination logic
5. Individual model predictions
6. Root cause diagnosis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import DataIngestionService
from backend.services.model_loader import ModelLoader
from backend.models.adaptive_ensemble import AdaptiveEnsemble
from backend.utils.logging import get_logger

logger = get_logger(__name__)

SYMBOL = "AAPL"
HORIZON_STEPS = 1  # For 24h with daily data


def step1_model_inspection(loaded_models):
    """STEP 1: Model Inspection"""
    print("\n" + "="*70)
    print("STEP 1: MODEL INSPECTION")
    print("="*70)
    
    model_configs = {}
    
    for model_type, (model, model_version) in loaded_models.items():
        print(f"\n--- {model_type} Model ---")
        config = {
            "is_fitted": model.is_fitted if hasattr(model, 'is_fitted') else "N/A",
            "horizon": getattr(model, 'horizon', 'N/A'),
            "symbol": getattr(model, 'symbol', 'N/A'),
        }
        
        # Check for scaler
        if hasattr(model, 'scaler'):
            scaler_type = type(model.scaler).__name__
            config["scaler"] = scaler_type
            if hasattr(model.scaler, 'scale_'):
                config["scaler_fitted"] = True
                config["scaler_scale"] = f"shape={model.scaler.scale_.shape if hasattr(model.scaler.scale_, 'shape') else 'N/A'}"
            else:
                config["scaler_fitted"] = False
        else:
            config["scaler"] = "None"
        
        # Check for ARIMA order
        if model_type == "ARIMA":
            config["order"] = getattr(model, 'order', 'N/A')
            config["predicts"] = "Close_diff (differenced)"
        
        # Check for LSTM/GRU/Transformer architecture
        if model_type in ["LSTM", "GRU", "Transformer"]:
            if hasattr(model, 'model') and model.model is not None:
                config["architecture"] = "Neural Network"
                config["predicts"] = "Close_diff (differenced)"
                if hasattr(model, 'units'):
                    config["units"] = model.units
                if hasattr(model, 'lag_window'):
                    config["lag_window"] = model.lag_window
        
        model_configs[model_type] = config
        
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    return model_configs


def step2_data_validation(service):
    """STEP 2: Data Validation"""
    print("\n" + "="*70)
    print("STEP 2: DATA VALIDATION")
    print("="*70)
    
    df = service.get_prices(SYMBOL, limit=200)
    
    if df.empty:
        print("ERROR: No data found!")
        return None
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get last 10 rows
    print("\n--- Last 10 rows of data ---")
    last_10 = df.tail(10)
    print(last_10[['date', 'close'] if 'close' in df.columns else ['date', 'Close']])
    
    # Check for NaNs
    print("\n--- NaN Check ---")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "No NaNs found")
    
    # Check for outliers
    close_col = 'close' if 'close' in df.columns else 'Close'
    if close_col in df.columns:
        last_10_values = df[close_col].tail(10).values
        print(f"\n--- Last 10 Close Prices ---")
        print(f"Values: {last_10_values}")
        print(f"Mean: {np.mean(last_10_values):.2f}")
        print(f"Min: {np.min(last_10_values):.2f}")
        print(f"Max: {np.max(last_10_values):.2f}")
        print(f"Std: {np.std(last_10_values):.2f}")
        
        # Check for discontinuities
        diffs = np.diff(last_10_values)
        print(f"\n--- Price Differences (last 10) ---")
        print(f"Diffs: {diffs}")
        print(f"Max diff: {np.max(np.abs(diffs)):.2f}")
        if np.max(np.abs(diffs)) > np.mean(np.abs(diffs)) * 3:
            print("⚠️  WARNING: Large discontinuity detected!")
    
    # Date alignment
    if 'date' in df.columns:
        last_date = pd.to_datetime(df['date'].iloc[-1])
        print(f"\n--- Date Alignment ---")
        print(f"Last data date: {last_date}")
        print(f"Forecast should start: {last_date + pd.Timedelta(days=1)}")
    
    return df


def step3_scaling_consistency(loaded_models, df):
    """STEP 3: Scaling Consistency"""
    print("\n" + "="*70)
    print("STEP 3: SCALING CONSISTENCY")
    print("="*70)
    
    # Get close prices
    close_col = 'close' if 'close' in df.columns else 'Close'
    prices = df[close_col].tail(100)  # Last 100 for prediction
    
    scaling_info = {}
    
    for model_type, (model, _) in loaded_models.items():
        print(f"\n--- {model_type} Scaling ---")
        
        if hasattr(model, 'scaler') and model.scaler is not None:
            scaler = model.scaler
            scaler_type = type(scaler).__name__
            print(f"Scaler type: {scaler_type}")
            
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                print(f"MinMaxScaler range: [{scaler.data_min_}, {scaler.data_max_}]")
            elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                print(f"StandardScaler mean: {scaler.mean_}, std: {scaler.scale_}")
            
            scaling_info[model_type] = {
                "has_scaler": True,
                "scaler_type": scaler_type,
                "needs_inverse_transform": True
            }
        else:
            print("No scaler found")
            scaling_info[model_type] = {
                "has_scaler": False,
                "needs_inverse_transform": False
            }
        
        # Check what the model predicts
        if model_type == "ARIMA":
            print("Predicts: Close_diff (differenced values)")
            print("⚠️  CRITICAL: Must convert back to absolute prices using cumsum!")
            scaling_info[model_type]["predicts_diff"] = True
        elif model_type == "LSTM":
            print("Predicts: Close_diff (differenced values)")
            print("⚠️  CRITICAL: Must convert back to absolute prices using cumsum!")
            scaling_info[model_type]["predicts_diff"] = True
        elif model_type in ["GRU", "Transformer"]:
            print("Predicts: Absolute prices (after inverse_transform)")
            print("✓ Already in absolute price format - no conversion needed")
            scaling_info[model_type]["predicts_diff"] = False
        else:
            scaling_info[model_type]["predicts_diff"] = False
    
    return scaling_info


def step4_ensemble_logic(ensemble):
    """STEP 4: Ensemble Combination Logic"""
    print("\n" + "="*70)
    print("STEP 4: ENSEMBLE COMBINATION LOGIC")
    print("="*70)
    
    print(f"\nNumber of models: {len(ensemble.forecasters)}")
    print(f"Model names: {ensemble.model_names}")
    print(f"\nWeights: {dict(zip(ensemble.model_names, ensemble.weights))}")
    print(f"Weights sum: {ensemble.weights.sum():.4f}")
    
    if abs(ensemble.weights.sum() - 1.0) > 0.01:
        print("⚠️  WARNING: Weights do not sum to 1!")
    
    # Check if any model dominates
    max_weight_idx = np.argmax(ensemble.weights)
    max_weight = ensemble.weights[max_weight_idx]
    print(f"\nMax weight: {max_weight:.4f} ({ensemble.model_names[max_weight_idx]})")
    
    if max_weight > 0.7:
        print(f"⚠️  WARNING: {ensemble.model_names[max_weight_idx]} dominates with {max_weight:.1%} weight!")
    
    print("\nCombination method: Weighted average")
    print("Formula: ensemble_pred = sum(weight_i * pred_i) for all models")


def step5_individual_predictions(loaded_models, df, prices):
    """STEP 5: Individual Model Predictions"""
    print("\n" + "="*70)
    print("STEP 5: INDIVIDUAL MODEL PREDICTIONS")
    print("="*70)
    
    close_col = 'close' if 'close' in df.columns else 'Close'
    last_5_actual = df[close_col].tail(5).values
    print(f"\n--- Last 5 Actual Values ---")
    print(f"Values: {last_5_actual}")
    print(f"Mean: {np.mean(last_5_actual):.2f}")
    
    individual_predictions = {}
    
    for model_type, (model, _) in loaded_models.items():
        print(f"\n--- {model_type} Prediction ---")
        
        try:
            if model_type == "LSTM":
                # LSTM needs DataFrame
                last_row = df.tail(1)
                pred = model.predict(last_row, steps=HORIZON_STEPS)
            else:
                # Others use Series
                pred = model.predict(prices, steps=HORIZON_STEPS)
            
            print(f"Raw prediction (first 5 steps): {pred[:5] if len(pred) >= 5 else pred}")
            print(f"Shape: {pred.shape}")
            print(f"Mean: {np.mean(pred):.2f}")
            print(f"Min: {np.min(pred):.2f}")
            print(f"Max: {np.max(pred):.2f}")
            
            # Check if this is Close_diff or absolute
            if model_type in ["ARIMA", "LSTM"]:
                print(f"⚠️  This is Close_diff (differenced)!")
                print(f"Last actual close: {prices.iloc[-1]:.2f}")
                
                # Convert to absolute prices
                last_close = prices.iloc[-1]
                if len(pred) > 0:
                    # Convert diff to absolute: pred_abs = last_close + cumsum(pred_diff)
                    pred_absolute = last_close + np.cumsum(pred)
                    print(f"Converted to absolute (first value): {pred_absolute[0]:.2f}")
                    print(f"Converted absolute values: {pred_absolute}")
                    individual_predictions[model_type] = {
                        "raw_diff": pred,
                        "absolute": pred_absolute
                    }
                else:
                    individual_predictions[model_type] = {
                        "raw_diff": pred,
                        "absolute": pred
                    }
            elif model_type in ["GRU", "Transformer"]:
                print(f"✓ This is absolute price (after inverse_transform)")
                print(f"Last actual close: {prices.iloc[-1]:.2f}")
                print(f"Prediction (absolute): {pred}")
                individual_predictions[model_type] = {
                    "raw": pred,
                    "absolute": pred  # Already absolute
                }
            else:
                individual_predictions[model_type] = {
                    "raw": pred,
                    "absolute": pred
                }
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            individual_predictions[model_type] = None
    
    return individual_predictions


def step6_diagnosis_and_fix(individual_predictions, ensemble_forecast, prices):
    """STEP 6: Diagnosis & Fix"""
    print("\n" + "="*70)
    print("STEP 6: DIAGNOSIS & FIX")
    print("="*70)
    
    last_close = prices.iloc[-1]
    print(f"\nLast actual close price: {last_close:.2f}")
    
    if ensemble_forecast is not None:
        print(f"\nEnsemble forecast (first 5): {ensemble_forecast[:5] if len(ensemble_forecast) >= 5 else ensemble_forecast}")
        print(f"Ensemble forecast mean: {np.mean(ensemble_forecast):.2f}")
        print(f"Ensemble forecast min: {np.min(ensemble_forecast):.2f}")
        print(f"Ensemble forecast max: {np.max(ensemble_forecast):.2f}")
    
    # Check if ensemble forecast is in diff or absolute
    print("\n--- DIAGNOSIS ---")
    
    issues_found = []
    
    # Check 1: Are predictions in diff format?
    diff_models = [m for m in individual_predictions.keys() 
                   if individual_predictions[m] and "raw_diff" in individual_predictions[m]]
    
    if diff_models:
        print(f"⚠️  ISSUE 1: Models {diff_models} predict Close_diff (differenced values)")
        print("   These need to be converted to absolute prices before ensembling!")
        print(f"   NOTE: GRU and Transformer predict absolute prices (already converted)")
        issues_found.append("diff_not_converted")
    
    # Check 2: Is ensemble forecast in diff format?
    if ensemble_forecast is not None:
        # If ensemble forecast is much smaller than last_close, it's likely in diff format
        if len(ensemble_forecast) > 0 and abs(ensemble_forecast[0]) < abs(last_close) * 0.1:
            print(f"⚠️  ISSUE 2: Ensemble forecast appears to be in diff format!")
            print(f"   First value: {ensemble_forecast[0]:.2f} vs Last close: {last_close:.2f}")
            issues_found.append("ensemble_in_diff_format")
    
    # Check 3: Negative or unrealistic values
    if ensemble_forecast is not None and len(ensemble_forecast) > 0:
        if np.any(ensemble_forecast < 0):
            print(f"⚠️  ISSUE 3: Negative forecast values found!")
            issues_found.append("negative_forecast")
        if np.any(ensemble_forecast < last_close * 0.5):
            print(f"⚠️  ISSUE 4: Forecast drops drastically below actual!")
            print(f"   Min forecast: {np.min(ensemble_forecast):.2f} vs Last close: {last_close:.2f}")
            issues_found.append("drastic_drop")
    
    # Recommended fix
    print("\n--- RECOMMENDED FIX ---")
    
    if "diff_not_converted" in issues_found or "ensemble_in_diff_format" in issues_found:
        print("""
FIX: Convert Close_diff predictions to absolute prices before ensembling.

For each model that predicts Close_diff:
1. Get the last actual close price: last_close = prices.iloc[-1]
2. Convert diff to absolute: pred_absolute = last_close + np.cumsum(pred_diff)
3. Use pred_absolute for ensemble averaging

Example code:
    last_close = prices.iloc[-1]
    for model_type, (model, _) in loaded_models.items():
        if model_type in ["ARIMA", "LSTM"]:
            # These predict Close_diff - convert to absolute
            pred_diff = model.predict(data, steps=steps)
            pred_absolute = last_close + np.cumsum(pred_diff)
        elif model_type in ["GRU", "Transformer"]:
            # These already predict absolute prices (after inverse_transform)
            pred_absolute = model.predict(data, steps=steps)
        else:
            # Other models (e.g., ExponentialSmoothing) predict absolute
            pred_absolute = model.predict(data, steps=steps)
        # Use pred_absolute for ensemble (all in same scale now)
""")
    
    if "drastic_drop" in issues_found:
        print("""
ADDITIONAL FIX: Apply sanity checks to ensemble forecast.

1. Ensure forecast is in absolute price format (not diff)
2. Clip negative values: forecast = np.maximum(forecast, last_close * 0.8)
3. Ensure forecast starts near last_close: forecast[0] should be close to last_close
""")
    
    return issues_found


def main():
    """Main diagnostic function"""
    print("="*70)
    print("ENSEMBLE FORECAST DIAGNOSTIC TOOL")
    print("="*70)
    print(f"Symbol: {SYMBOL}")
    print(f"Horizon steps: {HORIZON_STEPS}")
    
    try:
        # Load data
        service = DataIngestionService()
        df = step2_data_validation(service)
        if df is None:
            return
        
        # Get prices Series
        close_col = 'close' if 'close' in df.columns else 'Close'
        prices = df[close_col].tail(100)
        
        # Load models
        print("\n" + "="*70)
        print("LOADING MODELS...")
        print("="*70)
        
        loaded_models = ModelLoader.load_ensemble_models(SYMBOL, HORIZON_STEPS)
        print(f"\nLoaded {len(loaded_models)} models: {list(loaded_models.keys())}")
        
        # STEP 1: Model Inspection
        model_configs = step1_model_inspection(loaded_models)
        
        # STEP 3: Scaling Consistency
        scaling_info = step3_scaling_consistency(loaded_models, df)
        
        # Create ensemble
        ensemble = ModelLoader.create_ensemble_from_loaded_models(
            SYMBOL, HORIZON_STEPS, loaded_models
        )
        
        # STEP 4: Ensemble Logic
        step4_ensemble_logic(ensemble)
        
        # STEP 5: Individual Predictions
        individual_predictions = step5_individual_predictions(loaded_models, df, prices)
        
        # Get ensemble forecast
        print("\n--- Ensemble Forecast ---")
        try:
            ensemble_forecast = ensemble.predict(prices, steps=HORIZON_STEPS)
            print(f"Ensemble forecast shape: {ensemble_forecast.shape}")
            print(f"Ensemble forecast: {ensemble_forecast}")
        except Exception as e:
            print(f"❌ Error generating ensemble forecast: {e}")
            import traceback
            traceback.print_exc()
            ensemble_forecast = None
        
        # STEP 6: Diagnosis
        issues = step6_diagnosis_and_fix(individual_predictions, ensemble_forecast, prices)
        
        print("\n" + "="*70)
        print("DIAGNOSTIC COMPLETE")
        print("="*70)
        print(f"Issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

