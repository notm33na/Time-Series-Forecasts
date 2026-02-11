"""
Fix for ensemble forecast drop issue.

The problem: ARIMA, LSTM, GRU, and Transformer predict Close_diff (differenced values),
but the ensemble averages them without converting to absolute prices first.

This script shows the fix and provides corrected code.
"""

import numpy as np
import pandas as pd


def convert_diff_to_absolute(pred_diff: np.ndarray, last_close: float) -> np.ndarray:
    """
    Convert differenced predictions to absolute prices.
    
    Args:
        pred_diff: Array of Close_diff predictions
        last_close: Last actual close price
    
    Returns:
        Array of absolute price predictions
    """
    if len(pred_diff) == 0:
        return pred_diff
    
    # Convert diff to absolute: pred_abs[i] = last_close + sum(pred_diff[0:i+1])
    pred_absolute = last_close + np.cumsum(pred_diff)
    return pred_absolute


def fixed_ensemble_predict(ensemble, data: pd.Series, steps: int, last_close: float) -> np.ndarray:
    """
    Fixed version of ensemble.predict() that converts Close_diff to absolute prices.
    
    This is the corrected version that should be used in AdaptiveEnsemble.predict()
    """
    if not ensemble.forecasters:
        raise ValueError("No forecasters in ensemble")
    
    predictions = []
    valid_weights = []
    valid_names = []
    
    # Models that predict Close_diff (differenced values)
    diff_models = ["ARIMA", "LSTM", "GRU", "Transformer"]
    
    for i, forecaster in enumerate(ensemble.forecasters):
        if not forecaster.is_fitted:
            continue
        
        model_name = ensemble.model_names[i]
        
        try:
            # Get raw prediction
            if model_name == "LSTM":
                # LSTM needs DataFrame - convert Series to DataFrame
                if isinstance(data, pd.Series):
                    # Create DataFrame from Series (assuming it's Close prices)
                    data_df = pd.DataFrame({
                        'Close': data.values,
                        'Open': data.values,  # Approximate
                        'High': data.values,
                        'Low': data.values,
                        'Volume': np.ones(len(data)) * 1000000  # Dummy volume
                    })
                    pred = forecaster.predict(data_df.tail(1), steps=steps)
                else:
                    pred = forecaster.predict(data.tail(1), steps=steps)
            else:
                pred = forecaster.predict(data, steps=steps)
            
            if len(pred) == steps:
                # Convert Close_diff to absolute prices if needed
                if model_name in diff_models:
                    pred_absolute = convert_diff_to_absolute(pred, last_close)
                    print(f"  {model_name}: diff={pred[:3]}, absolute={pred_absolute[:3]}")
                else:
                    pred_absolute = pred
                    print(f"  {model_name}: absolute={pred_absolute[:3]}")
                
                predictions.append(pred_absolute)
                valid_weights.append(ensemble.weights[i])
                valid_names.append(model_name)
        except Exception as e:
            print(f"  Prediction failed for {model_name}: {e}")
    
    if not predictions:
        raise ValueError("No valid predictions from ensemble")
    
    # Normalize weights for valid models
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    # Weighted average of absolute prices
    predictions = np.array(predictions)
    ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
    
    print(f"\nEnsemble prediction (first 3): {ensemble_pred[:3]}")
    print(f"Last close: {last_close:.2f}")
    print(f"First forecast: {ensemble_pred[0]:.2f}")
    print(f"Difference: {ensemble_pred[0] - last_close:.2f}")
    
    return ensemble_pred


# Code to add to AdaptiveEnsemble.predict() method:
FIXED_PREDICT_CODE = '''
def predict(self, data: pd.Series = None, steps: int = None) -> np.ndarray:
    """Generate weighted ensemble prediction with proper Close_diff conversion."""
    if not self.forecasters:
        raise ValueError("No forecasters in ensemble")
    
    steps = steps or self.horizon
    
    # Get last close price for diff-to-absolute conversion
    if data is not None and len(data) > 0:
        last_close = float(data.iloc[-1])
    else:
        raise ValueError("Data required for prediction")
    
    predictions = []
    valid_weights = []
    valid_names = []
    
    # Models that predict Close_diff (differenced values)
    diff_models = ["ARIMA", "LSTM", "GRU", "Transformer"]
    
    for i, forecaster in enumerate(self.forecasters):
        if not forecaster.is_fitted:
            continue
        
        model_name = self.model_names[i]
        
        try:
            # Get raw prediction
            if model_name == "LSTM":
                # LSTM needs DataFrame
                if isinstance(data, pd.Series):
                    # Create minimal DataFrame for LSTM
                    data_df = pd.DataFrame({
                        'Close': data.values,
                        'Open': data.values,
                        'High': data.values,
                        'Low': data.values,
                        'Volume': np.ones(len(data)) * 1000000
                    })
                    pred = forecaster.predict(data_df.tail(1), steps=steps)
                else:
                    pred = forecaster.predict(data.tail(1), steps=steps)
            else:
                pred = forecaster.predict(data, steps=steps)
            
            if len(pred) == steps:
                # Convert Close_diff to absolute prices if needed
                if model_name in diff_models:
                    # Convert: pred_abs = last_close + cumsum(pred_diff)
                    pred_absolute = last_close + np.cumsum(pred)
                else:
                    pred_absolute = pred
                
                predictions.append(pred_absolute)
                valid_weights.append(self.weights[i])
                valid_names.append(model_name)
        except Exception as e:
            logger.warning(f"Prediction failed for {model_name}: {e}")
    
    if not predictions:
        raise ValueError("No valid predictions from ensemble")
    
    # Normalize weights for valid models
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    # Weighted average of absolute prices
    predictions = np.array(predictions)
    ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
    
    logger.debug(f"Ensemble prediction using {len(valid_names)} models: {valid_names}")
    
    return ensemble_pred
'''

if __name__ == "__main__":
    print("="*70)
    print("ENSEMBLE FORECAST FIX")
    print("="*70)
    print("\nROOT CAUSE:")
    print("ARIMA, LSTM, GRU, and Transformer predict Close_diff (differenced values),")
    print("but the ensemble averages them without converting to absolute prices.")
    print("\nSOLUTION:")
    print("Convert Close_diff predictions to absolute prices before averaging:")
    print("  pred_absolute = last_close + np.cumsum(pred_diff)")
    print("\nFIXED CODE:")
    print(FIXED_PREDICT_CODE)

