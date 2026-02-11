"""
Adaptive ensemble with dynamic reweighting based on recent error trends.
Uses exponential decay weighting for model selection.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .forecasting_models import (
    BaseForecaster,
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
)
from ..utils.logging import get_logger

try:
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class AdaptiveEnsemble:
    """
    Adaptive ensemble that dynamically reweights models based on recent performance.
    Uses exponential decay to emphasize recent errors.
    """
    
    def __init__(self, symbol: str, horizon: int = 1, decay_factor: float = 0.9):
        self.symbol = symbol
        self.horizon = horizon
        self.decay_factor = decay_factor  # Higher = more weight on recent errors
        self.forecasters: list[BaseForecaster] = []
        self.recent_errors: list[list[float]] = []  # List of error lists per model
        self.weights: np.ndarray = np.array([])
        self.model_names: list[str] = []
    
    def add_forecaster(self, forecaster: BaseForecaster, name: str) -> AdaptiveEnsemble:
        """Add a forecaster to the ensemble."""
        self.forecasters.append(forecaster)
        self.recent_errors.append([])
        self.model_names.append(name)
        self._update_weights()
        logger.info(f"Added {name} to ensemble for {self.symbol}")
        return self
    
    def _update_weights(self) -> None:
        """Update ensemble weights based on recent errors with exponential decay."""
        if not self.forecasters:
            return
        
        if not any(self.recent_errors) or not any(any(errs) for errs in self.recent_errors):
            # Equal weights if no error history
            self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
            return
        
        # Calculate weighted errors (exponential decay on recent errors)
        weighted_errors = []
        for error_list in self.recent_errors:
            if not error_list:
                weighted_errors.append(1.0)  # Default high error if no history
                continue
            
            # Apply exponential decay: more recent errors have higher weight
            weights = np.array([self.decay_factor ** (len(error_list) - 1 - i) 
                              for i in range(len(error_list))])
            weighted_error = np.average(error_list, weights=weights)
            weighted_errors.append(weighted_error)
        
        # Convert errors to weights (inverse relationship)
        weighted_errors = np.array(weighted_errors)
        # Add small epsilon to avoid division by zero
        inv_errors = 1.0 / (weighted_errors + 1e-6)
        
        # Normalize to sum to 1
        self.weights = inv_errors / inv_errors.sum()
        
        logger.debug(f"Ensemble weights updated: {dict(zip(self.model_names, self.weights))}")
    
    def fit(self, data: pd.Series) -> AdaptiveEnsemble:
        """Fit all forecasters in the ensemble."""
        logger.info(f"Fitting ensemble with {len(self.forecasters)} models for {self.symbol}")
        
        for i, forecaster in enumerate(self.forecasters):
            try:
                forecaster.fit(data)
                logger.info(f"✓ {self.model_names[i]} fitted successfully")
            except Exception as e:
                logger.warning(f"✗ {self.model_names[i]} failed to fit: {e}")
        
        return self
    
    def predict(self, data: pd.Series = None, data_df: pd.DataFrame = None, steps: int = None) -> np.ndarray:
        """
        Generate weighted ensemble prediction.
        
        CRITICAL: Converts Close_diff predictions to absolute prices before ensemble averaging.
        
        Process:
        1. ARIMA and LSTM predict Close_diff (differenced values)
        2. Convert Close_diff to absolute prices: pred_price = last_close + cumsum(predicted_close_diff)
        3. GRU and Transformer already predict absolute prices (no conversion needed)
        4. Ensemble all predictions in price-space using weighted average
        
        Args:
            data: pd.Series of close prices (for most models)
            data_df: pd.DataFrame with OHLCV columns (for LSTM, optional)
            steps: Number of forecast steps
        """
        if not self.forecasters:
            raise ValueError("No forecasters in ensemble")
        
        steps = steps or self.horizon
        
        # Get last close price for diff-to-absolute conversion
        if data is not None and len(data) > 0:
            last_close = float(data.iloc[-1])
        elif data_df is not None and len(data_df) > 0:
            # Extract close from DataFrame
            close_col = 'Close' if 'Close' in data_df.columns else 'close'
            if close_col in data_df.columns:
                last_close = float(data_df[close_col].iloc[-1])
            else:
                raise ValueError("DataFrame must contain 'Close' or 'close' column")
        else:
            raise ValueError("Data (Series or DataFrame) required for prediction")
        
        predictions = []
        valid_weights = []
        valid_names = []
        
        # Models that predict Close_diff (differenced values) - need conversion to absolute prices
        # ARIMA and LSTM predict Close_diff, which must be converted to absolute prices before ensemble
        # Conversion formula: pred_price = last_close + cumsum(predicted_close_diff)
        # NOTE: GRU and Transformer predict absolute prices directly (after inverse_transform),
        #       so they should NOT be converted. Only ARIMA and LSTM predict Close_diff.
        diff_models = ["ARIMA", "LSTM"]
        
        for i, forecaster in enumerate(self.forecasters):
            if not forecaster.is_fitted:
                continue
            
            model_name = self.model_names[i]
            
            try:
                # Get raw prediction
                # LSTM requires DataFrame, others use Series
                if model_name == "LSTM":
                    # Use provided DataFrame or create from Series
                    if data_df is not None:
                        # Ensure DataFrame has all required columns for LSTM
                        # LSTM expects: Open, High, Low, Close, Volume (or Adj Close)
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols = []
                        
                        # Check what columns we have
                        available_cols = list(data_df.columns)
                        close_col = None
                        for col in ['Close', 'close']:
                            if col in available_cols:
                                close_col = col
                                break
                        
                        if close_col:
                            # Create missing columns from Close if needed
                            for req_col in required_cols:
                                if req_col not in available_cols and req_col.lower() not in [c.lower() for c in available_cols]:
                                    if req_col == 'Volume':
                                        data_df[req_col] = np.ones(len(data_df)) * 1000000  # Dummy volume
                                    else:
                                        # Approximate OHLC from Close
                                        data_df[req_col] = data_df[close_col].values
                        
                        pred = forecaster.predict(data_df.tail(1), steps=steps)
                    elif isinstance(data, pd.Series):
                        # Create minimal DataFrame from Series for LSTM
                        # LSTM needs: Open, High, Low, Close, Volume
                        data_df_minimal = pd.DataFrame({
                            'Close': data.values,
                            'Open': data.values,  # Approximate from Close
                            'High': data.values,
                            'Low': data.values,
                            'Volume': np.ones(len(data)) * 1000000  # Dummy volume
                        })
                        pred = forecaster.predict(data_df_minimal.tail(1), steps=steps)
                    else:
                        raise ValueError("LSTM requires DataFrame or Series")
                else:
                    # Other models use Series
                    if data is None:
                        raise ValueError(f"{model_name} requires Series data")
                    pred = forecaster.predict(data, steps=steps)
                
                if len(pred) == steps:
                    # Convert Close_diff to absolute prices if needed
                    if model_name in diff_models:
                        # ARIMA and LSTM predict Close_diff (differenced values)
                        # Convert to absolute prices: pred_price = last_close + cumsum(predicted_close_diff)
                        # For multi-step: pred_absolute[i] = last_close + sum(pred_diff[0:i+1])
                        # CRITICAL: Ensure pred is a numpy array for cumsum to work correctly
                        if not isinstance(pred, np.ndarray):
                            pred = np.array(pred)
                        
                        # Convert Close_diff to absolute prices
                        pred_absolute = last_close + np.cumsum(pred)
                        
                        # Validation: Check if conversion produced reasonable values
                        if len(pred_absolute) > 0:
                            first_pred = pred_absolute[0]
                            if first_pred < 0 or abs(first_pred - last_close) > last_close * 2:
                                logger.error(
                                    f"⚠️ {model_name} conversion ERROR! "
                                    f"Raw diff (first 3): {pred[:3]}, "
                                    f"Last close: {last_close:.2f}, "
                                    f"Converted (first): {first_pred:.2f}"
                                )
                                # Skip this model's prediction - it's invalid
                                continue
                        
                        logger.info(
                            f"{model_name}: Close_diff predictions (first 3): {pred[:3]}, "
                            f"Absolute prices (first 3): {pred_absolute[:3]}, "
                            f"Last close: {last_close:.2f}"
                        )
                    else:
                        # Model already predicts absolute prices (e.g., ExponentialSmoothing, GRU, Transformer)
                        # No conversion needed - already in price-space
                        pred_absolute = pred
                        
                        # Validation: Check if absolute prices are reasonable
                        if len(pred_absolute) > 0:
                            first_pred = pred_absolute[0] if isinstance(pred_absolute, np.ndarray) else pred_absolute[0]
                            change_pct = ((first_pred - last_close) / last_close * 100) if last_close > 0 else 0
                            
                            # Strict validation for extreme predictions
                            if first_pred < 0:
                                logger.error(
                                    f"⚠️ {model_name} prediction is negative: ${first_pred:.2f}. "
                                    f"Excluding from ensemble."
                                )
                                # Skip this model's prediction - it's invalid
                                continue
                            elif abs(change_pct) > 25:  # More than 25% change
                                logger.warning(
                                    f"⚠️ {model_name} prediction is extreme: "
                                    f"${first_pred:.2f} ({change_pct:+.2f}% from ${last_close:.2f}). "
                                    f"This may indicate model/scaler mismatch. "
                                    f"Still including but consider retraining model."
                                )
                                # Still include it, but log strong warning
                            elif first_pred > last_close * 10:
                                logger.warning(
                                    f"⚠️ {model_name} absolute price suspicious: "
                                    f"First pred: {first_pred:.2f}, Last close: {last_close:.2f}"
                                )
                                # Still include it, but log warning
                        
                        logger.info(
                            f"{model_name}: Absolute price predictions (first 3): {pred_absolute[:3] if len(pred_absolute) >= 3 else pred_absolute}, "
                            f"Last close: {last_close:.2f}"
                        )
                    
                    predictions.append(pred_absolute)
                    valid_weights.append(self.weights[i])
                    valid_names.append(model_name)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble")
        
        # Normalize weights for valid models (after any down-weighting adjustments)
        valid_weights = np.array(valid_weights)
        if valid_weights.sum() > 0:
            valid_weights = valid_weights / valid_weights.sum()
            logger.debug(
                f"Ensemble weights normalized after adjustments. "
                f"Final weights: {dict(zip(valid_names, [f'{w:.3f}' for w in valid_weights]))}"
            )
        else:
            # Fallback: equal weights if all were zero
            valid_weights = np.ones(len(valid_names)) / len(valid_names)
            logger.warning("All weights were zero after adjustments, using equal weights as fallback")
        
        # Weighted average of absolute prices (all predictions are now in same scale)
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        logger.info(f"Ensemble prediction using {len(valid_names)} models: {valid_names}")
        logger.info(f"Model weights: {dict(zip(valid_names, valid_weights))}")
        logger.info(
            f"Ensemble forecast (first 3): {ensemble_pred[:3]}, "
            f"Range: [{np.min(ensemble_pred):.2f}, {np.max(ensemble_pred):.2f}], "
            f"Last close: {last_close:.2f}, "
            f"First prediction change: {ensemble_pred[0] - last_close:.2f} "
            f"({((ensemble_pred[0] - last_close) / last_close * 100):.2f}%)"
        )
        
        # Log individual model contributions for debugging
        if len(predictions) > 0:
            logger.debug("Individual model predictions (first value):")
            for name, pred_arr in zip(valid_names, predictions):
                logger.debug(f"  {name}: {pred_arr[0]:.2f} (weight: {dict(zip(valid_names, valid_weights))[name]:.3f})")
        
        return ensemble_pred
    
    def update_errors(self, actual: float, predicted: float, model_index: Optional[int] = None) -> None:
        """
        Update error history for a specific model or all models.
        If model_index is None, assumes the error is for the ensemble prediction.
        """
        error = abs(actual - predicted)
        
        if model_index is not None and 0 <= model_index < len(self.recent_errors):
            # Update specific model
            self.recent_errors[model_index].append(error)
            # Keep only last 100 errors
            if len(self.recent_errors[model_index]) > 100:
                self.recent_errors[model_index].pop(0)
        else:
            # Update all models with ensemble error (for simplicity)
            # In practice, you'd track individual model errors
            for error_list in self.recent_errors:
                error_list.append(error)
                if len(error_list) > 100:
                    error_list.pop(0)
        
        self._update_weights()
    
    def update_model_errors(self, actuals: np.ndarray, predictions_dict: dict[str, np.ndarray]) -> None:
        """
        Update error history for each model individually based on their predictions.
        
        Args:
            actuals: Actual values (array of length steps)
            predictions_dict: Dict mapping model_name -> predictions array
        """
        if len(actuals) == 0:
            return
        
        # Update errors for each model
        for i, model_name in enumerate(self.model_names):
            if model_name in predictions_dict:
                pred = predictions_dict[model_name]
                if len(pred) == len(actuals):
                    # Calculate errors for this model
                    errors = np.abs(actuals - pred)
                    # Add to error history
                    for error in errors:
                        self.recent_errors[i].append(float(error))
                        # Keep only last 100 errors
                        if len(self.recent_errors[i]) > 100:
                            self.recent_errors[i].pop(0)
        
        self._update_weights()
    
    def get_model_performance(self) -> dict[str, dict]:
        """Get performance summary for all models."""
        performance = {}
        
        for i, (name, error_list) in enumerate(zip(self.model_names, self.recent_errors)):
            if not error_list:
                performance[name] = {
                    "weight": float(self.weights[i]),
                    "recent_mae": None,
                    "recent_rmse": None,
                    "error_count": 0,
                    "trend": None,  # "improving", "degrading", "stable"
                }
            else:
                recent_errors = error_list[-10:] if len(error_list) >= 10 else error_list
                older_errors = error_list[-20:-10] if len(error_list) >= 20 else []
                
                recent_mae = float(np.mean(recent_errors))
                recent_rmse = float(np.sqrt(np.mean([e**2 for e in recent_errors])))
                
                # Determine trend
                trend = None
                if older_errors:
                    older_mae = float(np.mean(older_errors))
                    if recent_mae < older_mae * 0.95:
                        trend = "improving"
                    elif recent_mae > older_mae * 1.05:
                        trend = "degrading"
                    else:
                        trend = "stable"
                
                performance[name] = {
                    "weight": float(self.weights[i]),
                    "recent_mae": recent_mae,
                    "recent_rmse": recent_rmse,
                    "error_count": len(error_list),
                    "trend": trend,
                }
        
        return performance


def create_default_ensemble(symbol: str, horizon: int = 1) -> AdaptiveEnsemble:
    """Create an ensemble with default models."""
    ensemble = AdaptiveEnsemble(symbol, horizon)
    
    # Add traditional models
    try:
        ensemble.add_forecaster(ARIMAForecaster(symbol, horizon), "ARIMA")
    except Exception as e:
        logger.warning(f"Could not add ARIMA: {e}")
    
    try:
        ensemble.add_forecaster(ExponentialSmoothingForecaster(symbol, horizon), "ExponentialSmoothing")
    except Exception as e:
        logger.warning(f"Could not add Exponential Smoothing: {e}")
    
    # Add neural models
    try:
        ensemble.add_forecaster(LSTMForecaster(symbol, horizon), "LSTM")
    except Exception as e:
        logger.warning(f"Could not add LSTM: {e}")
    
    try:
        ensemble.add_forecaster(GRUForecaster(symbol, horizon), "GRU")
    except Exception as e:
        logger.warning(f"Could not add GRU: {e}")
    
    try:
        ensemble.add_forecaster(TransformerForecaster(symbol, horizon), "Transformer")
    except Exception as e:
        logger.warning(f"Could not add Transformer: {e}")
    
    return ensemble


__all__ = ["AdaptiveEnsemble", "create_default_ensemble"]

