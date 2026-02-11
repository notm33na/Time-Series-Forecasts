"""
Enhanced forecast API endpoints supporting multiple models, instruments, and horizons.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from pathlib import Path
import pandas as pd
import numpy as np

from ..services.multi_model_service import MultiModelService
from ..services.data_ingestion import DataIngestionService
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/forecast", tags=["forecast"])


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol format for database lookup.
    Converts forex symbols from EURUSD to EURUSD=X format if needed.
    Also handles the reverse (EURUSD=X -> EURUSD=X, no change).
    
    Args:
        symbol: Symbol in any format (EURUSD, EURUSD=X, etc.)
    
    Returns:
        Normalized symbol for database lookup
    """
    # Common forex pairs that need =X suffix
    forex_pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "GBPJPY", "AUDJPY", "CADJPY",
        "CHFJPY", "NZDJPY", "AUDNZD", "AUDCAD", "AUDCHF", "CADCHF", "EURNZD",
        "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "NZDCAD", "NZDCHF"
    ]
    
    # If symbol already has =X suffix, return as-is
    if symbol.endswith("=X"):
        return symbol
    
    # If symbol is a known forex pair without =X, add it
    if symbol.upper() in [fp.upper() for fp in forex_pairs]:
        return f"{symbol.upper()}=X"
    
    # For other symbols, return as-is (stocks, crypto, etc.)
    return symbol.upper()


def detect_data_frequency(df: pd.DataFrame) -> str:
    """
    Detect the frequency of the data (daily, hourly, etc.).
    DEFAULT: Assumes hourly data (1h interval) as the standard format.
    Returns: 'hourly', 'daily', or 'unknown'
    """
    if df.empty or 'date' not in df.columns:
        return 'hourly'  # Default to hourly
    
    # Check if _interval field exists in MongoDB data
    if '_interval' in df.columns:
        interval = df['_interval'].iloc[0] if len(df) > 0 else None
        if interval in ['1h', '60m']:
            return 'hourly'
        elif interval in ['1d', '1wk', '1mo']:
            return 'daily'
    
    # Get time differences between consecutive dates
    dates = pd.to_datetime(df['date']).sort_values()
    if len(dates) < 2:
        return 'hourly'  # Default to hourly
    
    time_diffs = dates.diff().dropna()
    if len(time_diffs) == 0:
        return 'hourly'  # Default to hourly
    
    # Get median time difference
    median_diff = time_diffs.median()
    hours_diff = median_diff.total_seconds() / 3600
    
    # Classify based on median difference
    if 0.5 <= hours_diff <= 2:  # ~1 hour (hourly)
        return 'hourly'
    elif 20 <= hours_diff <= 28:  # ~24 hours (daily)
        return 'daily'
    else:
        # Default to hourly as the standard format
        return 'hourly'


def convert_horizon_to_steps(horizon_hours: int, data_frequency: str) -> int:
    """
    Convert hour-based horizon to number of forecast steps based on data frequency.
    
    Args:
        horizon_hours: Number of hours for the forecast
        data_frequency: 'daily' or 'hourly'
    
    Returns:
        Number of steps to forecast
    """
    if data_frequency == 'hourly':
        # For hourly data, 1 hour = 1 step
        return max(1, horizon_hours)
    elif data_frequency == 'daily':
        # For daily data, convert hours to days
        # 1h, 3h → 1 day (minimum 1 step)
        # 24h → 1 day
        # 72h → 3 days
        days = max(1, round(horizon_hours / 24))
        return days
    else:
        # Default: assume daily
        return max(1, round(horizon_hours / 24))


class ForecastRequest(BaseModel):
    symbol: str
    model_type: Literal["ARIMA", "VAR", "ExponentialSmoothing", "MovingAverage", "LSTM", "GRU", "Transformer", "Ensemble"] = "Ensemble"
    horizon: Literal["1h", "3h", "24h", "72h"] = "24h"
    model_version_id: Optional[int] = None


class TrainRequest(BaseModel):
    symbol: str
    model_type: Literal["ARIMA", "VAR", "ExponentialSmoothing", "MovingAverage", "LSTM", "GRU", "Transformer", "Ensemble"] = "Ensemble"
    horizon: Literal["1h", "3h", "24h", "72h"] = "24h"
    lag_window: Optional[int] = None


class FineTuneRequest(BaseModel):
    symbol: str
    model_type: Literal["LSTM", "GRU", "Transformer"]
    horizon: Literal["1h", "3h", "24h", "72h"] = "24h"
    new_data_window: int = 50


@router.post("/run")
async def run_forecast(request: ForecastRequest):
    """
    Generate forecast using specified model and horizon.
    Only loads pre-trained models - no training happens here.
    """
    import warnings
    # Suppress expected warnings that don't affect functionality
    warnings.filterwarnings('ignore', category=UserWarning, module='keras.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.*')
    
    try:
        from ..services.model_loader import ModelLoader
        
        # Normalize symbol for database lookup (e.g., EURUSD -> EURUSD=X for forex)
        normalized_symbol = normalize_symbol(request.symbol)
        logger.info(f"Normalized symbol: {request.symbol} -> {normalized_symbol}")
        
        # Get data for prediction
        service = DataIngestionService()
        # Try normalized symbol first
        df = service.get_prices(normalized_symbol, limit=200)
        
        # If no data found with normalized symbol, try original symbol (for backward compatibility)
        if df.empty and normalized_symbol != request.symbol.upper():
            logger.info(f"No data found for {normalized_symbol}, trying original symbol {request.symbol}")
            df = service.get_prices(request.symbol, limit=200)
        
        if df.empty:
            error_msg = f"No price data found for {request.symbol} (normalized: {normalized_symbol}). Please ingest data first using /api/data/ingest."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(df) < 60:
            error_msg = f"Insufficient data for prediction: need at least 60 points, got {len(df)} for {request.symbol} (normalized: {normalized_symbol})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log data quality info
        logger.info(
            f"Data for {request.symbol}: {len(df)} records, "
            f"date range: {df['date'].min()} to {df['date'].max()}, "
            f"latest close: {df['close'].iloc[-1]:.2f}"
        )
        
        # Detect data frequency
        data_frequency = detect_data_frequency(df)
        logger.info(f"Detected data frequency: {data_frequency} for {request.symbol}")
        
        # Convert hour-based horizon to steps based on data frequency
        horizon_hours = MultiModelService.HORIZON_MAP.get(request.horizon, 24)
        horizon_steps = convert_horizon_to_steps(horizon_hours, data_frequency)
        logger.info(f"Converted {request.horizon} ({horizon_hours}h) to {horizon_steps} steps for {data_frequency} data")
        
        # Handle both uppercase and lowercase column names
        if "Close" in df.columns:
            close_col = "Close"
        elif "close" in df.columns:
            close_col = "close"
        else:
            raise ValueError(f"Price data missing 'close' column. Available columns: {list(df.columns)}")
        
        # Ensure date column exists and is datetime
        if 'date' not in df.columns:
            raise ValueError(f"Price data missing 'date' column. Available columns: {list(df.columns)}")
        
        # Create prices Series with date as index
        # Data should already be sorted by date (ascending) from get_prices()
        df['date'] = pd.to_datetime(df['date'], utc=True)
        prices = pd.Series(df[close_col].values, index=df['date'])
        
        # Ensure prices is sorted by index (chronological order, oldest to newest)
        prices = prices.sort_index()
        
        # Validate that prices are in ascending order
        if not prices.index.is_monotonic_increasing:
            logger.warning(f"Prices for {request.symbol} are not in ascending order. Re-sorting...")
            prices = prices.sort_index()
        
        # Log final data info
        logger.info(
            f"Prices Series for {request.symbol}: {len(prices)} points, "
            f"index range: {prices.index[0]} to {prices.index[-1]}, "
            f"last close: {prices.iloc[-1]:.2f}"
        )
        
        # For LSTM, we need the full DataFrame for prediction
        # For other models, we can use Series
        use_dataframe_for_prediction = (request.model_type == "LSTM")
        
        # For Ensemble, load individual models and create ensemble
        if request.model_type == "Ensemble":
            # Load individual models (no training)
            # Note: horizon_steps is the converted steps based on data frequency
            # Models are flexible and can predict any number of steps
            loaded_models = ModelLoader.load_ensemble_models(request.symbol, horizon_steps)
            
            # Create ensemble from loaded models
            ensemble = ModelLoader.create_ensemble_from_loaded_models(
                request.symbol, horizon_steps, loaded_models
            )
            
            # Check if ensemble has any fitted models
            fitted_count = sum(1 for f in ensemble.forecasters if hasattr(f, 'is_fitted') and f.is_fitted)
            if fitted_count == 0:
                raise ValueError(
                    f"Ensemble has no fitted models. "
                    f"Loaded {len(loaded_models)} models but none are fitted. "
                    f"Please ensure models are properly trained."
                )
            
            logger.info(f"Ensemble has {fitted_count}/{len(ensemble.forecasters)} fitted models")
            
            # Get last date before prediction (needed for saving predictions)
            last_date = prices.index[-1] if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(df['date'].iloc[-1])
            
            # Generate ensemble prediction
            # Pass both Series and DataFrame to ensemble (LSTM needs DataFrame)
            try:
                ensemble_forecast = ensemble.predict(prices, data_df=df, steps=horizon_steps)
            except ValueError as e:
                if "No valid predictions" in str(e):
                    raise ValueError(
                        f"All models in ensemble failed to predict. "
                        f"This usually means models need to be retrained or data format is incorrect. "
                        f"Original error: {e}"
                    )
                raise
            
            # Generate individual model forecasts for display purposes
            # ARIMA and LSTM predict Close_diff, which must be converted to absolute prices
            # Conversion formula: pred_price = last_close + cumsum(predicted_close_diff)
            # GRU and Transformer already predict absolute prices (no conversion needed)
            individual_forecasts = {}
            last_close_price = float(prices.iloc[-1])
            diff_models = ["ARIMA", "LSTM"]  # Models that predict Close_diff
            
            logger.info(f"Generating individual forecasts for models: {list(loaded_models.keys())}")
            
            for model_type, (model, _) in loaded_models.items():
                try:
                    logger.debug(f"Generating {model_type} individual forecast...")
                    # For LSTM, pass DataFrame; for others, pass Series
                    if model_type == "LSTM":
                        # LSTM needs the last row as DataFrame with all OHLCV columns
                        # CRITICAL: Must match the exact feature columns the model was trained on
                        last_row = df.iloc[[-1]].copy()
                        
                        # Get the feature columns the model expects (from training)
                        expected_features = getattr(model, 'feature_cols', None)
                        if expected_features is None:
                            logger.error(f"LSTM model missing feature_cols attribute. Cannot generate prediction.")
                            raise ValueError("LSTM model missing feature_cols. Model may not be properly loaded.")
                        
                        logger.debug(f"LSTM expects features: {expected_features}")
                        logger.debug(f"Available DataFrame columns: {list(last_row.columns)}")
                        
                        # Create a DataFrame with exactly the features the model expects
                        # CRITICAL: The scaler was fit on num_cols (including Close), so we need Close too
                        # even though it's not in feature_cols
                        input_df = pd.DataFrame()
                        all_needed_cols = list(expected_features)
                        
                        # Add Close if not already in expected_features (scaler needs it)
                        close_col_name = getattr(model, 'close_col_name', 'Close')
                        if close_col_name not in all_needed_cols and 'Close' not in all_needed_cols:
                            all_needed_cols.append(close_col_name)
                        
                        for feat_col in all_needed_cols:
                            # Try exact match first
                            if feat_col in last_row.columns:
                                input_df[feat_col] = last_row[feat_col].values
                            else:
                                # Try case-insensitive match
                                col_lower = feat_col.lower()
                                matching = [c for c in last_row.columns if c.lower() == col_lower]
                                if matching:
                                    input_df[feat_col] = last_row[matching[0]].values
                                else:
                                    # Create missing column using Close as fallback
                                    close_val = last_row.get('Close', last_row.get('close', prices.iloc[-1]))
                                    if isinstance(close_val, pd.Series):
                                        close_val = close_val.iloc[0] if len(close_val) > 0 else prices.iloc[-1]
                                    input_df[feat_col] = close_val
                                    logger.warning(f"LSTM: Created missing feature '{feat_col}' using Close price: {close_val}")
                        
                        # Validate input data quality
                        if len(input_df) > 0:
                            row = input_df.iloc[0]
                            ohlc_cols = [c for c in input_df.columns if c.lower() in ['open', 'high', 'low', 'close']]
                            if len(ohlc_cols) >= 3:
                                ohlc_values = [row[c] for c in ohlc_cols]
                                if len(set(ohlc_values)) == 1:
                                    logger.warning(
                                        f"⚠️ LSTM input data quality issue: All OHLC values are identical ({ohlc_values[0]:.2f}). "
                                        f"This will produce poor predictions. Data may be stale or incorrect."
                                    )
                            
                            if 'Volume' in input_df.columns and row.get('Volume', 0) == 0:
                                logger.warning(
                                    f"⚠️ LSTM input data quality issue: Volume is zero. "
                                    f"This may affect prediction quality."
                                )
                        
                        logger.debug(f"LSTM input DataFrame shape: {input_df.shape}, columns: {list(input_df.columns)}")
                        logger.debug(f"LSTM input DataFrame values (first row): {input_df.iloc[0].to_dict()}")
                        
                        pred = model.predict(input_df, steps=horizon_steps)
                        logger.debug(f"LSTM raw prediction shape: {pred.shape}, first 3 values: {pred[:3] if len(pred) >= 3 else pred}")
                    else:
                        pred = model.predict(prices, steps=horizon_steps)
                        logger.debug(f"{model_type} raw prediction shape: {pred.shape}, first 3 values: {pred[:3] if len(pred) >= 3 else pred}")
                    
                    # Convert Close_diff to absolute prices for consistency with ensemble
                    # ARIMA and LSTM predict Close_diff, which must be converted to absolute prices
                    # Conversion formula: pred_price = last_close + cumsum(predicted_close_diff)
                    if model_type in diff_models:
                        # Convert differenced predictions to absolute prices
                        # For multi-step: pred_absolute[i] = last_close + sum(pred_diff[0:i+1])
                        pred_absolute = last_close_price + np.cumsum(pred)
                        
                        # Validation: Check if conversion looks correct
                        if len(pred_absolute) > 0:
                            first_pred = pred_absolute[0]
                            # First prediction should be close to last_close (within reasonable range)
                            # If it's negative or way off, something is wrong
                            if first_pred < 0 or abs(first_pred - last_close_price) > last_close_price * 2:
                                logger.error(
                                    f"⚠️ {model_type} conversion looks wrong! "
                                    f"Raw diff (first 3): {pred[:3] if len(pred) >= 3 else pred}, "
                                    f"Last close: {last_close_price:.2f}, "
                                    f"Converted (first): {first_pred:.2f}"
                                )
                            else:
                                logger.info(
                                    f"{model_type} conversion OK: "
                                    f"Raw diff (first): {pred[0]:.4f}, "
                                    f"Last close: {last_close_price:.2f}, "
                                    f"Converted (first): {first_pred:.2f}"
                                )
                        
                        individual_forecasts[model_type.lower() + "_forecast"] = pred_absolute.tolist()
                        logger.info(f"{model_type} individual forecast (first 3 absolute): {pred_absolute[:3]}, last_close: {last_close_price:.2f}")
                    else:
                        # Already absolute prices - validate they're reasonable
                        if len(pred) > 0:
                            first_pred = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
                            change_pct = ((first_pred - last_close_price) / last_close_price * 100) if last_close_price > 0 else 0
                            
                            # More strict validation for extreme predictions
                            if first_pred < 0:
                                logger.error(
                                    f"⚠️ {model_type} prediction is negative: ${first_pred:.2f}. "
                                    f"This is invalid and will be excluded from ensemble."
                                )
                            elif abs(change_pct) > 25:  # More than 25% change
                                logger.warning(
                                    f"⚠️ {model_type} prediction is extreme: "
                                    f"${first_pred:.2f} ({change_pct:+.2f}% from ${last_close_price:.2f}). "
                                    f"This may indicate model/scaler mismatch or poor training."
                                )
                            elif first_pred > last_close_price * 10:
                                logger.warning(
                                    f"⚠️ {model_type} absolute price looks suspicious: "
                                    f"First pred: {first_pred:.2f}, Last close: {last_close_price:.2f}"
                                )
                        
                        individual_forecasts[model_type.lower() + "_forecast"] = pred.tolist()
                        logger.info(f"{model_type} individual forecast (first 3 absolute): {pred[:3] if len(pred) >= 3 else pred}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"❌ Could not generate {model_type} forecast: {error_msg}",
                        exc_info=True
                    )
                    # Log additional context for LSTM failures
                    if model_type == "LSTM":
                        logger.error(
                            f"LSTM Debug Info: "
                            f"Model has feature_cols: {hasattr(model, 'feature_cols')}, "
                            f"Model is_fitted: {getattr(model, 'is_fitted', False)}, "
                            f"Model has scaler: {hasattr(model, 'scaler')}, "
                            f"DataFrame columns: {list(df.columns) if 'df' in locals() else 'N/A'}"
                        )
                    # Don't include failed models in individual_forecasts
                    # This way the frontend can show "No prediction available"
            
            logger.info(f"Generated individual forecasts for: {list(individual_forecasts.keys())}")
            
            # Integrate portfolio management if enabled
            portfolio_data = None
            try:
                from ..trading.integration_example import integrate_portfolio_with_forecast
                
                current_price = float(prices.iloc[-1])
                forecast_price = float(ensemble_forecast[0]) if len(ensemble_forecast) > 0 else current_price
                
                # Get model version info for tracking
                model_version_id = ensemble_version_id
                model_type = request.model_type
                
                portfolio_data = integrate_portfolio_with_forecast(
                    symbol=request.symbol,
                    current_price=current_price,
                    forecast_price=forecast_price,
                    horizon=request.horizon,
                    model_version_id=model_version_id,
                    model_type=model_type
                )
                logger.info(f"Portfolio integration completed for {request.symbol}")
            except Exception as e:
                logger.warning(f"Portfolio integration failed (non-critical): {e}")
                # Don't fail the forecast if portfolio integration fails
            
            # Save ensemble predictions to MongoDB for frontend access
            try:
                from ..db.unified_db import UnifiedDataStore
                from datetime import timedelta
                
                store = UnifiedDataStore()
                
                # Calculate time delta based on data frequency
                if data_frequency == 'hourly':
                    time_delta = timedelta(hours=1)
                else:  # daily
                    time_delta = timedelta(days=1)
                
                # Save ensemble predictions (use a composite model_version_id for ensemble)
                ensemble_version_id = f"ensemble_{request.symbol}_{request.horizon}"
                for i, pred_value in enumerate(ensemble_forecast):
                    target_time = last_date + (i + 1) * time_delta
                    # Convert to datetime if needed
                    if isinstance(target_time, pd.Timestamp):
                        target_time = target_time.to_pydatetime()
                    
                    store.save_forecast_record(
                        symbol=request.symbol,
                        target_time=target_time,
                        horizon=horizon_steps,
                        predicted_value=float(pred_value),
                        model_version_id=ensemble_version_id,
                    )
                
                logger.info(f"Saved {len(ensemble_forecast)} ensemble forecast records to MongoDB for {request.symbol}")
            except Exception as e:
                logger.warning(f"Could not save ensemble predictions to MongoDB (non-critical): {e}")
            
            result = {
                "symbol": request.symbol,
                "model_type": request.model_type,
                "horizon": request.horizon,
                "data_frequency": data_frequency,
                "forecast_steps": horizon_steps,
                "ensemble_forecast": ensemble_forecast.tolist(),
                "predictions": [  # Format for frontend charts
                    {
                        "date": (last_date + (i + 1) * time_delta).isoformat(),
                        "prediction": float(pred_value)
                    }
                    for i, pred_value in enumerate(ensemble_forecast)
                ],
                **individual_forecasts,
                "status": "success",
            }
            
            if portfolio_data:
                result["portfolio"] = portfolio_data
            
            return result
        else:
            # For individual models, we need to load and predict with the correct steps
            # Since MultiModelService.predict uses HORIZON_MAP directly, we'll load the model
            # and predict manually with the correct steps
            model, model_version = ModelLoader.load_latest_model(request.symbol, request.model_type)
            
            # Get last date before prediction (needed for saving predictions)
            last_date = prices.index[-1] if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(df['date'].iloc[-1])
            
            # Predict with the converted steps
            # For LSTM, pass DataFrame; for others, pass Series
            if use_dataframe_for_prediction:
                # Get last row for LSTM prediction (needs DataFrame with features)
                last_row = df.iloc[[-1]]
                predictions = model.predict(last_row, steps=horizon_steps)
            else:
                predictions = model.predict(prices, steps=horizon_steps)
            
            # Integrate portfolio management if enabled
            portfolio_data = None
            try:
                from ..trading.integration_example import integrate_portfolio_with_forecast
                
                current_price = float(prices.iloc[-1])
                forecast_price = float(predictions[0]) if len(predictions) > 0 else current_price
                
                # Get model version info for tracking
                model_version_id = str(model_version["id"])
                model_type = request.model_type
                
                portfolio_data = integrate_portfolio_with_forecast(
                    symbol=request.symbol,
                    current_price=current_price,
                    forecast_price=forecast_price,
                    horizon=request.horizon,
                    model_version_id=model_version_id,
                    model_type=model_type
                )
                logger.info(f"Portfolio integration completed for {request.symbol}")
            except Exception as e:
                logger.warning(f"Portfolio integration failed (non-critical): {e}")
                # Don't fail the forecast if portfolio integration fails
            
            # Save predictions to MongoDB for frontend access
            try:
                from ..db.unified_db import UnifiedDataStore
                from datetime import timedelta
                
                store = UnifiedDataStore()
                
                # Calculate time delta based on data frequency
                if data_frequency == 'hourly':
                    time_delta = timedelta(hours=1)
                else:  # daily
                    time_delta = timedelta(days=1)
                
                # Save each prediction point
                for i, pred_value in enumerate(predictions):
                    target_time = last_date + (i + 1) * time_delta
                    # Convert to datetime if needed
                    if isinstance(target_time, pd.Timestamp):
                        target_time = target_time.to_pydatetime()
                    
                    store.save_forecast_record(
                        symbol=request.symbol,
                        target_time=target_time,
                        horizon=horizon_steps,
                        predicted_value=float(pred_value),
                        model_version_id=str(model_version["id"]),
                    )
                
                logger.info(f"Saved {len(predictions)} forecast records to MongoDB for {request.symbol}")
            except Exception as e:
                logger.warning(f"Could not save predictions to MongoDB (non-critical): {e}")
            
            result = {
                "symbol": request.symbol,
                "model_type": request.model_type,
                "horizon": request.horizon,
                "data_frequency": data_frequency,
                "forecast_steps": horizon_steps,
                "forecast": predictions.tolist(),
                "predictions": [  # Format for frontend charts
                    {
                        "date": (last_date + (i + 1) * time_delta).isoformat(),
                        "prediction": float(pred_value)
                    }
                    for i, pred_value in enumerate(predictions)
                ],
                "model_version_id": model_version["id"],
                "version_tag": model_version.get("version_tag", ""),
                "source": model_version.get("source"),
                "trained_at": model_version.get("trained_at").isoformat() if model_version.get("trained_at") else None,
                "status": "success",
            }
            
            if portfolio_data:
                result["portfolio"] = portfolio_data
            
            return result
    except ValueError as e:
        # No model found or data issue - return helpful error
        error_msg = str(e)
        logger.error(f"Forecast error: {error_msg}")
        
        # Check if it's a model not found error
        if "No" in error_msg and ("model" in error_msg.lower() or "models" in error_msg.lower()):
            status_code = 404
            detail = f"{error_msg}. Please train models first using training scripts (train_arima.py, train_lstm.py, etc.) or /api/forecast/train."
        elif "No price data" in error_msg or "Insufficient data" in error_msg:
            status_code = 404
            detail = f"{error_msg}. Please ingest data first using /api/data/ingest."
        elif "fitted" in error_msg.lower() or "No valid predictions" in error_msg:
            status_code = 400
            detail = f"{error_msg}. Models may need to be retrained."
        else:
            status_code = 400
            detail = error_msg
        
        raise HTTPException(status_code=status_code, detail=detail)
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(f"Model file not found: {error_msg}")
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact not found: {error_msg}. Please retrain the model."
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else f"{error_type} occurred"
        logger.error(f"Error generating forecast ({error_type}): {error_msg}\n{error_trace}")
        # Provide more detailed error message
        error_detail = f"{error_type}: {error_msg}. Check server logs for full traceback."
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a new forecasting model."""
    try:
        model, model_version = MultiModelService.train_model(
            request.symbol,
            model_type=request.model_type,
            horizon=request.horizon,
            lag_window=request.lag_window,
        )
        
        return {
            "symbol": request.symbol,
            "model_type": request.model_type,
            "horizon": request.horizon,
            "model_version_id": model_version.id,
            "version_tag": model_version.version_tag,
            "trained_at": model_version.trained_at.isoformat(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune")
async def fine_tune_model(request: FineTuneRequest):
    """Fine-tune a neural model with rolling window."""
    try:
        model, model_version = MultiModelService.fine_tune_neural_model(
            request.symbol,
            model_type=request.model_type,
            horizon=request.horizon,
            new_data_window=request.new_data_window,
        )
        
        return {
            "symbol": request.symbol,
            "model_type": request.model_type,
            "horizon": request.horizon,
            "model_version_id": model_version.id,
            "version_tag": model_version.version_tag,
            "trained_at": model_version.trained_at.isoformat(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error fine-tuning model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{symbol}")
async def get_predictions(symbol: str, model_version_id: Optional[str] = None, limit: int = 100):
    """
    Retrieve saved predictions for a symbol.
    Returns predictions in format suitable for frontend charts.
    """
    try:
        from ..db.unified_db import UnifiedDataStore
        from datetime import datetime, timezone
        
        store = UnifiedDataStore()
        
        # Build query
        query = {"symbol": symbol}
        if model_version_id:
            query["model_version_id"] = model_version_id
        
        # Get most recent predictions
        cursor = store.mongo_db.forecasts.find(query).sort("target_time", -1).limit(limit)
        forecasts = list(cursor)
        
        if not forecasts:
            return {
                "symbol": symbol,
                "predictions": [],
                "count": 0,
                "status": "success",
                "message": "No predictions found. Generate a forecast first using /api/forecast/run"
            }
        
        # Format for frontend
        predictions = []
        for forecast in forecasts:
            predictions.append({
                "date": forecast["target_time"].isoformat() if isinstance(forecast["target_time"], datetime) else str(forecast["target_time"]),
                "prediction": forecast["predicted_value"],
                "actual_value": forecast.get("actual_value"),
                "error": forecast.get("absolute_error"),
                "model_version_id": forecast.get("model_version_id"),
            })
        
        # Sort by date (oldest first for chart continuity)
        predictions.sort(key=lambda x: x["date"])
        
        return {
            "symbol": symbol,
            "predictions": predictions,
            "count": len(predictions),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{symbol}")
async def list_models(symbol: str, model_type: Optional[str] = None):
    """List all trained models for a symbol."""
    try:
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        models = store.list_model_versions(symbol, model_type)
        
        return {
            "symbol": symbol,
            "models": [
                {
                    "id": m["id"],
                    "model_type": m.get("model_type", "Unknown"),
                    "horizon": m.get("horizon", 1),
                    "horizon_string": m.get("training_info", {}).get("horizon", "unknown"),
                    "version_tag": m.get("version_tag", ""),
                    "trained_at": m.get("trained_at").isoformat() if m.get("trained_at") else None,
                    "artifact_path": m.get("artifact_path", ""),
                }
                for m in models
            ],
            "total_count": len(models),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/{symbol}")
async def debug_models(symbol: str):
    """Debug endpoint to check what models exist and what's available."""
    try:
        from ..services.data_ingestion import DataIngestionService
        from ..services.model_loader import ModelLoader
        
        result = {
            "symbol": symbol,
            "data_available": False,
            "data_count": 0,
            "models_available_db": False,
            "models_available_filesystem": False,
            "models_db": [],
            "models_filesystem": [],
            "search_paths": [],
            "recommendations": []
        }
        
        # Check data
        try:
            service = DataIngestionService()
            df = service.get_prices(symbol, limit=10)
            result["data_available"] = not df.empty
            result["data_count"] = len(df) if not df.empty else 0
            if df.empty:
                result["recommendations"].append("No price data found. Please ingest data first using /api/data/ingest")
        except Exception as e:
            result["data_error"] = str(e)
            result["recommendations"].append(f"Error checking data: {e}")
        
        # Check models in database
        try:
            from ..db.unified_db import UnifiedDataStore
            
            store = UnifiedDataStore()
            all_models = store.list_model_versions(symbol)
            
            result["models_available_db"] = len(all_models) > 0
            result["models_db"] = [
                {
                    "id": m["id"],
                    "model_type": m.get("training_info", {}).get("model_type", "Unknown"),
                    "horizon_steps": m.get("horizon", 1),
                    "horizon_string": m.get("training_info", {}).get("horizon", "unknown"),
                    "version_tag": m.get("version_tag", ""),
                    "trained_at": m.get("trained_at").isoformat() if m.get("trained_at") else None,
                    "artifact_exists": Path(m.get("artifact_path", "")).exists() if m.get("artifact_path") else False,
                }
                for m in all_models
            ]
            
            if not all_models:
                result["recommendations"].append("No models found in database. Checking filesystem...")
            else:
                model_types = set(m.get("training_info", {}).get("model_type", "Unknown") for m in all_models)
                result["available_model_types_db"] = list(model_types)
        except Exception as e:
            result["models_db_error"] = str(e)
            result["recommendations"].append(f"Error checking database models: {e}")
        
        # Check models in filesystem
        try:
            from ..config import get_settings
            
            settings = get_settings()
            possible_dirs = [
                settings.models_dir,
                Path(__file__).parent.parent / "artifacts",  # backend/artifacts
                Path(__file__).parent.parent.parent / "backend" / "artifacts",
                Path(__file__).parent.parent.parent / "artifacts",
                Path(__file__).parent.parent.parent / "notebooks",  # project/notebooks
            ]
            
            result["search_paths"] = [str(d) for d in possible_dirs]
            
            models_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    models_dir = dir_path
                    result["models_dir_found"] = str(models_dir)
                    break
            
            if models_dir:
                # Search for model files
                model_types = ["ARIMA", "LSTM", "GRU", "Transformer", "ExponentialSmoothing"]
                for model_type in model_types:
                    if model_type in ["LSTM", "GRU", "Transformer"]:
                        pattern = f"{symbol}_{model_type.lower()}_*.pkl"
                        files = list(models_dir.glob(pattern))
                        files = [f for f in files if '.weights.pkl' not in f.name]
                    else:
                        pattern = f"{symbol}_{model_type.lower()}_*.pkl"
                        files = list(models_dir.glob(pattern))
                    
                    if files:
                        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        result["models_filesystem"].append({
                            "model_type": model_type,
                            "file": files[0].name,
                            "path": str(files[0]),
                            "size": files[0].stat().st_size,
                            "modified": files[0].stat().st_mtime
                        })
                        result["models_available_filesystem"] = True
            else:
                result["recommendations"].append(f"Models directory not found in any of: {[str(d) for d in possible_dirs]}")
        
        except Exception as e:
            result["models_filesystem_error"] = str(e)
            result["recommendations"].append(f"Error checking filesystem models: {e}")
        
        # Try to actually load ensemble models
        try:
            loaded_models = ModelLoader.load_ensemble_models(symbol, 24)
            result["ensemble_load_success"] = True
            result["ensemble_models_loaded"] = list(loaded_models.keys())
        except Exception as e:
            result["ensemble_load_error"] = str(e)
            result["ensemble_load_success"] = False
        
        if not result["models_available_db"] and not result["models_available_filesystem"]:
            result["recommendations"].append("No models found in database or filesystem. Train models using training scripts (train_arima.py, train_lstm.py, etc.)")
        
        return result
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

