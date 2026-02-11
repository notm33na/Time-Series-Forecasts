"""
Evaluation and metrics API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.evaluation_service import EvaluationService
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


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


class EvaluateRequest(BaseModel):
    forecast_id: int
    actual_value: float


class MetricsRequest(BaseModel):
    symbol: str
    model_version_id: int
    horizon: int
    window: int = 100


@router.post("/evaluate")
async def evaluate_forecast(request: EvaluateRequest):
    """Evaluate a forecast with ground truth."""
    try:
        forecast = EvaluationService.evaluate_forecast(
            request.forecast_id,
            request.actual_value,
        )
        return {
            "forecast_id": forecast.id,
            "predicted_value": forecast.predicted_value,
            "actual_value": forecast.actual_value,
            "absolute_error": forecast.absolute_error,
            "percentage_error": forecast.percentage_error,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error evaluating forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics")
async def compute_metrics(request: MetricsRequest):
    """Compute metrics for a model version."""
    try:
        metric = EvaluationService.compute_metrics(
            request.symbol,
            request.model_version_id,
            request.horizon,
            window=request.window,
        )
        return {
            "symbol": metric.symbol,
            "mae": metric.mae,
            "rmse": metric.rmse,
            "mape": metric.mape,
            "calculated_at": metric.calculated_at.isoformat(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}")
async def get_metric_trends(
    symbol: str, 
    model_version_id: Optional[str] = None, 
    model_type: Optional[str] = None,
    limit: int = 50
):
    """Get metric trends over time, optionally filtered by model_type."""
    try:
        trends = EvaluationService.get_metric_trends(
            symbol, 
            model_version_id=model_version_id,
            model_type=model_type,
            limit=limit
        )
        return {"symbol": symbol, "trends": trends, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting metric trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}/all-models")
async def get_all_model_metrics(symbol: str, limit: int = 50):
    """Get metrics for all model types (Ensemble, ARIMA, LSTM, GRU, Transformer)."""
    try:
        all_metrics = EvaluationService.get_metrics_by_model_types(symbol, limit=limit)
        return {"symbol": symbol, "metrics_by_model": all_metrics, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting all model metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}/latest")
async def get_latest_metrics(symbol: str):
    """Get the latest metrics for each model type."""
    try:
        latest_metrics = EvaluationService.get_latest_metrics_by_model_types(symbol)
        return {"symbol": symbol, "latest_metrics": latest_metrics, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting latest metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}/weights")
async def get_ensemble_weights(symbol: str):
    """Get ensemble weights for individual models (ARIMA, LSTM, GRU, Transformer)."""
    try:
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        
        # Get latest ensemble model version
        ensemble_version = store.get_latest_model_version(symbol, "Ensemble")
        
        if not ensemble_version:
            return {
                "symbol": symbol,
                "weights": {},
                "status": "success",
                "message": "No ensemble model found"
            }
        
        # Extract weights from training_info
        training_info = ensemble_version.get("training_info", {})
        performance = training_info.get("performance", {})
        weights_list = training_info.get("weights", [])
        
        weights_dict = {}
        
        # First, try to get weights from performance dict (preferred method)
        # Performance dict structure: {"ARIMA": {"weight": 0.25, ...}, "LSTM": {"weight": 0.30, ...}, ...}
        if isinstance(performance, dict):
            for model_name, perf_data in performance.items():
                if isinstance(perf_data, dict) and "weight" in perf_data:
                    weights_dict[model_name] = float(perf_data["weight"])
        
        # If no weights found in performance, try to match weights list with model names
        if not weights_dict and isinstance(weights_list, list) and len(weights_list) > 0:
            # Try to get model_names from performance keys or from a separate field
            model_names_list = list(performance.keys()) if isinstance(performance, dict) else []
            
            # If we have model names, match them with weights
            if model_names_list and len(model_names_list) == len(weights_list):
                for i, model_name in enumerate(model_names_list):
                    weights_dict[model_name] = float(weights_list[i])
            elif len(weights_list) > 0:
                # Fallback: assume standard order (ARIMA, LSTM, GRU, Transformer)
                standard_order = ["ARIMA", "LSTM", "GRU", "Transformer"]
                for i, weight in enumerate(weights_list):
                    if i < len(standard_order):
                        weights_dict[standard_order[i]] = float(weight)
        
        return {
            "symbol": symbol,
            "weights": weights_dict,
            "ensemble_version": ensemble_version.get("version_tag"),
            "trained_at": ensemble_version.get("trained_at").isoformat() if ensemble_version.get("trained_at") else None,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting ensemble weights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-evaluate/{symbol}")
async def auto_evaluate(symbol: str):
    """Automatically evaluate pending forecasts."""
    try:
        count = EvaluationService.auto_evaluate_pending(symbol)
        return {"symbol": symbol, "evaluated_count": count, "status": "success"}
    except Exception as e:
        logger.error(f"Error auto-evaluating: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/{symbol}")
async def get_forecast_errors(symbol: str, limit: int = 50):
    """Get forecast errors for visualization (actual vs predicted)."""
    try:
        # Normalize symbol for database lookup (e.g., EURUSD -> EURUSD=X for forex)
        normalized_symbol = normalize_symbol(symbol)
        logger.info(f"Normalized symbol: {symbol} -> {normalized_symbol}")
        
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        # Query forecasts with actual values from MongoDB
        # Try normalized symbol first
        cursor = store.mongo_db.forecasts.find({
            "symbol": normalized_symbol,
            "actual_value": {"$ne": None}
        }).sort("target_time", -1).limit(limit)
        
        forecasts = list(cursor)
        
        # If no data found with normalized symbol, try original symbol (for backward compatibility)
        if not forecasts and normalized_symbol != symbol.upper():
            logger.info(f"No errors found for {normalized_symbol}, trying original symbol {symbol}")
            cursor = store.mongo_db.forecasts.find({
                "symbol": symbol,
                "actual_value": {"$ne": None}
            }).sort("target_time", -1).limit(limit)
            forecasts = list(cursor)
        
        errors = []
        for f in reversed(forecasts):  # Reverse to show chronological order
            errors.append({
                "date": f["target_time"].isoformat() if hasattr(f["target_time"], "isoformat") else str(f["target_time"]),
                "actual": f["actual_value"],
                "predicted": f["predicted_value"],
                "error": f.get("absolute_error", 0),
                "errorPercent": f.get("percentage_error", 0) or 0
            })
        
        return {
            "symbol": symbol,
            "errors": errors,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting forecast errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))