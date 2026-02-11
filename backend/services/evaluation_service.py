"""
Evaluation and metrics service for continuous monitoring.
Uses MongoDB exclusively for all data storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np

from ..db.unified_db import UnifiedDataStore
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """Service for continuous evaluation and metrics tracking."""

    @staticmethod
    def evaluate_forecast(
        forecast_id: str,  # MongoDB _id as string
        actual_value: float,
    ) -> Dict[str, Any]:
        """
        Evaluate a forecast when ground truth becomes available.
        Updates forecast record with errors in MongoDB.
        """
        from bson import ObjectId
        
        store = UnifiedDataStore()
        
        # Get forecast from MongoDB
        try:
            forecast_doc = store.mongo_db.forecasts.find_one({"_id": ObjectId(forecast_id)})
            if not forecast_doc:
                raise ValueError(f"Forecast {forecast_id} not found")
        except Exception as e:
            raise ValueError(f"Forecast {forecast_id} not found: {e}")

        predicted_value = forecast_doc["predicted_value"]
        absolute_error = abs(predicted_value - actual_value)
        percentage_error = (absolute_error / actual_value * 100) if actual_value != 0 else None

        # Update forecast in MongoDB
        store.mongo_db.forecasts.update_one(
            {"_id": ObjectId(forecast_id)},
            {
                "$set": {
                    "actual_value": actual_value,
                    "absolute_error": absolute_error,
                    "percentage_error": percentage_error,
                }
            }
        )

        logger.info(
            f"Evaluated forecast {forecast_id}: "
            f"pred={predicted_value:.2f}, actual={actual_value:.2f}, "
            f"error={absolute_error:.2f}"
        )

        forecast_doc["actual_value"] = actual_value
        forecast_doc["absolute_error"] = absolute_error
        forecast_doc["percentage_error"] = percentage_error
        forecast_doc["id"] = str(forecast_doc["_id"])
        del forecast_doc["_id"]
        
        return forecast_doc

    @staticmethod
    def compute_metrics(
        symbol: str,
        model_version_id: str,  # MongoDB _id as string
        horizon: int,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute MAE, RMSE, MAPE for a model version over recent forecasts.
        Creates a new Metric record in MongoDB.
        """
        store = UnifiedDataStore()
        
        # Get recent evaluated forecasts from MongoDB
        forecasts = list(
            store.mongo_db.forecasts.find({
                "symbol": symbol,
                "model_version_id": model_version_id,
                "actual_value": {"$ne": None}
            })
            .sort("target_time", -1)
            .limit(window)
        )

        if not forecasts:
            raise ValueError("No evaluated forecasts found for metrics computation")

        actuals = np.array([f["actual_value"] for f in forecasts])
        predictions = np.array([f["predicted_value"] for f in forecasts])

        mae = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

        # Get model_type from model_version
        model_version = store.get_model_version(model_version_id)
        model_type = "unknown"
        if model_version:
            model_type = model_version.get("model_type", "unknown")
            if model_type == "unknown" or not model_type:
                # Try to infer from version_tag or artifact_path
                version_tag = model_version.get("version_tag", "")
                if "arima" in version_tag.lower():
                    model_type = "ARIMA"
                elif "lstm" in version_tag.lower():
                    model_type = "LSTM"
                elif "gru" in version_tag.lower():
                    model_type = "GRU"
                elif "transformer" in version_tag.lower():
                    model_type = "Transformer"
                elif "ensemble" in version_tag.lower():
                    model_type = "Ensemble"

        # Save metrics to MongoDB with correct model_type
        store.save_metrics(
            symbol=symbol,
            model_type=model_type,
            metrics={
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            },
            model_version_id=model_version_id
        )

        logger.info(
            f"Computed metrics for {symbol} (model v{model_version_id}): "
            f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%"
        )

        return {
            "symbol": symbol,
            "calculated_at": datetime.now(timezone.utc),
            "horizon": horizon,
            "model_version_id": model_version_id,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }

    @staticmethod
    def get_metric_trends(
        symbol: str,
        model_version_id: Optional[str] = None,  # MongoDB _id as string
        model_type: Optional[str] = None,  # Filter by model type
        limit: int = 50,
    ) -> list[dict]:
        """Get metric history for visualization from MongoDB."""
        store = UnifiedDataStore()
        
        query = {"symbol": symbol}
        if model_version_id:
            query["model_version_id"] = model_version_id
        if model_type:
            query["model_type"] = model_type

        metrics = list(
            store.mongo_db.metrics.find(query)
            .sort("evaluated_at", -1)
            .limit(limit)
        )

        return [
            {
                "timestamp": m.get("evaluated_at").isoformat() if hasattr(m.get("evaluated_at"), "isoformat") else str(m.get("evaluated_at")),
                "mae": m.get("metrics", {}).get("mae", 0),
                "rmse": m.get("metrics", {}).get("rmse", 0),
                "mape": m.get("metrics", {}).get("mape", 0),
                "model_version_id": m.get("model_version_id"),
                "model_type": m.get("model_type", "unknown"),
            }
            for m in reversed(metrics)
        ]
    
    @staticmethod
    def get_metrics_by_model_types(
        symbol: str,
        model_types: Optional[list[str]] = None,
        limit: int = 50,
    ) -> dict[str, list[dict]]:
        """
        Get metrics for multiple model types at once.
        Returns a dictionary mapping model_type to list of metrics.
        """
        if model_types is None:
            model_types = ["Ensemble", "ARIMA", "LSTM", "GRU", "Transformer"]
        
        results = {}
        for model_type in model_types:
            try:
                metrics = EvaluationService.get_metric_trends(
                    symbol=symbol,
                    model_type=model_type,
                    limit=limit
                )
                if metrics:
                    results[model_type] = metrics
            except Exception as e:
                logger.warning(f"Error fetching metrics for {model_type}: {e}")
                results[model_type] = []
        
        return results
    
    @staticmethod
    def get_latest_metrics_by_model_types(
        symbol: str,
        model_types: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Get the latest metrics for each model type.
        Returns a dictionary mapping model_type to latest metrics dict.
        """
        if model_types is None:
            model_types = ["Ensemble", "ARIMA", "LSTM", "GRU", "Transformer"]
        
        store = UnifiedDataStore()
        results = {}
        
        for model_type in model_types:
            try:
                # Get the most recent metric for this model type
                metric = store.mongo_db.metrics.find_one(
                    {"symbol": symbol, "model_type": model_type},
                    sort=[("evaluated_at", -1)]
                )
                
                if metric:
                    results[model_type] = {
                        "mae": metric.get("metrics", {}).get("mae", 0),
                        "rmse": metric.get("metrics", {}).get("rmse", 0),
                        "mape": metric.get("metrics", {}).get("mape", 0),
                        "timestamp": metric.get("evaluated_at").isoformat() if hasattr(metric.get("evaluated_at"), "isoformat") else str(metric.get("evaluated_at")),
                        "model_version_id": metric.get("model_version_id"),
                    }
                else:
                    results[model_type] = {
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "timestamp": None,
                        "model_version_id": None,
                    }
            except Exception as e:
                logger.warning(f"Error fetching latest metrics for {model_type}: {e}")
                results[model_type] = {
                    "mae": None,
                    "rmse": None,
                    "mape": None,
                    "timestamp": None,
                    "model_version_id": None,
                }
        
        return results

    @staticmethod
    def auto_evaluate_pending(symbol: str) -> int:
        """
        Automatically evaluate forecasts when actual prices become available.
        Returns number of forecasts evaluated.
        Uses MongoDB exclusively.
        """
        from .data_ingestion import DataIngestionService

        store = UnifiedDataStore()
        
        # Get pending forecasts (no actual_value) from MongoDB
        pending = list(
            store.mongo_db.forecasts.find({
                "symbol": symbol,
                "actual_value": None
            })
        )

        evaluated_count = 0
        model_version_ids = set()
        
        for forecast in pending:
            # Get actual price at target_time
            actual_price = DataIngestionService.get_latest_price(symbol)
            if actual_price:
                target_time = forecast.get("target_time")
                if isinstance(target_time, str):
                    from dateutil import parser
                    target_time = parser.parse(target_time)
                
                if actual_price["timestamp"] >= target_time:
                    forecast_id = str(forecast["_id"])
                    EvaluationService.evaluate_forecast(forecast_id, actual_price["close"])
                    evaluated_count += 1
                    model_version_ids.add(forecast.get("model_version_id"))
        
        # Automatically compute metrics for affected model versions
        if evaluated_count > 0:
            for model_version_id in model_version_ids:
                try:
                    # Get the model version to get horizon
                    model_version = store.get_model_version(str(model_version_id))
                    if model_version:
                        # Compute metrics with a rolling window
                        EvaluationService.compute_metrics(
                            symbol=symbol,
                            model_version_id=str(model_version_id),
                            horizon=model_version.get("horizon", 1),
                            window=100
                        )
                except Exception as e:
                    logger.warning(f"Could not compute metrics for model {model_version_id}: {e}")

        return evaluated_count
    
    @staticmethod
    def auto_evaluate_all_symbols() -> dict[str, int]:
        """
        Automatically evaluate all pending forecasts for all symbols.
        Returns dict mapping symbol to count of evaluated forecasts.
        Uses MongoDB exclusively.
        """
        store = UnifiedDataStore()
        
        # Get all unique symbols with pending forecasts from MongoDB
        symbols = store.mongo_db.forecasts.distinct("symbol", {"actual_value": None})
        
        results = {}
        for symbol in symbols:
            try:
                count = EvaluationService.auto_evaluate_pending(symbol)
                if count > 0:
                    results[symbol] = count
            except Exception as e:
                logger.error(f"Error auto-evaluating {symbol}: {e}", exc_info=True)
        
        return results

