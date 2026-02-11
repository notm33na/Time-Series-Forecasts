"""
Scheduled retraining service for automatic model updates.
Supports scheduled retraining, performance-based retraining, and new data triggers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

from ..config import get_settings
from ..services.model_service import ModelService
from ..services.evaluation_service import EvaluationService
from ..services.data_ingestion import DataIngestionService
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ScheduledRetrainingService:
    """Service for automatic model retraining based on schedules and triggers."""
    
    def __init__(self):
        self.running = False
        self.tasks: dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start the scheduled retraining service."""
        self.running = True
        logger.info("Scheduled retraining service started")
        
        # Start background tasks
        asyncio.create_task(self._scheduled_retraining_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._new_data_monitoring_loop())
    
    async def stop(self):
        """Stop the scheduled retraining service."""
        self.running = False
        for task in self.tasks.values():
            task.cancel()
        logger.info("Scheduled retraining service stopped")
    
    async def _scheduled_retraining_loop(self):
        """Periodic retraining based on schedule."""
        retrain_interval_hours = getattr(settings, 'retrain_interval_hours', 24)
        
        while self.running:
            try:
                await asyncio.sleep(retrain_interval_hours * 3600)  # Convert hours to seconds
                
                if not self.running:
                    break
                
                logger.info("Scheduled retraining triggered")
                await self.retrain_all_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduled retraining loop: {e}", exc_info=True)
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _performance_monitoring_loop(self):
        """Monitor model performance and retrain if degradation detected."""
        check_interval_hours = getattr(settings, 'performance_check_interval_hours', 6)
        error_threshold = getattr(settings, 'performance_degradation_threshold', 0.15)  # 15% increase
        
        while self.running:
            try:
                await asyncio.sleep(check_interval_hours * 3600)
                
                if not self.running:
                    break
                
                logger.info("Performance monitoring check")
                await self.check_and_retrain_degraded_models(error_threshold)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(1800)  # Wait 30 minutes before retrying
    
    async def _new_data_monitoring_loop(self):
        """Monitor for new data and trigger incremental updates."""
        check_interval_minutes = getattr(settings, 'new_data_check_interval_minutes', 60)
        new_data_threshold = getattr(settings, 'new_data_retrain_threshold', 10)  # Retrain after 10 new points
        
        while self.running:
            try:
                await asyncio.sleep(check_interval_minutes * 60)
                
                if not self.running:
                    break
                
                logger.info("New data monitoring check")
                await self.check_and_adapt_for_new_data(new_data_threshold)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in new data monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(600)  # Wait 10 minutes before retrying
    
    async def retrain_all_models(self):
        """Retrain all active models. Uses MongoDB."""
        # Get list of symbols from MongoDB
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        symbols = store.mongo_db.model_versions.distinct("symbol")
        
        for symbol in symbols:
            try:
                logger.info(f"Retraining model for {symbol}")
                ModelService.train_new_model(symbol, initial=False)
                logger.info(f"Successfully retrained model for {symbol}")
            except Exception as e:
                logger.error(f"Error retraining model for {symbol}: {e}", exc_info=True)
    
    async def check_and_retrain_degraded_models(self, error_threshold: float):
        """Check model performance and retrain if degradation detected. Uses MongoDB."""
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        
        # Get latest models from MongoDB
        models = list(
            store.mongo_db.model_versions.find()
            .sort("trained_at", -1)
        )
        
        for model in models:
            try:
                model_id = str(model["_id"])
                # Get recent metrics from MongoDB
                recent_metrics = list(
                    store.mongo_db.metrics.find({"model_version_id": model_id})
                    .sort("evaluated_at", -1)
                    .limit(2)
                )
                
                if len(recent_metrics) < 2:
                    continue
                
                # Compare latest vs previous
                latest_mape = recent_metrics[0].get("metrics", {}).get("mape", 0)
                previous_mape = recent_metrics[1].get("metrics", {}).get("mape", 0)
                
                if previous_mape > 0:
                    degradation = (latest_mape - previous_mape) / previous_mape
                    
                    if degradation > error_threshold:
                        logger.warning(
                            f"Performance degradation detected for {model['symbol']} "
                            f"(MAPE increased by {degradation:.1%}), retraining..."
                        )
                        ModelService.train_new_model(model["symbol"], initial=False)
                        
            except Exception as e:
                logger.error(f"Error checking model {model.get('_id')}: {e}", exc_info=True)
    
    async def check_and_adapt_for_new_data(self, threshold: int):
        """Check for new data and trigger incremental updates. Uses MongoDB."""
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        symbols = store.mongo_db.model_versions.distinct("symbol")
        
        for symbol in symbols:
            try:
                # Get latest model from MongoDB
                latest_model = store.get_latest_model_version(symbol, None)
                
                if not latest_model:
                    continue
                
                # Check how many new data points since last training
                df = DataIngestionService.get_prices(symbol, limit=1000)
                if len(df) == 0:
                    continue
                
                latest_data_time = df.iloc[-1].get('date') or df.index[-1]
                if isinstance(latest_data_time, str):
                    from datetime import datetime
                    try:
                        latest_data_time = datetime.fromisoformat(latest_data_time.replace('Z', '+00:00'))
                    except:
                        from dateutil import parser
                        latest_data_time = parser.parse(latest_data_time)
                
                model_time = latest_model.get("trained_at")
                if isinstance(model_time, str):
                    from dateutil import parser
                    model_time = parser.parse(model_time)
                
                if latest_data_time.tzinfo is None:
                    latest_data_time = latest_data_time.replace(tzinfo=timezone.utc)
                if model_time.tzinfo is None:
                    model_time = model_time.replace(tzinfo=timezone.utc)
                
                # Count new points (simplified: check if significant time passed)
                time_diff = latest_data_time - model_time
                hours_diff = time_diff.total_seconds() / 3600
                
                # Estimate new data points (assuming hourly data)
                estimated_new_points = int(hours_diff)
                
                if estimated_new_points >= threshold:
                    logger.info(
                        f"New data detected for {symbol} (~{estimated_new_points} points), "
                        "triggering incremental update"
                    )
                    ModelService.adapt_model(symbol, new_data_window=min(estimated_new_points, 100))
                    
            except Exception as e:
                logger.error(f"Error checking new data for {symbol}: {e}", exc_info=True)
    
    async def trigger_retraining(self, symbol: str, reason: str = "manual"):
        """Manually trigger retraining for a symbol."""
        logger.info(f"Manual retraining triggered for {symbol} (reason: {reason})")
        try:
            ModelService.train_new_model(symbol, initial=False)
            return {"status": "success", "symbol": symbol, "reason": reason}
        except Exception as e:
            logger.error(f"Error in manual retraining for {symbol}: {e}", exc_info=True)
            raise


# Global instance
_retraining_service: Optional[ScheduledRetrainingService] = None


def get_retraining_service() -> ScheduledRetrainingService:
    """Get the global retraining service instance."""
    global _retraining_service
    if _retraining_service is None:
        _retraining_service = ScheduledRetrainingService()
    return _retraining_service


__all__ = ["ScheduledRetrainingService", "get_retraining_service"]

