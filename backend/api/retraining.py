"""
Scheduled retraining and adaptive learning API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from ..services.scheduled_retraining import get_retraining_service
from ..services.adaptive_learning_service import AdaptiveLearningService
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/retraining", tags=["retraining"])


class RetrainRequest(BaseModel):
    symbol: str
    reason: str = "manual"


@router.post("/trigger")
async def trigger_retraining(request: RetrainRequest):
    """Manually trigger retraining for a symbol."""
    try:
        service = get_retraining_service()
        result = await service.trigger_retraining(request.symbol, request.reason)
        return result
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_scheduled_retraining():
    """Start the scheduled retraining service."""
    try:
        service = get_retraining_service()
        await service.start()
        return {"status": "success", "message": "Scheduled retraining service started"}
    except Exception as e:
        logger.error(f"Error starting retraining service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_scheduled_retraining():
    """Stop the scheduled retraining service."""
    try:
        service = get_retraining_service()
        await service.stop()
        return {"status": "success", "message": "Scheduled retraining service stopped"}
    except Exception as e:
        logger.error(f"Error stopping retraining service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Adaptive Learning Endpoints

class OnlineUpdateRequest(BaseModel):
    symbol: str
    model_type: str = "LSTM"
    learning_rate: float = 0.01
    batch_size: int = 32


class RollingWindowRequest(BaseModel):
    symbol: str
    model_type: str = "LSTM"
    window_size: int = 200
    horizon: Optional[int] = None


class FineTuneRequest(BaseModel):
    symbol: str
    model_type: Literal["LSTM", "GRU", "Transformer"] = "LSTM"
    new_data_window: int = 50
    epochs: int = 5
    learning_rate: float = 0.0001
    freeze_base: bool = False


class EnsembleUpdateRequest(BaseModel):
    symbol: str
    horizon: int = 24
    evaluation_window: int = 50


class ModelComparisonRequest(BaseModel):
    symbol: str
    model_type: str
    limit: int = 10


@router.post("/online-update")
async def online_update(request: OnlineUpdateRequest):
    """Perform online/incremental learning update on an existing model."""
    try:
        learning_service = AdaptiveLearningService()
        
        # Get new data (last N points)
        from ..services.data_ingestion import DataIngestionService
        data_service = DataIngestionService()
        df = data_service.get_prices(request.symbol, limit=100)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient new data for online update")
        
        model, model_version = learning_service.online_update_model(
            symbol=request.symbol,
            model_type=request.model_type,
            new_data=df.tail(10),  # Use last 10 points
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_type": request.model_type,
            "model_version_id": model_version.get("id"),
            "version_tag": model_version.get("version_tag"),
            "trained_at": str(model_version.get("trained_at")),
        }
    except Exception as e:
        logger.error(f"Error in online update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rolling-window")
async def rolling_window_retrain(request: RollingWindowRequest):
    """Retrain model using a rolling window of recent data."""
    try:
        learning_service = AdaptiveLearningService()
        
        model, model_version = learning_service.rolling_window_retrain(
            symbol=request.symbol,
            model_type=request.model_type,
            window_size=request.window_size,
            horizon=request.horizon,
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_type": request.model_type,
            "model_version_id": model_version.get("id"),
            "version_tag": model_version.get("version_tag"),
            "trained_at": str(model_version.get("trained_at")),
            "window_size": request.window_size,
        }
    except Exception as e:
        logger.error(f"Error in rolling-window retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune")
async def fine_tune(request: FineTuneRequest):
    """Fine-tune a neural model with recent data."""
    try:
        learning_service = AdaptiveLearningService()
        
        model, model_version = learning_service.fine_tune_neural_model(
            symbol=request.symbol,
            model_type=request.model_type,
            new_data_window=request.new_data_window,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            freeze_base=request.freeze_base,
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_type": request.model_type,
            "model_version_id": model_version.get("id"),
            "version_tag": model_version.get("version_tag"),
            "trained_at": str(model_version.get("trained_at")),
            "epochs": request.epochs,
        }
    except Exception as e:
        logger.error(f"Error in fine-tuning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-ensemble")
async def update_ensemble(request: EnsembleUpdateRequest):
    """Update ensemble weights based on recent model performance."""
    try:
        learning_service = AdaptiveLearningService()
        
        ensemble, performance = learning_service.update_ensemble_weights(
            symbol=request.symbol,
            horizon=request.horizon,
            evaluation_window=request.evaluation_window,
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "performance": performance,
            "weights": {name: float(w) for name, w in zip(ensemble.model_names, ensemble.weights)},
        }
    except Exception as e:
        logger.error(f"Error updating ensemble: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-versions")
async def compare_versions(request: ModelComparisonRequest):
    """Compare performance across multiple model versions."""
    try:
        learning_service = AdaptiveLearningService()
        
        comparison = learning_service.get_model_version_comparison(
            symbol=request.symbol,
            model_type=request.model_type,
            limit=request.limit,
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "model_type": request.model_type,
            "comparison": comparison,
        }
    except Exception as e:
        logger.error(f"Error comparing versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

