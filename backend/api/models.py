"""
Model training and prediction API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.model_service import ModelService
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


class TrainRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    lag_window: Optional[int] = None


class PredictRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    model_version_id: Optional[int] = None


class AdaptRequest(BaseModel):
    symbol: str
    model_version_id: Optional[int] = None
    new_data_window: int = 50


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a new forecasting model."""
    try:
        forecaster, model_version = ModelService.train_new_model(
            request.symbol,
            horizon=request.horizon,
            lag_window=request.lag_window,
        )
        return {
            "symbol": request.symbol,
            "model_version_id": model_version.id,
            "version_tag": model_version.version_tag,
            "trained_at": model_version.trained_at.isoformat(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapt")
async def adapt_model(request: AdaptRequest):
    """Adapt existing model with new data."""
    try:
        forecaster, model_version = ModelService.adapt_model(
            request.symbol,
            model_version_id=request.model_version_id,
            new_data_window=request.new_data_window,
        )
        return {
            "symbol": request.symbol,
            "model_version_id": model_version.id,
            "version_tag": model_version.version_tag,
            "trained_at": model_version.trained_at.isoformat(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error adapting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(request: PredictRequest):
    """Generate forecast predictions."""
    try:
        predictions, model_version = ModelService.predict(
            request.symbol,
            horizon=request.horizon,
            model_version_id=request.model_version_id,
        )
        return {
            "symbol": request.symbol,
            "predictions": predictions.tolist(),
            "model_version_id": model_version.id,
            "version_tag": model_version.version_tag,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

