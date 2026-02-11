"""
Enhanced model service supporting multiple model types, instruments, and horizons.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..models.adaptive_ensemble import AdaptiveEnsemble, create_default_ensemble
from ..models.forecasting_models import (
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    MovingAverageForecaster,
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
)
# ModelVersion is now stored in MongoDB, not SQLite
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MultiModelService:
    """Service for managing multiple forecasting models with different types and horizons."""
    
    # Horizon mappings (in hours)
    HORIZON_MAP = {
        "1h": 1,
        "3h": 3,
        "24h": 24,
        "72h": 72,
    }
    
    MODEL_TYPES = Literal["ARIMA", "VAR", "ExponentialSmoothing", "MovingAverage", "LSTM", "GRU", "Transformer", "Ensemble"]
    
    @staticmethod
    def train_model(
        symbol: str,
        model_type: MODEL_TYPES = "Ensemble",
        horizon: str = "24h",
        lag_window: int = None,
    ) -> tuple[object, dict]:
        """
        Train a specific model type for a symbol and horizon.
        Returns (model, model_version).
        """
        horizon_steps = MultiModelService.HORIZON_MAP.get(horizon, 24)
        lag_window = lag_window or settings.lag_window
        
        logger.info(f"Training {model_type} for {symbol} (horizon: {horizon})")
        
        # Get data
        from .data_ingestion import DataIngestionService
        
        df = DataIngestionService.get_prices(symbol)
        if len(df) < settings.min_history:
            raise ValueError(f"Insufficient data: need {settings.min_history} points")
        
        prices = df["Close"]
        
        # Create and train model
        if model_type == "Ensemble":
            model = create_default_ensemble(symbol, horizon_steps)
            model.fit(prices)
        elif model_type == "ARIMA":
            model = ARIMAForecaster(symbol, horizon_steps)
            model.fit(prices)
        elif model_type == "ExponentialSmoothing":
            model = ExponentialSmoothingForecaster(symbol, horizon_steps)
            model.fit(prices)
        elif model_type == "MovingAverage":
            model = MovingAverageForecaster(symbol, horizon_steps, window=20, ma_type="SMA")
            model.fit(prices)
        elif model_type == "LSTM":
            model = LSTMForecaster(symbol, horizon_steps, lag_window=1)  # lag_window=1 for i221500_A02_nlp style
            model.fit(df)  # Pass DataFrame, not Series, for multi-feature input
        elif model_type == "GRU":
            model = GRUForecaster(symbol, horizon_steps, lag_window=lag_window)
            model.fit(prices)
        elif model_type == "Transformer":
            model = TransformerForecaster(symbol, horizon_steps, lag_window=lag_window)
            model.fit(prices)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create version tag
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{model_type}_{horizon}_{timestamp}"
        
        # Save artifact
        if hasattr(model, 'save'):
            artifact_path = model.save(timestamp)
        else:
            # For ensemble, save each component or use a different approach
            artifact_path = settings.models_dir / f"{symbol}_{model_type}_{timestamp}.joblib"
            import joblib
            joblib.dump(model, artifact_path)
        
        # Store version in MongoDB
        store = UnifiedDataStore()
        model_version_id = store.save_model_version(
                symbol=symbol,
            model_type=model_type,
                horizon=horizon_steps,
                version_tag=version_tag,
                artifact_path=str(artifact_path),
                source="local",  # Models trained via this service are local
                source_url=None,
                version_id=None,
                training_info={
                    "model_type": model_type,
                    "horizon": horizon,
                    "lag_window": lag_window,
                    "training_samples": len(prices),
                },
            )
        
        # Get the saved model version
        model_version = store.get_model_version(model_version_id)
        
        logger.info(f"Model trained: {version_tag}")
        return model, model_version
    
    @staticmethod
    def predict(
        symbol: str,
        model_type: MODEL_TYPES = "Ensemble",
        horizon: str = "24h",
        model_version_id: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate predictions using specified model type.
        Only loads pre-trained models - no training happens here.
        Returns (predictions, model_version).
        
        Raises:
            ValueError: If no model found (must train first)
        """
        from .model_loader import ModelLoader
        
        horizon_steps = MultiModelService.HORIZON_MAP.get(horizon, 24)
        
        # Load model (no training)
        if model_version_id:
            model, model_version = ModelLoader.load_model_from_version(str(model_version_id))
        else:
            # Load latest model of this type
            model, model_version = ModelLoader.load_latest_model(symbol, model_type)
        
        # Get recent data for prediction
        from .data_ingestion import DataIngestionService
        
        df = DataIngestionService.get_prices(symbol, limit=200)
        if len(df) < 60:  # Minimum for neural models
            raise ValueError("Insufficient data for prediction")
        
        # For LSTM, pass DataFrame; for others, pass Series
        if model_type == "LSTM":
            # Get last row for prediction (LSTM needs DataFrame with features)
            last_row = df.iloc[[-1]]
            predictions = model.predict(last_row, steps=horizon_steps)
        else:
            prices = df["Close"]
            predictions = model.predict(prices, steps=horizon_steps)
        
        return predictions, model_version
    
    @staticmethod
    def fine_tune_neural_model(
        symbol: str,
        model_type: Literal["LSTM", "GRU", "Transformer"],
        horizon: str = "24h",
        new_data_window: int = 50,
    ) -> tuple[object, dict]:
        """
        Fine-tune a neural model with rolling window of recent data.
        """
        horizon_steps = MultiModelService.HORIZON_MAP.get(horizon, 24)
        
        logger.info(f"Fine-tuning {model_type} for {symbol}")
        
        # Get recent data
        from .data_ingestion import DataIngestionService
        
        df = DataIngestionService.get_prices(symbol, limit=new_data_window + 100)
        if len(df) < new_data_window + 60:
            raise ValueError("Insufficient data for fine-tuning")
        
        # Create and train model on recent window
        if model_type == "LSTM":
            model = LSTMForecaster(symbol, horizon_steps, lag_window=1)  # lag_window=1 for i221500_A02_nlp style
            # For LSTM, use DataFrame with all features
            recent_data = df.iloc[-new_data_window:]
            model.fit(recent_data)
        elif model_type == "GRU":
            model = GRUForecaster(symbol, horizon_steps)
            prices = df["Close"]
            recent_prices = prices.iloc[-new_data_window:]
            model.fit(recent_prices)
        elif model_type == "Transformer":
            model = TransformerForecaster(symbol, horizon_steps)
            prices = df["Close"]
            recent_prices = prices.iloc[-new_data_window:]
            model.fit(recent_prices)
        else:
            raise ValueError(f"Fine-tuning only supported for LSTM, GRU, Transformer")
        
        # Create version
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{model_type}_{horizon}_{timestamp}_finetuned"
        
        artifact_path = settings.models_dir / f"{symbol}_{model_type}_{timestamp}_finetuned.joblib"
        import joblib
        joblib.dump(model, artifact_path)
        
        # Store version in MongoDB
        store = UnifiedDataStore()
        model_version_id = store.save_model_version(
                symbol=symbol,
            model_type=model_type,
                horizon=horizon_steps,
                version_tag=version_tag,
                artifact_path=str(artifact_path),
            source="local",
                training_info={
                    "model_type": model_type,
                    "horizon": horizon,
                    "fine_tuned": True,
                    "training_samples": len(recent_prices),
                },
            )
        
        # Get the saved model version
        model_version = store.get_model_version(model_version_id)
        
        logger.info(f"Model fine-tuned: {version_tag}")
        return model, model_version

