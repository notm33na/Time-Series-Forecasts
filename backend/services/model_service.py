"""
Model training and adaptation service.
Uses MongoDB exclusively for all data storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..models.adaptive_forecaster import AdaptiveForecaster, EnsembleForecaster
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ModelService:
    """Service for model training, adaptation, and versioning."""

    @staticmethod
    def train_new_model(
        symbol: str,
        horizon: int = None,
        lag_window: int = None,
        initial: bool = True,
    ) -> tuple[AdaptiveForecaster, Dict[str, Any]]:
        """
        Train a new model and create a version record.
        Returns (forecaster, model_version).
        """
        horizon = horizon or settings.forecast_horizon
        lag_window = lag_window or settings.lag_window

        logger.info(f"Training new model for {symbol} (horizon={horizon}, lag={lag_window})")

        # Get historical data
        from .data_ingestion import DataIngestionService

        df = DataIngestionService.get_prices(symbol)
        if len(df) < settings.min_history:
            raise ValueError(
                f"Insufficient data: need at least {settings.min_history} points, got {len(df)}"
            )

        prices = df["Close"]

        # Train model
        forecaster = AdaptiveForecaster(
            symbol=symbol,
            lag_window=lag_window,
            horizon=horizon,
        )
        forecaster.fit(prices, initial=initial)

        # Create version tag with microseconds for uniqueness
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{timestamp}"

        # Save artifact (save() will add symbol prefix, so pass just timestamp)
        artifact_path = forecaster.save(timestamp)

        # Store version in MongoDB
        # Ensure artifact_path is absolute for reliable loading
        artifact_path_abs = Path(artifact_path).resolve()
        
        store = UnifiedDataStore()
        model_version_id = store.save_model_version(
            symbol=symbol,
            model_type="AdaptiveForecaster",
            horizon=horizon,
            version_tag=version_tag,
            artifact_path=str(artifact_path_abs),
            source="local",  # Models trained via this service are local
            source_url=None,
            version_id=None,
            training_info={
                "model_type": "AdaptiveForecaster",
                "lag_window": lag_window,
                "training_samples": len(prices),
                "initial": initial,
            },
        )
        
        model_version = store.get_model_version(model_version_id)

        logger.info(f"Model trained and versioned: {version_tag}")
        return forecaster, model_version

    @staticmethod
    def adapt_model(
        symbol: str,
        model_version_id: Optional[str] = None,  # MongoDB _id as string
        new_data_window: int = 50,
    ) -> tuple[AdaptiveForecaster, Dict[str, Any]]:
        """
        Adapt existing model with new data using incremental learning.
        Returns (updated_forecaster, new_model_version).
        """
        logger.info(f"Adapting model for {symbol} (version_id={model_version_id})")

        # Load existing model or get latest from MongoDB
        store = UnifiedDataStore()
        if model_version_id:
            model_version = store.get_model_version(model_version_id)
        else:
            model_version = store.get_latest_model_version(symbol, "AdaptiveForecaster")

        if not model_version:
            logger.info("No existing model found, training new one")
            return ModelService.train_new_model(symbol, initial=True)

        # Load forecaster
        forecaster = AdaptiveForecaster.load(Path(model_version["artifact_path"]))

        # Get recent data
        from .data_ingestion import DataIngestionService

        df = DataIngestionService.get_prices(symbol, limit=new_data_window)
        if len(df) < forecaster.lag_window + forecaster.horizon:
            logger.warning("Insufficient new data for adaptation, skipping")
            return forecaster, model_version

        prices = df["Close"]

        # Incremental update
        forecaster.partial_fit(prices)

        # Create new version with microseconds for uniqueness
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{timestamp}_adapted"
        # Save artifact (save() will add symbol prefix, so pass just timestamp part)
        artifact_path = forecaster.save(f"{timestamp}_adapted")
        
        # Ensure artifact_path is absolute for reliable loading
        artifact_path_abs = Path(artifact_path).resolve()

        new_version_id = store.save_model_version(
            symbol=symbol,
            model_type="AdaptiveForecaster",
            horizon=forecaster.horizon,
            version_tag=version_tag,
            artifact_path=str(artifact_path_abs),
            source="local",
            source_url=None,
            version_id=None,
            training_info={
                "model_type": "AdaptiveForecaster",
                "lag_window": forecaster.lag_window,
                "adaptation_samples": len(prices),
                "previous_version": model_version.get("version_tag"),
                "total_trainings": forecaster.training_count,
            },
        )
        
        new_version = store.get_model_version(new_version_id)

        logger.info(f"Model adapted and new version created: {version_tag}")
        return forecaster, new_version

    @staticmethod
    def get_latest_model(symbol: str) -> Optional[tuple[AdaptiveForecaster, Dict[str, Any]]]:
        """Get the latest model version for a symbol from MongoDB."""
        store = UnifiedDataStore()
        model_version = store.get_latest_model_version(symbol, "AdaptiveForecaster")

        if not model_version:
            return None

        forecaster = AdaptiveForecaster.load(Path(model_version["artifact_path"]))
        return forecaster, model_version

    @staticmethod
    def predict(
        symbol: str,
        horizon: int = None,
        model_version_id: Optional[str] = None,  # MongoDB _id as string
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate forecast using latest or specified model.
        Only loads pre-trained models - no training happens here.
        Returns (predictions, model_version).
        
        Raises:
            ValueError: If no model found (must train first)
        """
        from .model_loader import ModelLoader
        
        horizon = horizon or settings.forecast_horizon

        # Load model (no training)
        if model_version_id:
            forecaster, model_version = ModelLoader.load_model_from_version(model_version_id)
        else:
            # Try to load latest AdaptiveForecaster
            try:
                forecaster, model_version = ModelLoader.load_latest_model(symbol, "AdaptiveForecaster")
            except ValueError:
                # Fallback: try to get from get_latest_model (for backward compatibility)
                result = ModelService.get_latest_model(symbol)
                if result:
                    forecaster, model_version = result
                else:
                    raise ValueError(
                        f"No model found for {symbol}. "
                        f"Please train a model first using /api/models/train or training scripts."
                    )

        # Get recent prices for prediction
        from .data_ingestion import DataIngestionService

        df = DataIngestionService.get_prices(symbol, limit=forecaster.lag_window + 10)
        if len(df) < forecaster.lag_window:
            raise ValueError(f"Insufficient data for prediction: need {forecaster.lag_window} points")

        prices = df["Close"]
        
        # Preprocess and predict (no training)
        predictions = forecaster.predict(prices, steps=horizon)

        return predictions, model_version

