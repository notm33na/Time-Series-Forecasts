"""
Service for loading pre-trained models from database or files.
No training happens here - only loading and prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..models.forecasting_models import (
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
)
from ..models.adaptive_forecaster import AdaptiveForecaster
from ..models.adaptive_ensemble import AdaptiveEnsemble, create_default_ensemble
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ModelLoader:
    """Service for loading pre-trained models (no training)."""

    MODEL_CLASS_MAP = {
        "ARIMA": ARIMAForecaster,
        "ExponentialSmoothing": ExponentialSmoothingForecaster,
        "LSTM": LSTMForecaster,
        "GRU": GRUForecaster,
        "Transformer": TransformerForecaster,
        "AdaptiveForecaster": AdaptiveForecaster,
    }

    @staticmethod
    def load_model_from_version(
        model_version_id: str,
    ) -> tuple[object, Dict[str, Any]]:
        """
        Load a model from a ModelVersion MongoDB record.
        
        Args:
            model_version_id: MongoDB _id (as string) of the model version to load
        
        Returns:
            (model_instance, model_version_dict)
        
        Raises:
            ValueError: If model version not found or cannot be loaded
        """
        store = UnifiedDataStore()
        model_version = store.get_model_version(model_version_id)
        
        if not model_version:
            raise ValueError(f"Model version {model_version_id} not found")
        
        # Get model type from training_info
        training_info = model_version.get("training_info", {})
        model_type = training_info.get("model_type", "AdaptiveForecaster")
        
        # Load model from artifact path
        artifact_path = Path(model_version["artifact_path"])
        
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}. "
                f"Model may need to be retrained or artifact path is incorrect."
            )
        
        # Load model based on type
        model_class = ModelLoader.MODEL_CLASS_MAP.get(model_type)
        
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            # Use load_model() to load the model
            # For AdaptiveForecaster, also support .load() method
            if hasattr(model_class, 'load_model'):
                model = model_class.load_model(artifact_path)
            elif hasattr(model_class, 'load'):
                model = model_class.load(artifact_path)
            else:
                raise ValueError(f"Model class {model_class} does not have load_model() or load() method")
            
            logger.info(f"Loaded {model_type} model from {artifact_path}")
            return model, model_version
        except Exception as e:
            logger.error(f"Failed to load model from {artifact_path}: {e}")
            raise ValueError(f"Cannot load model: {e}")

    @staticmethod
    def load_latest_model(
        symbol: str,
        model_type: str,
        horizon: Optional[int] = None,
    ) -> tuple[object, Dict[str, Any]]:
        """
        Load the latest model of a specific type for a symbol.
        Note: The horizon parameter is optional - models can predict any number of steps
        regardless of the horizon they were trained with.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model (ARIMA, LSTM, etc.)
            horizon: Optional - preferred horizon (in steps). If not provided, loads latest model.
        
        Returns:
            (model_instance, model_version_dict)
        
        Raises:
            ValueError: If no model found
        """
        store = UnifiedDataStore()
        model_version = store.get_latest_model_version(symbol, model_type, horizon)
        
        if not model_version:
            raise ValueError(
                f"No {model_type} model found for {symbol}. "
                f"Please train a model first using /api/forecast/train or training scripts."
            )
        
        if horizon is not None and model_version.get("horizon") != horizon:
            logger.info(
                f"Using model trained for horizon={model_version.get('horizon')} "
                f"to predict horizon={horizon} steps (models are flexible about step count)"
            )
        
        return ModelLoader.load_model_from_version(model_version["id"])

    @staticmethod
    def load_ensemble_models(
        symbol: str,
        horizon: int,
    ) -> dict[str, tuple[object, Dict[str, Any]]]:
        """
        Load individual models for ensemble prediction.
        Returns dict mapping model_type to (model, model_version).
        Tries database first, then falls back to filesystem.
        
        Args:
            symbol: Stock symbol
            horizon: Forecast horizon
        
        Returns:
            Dict of {model_type: (model, model_version)}
        """
        loaded_models = {}
        
        # Try to load each model type from database first
        # Note: horizon parameter is optional - models can predict any number of steps
        for model_type in ["ARIMA", "LSTM", "GRU", "Transformer", "ExponentialSmoothing"]:
            try:
                model, model_version = ModelLoader.load_latest_model(symbol, model_type, horizon=horizon)
                loaded_models[model_type] = (model, model_version)
                logger.info(f"Loaded {model_type} for ensemble from database (trained horizon={model_version.get('horizon')}, using for {horizon} steps)")
            except ValueError as e:
                logger.warning(f"Could not load {model_type} from database: {e}")
                # Try filesystem fallback
                try:
                    logger.info(f"Attempting to load {model_type} from filesystem for {symbol}...")
                    model, model_version = ModelLoader._load_model_from_filesystem(symbol, model_type, horizon)
                    loaded_models[model_type] = (model, model_version)
                    logger.info(f"✓ Loaded {model_type} for ensemble from filesystem (using for {horizon} steps)")
                except (ValueError, FileNotFoundError) as fs_error:
                    logger.warning(f"Could not load {model_type} from filesystem: {fs_error}")
                    # Continue with other models
                except Exception as fs_error:
                    logger.error(f"Unexpected error loading {model_type} from filesystem: {fs_error}", exc_info=True)
                    # Continue with other models
        
        if not loaded_models:
            raise ValueError(
                f"No models found for {symbol} in database or filesystem. "
                f"Please train at least one model first using training scripts (train_arima.py, train_lstm.py, etc.)."
            )
        
        return loaded_models
    
    @staticmethod
    def _load_model_from_filesystem(
        symbol: str,
        model_type: str,
        horizon: Optional[int] = None,
    ) -> tuple[object, Dict[str, Any]]:
        """
        Load model from filesystem when not found in database.
        This is a fallback for models trained via standalone scripts.
        """
        from ..models.forecasting_models import (
            ARIMAForecaster,
            ExponentialSmoothingForecaster,
            LSTMForecaster,
            GRUForecaster,
            TransformerForecaster,
        )
        from datetime import datetime, timezone
        
        model_type_map = {
            "ARIMA": (ARIMAForecaster, ["*.pkl"]),
            "ExponentialSmoothing": (ExponentialSmoothingForecaster, ["*.pkl"]),
            "LSTM": (LSTMForecaster, ["*.weights.h5", "*.h5"]),
            "GRU": (GRUForecaster, ["*.weights.h5", "*.h5"]),
            "Transformer": (TransformerForecaster, ["*.weights.h5", "*.h5"]),
        }
        
        if model_type not in model_type_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class, patterns = model_type_map[model_type]
        
        # Check multiple possible locations
        # model_loader.py is in backend/services/, so:
        # __file__ = backend/services/model_loader.py
        # parent = backend/services/
        # parent.parent = backend/
        # So backend/artifacts = parent.parent / "artifacts"
        possible_dirs = [
            settings.models_dir,
            Path(__file__).parent.parent / "artifacts",  # backend/artifacts
            Path(__file__).parent.parent.parent / "backend" / "artifacts",  # project/backend/artifacts
            Path(__file__).parent.parent.parent / "artifacts",  # project/artifacts
            Path(__file__).parent.parent.parent / "notebooks",  # project/notebooks (where training scripts might save)
        ]
        
        models_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                models_dir = dir_path
                break
        
        if not models_dir:
            logger.error(f"Models directory not found. Searched in: {[str(d) for d in possible_dirs]}")
            raise FileNotFoundError(
                f"Models directory not found in any of: {[str(d) for d in possible_dirs]}. "
                f"Please ensure models are trained and saved to one of these locations."
            )
        
        # For neural models (LSTM, GRU, Transformer), search for .pkl files first
        if model_type in ["LSTM", "GRU", "Transformer"]:
            # Try to find .pkl metadata files (not .weights.pkl)
            pkl_pattern = f"{symbol}_{model_type.lower()}_*.pkl"
            pkl_files = list(models_dir.glob(pkl_pattern))
            # Filter out .weights.pkl files - we want the base .pkl metadata file
            pkl_files = [f for f in pkl_files if '.weights.pkl' not in f.name]
            
            if not pkl_files:
                # If no regular .pkl found, try .weights.pkl as fallback
                weights_pkl_pattern = f"{symbol}_{model_type.lower()}_*.weights.pkl"
                weights_pkl_files = list(models_dir.glob(weights_pkl_pattern))
                if weights_pkl_files:
                    # Use .weights.pkl and try to find corresponding .weights.h5
                    weights_pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    weights_pkl_path = weights_pkl_files[0]
                    # Extract base name (remove .weights.pkl)
                    base_name = weights_pkl_path.stem.replace('.weights', '')
                    h5_path = weights_pkl_path.parent / f"{base_name}.weights.h5"
                    pkl_path = weights_pkl_path  # Use the .weights.pkl as metadata
                else:
                    raise FileNotFoundError(f"No {model_type} model found for {symbol} in {models_dir}")
            else:
                pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                pkl_path = pkl_files[0]
                base_name = pkl_path.stem
                h5_path = pkl_path.parent / f"{base_name}.weights.h5"
            
            # Try to load with .h5 file (load_model will find the .pkl)
            if h5_path.exists():
                model = model_class.load_model(h5_path)
                artifact_path = h5_path
            else:
                # If .h5 doesn't exist, try loading from .pkl directly (some models might be saved differently)
                try:
                    model = model_class.load_model(pkl_path)
                    artifact_path = pkl_path
                except Exception as e:
                    raise FileNotFoundError(
                        f"Could not find weights file for {model_type}: {h5_path}. "
                        f"Also tried loading from {pkl_path} but failed: {e}"
                    )
        else:
            # For non-neural models, search for pattern
            search_pattern = f"{symbol}_{model_type.lower()}_*{patterns[0]}"
            matching_files = list(models_dir.glob(search_pattern))
            
            if not matching_files:
                search_pattern = f"{symbol}_{model_type}_*{patterns[0]}"
                matching_files = list(models_dir.glob(search_pattern))
            
            if not matching_files:
                raise FileNotFoundError(f"No {model_type} model found for {symbol} in {models_dir}")
            
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = matching_files[0]
            model = model_class.load_model(model_path)
            artifact_path = model_path
        
        # Create a dummy model_version dict for compatibility
        return model, {
            "id": 0,
            "symbol": symbol,
            "horizon": horizon or 24,
            "version_tag": f"{symbol}_{model_type}_filesystem",
            "trained_at": datetime.now(timezone.utc),
            "artifact_path": str(artifact_path),
            "training_info": {"model_type": model_type}
        }

    @staticmethod
    def create_ensemble_from_loaded_models(
        symbol: str,
        horizon: int,
        loaded_models: dict[str, tuple[object, Dict[str, Any]]],
    ) -> AdaptiveEnsemble:
        """
        Create an ensemble from pre-loaded models.
        
        Args:
            symbol: Stock symbol
            horizon: Forecast horizon
            loaded_models: Dict of {model_type: (model, model_version)}
        
        Returns:
            AdaptiveEnsemble with loaded models
        """
        ensemble = AdaptiveEnsemble(symbol, horizon)
        
        for model_type, (model, model_version) in loaded_models.items():
            # Models loaded from files are already fitted
            # Mark them as fitted if the attribute exists
            if hasattr(model, 'is_fitted'):
                model.is_fitted = True
            elif hasattr(model, '_is_fitted'):
                model._is_fitted = True
            else:
                # If model doesn't have is_fitted attribute, assume it's fitted if it was loaded
                # This is a fallback for models that might not have this attribute
                logger.warning(f"Model {model_type} does not have is_fitted attribute, assuming it's fitted")
                try:
                    setattr(model, 'is_fitted', True)
                except:
                    pass
            
            # Verify model is actually fitted before adding
            is_fitted = getattr(model, 'is_fitted', False) or getattr(model, '_is_fitted', False)
            if not is_fitted:
                logger.warning(f"Model {model_type} is not marked as fitted, but adding anyway (may fail during prediction)")
            
            # Add to ensemble
            ensemble.add_forecaster(model, model_type)
            logger.info(f"Added {model_type} to ensemble (is_fitted={is_fitted})")
        
        return ensemble

