"""
Utility for registering model versions in MongoDB when loading from remote sources.
Uses MongoDB exclusively for all data storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import joblib

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def register_model_version(
    symbol: str,
    model_type: str,
    artifact_path: Path,
    source: str,
    source_url: Optional[str] = None,
    version_id: Optional[str] = None,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Register or update a model version in MongoDB.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model (ARIMA, LSTM, etc.)
        artifact_path: Path to model file
        source: Source of model ("local", "huggingface", "gdrive", "s3", "url")
        source_url: Full URL or repo path
        version_id: Version identifier from source
        horizon: Forecast horizon
    
    Returns:
        Model version document from MongoDB
    """
    artifact_path = Path(artifact_path)
    
    # Try to extract training date from model metadata
    training_date = None
    try:
        if artifact_path.suffix == '.pkl':
            # Try to load metadata
            metadata = joblib.load(artifact_path)
            if isinstance(metadata, dict):
                # Check for training date in metadata
                if 'train_size' in metadata or 'test_size' in metadata:
                    # Model was saved by training script, use current time as proxy
                    # In practice, training_date should be in metadata
                    training_date = datetime.now(timezone.utc)
                elif 'trained_at' in metadata:
                    training_date = metadata['trained_at']
        elif artifact_path.suffix == '.h5':
            # For neural models, check for corresponding .pkl metadata file
            metadata_path = artifact_path.with_suffix('.pkl')
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                if isinstance(metadata, dict) and 'train_size' in metadata:
                    training_date = datetime.now(timezone.utc)
    except Exception as e:
        logger.warning(f"Could not extract training date from {artifact_path}: {e}")
    
    if training_date is None:
        training_date = datetime.now(timezone.utc)
    
    # Create version tag
    timestamp = training_date.strftime('%Y%m%d_%H%M%S')
    version_tag = f"{symbol}_{model_type}_{source}_{timestamp}"
    
    store = UnifiedDataStore()
    artifact_path_str = str(artifact_path)
    
    # Check if version already exists in MongoDB
    existing = store.mongo_db.model_versions.find_one({
        "symbol": symbol,
        "artifact_path": artifact_path_str
    })
    
    if existing:
        # Update existing version with source info
        store.mongo_db.model_versions.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "source": source,
                    "source_url": source_url,
                    "version_id": version_id,
                    "trained_at": training_date,
                }
            }
        )
        existing["id"] = str(existing["_id"])
        del existing["_id"]
        logger.info(f"Updated model version {existing['id']} with source metadata")
        return existing
    
    # Create new version in MongoDB
    model_version_id = store.save_model_version(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
        version_tag=version_tag,
        artifact_path=artifact_path_str,
        source=source,
        source_url=source_url,
        version_id=version_id,
        training_info={
            "model_type": model_type,
            "source": source,
            "source_url": source_url,
            "version_id": version_id,
        },
    )
    
    model_version = store.get_model_version(model_version_id)
    logger.info(f"Registered model version {model_version_id} from {source}: {version_tag}")
    return model_version


def get_or_create_model_version_from_path(
    symbol: str,
    model_type: str,
    artifact_path: Path,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Get or create a ModelVersion for a local model file in MongoDB.
    
    Args:
        symbol: Stock symbol
        model_type: Type of model
        artifact_path: Path to model file
        horizon: Forecast horizon
    
    Returns:
        Model version document from MongoDB
    """
    return register_model_version(
        symbol=symbol,
        model_type=model_type,
        artifact_path=artifact_path,
        source="local",
        horizon=horizon,
    )

