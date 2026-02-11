"""
Adaptive forecasting models with incremental learning capabilities.
"""

from __future__ import annotations

import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from ..config import get_settings
from ..utils.logging import get_logger
from ..utils.model_downloader import ensure_model_file, parse_model_url
from ..utils.model_registry import register_model_version

logger = get_logger(__name__)
settings = get_settings()


class AdaptiveForecaster:
    """
    Adaptive forecaster using incremental SGD for online learning.
    Model must be loaded using load_model() - training is done in notebooks.
    """

    def __init__(
        self,
        symbol: str,
        lag_window: int = 24,
        horizon: int = 1,
        learning_rate: str = "adaptive",
        random_state: int = 42,
        model=None,
        scaler=None,
    ):
        self.symbol = symbol
        self.lag_window = lag_window
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = model or SGDRegressor(
            learning_rate=learning_rate,
            random_state=random_state,
            warm_start=True,
            max_iter=1000,
            tol=1e-3,
        )
        self.scaler = scaler or StandardScaler()
        self.is_fitted = model is not None
        self.training_count = 0
        self._model_version = None  # Will be set by load_model() if registered
    
    def get_model_version(self):
        """Get the ModelVersion database record for this model."""
        return self._model_version

    @classmethod
    def load_model(
        cls,
        model_path: Path | str,
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        gdrive_id: Optional[str] = None,
        s3_url: Optional[str] = None,
        remote_url: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> AdaptiveForecaster:
        """
        Load Adaptive Forecaster model from .pkl file.
        
        If local file doesn't exist, will attempt to download from remote sources.
        See ARIMAForecaster.load_model() for parameter details.
        """
        if isinstance(model_path, str):
            parsed = parse_model_url(model_path)
            if parsed["local_path"]:
                model_path = Path(parsed["local_path"])
            else:
                hf_repo = parsed["hf_repo"] or hf_repo
                hf_filename = parsed["hf_filename"] or hf_filename
                gdrive_id = parsed["gdrive_id"] or gdrive_id
                s3_url = parsed["s3_url"] or s3_url
                remote_url = parsed["remote_url"] or remote_url
                if hf_filename:
                    model_path = settings.models_dir / hf_filename
                elif remote_url:
                    model_path = settings.models_dir / Path(remote_url).name
                else:
                    model_path = settings.models_dir / "adaptive_forecaster.pkl"
        
        model_path = Path(model_path)
        
        try:
            model_path = ensure_model_file(
                local_path=model_path,
                remote_url=remote_url,
                hf_repo=hf_repo,
                hf_filename=hf_filename,
                gdrive_id=gdrive_id,
                s3_url=s3_url,
                hf_token=hf_token,
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to load Adaptive Forecaster model: {e}")
            raise
        
        state = joblib.load(model_path)
        forecaster = cls(
            symbol=state["symbol"],
            lag_window=state["lag_window"],
            horizon=state["horizon"],
            learning_rate=state.get("learning_rate", "adaptive"),
            random_state=state.get("random_state", 42),
            model=state["model"],
            scaler=state["scaler"]
        )
        forecaster.is_fitted = True
        forecaster.training_count = state.get("training_count", 0)
        
        # Determine source and register
        source = "local"
        source_url = None
        version_id = None
        
        if hf_repo and hf_filename:
            source = "huggingface"
            source_url = f"{hf_repo}/{hf_filename}"
            version_id = hf_filename
        elif gdrive_id:
            source = "gdrive"
            source_url = f"gdrive://{gdrive_id}"
            version_id = gdrive_id
        elif s3_url:
            source = "s3"
            source_url = s3_url
            version_id = Path(s3_url).name
        elif remote_url:
            source = "url"
            source_url = remote_url
            version_id = Path(remote_url).name
        
        # Register model version
        try:
            model_version = register_model_version(
                symbol=state["symbol"],
                model_type="AdaptiveForecaster",
                artifact_path=model_path,
                source=source,
                source_url=source_url,
                version_id=version_id,
                horizon=state.get("horizon", 1),
            )
            forecaster._model_version = model_version
            logger.info(f"Registered AdaptiveForecaster model version {model_version.id} from {source}")
        except Exception as e:
            logger.warning(f"Could not register model version: {e}")
            forecaster._model_version = None
        
        logger.info(f"Loaded Adaptive Forecaster from {model_path}")
        return forecaster

    def preprocess_input(self, last_prices: pd.Series) -> np.ndarray:
        """Preprocess input data for prediction."""
        if len(last_prices) < self.lag_window:
            raise ValueError(f"Need at least {self.lag_window} historical points")
        
        current_window = last_prices.iloc[-self.lag_window :].values.copy()
        X = current_window.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict(self, last_prices: pd.Series, steps: int = 1) -> np.ndarray:
        """Generate forecast for next steps."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() to load a trained model.")

        if len(last_prices) < self.lag_window:
            raise ValueError(f"Need at least {self.lag_window} historical points")

        predictions = []
        current_window = last_prices.iloc[-self.lag_window :].values.copy()

        for _ in range(steps):
            X_scaled = self.preprocess_input(pd.Series(current_window))
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)

            # Update window for next step
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred

        return np.array(predictions)

    def evaluate(self, prices: pd.Series) -> dict[str, float]:
        """Evaluate model on test data (requires creating features - typically done in notebooks)."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() to load a trained model.")
        
        # Note: Feature creation should be done in training/evaluation scripts
        # This is kept for backward compatibility but may not work without proper feature creation
        raise NotImplementedError(
            "Evaluation should be done in training notebooks. "
            "This method requires feature creation which is part of training logic."
        )

    @classmethod
    def load(cls, artifact_path: Path) -> AdaptiveForecaster:
        """Load model from artifact (alias for load_model for backward compatibility)."""
        return cls.load_model(artifact_path)


class EnsembleForecaster:
    """
    Adaptive ensemble that reweights models based on recent errors.
    """

    def __init__(self, symbol: str, horizon: int = 1):
        self.symbol = symbol
        self.horizon = horizon
        self.forecasters: list[AdaptiveForecaster] = []
        self.weights: np.ndarray = np.array([])
        self.recent_errors: list[list[float]] = []

    def add_forecaster(self, forecaster: AdaptiveForecaster) -> EnsembleForecaster:
        """Add a forecaster to the ensemble."""
        self.forecasters.append(forecaster)
        self.recent_errors.append([])
        self._update_weights()
        return self

    def _update_weights(self) -> None:
        """Update weights based on recent errors (inverse error weighting)."""
        if not self.recent_errors or not any(self.recent_errors):
            self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
            return

        errors = [np.mean(errs) if errs else 1.0 for errs in self.recent_errors]
        inv_errors = 1.0 / (np.array(errors) + 1e-6)
        self.weights = inv_errors / inv_errors.sum()

    def predict(self, last_prices: pd.Series, steps: int = 1) -> np.ndarray:
        """Weighted ensemble prediction."""
        if not self.forecasters:
            raise ValueError("No forecasters in ensemble")

        predictions = np.array([f.predict(last_prices, steps) for f in self.forecasters])
        weighted = np.average(predictions, axis=0, weights=self.weights)
        return weighted

    def update_errors(self, actual: float, predicted: float) -> None:
        """Update recent errors for adaptive reweighting."""
        error = abs(actual - predicted)
        for i, forecaster in enumerate(self.forecasters):
            self.recent_errors[i].append(error)
            if len(self.recent_errors[i]) > 50:  # Keep last 50 errors
                self.recent_errors[i].pop(0)
        self._update_weights()


__all__ = ["AdaptiveForecaster", "EnsembleForecaster"]

