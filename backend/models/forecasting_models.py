"""
Multiple forecasting models: ARIMA, VAR, Exponential Smoothing, LSTM, GRU, Transformer.
"""

from __future__ import annotations

import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from ..config import get_settings
from ..utils.logging import get_logger
from ..utils.model_downloader import ensure_model_file, parse_model_url
from ..utils.model_registry import register_model_version

logger = get_logger(__name__)
settings = get_settings()


def _determine_source_and_register(
    model_path: Path,
    forecaster_instance: BaseForecaster,
    model_type: str,
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None,
    gdrive_id: Optional[str] = None,
    s3_url: Optional[str] = None,
    remote_url: Optional[str] = None,
) -> None:
    """Helper to determine source and register model version."""
    source = "local"
    source_url = None
    version_id = None
    
    # Determine source from parameters
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
    
    # Register model version in database
    try:
        model_version = register_model_version(
            symbol=forecaster_instance.symbol,
            model_type=model_type,
            artifact_path=model_path,
            source=source,
            source_url=source_url,
            version_id=version_id,
            horizon=forecaster_instance.horizon,
        )
        forecaster_instance._model_version = model_version
        logger.info(f"Registered {model_type} model version {model_version.id} from {source}")
    except Exception as e:
        logger.warning(f"Could not register model version: {e}")
        forecaster_instance._model_version = None


class BaseForecaster:
    """Base class for all forecasters."""
    
    def __init__(self, symbol: str, horizon: int = 1):
        self.symbol = symbol
        self.horizon = horizon
        self.is_fitted = False
        self._model_version = None  # Will be set by load_model() if registered
    
    @classmethod
    def load_model(cls, model_path: Path) -> BaseForecaster:
        """Load a trained model from file."""
        raise NotImplementedError
    
    def preprocess_input(self, data: pd.Series) -> np.ndarray:
        """Preprocess input data for prediction."""
        raise NotImplementedError
    
    def get_model_version(self):
        """Get the ModelVersion database record for this model."""
        return self._model_version
    
    def predict(self, data: pd.Series = None, steps: int = None) -> np.ndarray:
        """Generate predictions."""
        raise NotImplementedError
    
    def evaluate(self, actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
        """Evaluate model performance."""
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual.iloc[:min_len]
            predicted = predicted[:min_len]
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        
        return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


class ARIMAForecaster(BaseForecaster):
    """ARIMA model forecaster - matches i221500_A02_nlp logic: predicts Close_diff with order (5,1,2)."""
    
    def __init__(self, symbol: str, horizon: int = 1, order: tuple = (5, 1, 2), model=None):
        super().__init__(symbol, horizon)
        self.order = order
        self.model = model
        if model is not None:
            self.is_fitted = True
    
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
    ) -> ARIMAForecaster:
        """
        Load ARIMA model from .pkl file.
        
        If local file doesn't exist, will attempt to download from:
        - Hugging Face Hub (if hf_repo and hf_filename provided)
        - Google Drive (if gdrive_id provided)
        - S3 (if s3_url provided)
        - Generic URL (if remote_url provided)
        
        Args:
            model_path: Local path to model file (or URL string)
            hf_repo: Hugging Face repository ID (e.g., "username/repo-name")
            hf_filename: Filename in Hugging Face repo
            gdrive_id: Google Drive file ID
            s3_url: S3 URL
            remote_url: Generic download URL
            hf_token: Hugging Face token (for private repos)
        
        Returns:
            Loaded ARIMAForecaster instance
        """
        # Convert string to Path if needed
        if isinstance(model_path, str):
            # Check if it's a URL format
            parsed = parse_model_url(model_path)
            if parsed["local_path"]:
                model_path = Path(parsed["local_path"])
            else:
                # Use parsed URL info
                hf_repo = parsed["hf_repo"] or hf_repo
                hf_filename = parsed["hf_filename"] or hf_filename
                gdrive_id = parsed["gdrive_id"] or gdrive_id
                s3_url = parsed["s3_url"] or s3_url
                remote_url = parsed["remote_url"] or remote_url
                # Default local path based on filename from URL
                if hf_filename:
                    model_path = settings.models_dir / hf_filename
                elif remote_url:
                    model_path = settings.models_dir / Path(remote_url).name
                else:
                    model_path = settings.models_dir / "arima_model.pkl"
        
        model_path = Path(model_path)
        
        # Ensure model file exists (download if needed)
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
            logger.error(f"Failed to load ARIMA model: {e}")
            raise
        
        # Load the model
        state = joblib.load(model_path)
        forecaster = cls(
            symbol=state["symbol"],
            horizon=state.get("horizon", 1),
            order=state.get("order", (5, 1, 0)),
            model=state["model"]
        )
        forecaster.is_fitted = True
        
        # Register model version
        _determine_source_and_register(
            model_path=model_path,
            forecaster_instance=forecaster,
            model_type="ARIMA",
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            gdrive_id=gdrive_id,
            s3_url=s3_url,
            remote_url=remote_url,
        )
        
        logger.info(f"Loaded ARIMA model from {model_path}")
        return forecaster
    
    def save(self, timestamp: str = None) -> Path:
        """Save ARIMA model to file."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        
        model_path = settings.models_dir / f"{self.symbol}_arima_{timestamp}.pkl"
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "order": self.order,
            "model": self.model,
        }
        
        joblib.dump(state, model_path)
        logger.info(f"Saved ARIMA model to {model_path}")
        return model_path
    
    def fit(self, data: pd.Series) -> ARIMAForecaster:
        """
        Fit ARIMA model on Close_diff (difference of Close prices).
        Matches i221500_A02_nlp logic: predicts Close_diff with order (5,1,2).
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for ARIMA")
        
        logger.info(f"Fitting ARIMA(order={self.order}) for {self.symbol} on Close_diff")
        
        # Calculate Close_diff (difference of Close prices)
        if isinstance(data, pd.Series):
            close_diff = data.diff().dropna()
        else:
            # If DataFrame, extract Close column
            if 'Close' in data.columns:
                close_diff = data['Close'].diff().dropna()
            elif 'close' in data.columns:
                close_diff = data['close'].diff().dropna()
            else:
                raise ValueError("Data must contain 'Close' column or be a Series of Close prices")
        
        if len(close_diff) < 50:
            raise ValueError(f"Insufficient data: need at least 50 points after differencing, got {len(close_diff)}")
        
        # Fit ARIMA model on Close_diff
        self.model = ARIMA(close_diff, order=self.order).fit()
        self.is_fitted = True
        
        logger.info(f"ARIMA model fitted successfully")
        return self
    
    def preprocess_input(self, data: pd.Series) -> pd.Series:
        """Preprocess input data: calculate Close_diff."""
        if isinstance(data, pd.Series):
            return data.diff().dropna()
        else:
            if 'Close' in data.columns:
                return data['Close'].diff().dropna()
            elif 'close' in data.columns:
                return data['close'].diff().dropna()
            else:
                raise ValueError("Data must contain 'Close' column")
    
    def predict(self, data: pd.Series = None, steps: int = None) -> np.ndarray:
        """
        Generate ARIMA forecast on Close_diff.
        Returns predictions of Close_diff (not absolute Close prices).
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first. Use fit() to train the model.")
        
        steps = steps or self.horizon
        forecast = self.model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else np.array(forecast)


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential Smoothing forecaster."""
    
    def __init__(self, symbol: str, horizon: int = 1, trend: str = 'add', seasonal: Optional[int] = None, model=None):
        super().__init__(symbol, horizon)
        self.trend = trend
        self.seasonal = seasonal
        self.model = model
        if model is not None:
            self.is_fitted = True
    
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
    ) -> ExponentialSmoothingForecaster:
        """
        Load Exponential Smoothing model from .pkl file.
        
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
                    model_path = settings.models_dir / "exponential_smoothing_model.pkl"
        
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
            logger.error(f"Failed to load Exponential Smoothing model: {e}")
            raise
        
        state = joblib.load(model_path)
        forecaster = cls(
            symbol=state["symbol"],
            horizon=state.get("horizon", 1),
            trend=state.get("trend", "add"),
            seasonal=state.get("seasonal"),
            model=state["model"]
        )
        forecaster.is_fitted = True
        
        # Register model version
        _determine_source_and_register(
            model_path=model_path,
            forecaster_instance=forecaster,
            model_type="ExponentialSmoothing",
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            gdrive_id=gdrive_id,
            s3_url=s3_url,
            remote_url=remote_url,
        )
        
        logger.info(f"Loaded Exponential Smoothing model from {model_path}")
        return forecaster
    
    def preprocess_input(self, data: pd.Series) -> pd.Series:
        """Preprocess input data (no transformation needed)."""
        return data.sort_index()
    
    def predict(self, data: pd.Series = None, steps: int = None) -> np.ndarray:
        """Generate Exponential Smoothing forecast."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() to load a trained model.")
        
        steps = steps or self.horizon
        forecast = self.model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else np.array(forecast)


class LSTMForecaster(BaseForecaster):
    """
    LSTM neural network forecaster - matches i221500_A02_nlp logic:
    - Uses multiple features: ['Open','High','Low','Close','Adj Close','Volume'] (normalized)
    - Predicts Close_diff (difference of Close prices)
    - Architecture: LSTM(50, return_sequences=True) -> LSTM(50) -> Dense(1)
    - Input shape: (samples, 1, features)
    """
    
    def __init__(self, symbol: str, horizon: int = 1, lag_window: int = 1, units: int = 50, model=None, scaler=None, feature_cols=None):
        super().__init__(symbol, horizon)
        self.lag_window = lag_window  # For i221500_A02_nlp, this is 1 (single timestep with multiple features)
        self.units = units
        self.model = model
        self.scaler = scaler or MinMaxScaler()
        self.feature_cols = feature_cols  # Store which features were used
        if model is not None:
            self.is_fitted = True
    
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
    ) -> LSTMForecaster:
        """
        Load LSTM model from .h5 (weights) and .pkl (metadata) files.
        
        If local files don't exist, will attempt to download from remote sources.
        For neural models, both weights (.h5) and metadata (.pkl) files are needed.
        If hf_filename is provided, will look for {hf_filename}.h5 and {hf_filename}.pkl.
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
                    # Remove extension if present, we'll add .h5
                    base_name = Path(hf_filename).stem
                    model_path = settings.models_dir / f"{base_name}.h5"
                elif remote_url:
                    model_path = settings.models_dir / Path(remote_url).name
                else:
                    model_path = settings.models_dir / "lstm_model.h5"
        
        model_path = Path(model_path)
        weights_path = model_path if model_path.suffix == '.h5' else model_path.with_suffix('.h5')
        
        # For .weights.h5 files, metadata is {base}.pkl (not .weights.pkl)
        # Example: AAPL_gru_20251110_203709.weights.h5 -> AAPL_gru_20251110_203709.pkl
        if weights_path.name.endswith('.weights.h5'):
            # Remove .weights.h5 and add .pkl
            base_name = weights_path.stem.replace('.weights', '')  # Remove .weights from stem
            metadata_path = weights_path.parent / f"{base_name}.pkl"
        else:
            metadata_path = model_path.with_suffix('.pkl') if model_path.suffix == '.h5' else model_path
        
        # Determine filenames for remote download
        hf_weights_filename = None
        hf_metadata_filename = None
        if hf_filename:
            base_name = Path(hf_filename).stem
            hf_weights_filename = f"{base_name}.h5"
            hf_metadata_filename = f"{base_name}.pkl"
        
        # Ensure weights file exists
        try:
            weights_path = ensure_model_file(
                local_path=weights_path,
                remote_url=remote_url if weights_path.suffix == '.h5' else None,
                hf_repo=hf_repo,
                hf_filename=hf_weights_filename,
                gdrive_id=gdrive_id,
                s3_url=s3_url,
                hf_token=hf_token,
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to load LSTM weights: {e}")
            raise
        
        # Ensure metadata file exists
        try:
            metadata_path = ensure_model_file(
                local_path=metadata_path,
                remote_url=None,  # Metadata usually comes from same source
                hf_repo=hf_repo,
                hf_filename=hf_metadata_filename,
                gdrive_id=None,  # Would need separate ID for metadata
                s3_url=None,  # Would need separate URL for metadata
                hf_token=hf_token,
            )
        except FileNotFoundError:
            # Try to construct metadata path from weights path
            # For .weights.h5 files, metadata is {base}.pkl (not .weights.pkl)
            if weights_path.name.endswith('.weights.h5'):
                base_name = weights_path.stem.replace('.weights', '')
                metadata_path = weights_path.parent / f"{base_name}.pkl"
            else:
                metadata_path = weights_path.with_suffix('.pkl')
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found: {metadata_path}. Attempting download...")
                try:
                    metadata_path = ensure_model_file(
                        local_path=metadata_path,
                        hf_repo=hf_repo,
                        hf_filename=hf_metadata_filename,
                        hf_token=hf_token,
                    )
                except FileNotFoundError as e:
                    logger.error(f"Failed to load LSTM metadata: {e}")
                    raise
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        
        # Rebuild model from config
        model_config = metadata["model_config"]
        model = keras.Sequential.from_config(model_config)
        model.compile(optimizer='adam', loss='mse')
        
        # Load weights
        model.load_weights(str(weights_path))
        
        forecaster = cls(
            symbol=metadata["symbol"],
            horizon=metadata.get("horizon", 1),
            lag_window=metadata.get("lag_window", 1),
            units=metadata.get("units", 50),
            model=model,
            scaler=metadata["scaler"],
            feature_cols=metadata.get("feature_cols")  # Load feature columns if available
        )
        forecaster.is_fitted = True
        
        # Load Close range for denormalization if available
        if "close_range" in metadata and metadata["close_range"] is not None:
            forecaster.close_range = float(metadata["close_range"])
            forecaster.close_min = float(metadata.get("close_min", 0))
            forecaster.close_max = float(metadata.get("close_max", 0))
            forecaster.close_col_idx_in_num_cols = metadata.get("close_col_idx_in_num_cols")
            forecaster.close_col_name = metadata.get("close_col_name")
            logger.info(f"LSTM: Loaded Close range from metadata: {forecaster.close_range:.4f}")
        else:
            # Try to calculate from scaler if available
            if hasattr(forecaster.scaler, 'data_min_') and hasattr(forecaster.scaler, 'data_max_'):
                # The scaler was fit on num_cols which includes Close
                # We need to find which index corresponds to Close
                close_col_idx = metadata.get("close_col_idx_in_num_cols")
                if close_col_idx is not None and close_col_idx < len(forecaster.scaler.data_min_):
                    forecaster.close_range = float(
                        forecaster.scaler.data_max_[close_col_idx] - forecaster.scaler.data_min_[close_col_idx]
                    )
                    forecaster.close_min = float(forecaster.scaler.data_min_[close_col_idx])
                    forecaster.close_max = float(forecaster.scaler.data_max_[close_col_idx])
                    forecaster.close_col_idx_in_num_cols = close_col_idx
                    forecaster.close_col_name = metadata.get("close_col_name", "Close")
                    logger.info(f"LSTM: Calculated Close range from scaler: {forecaster.close_range:.4f}")
                else:
                    # Last resort: assume Close is the last column or find it by name
                    # If scaler has feature_names_in_, use that
                    if hasattr(forecaster.scaler, 'feature_names_in_'):
                        feature_names = list(forecaster.scaler.feature_names_in_)
                        close_idx = None
                        for i, name in enumerate(feature_names):
                            if 'close' in name.lower() and 'adj' not in name.lower():
                                close_idx = i
                                break
                        if close_idx is not None:
                            forecaster.close_range = float(
                                forecaster.scaler.data_max_[close_idx] - forecaster.scaler.data_min_[close_idx]
                            )
                            forecaster.close_min = float(forecaster.scaler.data_min_[close_idx])
                            forecaster.close_max = float(forecaster.scaler.data_max_[close_idx])
                            forecaster.close_col_idx_in_num_cols = close_idx
                            forecaster.close_col_name = feature_names[close_idx]
                            logger.info(f"LSTM: Calculated Close range from scaler feature_names: {forecaster.close_range:.4f}")
                        else:
                            logger.warning("LSTM: Could not find Close column in scaler feature_names. Will calculate during prediction.")
                    else:
                        logger.warning("LSTM: close_range not in metadata and cannot infer from scaler. Will calculate during prediction.")
            else:
                logger.warning("LSTM: close_range not in metadata and scaler missing data_min_/data_max_. Will calculate during prediction.")
        
        # Register model version
        _determine_source_and_register(
            model_path=weights_path,
            forecaster_instance=forecaster,
            model_type="LSTM",
            hf_repo=hf_repo,
            hf_filename=hf_weights_filename,
            gdrive_id=gdrive_id,
            s3_url=s3_url,
            remote_url=remote_url,
        )
        
        logger.info(f"Loaded LSTM model from {weights_path}")
        return forecaster
    
    def save(self, timestamp: str = None) -> Path:
        """Save LSTM model to .h5 (weights) and .pkl (metadata) files."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        weights_path = settings.models_dir / f"{self.symbol}_lstm_{timestamp}.weights.h5"
        metadata_path = settings.models_dir / f"{self.symbol}_lstm_{timestamp}.pkl"
        
        # Save weights
        self.model.save_weights(str(weights_path))
        
        # Save metadata
        metadata = {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "lag_window": self.lag_window,
            "units": self.units,
            "model_config": self.model.get_config(),
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,  # Save feature columns
            # Save Close range for denormalization
            "close_range": getattr(self, 'close_range', None),
            "close_min": getattr(self, 'close_min', None),
            "close_max": getattr(self, 'close_max', None),
            "close_col_idx_in_num_cols": getattr(self, 'close_col_idx_in_num_cols', None),
            "close_col_name": getattr(self, 'close_col_name', None),
        }
        
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved LSTM model to {weights_path} and {metadata_path}")
        return weights_path
    
    def fit(self, data: pd.DataFrame) -> LSTMForecaster:
        """
        Fit LSTM model on multiple features to predict Close_diff.
        Matches i221500_A02_nlp logic:
        - Normalizes ['Open','High','Low','Close','Adj Close','Volume']
        - Predicts Close_diff
        - Architecture: LSTM(50, return_sequences=True) -> LSTM(50) -> Dense(1)
        """
        logger.info(f"Fitting LSTM(units={self.units}) for {self.symbol} on Close_diff with multiple features")
        
        # Prepare data - need DataFrame with OHLCV columns
        if isinstance(data, pd.Series):
            raise ValueError("LSTM requires DataFrame with multiple features, not Series")
        
        # Use available columns (case-insensitive)
        num_cols = []
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            # Try exact match first
            if col in data.columns:
                num_cols.append(col)
            else:
                # Try case-insensitive
                col_lower = col.lower()
                matching = [c for c in data.columns if c.lower() == col_lower]
                if matching:
                    num_cols.append(matching[0])
        
        if len(num_cols) < 4:
            raise ValueError(f"Need at least 4 features (Open, High, Low, Close). Found: {num_cols}")
        
        # Normalize features
        scaled = self.scaler.fit_transform(data[num_cols])
        df_scaled = pd.DataFrame(scaled, columns=num_cols)
        
        # Calculate Close_diff on normalized Close
        close_col = [c for c in num_cols if 'close' in c.lower()][0]
        close_col_idx = num_cols.index(close_col)  # Store index for denormalization
        df_scaled['Close_diff'] = df_scaled[close_col].diff()
        df_scaled = df_scaled.dropna()
        
        if len(df_scaled) < 50:
            raise ValueError(f"Insufficient data: need at least 50 points after differencing, got {len(df_scaled)}")
        
        # Features: all columns except Close and Close_diff
        feature_cols = [c for c in num_cols if c != close_col]
        X = df_scaled[feature_cols].values
        y = df_scaled['Close_diff'].values
        
        # Store Close column info for denormalization during prediction
        self.close_col_name = close_col
        self.close_col_idx_in_num_cols = close_col_idx
        # Store original Close min/max for denormalization
        if hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
            self.close_min = float(self.scaler.data_min_[close_col_idx])
            self.close_max = float(self.scaler.data_max_[close_col_idx])
            self.close_range = self.close_max - self.close_min
            logger.info(f"LSTM: Stored Close range for denormalization: {self.close_range:.4f} (min={self.close_min:.4f}, max={self.close_max:.4f})")
        else:
            # Calculate from original data as fallback
            self.close_range = float(data[close_col].max() - data[close_col].min())
            logger.warning(f"LSTM: Calculated Close range from data: {self.close_range:.4f} (scaler missing min/max)")
        
        # Reshape for LSTM: (samples, 1, features) - matches i221500_A02_nlp
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Build model architecture matching i221500_A02_nlp
        self.model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])),
            layers.LSTM(self.units),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train with parameters from i221500_A02_nlp: epochs=5, batch_size=16
        self.model.fit(X_reshaped, y, epochs=5, batch_size=16, verbose=1)
        
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        logger.info(f"LSTM model fitted successfully on {len(feature_cols)} features")
        return self
    
    def preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for LSTM prediction: normalize features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if isinstance(data, pd.Series):
            raise ValueError("LSTM requires DataFrame with multiple features")
        
        # CRITICAL: The scaler was fit on num_cols (which includes Close),
        # but feature_cols excludes Close. We need to provide all columns
        # that the scaler expects, then select only feature_cols for the model.
        
        # Determine what columns the scaler expects (all columns used during training)
        # This includes Close even though it's not in feature_cols
        scaler_expected_cols = None
        if hasattr(self.scaler, 'feature_names_in_'):
            scaler_expected_cols = list(self.scaler.feature_names_in_)
        elif hasattr(self, 'close_col_name'):
            # If we know Close was used, scaler expects: feature_cols + Close
            scaler_expected_cols = list(self.feature_cols) + [self.close_col_name] if self.feature_cols else None
        
        # If feature_cols is None (old model), try to infer
        if self.feature_cols is None:
            logger.warning("LSTM model missing feature_cols. Attempting to infer from data...")
            # Try to infer from common column names
            common_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_features = [col for col in common_features if col in data.columns or col.lower() in [c.lower() for c in data.columns]]
            
            if len(available_features) == 0:
                # Last resort: use all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) == 0:
                    raise ValueError(
                        f"Could not infer feature columns. "
                        f"Model missing feature_cols and data has no matching columns. "
                        f"Available columns: {list(data.columns)}. "
                        f"Model may need to be retrained."
                    )
                available_features = numeric_cols[:5]  # Use first 5 numeric columns
                logger.warning(f"Using inferred features from numeric columns: {available_features}")
            else:
                logger.warning(f"Using inferred features: {available_features}")
            
            self.feature_cols = available_features
        
        # Build list of all columns needed for scaler (feature_cols + Close if Close was used)
        all_cols_for_scaler = list(self.feature_cols)
        close_col_name = getattr(self, 'close_col_name', None)
        if close_col_name and close_col_name not in all_cols_for_scaler:
            all_cols_for_scaler.append(close_col_name)
        
        # If scaler has feature_names_in_, use those
        if scaler_expected_cols:
            all_cols_for_scaler = scaler_expected_cols
        
        # Get available columns for scaler (case-insensitive)
        available_for_scaler = []
        for col in all_cols_for_scaler:
            if col in data.columns:
                available_for_scaler.append(col)
            else:
                col_lower = col.lower()
                matching = [c for c in data.columns if c.lower() == col_lower]
                if matching:
                    available_for_scaler.append(matching[0])
                else:
                    logger.warning(f"Scaler expects '{col}' but not found in data. Available: {list(data.columns)}")
        
        # Get available columns for model (feature_cols only)
        available_for_model = []
        for col in self.feature_cols:
            if col in data.columns:
                available_for_model.append(col)
            else:
                col_lower = col.lower()
                matching = [c for c in data.columns if c.lower() == col_lower]
                if matching:
                    available_for_model.append(matching[0])
                else:
                    logger.error(f"Required feature '{col}' not found in data. Available: {list(data.columns)}")
        
        if len(available_for_model) != len(self.feature_cols):
            raise ValueError(
                f"Feature mismatch: Need {len(self.feature_cols)} features ({self.feature_cols}), "
                f"but only found {len(available_for_model)} matching columns ({available_for_model}) in data. "
                f"Data columns: {list(data.columns)}"
            )
        
        # Normalize all columns that scaler expects
        # Create a DataFrame with all required columns in the order scaler expects
        scaler_input_df = pd.DataFrame()
        for col in all_cols_for_scaler:
            if col in available_for_scaler:
                scaler_input_df[col] = data[col].values if col in data.columns else data[[c for c in data.columns if c.lower() == col.lower()][0]].values
            elif col in available_for_model:
                # Use from model features
                matching_col = [c for c in data.columns if c.lower() == col.lower()][0]
                scaler_input_df[col] = data[matching_col].values
            else:
                # Missing column - use Close as fallback
                close_val = data.get('Close', data.get('close', data.iloc[:, 0])).iloc[0] if len(data) > 0 else 0
                scaler_input_df[col] = [close_val]
                logger.warning(f"Using fallback value for missing column '{col}'")
        
        # Transform with scaler (expects all columns it was fit on)
        try:
            scaled_all = self.scaler.transform(scaler_input_df.values)
        except ValueError as e:
            # If scaler expects different number of features, try with just feature_cols
            logger.warning(f"Scaler transform failed with all columns, trying with feature_cols only: {e}")
            scaled_all = self.scaler.transform(data[available_for_model].values)
            # Extract only feature_cols from scaled result
            # This is a fallback - ideally scaler should match
            if scaled_all.shape[1] > len(self.feature_cols):
                # Assume first N columns are feature_cols
                scaled = scaled_all[:, :len(self.feature_cols)]
            else:
                scaled = scaled_all
        else:
            # Extract only feature_cols columns from scaled result
            # Find indices of feature_cols in all_cols_for_scaler
            feature_indices = [all_cols_for_scaler.index(col) for col in self.feature_cols if col in all_cols_for_scaler]
            if len(feature_indices) == len(self.feature_cols):
                scaled = scaled_all[:, feature_indices]
            else:
                # Fallback: use first N columns
                scaled = scaled_all[:, :len(self.feature_cols)]
        
        # Reshape: (1, 1, features) - single sample, single timestep, multiple features
        return scaled.reshape((1, 1, scaled.shape[1]))
    
    def predict(self, data: pd.DataFrame = None, steps: int = None) -> np.ndarray:
        """
        Generate LSTM forecast on Close_diff.
        Returns predictions of Close_diff (not absolute Close prices).
        
        CRITICAL: LSTM predicts normalized Close_diff, which must be denormalized.
        The scaler was fit on features (Open, High, Low, Close, Volume), and Close_diff
        was calculated on normalized Close. To denormalize Close_diff, we multiply by
        the Close column's range from the scaler.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first. Use fit() to train the model.")
        
        steps = steps or self.horizon
        
        if data is None:
            raise ValueError("LSTM requires recent data for prediction")
        
        # Preprocess input
        X = self.preprocess_input(data)
        
        # DEBUG: Log input to model
        logger.debug(
            f"LSTM input shape: {X.shape}, "
            f"Input range: [{np.min(X):.6f}, {np.max(X):.6f}], "
            f"Input mean: {np.mean(X):.6f}, "
            f"Input std: {np.std(X):.6f}"
        )
        
        # Validate input data quality
        if np.all(X == 0) or np.std(X) < 1e-6:
            logger.warning(
                f"⚠️ LSTM received flat/zero input! "
                f"This will produce poor predictions. "
                f"Input stats: min={np.min(X):.6f}, max={np.max(X):.6f}, std={np.std(X):.6f}"
            )
        
        # Predict normalized Close_diff for each step
        predictions_normalized = []
        for step in range(steps):
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions_normalized.append(pred)
            # For next prediction, we could update X, but for simplicity, use same input
            # (In practice, you'd use the predicted value to update features)
            if step == 0:
                logger.debug(f"LSTM first prediction (normalized): {pred:.6f}")
        
        predictions_normalized = np.array(predictions_normalized)
        
        # DEBUG: Log raw predictions before denormalization
        logger.debug(
            f"LSTM normalized predictions: "
            f"First 3: {predictions_normalized[:3]}, "
            f"Range: [{np.min(predictions_normalized):.6f}, {np.max(predictions_normalized):.6f}], "
            f"Mean: {np.mean(predictions_normalized):.6f}, "
            f"All zero: {np.all(predictions_normalized == 0)}"
        )
        
        # CRITICAL FIX: Denormalize Close_diff predictions
        # The scaler was fit on features including Close, and Close_diff was calculated
        # on normalized Close. To denormalize: pred_actual = pred_normalized * close_range
        
        # Use stored close_range from training (most accurate)
        if hasattr(self, 'close_range') and self.close_range is not None and self.close_range > 0:
            predictions_denormalized = predictions_normalized * self.close_range
            logger.info(
                f"LSTM denormalization (using stored range): Close range={self.close_range:.4f}, "
                f"Normalized pred (first 3): {predictions_normalized[:3]}, "
                f"Denormalized pred (first 3): {predictions_denormalized[:3]}"
            )
            
            # Validate denormalized predictions
            if np.all(predictions_denormalized == 0) and not np.all(predictions_normalized == 0):
                logger.error(
                    f"⚠️ LSTM denormalization produced all zeros! "
                    f"This suggests close_range={self.close_range:.4f} may be incorrect or predictions are truly zero."
                )
            elif np.all(predictions_normalized == 0):
                logger.warning(
                    f"⚠️ LSTM model produced all-zero normalized predictions! "
                    f"This indicates the model received bad input or is not properly trained. "
                    f"Input data may be flat or incorrectly scaled."
                )
        elif hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
            # Fallback: try to get from scaler if we know Close column index
            if hasattr(self, 'close_col_idx_in_num_cols'):
                close_range = self.scaler.data_max_[self.close_col_idx_in_num_cols] - self.scaler.data_min_[self.close_col_idx_in_num_cols]
                predictions_denormalized = predictions_normalized * close_range
                logger.debug(
                    f"LSTM denormalization (from scaler): Close range={close_range:.4f}, "
                    f"Normalized pred (first 3): {predictions_normalized[:3]}, "
                    f"Denormalized pred (first 3): {predictions_denormalized[:3]}"
                )
            else:
                # Last resort: calculate from input data
                try:
                    close_col_name = None
                    for col in data.columns:
                        if 'close' in col.lower() and 'adj' not in col.lower():
                            close_col_name = col
                            break
                    
                    if close_col_name:
                        close_values = data[close_col_name].values
                        close_range = float(np.max(close_values) - np.min(close_values))
                        predictions_denormalized = predictions_normalized * close_range
                        logger.warning(
                            f"LSTM denormalization (from input data): Close range={close_range:.4f}"
                        )
                    else:
                        # Use average of all feature ranges
                        feature_ranges = self.scaler.data_max_ - self.scaler.data_min_
                        avg_range = np.mean(feature_ranges)
                        predictions_denormalized = predictions_normalized * avg_range
                        logger.warning(
                            f"LSTM: Using average feature range {avg_range:.4f} for denormalization (less accurate)"
                        )
                except Exception as e:
                    logger.error(f"LSTM denormalization failed: {e}. Returning normalized predictions (WILL CAUSE ERRORS).")
                    return predictions_normalized
        else:
            logger.error("LSTM: Cannot denormalize - missing scaler attributes and close_range. Returning normalized (WILL CAUSE ERRORS).")
            return predictions_normalized
        
        return predictions_denormalized


class GRUForecaster(BaseForecaster):
    """GRU neural network forecaster."""
    
    def __init__(self, symbol: str, horizon: int = 1, lag_window: int = 60, units: int = 50, model=None, scaler=None):
        super().__init__(symbol, horizon)
        self.lag_window = lag_window
        self.units = units
        self.model = model
        self.scaler = scaler or MinMaxScaler()
        if model is not None:
            self.is_fitted = True
    
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
    ) -> GRUForecaster:
        """
        Load GRU model from .h5 (weights) and .pkl (metadata) files.
        
        If local files don't exist, will attempt to download from remote sources.
        See LSTMForecaster.load_model() for details.
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
                    base_name = Path(hf_filename).stem
                    model_path = settings.models_dir / f"{base_name}.h5"
                elif remote_url:
                    model_path = settings.models_dir / Path(remote_url).name
                else:
                    model_path = settings.models_dir / "gru_model.h5"
        
        model_path = Path(model_path)
        weights_path = model_path if model_path.suffix == '.h5' else model_path.with_suffix('.h5')
        
        # For .weights.h5 files, metadata is {base}.pkl (not .weights.pkl)
        # Example: AAPL_gru_20251110_203709.weights.h5 -> AAPL_gru_20251110_203709.pkl
        if weights_path.name.endswith('.weights.h5'):
            # Remove .weights.h5 and add .pkl
            base_name = weights_path.stem.replace('.weights', '')  # Remove .weights from stem
            metadata_path = weights_path.parent / f"{base_name}.pkl"
        else:
            metadata_path = model_path.with_suffix('.pkl') if model_path.suffix == '.h5' else model_path
        
        hf_weights_filename = None
        hf_metadata_filename = None
        if hf_filename:
            base_name = Path(hf_filename).stem
            hf_weights_filename = f"{base_name}.h5"
            hf_metadata_filename = f"{base_name}.pkl"
        
        try:
            weights_path = ensure_model_file(
                local_path=weights_path,
                remote_url=remote_url if weights_path.suffix == '.h5' else None,
                hf_repo=hf_repo,
                hf_filename=hf_weights_filename,
                gdrive_id=gdrive_id,
                s3_url=s3_url,
                hf_token=hf_token,
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to load GRU weights: {e}")
            raise
        
        try:
            metadata_path = ensure_model_file(
                local_path=metadata_path,
                hf_repo=hf_repo,
                hf_filename=hf_metadata_filename,
                hf_token=hf_token,
            )
        except FileNotFoundError:
            metadata_path = weights_path.with_suffix('.pkl')
            if not metadata_path.exists():
                try:
                    metadata_path = ensure_model_file(
                        local_path=metadata_path,
                        hf_repo=hf_repo,
                        hf_filename=hf_metadata_filename,
                        hf_token=hf_token,
                    )
                except FileNotFoundError as e:
                    logger.error(f"Failed to load GRU metadata: {e}")
                    raise
        
        metadata = joblib.load(metadata_path)
        model_config = metadata["model_config"]
        model = keras.Sequential.from_config(model_config)
        model.compile(optimizer='adam', loss='mse')
        model.load_weights(str(weights_path))
        
        forecaster = cls(
            symbol=metadata["symbol"],
            horizon=metadata.get("horizon", 1),
            lag_window=metadata["lag_window"],
            units=metadata["units"],
            model=model,
            scaler=metadata["scaler"]
        )
        forecaster.is_fitted = True
        
        # Register model version
        _determine_source_and_register(
            model_path=weights_path,
            forecaster_instance=forecaster,
            model_type="GRU",
            hf_repo=hf_repo,
            hf_filename=hf_weights_filename,
            gdrive_id=gdrive_id,
            s3_url=s3_url,
            remote_url=remote_url,
        )
        
        logger.info(f"Loaded GRU model from {weights_path}")
        return forecaster
    
    def preprocess_input(self, data: pd.Series) -> np.ndarray:
        """Preprocess input data for GRU prediction."""
        if len(data) < self.lag_window:
            raise ValueError(f"Need at least {self.lag_window} points for prediction")
        
        last_data = data.iloc[-self.lag_window:].values
        scaled = self.scaler.transform(last_data.reshape(-1, 1)).flatten()
        return scaled
    
    def predict(self, data: pd.Series, steps: int = None) -> np.ndarray:
        """Generate GRU forecast."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() to load a trained model.")
        
        steps = steps or self.horizon
        
        # Preprocess input
        scaled = self.preprocess_input(data)
        
        predictions = []
        current_window = scaled.copy()
        
        for _ in range(steps):
            X = current_window.reshape((1, self.lag_window, 1))
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions


class TransformerForecaster(BaseForecaster):
    """Transformer-based forecaster (simplified version)."""
    
    def __init__(self, symbol: str, horizon: int = 1, lag_window: int = 60, d_model: int = 64, model=None, scaler=None):
        super().__init__(symbol, horizon)
        self.lag_window = lag_window
        self.d_model = d_model
        self.model = model
        self.scaler = scaler or MinMaxScaler()
        if model is not None:
            self.is_fitted = True
    
    @classmethod
    def _transformer_block(cls, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = layers.Dropout(dropout)(attention_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(inputs.shape[-1]),
        ])
        ffn_output = ffn(out1)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
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
    ) -> TransformerForecaster:
        """
        Load Transformer model from .h5 (weights) and .pkl (metadata) files.
        
        If local files don't exist, will attempt to download from remote sources.
        See LSTMForecaster.load_model() for details.
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
                    base_name = Path(hf_filename).stem
                    model_path = settings.models_dir / f"{base_name}.h5"
                elif remote_url:
                    model_path = settings.models_dir / Path(remote_url).name
                else:
                    model_path = settings.models_dir / "transformer_model.h5"
        
        model_path = Path(model_path)
        weights_path = model_path if model_path.suffix == '.h5' else model_path.with_suffix('.h5')
        
        # For .weights.h5 files, metadata is {base}.pkl (not .weights.pkl)
        # Example: AAPL_gru_20251110_203709.weights.h5 -> AAPL_gru_20251110_203709.pkl
        if weights_path.name.endswith('.weights.h5'):
            # Remove .weights.h5 and add .pkl
            base_name = weights_path.stem.replace('.weights', '')  # Remove .weights from stem
            metadata_path = weights_path.parent / f"{base_name}.pkl"
        else:
            metadata_path = model_path.with_suffix('.pkl') if model_path.suffix == '.h5' else model_path
        
        hf_weights_filename = None
        hf_metadata_filename = None
        if hf_filename:
            base_name = Path(hf_filename).stem
            hf_weights_filename = f"{base_name}.h5"
            hf_metadata_filename = f"{base_name}.pkl"
        
        try:
            weights_path = ensure_model_file(
                local_path=weights_path,
                remote_url=remote_url if weights_path.suffix == '.h5' else None,
                hf_repo=hf_repo,
                hf_filename=hf_weights_filename,
                gdrive_id=gdrive_id,
                s3_url=s3_url,
                hf_token=hf_token,
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to load Transformer weights: {e}")
            raise
        
        try:
            metadata_path = ensure_model_file(
                local_path=metadata_path,
                hf_repo=hf_repo,
                hf_filename=hf_metadata_filename,
                hf_token=hf_token,
            )
        except FileNotFoundError:
            metadata_path = weights_path.with_suffix('.pkl')
            if not metadata_path.exists():
                try:
                    metadata_path = ensure_model_file(
                        local_path=metadata_path,
                        hf_repo=hf_repo,
                        hf_filename=hf_metadata_filename,
                        hf_token=hf_token,
                    )
                except FileNotFoundError as e:
                    logger.error(f"Failed to load Transformer metadata: {e}")
                    raise
        
        metadata = joblib.load(metadata_path)
        model_config = metadata["model_config"]
        model = keras.Model.from_config(model_config)
        model.compile(optimizer='adam', loss='mse')
        model.load_weights(str(weights_path))
        
        forecaster = cls(
            symbol=metadata["symbol"],
            horizon=metadata.get("horizon", 1),
            lag_window=metadata["lag_window"],
            d_model=metadata["d_model"],
            model=model,
            scaler=metadata["scaler"]
        )
        forecaster.is_fitted = True
        
        # Register model version
        _determine_source_and_register(
            model_path=weights_path,
            forecaster_instance=forecaster,
            model_type="Transformer",
            hf_repo=hf_repo,
            hf_filename=hf_weights_filename,
            gdrive_id=gdrive_id,
            s3_url=s3_url,
            remote_url=remote_url,
        )
        
        logger.info(f"Loaded Transformer model from {weights_path}")
        return forecaster
    
    def preprocess_input(self, data: pd.Series) -> np.ndarray:
        """Preprocess input data for Transformer prediction."""
        if len(data) < self.lag_window:
            raise ValueError(f"Need at least {self.lag_window} points for prediction")
        
        last_data = data.iloc[-self.lag_window:].values
        
        # Handle both MinMaxScaler and StandardScaler
        is_minmax = hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_')
        is_standard = hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_')
        
        if is_minmax:
            # MinMaxScaler: clip to training range
            scaler_min = self.scaler.data_min_[0]
            scaler_max = self.scaler.data_max_[0]
            
            data_clipped = np.clip(last_data, scaler_min, scaler_max)
            
            if np.any(last_data != data_clipped):
                clipped_count = np.sum(last_data != data_clipped)
                logger.warning(
                    f"Transformer: Clipped {clipped_count}/{len(last_data)} input values "
                    f"to scaler range [${scaler_min:.2f}, ${scaler_max:.2f}]. "
                    f"Original range: [${np.min(last_data):.2f}, ${np.max(last_data):.2f}]."
                )
            
            scaled = self.scaler.transform(data_clipped.reshape(-1, 1)).flatten()
            # Clip to [0, 1] for MinMaxScaler
            scaled = np.clip(scaled, 0, 1)
        elif is_standard:
            # StandardScaler: no hard bounds, just standardize
            scaled = self.scaler.transform(last_data.reshape(-1, 1)).flatten()
            # Log if values are very far from mean (beyond 3 std)
            if np.any(np.abs(scaled) > 3):
                extreme_count = np.sum(np.abs(scaled) > 3)
                logger.warning(
                    f"Transformer: {extreme_count}/{len(scaled)} input values are >3 std from mean. "
                    f"This may indicate extrapolation."
                )
        else:
            # Unknown scaler type, just transform
            scaled = self.scaler.transform(last_data.reshape(-1, 1)).flatten()
        
        return scaled
    
    def predict(self, data: pd.Series, steps: int = None) -> np.ndarray:
        """Generate Transformer forecast."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() to load a trained model.")
        
        steps = steps or self.horizon
        
        # Get last close price for validation
        last_close = float(data.iloc[-1])
        
        # Preprocess input
        scaled = self.preprocess_input(data)
        
        # DEBUG: Log scaler range and input
        is_minmax = hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_')
        is_standard = hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_')
        
        if is_minmax:
            logger.debug(
                f"Transformer scaler (MinMax): range [{self.scaler.data_min_[0]:.2f}, {self.scaler.data_max_[0]:.2f}], "
                f"Last close: {last_close:.2f}, "
                f"Input scaled range: [{np.min(scaled):.6f}, {np.max(scaled):.6f}]"
            )
        elif is_standard:
            logger.debug(
                f"Transformer scaler (Standard): mean={self.scaler.mean_[0]:.2f}, std={self.scaler.scale_[0]:.2f}, "
                f"Last close: {last_close:.2f}, "
                f"Input scaled range: [{np.min(scaled):.6f}, {np.max(scaled):.6f}]"
            )
        
        predictions = []
        current_window = scaled.copy()
        
        for step in range(steps):
            X = current_window.reshape((1, self.lag_window, 1))
            pred_normalized = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred_normalized)
            
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred_normalized
            
            if step == 0:
                logger.debug(f"Transformer first prediction (normalized): {pred_normalized:.6f}")
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Log normalized prediction before clipping to diagnose saturation
        first_pred_normalized = predictions[0, 0]
        if is_minmax:
            scaler_max_val = self.scaler.data_max_[0]
            logger.info(
                f"Transformer: First normalized prediction: {first_pred_normalized:.6f} "
                f"(range [0, 1], where 1.0 = scaler max ${scaler_max_val:.2f})"
            )
            # Clip normalized predictions to [0, 1] for MinMaxScaler
            predictions_clipped = np.clip(predictions, 0, 1)
            if np.any(predictions != predictions_clipped):
                clipped_count = np.sum(predictions != predictions_clipped)
                logger.warning(
                    f"Transformer: Clipped {clipped_count}/{len(predictions)} normalized predictions "
                    f"to [0, 1] range. This prevents extreme inverse-transformed values."
                )
            predictions = predictions_clipped
        elif is_standard:
            logger.info(
                f"Transformer: First normalized prediction: {first_pred_normalized:.6f} "
                f"(standardized, mean=0, std=1)"
            )
            # For StandardScaler, clip extreme values (>3 std) to prevent wild predictions
            predictions_clipped = np.clip(predictions, -3, 3)
            if np.any(np.abs(predictions) > 3):
                clipped_count = np.sum(np.abs(predictions) > 3)
                logger.warning(
                    f"Transformer: Clipped {clipped_count}/{len(predictions)} predictions "
                    f"beyond ±3 std to prevent extreme values."
                )
            predictions = predictions_clipped
        else:
            logger.info(
                f"Transformer: First normalized prediction: {first_pred_normalized:.6f}"
            )
        
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Validate and potentially clip predictions to reasonable range
        first_pred = predictions[0]
        change_pct = ((first_pred - last_close) / last_close * 100) if last_close > 0 else 0
        
        # Check if retraining is needed (automatic retrain trigger)
        # If current price exceeds scaler_max * 1.02, recommend retraining
        retrain_needed = False
        if is_minmax:
            scaler_max = self.scaler.data_max_[0]
            if last_close > scaler_max * 1.02:
                retrain_needed = True
                logger.warning(
                    f"🔄 RETRAIN TRIGGER: Current price ${last_close:.2f} exceeds scaler max * 1.02 "
                    f"(${scaler_max * 1.02:.2f}). Model should be retrained to include new price range."
                )
        elif is_standard:
            mean_price = self.scaler.mean_[0]
            std_price = self.scaler.scale_[0]
            # For StandardScaler, trigger if price is > mean + 3.5*std (very extreme)
            threshold = mean_price + 3.5 * std_price
            if last_close > threshold:
                retrain_needed = True
                logger.warning(
                    f"🔄 RETRAIN TRIGGER: Current price ${last_close:.2f} exceeds mean + 3.5*std "
                    f"(${threshold:.2f}). Model should be retrained to include new price range."
                )
        
        # Check if scaler range matches current data
        # Only apply conservative bounds if price is SIGNIFICANTLY outside range (>1% away)
        extrapolation_detected = False
        if is_minmax:
            scaler_min = self.scaler.data_min_[0]
            scaler_max = self.scaler.data_max_[0]
            
            # Calculate how far outside the range we are (as percentage)
            if last_close < scaler_min:
                pct_below = ((scaler_min - last_close) / scaler_min) * 100
                if pct_below > 1.0:  # More than 1% below min
                    extrapolation_detected = True
            elif last_close > scaler_max:
                pct_above = ((last_close - scaler_max) / scaler_max) * 100
                if pct_above > 1.0:  # More than 1% above max
                    extrapolation_detected = True
            # If within 1% of bounds, don't treat as extrapolation
            
            if extrapolation_detected:
                logger.warning(
                    f"⚠️ Transformer: Current price ${last_close:.2f} is significantly outside scaler range "
                    f"[${scaler_min:.2f}, ${scaler_max:.2f}] (>1% away). "
                    f"Using conservative prediction bounds."
                )
                
                # Apply conservative bounds: limit prediction to reasonable range around current price
                # Allow up to 10% change if extrapolating, otherwise use normal bounds
                max_change_pct = 10.0 if extrapolation_detected else 25.0
                pred_min = last_close * (1 - max_change_pct / 100)
                pred_max = last_close * (1 + max_change_pct / 100)
                
                predictions_clipped = np.clip(predictions, pred_min, pred_max)
                if np.any(predictions != predictions_clipped):
                    logger.info(
                        f"Transformer: Clipped predictions to conservative range "
                        f"[${pred_min:.2f}, ${pred_max:.2f}] due to extrapolation."
                    )
                    predictions = predictions_clipped
                    first_pred = predictions[0]
                    change_pct = ((first_pred - last_close) / last_close * 100) if last_close > 0 else 0
            else:
                # Price is within 1% of scaler bounds - log but don't clamp
                if last_close < scaler_min or last_close > scaler_max:
                    logger.debug(
                        f"Transformer: Current price ${last_close:.2f} is slightly outside scaler range "
                        f"[${scaler_min:.2f}, ${scaler_max:.2f}] (<1% away). "
                        f"No conservative bounds applied."
                    )
        elif is_standard:
            # For StandardScaler, check if current price is very far from mean (>3 std)
            mean_price = self.scaler.mean_[0]
            std_price = self.scaler.scale_[0]
            z_score = (last_close - mean_price) / std_price
            
            if abs(z_score) > 3:
                extrapolation_detected = True
                logger.warning(
                    f"⚠️ Transformer: Current price ${last_close:.2f} is {z_score:.2f} std from mean "
                    f"(${mean_price:.2f}). This may indicate extrapolation."
                )
                # Apply conservative bounds for extreme extrapolation
                max_change_pct = 10.0
                pred_min = last_close * (1 - max_change_pct / 100)
                pred_max = last_close * (1 + max_change_pct / 100)
                
                predictions_clipped = np.clip(predictions, pred_min, pred_max)
                if np.any(predictions != predictions_clipped):
                    logger.info(
                        f"Transformer: Clipped predictions to conservative range "
                        f"[${pred_min:.2f}, ${pred_max:.2f}] due to extreme extrapolation."
                    )
                    predictions = predictions_clipped
                    first_pred = predictions[0]
                    change_pct = ((first_pred - last_close) / last_close * 100) if last_close > 0 else 0
        
        # Check for extreme predictions
        if abs(change_pct) > 20:  # More than 20% change
            logger.warning(
                f"⚠️ Transformer prediction is extreme! "
                f"First pred: ${first_pred:.2f}, Last close: ${last_close:.2f}, "
                f"Change: {change_pct:.2f}%. "
                f"This may indicate:"
                f"  - Scaler was fit on different data range (extrapolation)"
                f"  - Model is poorly trained or overfitting"
                f"  - Input data is outside training distribution"
            )
        
        # Log prediction stats
        logger.info(
            f"Transformer predictions: "
            f"First: ${first_pred:.2f}, "
            f"Range: [${np.min(predictions):.2f}, ${np.max(predictions):.2f}], "
            f"Change from close: {change_pct:.2f}%"
        )
        
        return predictions


class VARForecaster(BaseForecaster):
    """Vector Autoregression (VAR) model forecaster for multivariate time series."""
    
    def __init__(self, symbol: str, horizon: int = 1, maxlags: int = 5):
        super().__init__(symbol, horizon)
        self.maxlags = maxlags
        self.model = None
        self.columns = None
    
    def fit(self, data: pd.DataFrame) -> VARForecaster:
        """Fit VAR model on multivariate data (requires DataFrame with multiple columns)."""
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for VAR")
        
        if isinstance(data, pd.Series):
            raise ValueError("VAR requires multivariate data (DataFrame), not Series")
        
        logger.info(f"Fitting VAR(maxlags={self.maxlags}) for {self.symbol}")
        
        # Use OHLC columns if available
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in ohlc_cols):
            var_data = data[ohlc_cols].copy()
        elif all(col.lower() in [c.lower() for c in data.columns] for col in ohlc_cols):
            # Case-insensitive match
            var_data = data[[c for c in data.columns if c.lower() in [col.lower() for col in ohlc_cols]]].copy()
        else:
            # Use numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError("VAR requires at least 2 variables")
            var_data = data[numeric_cols[:4]].copy()  # Use first 4 numeric columns
        
        self.columns = var_data.columns.tolist()
        
        # Remove any NaN values
        var_data = var_data.dropna()
        
        if len(var_data) < self.maxlags + 10:
            raise ValueError(f"Insufficient data for VAR: need at least {self.maxlags + 10} points")
        
        self.model = VAR(var_data)
        self.model = self.model.fit(maxlags=self.maxlags, ic='aic')
        self.is_fitted = True
        logger.info(f"VAR model fitted with {len(self.columns)} variables")
        return self
    
    def predict(self, data: pd.DataFrame = None, steps: int = None) -> np.ndarray:
        """Generate VAR forecast (returns Close price predictions)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        steps = steps or self.horizon
        
        forecast = self.model.forecast(steps=steps)
        
        # Convert to DataFrame for easier column access
        forecast_df = pd.DataFrame(forecast, columns=self.columns)
        
        # Return Close price predictions (or last column if Close not available)
        if 'Close' in forecast_df.columns:
            return forecast_df['Close'].values
        elif 'close' in forecast_df.columns:
            return forecast_df['close'].values
        else:
            # Return the last column (typically Close)
            return forecast_df.iloc[:, -1].values


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecaster (Simple and Exponential Moving Averages)."""
    
    def __init__(self, symbol: str, horizon: int = 1, window: int = 20, ma_type: str = "SMA"):
        """
        Initialize Moving Average forecaster.
        
        Args:
            symbol: Symbol identifier
            horizon: Forecast horizon
            window: Window size for moving average
            ma_type: Type of moving average - "SMA" (Simple) or "EMA" (Exponential)
        """
        super().__init__(symbol, horizon)
        self.window = window
        self.ma_type = ma_type.upper()
        if self.ma_type not in ["SMA", "EMA"]:
            raise ValueError("ma_type must be 'SMA' or 'EMA'")
        self.last_ma_value = None
    
    def fit(self, data: pd.Series) -> MovingAverageForecaster:
        """Fit Moving Average model (compute the last MA value)."""
        logger.info(f"Fitting {self.ma_type}(window={self.window}) for {self.symbol}")
        
        if len(data) < self.window:
            raise ValueError(f"Insufficient data: need at least {self.window} points for MA")
        
        if self.ma_type == "SMA":
            # Simple Moving Average
            self.last_ma_value = data.rolling(window=self.window).mean().iloc[-1]
        else:
            # Exponential Moving Average
            self.last_ma_value = data.ewm(span=self.window, adjust=False).mean().iloc[-1]
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.Series = None, steps: int = None) -> np.ndarray:
        """Generate Moving Average forecast (constant forecast based on last MA value)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        steps = steps or self.horizon
        
        # For moving averages, we typically forecast the last MA value forward
        # This is a naive forecast but commonly used
        if data is not None and len(data) >= self.window:
            # Recompute MA from the latest data
            if self.ma_type == "SMA":
                self.last_ma_value = data.rolling(window=self.window).mean().iloc[-1]
            else:
                self.last_ma_value = data.ewm(span=self.window, adjust=False).mean().iloc[-1]
        
        # Return constant forecast (last MA value repeated)
        return np.full(steps, self.last_ma_value)


__all__ = [
    "BaseForecaster",
    "ARIMAForecaster",
    "ExponentialSmoothingForecaster",
    "VARForecaster",
    "MovingAverageForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "TransformerForecaster",
]

