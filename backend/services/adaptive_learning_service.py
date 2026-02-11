"""
Adaptive & Continuous Learning Service

Implements:
- Online learning / incremental updates
- Rolling-window retraining
- LSTM/GRU/Transformer fine-tuning
- Dynamic ensemble reweighting based on recent accuracy
- Model version tracking with performance comparison
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..models.adaptive_ensemble import AdaptiveEnsemble
from ..models.forecasting_models import (
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
)
from ..services.model_loader import ModelLoader
from ..services.data_ingestion import DataIngestionService
from ..services.evaluation_service import EvaluationService
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AdaptiveLearningService:
    """
    Service for adaptive and continuous learning.
    Supports online learning, rolling-window retraining, fine-tuning, and dynamic ensembles.
    """
    
    def __init__(self):
        self.store = UnifiedDataStore()
        self.data_service = DataIngestionService()
        self.eval_service = EvaluationService()
    
    def online_update_model(
        self,
        symbol: str,
        model_type: str,
        new_data: pd.DataFrame,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ) -> Tuple[object, Dict[str, Any]]:
        """
        Perform online/incremental learning update on an existing model.
        Updates the model with new data without full retraining.
        
        Args:
            symbol: Stock symbol
            model_type: Model type (LSTM, GRU, Transformer)
            new_data: New data points to learn from
            learning_rate: Learning rate for incremental update
            batch_size: Batch size for update
        
        Returns:
            (updated_model, model_version_dict)
        """
        logger.info(f"Online update for {symbol}/{model_type} with {len(new_data)} new points")
        
        # Load latest model
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, model_type)
        except ValueError as e:
            raise ValueError(f"No existing model found for {symbol}/{model_type}: {e}")
        
        # Prepare new data
        if model_type == "LSTM":
            # LSTM needs DataFrame with features
            if 'Close' not in new_data.columns and 'close' in new_data.columns:
                new_data = new_data.rename(columns={'close': 'Close'})
            X_new = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        else:
            # GRU/Transformer use Series
            prices = new_data['Close'] if 'Close' in new_data.columns else new_data['close']
            X_new = prices.values
        
        # For neural models, perform incremental training
        if model_type in ["LSTM", "GRU", "Transformer"]:
            if hasattr(model, 'model') and hasattr(model.model, 'fit'):
                # Get scaler from model metadata
                scaler = getattr(model, 'scaler', None)
                if scaler is None:
                    # Try to load from metadata
                    metadata_path = Path(model_version.get('artifact_path', '')).with_suffix('.pkl')
                    if metadata_path.exists():
                        metadata = joblib.load(metadata_path)
                        scaler = metadata.get('scaler')
                
                if scaler:
                    # Scale new data
                    if model_type == "LSTM":
                        X_scaled = scaler.transform(X_new)
                        # Reshape for LSTM: (samples, 1, features)
                        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        # For online learning, we need targets (next Close_diff)
                        # Use Close_diff as target
                        close_col_idx = list(new_data.columns).index('Close') if 'Close' in new_data.columns else 0
                        y_new = np.diff(X_new[:, close_col_idx])
                        X_scaled = X_scaled[:-1]  # Remove last sample (no target)
                    else:
                        X_scaled = scaler.transform(X_new.reshape(-1, 1))
                        y_new = X_scaled[1:].flatten()
                        X_scaled = X_scaled[:-1].reshape(-1, 1, 1)
                    
                    # Perform incremental update with small learning rate
                    model.model.fit(
                        X_scaled, y_new,
                        epochs=1,
                        batch_size=min(batch_size, len(X_scaled)),
                        verbose=0,
                        shuffle=False
                    )
                    
                    logger.info(f"✓ Online update completed for {model_type}")
                else:
                    logger.warning(f"No scaler found, skipping online update")
            else:
                logger.warning(f"Model {model_type} does not support online updates")
        else:
            # For non-neural models, use rolling-window retraining instead
            logger.info(f"{model_type} does not support online updates, using rolling-window retraining")
            return self.rolling_window_retrain(symbol, model_type, window_size=len(new_data) + 100)
        
        # Save updated model
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{model_type}_{timestamp}_online"
        
        artifact_path = model.save(timestamp) if hasattr(model, 'save') else None
        if not artifact_path:
            artifact_path = settings.models_dir / f"{symbol}_{model_type}_{timestamp}_online.pkl"
            joblib.dump(model, artifact_path)
        
        # Save new version
        model_version_id = self.store.save_model_version(
            symbol=symbol,
            model_type=model_type,
            horizon=model_version.get('horizon', 24),
            version_tag=version_tag,
            artifact_path=str(artifact_path),
            source="local",
            training_info={
                "update_type": "online",
                "learning_rate": learning_rate,
                "new_data_points": len(new_data),
                "previous_version": model_version.get('id'),
            },
            notes=f"Online update with {len(new_data)} new points"
        )
        
        new_model_version = self.store.get_model_version(model_version_id)
        logger.info(f"Saved online-updated model version: {version_tag}")
        
        return model, new_model_version
    
    def rolling_window_retrain(
        self,
        symbol: str,
        model_type: str,
        window_size: int = 200,
        horizon: Optional[int] = None,
    ) -> Tuple[object, Dict[str, Any]]:
        """
        Retrain model using a rolling window of recent data.
        
        Args:
            symbol: Stock symbol
            model_type: Model type
            window_size: Number of recent data points to use
            horizon: Forecast horizon (uses model default if None)
        
        Returns:
            (retrained_model, model_version_dict)
        """
        logger.info(f"Rolling-window retraining for {symbol}/{model_type} (window={window_size})")
        
        # Get recent data
        df = self.data_service.get_prices(symbol, limit=window_size + 100)
        if len(df) < window_size:
            raise ValueError(f"Insufficient data: need {window_size} points, got {len(df)}")
        
        # Use most recent window_size points
        recent_data = df.tail(window_size).copy()
        
        # Determine horizon
        if horizon is None:
            # Try to get from latest model
            try:
                _, latest_version = ModelLoader.load_latest_model(symbol, model_type)
                horizon = latest_version.get('horizon', 24)
            except:
                horizon = 24
        
        # Create and train model
        if model_type == "LSTM":
            model = LSTMForecaster(symbol, horizon, lag_window=1)
            model.fit(recent_data)
        elif model_type == "GRU":
            model = GRUForecaster(symbol, horizon)
            prices = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']
            model.fit(prices)
        elif model_type == "Transformer":
            model = TransformerForecaster(symbol, horizon)
            prices = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']
            model.fit(prices)
        elif model_type == "ARIMA":
            model = ARIMAForecaster(symbol, horizon)
            prices = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']
            model.fit(prices)
        elif model_type == "ExponentialSmoothing":
            model = ExponentialSmoothingForecaster(symbol, horizon)
            prices = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']
            model.fit(prices)
        else:
            raise ValueError(f"Unsupported model type for rolling-window retraining: {model_type}")
        
        # Save model
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{model_type}_{timestamp}_rolling"
        
        artifact_path = model.save(timestamp) if hasattr(model, 'save') else None
        if not artifact_path:
            artifact_path = settings.models_dir / f"{symbol}_{model_type}_{timestamp}_rolling.pkl"
            joblib.dump(model, artifact_path)
        
        # Save version
        model_version_id = self.store.save_model_version(
            symbol=symbol,
            model_type=model_type,
            horizon=horizon,
            version_tag=version_tag,
            artifact_path=str(artifact_path),
            source="local",
            training_info={
                "update_type": "rolling_window",
                "window_size": window_size,
                "training_samples": len(recent_data),
            },
            notes=f"Rolling-window retraining with {window_size} points"
        )
        
        model_version = self.store.get_model_version(model_version_id)
        logger.info(f"✓ Rolling-window retraining completed: {version_tag}")
        
        return model, model_version
    
    def fine_tune_neural_model(
        self,
        symbol: str,
        model_type: Literal["LSTM", "GRU", "Transformer"],
        new_data_window: int = 50,
        epochs: int = 5,
        learning_rate: float = 0.0001,
        freeze_base: bool = False,
    ) -> Tuple[object, Dict[str, Any]]:
        """
        Fine-tune a neural model with recent data.
        Optionally freezes base layers for transfer learning.
        
        Args:
            symbol: Stock symbol
            model_type: Neural model type
            new_data_window: Number of recent points for fine-tuning
            epochs: Fine-tuning epochs
            learning_rate: Learning rate (typically lower than initial training)
            freeze_base: If True, freeze base layers and only train top layers
        
        Returns:
            (fine_tuned_model, model_version_dict)
        """
        logger.info(f"Fine-tuning {model_type} for {symbol} (window={new_data_window}, epochs={epochs})")
        
        # Load latest model
        try:
            model, model_version = ModelLoader.load_latest_model(symbol, model_type)
        except ValueError as e:
            raise ValueError(f"No existing model found for {symbol}/{model_type}: {e}")
        
        # Get recent data
        df = self.data_service.get_prices(symbol, limit=new_data_window + 100)
        if len(df) < new_data_window:
            raise ValueError(f"Insufficient data: need {new_data_window} points, got {len(df)}")
        
        recent_data = df.tail(new_data_window).copy()
        horizon = model_version.get('horizon', 24)
        
        # Prepare data
        if model_type == "LSTM":
            # LSTM needs DataFrame
            if 'Close' not in recent_data.columns and 'close' in recent_data.columns:
                recent_data = recent_data.rename(columns={'close': 'Close'})
            # Get scaler
            scaler = getattr(model, 'scaler', None)
            if scaler:
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                X = recent_data[feature_cols].values
                X_scaled = scaler.transform(X)
                X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                # Calculate Close_diff as target
                close_col_idx = feature_cols.index('Close')
                y = np.diff(X_scaled[:, 0, close_col_idx])  # Close_diff
                X_scaled = X_scaled[:-1]
        else:
            # GRU/Transformer use Series
            prices = recent_data['Close'] if 'Close' in recent_data.columns else recent_data['close']
            scaler = getattr(model, 'scaler', None)
            if scaler:
                prices_scaled = scaler.transform(prices.values.reshape(-1, 1)).flatten()
                # Create sequences
                lag_window = getattr(model, 'lag_window', 60)
                X_scaled, y = [], []
                for i in range(lag_window, len(prices_scaled)):
                    X_scaled.append(prices_scaled[i-lag_window:i])
                    y.append(prices_scaled[i])
                X_scaled = np.array(X_scaled).reshape(-1, lag_window, 1)
                y = np.array(y)
        
        # Fine-tune model
        if hasattr(model, 'model') and hasattr(model.model, 'fit'):
            # Freeze base layers if requested
            if freeze_base and hasattr(model.model, 'layers'):
                # Freeze all but last layer
                for layer in model.model.layers[:-1]:
                    layer.trainable = False
                logger.info("Froze base layers for fine-tuning")
            
            # Compile with lower learning rate
            if hasattr(model.model, 'compile'):
                import tensorflow as tf
                model.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='mse'
                )
            
            # Fine-tune
            model.model.fit(
                X_scaled, y,
                epochs=epochs,
                batch_size=min(32, len(X_scaled)),
                verbose=1,
                shuffle=False
            )
            
            logger.info(f"✓ Fine-tuning completed for {model_type}")
        else:
            raise ValueError(f"Model {model_type} does not support fine-tuning")
        
        # Save fine-tuned model
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        version_tag = f"{symbol}_{model_type}_{timestamp}_finetuned"
        
        artifact_path = model.save(timestamp) if hasattr(model, 'save') else None
        if not artifact_path:
            artifact_path = settings.models_dir / f"{symbol}_{model_type}_{timestamp}_finetuned.pkl"
            joblib.dump(model, artifact_path)
        
        # Save version
        model_version_id = self.store.save_model_version(
            symbol=symbol,
            model_type=model_type,
            horizon=horizon,
            version_tag=version_tag,
            artifact_path=str(artifact_path),
            source="local",
            training_info={
                "update_type": "fine_tuning",
                "epochs": epochs,
                "learning_rate": learning_rate,
                "freeze_base": freeze_base,
                "new_data_points": len(recent_data),
                "previous_version": model_version.get('id'),
            },
            notes=f"Fine-tuned with {new_data_window} points, {epochs} epochs"
        )
        
        new_model_version = self.store.get_model_version(model_version_id)
        logger.info(f"Saved fine-tuned model version: {version_tag}")
        
        return model, new_model_version
    
    def update_ensemble_weights(
        self,
        symbol: str,
        horizon: int = 24,
        evaluation_window: int = 50,
    ) -> Tuple[AdaptiveEnsemble, Dict[str, Any]]:
        """
        Update ensemble weights based on recent model performance.
        Evaluates each model on recent data and reweights accordingly.
        
        Args:
            symbol: Stock symbol
            horizon: Forecast horizon
            evaluation_window: Number of recent points to evaluate on
        
        Returns:
            (updated_ensemble, performance_summary)
        """
        logger.info(f"Updating ensemble weights for {symbol} (window={evaluation_window})")
        
        # Load ensemble
        try:
            ensemble, ensemble_version = ModelLoader.load_latest_model(symbol, "Ensemble")
        except ValueError:
            # Create new ensemble
            from ..models.adaptive_ensemble import create_default_ensemble
            ensemble = create_default_ensemble(symbol, horizon)
            ensemble_version = None
        
        # Get recent data for evaluation
        df = self.data_service.get_prices(symbol, limit=evaluation_window + horizon + 100)
        if len(df) < evaluation_window + horizon:
            raise ValueError(f"Insufficient data for evaluation: need {evaluation_window + horizon} points")
        
        prices = df['Close'] if 'Close' in df.columns else df['close']
        
        # Evaluate each model in ensemble
        model_errors = {}
        
        for i, (forecaster, name) in enumerate(zip(ensemble.forecasters, ensemble.model_names)):
            if not forecaster.is_fitted:
                continue
            
            errors = []
            
            # Evaluate on rolling window
            for j in range(evaluation_window):
                train_end = len(prices) - evaluation_window + j
                test_start = train_end
                test_end = min(test_start + horizon, len(prices))
                
                if test_end <= test_start:
                    continue
                
                # Get training data up to this point
                train_data = prices[:train_end]
                test_data = prices[test_start:test_end]
                
                try:
                    # Make prediction
                    if name == "LSTM":
                        # LSTM needs DataFrame
                        train_df = df.iloc[:train_end]
                        pred = forecaster.predict(train_df.tail(1), steps=len(test_data))
                    else:
                        pred = forecaster.predict(train_data, steps=len(test_data))
                    
                    # Calculate error
                    if len(pred) == len(test_data):
                        error = np.mean(np.abs(pred - test_data.values))
                        errors.append(error)
                except Exception as e:
                    logger.warning(f"Evaluation failed for {name} at step {j}: {e}")
                    continue
            
            if errors:
                model_errors[name] = np.mean(errors)
                # Update ensemble error history
                ensemble.update_errors(
                    actual=0,  # Not used when updating by model
                    predicted=0,  # Not used when updating by model
                    model_index=i
                )
                # Manually set recent errors
                ensemble.recent_errors[i] = errors[-min(20, len(errors)):]  # Keep last 20
                logger.info(f"{name}: avg error = {model_errors[name]:.4f}")
        
        # Update weights based on errors
        ensemble._update_weights()
        
        # Get performance summary
        performance = ensemble.get_model_performance()
        
        # Save updated ensemble if it was loaded
        if ensemble_version:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            version_tag = f"{symbol}_ensemble_{timestamp}_updated"
            
            # Save ensemble metadata (not full model)
            artifact_path = settings.models_dir / f"{symbol}_ensemble_{timestamp}_metadata.pkl"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "weights": ensemble.weights.tolist(),
                "model_names": ensemble.model_names,
                "performance": performance,
            }, artifact_path)
            
            model_version_id = self.store.save_model_version(
                symbol=symbol,
                model_type="Ensemble",
                horizon=horizon,
                version_tag=version_tag,
                artifact_path=str(artifact_path),
                source="local",
                training_info={
                    "update_type": "weight_update",
                    "evaluation_window": evaluation_window,
                    "weights": ensemble.weights.tolist(),
                    "performance": performance,
                },
                notes="Ensemble weights updated based on recent performance"
            )
            
            logger.info(f"Saved updated ensemble: {version_tag}")
        
        logger.info(f"✓ Ensemble weights updated")
        logger.info(f"Performance: {performance}")
        
        return ensemble, performance
    
    def track_model_performance(
        self,
        symbol: str,
        model_type: str,
        model_version_id: str,
        actual_values: np.ndarray,
        predicted_values: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Track and save model performance metrics.
        Compares with previous versions to detect performance changes.
        
        Args:
            symbol: Stock symbol
            model_type: Model type
            model_version_id: Model version ID
            actual_values: Actual values
            predicted_values: Predicted values
        
        Returns:
            Performance metrics dict
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
        }
        
        # Save metrics
        self.store.save_metrics(
            symbol=symbol,
            model_type=model_type,
            metrics=metrics,
            model_version_id=model_version_id,
        )
        
        # Get previous version metrics for comparison
        try:
            model_version = self.store.get_model_version(model_version_id)
            previous_versions = list(
                self.store.mongo_db.model_versions.find({
                    "symbol": symbol,
                    "model_type": model_type,
                    "trained_at": {"$lt": model_version.get("trained_at")}
                }).sort("trained_at", -1).limit(1)
            )
            
            if previous_versions:
                prev_id = str(previous_versions[0]["_id"])
                prev_metrics_list = list(
                    self.store.mongo_db.metrics.find({"model_version_id": prev_id})
                    .sort("evaluated_at", -1)
                    .limit(1)
                )
                
                if prev_metrics_list:
                    prev_metrics = prev_metrics_list[0].get("metrics", {})
                    comparison = {
                        "mae_change": ((mae - prev_metrics.get("mae", 0)) / prev_metrics.get("mae", 1)) * 100 if prev_metrics.get("mae") else None,
                        "rmse_change": ((rmse - prev_metrics.get("rmse", 0)) / prev_metrics.get("rmse", 1)) * 100 if prev_metrics.get("rmse") else None,
                        "mape_change": ((mape - prev_metrics.get("mape", 0)) / prev_metrics.get("mape", 1)) * 100 if prev_metrics.get("mape") else None,
                    }
                    metrics["comparison"] = comparison
                    
                    logger.info(f"Performance comparison: MAE {comparison['mae_change']:+.2f}%, "
                              f"RMSE {comparison['rmse_change']:+.2f}%, "
                              f"MAPE {comparison['mape_change']:+.2f}%")
        except Exception as e:
            logger.warning(f"Could not compare with previous version: {e}")
        
        logger.info(f"Tracked performance for {symbol}/{model_type}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def get_model_version_comparison(
        self,
        symbol: str,
        model_type: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get performance comparison across multiple model versions.
        
        Args:
            symbol: Stock symbol
            model_type: Model type
            limit: Maximum number of versions to compare
        
        Returns:
            List of version performance dicts
        """
        versions = list(
            self.store.mongo_db.model_versions.find({
                "symbol": symbol,
                "model_type": model_type,
            }).sort("trained_at", -1).limit(limit)
        )
        
        comparison = []
        
        for version in versions:
            version_id = str(version["_id"])
            
            # Get latest metrics for this version
            metrics_list = list(
                self.store.mongo_db.metrics.find({"model_version_id": version_id})
                .sort("evaluated_at", -1)
                .limit(1)
            )
            
            metrics = metrics_list[0].get("metrics", {}) if metrics_list else {}
            
            comparison.append({
                "version_id": version_id,
                "version_tag": version.get("version_tag"),
                "trained_at": version.get("trained_at"),
                "training_info": version.get("training_info", {}),
                "metrics": metrics,
            })
        
        return comparison


__all__ = ["AdaptiveLearningService"]

