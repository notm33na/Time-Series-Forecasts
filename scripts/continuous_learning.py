"""
Continuous Learning Automation Script

Automatically updates models as new data arrives using:
- Online learning / incremental updates
- Rolling-window retraining
- Fine-tuning
- Dynamic ensemble reweighting
- Performance tracking

Usage:
    python continuous_learning.py --symbol AAPL --mode auto
    python continuous_learning.py --symbol AAPL --mode rolling --window 200
    python continuous_learning.py --symbol AAPL --mode fine-tune --epochs 10
    python continuous_learning.py --symbol AAPL --mode ensemble --update-weights
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from backend.services.adaptive_learning_service import AdaptiveLearningService
from backend.services.data_ingestion import DataIngestionService
from backend.services.model_loader import ModelLoader
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def continuous_learning_auto(symbol: str, check_interval_hours: int = 24):
    """
    Automatic continuous learning: monitors for new data and updates models.
    
    Args:
        symbol: Stock symbol
        check_interval_hours: How often to check for new data (hours)
    """
    logger.info(f"Starting automatic continuous learning for {symbol}")
    
    learning_service = AdaptiveLearningService()
    data_service = DataIngestionService()
    
    # Get latest model training time
    try:
        _, latest_version = ModelLoader.load_latest_model(symbol, "LSTM")
        last_training_time = latest_version.get("trained_at")
        if isinstance(last_training_time, str):
            from dateutil import parser
            last_training_time = parser.parse(last_training_time)
    except:
        logger.warning(f"No existing model found for {symbol}, will train initial model")
        last_training_time = None
    
    # Check for new data
    df = data_service.get_prices(symbol, limit=1000)
    if len(df) == 0:
        logger.error(f"No data available for {symbol}")
        return
    
    # Determine how much new data we have
    if last_training_time:
        # Count new points since last training
        df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
        new_data = df[df['date'] > last_training_time]
        new_points = len(new_data)
        
        logger.info(f"Found {new_points} new data points since last training")
        
        if new_points >= 10:
            # Enough new data for online update
            logger.info("Performing online update...")
            try:
                model, version = learning_service.online_update_model(
                    symbol=symbol,
                    model_type="LSTM",
                    new_data=new_data,
                )
                logger.info(f"✓ Online update completed: {version.get('version_tag')}")
            except Exception as e:
                logger.error(f"Online update failed: {e}")
                logger.info("Falling back to rolling-window retraining...")
                try:
                    model, version = learning_service.rolling_window_retrain(
                        symbol=symbol,
                        model_type="LSTM",
                        window_size=200,
                    )
                    logger.info(f"✓ Rolling-window retraining completed: {version.get('version_tag')}")
                except Exception as e2:
                    logger.error(f"Rolling-window retraining also failed: {e2}")
        
        elif new_points >= 50:
            # Enough for fine-tuning
            logger.info("Performing fine-tuning...")
            try:
                model, version = learning_service.fine_tune_neural_model(
                    symbol=symbol,
                    model_type="LSTM",
                    new_data_window=min(new_points, 100),
                    epochs=5,
                )
                logger.info(f"✓ Fine-tuning completed: {version.get('version_tag')}")
            except Exception as e:
                logger.error(f"Fine-tuning failed: {e}")
    else:
        # No existing model, need initial training
        logger.info("No existing model found. Please train initial model first.")
        logger.info(f"Run: python notebooks/train_lstm.py --symbol {symbol}")


def continuous_learning_rolling(symbol: str, model_type: str, window_size: int = 200):
    """Rolling-window retraining."""
    logger.info(f"Rolling-window retraining for {symbol}/{model_type} (window={window_size})")
    
    learning_service = AdaptiveLearningService()
    
    try:
        model, version = learning_service.rolling_window_retrain(
            symbol=symbol,
            model_type=model_type,
            window_size=window_size,
        )
        logger.info(f"✓ Retraining completed: {version.get('version_tag')}")
        logger.info(f"   Artifact: {version.get('artifact_path')}")
        return model, version
    except Exception as e:
        logger.error(f"Rolling-window retraining failed: {e}")
        raise


def continuous_learning_finetune(
    symbol: str,
    model_type: str,
    new_data_window: int = 50,
    epochs: int = 5,
):
    """Fine-tune neural model."""
    logger.info(f"Fine-tuning {symbol}/{model_type} (window={new_data_window}, epochs={epochs})")
    
    learning_service = AdaptiveLearningService()
    
    try:
        model, version = learning_service.fine_tune_neural_model(
            symbol=symbol,
            model_type=model_type,
            new_data_window=new_data_window,
            epochs=epochs,
        )
        logger.info(f"✓ Fine-tuning completed: {version.get('version_tag')}")
        logger.info(f"   Artifact: {version.get('artifact_path')}")
        return model, version
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


def continuous_learning_ensemble(symbol: str, update_weights: bool = True):
    """Update ensemble weights based on recent performance."""
    logger.info(f"Updating ensemble for {symbol}")
    
    learning_service = AdaptiveLearningService()
    
    try:
        ensemble, performance = learning_service.update_ensemble_weights(
            symbol=symbol,
            horizon=24,
            evaluation_window=50,
        )
        logger.info(f"✓ Ensemble weights updated")
        logger.info(f"Performance summary:")
        for model_name, perf in performance.items():
            logger.info(f"   {model_name}: weight={perf['weight']:.3f}, "
                       f"MAE={perf.get('recent_mae', 'N/A')}, "
                       f"trend={perf.get('trend', 'N/A')}")
        return ensemble, performance
    except Exception as e:
        logger.error(f"Ensemble update failed: {e}")
        raise


def compare_model_versions(symbol: str, model_type: str, limit: int = 10):
    """Compare performance across multiple model versions."""
    logger.info(f"Comparing model versions for {symbol}/{model_type}")
    
    learning_service = AdaptiveLearningService()
    
    comparison = learning_service.get_model_version_comparison(
        symbol=symbol,
        model_type=model_type,
        limit=limit,
    )
    
    print(f"\n{'='*80}")
    print(f"Model Version Comparison: {symbol}/{model_type}")
    print(f"{'='*80}")
    print(f"{'Version':<30} {'Trained At':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print(f"{'-'*80}")
    
    for version in comparison:
        metrics = version.get('metrics', {})
        trained_at = version.get('trained_at', 'N/A')
        if isinstance(trained_at, str):
            trained_at = trained_at[:19]  # Truncate to date-time
        
        print(f"{version.get('version_tag', 'N/A'):<30} "
              f"{str(trained_at):<20} "
              f"{metrics.get('mae', 'N/A'):<10.4f} "
              f"{metrics.get('rmse', 'N/A'):<10.4f} "
              f"{metrics.get('mape', 'N/A'):<10.2f}")
    
    print(f"{'='*80}\n")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous Learning Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic continuous learning (monitors and updates)
  python continuous_learning.py --symbol AAPL --mode auto

  # Rolling-window retraining
  python continuous_learning.py --symbol AAPL --mode rolling --model-type LSTM --window 200

  # Fine-tune neural model
  python continuous_learning.py --symbol AAPL --mode fine-tune --model-type LSTM --epochs 10

  # Update ensemble weights
  python continuous_learning.py --symbol AAPL --mode ensemble --update-weights

  # Compare model versions
  python continuous_learning.py --symbol AAPL --mode compare --model-type LSTM --limit 10
        """
    )
    
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "rolling", "fine-tune", "ensemble", "compare"],
        default="auto",
        help="Learning mode"
    )
    parser.add_argument("--model-type", type=str, default="LSTM",
                       choices=["LSTM", "GRU", "Transformer", "ARIMA", "ExponentialSmoothing"],
                       help="Model type (for rolling/fine-tune modes)")
    parser.add_argument("--window", type=int, default=200,
                       help="Window size for rolling-window retraining")
    parser.add_argument("--new-data-window", type=int, default=50,
                       help="New data window for fine-tuning")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Epochs for fine-tuning")
    parser.add_argument("--update-weights", action="store_true",
                       help="Update ensemble weights (for ensemble mode)")
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of versions to compare (for compare mode)")
    parser.add_argument("--check-interval", type=int, default=24,
                       help="Check interval in hours (for auto mode)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "auto":
            continuous_learning_auto(args.symbol, args.check_interval)
        elif args.mode == "rolling":
            continuous_learning_rolling(args.symbol, args.model_type, args.window)
        elif args.mode == "fine-tune":
            continuous_learning_finetune(
                args.symbol,
                args.model_type,
                args.new_data_window,
                args.epochs,
            )
        elif args.mode == "ensemble":
            continuous_learning_ensemble(args.symbol, args.update_weights)
        elif args.mode == "compare":
            compare_model_versions(args.symbol, args.model_type, args.limit)
        
        print("\n✅ Continuous learning operation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Continuous learning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

