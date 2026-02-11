"""
Training script optimized for Hugging Face Spaces/Inference Endpoints.

This script can be run on Hugging Face infrastructure to train models.
It fetches data from the database (or yfinance if DB is empty) and trains models.

Usage in Hugging Face Space:
    python train_on_huggingface.py --symbol AAPL --model-type ARIMA --upload-to-hf
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.services.data_ingestion import DataIngestionService
from backend.services.multi_model_service import MultiModelService
from backend.utils.logging import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


def train_model_on_hf(
    symbol: str,
    model_type: str,
    horizon: str = "24h",
    upload_to_hf: bool = False,
    hf_repo: str = None,
    use_db: bool = True,
) -> dict:
    """
    Train a model on Hugging Face infrastructure.
    
    Args:
        symbol: Stock symbol
        model_type: Model type (ARIMA, LSTM, GRU, Transformer, etc.)
        horizon: Forecast horizon (1h, 3h, 24h, 72h)
        upload_to_hf: Whether to upload trained model to HF Hub
        hf_repo: Hugging Face repository ID (e.g., "username/repo-name")
        use_db: Whether to fetch data from database (False = use yfinance directly)
    
    Returns:
        Dict with training results
    """
    logger.info(f"Training {model_type} model for {symbol} on Hugging Face")
    
    # Get data
    if use_db:
        try:
            service = DataIngestionService()
            df = service.get_prices(symbol, limit=500)
            if df.empty:
                logger.warning(f"No data in DB for {symbol}, fetching from yfinance")
                use_db = False
        except Exception as e:
            logger.warning(f"Could not fetch from DB: {e}, using yfinance")
            use_db = False
    
    if not use_db:
        # Fetch directly from yfinance
        import yfinance as yf
        logger.info(f"Fetching data from yfinance for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        df = df.reset_index()
        df['date'] = df['Date'] if 'Date' in df.columns else df.index
    
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} records (need at least 60)")
    
    logger.info(f"Using {len(df)} records for training")
    
    # If we fetched data directly, save it to database for training service
    # MultiModelService.train_model() fetches from DB, so we need data there
    if not use_db:
        try:
            service = DataIngestionService()
            service.ingest_from_dataframe(symbol, df)
            logger.info(f"Saved {len(df)} records to database for training")
        except Exception as e:
            logger.error(f"Could not save to DB: {e}")
            logger.error("Training requires database. Please:")
            logger.error("  1. Set up MongoDB, OR")
            logger.error("  2. Run: python -m backend.scripts.populate_database --symbol AAPL")
            raise ValueError(f"Cannot train without database: {e}")
    
    # Train model (fetches from DB)
    model, model_version = MultiModelService.train_model(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
    )
    
    logger.info(f"✓ Model trained: {model_version.version_tag}")
    logger.info(f"  Artifact path: {model_version.artifact_path}")
    
    result = {
        "symbol": symbol,
        "model_type": model_type,
        "horizon": horizon,
        "model_version_id": model_version.id,
        "version_tag": model_version.version_tag,
        "artifact_path": str(model_version.artifact_path),
        "trained_at": model_version.trained_at.isoformat(),
    }
    
    # Upload to Hugging Face if requested
    if upload_to_hf:
        if not hf_repo:
            hf_repo = f"{settings.hf_username or 'username'}/forecast-models"
        
        try:
            from backend.scripts.upload_model import upload_model
            
            artifact_path = Path(model_version.artifact_path)
            commit_message = f"Train {model_type} for {symbol} on HF"
            
            url = upload_model(
                model_path=artifact_path,
                repo_id=hf_repo,
                commit_message=commit_message,
            )
            
            result["hf_repo"] = hf_repo
            result["hf_url"] = url
            logger.info(f"✓ Model uploaded to {url}")
        except Exception as e:
            logger.error(f"Failed to upload to HF: {e}")
            result["upload_error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train models on Hugging Face infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ARIMA model
  python train_on_huggingface.py --symbol AAPL --model-type ARIMA
  
  # Train LSTM and upload to HF
  python train_on_huggingface.py --symbol AAPL --model-type LSTM --upload-to-hf --hf-repo username/forecast-models
  
  # Train multiple models
  python train_on_huggingface.py --symbol AAPL --model-type ARIMA,LSTM,GRU
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol (e.g., AAPL)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type(s): ARIMA, LSTM, GRU, Transformer, ExponentialSmoothing (comma-separated for multiple)"
    )
    
    parser.add_argument(
        "--horizon",
        type=str,
        default="24h",
        choices=["1h", "3h", "24h", "72h"],
        help="Forecast horizon (default: 24h)"
    )
    
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload trained model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., username/repo-name)"
    )
    
    parser.add_argument(
        "--use-db",
        action="store_true",
        default=False,
        help="Fetch data from database (default: fetch from yfinance directly)"
    )
    
    args = parser.parse_args()
    
    # Parse model types
    model_types = [mt.strip() for mt in args.model_type.split(",")]
    
    print(f"Training {len(model_types)} model(s) for {args.symbol}...")
    print()
    
    results = []
    
    for model_type in model_types:
        try:
            print(f"Training {model_type}...")
            result = train_model_on_hf(
                symbol=args.symbol,
                model_type=model_type,
                horizon=args.horizon,
                upload_to_hf=args.upload_to_hf,
                hf_repo=args.hf_repo,
                use_db=args.use_db,
            )
            results.append(result)
            print(f"✓ {model_type} trained successfully")
            if args.upload_to_hf and "hf_url" in result:
                print(f"  Uploaded to: {result['hf_url']}")
            print()
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}", exc_info=True)
            print(f"✗ {model_type} failed: {e}")
            print()
    
    print(f"✓ Completed: {len(results)}/{len(model_types)} models trained")
    
    if len(results) < len(model_types):
        sys.exit(1)


if __name__ == "__main__":
    main()

