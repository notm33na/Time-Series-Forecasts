"""
Script to register the newly trained Transformer model in the database.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.db.unified_db import UnifiedDataStore
from backend.utils.model_registry import register_model_version
from backend.config import get_settings
import joblib

settings = get_settings()

def register_newest_transformer(symbol: str = "GOOGL"):
    """Register the newest Transformer model in the database."""
    print(f"\n{'='*70}")
    print(f"Registering Newest Transformer Model for {symbol}")
    print(f"{'='*70}\n")
    
    # Find newest model
    models_dir = settings.models_dir
    pkl_files = list(models_dir.glob(f"{symbol}_transformer_*.pkl"))
    # Filter out .weights.pkl files
    pkl_files = [f for f in pkl_files if '.weights.pkl' not in f.name]
    
    if not pkl_files:
        print(f"❌ No Transformer model found for {symbol}")
        return False
    
    # Sort by modification time (newest first)
    pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    newest_pkl = pkl_files[0]
    
    # Find corresponding .h5 file
    base_name = newest_pkl.stem
    h5_file = newest_pkl.parent / f"{base_name}.weights.h5"
    
    if not h5_file.exists():
        print(f"❌ Could not find weights file: {h5_file}")
        return False
    
    print(f"Found newest model:")
    print(f"  Metadata: {newest_pkl.name}")
    print(f"  Weights: {h5_file.name}")
    
    # Load metadata to get training info
    try:
        metadata = joblib.load(newest_pkl)
        print(f"\nModel metadata:")
        print(f"  Symbol: {metadata.get('symbol', 'N/A')}")
        print(f"  Lag window: {metadata.get('lag_window', 'N/A')}")
        print(f"  Train size: {metadata.get('train_size', 'N/A')}")
        print(f"  Test size: {metadata.get('test_size', 'N/A')}")
        if 'metrics' in metadata:
            print(f"  Metrics: MAE={metadata['metrics'].get('mae', 'N/A'):.4f}, "
                  f"RMSE={metadata['metrics'].get('rmse', 'N/A'):.4f}, "
                  f"MAPE={metadata['metrics'].get('mape', 'N/A'):.2f}%")
        
        # Check scaler range
        if 'scaler' in metadata:
            scaler = metadata['scaler']
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                print(f"  Scaler range: [${scaler.data_min_[0]:.2f}, ${scaler.data_max_[0]:.2f}]")
    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")
    
    # Register in database
    print(f"\nRegistering model in database...")
    try:
        model_version = register_model_version(
            symbol=symbol,
            model_type="Transformer",
            artifact_path=h5_file,
            source="local",
            horizon=1,  # Default horizon, models are flexible
        )
        print(f"✓ Model registered successfully!")
        print(f"  Version ID: {model_version.get('id', 'N/A')}")
        print(f"  Version tag: {model_version.get('version_tag', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register Transformer model in database")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Symbol to register")
    
    args = parser.parse_args()
    
    success = register_newest_transformer(args.symbol)
    sys.exit(0 if success else 1)

