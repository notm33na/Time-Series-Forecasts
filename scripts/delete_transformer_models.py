"""
Script to delete Transformer model files and MongoDB records for a symbol.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.db.unified_db import UnifiedDataStore
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def delete_transformer_files(symbol: str):
    """Delete Transformer model files from filesystem."""
    settings = get_settings()
    artifacts_dir = settings.models_dir
    
    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory does not exist: {artifacts_dir}")
        return 0
    
    # Find all Transformer model files for this symbol
    model_patterns = [
        f"{symbol}_transformer_*.pkl",
        f"{symbol}_transformer_*.h5",
        f"{symbol}_transformer_*.weights.h5",
        f"{symbol}_transformer_*.weights.pkl",
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(list(artifacts_dir.glob(pattern)))
    
    if not model_files:
        print(f"✓ No Transformer model files found for {symbol} in {artifacts_dir}")
        return 0
    
    print(f"\n📁 Found {len(model_files)} Transformer model files for {symbol}")
    
    # Delete each file
    deleted_count = 0
    for file_path in model_files:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"   ✓ Deleted: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to delete {file_path.name}: {e}")
    
    return deleted_count


def delete_transformer_mongodb(symbol: str):
    """Delete Transformer model versions from MongoDB."""
    try:
        store = UnifiedDataStore()
    except Exception as e:
        print(f"⚠️  Could not connect to MongoDB: {e}")
        return 0
    
    # Delete model versions
    result = store.mongo_db.model_versions.delete_many({
        "symbol": symbol,
        "model_type": "Transformer"
    })
    
    deleted_count = result.deleted_count
    print(f"\n🗄️  Deleted {deleted_count} Transformer model version(s) from MongoDB for {symbol}")
    
    # Also delete related forecasts (optional - you might want to keep historical forecasts)
    forecast_result = store.mongo_db.forecasts.delete_many({
        "symbol": symbol,
        "model_version_id": {"$in": []}  # This won't match anything, but we could enhance this
    })
    
    # Actually, let's find model version IDs first, then delete forecasts
    model_versions = store.mongo_db.model_versions.find({
        "symbol": symbol,
        "model_type": "Transformer"
    })
    model_version_ids = [str(mv["_id"]) for mv in model_versions]
    
    if model_version_ids:
        forecast_result = store.mongo_db.forecasts.delete_many({
            "symbol": symbol,
            "model_version_id": {"$in": model_version_ids}
        })
        print(f"   ✓ Deleted {forecast_result.deleted_count} forecast record(s) for Transformer models")
    
    return deleted_count


def main():
    """Main function to delete Transformer models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete Transformer models for a symbol")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--skip-filesystem", action="store_true", help="Skip filesystem deletion")
    parser.add_argument("--skip-mongodb", action="store_true", help="Skip MongoDB deletion")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Deleting Transformer Models for {args.symbol}")
    print(f"{'='*70}\n")
    
    total_deleted = 0
    
    # Delete from filesystem
    if not args.skip_filesystem:
        filesystem_count = delete_transformer_files(args.symbol)
        total_deleted += filesystem_count
    
    # Delete from MongoDB
    if not args.skip_mongodb:
        mongodb_count = delete_transformer_mongodb(args.symbol)
        total_deleted += mongodb_count
    
    print(f"\n{'='*70}")
    print(f"✅ Deletion Complete: {total_deleted} item(s) deleted")
    print(f"{'='*70}\n")
    
    return total_deleted


if __name__ == "__main__":
    main()

