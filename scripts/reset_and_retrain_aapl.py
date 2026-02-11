"""
Script to delete all AAPL artifacts, data, and retrain all models.

This script will:
1. Delete all AAPL model files from artifacts directory
2. Delete all AAPL data from MongoDB (prices, models, forecasts, metrics)
3. Retrain all models for AAPL

Usage:
    python scripts/reset_and_retrain_aapl.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_settings
from backend.db.unified_db import get_mongo_db
from backend.utils.logging import get_logger

logger = get_logger(__name__)

SYMBOL = "AAPL"


def delete_aapl_artifacts():
    """Delete all AAPL model files from the artifacts directory."""
    settings = get_settings()
    artifacts_dir = settings.models_dir
    
    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory does not exist: {artifacts_dir}")
        return 0
    
    # Find all AAPL model files
    model_extensions = ['.pkl', '.h5', '.joblib', '.weights.h5', '.weights.pkl']
    aapl_files = []
    
    for ext in model_extensions:
        # Match files that start with AAPL_ or contain AAPL in the name
        pattern = f"*{SYMBOL}*{ext}"
        aapl_files.extend(list(artifacts_dir.glob(pattern)))
        # Also check for lowercase
        pattern_lower = f"*{SYMBOL.lower()}*{ext}"
        aapl_files.extend(list(artifacts_dir.glob(pattern_lower)))
    
    if not aapl_files:
        print(f"✓ No {SYMBOL} model files found in {artifacts_dir}")
        return 0
    
    print(f"\n📁 Found {len(aapl_files)} {SYMBOL} model files in {artifacts_dir}")
    
    # Delete each file
    deleted_count = 0
    for file_path in aapl_files:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"   ✓ Deleted: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to delete {file_path.name}: {e}")
    
    print(f"\n✅ Deleted {deleted_count}/{len(aapl_files)} {SYMBOL} files from filesystem")
    return deleted_count


def delete_aapl_mongodb_data():
    """Delete all AAPL-related documents from MongoDB."""
    try:
        mongo_db = get_mongo_db()
    except Exception as e:
        print(f"⚠️  Could not connect to MongoDB: {e}")
        return 0
    
    total_deleted = 0
    
    # Collections to clean
    collections_to_clean = {
        'prices': f'{SYMBOL} price data',
        'models': f'{SYMBOL} model metadata',
        'model_versions': f'{SYMBOL} model versions',
        'metrics': f'{SYMBOL} evaluation metrics',
        'forecasts': f'{SYMBOL} forecast predictions',
    }
    
    print(f"\n🗄️  Deleting {SYMBOL} data from MongoDB...")
    
    for collection_name, description in collections_to_clean.items():
        try:
            collection = getattr(mongo_db, collection_name)
            
            # Count documents for this symbol
            count = collection.count_documents({"symbol": SYMBOL})
            
            if count == 0:
                print(f"   ✓ {description}: No documents found")
                continue
            
            # Delete all documents for this symbol
            result = collection.delete_many({"symbol": SYMBOL})
            deleted = result.deleted_count
            total_deleted += deleted
            print(f"   ✓ {description}: Deleted {deleted} document(s)")
            
        except AttributeError:
            # Collection doesn't exist
            print(f"   ⚠️  {description}: Collection '{collection_name}' does not exist")
        except Exception as e:
            print(f"   ⚠️  {description}: Error - {e}")
    
    print(f"\n✅ Deleted {total_deleted} {SYMBOL} document(s) from MongoDB")
    return total_deleted


def retrain_all_models():
    """Retrain all models for AAPL using the train_all_models script."""
    import subprocess
    
    notebooks_dir = project_root / "notebooks"
    train_script = notebooks_dir / "train_all_models.py"
    
    if not train_script.exists():
        print(f"⚠️  Training script not found: {train_script}")
        print("   Please ensure notebooks/train_all_models.py exists")
        return False
    
    print(f"\n🚀 Starting training for all models on {SYMBOL}...")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        str(train_script),
        "--symbols", SYMBOL,
        "--period", "max"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=notebooks_dir,
            check=True
        )
        
        print("\n✅ All models trained successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to reset and retrain AAPL models."""
    print("=" * 70)
    print(f"RESET AND RETRAIN {SYMBOL} MODELS")
    print("=" * 70)
    print("\n⚠️  WARNING: This will permanently delete:")
    print(f"   • All {SYMBOL} model files from filesystem (artifacts directory)")
    print(f"   • All {SYMBOL} data from MongoDB (prices, models, forecasts, metrics)")
    print(f"   • Then retrain all models for {SYMBOL}")
    print("\nThis action is IRREVERSIBLE!")
    
    # Ask for confirmation
    response = input(f"\nAre you sure you want to reset and retrain {SYMBOL}? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Operation cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("STEP 1: DELETING AAPL ARTIFACTS...")
    print("=" * 70)
    files_deleted = delete_aapl_artifacts()
    
    print("\n" + "=" * 70)
    print("STEP 2: DELETING AAPL DATA FROM MONGODB...")
    print("=" * 70)
    docs_deleted = delete_aapl_mongodb_data()
    
    print("\n" + "=" * 70)
    print("STEP 3: RETRAINING ALL MODELS...")
    print("=" * 70)
    training_success = retrain_all_models()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Files deleted from filesystem: {files_deleted}")
    print(f"✅ Documents deleted from MongoDB: {docs_deleted}")
    if training_success:
        print(f"✅ All models retrained successfully for {SYMBOL}")
    else:
        print(f"❌ Model training failed for {SYMBOL}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

