"""
Script to delete all model data from filesystem and MongoDB.

This script will:
1. Delete all model files from the artifacts directory
2. Delete all model-related documents from MongoDB collections:
   - models / model_versions
   - metrics
   - forecasts

WARNING: This action is irreversible!
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


def delete_filesystem_models():
    """Delete all model files from the artifacts directory."""
    settings = get_settings()
    artifacts_dir = settings.models_dir
    
    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory does not exist: {artifacts_dir}")
        return 0
    
    # Find all model files
    model_extensions = ['.pkl', '.h5', '.joblib', '.weights.h5', '.weights.pkl']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(list(artifacts_dir.glob(f"*{ext}")))
    
    if not model_files:
        print(f"✓ No model files found in {artifacts_dir}")
        return 0
    
    print(f"\n📁 Found {len(model_files)} model files in {artifacts_dir}")
    
    # Delete each file
    deleted_count = 0
    for file_path in model_files:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"   ✓ Deleted: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to delete {file_path.name}: {e}")
    
    print(f"\n✅ Deleted {deleted_count}/{len(model_files)} files from filesystem")
    return deleted_count


def delete_mongodb_models():
    """Delete all model-related documents from MongoDB."""
    try:
        mongo_db = get_mongo_db()
    except Exception as e:
        print(f"⚠️  Could not connect to MongoDB: {e}")
        return 0
    
    total_deleted = 0
    
    # Collections to clean
    collections_to_clean = {
        'models': 'Model metadata',
        'model_versions': 'Model versions',
        'metrics': 'Evaluation metrics',
        'forecasts': 'Forecast predictions',
    }
    
    print(f"\n🗄️  Deleting model data from MongoDB...")
    
    for collection_name, description in collections_to_clean.items():
        try:
            collection = getattr(mongo_db, collection_name)
            count = collection.count_documents({})
            
            if count == 0:
                print(f"   ✓ {description}: No documents found")
                continue
            
            result = collection.delete_many({})
            deleted = result.deleted_count
            total_deleted += deleted
            print(f"   ✓ {description}: Deleted {deleted} document(s)")
            
        except AttributeError:
            # Collection doesn't exist
            print(f"   ⚠️  {description}: Collection '{collection_name}' does not exist")
        except Exception as e:
            print(f"   ⚠️  {description}: Error - {e}")
    
    print(f"\n✅ Deleted {total_deleted} document(s) from MongoDB")
    return total_deleted


def main():
    """Main function to delete all model data."""
    print("=" * 70)
    print("DELETE ALL MODEL DATA")
    print("=" * 70)
    print("\n⚠️  WARNING: This will permanently delete:")
    print("   • All model files from filesystem (artifacts directory)")
    print("   • All model metadata from MongoDB")
    print("   • All evaluation metrics from MongoDB")
    print("   • All forecast predictions from MongoDB")
    print("\nThis action is IRREVERSIBLE!")
    
    # Ask for confirmation
    response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Operation cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("DELETING MODEL DATA...")
    print("=" * 70)
    
    # Delete from filesystem
    files_deleted = delete_filesystem_models()
    
    # Delete from MongoDB
    docs_deleted = delete_mongodb_models()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Files deleted from filesystem: {files_deleted}")
    print(f"✅ Documents deleted from MongoDB: {docs_deleted}")
    print("\n✅ All model data has been deleted!")
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

