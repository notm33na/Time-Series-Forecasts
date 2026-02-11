"""
Script to completely clear all data in MongoDB.
WARNING: This will delete ALL collections and data!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.db.unified_db import get_mongo_db
from backend.config import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def clear_all_mongodb_data():
    """
    Clear all data from MongoDB collections.
    WARNING: This permanently deletes all data!
    """
    try:
        mongo_db = get_mongo_db()
        
        # List all collections
        collections = mongo_db.list_collection_names()
        
        if not collections:
            print("No collections found in MongoDB. Database is already empty.")
            return
        
        print(f"Found {len(collections)} collections in MongoDB:")
        for col in collections:
            count = mongo_db[col].count_documents({})
            print(f"  - {col}: {count} documents")
        
        # Confirm deletion
        print("\n⚠️  WARNING: This will DELETE ALL DATA from MongoDB!")
        print("Collections to be deleted:")
        for col in collections:
            print(f"  - {col}")
        
        response = input("\nType 'DELETE ALL' to confirm: ")
        
        if response != "DELETE ALL":
            print("❌ Deletion cancelled.")
            return
        
        # Delete all collections
        print("\n🗑️  Deleting all collections...")
        deleted_count = 0
        for col in collections:
            result = mongo_db[col].delete_many({})
            deleted_count += result.deleted_count
            print(f"  ✓ Deleted {result.deleted_count} documents from {col}")
        
        # Drop collections to remove indexes and metadata
        print("\n🗑️  Dropping collections...")
        for col in collections:
            mongo_db[col].drop()
            print(f"  ✓ Dropped collection: {col}")
        
        print(f"\n✅ Successfully cleared MongoDB!")
        print(f"   Total documents deleted: {deleted_count}")
        print(f"   Collections dropped: {len(collections)}")
        
        # Verify
        remaining_collections = mongo_db.list_collection_names()
        if remaining_collections:
            print(f"\n⚠️  Warning: {len(remaining_collections)} collections still exist:")
            for col in remaining_collections:
                print(f"  - {col}")
        else:
            print("\n✓ MongoDB is now completely empty.")
            
    except Exception as e:
        logger.error(f"Error clearing MongoDB: {e}", exc_info=True)
        print(f"\n❌ Error clearing MongoDB: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear all data from MongoDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    if args.force:
        # Force mode - skip confirmation
        try:
            mongo_db = get_mongo_db()
            collections = mongo_db.list_collection_names()
            
            if not collections:
                print("MongoDB is already empty.")
                sys.exit(0)
            
            print(f"Force mode: Clearing {len(collections)} collections...")
            deleted_count = 0
            for col in collections:
                result = mongo_db[col].delete_many({})
                deleted_count += result.deleted_count
                mongo_db[col].drop()
                print(f"  ✓ Cleared {col} ({result.deleted_count} documents)")
            
            print(f"\n✅ Cleared MongoDB: {deleted_count} documents deleted, {len(collections)} collections dropped")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode with confirmation
        clear_all_mongodb_data()

