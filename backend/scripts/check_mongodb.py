"""
Script to check MongoDB for data.
Shows connection status, collections, document counts, and sample data.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir.parent))

from backend.db import get_mongo_db
from backend.config import get_settings
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


def check_mongodb():
    """Check MongoDB connection and data."""
    settings = get_settings()
    
    print("=" * 70)
    print("MongoDB Data Check")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  URI: {settings.mongo_uri}")
    print(f"  Database: {settings.mongo_db}")
    print(f"  Use MongoDB: {settings.use_mongo}")
    
    try:
        # Connect to MongoDB
        print("\n" + "-" * 70)
        print("Connecting to MongoDB...")
        db = get_mongo_db()
        print("✅ Successfully connected to MongoDB!")
        
        # List all collections
        print("\n" + "-" * 70)
        print("Collections:")
        collections = db.list_collection_names()
        if collections:
            for coll_name in sorted(collections):
                count = db[coll_name].count_documents({})
                print(f"  • {coll_name}: {count:,} documents")
        else:
            print("  ⚠️  No collections found (database is empty)")
        
        # Check each expected collection
        expected_collections = ["prices", "forecasts", "models", "metrics", "portfolio"]
        
        print("\n" + "-" * 70)
        print("Detailed Collection Information:")
        print("-" * 70)
        
        for coll_name in expected_collections:
            collection = db[coll_name]
            count = collection.count_documents({})
            
            print(f"\n📊 {coll_name.upper()}")
            print(f"   Total documents: {count:,}")
            
            if count > 0:
                # Get sample document
                sample = collection.find_one()
                if sample:
                    print(f"   Sample document keys: {list(sample.keys())}")
                
                # Collection-specific stats
                if coll_name == "prices":
                    # Count unique symbols
                    symbols = collection.distinct("symbol")
                    print(f"   Unique symbols: {len(symbols)}")
                    if symbols:
                        print(f"   Symbols: {', '.join(sorted(symbols)[:10])}")
                        if len(symbols) > 10:
                            print(f"   ... and {len(symbols) - 10} more")
                    
                    # Date range
                    dates = collection.find({}, {"date": 1}).sort("date", 1)
                    dates_list = [doc.get("date") for doc in dates if doc.get("date")]
                    if dates_list:
                        print(f"   Date range: {dates_list[0]} to {dates_list[-1]}")
                
                elif coll_name == "forecasts":
                    # Count by model type
                    model_types = collection.distinct("model_type")
                    print(f"   Model types: {len(model_types)}")
                    if model_types:
                        for mt in model_types:
                            mt_count = collection.count_documents({"model_type": mt})
                            print(f"     • {mt}: {mt_count:,} forecasts")
                
                elif coll_name == "models":
                    # Count by model type
                    model_types = collection.distinct("model_type")
                    print(f"   Model types: {len(model_types)}")
                    if model_types:
                        for mt in model_types:
                            mt_count = collection.count_documents({"model_type": mt})
                            print(f"     • {mt}: {mt_count:,} models")
                
                elif coll_name == "metrics":
                    # Count by model type
                    model_types = collection.distinct("model_type")
                    print(f"   Model types: {len(model_types)}")
                    if model_types:
                        for mt in model_types:
                            mt_count = collection.count_documents({"model_type": mt})
                            print(f"     • {mt}: {mt_count:,} metric records")
                
                elif coll_name == "portfolio":
                    # Count unique symbols
                    symbols = collection.distinct("symbol")
                    print(f"   Unique symbols: {len(symbols)}")
                    if symbols:
                        print(f"   Symbols: {', '.join(sorted(symbols)[:10])}")
            
            else:
                print(f"   ⚠️  No data in this collection")
        
        # Show sample data from prices if available
        print("\n" + "-" * 70)
        print("Sample Data (from 'prices' collection):")
        print("-" * 70)
        
        prices_collection = db["prices"]
        sample_prices = list(prices_collection.find().limit(5).sort("date", -1))
        
        if sample_prices:
            print(f"\nShowing {len(sample_prices)} most recent price records:\n")
            for i, doc in enumerate(sample_prices, 1):
                print(f"Record {i}:")
                print(f"  Symbol: {doc.get('symbol', 'N/A')}")
                print(f"  Date: {doc.get('date', 'N/A')}")
                print(f"  Close: ${doc.get('close', 0):.2f}")
                print(f"  Volume: {doc.get('volume', 0):,}")
                print()
        else:
            print("  ⚠️  No price data available")
        
        print("=" * 70)
        print("✅ MongoDB check completed successfully!")
        print("=" * 70)
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print("\n❌ Failed to connect to MongoDB!")
        print(f"   Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure MongoDB is running:")
        print("     - Docker: docker ps (check if mongodb container is running)")
        print("     - Windows Service: Get-Service -Name MongoDB")
        print("  2. Check connection string in .env file:")
        print(f"     FORECAST_MONGO_URI={settings.mongo_uri}")
        print("  3. Try connecting manually:")
        print("     mongosh mongodb://localhost:27017/")
        return False
    
    except Exception as e:
        print(f"\n❌ Error checking MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = check_mongodb()
    sys.exit(0 if success else 1)

