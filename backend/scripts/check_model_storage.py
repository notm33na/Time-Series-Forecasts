"""
Script to check where model data is stored.
Shows both filesystem (artifacts) and MongoDB storage locations.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir.parent))

from backend.db import get_mongo_db
from backend.config import get_settings
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def check_filesystem_models():
    """Check filesystem for model artifacts."""
    settings = get_settings()
    
    print("=" * 70)
    print("FILESYSTEM MODEL STORAGE")
    print("=" * 70)
    
    # Check default models directory
    models_dir = settings.models_dir
    print(f"\n📁 Default Models Directory:")
    print(f"   Path: {models_dir}")
    print(f"   Exists: {models_dir.exists()}")
    
    if models_dir.exists():
        print(f"   Absolute: {models_dir.resolve()}")
        
        # Count files by type
        pkl_files = list(models_dir.glob("*.pkl"))
        h5_files = list(models_dir.glob("*.h5"))
        joblib_files = list(models_dir.glob("*.joblib"))
        all_files = list(models_dir.glob("*"))
        
        print(f"\n   File Counts:")
        print(f"     • .pkl files: {len(pkl_files)}")
        print(f"     • .h5 files: {len(h5_files)}")
        print(f"     • .joblib files: {len(joblib_files)}")
        print(f"     • Total files: {len(all_files)}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        print(f"     • Total size: {format_size(total_size)}")
        
        # Group by model type
        print(f"\n   Models by Type:")
        model_types = {}
        for file in all_files:
            if file.is_file():
                name = file.name.lower()
                if 'arima' in name:
                    model_types.setdefault('ARIMA', []).append(file)
                elif 'lstm' in name:
                    model_types.setdefault('LSTM', []).append(file)
                elif 'gru' in name:
                    model_types.setdefault('GRU', []).append(file)
                elif 'transformer' in name:
                    model_types.setdefault('Transformer', []).append(file)
                elif 'exponential' in name or 'smoothing' in name:
                    model_types.setdefault('ExponentialSmoothing', []).append(file)
                elif 'adaptive' in name:
                    model_types.setdefault('AdaptiveForecaster', []).append(file)
                elif 'ensemble' in name:
                    model_types.setdefault('Ensemble', []).append(file)
        
        for model_type, files in sorted(model_types.items()):
            print(f"     • {model_type}: {len(files)} file(s)")
            # Show first few files
            for f in sorted(files)[:3]:
                size = format_size(f.stat().st_size)
                print(f"       - {f.name} ({size})")
            if len(files) > 3:
                print(f"       ... and {len(files) - 3} more")
        
        # Group by symbol
        print(f"\n   Models by Symbol:")
        symbols = {}
        for file in all_files:
            if file.is_file():
                # Extract symbol from filename (usually first part before underscore)
                parts = file.stem.split('_')
                if parts:
                    symbol = parts[0].upper()
                    symbols.setdefault(symbol, []).append(file)
        
        for symbol, files in sorted(symbols.items()):
            print(f"     • {symbol}: {len(files)} file(s)")
        
        # Show recent files
        if all_files:
            print(f"\n   Recent Files (last 5):")
            recent_files = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
            for f in recent_files:
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    size = format_size(f.stat().st_size)
                    print(f"     • {f.name}")
                    print(f"       Size: {size}, Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"   ⚠️  Directory does not exist!")
        print(f"   Models will be saved here when training scripts run.")
    
    # Check alternative locations
    print(f"\n📁 Alternative Locations Checked:")
    alt_locations = [
        backend_dir.parent / "artifacts",
        backend_dir / "artifacts",
        Path.cwd() / "artifacts",
    ]
    for alt_path in alt_locations:
        exists = alt_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {alt_path}")
        if exists:
            file_count = len(list(alt_path.glob("*")))
            print(f"      ({file_count} items)")


def check_mongodb_models():
    """Check MongoDB for model metadata."""
    settings = get_settings()
    
    print("\n" + "=" * 70)
    print("MONGODB MODEL STORAGE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  URI: {settings.mongo_uri}")
    print(f"  Database: {settings.mongo_db}")
    
    try:
        db = get_mongo_db()
        print("\n✅ Successfully connected to MongoDB!")
        
        # Check models collection
        models_collection = db["models"]
        model_count = models_collection.count_documents({})
        
        print(f"\n📊 Models Collection:")
        print(f"   Total model records: {model_count:,}")
        
        if model_count > 0:
            # Get sample
            sample = models_collection.find_one()
            if sample:
                print(f"   Sample document keys: {list(sample.keys())}")
            
            # Group by model type
            print(f"\n   Models by Type:")
            model_types = models_collection.distinct("training_info.model_type")
            for mt in sorted(model_types):
                count = models_collection.count_documents({"training_info.model_type": mt})
                print(f"     • {mt}: {count:,} model(s)")
            
            # Group by symbol
            print(f"\n   Models by Symbol:")
            symbols = models_collection.distinct("symbol")
            for symbol in sorted(symbols):
                count = models_collection.count_documents({"symbol": symbol})
                print(f"     • {symbol}: {count:,} model(s)")
            
            # Show recent models
            print(f"\n   Recent Models (last 5):")
            recent_models = list(models_collection.find().sort("trained_at", -1).limit(5))
            for doc in recent_models:
                model_type = doc.get("training_info", {}).get("model_type", "Unknown")
                symbol = doc.get("symbol", "Unknown")
                version_tag = doc.get("version_tag", "Unknown")
                artifact_path = doc.get("artifact_path", "N/A")
                trained_at = doc.get("trained_at", "N/A")
                
                print(f"     • {symbol} - {model_type}")
                print(f"       Version: {version_tag}")
                print(f"       Artifact: {artifact_path}")
                print(f"       Trained: {trained_at}")
                
                # Check if artifact file exists
                if artifact_path and artifact_path != "N/A":
                    artifact_file = Path(artifact_path)
                    if artifact_file.exists():
                        size = format_size(artifact_file.stat().st_size)
                        print(f"       ✅ File exists ({size})")
                    else:
                        print(f"       ❌ File not found!")
        else:
            print(f"   ⚠️  No model records in MongoDB")
            print(f"   Models will be registered here when trained via API or scripts.")
        
        # Check metrics collection
        metrics_collection = db["metrics"]
        metrics_count = metrics_collection.count_documents({})
        print(f"\n📊 Metrics Collection:")
        print(f"   Total metric records: {metrics_count:,}")
        
        # Check forecasts collection
        forecasts_collection = db["forecasts"]
        forecasts_count = forecasts_collection.count_documents({})
        print(f"\n📊 Forecasts Collection:")
        print(f"   Total forecast records: {forecasts_count:,}")
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print("\n❌ Failed to connect to MongoDB!")
        print(f"   Error: {e}")
        print("\n   MongoDB is used for:")
        print("     • Model version metadata")
        print("     • Model evaluation metrics")
        print("     • Forecast predictions")
        print("     • Tracking artifact file paths")
    
    except Exception as e:
        print(f"\n❌ Error checking MongoDB: {e}")
        import traceback
        traceback.print_exc()


def check_model_storage():
    """Main function to check all model storage locations."""
    print("\n" + "=" * 70)
    print("MODEL STORAGE LOCATION CHECK")
    print("=" * 70)
    print("\nThis script checks where model data is stored:")
    print("  1. Filesystem: Model artifact files (.pkl, .h5, .joblib)")
    print("  2. MongoDB: Model metadata, versions, metrics, forecasts")
    
    # Check filesystem
    check_filesystem_models()
    
    # Check MongoDB
    check_mongodb_models()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nModel Storage Architecture:")
    print("  • Filesystem (Artifacts):")
    print("    - Actual model files (.pkl, .h5, .joblib)")
    print("    - Contains trained model weights and metadata")
    print("    - Default location: backend/artifacts/")
    print("  • MongoDB (Metadata):")
    print("    - Model version information")
    print("    - Training metadata and configuration")
    print("    - Evaluation metrics")
    print("    - Forecast predictions")
    print("    - References to artifact file paths")
    print("\nNote: Both storage locations work together:")
    print("  - MongoDB tracks which models exist and their metadata")
    print("  - Filesystem stores the actual model files")
    print("  - ModelLoader service connects both to load models")
    print("=" * 70)


if __name__ == "__main__":
    try:
        check_model_storage()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

