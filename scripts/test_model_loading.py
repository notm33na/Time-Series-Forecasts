"""Test model loading directly to see what fails."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.model_loader import ModelLoader

print("Testing model loading for AAPL...\n")

try:
    # Try to load ensemble models
    loaded_models = ModelLoader.load_ensemble_models("AAPL", 24)
    print(f"✓ Successfully loaded {len(loaded_models)} models:")
    for model_type in loaded_models.keys():
        print(f"  - {model_type}")
    
    # Try to create ensemble
    ensemble = ModelLoader.create_ensemble_from_loaded_models("AAPL", 24, loaded_models)
    print(f"\n✓ Successfully created ensemble with {len(ensemble.forecasters)} forecasters")
    
    # Check if models are fitted
    fitted_count = sum(1 for f in ensemble.forecasters if hasattr(f, 'is_fitted') and f.is_fitted)
    print(f"✓ {fitted_count}/{len(ensemble.forecasters)} models are fitted")
    
except Exception as e:
    import traceback
    print(f"✗ ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

