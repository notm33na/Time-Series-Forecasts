"""Check if model artifact paths in database actually exist."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db.unified_db import UnifiedDataStore

store = UnifiedDataStore()
models = store.list_model_versions('AAPL')

print(f"Found {len(models)} models for AAPL\n")
for m in models[:10]:
    artifact_path = m.get('artifact_path', '')
    exists = Path(artifact_path).exists() if artifact_path else False
    print(f"{m.get('model_type')}:")
    print(f"  Path: {artifact_path}")
    print(f"  Exists: {exists}")
    if artifact_path and not exists:
        # Try to find the file
        filename = Path(artifact_path).name
        from backend.config import get_settings
        settings = get_settings()
        found = list(settings.models_dir.glob(filename))
        if found:
            print(f"  Found at: {found[0]}")
    print()

