"""
Diagnostic script to check why /api/forecast/run returns 404.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("404 DIAGNOSTIC - /api/forecast/run")
print("=" * 80)
print()

# Step 1: Check if route is defined
print("Step 1: Checking route definition...")
try:
    from backend.api.forecast import router
    print(f"  ✓ Router imported successfully")
    print(f"  ✓ Router prefix: {router.prefix}")
    
    routes = [r for r in router.routes if hasattr(r, 'path')]
    print(f"  ✓ Routes in forecast router: {len(routes)}")
    for r in routes:
        methods = getattr(r, 'methods', set())
        print(f"    - {r.path} {methods}")
    
    # Check specifically for /run
    run_routes = [r for r in routes if 'run' in r.path.lower()]
    if run_routes:
        print(f"  ✓ /run route found: {run_routes[0].path}")
    else:
        print(f"  ✗ /run route NOT found in router!")
        
except Exception as e:
    print(f"  ✗ Error importing forecast router: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 2: Check if router is registered in main app
print("Step 2: Checking router registration in main app...")
try:
    from backend.main import app
    print(f"  ✓ Main app imported successfully")
    
    # Get all routes
    all_routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            methods = getattr(route, 'methods', set()) if hasattr(route, 'methods') else set()
            all_routes.append((route.path, methods))
    
    print(f"  ✓ Total routes in app: {len(all_routes)}")
    
    # Check for forecast routes
    forecast_routes = [r for r in all_routes if '/forecast' in r[0]]
    print(f"  ✓ Forecast routes found: {len(forecast_routes)}")
    for path, methods in forecast_routes:
        print(f"    - {path} {methods}")
    
    # Check specifically for /api/forecast/run
    run_route = [r for r in all_routes if r[0] == '/api/forecast/run']
    if run_route:
        print(f"  ✓ /api/forecast/run is registered: {run_route[0]}")
    else:
        print(f"  ✗ /api/forecast/run is NOT registered in app!")
        print(f"  Available forecast routes:")
        for path, methods in forecast_routes:
            print(f"    - {path} {methods}")
            
except Exception as e:
    print(f"  ✗ Error importing main app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Check for import errors
print("Step 3: Checking for import errors...")
try:
    from backend.api import forecast, data, evaluation, models, portfolio, retraining
    print(f"  ✓ All API modules imported successfully")
except Exception as e:
    print(f"  ✗ Error importing API modules: {e}")
    import traceback
    traceback.print_exc()

print()

# Step 4: Check if server can start
print("Step 4: Testing if app can be instantiated...")
try:
    from backend.main import app
    print(f"  ✓ App instantiated successfully")
    print(f"  ✓ App title: {app.title}")
    
    # Try to get OpenAPI schema (this will fail if there are route issues)
    try:
        schema = app.openapi()
        print(f"  ✓ OpenAPI schema generated successfully")
        print(f"  ✓ Schema has {len(schema.get('paths', {}))} paths")
        
        # Check if our route is in the schema
        if '/api/forecast/run' in schema.get('paths', {}):
            print(f"  ✓ /api/forecast/run is in OpenAPI schema")
        else:
            print(f"  ✗ /api/forecast/run is NOT in OpenAPI schema")
            print(f"  Available paths in schema:")
            for path in list(schema.get('paths', {}).keys())[:10]:
                print(f"    - {path}")
    except Exception as e:
        print(f"  ⚠ Could not generate OpenAPI schema: {e}")
        
except Exception as e:
    print(f"  ✗ Error instantiating app: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print()
print("If /api/forecast/run is registered but still returns 404:")
print("  1. Make sure the server is running: python -m backend.main")
print("  2. Check server logs for startup errors")
print("  3. Verify the server is listening on port 8000")
print("  4. Try accessing: http://localhost:8000/docs to see all routes")
print()

