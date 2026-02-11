"""
Test script to verify forecast API is working for all horizons.
Tests 1h, 3h, 24h, and 72h forecasts.
"""

import requests
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# API base URL
API_BASE = "http://localhost:8000"

def test_debug_endpoint(symbol: str = "AAPL"):
    """Test the debug endpoint to see what's available."""
    print(f"\n{'='*70}")
    print(f"Testing Debug Endpoint for {symbol}")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(f"{API_BASE}/api/forecast/debug/{symbol}")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Debug endpoint responded successfully")
        print(f"\nData Status:")
        print(f"  - Data available: {data.get('data_available', False)}")
        print(f"  - Data count: {data.get('data_count', 0)}")
        
        print(f"\nModels Status:")
        print(f"  - Database models: {data.get('models_available_db', False)}")
        print(f"  - Filesystem models: {data.get('models_available_filesystem', False)}")
        
        if data.get('models_dir_found'):
            print(f"  - Models directory: {data.get('models_dir_found')}")
        
        if data.get('models_filesystem'):
            print(f"\n  Filesystem Models Found:")
            for model in data.get('models_filesystem', []):
                print(f"    - {model['model_type']}: {model['file']}")
        
        if data.get('ensemble_load_success'):
            print(f"\n✓ Ensemble can be loaded!")
            print(f"  Models: {', '.join(data.get('ensemble_models_loaded', []))}")
        else:
            print(f"\n✗ Ensemble load failed: {data.get('ensemble_load_error', 'Unknown error')}")
        
        if data.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in data.get('recommendations', []):
                print(f"  - {rec}")
        
        return data
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: Cannot connect to {API_BASE}")
        print(f"  Make sure the backend server is running!")
        return None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None


def test_forecast(symbol: str = "AAPL", horizon: str = "24h"):
    """Test forecast endpoint for a specific horizon."""
    print(f"\n{'='*70}")
    print(f"Testing Forecast: {symbol} - {horizon}")
    print(f"{'='*70}\n")
    
    try:
        payload = {
            "symbol": symbol,
            "model_type": "Ensemble",
            "horizon": horizon
        }
        
        print(f"Request: {json.dumps(payload, indent=2)}")
        print(f"\nSending request to {API_BASE}/api/forecast/run...")
        
        response = requests.post(
            f"{API_BASE}/api/forecast/run",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ SUCCESS! Forecast generated")
            print(f"\nForecast Details:")
            print(f"  - Symbol: {data.get('symbol')}")
            print(f"  - Model Type: {data.get('model_type')}")
            print(f"  - Horizon: {data.get('horizon')}")
            print(f"  - Data Frequency: {data.get('data_frequency', 'unknown')}")
            print(f"  - Forecast Steps: {data.get('forecast_steps', 0)}")
            
            if 'ensemble_forecast' in data:
                forecast = data['ensemble_forecast']
                print(f"  - Ensemble Forecast Points: {len(forecast)}")
                if forecast:
                    print(f"    First 5 values: {forecast[:5]}")
                    print(f"    Last 5 values: {forecast[-5:]}")
            
            # Show individual model forecasts if available
            individual_forecasts = {k: v for k, v in data.items() if k.endswith('_forecast')}
            if individual_forecasts:
                print(f"\n  Individual Model Forecasts:")
                for model_name, forecast in individual_forecasts.items():
                    print(f"    - {model_name}: {len(forecast)} points")
            
            return True
        else:
            print(f"\n✗ ERROR: Status {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error Detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: Cannot connect to {API_BASE}")
        print(f"  Make sure the backend server is running!")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print(f"\n{'='*70}")
    print("FORECAST API TEST SUITE")
    print(f"{'='*70}")
    print(f"\nTesting API at: {API_BASE}")
    print(f"Make sure the backend server is running at {API_BASE}")
    
    symbol = "AAPL"
    
    # Test 1: Debug endpoint
    debug_data = test_debug_endpoint(symbol)
    
    if debug_data is None:
        print("\n✗ Cannot proceed - server not accessible or debug endpoint failed")
        return
    
    # Test 2: Forecast for each horizon
    horizons = ["1h", "3h", "24h", "72h"]
    results = {}
    
    print(f"\n{'='*70}")
    print("TESTING ALL FORECAST HORIZONS")
    print(f"{'='*70}")
    
    for horizon in horizons:
        success = test_forecast(symbol, horizon)
        results[horizon] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}\n")
    
    for horizon, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {horizon:>4}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All tests passed! Forecast API is working correctly.")
    else:
        print(f"\n✗ Some tests failed. Check the errors above.")
        print(f"\nTroubleshooting:")
        print(f"  1. Check server logs for detailed error messages")
        print(f"  2. Verify models are trained: python notebooks/train_arima.py --symbol {symbol}")
        print(f"  3. Check debug endpoint: {API_BASE}/api/forecast/debug/{symbol}")
        print(f"  4. Verify data is available: {API_BASE}/api/data/prices/{symbol}")


if __name__ == "__main__":
    main()

