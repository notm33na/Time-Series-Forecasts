"""Test hourly data ingestion to debug the issue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from backend.services.data_ingestion import DataIngestionService
from backend.db.unified_db import UnifiedDataStore

print("Testing hourly data ingestion...")
print("=" * 80)

# Test yfinance directly
print("\n1. Testing yfinance directly:")
ticker = yf.Ticker('AAPL')
df_raw = ticker.history(period='30d', interval='1h')
print(f"   Raw data from yfinance: {len(df_raw)} records")

if not df_raw.empty:
    print(f"   Date range: {df_raw.index.min()} to {df_raw.index.max()}")
    print(f"   Columns: {list(df_raw.columns)}")
    
    # Test the processing steps
    print("\n2. Testing data processing:")
    df = df_raw.reset_index()
    print(f"   After reset_index: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Handle different date column names
    if 'Datetime' in df.columns:
        df['date'] = df['Datetime']
    elif 'Date' in df.columns:
        df['date'] = df['Date']
    else:
        df['date'] = df.index
    print(f"   After adding date column: {len(df)} records")
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    print(f"   After datetime conversion: {len(df)} records")
    print(f"   Sample dates: {df['date'].head(3).tolist()}")
    
    df_filtered = df[df['date'] > pd.Timestamp('2000-01-01', tz='UTC')]
    print(f"   After filtering (>2000): {len(df_filtered)} records")
    
    if len(df_filtered) > 0:
        print(f"   Sample filtered dates: {df_filtered['date'].head(3).tolist()}")
        
        # Test saving
        print("\n3. Testing save to MongoDB:")
        service = DataIngestionService()
        count = service.ingest_from_yfinance('AAPL', period='30d', interval='1h')
        print(f"   Ingested: {count} records")
        
        # Check what's in MongoDB
        print("\n4. Checking MongoDB:")
        store = UnifiedDataStore()
        mongo_count = store.mongo_db.prices.count_documents({'symbol': 'AAPL', '_interval': '1h'})
        print(f"   Records in MongoDB with _interval='1h': {mongo_count}")
        
        all_count = store.mongo_db.prices.count_documents({'symbol': 'AAPL'})
        print(f"   Total AAPL records: {all_count}")
        
        if mongo_count > 0:
            sample = list(store.mongo_db.prices.find({'symbol': 'AAPL', '_interval': '1h'}).limit(3))
            print(f"   Sample records: {[s.get('date') for s in sample]}")
    else:
        print("   ERROR: All data filtered out!")
else:
    print("   ERROR: No data from yfinance!")

print("=" * 80)

