"""Check what data is in MongoDB for a symbol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.unified_db import UnifiedDataStore
import pandas as pd

store = UnifiedDataStore()
# Prefer hourly data (1h interval)
all_records = list(store.mongo_db.prices.find({'symbol': 'AAPL', '_interval': '1h'}).sort('date', 1))
if not all_records:
    # Fallback to all data if no hourly data
    all_records = list(store.mongo_db.prices.find({'symbol': 'AAPL'}).sort('date', 1))

print(f'Total AAPL records: {len(all_records)}')

if all_records:
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f'\nDate range: {df["date"].min()} to {df["date"].max()}')
    print(f'\nFirst 10 records:')
    print(df[['date', 'close']].head(10))
    
    # Check time differences
    time_diffs = df['date'].diff().dt.total_seconds().div(3600).dropna()
    print(f'\nTime differences (hours):')
    print(f'Min: {time_diffs.min():.2f}h, Max: {time_diffs.max():.2f}h, Mean: {time_diffs.mean():.2f}h')
    print(f'\nUnique time differences: {sorted(time_diffs.unique())[:20]}')
    
    # Check if data is hourly or daily
    most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
    if most_common_diff:
        if 0.9 <= most_common_diff <= 1.1:
            print(f'\n✓ Data appears to be HOURLY (most common interval: {most_common_diff:.2f} hours)')
        elif 23 <= most_common_diff <= 25:
            print(f'\n✓ Data appears to be DAILY (most common interval: {most_common_diff:.2f} hours)')
        else:
            print(f'\n? Data interval unclear (most common: {most_common_diff:.2f} hours)')

