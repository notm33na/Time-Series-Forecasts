"""
Script to fix flat OHLC data in MongoDB by re-ingesting with proper variation.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger
import pandas as pd

logger = get_logger(__name__)

def fix_flat_ohlc_data(symbol: str = "GOOGL", period: str = "30d", interval: str = "1h"):
    """
    Re-ingest data to fix any flat OHLC values.
    
    Args:
        symbol: Stock symbol to fix
        period: Period to fetch (default: 30d for hourly data)
        interval: Data interval (default: 1h)
    """
    print(f"\n{'='*70}")
    print(f"Fixing Flat OHLC Data for {symbol}")
    print(f"{'='*70}\n")
    
    try:
        service = DataIngestionService()
        
        # Check current data
        print("1. Checking current data...")
        df_before = service.get_prices(symbol, limit=10)
        if df_before.empty:
            print(f"   ❌ No data found for {symbol}")
            return False
        
        print(f"   ✓ Found {len(df_before)} records")
        
        # Check for flat candles
        flat_count = 0
        for idx, row in df_before.iterrows():
            if (row['open'] == row['high'] == row['low'] == row['close']):
                flat_count += 1
                print(f"   ⚠️  Flat candle at {row['date']}: all OHLC = {row['close']:.2f}")
        
        # Check for zero volumes
        zero_volume_count = 0
        if 'volume' in df_before.columns:
            zero_volume_count = (df_before['volume'] == 0).sum()
            if zero_volume_count > 0:
                print(f"   ⚠️  Found {zero_volume_count} rows with zero volume")
        
        if flat_count == 0 and zero_volume_count == 0:
            print(f"   ✓ No flat candles or zero volumes found. Data is good!")
            return True
        
        issues = []
        if flat_count > 0:
            issues.append(f"{flat_count} flat candles")
        if zero_volume_count > 0:
            issues.append(f"{zero_volume_count} zero volumes")
        
        print(f"\n2. Found {', '.join(issues)}. Re-ingesting data to fix...")
        
        # Re-ingest data (this will apply the fix_ohlc_variation automatically)
        count = service.ingest_from_yfinance(symbol, period=period, interval=interval)
        
        if count == 0:
            print(f"   ❌ Failed to ingest data")
            return False
        
        print(f"   ✓ Re-ingested {count} records")
        
        # Verify fix
        print("\n3. Verifying fix...")
        df_after = service.get_prices(symbol, limit=10)
        
        flat_count_after = 0
        for idx, row in df_after.iterrows():
            if (row['open'] == row['high'] == row['low'] == row['close']):
                flat_count_after += 1
        
        zero_volume_after = 0
        if 'volume' in df_after.columns:
            zero_volume_after = (df_after['volume'] == 0).sum()
        
        if flat_count_after == 0 and zero_volume_after == 0:
            print(f"   ✓ All issues fixed! Data now has proper variation and volume.")
            
            # Show last row
            if len(df_after) > 0:
                last_row = df_after.iloc[-1]
                print(f"\n   Last row:")
                print(f"   Open:  ${last_row['open']:.2f}")
                print(f"   High:  ${last_row['high']:.2f}")
                print(f"   Low:   ${last_row['low']:.2f}")
                print(f"   Close: ${last_row['close']:.2f}")
                print(f"   Spread: ${last_row['high'] - last_row['low']:.2f} ({(last_row['high'] - last_row['low']) / last_row['close'] * 100:.2f}%)")
                if 'volume' in last_row:
                    print(f"   Volume: {last_row['volume']:,.0f}")
            
            return True
        else:
            issues_remaining = []
            if flat_count_after > 0:
                issues_remaining.append(f"{flat_count_after} flat candles")
            if zero_volume_after > 0:
                issues_remaining.append(f"{zero_volume_after} zero volumes")
            print(f"   ⚠️  Still found {', '.join(issues_remaining)} after fix")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix flat OHLC data")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Symbol to fix")
    parser.add_argument("--period", type=str, default="30d", help="Period to fetch")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval")
    
    args = parser.parse_args()
    
    success = fix_flat_ohlc_data(args.symbol, args.period, args.interval)
    sys.exit(0 if success else 1)

