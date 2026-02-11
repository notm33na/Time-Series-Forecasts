"""
Script to populate MongoDB with GOOGL data at multiple intervals:
- 1 hour interval
- 3 hour interval  
- 24 hour interval (daily)
- 72 hour interval (3 days)

This script fetches data from yfinance and stores it in MongoDB with proper interval tags.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings

from backend.config import get_settings
from backend.services.data_ingestion import DataIngestionService

warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning)

settings = get_settings()


def resample_to_interval(df: pd.DataFrame, target_interval_hours: int) -> pd.DataFrame:
    """
    Resample hourly data to a target interval in hours.
    
    Args:
        df: DataFrame with datetime index and OHLCV columns
        target_interval_hours: Target interval in hours (e.g., 3 for 3-hour, 72 for 72-hour)
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    # Ensure date is the index
    if 'date' in df.columns:
        df = df.set_index('date')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or 'date' column")
    
    # Ensure index is timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Normalize column names to uppercase
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'open':
            column_mapping[col] = 'Open'
        elif col.lower() == 'high':
            column_mapping[col] = 'High'
        elif col.lower() == 'low':
            column_mapping[col] = 'Low'
        elif col.lower() == 'close':
            column_mapping[col] = 'Close'
        elif col.lower() == 'volume':
            column_mapping[col] = 'Volume'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Resample to target interval
    # For OHLCV data, we need to aggregate properly:
    # - Open: first value in period
    # - High: max value in period
    # - Low: min value in period
    # - Close: last value in period
    # - Volume: sum of volumes in period
    
    resampled = df.resample(f'{target_interval_hours}H', label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def populate_googl_data(
    symbol: str = "GOOGL",
    intervals: list = None,
    period: str = "2y"  # Get 2 years of data to ensure we have enough
):
    """
    Populate MongoDB with GOOGL data at multiple intervals.
    
    Args:
        symbol: Stock symbol (default: GOOGL)
        intervals: List of intervals to populate. Each interval can be:
                   - "1h" for 1-hour (uses yfinance directly)
                   - "3h" for 3-hour (resampled from 1h)
                   - "24h" or "1d" for daily (uses yfinance directly)
                   - "72h" for 72-hour/3-day (resampled from 1h)
        period: Period to fetch from yfinance (default: "2y" for 2 years)
    """
    if intervals is None:
        intervals = ["1h", "3h", "24h", "72h"]
    
    print(f"🚀 Populating MongoDB with {symbol} data")
    print(f"   Intervals: {', '.join(intervals)}")
    print(f"   Period: {period}")
    print(f"{'='*70}\n")
    
    data_service = DataIngestionService()
    results = {}
    
    # First, fetch 1-hour data (we'll need it for resampling)
    print("📥 Step 1: Fetching 1-hour base data from yfinance...")
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # For hourly data, yfinance limits how far back we can go
        # Try to get maximum available hourly data
        # For 1h data, max period is typically 60 days, but let's try 2y and see what we get
        print(f"   Fetching hourly data (period={period})...")
        df_1h = ticker.history(period=period, interval="1h")
        
        if df_1h.empty:
            # Try shorter period for hourly data
            print("   ⚠️  No hourly data with 2y period, trying 60d...")
            df_1h = ticker.history(period="60d", interval="1h")
        
        if df_1h.empty:
            raise ValueError(f"No hourly data available for {symbol}")
        
        # Prepare DataFrame
        df_1h = df_1h.reset_index()
        if 'Datetime' in df_1h.columns:
            df_1h['date'] = df_1h['Datetime']
        elif 'Date' in df_1h.columns:
            df_1h['date'] = df_1h['Date']
        else:
            df_1h['date'] = df_1h.index
        
        df_1h['date'] = pd.to_datetime(df_1h['date'], utc=True)
        df_1h = df_1h[df_1h['date'] > pd.Timestamp('2000-01-01', tz='UTC')]
        
        # Normalize column names to uppercase
        if 'close' in df_1h.columns:
            df_1h = df_1h.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
        
        print(f"   ✓ Fetched {len(df_1h)} hourly records")
        print(f"   Date range: {df_1h['date'].min()} to {df_1h['date'].max()}\n")
        
    except Exception as e:
        raise ValueError(f"Failed to fetch hourly data: {e}")
    
    # Process each interval
    for interval in intervals:
        print(f"📊 Processing {interval} interval...")
        
        try:
            if interval == "1h":
                # Use 1h data directly
                df_interval = df_1h.copy()
                yfinance_interval = "1h"
                
            elif interval == "3h":
                # Resample 1h data to 3h
                df_interval = df_1h.set_index('date').copy()
                df_interval = resample_to_interval(df_interval, 3)
                yfinance_interval = "3h"  # For metadata
                
            elif interval in ["24h", "1d"]:
                # Fetch daily data directly from yfinance (more reliable than resampling)
                print(f"   Fetching daily data from yfinance...")
                df_daily = ticker.history(period=period, interval="1d")
                df_interval = df_daily.reset_index()
                if 'Datetime' in df_interval.columns:
                    df_interval['date'] = df_interval['Datetime']
                elif 'Date' in df_interval.columns:
                    df_interval['date'] = df_interval['Date']
                else:
                    df_interval['date'] = df_interval.index
                df_interval['date'] = pd.to_datetime(df_interval['date'], utc=True)
                df_interval = df_interval[df_interval['date'] > pd.Timestamp('2000-01-01', tz='UTC')]
                
                # Normalize column names
                if 'close' in df_interval.columns:
                    df_interval = df_interval.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    })
                yfinance_interval = "1d"
                
            elif interval == "72h":
                # Resample 1h data to 72h (3 days)
                df_interval = df_1h.set_index('date').copy()
                df_interval = resample_to_interval(df_interval, 72)
                yfinance_interval = "72h"  # For metadata
                
            else:
                print(f"   ⚠️  Unknown interval: {interval}, skipping...")
                continue
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df_interval.columns]
            if missing_cols:
                # Try lowercase
                required_cols_lower = [col.lower() for col in required_cols]
                if all(col.lower() in df_interval.columns for col in required_cols):
                    df_interval = df_interval.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    })
                else:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add interval metadata
            df_interval['_interval'] = interval
            
            # Store in MongoDB using the data ingestion service
            # First clear existing data for this interval
            print(f"   Clearing existing {interval} data...")
            data_service.store.mongo_db.prices.delete_many({
                "symbol": symbol,
                "_interval": interval
            })
            
            # Store new data
            print(f"   Storing {len(df_interval)} records in MongoDB...")
            data_service.store.save_prices(symbol, df_interval, interval=interval)
            
            results[interval] = {
                "count": len(df_interval),
                "date_range": (str(df_interval['date'].min()), str(df_interval['date'].max())),
                "status": "success"
            }
            
            print(f"   ✓ Successfully stored {len(df_interval)} records for {interval} interval")
            print(f"      Date range: {df_interval['date'].min()} to {df_interval['date'].max()}\n")
            
        except Exception as e:
            print(f"   ❌ Failed to process {interval} interval: {e}")
            results[interval] = {
                "status": "error",
                "error": str(e)
            }
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for interval, result in results.items():
        if result.get("status") == "success":
            print(f"✅ {interval:6s}: {result['count']:6d} records ({result['date_range'][0]} to {result['date_range'][1]})")
        else:
            print(f"❌ {interval:6s}: Failed - {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with GOOGL data at multiple intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate all intervals (1h, 3h, 24h, 72h)
  python populate_googl_data.py

  # Populate specific intervals
  python populate_googl_data.py --intervals 1h 24h

  # Use different symbol
  python populate_googl_data.py --symbol AAPL

  # Fetch more historical data
  python populate_googl_data.py --period 5y

Note: 
  - 1h and 24h intervals are fetched directly from yfinance
  - 3h and 72h intervals are resampled from 1h data
  - Hourly data from yfinance is typically limited to ~60 days
  - Daily data can go back much further (years)
        """
    )
    
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol (default: GOOGL)")
    parser.add_argument("--intervals", nargs="+", default=["1h", "3h", "24h", "72h"], 
                       choices=["1h", "3h", "24h", "1d", "72h"],
                       help="Intervals to populate (default: all)")
    parser.add_argument("--period", type=str, default="2y", 
                       help="Period to fetch from yfinance (default: 2y). For hourly data, max is typically 60d")
    
    args = parser.parse_args()
    
    # Normalize 1d to 24h
    intervals = ["24h" if i == "1d" else i for i in args.intervals]
    
    try:
        results = populate_googl_data(
            symbol=args.symbol,
            intervals=intervals,
            period=args.period
        )
        
        # Check if all succeeded
        all_success = all(r.get("status") == "success" for r in results.values())
        if all_success:
            print("\n✅ All intervals populated successfully!")
        else:
            print("\n⚠️  Some intervals failed. Check errors above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

