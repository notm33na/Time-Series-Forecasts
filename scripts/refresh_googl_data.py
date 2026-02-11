"""
Script to refresh GOOGL data in MongoDB with the most recent prices.
Fetches maximum available data from yfinance to ensure training data includes current prices.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def refresh_googl_data(symbol: str = "GOOGL", period: str = "max", interval: str = "1d", also_refresh_hourly: bool = True):
    """
    Refresh data for a symbol by fetching from yfinance and storing in MongoDB.
    Always ingests both daily and hourly data to ensure complete coverage.
    
    Args:
        symbol: Stock symbol (default: GOOGL)
        period: Data period for daily data - use "max" for maximum available, "2y" for 2 years, etc.
        interval: Primary data interval - use "1d" for daily (better for training), "1h" for hourly
        also_refresh_hourly: Always True - always refreshes hourly data for recent prices
    """
    print(f"\n{'='*70}")
    print(f"Refreshing {symbol} Data in MongoDB")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {period} (maximum available data)")
    print(f"  Interval: {interval} ({'daily' if interval == '1d' else 'hourly' if interval == '1h' else interval})")
    print()
    
    try:
        service = DataIngestionService()
        
        # Check current data
        print("Checking current data in MongoDB...")
        current_df = service.get_prices(symbol)
        if not current_df.empty:
            print(f"  Current records: {len(current_df)}")
            print(f"  Date range: {current_df['date'].min()} to {current_df['date'].max()}")
            print(f"  Price range: ${current_df['close'].min():.2f} - ${current_df['close'].max():.2f}")
            print(f"  Latest price: ${current_df['close'].iloc[-1]:.2f}")
        else:
            print("  No existing data found")
        
        print(f"\n📥 Fetching fresh data from yfinance...")
        print(f"   This may take a moment for '{period}' period...")
        
        # Fetch and ingest data
        # For training, we want daily data with maximum period to get all historical prices
        records_ingested = service.ingest_from_yfinance(
            symbol=symbol,
            period=period,
            interval=interval
        )
        
        print(f"\n✅ Successfully ingested {records_ingested} records")
        
        # Verify new data
        print("\nVerifying updated data...")
        new_df = service.get_prices(symbol)
        if not new_df.empty:
            print(f"  Total records: {len(new_df)}")
            print(f"  Date range: {new_df['date'].min()} to {new_df['date'].max()}")
            print(f"  Price range: ${new_df['close'].min():.2f} - ${new_df['close'].max():.2f}")
            print(f"  Latest price: ${new_df['close'].iloc[-1]:.2f}")
            print(f"\n  Last 5 prices:")
            last_5 = new_df[['date', 'close']].tail(5)
            for _, row in last_5.iterrows():
                print(f"    {row['date'].strftime('%Y-%m-%d')}: ${row['close']:.2f}")
        
        # Always refresh hourly data for more recent prices
        # Hourly data provides more granular recent data and ensures we have the latest prices
        if also_refresh_hourly:
            print(f"\n📥 Refreshing hourly data (for more recent prices)...")
            hourly_records = service.ingest_from_yfinance(
                symbol=symbol,
                period="30d",  # 30 days is reliable for hourly data from yfinance
                interval="1h"
            )
            print(f"✅ Ingested {hourly_records} hourly records")
            
            # Verify hourly data
            hourly_df = service.store.fetch_prices(symbol, limit_days=None, prefer_interval="1h")
            if not hourly_df.empty:
                print(f"  Hourly data: {len(hourly_df)} records")
                print(f"  Hourly date range: {hourly_df['date'].min()} to {hourly_df['date'].max()}")
                print(f"  Hourly price range: ${hourly_df['close'].min():.2f} - ${hourly_df['close'].max():.2f}")
        else:
            print(f"\n⚠️  Skipping hourly data refresh (use --skip-hourly to explicitly skip)")
        
        print(f"\n{'='*70}")
        print(f"✅ Data Refresh Complete!")
        print(f"{'='*70}\n")
        
        return records_ingested
        
    except Exception as e:
        print(f"\n❌ Error refreshing data: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refresh stock data in MongoDB")
    parser.add_argument("--symbol", type=str, default="GOOGL", help="Stock symbol")
    parser.add_argument("--period", type=str, default="max", 
                       help="Data period: '1y', '2y', '5y', 'max' (default: max)")
    parser.add_argument("--interval", type=str, default="1d",
                       help="Primary data interval: '1d' for daily (recommended for training), '1h' for hourly")
    parser.add_argument("--skip-hourly", action="store_true",
                       help="Skip hourly data refresh (not recommended - hourly data provides latest prices)")
    
    args = parser.parse_args()
    
    # Always refresh hourly unless explicitly skipped
    refresh_hourly = not args.skip_hourly
    refresh_googl_data(args.symbol, args.period, args.interval, refresh_hourly)

