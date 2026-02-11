"""
Data ingestion script for Forex and Crypto pairs.
Ingests historical data from Yahoo Finance and stores in MongoDB.

Forex pairs: EURUSD=X, GBPUSD=X
Crypto pairs: BTC-USD, ETH-USD

For model training, we use daily data (1d interval) with maximum available period.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Forex and Crypto symbols
FOREX_SYMBOLS = [
    "EURUSD=X",  # EUR/USD
    "GBPUSD=X",  # GBP/USD
]

CRYPTO_SYMBOLS = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
]

ALL_SYMBOLS = FOREX_SYMBOLS + CRYPTO_SYMBOLS


def ingest_symbol(symbol: str, period: str = "2y", interval: str = "1d") -> dict:
    """
    Ingest data for a single symbol.
    
    Args:
        symbol: Trading pair symbol
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1d' for daily, '1h' for hourly, etc.)
    
    Returns:
        dict with ingestion results
    """
    try:
        print(f"\n{'='*70}")
        print(f"Ingesting data for {symbol}")
        print(f"{'='*70}")
        print(f"  Period: {period}")
        print(f"  Interval: {interval}")
        print()
        
        service = DataIngestionService()
        count = service.ingest_from_yfinance(symbol, period=period, interval=interval)
        
        if count > 0:
            # Verify data was ingested
            df = service.get_prices(symbol, limit=10)
            if not df.empty:
                print(f"✓ Successfully ingested {count} records for {symbol}")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                print(f"  Latest price: ${df['close'].iloc[-1]:.2f}")
                return {
                    "symbol": symbol,
                    "success": True,
                    "records": count,
                    "date_range": (str(df['date'].min()), str(df['date'].max())),
                    "price_range": (float(df['close'].min()), float(df['close'].max())),
                }
            else:
                print(f"⚠️  Ingested {count} records but could not retrieve for verification")
                return {
                    "symbol": symbol,
                    "success": True,
                    "records": count,
                    "date_range": None,
                    "price_range": None,
                }
        else:
            print(f"⚠️  No data ingested for {symbol}")
            return {
                "symbol": symbol,
                "success": False,
                "records": 0,
                "date_range": None,
                "price_range": None,
            }
            
    except Exception as e:
        print(f"✗ Failed to ingest {symbol}: {e}")
        logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "success": False,
            "records": 0,
            "error": str(e),
        }


def ingest_all(
    symbols: list = None,
    period: str = "2y",
    interval: str = "1d",
    forex_only: bool = False,
    crypto_only: bool = False
):
    """
    Ingest data for all specified symbols.
    
    Args:
        symbols: List of symbols to ingest (default: ALL_SYMBOLS)
        period: Data period (default: "2y" for 2 years)
        interval: Data interval (default: "1d" for daily data)
        forex_only: Only ingest forex pairs
        crypto_only: Only ingest crypto pairs
    """
    if forex_only:
        symbols = FOREX_SYMBOLS
    elif crypto_only:
        symbols = CRYPTO_SYMBOLS
    elif symbols is None:
        symbols = ALL_SYMBOLS
    
    print("="*70)
    print("FOREX & CRYPTO DATA INGESTION")
    print("="*70)
    print(f"Forex Pairs: {', '.join(FOREX_SYMBOLS)}")
    print(f"Crypto Pairs: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"Symbols to ingest: {', '.join(symbols)}")
    print(f"Period: {period}")
    print(f"Interval: {interval}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    total_records = 0
    successful = 0
    failed = 0
    
    for symbol in symbols:
        result = ingest_symbol(symbol, period=period, interval=interval)
        results.append(result)
        
        if result["success"]:
            total_records += result["records"]
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print("INGESTION SUMMARY")
    print("="*70)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total records ingested: {total_records}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Detailed results
    print("\nDetailed Results:")
    print("-"*70)
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} {result['symbol']}")
        if result["success"]:
            print(f"   Records: {result['records']}")
            if result.get("date_range"):
                print(f"   Date range: {result['date_range'][0]} to {result['date_range'][1]}")
            if result.get("price_range"):
                print(f"   Price range: ${result['price_range'][0]:.2f} - ${result['price_range'][1]:.2f}")
        else:
            if result.get("error"):
                print(f"   Error: {result['error']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest Forex and Crypto data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all forex and crypto pairs (default: 2y daily data)
  python scripts/ingest_forex_crypto.py

  # Ingest with maximum available period
  python scripts/ingest_forex_crypto.py --period max

  # Ingest only forex pairs
  python scripts/ingest_forex_crypto.py --forex-only

  # Ingest only crypto pairs
  python scripts/ingest_forex_crypto.py --crypto-only

  # Ingest specific symbols
  python scripts/ingest_forex_crypto.py --symbols EURUSD=X,BTC-USD

  # Ingest hourly data (for more granular analysis)
  python scripts/ingest_forex_crypto.py --interval 1h --period 30d

Note: For model training, daily data (1d) with 2y period is recommended.
      Hourly data (1h) is useful for shorter-term analysis but requires more storage.
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: all forex and crypto)"
    )
    
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max' (default: 2y)"
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval: '1d' for daily, '1h' for hourly, etc. (default: 1d for training)"
    )
    
    parser.add_argument(
        "--forex-only",
        action="store_true",
        help="Ingest only forex pairs"
    )
    
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Ingest only crypto pairs"
    )
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Validate period and interval
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    if args.period not in valid_periods:
        print(f"⚠️  Warning: Period '{args.period}' may not be valid. Valid periods: {', '.join(valid_periods)}")
    
    # Run ingestion
    results = ingest_all(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        forex_only=args.forex_only,
        crypto_only=args.crypto_only
    )
    
    # Exit with error code if any failed
    failed_count = sum(1 for r in results if not r["success"])
    
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} symbol(s) failed to ingest")
        sys.exit(1)
    else:
        print("\n✅ All data ingestion completed successfully!")
        print("\nNext step: Train models using:")
        print("  python scripts/train_forex_crypto.py")
        sys.exit(0)

