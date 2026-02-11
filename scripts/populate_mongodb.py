"""
Script to populate MongoDB with historical price data from Yahoo Finance.

Usage:
    python scripts/populate_mongodb.py --symbol AAPL --period 2y --interval 1h
    python scripts/populate_mongodb.py --symbol AAPL,MSFT,GOOGL --period 5y --interval 1d
    python scripts/populate_mongodb.py --symbol AAPL --period max --interval 1d
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def populate_symbol(symbol: str, period: str = "2y", interval: str = "1d") -> int:
    """
    Populate MongoDB with data for a single symbol.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
                 For hourly data, use '1h' or '60m'
    
    Returns:
        Number of records ingested
    """
    try:
        print(f"Fetching data for {symbol} (period: {period}, interval: {interval})...")
        service = DataIngestionService()
        count = service.ingest_from_yfinance(symbol, period=period, interval=interval)
        print(f"✓ Successfully ingested {count} records for {symbol} to MongoDB")
        return count
    except Exception as e:
        print(f"✗ Failed to ingest {symbol}: {e}")
        logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with historical price data from Yahoo Finance"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol(s), comma-separated for multiple (e.g., AAPL or AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="30d",
        help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max' (default: 30d - reliable period for hourly data)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Data interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo' (default: 1h - hourly format). For hourly: '1h' or '60m'"
    )
    
    args = parser.parse_args()
    
    # Parse symbols (support comma-separated)
    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    
    print("=" * 80)
    print("MongoDB Data Population Script")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")
    print("=" * 80)
    print()
    
    total_records = 0
    successful = 0
    failed = 0
    
    for symbol in symbols:
        count = populate_symbol(symbol, period=args.period, interval=args.interval)
        if count > 0:
            total_records += count
            successful += 1
        else:
            failed += 1
        print()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total symbols processed: {len(symbols)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total records ingested: {total_records}")
    print("=" * 80)


if __name__ == "__main__":
    main()

