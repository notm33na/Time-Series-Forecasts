"""
Script to populate the database with historical price data from Yahoo Finance.

Usage:
    python -m backend.scripts.populate_database --symbol AAPL --period 2y
    python -m backend.scripts.populate_database --symbol AAPL,MSFT,GOOGL --period 5y
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.data_ingestion import DataIngestionService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def populate_symbol(symbol: str, period: str = "2y") -> int:
    """
    Populate database with data for a single symbol.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        period: Time period (e.g., "2y", "5y", "1mo", "max")
    
    Returns:
        Number of records ingested
    """
    try:
        service = DataIngestionService()
        count = service.ingest_from_yfinance(symbol, period=period)
        logger.info(f"✓ Ingested {count} records for {symbol}")
        return count
    except Exception as e:
        logger.error(f"✗ Failed to ingest {symbol}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Populate database with historical price data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate single symbol
  python -m backend.scripts.populate_database --symbol AAPL --period 2y
  
  # Populate multiple symbols
  python -m backend.scripts.populate_database --symbol AAPL,MSFT,GOOGL --period 5y
  
  # Populate with maximum history
  python -m backend.scripts.populate_database --symbol AAPL --period max
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol(s), comma-separated (e.g., 'AAPL' or 'AAPL,MSFT,GOOGL')"
    )
    
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Time period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max' (default: 2y)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    
    print(f"Populating database with {len(symbols)} symbol(s)...")
    print(f"Period: {args.period}")
    print()
    
    total_records = 0
    successful = 0
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        count = populate_symbol(symbol, period=args.period)
        if count > 0:
            total_records += count
            successful += 1
        print()
    
    print(f"✓ Successfully populated {successful}/{len(symbols)} symbols")
    print(f"✓ Total records ingested: {total_records}")
    
    if successful < len(symbols):
        print(f"⚠ {len(symbols) - successful} symbol(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

