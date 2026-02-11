"""
Unified data ingestion service using MongoDB exclusively.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from ..db import UnifiedDataStore
from ..config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DataIngestionService:
    """Service for ingesting and storing market data using MongoDB only."""

    def __init__(self, use_mongo: Optional[bool] = None):
        """Initialize with MongoDB (required)."""
        use_mongo = use_mongo if use_mongo is not None else settings.use_mongo
        if not use_mongo:
            raise ValueError("MongoDB is required. Please set FORECAST_USE_MONGO=true and configure FORECAST_MONGO_URI")
        self.store = UnifiedDataStore(use_mongo=True)

    def ingest_from_yfinance(self, symbol: str, period: str = "30d", interval: str = "1h") -> int:
        """
        Fetch data from Yahoo Finance and store in database(s).
        DEFAULT: Uses 1-hour interval for all data ingestion.
        
        Args:
            symbol: Stock/crypto/forex symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
                    Default: '30d' (30 days) - reliable period for hourly data from yfinance
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
                    DEFAULT: '1h' (1 hour) - all data is stored in hourly format
                    For hourly data, use '1h' or '60m'
        
        Returns number of records ingested.
        """
        logger.info(f"Ingesting data for {symbol} (period: {period}, interval: {interval})")

        try:
            ticker = yf.Ticker(symbol)
            # Use interval parameter to support hourly data
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return 0

            # Reset index to make date a column
            df = df.reset_index()
            
            # Handle different date column names from yfinance
            # yfinance uses 'Datetime' for intraday data, 'Date' for daily data
            if 'Datetime' in df.columns:
                df['date'] = df['Datetime']
            elif 'Date' in df.columns:
                df['date'] = df['Date']
            elif df.index.name in ['Datetime', 'Date', 'date']:
                # If the index was a datetime index, it should be preserved
                df['date'] = df.index
            else:
                # Last resort: check if index is datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df['date'] = df.index
                else:
                    raise ValueError(f"Could not find date column. Available columns: {list(df.columns)}, index: {df.index.name}")
            
            # Ensure date column is datetime and handle timezone
            df['date'] = pd.to_datetime(df['date'], utc=True)
            # Remove any invalid dates (like 1970-01-01 which indicates a parsing error)
            df = df[df['date'] > pd.Timestamp('2000-01-01', tz='UTC')]
            
            if df.empty:
                logger.warning(f"No valid data after filtering for {symbol}")
                return 0
            
            # Normalize column names to ensure consistency
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            for col in ohlc_cols:
                if col.lower() in df.columns and col not in df.columns:
                    df[col] = df[col.lower()]
            
            # Validate and fix OHLC data quality
            df = self._fix_ohlc_variation(df, symbol)
            df = self._fix_zero_volume(df, symbol)
            
            # For hourly or higher frequency data, normalize timestamps to avoid conflicts
            # yfinance sometimes returns daily data at 5:00 AM UTC which can conflict with hourly data
            # We'll store the interval in metadata to help identify data type
            df['_interval'] = interval  # Store interval for reference
            
            # Save using unified store (MongoDB)
            # IMPORTANT: Clear existing data for this symbol/interval combination to avoid overwrites
            # This ensures we don't mix hourly and daily data
            logger.info(f"Clearing existing {interval} data for {symbol} before ingesting new data...")
            self.store.mongo_db.prices.delete_many({
                "symbol": symbol,
                "_interval": interval
            })
            
            self.store.save_prices(symbol, df, interval=interval)

            logger.info(f"Ingested {len(df)} records for {symbol} to MongoDB (interval: {interval})")
            
            # Trigger automatic evaluation for this symbol
            try:
                from .evaluation_service import EvaluationService
                evaluated_count = EvaluationService.auto_evaluate_pending(symbol)
                if evaluated_count > 0:
                    logger.info(f"Auto-evaluated {evaluated_count} forecasts for {symbol} after data ingestion")
            except Exception as e:
                logger.warning(f"Could not auto-evaluate after ingestion: {e}")
            
            return len(df)

        except Exception as e:
            logger.error(f"Error ingesting data for {symbol}: {e}", exc_info=True)
            raise

    def ingest_from_dataframe(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Ingest data from a pandas DataFrame.
        DataFrame should have columns: Open, High, Low, Close, Volume, and DatetimeIndex.
        """
        logger.info(f"Ingesting {len(df)} records from DataFrame for {symbol}")

        # Prepare DataFrame
        if df.index.name != 'date' and 'date' not in df.columns:
            df = df.reset_index()
            if 'Date' in df.columns:
                df['date'] = df['Date']
            elif df.index.name:
                df['date'] = df.index

        # Save using unified store (MongoDB)
        self.store.save_prices(symbol, df)

        logger.info(f"Ingested {len(df)} records for {symbol} to MongoDB")
        
        # Trigger automatic evaluation for this symbol
        try:
            from .evaluation_service import EvaluationService
            evaluated_count = EvaluationService.auto_evaluate_pending(symbol)
            if evaluated_count > 0:
                logger.info(f"Auto-evaluated {evaluated_count} forecasts for {symbol} after data ingestion")
        except Exception as e:
            logger.warning(f"Could not auto-evaluate after ingestion: {e}")
        
        return len(df)

    def get_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve prices from MongoDB as DataFrame.
        PREFERS hourly data (1h interval) by default.
        
        Returns data sorted by date (ascending, oldest to newest).
        """
        # Fetch from MongoDB - prefer hourly data (1h interval)
        # limit_days parameter in fetch_prices is for limiting by days, not records
        # For training, we want all data, so pass None
        df = self.store.fetch_prices(symbol, limit_days=None, prefer_interval="1h")
        
        if not df.empty:
            # Ensure date column exists and is datetime
            if 'date' not in df.columns:
                logger.warning(f"No 'date' column found for {symbol}. Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Remove any duplicate dates (keep the last one)
            if df['date'].duplicated().any():
                logger.warning(f"Found duplicate dates for {symbol}. Removing duplicates (keeping last).")
                df = df.drop_duplicates(subset=['date'], keep='last')
            
            # Sort by date to ensure chronological order (oldest to newest)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Filter by date range if provided
            if start_date or end_date:
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date, utc=True)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date, utc=True)]
            
            # Apply limit if specified (limit number of records, not days)
            # tail() gets the most recent records since data is sorted ascending
            if limit is not None and limit > 0:
                df = df.tail(limit)
            
            # Final sort to ensure order (in case tail() changed it)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Validate data quality
            if len(df) > 0:
                # Check for missing values in critical columns
                critical_cols = ['open', 'high', 'low', 'close']
                missing = df[critical_cols].isnull().sum()
                if missing.any():
                    logger.warning(f"Found missing values for {symbol}: {missing.to_dict()}")
                    # Fill missing values with forward fill, then backward fill
                    df[critical_cols] = df[critical_cols].ffill().bfill()
                
                # Check for invalid prices (zero or negative)
                invalid_prices = (df[critical_cols] <= 0).any(axis=1)
                if invalid_prices.any():
                    logger.warning(f"Found {invalid_prices.sum()} rows with invalid prices (<=0) for {symbol}. Removing them.")
                    df = df[~invalid_prices]
                
                # Validate OHLC relationships
                invalid_ohlc = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                )
                if invalid_ohlc.any():
                    logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships for {symbol}. Removing them.")
                    df = df[~invalid_ohlc]
            
            # Validate and fix OHLC data quality on retrieval
            if len(df) > 0:
                df = self._fix_ohlc_variation(df, symbol)
                df = self._fix_zero_volume(df, symbol)
            
            return df
        
        return pd.DataFrame()
    
    def _fix_ohlc_variation(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Fix OHLC values that are all identical (flat candles).
        This can happen when yfinance returns data with no price variation.
        
        Strategy:
        1. Detect rows where Open=High=Low=Close
        2. For such rows, add small variation based on:
           - Previous period's variation (if available)
           - Small percentage variation (0.1% spread) if no previous data
        3. Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        """
        if df.empty:
            return df
        
        # Normalize column names
        ohlc_map = {}
        for col in ['Open', 'High', 'Low', 'Close', 'open', 'high', 'low', 'close']:
            if col in df.columns:
                ohlc_map[col.lower()] = col
        
        # Use lowercase for consistency
        open_col = ohlc_map.get('open', 'open')
        high_col = ohlc_map.get('high', 'high')
        low_col = ohlc_map.get('low', 'low')
        close_col = ohlc_map.get('close', 'close')
        
        # Ensure we have the columns
        if not all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
            logger.warning(f"Cannot fix OHLC variation for {symbol}: missing OHLC columns")
            return df
        
        # Detect flat candles (all OHLC values identical)
        flat_mask = (
            (df[open_col] == df[high_col]) &
            (df[high_col] == df[low_col]) &
            (df[low_col] == df[close_col])
        )
        
        flat_count = flat_mask.sum()
        if flat_count == 0:
            return df  # No flat candles, return as-is
        
        logger.warning(
            f"Found {flat_count} flat candles (identical OHLC) for {symbol}. "
            f"Adding variation to improve data quality."
        )
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Fix flat candles - iterate by position to access previous row
        flat_indices = df[flat_mask].index.tolist()
        for i, idx in enumerate(flat_indices):
            base_price = df.loc[idx, close_col]
            
            # Try to use previous period's variation
            # Find position of current index
            pos = df.index.get_loc(idx)
            if pos > 0:
                prev_row = df.iloc[pos - 1]
                prev_high = prev_row[high_col]
                prev_low = prev_row[low_col]
                prev_close = prev_row[close_col]
                
                # Calculate previous period's spread
                if prev_high != prev_low and prev_close > 0:
                    prev_spread_pct = (prev_high - prev_low) / prev_close
                else:
                    prev_spread_pct = 0.001  # Default 0.1% spread
            else:
                # No previous data, use default small variation
                prev_spread_pct = 0.001  # 0.1% spread
            
            # Apply variation: High = base * (1 + spread/2), Low = base * (1 - spread/2)
            spread = base_price * prev_spread_pct
            df.loc[idx, high_col] = base_price + spread / 2
            df.loc[idx, low_col] = base_price - spread / 2
            
            # Ensure Open and Close are within High-Low range
            # If they're still equal, add tiny variation
            if df.loc[idx, open_col] == df.loc[idx, close_col]:
                # Small random variation to Open and Close
                tiny_var = base_price * 0.0001  # 0.01% variation
                df.loc[idx, open_col] = base_price - tiny_var / 2
                df.loc[idx, close_col] = base_price + tiny_var / 2
            
            # Final validation: ensure High >= all others and Low <= all others
            df.loc[idx, high_col] = max(
                df.loc[idx, open_col],
                df.loc[idx, high_col],
                df.loc[idx, low_col],
                df.loc[idx, close_col]
            )
            df.loc[idx, low_col] = min(
                df.loc[idx, open_col],
                df.loc[idx, high_col],
                df.loc[idx, low_col],
                df.loc[idx, close_col]
            )
        
        # Validate final OHLC relationships
        invalid_count = (
            (df[high_col] < df[low_col]) |
            (df[high_col] < df[open_col]) |
            (df[high_col] < df[close_col]) |
            (df[low_col] > df[open_col]) |
            (df[low_col] > df[close_col])
        ).sum()
        
        if invalid_count > 0:
            logger.error(
                f"After fixing OHLC variation, {invalid_count} rows still have invalid OHLC relationships for {symbol}"
            )
        else:
            logger.info(
                f"Fixed {flat_count} flat candles for {symbol}. "
                f"OHLC values now have proper variation."
            )
        
        return df
    
    def _fix_zero_volume(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Fix zero Volume values in the data.
        Zero volume can occur during after-hours trading or data quality issues.
        
        Strategy:
        1. Detect rows where Volume = 0 or is missing
        2. For such rows, use:
           - Previous period's volume (if available and > 0)
           - Average volume from recent periods (if available)
           - Default reasonable volume based on symbol type
        3. Ensure Volume is always positive
        """
        if df.empty:
            return df
        
        # Normalize column names
        volume_col = None
        for col in ['Volume', 'volume']:
            if col in df.columns:
                volume_col = col
                break
        
        if volume_col is None:
            logger.warning(f"Cannot fix zero volume for {symbol}: missing Volume column")
            return df
        
        # Detect zero or missing volume
        zero_volume_mask = (df[volume_col] == 0) | (df[volume_col].isna())
        zero_count = zero_volume_mask.sum()
        
        if zero_count == 0:
            return df  # No zero volumes, return as-is
        
        logger.warning(
            f"Found {zero_count} rows with zero/missing volume for {symbol}. "
            f"Fixing with reasonable defaults."
        )
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate average volume from non-zero periods
        non_zero_volumes = df[df[volume_col] > 0][volume_col]
        avg_volume = non_zero_volumes.mean() if len(non_zero_volumes) > 0 else None
        
        # Fix zero volumes
        for idx in df[zero_volume_mask].index:
            # Try to use previous period's volume
            pos = df.index.get_loc(idx)
            if pos > 0:
                prev_volume = df.iloc[pos - 1][volume_col]
                if prev_volume > 0:
                    # Use previous period's volume (reasonable for after-hours continuation)
                    df.loc[idx, volume_col] = prev_volume
                    continue
            
            # Try to use next period's volume (if available)
            if pos < len(df) - 1:
                next_volume = df.iloc[pos + 1][volume_col]
                if next_volume > 0:
                    df.loc[idx, volume_col] = next_volume
                    continue
            
            # Use average volume if available
            if avg_volume is not None and avg_volume > 0:
                df.loc[idx, volume_col] = avg_volume
                continue
            
            # Last resort: use a default reasonable volume
            # For stocks, typical hourly volume might be 100k-1M shares
            # Use a conservative default
            default_volume = 100000.0  # 100k shares default
            df.loc[idx, volume_col] = default_volume
            logger.debug(
                f"Using default volume {default_volume} for {symbol} at {df.loc[idx, 'date'] if 'date' in df.columns else idx}"
            )
        
        # Validate final volume
        still_zero = (df[volume_col] <= 0).sum()
        if still_zero > 0:
            logger.error(
                f"After fixing zero volume, {still_zero} rows still have zero/negative volume for {symbol}"
            )
        else:
            logger.info(
                f"Fixed {zero_count} zero volume rows for {symbol}. "
                f"All volumes are now positive."
            )
        
        return df

    def _get_latest_price_instance(self, symbol: str) -> Optional[dict]:
        """Instance method to get the most recent price for a symbol from MongoDB."""
        try:
            latest = self.store.mongo_db.prices.find_one(
                {"symbol": symbol},
                sort=[("date", -1)]
            )
            if latest:
                return {
                    "symbol": latest["symbol"],
                    "timestamp": latest["date"],
                    "open": latest["open"],
                    "high": latest["high"],
                    "low": latest["low"],
                    "close": latest["close"],
                    "volume": latest["volume"],
                }
        except Exception as e:
            logger.error(f"MongoDB fetch failed: {e}")
        return None
    
    @staticmethod
    def get_latest_price(symbol: str) -> Optional[dict]:
        """Static method to get the most recent price for a symbol."""
        service = DataIngestionService()
        return service._get_latest_price_instance(symbol)

