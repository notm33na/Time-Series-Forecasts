"""
Data ingestion API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd

from ..services.data_ingestion import DataIngestionService
from ..config import get_settings
from ..utils.logging import get_logger

settings = get_settings()

logger = get_logger(__name__)
router = APIRouter(prefix="/api/data", tags=["data"])


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol format for database lookup.
    Converts forex symbols from EURUSD to EURUSD=X format if needed.
    Also handles the reverse (EURUSD=X -> EURUSD=X, no change).
    
    Args:
        symbol: Symbol in any format (EURUSD, EURUSD=X, etc.)
    
    Returns:
        Normalized symbol for database lookup
    """
    # Common forex pairs that need =X suffix
    forex_pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "GBPJPY", "AUDJPY", "CADJPY",
        "CHFJPY", "NZDJPY", "AUDNZD", "AUDCAD", "AUDCHF", "CADCHF", "EURNZD",
        "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "NZDCAD", "NZDCHF"
    ]
    
    # If symbol already has =X suffix, return as-is
    if symbol.endswith("=X"):
        return symbol
    
    # If symbol is a known forex pair without =X, add it
    if symbol.upper() in [fp.upper() for fp in forex_pairs]:
        return f"{symbol.upper()}=X"
    
    # For other symbols, return as-is (stocks, crypto, etc.)
    return symbol.upper()


class IngestRequest(BaseModel):
    symbol: str
    period: str = "30d"  # Default: 30 days (reliable period for hourly data)
    interval: str = "1h"  # DEFAULT: 1 hour - all data is stored in hourly format


@router.post("/ingest")
async def ingest_data(request: IngestRequest):
    """Ingest data from Yahoo Finance. Supports hourly intervals (e.g., interval='1h' or '60m')."""
    try:
        service = DataIngestionService()
        count = service.ingest_from_yfinance(request.symbol, request.period, request.interval)
        return {
            "symbol": request.symbol, 
            "records_ingested": count, 
            "interval": request.interval,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error ingesting data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prices/{symbol}")
async def get_prices(symbol: str, limit: int = 1000):
    """Get price history for a symbol."""
    try:
        # Normalize symbol for database lookup (e.g., EURUSD -> EURUSD=X for forex)
        normalized_symbol = normalize_symbol(symbol)
        logger.info(f"Normalized symbol: {symbol} -> {normalized_symbol}")
        
        service = DataIngestionService()
        # Try normalized symbol first
        df = service.get_prices(normalized_symbol, limit=limit)
        
        # If no data found with normalized symbol, try original symbol (for backward compatibility)
        if df.empty and normalized_symbol != symbol.upper():
            logger.info(f"No data found for {normalized_symbol}, trying original symbol {symbol}")
            df = service.get_prices(symbol, limit=limit)
        
        if df.empty:
            return {"symbol": symbol, "prices": []}

        # Handle different date column names
        date_col = 'date' if 'date' in df.columns else df.index
        
        prices = []
        for idx, row in df.iterrows():
            date_val = row.get('date', idx) if 'date' in df.columns else idx
            prices.append({
                "timestamp": pd.Timestamp(date_val).isoformat() if hasattr(date_val, 'isoformat') else str(date_val),
                "open": float(row.get("Open", row.get("open", 0))),
                "high": float(row.get("High", row.get("high", 0))),
                "low": float(row.get("Low", row.get("low", 0))),
                "close": float(row.get("Close", row.get("close", 0))),
                "volume": float(row.get("Volume", row.get("volume", 0))),
            })
        
        return {"symbol": symbol, "prices": prices}
    except Exception as e:
        logger.error(f"Error getting prices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

