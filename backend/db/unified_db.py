"""
Unified database layer using MongoDB exclusively.
Provides a consistent interface for data persistence with MongoDB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne

from ..config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# MongoDB setup
_mongo_client: Optional[MongoClient] = None
_mongo_db = None


def get_mongo_db():
    """Get MongoDB database instance (required)."""
    global _mongo_client, _mongo_db
    
    mongo_uri = getattr(settings, 'mongo_uri', None)
    mongo_db_name = getattr(settings, 'mongo_db', 'forecast_db')
    
    if not mongo_uri:
        raise RuntimeError(
            "MongoDB URI not configured. Please set FORECAST_MONGO_URI environment variable. "
            "Example: mongodb://localhost:27017/"
        )
    
    if _mongo_client is None:
        _mongo_client = MongoClient(mongo_uri, appname="forecast_dash_pro")
        _mongo_db = _mongo_client[mongo_db_name]
        
        # Create indexes for all collections
        _mongo_db.prices.create_index([("symbol", ASCENDING), ("date", ASCENDING)], unique=True)
        _mongo_db.forecasts.create_index([("symbol", ASCENDING), ("target_time", ASCENDING), ("model_version_id", ASCENDING)], unique=True)
        _mongo_db.model_versions.create_index([("symbol", ASCENDING), ("model_type", ASCENDING), ("trained_at", ASCENDING)])
        _mongo_db.model_versions.create_index([("version_tag", ASCENDING)], unique=True)
        _mongo_db.metrics.create_index([("symbol", ASCENDING), ("model_type", ASCENDING), ("evaluated_at", ASCENDING)])
        _mongo_db.portfolio_snapshots.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)])
        _mongo_db.trades.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)])
        _mongo_db.portfolios.create_index([("portfolio_id", ASCENDING)], unique=True)
        
        logger.info(f"Connected to MongoDB: {mongo_db_name}")
    
    return _mongo_db


def _ensure_datetime(value: Any) -> datetime:
    """Ensure value is a timezone-aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(value, str):
        dt = pd.to_datetime(value).to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return value


class UnifiedDataStore:
    """Unified interface for data storage using MongoDB only."""
    
    def __init__(self, use_mongo: bool = True):
        self.use_mongo = use_mongo
        if not use_mongo:
            raise ValueError("MongoDB is required. SQLite fallback is not supported.")
        self.mongo_db = get_mongo_db()
        if self.mongo_db is None:
            raise RuntimeError("MongoDB connection failed. Please ensure MongoDB is running and configured.")
    
    def save_prices(self, symbol: str, prices: pd.DataFrame, interval: Optional[str] = None):
        """Save price data to MongoDB."""
        if prices.empty:
            return
        
        # Prepare records
        records = []
        for _, row in prices.iterrows():
            date_val = _ensure_datetime(row.get("date", row.name))
            # Skip invalid dates (like 1970-01-01 which indicates parsing errors)
            if date_val < datetime(2000, 1, 1, tzinfo=timezone.utc):
                continue
                
            record = {
                "symbol": symbol,
                "date": date_val,
                "open": float(row.get("Open", row.get("open", 0))),
                "high": float(row.get("High", row.get("high", 0))),
                "low": float(row.get("Low", row.get("low", 0))),
                "close": float(row.get("Close", row.get("close", 0))),
                "volume": float(row.get("Volume", row.get("volume", 0))),
                "updated_at": datetime.now(timezone.utc),
            }
            # Store interval if provided (helps identify data frequency)
            if interval:
                record["_interval"] = interval
            elif "_interval" in row:
                record["_interval"] = row["_interval"]
            
            records.append(record)
        
        if not records:
            logger.warning(f"No valid records to save for {symbol}")
            return
        
        # Save to MongoDB
        # Use symbol + date as unique key (MongoDB index ensures uniqueness)
        # The _interval field helps identify data type but doesn't affect uniqueness
        ops = []
        for r in records:
            key = {"symbol": r["symbol"], "date": r["date"]}
            ops.append(UpdateOne(key, {"$set": r}, upsert=True))
        if ops:
            self.mongo_db.prices.bulk_write(ops, ordered=False)
            logger.info(f"Saved {len(records)} price records to MongoDB for {symbol} (interval: {interval or 'unknown'})")
    
    def fetch_prices(self, symbol: str, limit_days: Optional[int] = None, prefer_interval: Optional[str] = "1h") -> pd.DataFrame:
        """
        Fetch price data from MongoDB.
        PREFERS hourly data (1h interval) by default.
        
        Args:
            symbol: Stock symbol
            limit_days: Optional limit in days
            prefer_interval: Preferred interval ('1h' for hourly, '1d' for daily). Default: '1h'
        """
        try:
            query = {"symbol": symbol}
            
            # Prefer hourly data if available
            if prefer_interval:
                # First try to get data with preferred interval
                query_with_interval = {**query, "_interval": prefer_interval}
                cursor = self.mongo_db.prices.find(query_with_interval).sort("date", 1)
                rows = list(cursor)
                
                # If no data with preferred interval, fall back to all data
                if not rows:
                    cursor = self.mongo_db.prices.find(query).sort("date", 1)
                    rows = list(cursor)
            else:
                cursor = self.mongo_db.prices.find(query).sort("date", 1)
                rows = list(cursor)
            
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                # Remove invalid dates
                df = df[df["date"] > pd.Timestamp('2000-01-01', tz='UTC')]
                
                # If limit_days is specified, filter to last N days
                if limit_days is not None and limit_days > 0:
                    if len(df) > 0:
                        latest_date = df["date"].max()
                        cutoff_date = latest_date - pd.Timedelta(days=limit_days)
                        df = df[df["date"] >= cutoff_date]
                return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"MongoDB fetch failed: {e}")
            raise
        
        return pd.DataFrame()
    
    def save_forecast(
        self,
        symbol: str,
        model_type: str,
        predictions: List[Dict[str, Any]],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Save forecast predictions to MongoDB."""
        doc = {
            "symbol": symbol,
            "model_type": model_type,
            "created_at": datetime.now(timezone.utc),
            "predictions": predictions,
            "metrics": metrics or {},
        }
        
        self.mongo_db.forecasts.insert_one(doc)
        logger.info(f"Saved forecast to MongoDB for {symbol} using {model_type}")
    
    def save_model_metadata(
        self,
        symbol: str,
        model_type: str,
        version_tag: str,
        training_info: Dict[str, Any],
    ):
        """Save model metadata to MongoDB."""
        doc = {
            "symbol": symbol,
            "model_type": model_type,
            "version_tag": version_tag,
            "trained_at": datetime.now(timezone.utc),
            "training_info": training_info,
        }
        
        self.mongo_db.models.insert_one(doc)
        logger.info(f"Saved model metadata to MongoDB: {symbol}/{model_type}/{version_tag}")
    
    def save_metrics(
        self,
        symbol: str,
        model_type: str,
        metrics: Dict[str, float],
        evaluated_at: Optional[datetime] = None,
        model_version_id: Optional[str] = None,
    ):
        """Save evaluation metrics to MongoDB."""
        doc = {
            "symbol": symbol,
            "model_type": model_type,
            "evaluated_at": evaluated_at or datetime.now(timezone.utc),
            "metrics": metrics,
        }
        if model_version_id:
            doc["model_version_id"] = model_version_id
        
        self.mongo_db.metrics.insert_one(doc)
        logger.info(f"Saved metrics to MongoDB for {symbol}/{model_type}")
    
    def save_model_version(
        self,
        symbol: str,
        model_type: str,
        horizon: int,
        version_tag: str,
        artifact_path: str,
        training_info: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        version_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Save model version metadata to MongoDB. Returns the document _id as string."""
        doc = {
            "symbol": symbol,
            "model_type": model_type,
            "horizon": horizon,
            "version_tag": version_tag,
            "trained_at": datetime.now(timezone.utc),
            "artifact_path": artifact_path,
            "training_info": training_info or {},
            "source": source,
            "source_url": source_url,
            "version_id": version_id,
            "notes": notes,
        }
        
        result = self.mongo_db.model_versions.insert_one(doc)
        logger.info(f"Saved model version to MongoDB: {symbol}/{model_type}/{version_tag} (id: {result.inserted_id})")
        return str(result.inserted_id)
    
    def get_model_version(self, model_version_id: str) -> Optional[Dict[str, Any]]:
        """Get model version by ID."""
        from bson import ObjectId
        try:
            doc = self.mongo_db.model_versions.find_one({"_id": ObjectId(model_version_id)})
            if doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            return doc
        except Exception as e:
            logger.error(f"Error getting model version {model_version_id}: {e}")
            return None
    
    def get_latest_model_version(
        self,
        symbol: str,
        model_type: str,
        horizon: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get latest model version for symbol and model_type."""
        query = {"symbol": symbol, "model_type": model_type}
        cursor = self.mongo_db.model_versions.find(query).sort("trained_at", -1).limit(1)
        # Use next() with default None to avoid StopIteration
        doc = next(cursor, None)
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            # Filter by horizon if specified
            if horizon is not None and doc.get("horizon") != horizon:
                # Try to find one with matching horizon, but if not found, use the one we have
                # (models are flexible about horizon)
                query["horizon"] = horizon
                cursor = self.mongo_db.model_versions.find(query).sort("trained_at", -1).limit(1)
                horizon_doc = next(cursor, None)
                if horizon_doc:
                    doc = horizon_doc
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
        return doc
    
    def list_model_versions(
        self,
        symbol: str,
        model_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all model versions for a symbol."""
        query = {"symbol": symbol}
        if model_type:
            query["model_type"] = model_type
        cursor = self.mongo_db.model_versions.find(query).sort("trained_at", -1)
        models = []
        for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            models.append(doc)
        return models
    
    def save_forecast_record(
        self,
        symbol: str,
        target_time: datetime,
        horizon: int,
        predicted_value: float,
        model_version_id: str,
        actual_value: Optional[float] = None,
        absolute_error: Optional[float] = None,
        percentage_error: Optional[float] = None,
    ):
        """Save individual forecast record to MongoDB."""
        doc = {
            "symbol": symbol,
            "created_at": datetime.now(timezone.utc),
            "target_time": _ensure_datetime(target_time),
            "horizon": horizon,
            "predicted_value": float(predicted_value),
            "model_version_id": model_version_id,
            "actual_value": actual_value,
            "absolute_error": absolute_error,
            "percentage_error": percentage_error,
        }
        
        # Use upsert to avoid duplicates
        key = {"symbol": symbol, "target_time": doc["target_time"], "model_version_id": model_version_id}
        self.mongo_db.forecasts.update_one(key, {"$set": doc}, upsert=True)
        logger.debug(f"Saved forecast record to MongoDB for {symbol} at {target_time}")
    
    def save_portfolio_snapshot(
        self,
        symbol: str,
        cash: float,
        holdings: float,
        total_value: float,
        daily_return: float,
        volatility: float,
        sharpe_ratio: float,
        timestamp: Optional[datetime] = None,
    ):
        """Save portfolio snapshot to MongoDB."""
        doc = {
            "symbol": symbol,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "cash": float(cash),
            "holdings": float(holdings),
            "total_value": float(total_value),
            "daily_return": float(daily_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
        }
        
        self.mongo_db.portfolio_snapshots.insert_one(doc)
        logger.debug(f"Saved portfolio snapshot to MongoDB for {symbol}")
    
    def get_portfolio_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio snapshot history."""
        cursor = self.mongo_db.portfolio_snapshots.find({"symbol": symbol}).sort("timestamp", -1).limit(limit)
        snapshots = []
        for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            snapshots.append(doc)
        return list(reversed(snapshots))  # Return in chronological order
    
    def save_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        note: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Save trade to MongoDB. Returns the document _id as string."""
        doc = {
            "symbol": symbol,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "action": action,
            "quantity": float(quantity),
            "price": float(price),
            "note": note,
        }
        
        result = self.mongo_db.trades.insert_one(doc)
        logger.info(f"Saved trade to MongoDB for {symbol}: {action} {quantity} @ {price}")
        return str(result.inserted_id)
    
    def get_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trades for a symbol."""
        cursor = self.mongo_db.trades.find({"symbol": symbol}).sort("timestamp", -1).limit(limit)
        trades = []
        for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            trades.append(doc)
        return trades
    
    def save_portfolio_state(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Save portfolio state to MongoDB.
        
        Args:
            portfolio_data: Dictionary with portfolio_id, cash_balance, positions, etc.
        """
        portfolio_id = portfolio_data.get("portfolio_id", "default")
        
        doc = {
            "portfolio_id": portfolio_id,
            "cash_balance": float(portfolio_data.get("cash_balance", 0)),
            "positions": portfolio_data.get("positions", {}),
            "total_value": float(portfolio_data.get("total_value", 0)),
            "last_updated": portfolio_data.get("last_updated", datetime.now(timezone.utc)),
        }
        
        # Upsert by portfolio_id
        self.mongo_db.portfolios.update_one(
            {"portfolio_id": portfolio_id},
            {"$set": doc},
            upsert=True
        )
        logger.debug(f"Saved portfolio state for {portfolio_id}")
    
    def get_portfolio_state(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Get portfolio state from MongoDB.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Portfolio state dictionary or None if not found
        """
        doc = self.mongo_db.portfolios.find_one({"portfolio_id": portfolio_id})
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            return doc
        return None


__all__ = ["get_mongo_db", "UnifiedDataStore"]

