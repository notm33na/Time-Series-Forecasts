"""
Portfolio management service with buy/sell/hold logic.
Uses MongoDB exclusively for all data storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np

from ..config import get_settings
from ..db.unified_db import UnifiedDataStore
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PortfolioService:
    """Service for portfolio simulation and tracking."""

    def __init__(self, symbol: str, initial_cash: float = None):
        self.symbol = symbol
        self.cash = initial_cash or settings.portfolio_initial_cash
        self.holdings = 0.0
        self.trades: list[Dict[str, Any]] = []
        self.store = UnifiedDataStore()

    def get_total_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        return self.cash + (self.holdings * current_price)

    def buy(self, quantity: float, price: float, note: Optional[str] = None) -> bool:
        """Execute buy order. Saves to MongoDB."""
        cost = quantity * price
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy: need {cost:.2f}, have {self.cash:.2f}")
            return False

        self.cash -= cost
        self.holdings += quantity

        # Save trade to MongoDB
        trade_id = self.store.save_trade(
            symbol=self.symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            note=note,
        )
        
        trade = {
            "id": trade_id,
            "symbol": self.symbol,
            "timestamp": datetime.now(timezone.utc),
            "action": "BUY",
            "quantity": quantity,
            "price": price,
            "note": note,
        }
        self.trades.append(trade)

        logger.info(f"BUY {quantity:.4f} {self.symbol} @ {price:.2f}")
        return True

    def sell(self, quantity: float, price: float, note: Optional[str] = None) -> bool:
        """Execute sell order. Saves to MongoDB."""
        if quantity > self.holdings:
            logger.warning(
                f"Insufficient holdings for sell: need {quantity:.4f}, have {self.holdings:.4f}"
            )
            return False

        self.cash += quantity * price
        self.holdings -= quantity

        # Save trade to MongoDB
        trade_id = self.store.save_trade(
            symbol=self.symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            note=note,
        )
        
        trade = {
            "id": trade_id,
            "symbol": self.symbol,
            "timestamp": datetime.now(timezone.utc),
            "action": "SELL",
            "quantity": quantity,
            "price": price,
            "note": note,
        }
        self.trades.append(trade)

        logger.info(f"SELL {quantity:.4f} {self.symbol} @ {price:.2f}")
        return True

    def decide_action(
        self,
        current_price: float,
        forecast_price: float,
        threshold: float = None,
        strategy: str = "forecast",
        ma_short: Optional[float] = None,
        ma_long: Optional[float] = None,
    ) -> str:
        """
        Trading strategy decision logic.
        
        Strategies:
        - "forecast": Based on forecast vs current price
        - "ma_crossover": Moving average crossover (requires ma_short and ma_long)
        - "combined": Both forecast and MA signals
        """
        threshold = threshold or settings.error_threshold
        
        if strategy == "forecast":
            change_pct = (forecast_price - current_price) / current_price
            if change_pct > threshold:
                return "BUY"
            elif change_pct < -threshold:
                return "SELL"
            else:
                return "HOLD"
        
        elif strategy == "ma_crossover":
            if ma_short is None or ma_long is None:
                return "HOLD"
            
            # Golden cross: short MA crosses above long MA = BUY
            # Death cross: short MA crosses below long MA = SELL
            if ma_short > ma_long and current_price > ma_short:
                return "BUY"
            elif ma_short < ma_long and current_price < ma_short:
                return "SELL"
            else:
                return "HOLD"
        
        elif strategy == "combined":
            # Combine forecast and MA signals
            forecast_signal = self.decide_action(current_price, forecast_price, threshold, "forecast")
            ma_signal = self.decide_action(current_price, forecast_price, threshold, "ma_crossover", ma_short, ma_long)
            
            # Both signals agree
            if forecast_signal == ma_signal:
                return forecast_signal
            # Conflicting signals = HOLD
            else:
                return "HOLD"
        
        else:
            return "HOLD"

    def execute_strategy(
        self,
        current_price: float,
        forecast_price: float,
        position_size: float = 0.1,
    ) -> Optional[str]:
        """
        Execute trading strategy based on forecast.
        position_size: fraction of portfolio to trade (0.0 to 1.0)
        """
        action = self.decide_action(current_price, forecast_price)

        total_value = self.get_total_value(current_price)
        trade_value = total_value * position_size

        if action == "BUY":
            quantity = trade_value / current_price
            if self.buy(quantity, current_price, f"Forecast: {forecast_price:.2f}"):
                return "BUY"
        elif action == "SELL" and self.holdings > 0:
            quantity = min(self.holdings, trade_value / current_price)
            if self.sell(quantity, current_price, f"Forecast: {forecast_price:.2f}"):
                return "SELL"

        return "HOLD"

    def snapshot(self, current_price: float) -> Dict[str, Any]:
        """Create a portfolio snapshot with metrics. Saves to MongoDB."""
        total_value = self.get_total_value(current_price)

        # Calculate returns and volatility from trade history
        returns = []
        if len(self.trades) > 1:
            prev_value = settings.portfolio_initial_cash
            for trade in self.trades:
                if trade.get("action") == "BUY":
                    prev_value -= trade.get("quantity", 0) * trade.get("price", 0)
                else:
                    prev_value += trade.get("quantity", 0) * trade.get("price", 0)
                returns.append((prev_value - settings.portfolio_initial_cash) / settings.portfolio_initial_cash)

        daily_return = returns[-1] if returns else 0.0
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
        sharpe_ratio = (daily_return / volatility) if volatility > 0 else 0.0

        # Save snapshot to MongoDB
        self.store.save_portfolio_snapshot(
            symbol=self.symbol,
            cash=self.cash,
            holdings=self.holdings,
            total_value=total_value,
            daily_return=daily_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
        )

        snapshot = {
            "symbol": self.symbol,
            "timestamp": datetime.now(timezone.utc),
            "cash": self.cash,
            "holdings": self.holdings,
            "total_value": total_value,
            "daily_return": daily_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
        }

        logger.info(
            f"Portfolio snapshot: value={total_value:.2f}, "
            f"return={daily_return:.2%}, sharpe={sharpe_ratio:.2f}"
        )

        return snapshot

    @staticmethod
    def get_portfolio_history(symbol: str, limit: int = 100) -> list[dict]:
        """Get portfolio snapshot history for visualization from MongoDB."""
        store = UnifiedDataStore()
        snapshots = store.get_portfolio_history(symbol, limit)
        
        # Convert timestamps to ISO strings
        for snapshot in snapshots:
            timestamp = snapshot.get("timestamp")
            if timestamp and hasattr(timestamp, "isoformat"):
                snapshot["timestamp"] = timestamp.isoformat()
            elif timestamp:
                snapshot["timestamp"] = str(timestamp)
        
        return snapshots

