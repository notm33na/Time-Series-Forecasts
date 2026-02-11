"""
Trading Strategy Module

Implements rule-based trading strategies for buy/sell/hold decisions.
Strategies are based on forecasted prices vs current prices, with configurable thresholds.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TradingAction(str, Enum):
    """Trading action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingStrategy:
    """
    Rule-based trading strategy implementation.
    
    Strategies:
    - Simple forecast-based: Buy when predicted > current * (1 + threshold)
    - Moving average crossover: Golden/death cross signals
    - Combined: Requires both forecast and MA signals to agree
    """
    
    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = 0.02,
        min_position_size: float = 0.01,
        max_position_size: float = 0.25
    ):
        """
        Initialize trading strategy.
        
        Args:
            buy_threshold: Minimum price increase % to trigger BUY (default: 2%)
            sell_threshold: Minimum price decrease % to trigger SELL (default: 2%)
            min_position_size: Minimum position size as fraction of portfolio (default: 1%)
            max_position_size: Maximum position size as fraction of portfolio (default: 25%)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
    
    def get_action(
        self,
        current_price: float,
        predicted_price: float,
        strategy_type: str = "forecast",
        ma_short: Optional[float] = None,
        ma_long: Optional[float] = None
    ) -> TradingAction:
        """
        Determine trading action based on strategy.
        
        Args:
            current_price: Current market price
            predicted_price: Forecasted future price
            strategy_type: Strategy to use ("forecast", "ma_crossover", "combined")
            ma_short: Short-term moving average (for MA strategies)
            ma_long: Long-term moving average (for MA strategies)
            
        Returns:
            TradingAction (BUY, SELL, or HOLD)
        """
        if strategy_type == "forecast":
            return self._forecast_strategy(current_price, predicted_price)
        elif strategy_type == "ma_crossover":
            return self._ma_crossover_strategy(current_price, ma_short, ma_long)
        elif strategy_type == "combined":
            return self._combined_strategy(
                current_price, predicted_price, ma_short, ma_long
            )
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}. Using forecast strategy.")
            return self._forecast_strategy(current_price, predicted_price)
    
    def _forecast_strategy(
        self,
        current_price: float,
        predicted_price: float
    ) -> TradingAction:
        """
        Simple forecast-based strategy.
        
        Rules:
        - BUY if predicted_price > current_price * (1 + buy_threshold)
        - SELL if predicted_price < current_price * (1 - sell_threshold)
        - HOLD otherwise
        """
        if current_price <= 0:
            logger.warning(f"Invalid current_price: {current_price}")
            return TradingAction.HOLD
        
        if predicted_price <= 0:
            logger.warning(f"Invalid predicted_price: {predicted_price}")
            return TradingAction.HOLD
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        if expected_return > self.buy_threshold:
            return TradingAction.BUY
        elif expected_return < -self.sell_threshold:
            return TradingAction.SELL
        else:
            return TradingAction.HOLD
    
    def _ma_crossover_strategy(
        self,
        current_price: float,
        ma_short: Optional[float],
        ma_long: Optional[float]
    ) -> TradingAction:
        """
        Moving average crossover strategy.
        
        Rules:
        - BUY: Golden cross (short MA crosses above long MA) AND price > short MA
        - SELL: Death cross (short MA crosses below long MA) AND price < short MA
        - HOLD otherwise
        """
        if ma_short is None or ma_long is None:
            logger.warning("MA crossover strategy requires ma_short and ma_long")
            return TradingAction.HOLD
        
        if current_price <= 0 or ma_short <= 0 or ma_long <= 0:
            logger.warning("Invalid prices for MA strategy")
            return TradingAction.HOLD
        
        # Golden cross: short MA > long MA (bullish signal)
        if ma_short > ma_long and current_price > ma_short:
            return TradingAction.BUY
        
        # Death cross: short MA < long MA (bearish signal)
        elif ma_short < ma_long and current_price < ma_short:
            return TradingAction.SELL
        
        else:
            return TradingAction.HOLD
    
    def _combined_strategy(
        self,
        current_price: float,
        predicted_price: float,
        ma_short: Optional[float],
        ma_long: Optional[float]
    ) -> TradingAction:
        """
        Combined strategy requiring both forecast and MA signals to agree.
        
        Rules:
        - BUY: Both forecast and MA signals indicate BUY
        - SELL: Both forecast and MA signals indicate SELL
        - HOLD: Signals conflict or neither indicates action
        """
        forecast_action = self._forecast_strategy(current_price, predicted_price)
        ma_action = self._ma_crossover_strategy(current_price, ma_short, ma_long)
        
        # Both signals must agree
        if forecast_action == ma_action and forecast_action != TradingAction.HOLD:
            return forecast_action
        else:
            return TradingAction.HOLD
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        action: TradingAction,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on portfolio value and confidence.
        
        Args:
            portfolio_value: Total portfolio value
            current_price: Current asset price
            action: Trading action (BUY or SELL)
            confidence: Confidence level (0.0 to 1.0) for position sizing
            
        Returns:
            Quantity to trade (0 if HOLD or invalid)
        """
        if action == TradingAction.HOLD:
            return 0.0
        
        if portfolio_value <= 0 or current_price <= 0:
            return 0.0
        
        # Base position size (scaled by confidence)
        base_size = self.max_position_size * confidence
        
        # Clamp to min/max bounds
        position_size = max(self.min_position_size, min(base_size, self.max_position_size))
        
        # Calculate quantity
        trade_value = portfolio_value * position_size
        quantity = trade_value / current_price
        
        return quantity
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy configuration information."""
        return {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "min_position_size": self.min_position_size,
            "max_position_size": self.max_position_size,
            "buy_threshold_pct": self.buy_threshold * 100,
            "sell_threshold_pct": self.sell_threshold * 100
        }

