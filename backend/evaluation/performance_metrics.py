"""
Performance Metrics Module

Calculates comprehensive portfolio performance metrics including:
- Daily and cumulative returns
- Volatility (standard deviation of returns)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Win rate and profit factor
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculate and track portfolio performance metrics.
    
    Metrics calculated:
    - Returns: Daily, cumulative, annualized
    - Volatility: Standard deviation of returns
    - Sharpe Ratio: Risk-adjusted return (annualized)
    - Maximum Drawdown: Largest peak-to-trough decline
    - Win Rate: Percentage of profitable trades
    - Profit Factor: Gross profit / Gross loss
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2% for Sharpe ratio)
        """
        self.risk_free_rate = risk_free_rate
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        self.returns: List[float] = []
    
    def add_portfolio_snapshot(
        self,
        portfolio_value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a portfolio value snapshot for metrics calculation.
        
        Args:
            portfolio_value: Total portfolio value at this time
            timestamp: Timestamp of snapshot (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.portfolio_values.append(portfolio_value)
        self.timestamps.append(timestamp)
        
        # Calculate return if we have previous value
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                self.returns.append(daily_return)
            else:
                self.returns.append(0.0)
        else:
            # First snapshot - no return yet
            self.returns.append(0.0)
    
    def calculate_daily_returns(self) -> np.ndarray:
        """
        Calculate daily returns from portfolio values.
        
        Returns:
            Array of daily returns
        """
        if len(self.portfolio_values) < 2:
            return np.array([])
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        return returns
    
    def calculate_cumulative_returns(self) -> np.ndarray:
        """
        Calculate cumulative returns from initial value.
        
        Returns:
            Array of cumulative returns
        """
        if len(self.portfolio_values) < 2:
            return np.array([0.0])
        
        initial_value = self.portfolio_values[0]
        if initial_value == 0:
            return np.array([0.0] * len(self.portfolio_values))
        
        values = np.array(self.portfolio_values)
        cumulative_returns = (values - initial_value) / initial_value
        return cumulative_returns
    
    def calculate_volatility(self, annualized: bool = True) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            annualized: Whether to annualize volatility (default: True)
            
        Returns:
            Volatility (as decimal, e.g., 0.15 for 15%)
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        volatility = float(np.std(returns_array))
        
        if annualized:
            # Annualize: multiply by sqrt(252) for daily returns
            volatility *= np.sqrt(252)
        
        return volatility
    
    def calculate_sharpe_ratio(self, annualized: bool = True) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Formula: (Return - RiskFreeRate) / Volatility
        
        Args:
            annualized: Whether to use annualized metrics (default: True)
            
        Returns:
            Sharpe ratio (dimensionless)
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        mean_return = float(np.mean(returns_array))
        
        if annualized:
            # Annualize mean return
            mean_return *= 252
            # Annualize risk-free rate (assume daily returns)
            daily_rf = self.risk_free_rate / 252
        else:
            daily_rf = self.risk_free_rate / 252
        
        volatility = self.calculate_volatility(annualized=annualized)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (mean_return - daily_rf) / volatility if not annualized else (mean_return - self.risk_free_rate) / volatility
        return sharpe
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Returns:
            Dictionary with 'max_drawdown' (as decimal) and 'max_drawdown_pct' (as percentage)
        """
        if len(self.portfolio_values) < 2:
            return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}
        
        values = np.array(self.portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max
        
        max_drawdown = float(np.min(drawdown))
        max_drawdown_pct = max_drawdown * 100
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct
        }
    
    def calculate_trade_metrics(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate trade-based metrics (win rate, profit factor, etc.).
        
        Args:
            trades: List of trade dictionaries with 'action', 'price', 'quantity', 'timestamp'
            
        Returns:
            Dictionary with trade metrics
        """
        if not trades:
            return {
                "num_trades": 0,
                "num_buys": 0,
                "num_sells": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }
        
        # Separate buys and sells
        buys = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        
        # Calculate P&L for round trips (simplified: assumes FIFO)
        profits = []
        losses = []
        
        # Simple approach: track buy/sell pairs
        buy_queue = []
        for trade in sorted(trades, key=lambda x: x.get("timestamp", "")):
            if trade.get("action") == "BUY":
                buy_queue.append(trade)
            elif trade.get("action") == "SELL" and buy_queue:
                buy_trade = buy_queue.pop(0)
                buy_price = buy_trade.get("price", 0)
                sell_price = trade.get("price", 0)
                quantity = min(buy_trade.get("quantity", 0), trade.get("quantity", 0))
                
                pnl = (sell_price - buy_price) * quantity
                
                if pnl > 0:
                    profits.append(pnl)
                else:
                    losses.append(abs(pnl))
        
        num_trades = len(profits) + len(losses)
        num_wins = len(profits)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0
        
        gross_profit = sum(profits) if profits else 0.0
        gross_loss = sum(losses) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        avg_win = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return {
            "num_trades": num_trades,
            "num_buys": len(buys),
            "num_sells": len(sells),
            "num_wins": num_wins,
            "num_losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }
    
    def get_all_metrics(
        self,
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            trades: Optional list of trades for trade-based metrics
            
        Returns:
            Dictionary with all calculated metrics
        """
        if len(self.portfolio_values) < 2:
            return {
                "error": "Insufficient data for metrics calculation",
                "num_snapshots": len(self.portfolio_values)
            }
        
        daily_returns = self.calculate_daily_returns()
        cumulative_returns = self.calculate_cumulative_returns()
        volatility = self.calculate_volatility()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        
        # Calculate total return
        initial_value = self.portfolio_values[0]
        current_value = self.portfolio_values[-1]
        total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0.0
        
        # Annualized return (if we have enough data)
        if len(daily_returns) > 0:
            mean_daily_return = float(np.mean(daily_returns))
            annualized_return = mean_daily_return * 252
        else:
            annualized_return = 0.0
        
        metrics = {
            "initial_value": initial_value,
            "current_value": current_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown["max_drawdown"],
            "max_drawdown_pct": max_drawdown["max_drawdown_pct"],
            "num_snapshots": len(self.portfolio_values),
            "num_returns": len(daily_returns)
        }
        
        # Add trade metrics if provided
        if trades:
            trade_metrics = self.calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
        
        return metrics
    
    def log_metrics(self, trades: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Log all metrics to console/file.
        
        Args:
            trades: Optional list of trades for trade-based metrics
        """
        metrics = self.get_all_metrics(trades)
        
        logger.info("=" * 60)
        logger.info("PORTFOLIO PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"Initial Value: ${metrics.get('initial_value', 0):,.2f}")
        logger.info(f"Current Value: ${metrics.get('current_value', 0):,.2f}")
        logger.info(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
        logger.info(f"Volatility: {metrics.get('volatility_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        
        if trades:
            logger.info(f"Number of Trades: {metrics.get('num_trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
            logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        logger.info("=" * 60)

