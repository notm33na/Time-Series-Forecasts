"""
Integration Example: Portfolio Management with Forecasting Loop

This module demonstrates how to integrate the portfolio management system
with the existing forecasting pipeline.

Usage:
    from backend.trading.integration_example import integrate_portfolio_with_forecast
    
    # In your forecast endpoint or service
    result = integrate_portfolio_with_forecast(
        symbol="AAPL",
        current_price=150.0,
        forecast_price=155.0,
        horizon="24h"
    )
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np

from .portfolio_manager import PortfolioManager
from .trading_strategy import TradingStrategy, TradingAction
from ..evaluation.performance_metrics import PerformanceMetrics
from ..visualization.portfolio_dashboard import PortfolioDashboard
from ..utils.logging import get_logger

logger = get_logger(__name__)


def integrate_portfolio_with_forecast(
    symbol: str,
    current_price: float,
    forecast_price: float,
    horizon: str = "24h",
    strategy_type: str = "forecast",
    ma_short: Optional[float] = None,
    ma_long: Optional[float] = None,
    portfolio_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Integrate portfolio management with forecast generation.
    
    This function:
    1. Gets trading action from strategy
    2. Executes trade if needed
    3. Updates portfolio value
    4. Calculates performance metrics
    5. Saves portfolio snapshot with model version tracking
    6. Returns comprehensive result
    
    Args:
        symbol: Asset symbol
        current_price: Current market price
        forecast_price: Forecasted price
        horizon: Forecast horizon (e.g., "24h")
        strategy_type: Trading strategy ("forecast", "ma_crossover", "combined")
        ma_short: Short-term MA (for MA strategies)
        ma_long: Long-term MA (for MA strategies)
        portfolio_id: Portfolio identifier
        model_version_id: Model version ID used for this forecast (for continuous learning tracking)
        model_type: Model type used (e.g., "Ensemble", "LSTM", "Transformer")
        
    Returns:
        Dictionary with trade result, portfolio metrics, and action taken
    """
    # Initialize components
    portfolio = PortfolioManager(portfolio_id=portfolio_id or symbol)
    strategy = TradingStrategy()
    metrics = PerformanceMetrics()
    
    # Get current portfolio value
    current_prices = {symbol: current_price}
    portfolio_value = portfolio.update_portfolio_value(current_prices)
    
    # Get trading action
    action = strategy.get_action(
        current_price=current_price,
        predicted_price=forecast_price,
        strategy_type=strategy_type,
        ma_short=ma_short,
        ma_long=ma_long
    )
    
    # Execute trade if not HOLD
    trade_executed = False
    if action != TradingAction.HOLD:
        # Calculate position size
        quantity = strategy.calculate_position_size(
            portfolio_value=portfolio_value,
            current_price=current_price,
            action=action,
            confidence=1.0
        )
        
        if quantity > 0:
            # Build note with model information
            model_info = ""
            if model_version_id:
                model_info = f" [Model: {model_type or 'Unknown'} v{model_version_id}]"
            
            note = f"Forecast-based: {forecast_price:.2f} vs {current_price:.2f} ({horizon}){model_info}"
            trade_executed = portfolio.execute_trade(
                symbol=symbol,
                action=action.value,
                price=current_price,
                quantity=quantity,
                note=note
            )
    
    # Update portfolio value after trade
    updated_portfolio_value = portfolio.update_portfolio_value(current_prices)
    
    # Add snapshot to metrics
    metrics.add_portfolio_snapshot(updated_portfolio_value)
    
    # Get portfolio metrics
    portfolio_metrics = portfolio.get_portfolio_metrics(current_prices)
    
    # Get performance metrics
    performance_metrics = metrics.get_all_metrics(
        trades=portfolio.get_trade_history(symbol=symbol)
    )
    
    # Save portfolio snapshot with model version tracking
    try:
        from ..db.unified_db import UnifiedDataStore
        store = UnifiedDataStore()
        
        # Calculate daily return
        portfolio_history = store.get_portfolio_history(symbol, limit=2)
        daily_return = 0.0
        if len(portfolio_history) > 0:
            prev_value = portfolio_history[-1].get("total_value", portfolio.initial_cash)
            if prev_value > 0:
                daily_return = ((updated_portfolio_value - prev_value) / prev_value) * 100
        
        # Calculate volatility from recent history
        returns = []
        if len(portfolio_history) > 1:
            for i in range(1, len(portfolio_history)):
                prev_val = portfolio_history[i-1].get("total_value", portfolio.initial_cash)
                curr_val = portfolio_history[i].get("total_value", updated_portfolio_value)
                if prev_val > 0:
                    ret = ((curr_val - prev_val) / prev_val) * 100
                    returns.append(ret)
        
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = 0.0
        if len(returns) > 1 and volatility > 0:
            avg_return = np.mean(returns)
            annual_return = avg_return * 252
            annual_std = volatility * np.sqrt(252)
            risk_free_rate = 0.02 * 100  # 2% annual
            sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0.0
        
        # Save snapshot with model version info
        store.save_portfolio_snapshot(
            symbol=symbol,
            cash=portfolio.cash_balance,
            holdings=updated_portfolio_value - portfolio.cash_balance,
            total_value=updated_portfolio_value,
            daily_return=daily_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio
        )
        
        # Save model version tracking if available
        if model_version_id:
            try:
                store.mongo_db.portfolio_model_tracking.insert_one({
                    "symbol": symbol,
                    "portfolio_id": portfolio_id or symbol,
                    "model_version_id": model_version_id,
                    "model_type": model_type,
                    "timestamp": datetime.now(timezone.utc),
                    "portfolio_value": updated_portfolio_value,
                    "action_taken": action.value,
                    "forecast_price": forecast_price,
                    "current_price": current_price,
                    "trade_executed": trade_executed
                })
            except Exception as e:
                logger.warning(f"Could not save model version tracking: {e}")
    except Exception as e:
        logger.warning(f"Could not save portfolio snapshot: {e}")
    
    result = {
        "symbol": symbol,
        "action": action.value,
        "trade_executed": trade_executed,
        "current_price": current_price,
        "forecast_price": forecast_price,
        "forecast_horizon": horizon,
        "portfolio_value": updated_portfolio_value,
        "portfolio_metrics": portfolio_metrics,
        "performance_metrics": performance_metrics,
        "model_version_id": model_version_id,
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(
        f"Portfolio integration complete: {symbol} - {action.value} - "
        f"Value: ${updated_portfolio_value:.2f} "
        f"(Model: {model_type or 'N/A'} v{model_version_id or 'N/A'})"
    )
    
    return result


def integrate_with_forecast_endpoint(
    symbol: str,
    forecast_result: Dict[str, Any],
    current_price: float
) -> Dict[str, Any]:
    """
    Integration helper for use in forecast API endpoints.
    
    Args:
        symbol: Asset symbol
        forecast_result: Result from forecast API endpoint
        current_price: Current market price
        
    Returns:
        Enhanced forecast result with portfolio integration
    """
    forecast_price = forecast_result.get("forecast", [])
    if isinstance(forecast_price, list) and len(forecast_price) > 0:
        # Use first forecast value
        forecast_price = forecast_price[0]
    elif isinstance(forecast_price, (int, float)):
        forecast_price = float(forecast_price)
    else:
        logger.warning(f"Could not extract forecast price from {forecast_result}")
        return forecast_result
    
    horizon = forecast_result.get("horizon", "24h")
    
    # Integrate portfolio
    portfolio_result = integrate_portfolio_with_forecast(
        symbol=symbol,
        current_price=current_price,
        forecast_price=forecast_price,
        horizon=horizon
    )
    
    # Merge results
    enhanced_result = {
        **forecast_result,
        "portfolio": portfolio_result
    }
    
    return enhanced_result


def get_portfolio_dashboard_data(
    symbol: str,
    portfolio_id: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get all data needed for portfolio dashboard visualization.
    
    Args:
        symbol: Asset symbol
        portfolio_id: Portfolio identifier
        limit: Maximum number of historical records
        
    Returns:
        Dictionary with portfolio history, trades, and metrics
    """
    try:
        from ..db.unified_db import UnifiedDataStore
        from ..services.data_ingestion import DataIngestionService
        
        store = UnifiedDataStore()
        portfolio = PortfolioManager(portfolio_id=portfolio_id or symbol, load_from_db=True)
        
        # Get portfolio history
        try:
            portfolio_history = store.get_portfolio_history(symbol, limit)
            logger.info(f"Retrieved {len(portfolio_history)} portfolio history records for {symbol}")
            # If no history exists, create initial snapshot
            if not portfolio_history:
                portfolio_value = portfolio.update_portfolio_value({}) if current_price == 0 else portfolio.update_portfolio_value({symbol: current_price})
                store.save_portfolio_snapshot(
                    symbol=symbol,
                    cash=portfolio.cash_balance,
                    holdings=portfolio_value - portfolio.cash_balance,
                    total_value=portfolio_value,
                    daily_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0
                )
                portfolio_history = store.get_portfolio_history(symbol, limit)
        except Exception as e:
            logger.warning(f"Could not get portfolio history: {e}")
            portfolio_history = []
        
        # Get trades from database (not just in-memory list)
        try:
            # Query database directly to get all trades for this symbol
            trades_from_db = store.get_trades(symbol, limit)
            logger.info(f"Retrieved {len(trades_from_db)} trades from database for {symbol}")
            # Convert to the format expected by frontend
            trades = []
            for trade in trades_from_db:
                timestamp = trade.get("timestamp")
                # Convert datetime to ISO string if needed
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                elif timestamp and hasattr(timestamp, "isoformat"):
                    timestamp = timestamp.isoformat()
                
                trades.append({
                    "id": trade.get("id", str(trade.get("_id", ""))),
                    "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                    "action": trade.get("action"),
                    "quantity": trade.get("quantity", 0.0),
                    "price": trade.get("price", 0.0),
                    "note": trade.get("note")
                })
            # Sort by timestamp (oldest first for display)
            # get_trades already returns in reverse chronological order, so reverse it
            trades.reverse()
        except Exception as e:
            logger.warning(f"Could not get trades: {e}")
            trades = []
        
        # Get price data
        try:
            service = DataIngestionService()
            price_data = service.get_prices(symbol, limit=limit)
        except Exception as e:
            logger.warning(f"Could not get price data: {e}")
            price_data = None
        
        # Get current price
        try:
            current_price_obj = DataIngestionService.get_latest_price(symbol)
            current_price = current_price_obj.get("close", 0.0) if current_price_obj else 0.0
        except Exception as e:
            logger.warning(f"Could not get current price: {e}")
            current_price = 0.0
        
        # Calculate metrics
        try:
            current_prices = {symbol: current_price} if current_price > 0 else {}
            portfolio_metrics = portfolio.get_portfolio_metrics(current_prices)
            
            # Ensure initial snapshot exists if no history
            if not portfolio_history and current_price > 0:
                # Create initial snapshot
                store.save_portfolio_snapshot(
                    symbol=symbol,
                    cash=portfolio.cash_balance,
                    holdings=portfolio_metrics.get("total_value", portfolio.cash_balance) - portfolio.cash_balance,
                    total_value=portfolio_metrics.get("total_value", portfolio.cash_balance),
                    daily_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0
                )
                # Reload history
                portfolio_history = store.get_portfolio_history(symbol, limit)
        except Exception as e:
            logger.warning(f"Could not calculate portfolio metrics: {e}")
            portfolio_metrics = {
                "portfolio_id": portfolio_id or symbol,
                "cash_balance": portfolio.cash_balance,
                "total_value": portfolio.cash_balance,
                "initial_cash": portfolio.initial_cash,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "unrealized_pnl": 0.0,
                "holdings": {},
                "num_positions": 0,
                "num_trades": len(trades)
            }
        
        # Build metrics calculator
        try:
            metrics = PerformanceMetrics()
            for snapshot in portfolio_history:
                total_value = snapshot.get("total_value", snapshot.get("total_value", 0))
                if total_value:
                    metrics.add_portfolio_snapshot(float(total_value))
            
            performance_metrics = metrics.get_all_metrics(trades=trades)
        except Exception as e:
            logger.warning(f"Could not calculate performance metrics: {e}")
            performance_metrics = {
                "initial_value": portfolio.initial_cash,
                "current_value": portfolio_metrics.get("total_value", portfolio.initial_cash),
                "total_return": 0.0,
                "total_return_pct": 0.0
            }
        
        # Create visualizations
        chart_json = {}
        try:
            if price_data is not None and not price_data.empty:
                charts = PortfolioDashboard.create_combined_dashboard(
                    portfolio_history=portfolio_history,
                    price_data=price_data,
                    trades=trades,
                    metrics=performance_metrics
                )
                chart_json = PortfolioDashboard.charts_to_json(charts)
        except Exception as e:
            logger.warning(f"Could not create portfolio charts: {e}")
            # Continue without charts - non-critical
        
        return {
            "symbol": symbol,
            "portfolio_history": portfolio_history,
            "trades": trades,
            "portfolio_metrics": portfolio_metrics,
            "performance_metrics": performance_metrics,
            "charts": chart_json,
            "current_price": current_price
        }
    except Exception as e:
        logger.error(f"Error in get_portfolio_dashboard_data: {e}", exc_info=True)
        # Return minimal response to prevent complete failure
        return {
            "symbol": symbol,
            "portfolio_history": [],
            "trades": [],
            "portfolio_metrics": {},
            "performance_metrics": {},
            "charts": {},
            "current_price": 0.0,
            "error": str(e)
        }

