"""
Portfolio management API endpoints.
Comprehensive portfolio management with NLP-A3 features.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime, timezone

from ..config import get_settings
from ..services.portfolio_service import PortfolioService
from ..services.data_ingestion import DataIngestionService
from ..trading.portfolio_manager import PortfolioManager
from ..evaluation.performance_metrics import PerformanceMetrics
from ..db.unified_db import UnifiedDataStore
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])
settings = get_settings()


class TradeRequest(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: Optional[float] = None
    price: Optional[float] = None
    portfolio_id: Optional[str] = "default"
    reason: Optional[str] = "user_manual"


class StrategyRequest(BaseModel):
    symbol: str
    forecast_price: Optional[float] = None
    position_size: float = 0.1
    strategy: str = "momentum"  # momentum, conservative, aggressive
    portfolio_id: Optional[str] = "default"
    forecast_id: Optional[str] = None


@router.get("/summary")
async def get_portfolio_summary(portfolio_id: str = Query("default", description="Portfolio identifier")):
    """Get comprehensive portfolio summary with all metrics."""
    try:
        from ..services.data_ingestion import DataIngestionService
        
        portfolio = PortfolioManager(portfolio_id=portfolio_id)
        
        # Get current prices for all positions
        current_prices = {}
        for symbol in portfolio.positions.keys():
            price_obj = DataIngestionService.get_latest_price(symbol)
            if price_obj:
                current_prices[symbol] = price_obj.get("close", 0.0)
        
        # Get portfolio metrics
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        # Calculate additional metrics
        total_value = metrics["total_value"]
        total_return_pct = metrics["total_return_pct"]
        
        # Get performance history for volatility and Sharpe calculation
        store = UnifiedDataStore()
        history = store.get_portfolio_history(portfolio_id, limit=30)
        
        # Calculate volatility and Sharpe ratio from history
        returns = []
        if len(history) > 1:
            for i in range(1, len(history)):
                prev_value = history[i-1].get("total_value", portfolio.initial_cash)
                curr_value = history[i].get("total_value", total_value)
                if prev_value > 0:
                    daily_return = ((curr_value - prev_value) / prev_value) * 100
                    returns.append(daily_return)
        
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = 0.0
        if len(returns) > 1 and volatility > 0:
            avg_return = np.mean(returns)
            annual_return = avg_return * 252  # Trading days
            annual_std = volatility * np.sqrt(252)
            risk_free_rate = 0.02 * 100  # 2% annual
            sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0.0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        if len(history) > 1:
            values = [h.get("total_value", portfolio.initial_cash) for h in history]
            values.append(total_value)
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = ((peak - value) / peak) * 100 if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Get positions with current prices
        positions = []
        for symbol, position in portfolio.positions.items():
            current_price = current_prices.get(symbol, position.avg_price)
            current_value = position.quantity * current_price
            pnl = (current_price - position.avg_price) * position.quantity
            pnl_percent = ((current_price - position.avg_price) / position.avg_price * 100) if position.avg_price > 0 else 0.0
            
            positions.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "averagePrice": position.avg_price,
                "currentPrice": current_price,
                "totalCost": position.quantity * position.avg_price,
                "currentValue": current_value,
                "pnl": pnl,
                "pnlPercent": pnl_percent
            })
        
        # Calculate allocation
        allocation = []
        for pos in positions:
            if total_value > 0:
                allocation.append({
                    "name": pos["symbol"],
                    "value": (pos["currentValue"] / total_value) * 100
                })
        
        if portfolio.cash_balance > 0 and total_value > 0:
            allocation.append({
                "name": "Cash",
                "value": (portfolio.cash_balance / total_value) * 100
            })
        
        # Save performance snapshot
        store.save_portfolio_snapshot(
            symbol=portfolio_id,
            cash=portfolio.cash_balance,
            holdings=total_value - portfolio.cash_balance,
            total_value=total_value,
            daily_return=total_return_pct,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio
        )
        
        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "initial_cash": portfolio.initial_cash,
                "current_cash": portfolio.cash_balance,
                "total_value": total_value,
                "total_return": total_return_pct,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "positions": positions,
                "allocation": allocation,
                "positions_count": len(positions),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(portfolio_id: str = Query("default", description="Portfolio identifier")):
    """Get all positions in the portfolio."""
    try:
        from ..services.data_ingestion import DataIngestionService
        
        portfolio = PortfolioManager(portfolio_id=portfolio_id)
        
        # Get current prices
        current_prices = {}
        for symbol in portfolio.positions.keys():
            price_obj = DataIngestionService.get_latest_price(symbol)
            if price_obj:
                current_prices[symbol] = price_obj.get("close", 0.0)
        
        positions = []
        for symbol, position in portfolio.positions.items():
            if position.quantity > 0:
                current_price = current_prices.get(symbol, position.avg_price)
                current_value = position.quantity * current_price
                pnl = (current_price - position.avg_price) * position.quantity
                pnl_percent = ((current_price - position.avg_price) / position.avg_price * 100) if position.avg_price > 0 else 0.0
                
                positions.append({
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "averagePrice": position.avg_price,
                    "currentPrice": current_price,
                    "totalCost": position.quantity * position.avg_price,
                    "currentValue": current_value,
                    "pnl": pnl,
                    "pnlPercent": pnl_percent
                })
        
        return {
            "success": True,
            "data": positions
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance(
    portfolio_id: str = Query("default", description="Portfolio identifier"),
    days: int = Query(30, description="Number of days of history")
):
    """Get performance history for visualization."""
    try:
        store = UnifiedDataStore()
        history = store.get_portfolio_history(portfolio_id, limit=days)
        
        # Format for frontend
        performance_data = []
        for h in history:
            performance_data.append({
                "date": h.get("timestamp").isoformat() if hasattr(h.get("timestamp"), "isoformat") else str(h.get("timestamp")),
                "value": h.get("total_value", 0),
                "returns": h.get("daily_return", 0)
            })
        
        return {
            "success": True,
            "data": performance_data
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_portfolio_history(symbol: str, limit: int = 100):
    """Get portfolio snapshot history."""
    try:
        history = PortfolioService.get_portfolio_history(symbol, limit)
        return {"symbol": symbol, "history": history, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-strategy")
async def execute_strategy(request: StrategyRequest):
    """
    Execute trading strategy based on forecast predictions.
    
    Strategies:
    - momentum: Buy if forecast predicts >2% increase, sell if >2% decrease
    - conservative: Only buy if forecast predicts >5% increase
    - aggressive: Buy/sell with 1% thresholds, uses larger position sizes
    """
    try:
        # Get current price
        current_price_obj = DataIngestionService.get_latest_price(request.symbol)
        if not current_price_obj:
            raise ValueError(f"No price data available for {request.symbol}")

        current_price = current_price_obj.get("close", 0.0)

        # Initialize data store (needed for forecast lookup and snapshot creation)
        store = UnifiedDataStore()

        # Get forecast price if not provided
        forecast_price = request.forecast_price
        if forecast_price is None:
            # Try to get latest forecast
            # Get latest forecast for symbol
            forecast_doc = store.mongo_db.forecasts.find_one(
                {"symbol": request.symbol},
                sort=[("target_time", -1)]
            )
            if forecast_doc and forecast_doc.get("forecast_data"):
                # Use the last predicted price
                forecast_data = forecast_doc.get("forecast_data", [])
                if isinstance(forecast_data, list) and len(forecast_data) > 0:
                    last_forecast = forecast_data[-1]
                    forecast_price = last_forecast.get("Close", current_price)
                else:
                    forecast_price = current_price
            else:
                forecast_price = current_price

        # Initialize portfolio manager
        portfolio = PortfolioManager(portfolio_id=request.portfolio_id)
        
        # Get current position
        current_position = portfolio.positions.get(request.symbol)
        current_quantity = current_position.quantity if current_position else 0.0
        
        # Analyze forecast trend
        price_change = forecast_price - current_price
        price_change_percent = (price_change / current_price) * 100 if current_price > 0 else 0.0
        
        # Strategy logic
        action_taken = "hold"
        result = {
            "success": True,
            "strategy": request.strategy,
            "symbol": request.symbol,
            "forecast_analysis": {
                "current_price": current_price,
                "forecast_price": forecast_price,
                "predicted_change": price_change,
                "predicted_change_percent": price_change_percent,
                "action_taken": action_taken
            }
        }
        
        if request.strategy == "momentum":
            # Buy if forecast predicts >2% increase, sell if >2% decrease
            if price_change_percent > 2.0 and current_quantity == 0:
                # Buy signal
                available_cash = portfolio.cash_balance
                investment_amount = available_cash * 0.1  # 10% of cash
                quantity = investment_amount / current_price
                if quantity > 0:
                    success = portfolio.execute_trade(
                        request.symbol, "BUY", current_price, quantity,
                        f"forecast_momentum_{request.forecast_id or ''}"
                    )
                    if success:
                        action_taken = "buy"
                        result["message"] = f"Bought {quantity:.4f} {request.symbol} at ${current_price:.2f}"
                        result["transaction"] = {"action": "BUY", "quantity": quantity, "price": current_price}
            
            elif price_change_percent < -2.0 and current_quantity > 0:
                # Sell signal - sell 50% of position
                sell_quantity = current_quantity * 0.5
                success = portfolio.execute_trade(
                    request.symbol, "SELL", current_price, sell_quantity,
                    f"forecast_momentum_{request.forecast_id or ''}"
                )
                if success:
                    action_taken = "sell"
                    result["message"] = f"Sold {sell_quantity:.4f} {request.symbol} at ${current_price:.2f}"
                    result["transaction"] = {"action": "SELL", "quantity": sell_quantity, "price": current_price}
            else:
                action_taken = "hold"
                result["message"] = f"Hold - Forecast change: {price_change_percent:.2f}%"
        
        elif request.strategy == "conservative":
            # Only buy if forecast predicts >5% increase
            if price_change_percent > 5.0 and current_quantity == 0:
                available_cash = portfolio.cash_balance
                investment_amount = available_cash * 0.05  # Only 5% of cash
                quantity = investment_amount / current_price
                if quantity > 0:
                    success = portfolio.execute_trade(
                        request.symbol, "BUY", current_price, quantity,
                        f"forecast_conservative_{request.forecast_id or ''}"
                    )
                    if success:
                        action_taken = "buy"
                        result["message"] = f"Bought {quantity:.4f} {request.symbol} at ${current_price:.2f}"
                        result["transaction"] = {"action": "BUY", "quantity": quantity, "price": current_price}
            else:
                action_taken = "hold"
                result["message"] = f"Hold - Forecast change: {price_change_percent:.2f}% (threshold: 5%)"
        
        elif request.strategy == "aggressive":
            # More aggressive trading
            if price_change_percent > 1.0 and current_quantity == 0:
                available_cash = portfolio.cash_balance
                investment_amount = available_cash * 0.2  # 20% of cash
                quantity = investment_amount / current_price
                if quantity > 0:
                    success = portfolio.execute_trade(
                        request.symbol, "BUY", current_price, quantity,
                        f"forecast_aggressive_{request.forecast_id or ''}"
                    )
                    if success:
                        action_taken = "buy"
                        result["message"] = f"Bought {quantity:.4f} {request.symbol} at ${current_price:.2f}"
                        result["transaction"] = {"action": "BUY", "quantity": quantity, "price": current_price}
            
            elif price_change_percent < -1.0 and current_quantity > 0:
                sell_quantity = current_quantity * 0.75  # Sell 75%
                success = portfolio.execute_trade(
                    request.symbol, "SELL", current_price, sell_quantity,
                    f"forecast_aggressive_{request.forecast_id or ''}"
                )
                if success:
                    action_taken = "sell"
                    result["message"] = f"Sold {sell_quantity:.4f} {request.symbol} at ${current_price:.2f}"
                    result["transaction"] = {"action": "SELL", "quantity": sell_quantity, "price": current_price}
            else:
                action_taken = "hold"
                result["message"] = f"Hold - Forecast change: {price_change_percent:.2f}%"
        else:
            raise ValueError(f"Invalid strategy: {request.strategy}. Must be one of: momentum, conservative, aggressive")
        
        result["forecast_analysis"]["action_taken"] = action_taken
        
        # Get updated portfolio metrics
        current_prices = {request.symbol: current_price}
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        result["portfolio_value"] = metrics["total_value"]
        result["cash"] = metrics["cash_balance"]
        result["holdings"] = metrics["total_value"] - metrics["cash_balance"]
        
        # Create portfolio snapshot after trade execution
        if action_taken != "hold":
            try:
                # Get portfolio history to calculate daily return
                # Note: get_portfolio_history uses symbol, but we need to ensure we're getting the right data
                history = store.get_portfolio_history(request.symbol, limit=10)
                daily_return = 0.0
                if len(history) > 0:
                    prev_value = history[-1].get("total_value", portfolio.initial_cash)
                    if prev_value > 0:
                        daily_return = ((metrics["total_value"] - prev_value) / prev_value) * 100
                
                # Calculate volatility from recent history
                returns = []
                if len(history) > 1:
                    for i in range(1, len(history)):
                        prev_val = history[i-1].get("total_value", portfolio.initial_cash)
                        curr_val = history[i].get("total_value", metrics["total_value"])
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
                
                # Save portfolio snapshot
                store.save_portfolio_snapshot(
                    symbol=request.symbol,
                    cash=metrics["cash_balance"],
                    holdings=metrics["total_value"] - metrics["cash_balance"],
                    total_value=metrics["total_value"],
                    daily_return=daily_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio
                )
                logger.info(f"Saved portfolio snapshot for {request.symbol} after {action_taken}")
            except Exception as e:
                logger.warning(f"Could not save portfolio snapshot: {e}")
        
        # Log the result for debugging
        logger.info(f"Strategy execution result: action={action_taken}, message={result.get('message')}, portfolio_value={result['portfolio_value']}")
        
        return result
    except Exception as e:
        logger.error(f"Error executing strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy")
async def buy_asset(request: TradeRequest):
    """Execute a buy order."""
    try:
        if not request.symbol:
            raise ValueError("Symbol is required")
        
        if request.quantity and request.quantity <= 0:
            raise ValueError("Quantity must be greater than 0")
        
        # Get current price if not provided
        if not request.price:
            current_price_obj = DataIngestionService.get_latest_price(request.symbol)
            if not current_price_obj:
                raise ValueError(f"No price data available for {request.symbol}")
            price = current_price_obj.get("close", 0.0)
        else:
            price = request.price

        # Initialize portfolio manager
        portfolio = PortfolioManager(portfolio_id=request.portfolio_id)
        
        # Debug: Log current state before buy
        logger.info(
            f"Buy request for {request.symbol}: "
            f"portfolio_id={request.portfolio_id}, "
            f"existing_positions={list(portfolio.positions.keys())}, "
            f"cash_balance=${portfolio.cash_balance:.2f}"
        )
        
        # Calculate quantity if not provided (use 10% of available cash)
        if not request.quantity:
            trade_value = portfolio.cash_balance * 0.1
            quantity = trade_value / price
        else:
            quantity = request.quantity

        # Execute buy
        success = portfolio.execute_trade(
            request.symbol, "BUY", price, quantity, request.reason or "user_manual"
        )

        if not success:
            raise ValueError("Buy failed: insufficient cash")
        
        # Verify position was created and saved
        position_after = portfolio.positions.get(request.symbol)
        if not position_after or position_after.quantity <= 0:
            logger.error(
                f"Buy executed but position not found for {request.symbol}. "
                f"This indicates a bug in position tracking."
            )
            raise ValueError("Buy executed but position was not created correctly")
        
        logger.info(
            f"Buy successful for {request.symbol}: "
            f"quantity={position_after.quantity:.4f}, "
            f"avg_price=${position_after.avg_price:.2f}, "
            f"cash_remaining=${portfolio.cash_balance:.2f}"
        )

        # Get updated metrics
        current_prices = {request.symbol: price}
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        # Create portfolio snapshot after trade
        try:
            store = UnifiedDataStore()
            history = store.get_portfolio_history(request.symbol, limit=2)
            daily_return = 0.0
            if len(history) > 0:
                prev_value = history[-1].get("total_value", portfolio.initial_cash)
                if prev_value > 0:
                    daily_return = ((metrics["total_value"] - prev_value) / prev_value) * 100
            
            # Calculate volatility
            returns = []
            if len(history) > 1:
                for i in range(1, len(history)):
                    prev_val = history[i-1].get("total_value", portfolio.initial_cash)
                    curr_val = history[i].get("total_value", metrics["total_value"])
                    if prev_val > 0:
                        ret = ((curr_val - prev_val) / prev_val) * 100
                        returns.append(ret)
            
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            sharpe_ratio = 0.0
            if len(returns) > 1 and volatility > 0:
                avg_return = np.mean(returns)
                annual_return = avg_return * 252
                annual_std = volatility * np.sqrt(252)
                risk_free_rate = 0.02 * 100
                sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0.0
            
            store.save_portfolio_snapshot(
                symbol=request.symbol,
                cash=metrics["cash_balance"],
                holdings=metrics["total_value"] - metrics["cash_balance"],
                total_value=metrics["total_value"],
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio
            )
        except Exception as e:
            logger.warning(f"Could not save portfolio snapshot after buy: {e}")
        
        # Get position info
        position = portfolio.positions.get(request.symbol)
        
        return {
            "success": True,
            "message": f"Bought {quantity:.4f} {request.symbol} at ${price:.2f}",
            "transaction": {
                "symbol": request.symbol,
                "action": "BUY",
                "quantity": quantity,
                "price": price
            },
            "remaining_cash": portfolio.cash_balance,
            "position": {
                "symbol": request.symbol,
                "quantity": position.quantity,
                "average_price": position.avg_price,
                "total_cost": position.quantity * position.avg_price
            } if position else {
                "symbol": request.symbol,
                "quantity": quantity,
                "average_price": price,
                "total_cost": quantity * price
            },
            "portfolio_value": metrics["total_value"]
        }
    except Exception as e:
        logger.error(f"Error executing buy: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sell")
async def sell_asset(request: TradeRequest):
    """Execute a sell order."""
    try:
        if not request.symbol:
            raise ValueError("Symbol is required")
        
        if request.quantity and request.quantity <= 0:
            raise ValueError("Quantity must be greater than 0")
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(portfolio_id=request.portfolio_id)
        
        # Debug: Log current positions
        logger.info(
            f"Sell request for {request.symbol}: "
            f"portfolio_id={request.portfolio_id}, "
            f"available_positions={list(portfolio.positions.keys())}, "
            f"cash_balance=${portfolio.cash_balance:.2f}"
        )
        
        # Check if position exists
        position = portfolio.positions.get(request.symbol)
        if not position or position.quantity <= 0:
            # Try to rebuild position from trade history
            logger.info(f"Position not found for {request.symbol}, attempting to rebuild from trade history...")
            rebuilt = portfolio.rebuild_position_for_symbol(request.symbol)
            
            if rebuilt:
                position = portfolio.positions.get(request.symbol)
                logger.info(f"Successfully rebuilt position for {request.symbol} from trade history")
            else:
                available_symbols = list(portfolio.positions.keys())
                error_msg = (
                    f"No position found for {request.symbol}. "
                    f"You need to buy shares before selling. "
                    f"Available positions: {available_symbols if available_symbols else 'none'}"
                )
                logger.warning(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        
        # Get current price if not provided
        if not request.price:
            current_price_obj = DataIngestionService.get_latest_price(request.symbol)
            if not current_price_obj:
                price = position.avg_price  # Fallback to average price
            else:
                price = current_price_obj.get("close", position.avg_price)
        else:
            price = request.price
        
        # Use all position if quantity not specified
        if not request.quantity:
            quantity = position.quantity
        else:
            quantity = request.quantity
        
        if quantity > position.quantity:
            raise ValueError(f"Insufficient shares. Requested: {quantity:.4f}, Owned: {position.quantity:.4f}")

        # Execute sell
        success = portfolio.execute_trade(
            request.symbol, "SELL", price, quantity, request.reason or "user_manual"
        )

        if not success:
            raise ValueError("Sell failed")

        # Get updated metrics
        current_prices = {request.symbol: price}
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        # Create portfolio snapshot after trade
        try:
            store = UnifiedDataStore()
            history = store.get_portfolio_history(request.symbol, limit=2)
            daily_return = 0.0
            if len(history) > 0:
                prev_value = history[-1].get("total_value", portfolio.initial_cash)
                if prev_value > 0:
                    daily_return = ((metrics["total_value"] - prev_value) / prev_value) * 100
            
            # Calculate volatility
            returns = []
            if len(history) > 1:
                for i in range(1, len(history)):
                    prev_val = history[i-1].get("total_value", portfolio.initial_cash)
                    curr_val = history[i].get("total_value", metrics["total_value"])
                    if prev_val > 0:
                        ret = ((curr_val - prev_val) / prev_val) * 100
                        returns.append(ret)
            
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            sharpe_ratio = 0.0
            if len(returns) > 1 and volatility > 0:
                avg_return = np.mean(returns)
                annual_return = avg_return * 252
                annual_std = volatility * np.sqrt(252)
                risk_free_rate = 0.02 * 100
                sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0.0
            
            store.save_portfolio_snapshot(
                symbol=request.symbol,
                cash=metrics["cash_balance"],
                holdings=metrics["total_value"] - metrics["cash_balance"],
                total_value=metrics["total_value"],
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio
            )
        except Exception as e:
            logger.warning(f"Could not save portfolio snapshot after sell: {e}")
        
        # Get updated position (if still exists)
        updated_position = portfolio.positions.get(request.symbol)
        
        return {
            "success": True,
            "message": f"Sold {quantity:.4f} {request.symbol} at ${price:.2f}",
            "transaction": {
                "symbol": request.symbol,
                "action": "SELL",
                "quantity": quantity,
                "price": price
            },
            "remaining_cash": portfolio.cash_balance,
            "position": {
                "symbol": request.symbol,
                "quantity": updated_position.quantity,
                "average_price": updated_position.avg_price,
                "total_cost": updated_position.quantity * updated_position.avg_price
            } if updated_position else None,
            "portfolio_value": metrics["total_value"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing sell: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trade")
async def execute_trade(request: TradeRequest):
    """Execute a manual trade (BUY or SELL)."""
    try:
        if request.action == "BUY":
            return await buy_asset(request)
        elif request.action == "SELL":
            return await sell_asset(request)
        elif request.action == "HOLD":
            return {
                "success": True,
                "message": f"Holding position for {request.symbol}",
                "action": "hold",
                "reason": request.reason or "user_manual"
            }
        else:
            raise ValueError(f"Invalid action: {request.action}. Must be BUY, SELL, or HOLD")
    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes (400, 404, etc.)
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{symbol}")
async def get_trades(symbol: str, limit: int = 50):
    """Get trade history for a symbol."""
    try:
        from ..db.unified_db import UnifiedDataStore
        
        store = UnifiedDataStore()
        trades = store.get_trades(symbol, limit)
        
        return {
            "symbol": symbol,
            "trades": [
                {
                    "id": t["id"],
                    "timestamp": t["timestamp"].isoformat() if hasattr(t["timestamp"], "isoformat") else str(t["timestamp"]),
                    "action": t["action"],
                    "quantity": t["quantity"],
                    "price": t["price"],
                    "note": t.get("note"),
                }
                for t in reversed(trades)  # Reverse to show chronological order
            ],
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/{symbol}")
async def get_dashboard(symbol: str, limit: int = 100):
    """Get complete portfolio dashboard data with visualizations."""
    try:
        from ..trading.integration_example import get_portfolio_dashboard_data
        from ..trading.portfolio_manager import PortfolioManager
        
        dashboard_data = get_portfolio_dashboard_data(symbol, limit=limit)
        
        # If no portfolio history exists, create initial snapshot
        if not dashboard_data.get("portfolio_history") or len(dashboard_data["portfolio_history"]) == 0:
            portfolio = PortfolioManager(portfolio_id="default", load_from_db=True)
            current_price = dashboard_data.get("current_price", 0.0)
            
            if current_price > 0:
                current_prices = {symbol: current_price}
                portfolio_value = portfolio.update_portfolio_value(current_prices)
                
                # Create initial snapshot
                store = UnifiedDataStore()
                store.save_portfolio_snapshot(
                    symbol=symbol,
                    cash=portfolio.cash_balance,
                    holdings=portfolio_value - portfolio.cash_balance,
                    total_value=portfolio_value,
                    daily_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0
                )
                
                # Update dashboard data with initial state
                dashboard_data["portfolio_history"] = [{
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cash": portfolio.cash_balance,
                    "holdings": portfolio_value - portfolio.cash_balance,
                    "total_value": portfolio_value,
                    "daily_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0
                }]
                
                # Update portfolio metrics
                portfolio_metrics = portfolio.get_portfolio_metrics(current_prices)
                dashboard_data["portfolio_metrics"] = portfolio_metrics
        
        return {
            "symbol": symbol,
            **dashboard_data,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error getting portfolio dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}")
async def get_portfolio_metrics(symbol: str, portfolio_id: str = Query("default")):
    """Get current portfolio metrics for a symbol."""
    try:
        # Get current price
        current_price_obj = DataIngestionService.get_latest_price(symbol)
        if not current_price_obj:
            raise ValueError(f"No price data available for {symbol}")
        
        current_price = current_price_obj.get("close", 0.0)
        
        # Load portfolio
        portfolio = PortfolioManager(portfolio_id=portfolio_id)
        
        # Get metrics
        current_prices = {symbol: current_price}
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        return {
            "symbol": symbol,
            "metrics": metrics,
            "current_price": current_price,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error getting portfolio metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))