"""
Portfolio Manager Module

Manages portfolio state including cash balance, positions, and trade history.
Integrates with MongoDB for persistent storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from ..db.unified_db import UnifiedDataStore
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a position in a single asset."""
    symbol: str
    quantity: float
    avg_price: float


@dataclass
class Trade:
    """Represents a single trade execution."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    price: float
    quantity: float
    timestamp: datetime
    note: Optional[str] = None
    execution_price: Optional[float] = None  # Actual execution price after slippage
    fees: Optional[float] = None  # Transaction fees paid
    slippage: Optional[float] = None  # Slippage amount


class PortfolioManager:
    """
    Manages portfolio state and trade execution.
    
    Features:
    - Tracks cash balance and positions across multiple symbols
    - Executes buy/sell trades with validation
    - Maintains trade history
    - Updates portfolio value based on current prices
    - Integrates with MongoDB for persistence
    - Supports transaction fees and slippage for realistic trading
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        portfolio_id: Optional[str] = None,
        load_from_db: bool = True,
        transaction_fee_rate: float = 0.001,  # 0.1% transaction fee
        slippage_rate: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_cash: Starting cash balance in USD
            portfolio_id: Unique identifier for this portfolio (for multi-portfolio support)
            load_from_db: Whether to load existing portfolio state from database
            transaction_fee_rate: Transaction fee as fraction of trade value (default: 0.1%)
            slippage_rate: Slippage as fraction of price (default: 0.05%)
        """
        self.portfolio_id = portfolio_id or "default"
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trade_history: List[Trade] = []
        self.store = UnifiedDataStore()
        self.transaction_fee_rate = transaction_fee_rate
        self.slippage_rate = slippage_rate
        
        if load_from_db:
            self._load_from_database()
    
    def _load_from_database(self) -> None:
        """Load portfolio state from MongoDB if it exists."""
        try:
            portfolio_data = self.store.get_portfolio_state(self.portfolio_id)
            if portfolio_data:
                self.cash_balance = portfolio_data.get("cash_balance", self.initial_cash)
                
                # Load positions (filter out positions with quantity 0)
                positions_data = portfolio_data.get("positions", {})
                for symbol, pos_data in positions_data.items():
                    quantity = pos_data.get("quantity", 0.0)
                    # Only load positions with quantity > 0
                    if quantity > 0:
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=quantity,
                            avg_price=pos_data.get("avg_price", 0.0)
                        )
                
                logger.info(
                    f"Loaded portfolio {self.portfolio_id} from database: "
                    f"cash=${self.cash_balance:.2f}, positions={list(self.positions.keys())}"
                )
                
                # If no positions loaded but we have trades, try to rebuild from trade history
                if not self.positions:
                    logger.info("No positions found in portfolio state, attempting to rebuild from trade history...")
                    self._rebuild_positions_from_trades()
        except Exception as e:
            logger.warning(f"Could not load portfolio from database: {e}. Starting fresh.")
    
    def _rebuild_positions_from_trades(self) -> None:
        """Rebuild positions and cash balance from trade history."""
        try:
            # Reset to initial state
            self.cash_balance = self.initial_cash
            self.positions = {}
            
            # Get all trades from database (we'll process all symbols)
            # Query MongoDB directly to get all trades
            trades_cursor = self.store.mongo_db.trades.find().sort("timestamp", 1)  # Oldest first
            
            for trade_doc in trades_cursor:
                symbol = trade_doc.get("symbol")
                action = trade_doc.get("action")
                quantity = trade_doc.get("quantity", 0.0)
                price = trade_doc.get("price", 0.0)
                
                if not symbol or action not in ["BUY", "SELL"] or quantity <= 0 or price <= 0:
                    continue
                
                # Apply slippage and fees (use same rates)
                if action == "BUY":
                    slippage = price * self.slippage_rate
                    execution_price = price + slippage
                    trade_value = execution_price * quantity
                    transaction_fee = trade_value * self.transaction_fee_rate
                    total_cost = trade_value + transaction_fee
                    
                    # Deduct from cash
                    self.cash_balance -= total_cost
                    
                    # Update position
                    if symbol in self.positions:
                        existing = self.positions[symbol]
                        total_quantity = existing.quantity + quantity
                        total_cost_basis = (existing.quantity * existing.avg_price) + trade_value
                        new_avg_price = total_cost_basis / total_quantity if total_quantity > 0 else execution_price
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=total_quantity,
                            avg_price=new_avg_price
                        )
                    else:
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=quantity,
                            avg_price=execution_price
                        )
                elif action == "SELL":
                    slippage = price * self.slippage_rate
                    execution_price = price - slippage
                    trade_value = execution_price * quantity
                    transaction_fee = trade_value * self.transaction_fee_rate
                    net_proceeds = trade_value - transaction_fee
                    
                    # Add to cash
                    self.cash_balance += net_proceeds
                    
                    # Update position
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        remaining_quantity = position.quantity - quantity
                        if remaining_quantity > 0:
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=remaining_quantity,
                                avg_price=position.avg_price
                            )
                        else:
                            del self.positions[symbol]
            
            # Filter out positions with quantity 0
            self.positions = {k: v for k, v in self.positions.items() if v.quantity > 0}
            
            # Save the rebuilt state
            if self.positions:
                self._save_to_database()
                logger.info(
                    f"Rebuilt positions from trade history: "
                    f"cash=${self.cash_balance:.2f}, positions={list(self.positions.keys())}"
                )
            else:
                logger.warning("No positions found after rebuilding from trade history")
            
        except Exception as e:
            logger.error(f"Error rebuilding positions from trades: {e}", exc_info=True)
    
    def rebuild_position_for_symbol(self, symbol: str) -> bool:
        """Rebuild position for a specific symbol from its trade history."""
        try:
            # Get all trades for this symbol
            trades = self.store.get_trades(symbol, limit=1000)  # Get many trades
            
            if not trades:
                return False
            
            # Sort by timestamp (oldest first)
            def get_timestamp(trade):
                ts = trade.get("timestamp")
                if isinstance(ts, datetime):
                    return ts
                elif isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        return datetime.min.replace(tzinfo=timezone.utc)
                return datetime.min.replace(tzinfo=timezone.utc)
            
            trades.sort(key=get_timestamp)
            
            # Reset position for this symbol
            if symbol in self.positions:
                del self.positions[symbol]
            
            # Track position quantity and cost basis
            position_quantity = 0.0
            total_cost_basis = 0.0
            
            # Process trades
            # Note: The price saved in trades is already the execution_price (with slippage applied)
            for trade in trades:
                action = trade.get("action")
                quantity = trade.get("quantity", 0.0)
                execution_price = trade.get("price", 0.0)  # This is already execution_price
                
                if action == "BUY" and quantity > 0 and execution_price > 0:
                    # execution_price already includes slippage
                    # We need to calculate what the cost basis was (including fees)
                    trade_value = execution_price * quantity
                    # Note: We can't perfectly reconstruct fees, but we can approximate
                    # The avg_price should reflect the execution price, not including fees
                    # Fees affect cash but not the cost basis for position tracking
                    total_cost_basis += trade_value
                    position_quantity += quantity
                elif action == "SELL" and quantity > 0 and execution_price > 0:
                    if position_quantity >= quantity:
                        # Calculate cost basis for sold shares (FIFO approximation)
                        avg_cost = total_cost_basis / position_quantity if position_quantity > 0 else 0
                        sold_cost_basis = avg_cost * quantity
                        total_cost_basis -= sold_cost_basis
                        position_quantity -= quantity
            
            # Create position if we have quantity > 0
            if position_quantity > 0:
                avg_price = total_cost_basis / position_quantity
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=position_quantity,
                    avg_price=avg_price
                )
                logger.info(
                    f"Rebuilt position for {symbol}: quantity={position_quantity:.4f}, "
                    f"avg_price=${avg_price:.2f}"
                )
                # Save the updated state
                self._save_to_database()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error rebuilding position for {symbol}: {e}", exc_info=True)
            return False
    
    def _save_to_database(self) -> None:
        """Save current portfolio state to MongoDB."""
        try:
            # Only save positions with quantity > 0
            positions_to_save = {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price
                }
                for symbol, pos in self.positions.items()
                if pos.quantity > 0
            }
            
            # Calculate total value (use avg_price for positions if no current prices available)
            holdings_value = sum(
                pos.quantity * pos.avg_price 
                for pos in self.positions.values() 
                if pos.quantity > 0
            )
            total_value = self.cash_balance + holdings_value
            
            portfolio_data = {
                "portfolio_id": self.portfolio_id,
                "cash_balance": self.cash_balance,
                "positions": positions_to_save,
                "total_value": total_value,
                "last_updated": datetime.now(timezone.utc)
            }
            self.store.save_portfolio_state(portfolio_data)
            logger.debug(
                f"Saved portfolio {self.portfolio_id} to database: "
                f"cash=${self.cash_balance:.2f}, positions={list(positions_to_save.keys())}"
            )
        except Exception as e:
            logger.error(f"Error saving portfolio to database: {e}")
    
    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        note: Optional[str] = None
    ) -> bool:
        """
        Execute a trade (BUY or SELL).
        
        Args:
            symbol: Asset symbol (e.g., "AAPL", "BTC")
            action: "BUY" or "SELL"
            price: Execution price per unit
            quantity: Number of units to trade
            note: Optional note/strategy identifier
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        if action not in ["BUY", "SELL"]:
            logger.error(f"Invalid action: {action}. Must be BUY or SELL")
            return False
        
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}. Must be positive")
            return False
        
        if price <= 0:
            logger.error(f"Invalid price: {price}. Must be positive")
            return False
        
        if action == "BUY":
            return self._execute_buy(symbol, price, quantity, note)
        else:  # SELL
            return self._execute_sell(symbol, price, quantity, note)
    
    def _execute_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        note: Optional[str] = None
    ) -> bool:
        """Execute a buy order with fees and slippage."""
        # Apply slippage (buy orders typically execute at slightly higher price)
        slippage = price * self.slippage_rate
        execution_price = price + slippage
        
        # Calculate costs
        trade_value = execution_price * quantity
        transaction_fee = trade_value * self.transaction_fee_rate
        total_cost = trade_value + transaction_fee
        
        if total_cost > self.cash_balance:
            logger.warning(
                f"Insufficient cash for BUY: need ${total_cost:.2f} "
                f"(price: ${execution_price:.2f}, fee: ${transaction_fee:.2f}), "
                f"have ${self.cash_balance:.2f}"
            )
            return False
        
        # Update cash (deduct total cost including fees)
        self.cash_balance -= total_cost
        
        # Update position (use execution price for cost basis)
        if symbol in self.positions:
            # Calculate new average price (weighted average)
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            total_cost_basis = (existing.quantity * existing.avg_price) + trade_value
            new_avg_price = total_cost_basis / total_quantity if total_quantity > 0 else execution_price
            
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_price=new_avg_price
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=execution_price
            )
        
        # Record trade with execution details
        trade = Trade(
            symbol=symbol,
            action="BUY",
            price=price,  # Intended price
            quantity=quantity,
            timestamp=datetime.now(timezone.utc),
            note=note,
            execution_price=execution_price,
            fees=transaction_fee,
            slippage=slippage
        )
        self.trade_history.append(trade)
        
        # Save trade to database
        self.store.save_trade(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=execution_price,  # Save execution price
            note=f"{note or ''} (fee: ${transaction_fee:.2f}, slippage: ${slippage:.2f})"
        )
        
        # Save portfolio state
        self._save_to_database()
        
        logger.info(
            f"BUY executed: {quantity:.4f} {symbol} @ ${execution_price:.2f} "
            f"(intended: ${price:.2f}, fee: ${transaction_fee:.2f}, slippage: ${slippage:.2f}) "
            f"(Cash: ${self.cash_balance:.2f})"
        )
        return True
    
    def _execute_sell(
        self,
        symbol: str,
        price: float,
        quantity: float,
        note: Optional[str] = None
    ) -> bool:
        """Execute a sell order with fees and slippage."""
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to sell")
            return False
        
        position = self.positions[symbol]
        
        if quantity > position.quantity:
            logger.warning(
                f"Insufficient holdings for SELL: need {quantity:.4f}, have {position.quantity:.4f}"
            )
            return False
        
        # Apply slippage (sell orders typically execute at slightly lower price)
        slippage = price * self.slippage_rate
        execution_price = price - slippage
        
        # Calculate proceeds
        trade_value = execution_price * quantity
        transaction_fee = trade_value * self.transaction_fee_rate
        net_proceeds = trade_value - transaction_fee
        
        # Update cash (add net proceeds after fees)
        self.cash_balance += net_proceeds
        
        # Update position
        remaining_quantity = position.quantity - quantity
        if remaining_quantity > 0:
            # Partial sell - keep position with same avg_price
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                avg_price=position.avg_price
            )
        else:
            # Full sell - remove position
            del self.positions[symbol]
        
        # Record trade with execution details
        trade = Trade(
            symbol=symbol,
            action="SELL",
            price=price,  # Intended price
            quantity=quantity,
            timestamp=datetime.now(timezone.utc),
            note=note,
            execution_price=execution_price,
            fees=transaction_fee,
            slippage=slippage
        )
        self.trade_history.append(trade)
        
        # Save trade to database
        self.store.save_trade(
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=execution_price,  # Save execution price
            note=f"{note or ''} (fee: ${transaction_fee:.2f}, slippage: ${slippage:.2f})"
        )
        
        # Save portfolio state
        self._save_to_database()
        
        logger.info(
            f"SELL executed: {quantity:.4f} {symbol} @ ${execution_price:.2f} "
            f"(intended: ${price:.2f}, fee: ${transaction_fee:.2f}, slippage: ${slippage:.2f}) "
            f"(Cash: ${self.cash_balance:.2f})"
        )
        return True
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate and return total portfolio value based on current prices.
        
        Args:
            current_prices: Dictionary mapping symbol -> current price
            
        Returns:
            Total portfolio value (cash + holdings value)
        """
        holdings_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                holdings_value += position.quantity * current_prices[symbol]
            else:
                logger.warning(f"No current price available for {symbol}, using avg_price")
                holdings_value += position.quantity * position.avg_price
        
        total_value = self.cash_balance + holdings_value
        return total_value
    
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive portfolio metrics.
        
        Args:
            current_prices: Dictionary mapping symbol -> current price
            
        Returns:
            Dictionary with portfolio metrics
        """
        total_value = self.update_portfolio_value(current_prices)
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                cost_basis = position.quantity * position.avg_price
                current_value = position.quantity * current_prices[symbol]
                unrealized_pnl += current_value - cost_basis
        
        # Calculate total return
        total_return = total_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0
        
        # Holdings breakdown
        holdings = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                holdings[symbol] = {
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": current_price,
                    "value": position.quantity * current_price,
                    "unrealized_pnl": (current_price - position.avg_price) * position.quantity,
                    "unrealized_pnl_pct": ((current_price - position.avg_price) / position.avg_price * 100) if position.avg_price > 0 else 0.0
                }
        
        # Calculate total fees and slippage
        total_fees = self.get_total_fees_paid()
        total_slippage_cost = self.get_total_slippage()
        
        return {
            "portfolio_id": self.portfolio_id,
            "cash_balance": self.cash_balance,
            "total_value": total_value,
            "initial_cash": self.initial_cash,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "unrealized_pnl": unrealized_pnl,
            "holdings": holdings,
            "num_positions": len(self.positions),
            "num_trades": len(self.trade_history),
            "total_fees_paid": total_fees,
            "total_slippage_cost": total_slippage_cost,
            "net_return_after_costs": total_return - total_fees - total_slippage_cost,
            "net_return_pct_after_costs": ((total_return - total_fees - total_slippage_cost) / self.initial_cash * 100) if self.initial_cash > 0 else 0.0
        }
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        trades = self.trade_history
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Convert to dictionaries and sort by timestamp (newest first)
        trade_dicts = [asdict(trade) for trade in trades]
        trade_dicts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Convert datetime to ISO string for JSON serialization
        for trade in trade_dicts:
            if isinstance(trade["timestamp"], datetime):
                trade["timestamp"] = trade["timestamp"].isoformat()
        
        return trade_dicts[:limit]
    
    def get_total_fees_paid(self) -> float:
        """Calculate total transaction fees paid across all trades."""
        total_fees = 0.0
        for trade in self.trade_history:
            if trade.fees is not None:
                total_fees += trade.fees
        return total_fees
    
    def get_total_slippage(self) -> float:
        """Calculate total slippage cost across all trades."""
        total_slippage = 0.0
        for trade in self.trade_history:
            if trade.slippage is not None:
                # Slippage cost = slippage * quantity
                slippage_cost = trade.slippage * trade.quantity
                total_slippage += slippage_cost
        return total_slippage

