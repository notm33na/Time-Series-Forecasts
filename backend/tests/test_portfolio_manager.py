"""
Unit tests for Portfolio Manager module.
"""

import pytest
from datetime import datetime, timezone
from backend.trading.portfolio_manager import PortfolioManager, Position, Trade
from backend.trading.trading_strategy import TradingStrategy, TradingAction


class TestPortfolioManager:
    """Test suite for PortfolioManager."""
    
    def test_initialization(self):
        """Test portfolio initialization."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        assert portfolio.cash_balance == 10000.0
        assert portfolio.initial_cash == 10000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.trade_history) == 0
    
    def test_buy_trade(self):
        """Test executing a buy trade."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        success = portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        assert success == True
        assert portfolio.cash_balance == 8500.0  # 10000 - (150 * 10)
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 10.0
        assert portfolio.positions["AAPL"].avg_price == 150.0
        assert len(portfolio.trade_history) == 1
    
    def test_buy_insufficient_cash(self):
        """Test buy trade with insufficient cash."""
        portfolio = PortfolioManager(initial_cash=1000.0, load_from_db=False)
        
        success = portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        assert success == False
        assert portfolio.cash_balance == 1000.0  # Unchanged
        assert len(portfolio.positions) == 0
    
    def test_sell_trade(self):
        """Test executing a sell trade."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        # First buy
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        # Then sell
        success = portfolio.execute_trade("AAPL", "SELL", 155.0, 5.0)
        
        assert success == True
        assert portfolio.cash_balance == 9275.0  # 8500 + (155 * 5)
        assert portfolio.positions["AAPL"].quantity == 5.0
        assert len(portfolio.trade_history) == 2
    
    def test_sell_insufficient_holdings(self):
        """Test sell trade with insufficient holdings."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        success = portfolio.execute_trade("AAPL", "SELL", 150.0, 10.0)
        
        assert success == False
        assert portfolio.cash_balance == 10000.0  # Unchanged
    
    def test_sell_all_holdings(self):
        """Test selling all holdings removes position."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        # Buy
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        # Sell all
        portfolio.execute_trade("AAPL", "SELL", 155.0, 10.0)
        
        assert "AAPL" not in portfolio.positions
        assert portfolio.cash_balance == 10050.0  # 10000 - 1500 + 1550
    
    def test_average_price_calculation(self):
        """Test average price calculation for multiple buys."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        # First buy
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        # Second buy at different price
        portfolio.execute_trade("AAPL", "BUY", 160.0, 10.0)
        
        # Average should be weighted: (150*10 + 160*10) / 20 = 155.0
        assert portfolio.positions["AAPL"].quantity == 20.0
        assert portfolio.positions["AAPL"].avg_price == 155.0
    
    def test_update_portfolio_value(self):
        """Test portfolio value calculation."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        current_prices = {"AAPL": 155.0}
        total_value = portfolio.update_portfolio_value(current_prices)
        
        # Cash: 8500, Holdings: 10 * 155 = 1550, Total: 10050
        assert total_value == 10050.0
    
    def test_get_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        
        current_prices = {"AAPL": 155.0}
        metrics = portfolio.get_portfolio_metrics(current_prices)
        
        assert metrics["cash_balance"] == 8500.0
        assert metrics["total_value"] == 10050.0
        assert metrics["total_return"] == 50.0
        assert metrics["total_return_pct"] == 0.5
        assert "AAPL" in metrics["holdings"]
        assert metrics["holdings"]["AAPL"]["unrealized_pnl"] == 50.0
    
    def test_get_trade_history(self):
        """Test retrieving trade history."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        portfolio.execute_trade("AAPL", "SELL", 155.0, 5.0)
        
        history = portfolio.get_trade_history()
        
        assert len(history) == 2
        assert history[0]["action"] == "SELL"  # Most recent first
        assert history[1]["action"] == "BUY"
    
    def test_get_trade_history_filtered(self):
        """Test trade history filtered by symbol."""
        portfolio = PortfolioManager(initial_cash=10000.0, load_from_db=False)
        
        portfolio.execute_trade("AAPL", "BUY", 150.0, 10.0)
        portfolio.execute_trade("GOOGL", "BUY", 200.0, 5.0)
        
        aapl_history = portfolio.get_trade_history(symbol="AAPL")
        
        assert len(aapl_history) == 1
        assert aapl_history[0]["symbol"] == "AAPL"


class TestTradingStrategy:
    """Test suite for TradingStrategy."""
    
    def test_forecast_strategy_buy(self):
        """Test forecast strategy triggers BUY."""
        strategy = TradingStrategy(buy_threshold=0.02, sell_threshold=0.02)
        
        action = strategy.get_action(100.0, 105.0, strategy_type="forecast")
        
        assert action == TradingAction.BUY  # 5% increase > 2% threshold
    
    def test_forecast_strategy_sell(self):
        """Test forecast strategy triggers SELL."""
        strategy = TradingStrategy(buy_threshold=0.02, sell_threshold=0.02)
        
        action = strategy.get_action(100.0, 95.0, strategy_type="forecast")
        
        assert action == TradingAction.SELL  # 5% decrease > 2% threshold
    
    def test_forecast_strategy_hold(self):
        """Test forecast strategy triggers HOLD."""
        strategy = TradingStrategy(buy_threshold=0.02, sell_threshold=0.02)
        
        action = strategy.get_action(100.0, 101.0, strategy_type="forecast")
        
        assert action == TradingAction.HOLD  # 1% increase < 2% threshold
    
    def test_ma_crossover_strategy_golden_cross(self):
        """Test MA crossover strategy with golden cross."""
        strategy = TradingStrategy()
        
        action = strategy.get_action(
            current_price=152.0,
            predicted_price=150.0,
            strategy_type="ma_crossover",
            ma_short=151.0,
            ma_long=148.0
        )
        
        assert action == TradingAction.BUY  # Golden cross + price > short MA
    
    def test_ma_crossover_strategy_death_cross(self):
        """Test MA crossover strategy with death cross."""
        strategy = TradingStrategy()
        
        action = strategy.get_action(
            current_price=147.0,
            predicted_price=150.0,
            strategy_type="ma_crossover",
            ma_short=148.0,
            ma_long=151.0
        )
        
        assert action == TradingAction.SELL  # Death cross + price < short MA
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        strategy = TradingStrategy(max_position_size=0.25)
        
        quantity = strategy.calculate_position_size(
            portfolio_value=10000.0,
            current_price=150.0,
            action=TradingAction.BUY,
            confidence=1.0
        )
        
        # 25% of 10000 = 2500, quantity = 2500 / 150 = 16.67
        assert quantity > 16.0
        assert quantity < 17.0
    
    def test_calculate_position_size_hold(self):
        """Test position size for HOLD action."""
        strategy = TradingStrategy()
        
        quantity = strategy.calculate_position_size(
            portfolio_value=10000.0,
            current_price=150.0,
            action=TradingAction.HOLD,
            confidence=1.0
        )
        
        assert quantity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

