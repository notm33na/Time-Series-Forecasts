"""
Unit tests for Performance Metrics module.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from backend.evaluation.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics(risk_free_rate=0.02)
        
        assert metrics.risk_free_rate == 0.02
        assert len(metrics.portfolio_values) == 0
        assert len(metrics.timestamps) == 0
        assert len(metrics.returns) == 0
    
    def test_add_portfolio_snapshot(self):
        """Test adding portfolio snapshots."""
        metrics = PerformanceMetrics()
        
        metrics.add_portfolio_snapshot(10000.0)
        metrics.add_portfolio_snapshot(10500.0)
        
        assert len(metrics.portfolio_values) == 2
        assert len(metrics.returns) == 2
        assert metrics.returns[0] == 0.0  # First snapshot has no return
        assert metrics.returns[1] == 0.05  # 5% return
    
    def test_calculate_daily_returns(self):
        """Test daily returns calculation."""
        metrics = PerformanceMetrics()
        
        metrics.add_portfolio_snapshot(10000.0)
        metrics.add_portfolio_snapshot(10500.0)
        metrics.add_portfolio_snapshot(10200.0)
        
        returns = metrics.calculate_daily_returns()
        
        assert len(returns) == 2
        assert abs(returns[0] - 0.05) < 0.001  # 5% return
        assert abs(returns[1] - (-0.02857)) < 0.001  # ~-2.86% return
    
    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation."""
        metrics = PerformanceMetrics()
        
        metrics.add_portfolio_snapshot(10000.0)
        metrics.add_portfolio_snapshot(10500.0)
        metrics.add_portfolio_snapshot(11000.0)
        
        cumulative = metrics.calculate_cumulative_returns()
        
        assert len(cumulative) == 3
        assert cumulative[0] == 0.0
        assert abs(cumulative[1] - 0.05) < 0.001
        assert abs(cumulative[2] - 0.10) < 0.001
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        metrics = PerformanceMetrics()
        
        # Add some snapshots with varying returns
        values = [10000.0, 10500.0, 10200.0, 10800.0, 10400.0]
        for v in values:
            metrics.add_portfolio_snapshot(v)
        
        volatility = metrics.calculate_volatility(annualized=False)
        
        assert volatility > 0
        assert volatility < 1.0  # Should be reasonable
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics(risk_free_rate=0.02)
        
        # Add positive returns
        values = [10000.0, 10200.0, 10400.0, 10600.0, 10800.0]
        for v in values:
            metrics.add_portfolio_snapshot(v)
        
        sharpe = metrics.calculate_sharpe_ratio(annualized=False)
        
        # With positive returns and low volatility, Sharpe should be positive
        assert sharpe > 0
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        metrics = PerformanceMetrics()
        
        # Create a scenario with a drawdown
        values = [10000.0, 11000.0, 12000.0, 10500.0, 9500.0, 10000.0]
        for v in values:
            metrics.add_portfolio_snapshot(v)
        
        drawdown = metrics.calculate_max_drawdown()
        
        assert "max_drawdown" in drawdown
        assert "max_drawdown_pct" in drawdown
        assert drawdown["max_drawdown"] < 0  # Should be negative
        assert abs(drawdown["max_drawdown"]) < 0.3  # Should be reasonable
    
    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        metrics = PerformanceMetrics()
        
        trades = [
            {
                "action": "BUY",
                "price": 150.0,
                "quantity": 10.0,
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "action": "SELL",
                "price": 155.0,
                "quantity": 10.0,
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "action": "BUY",
                "price": 160.0,
                "quantity": 10.0,
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "action": "SELL",
                "price": 158.0,
                "quantity": 10.0,
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        
        trade_metrics = metrics.calculate_trade_metrics(trades)
        
        assert trade_metrics["num_trades"] == 2  # Two round trips
        assert trade_metrics["num_buys"] == 2
        assert trade_metrics["num_sells"] == 2
        assert trade_metrics["num_wins"] >= 0
        assert "win_rate" in trade_metrics
        assert "profit_factor" in trade_metrics
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        metrics = PerformanceMetrics()
        
        values = [10000.0, 10500.0, 10200.0, 10800.0]
        for v in values:
            metrics.add_portfolio_snapshot(v)
        
        all_metrics = metrics.get_all_metrics()
        
        assert "initial_value" in all_metrics
        assert "current_value" in all_metrics
        assert "total_return" in all_metrics
        assert "volatility" in all_metrics
        assert "sharpe_ratio" in all_metrics
        assert "max_drawdown" in all_metrics
    
    def test_get_all_metrics_insufficient_data(self):
        """Test metrics with insufficient data."""
        metrics = PerformanceMetrics()
        
        metrics.add_portfolio_snapshot(10000.0)
        
        all_metrics = metrics.get_all_metrics()
        
        # Should handle gracefully
        assert "error" in all_metrics or "num_snapshots" in all_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

