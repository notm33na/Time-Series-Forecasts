"""
Tests for portfolio service.
"""

import pytest

from ..services.portfolio_service import PortfolioService


def test_portfolio_initialization():
    """Test portfolio initialization."""
    portfolio = PortfolioService("TEST", initial_cash=10000.0)
    assert portfolio.symbol == "TEST"
    assert portfolio.cash == 10000.0
    assert portfolio.holdings == 0.0


def test_portfolio_buy():
    """Test buy operation."""
    portfolio = PortfolioService("TEST", initial_cash=10000.0)
    success = portfolio.buy(10.0, 100.0)

    assert success
    assert portfolio.cash == 9000.0
    assert portfolio.holdings == 10.0


def test_portfolio_sell():
    """Test sell operation."""
    portfolio = PortfolioService("TEST", initial_cash=10000.0)
    portfolio.buy(10.0, 100.0)
    success = portfolio.sell(5.0, 110.0)

    assert success
    assert portfolio.cash == 9550.0
    assert portfolio.holdings == 5.0


def test_portfolio_insufficient_cash():
    """Test buy with insufficient cash."""
    portfolio = PortfolioService("TEST", initial_cash=100.0)
    success = portfolio.buy(10.0, 100.0)

    assert not success
    assert portfolio.cash == 100.0
    assert portfolio.holdings == 0.0


def test_portfolio_decide_action():
    """Test trading decision logic."""
    portfolio = PortfolioService("TEST")

    # Bullish forecast
    action = portfolio.decide_action(100.0, 105.0, threshold=0.02)
    assert action == "BUY"

    # Bearish forecast
    action = portfolio.decide_action(100.0, 95.0, threshold=0.02)
    assert action == "SELL"

    # Neutral
    action = portfolio.decide_action(100.0, 101.0, threshold=0.02)
    assert action == "HOLD"


def test_portfolio_total_value():
    """Test total value calculation."""
    portfolio = PortfolioService("TEST", initial_cash=10000.0)
    portfolio.buy(10.0, 100.0)

    total = portfolio.get_total_value(110.0)
    assert total == 9000.0 + (10.0 * 110.0)

