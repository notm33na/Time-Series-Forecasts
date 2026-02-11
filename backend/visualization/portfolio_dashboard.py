"""
Portfolio Dashboard Module

Generates visualizations for portfolio performance including:
- Portfolio growth over time
- Candlestick charts with buy/sell markers
- Performance metrics summary
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioDashboard:
    """
    Generate portfolio visualization charts and dashboards.
    
    Uses Plotly for interactive charts that can be embedded in web frontend.
    """
    
    @staticmethod
    def create_portfolio_growth_chart(
        portfolio_history: List[Dict[str, Any]],
        title: str = "Portfolio Growth Over Time"
    ) -> go.Figure:
        """
        Create a line chart showing portfolio value over time.
        
        Args:
            portfolio_history: List of portfolio snapshots with 'timestamp' and 'total_value'
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if not portfolio_history:
            logger.warning("Empty portfolio history provided")
            return go.Figure()
        
        # Extract data
        timestamps = [pd.to_datetime(s["timestamp"]) for s in portfolio_history]
        values = [s.get("total_value", 0) for s in portfolio_history]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_candlestick_with_trades(
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        forecasts: Optional[List[Dict[str, Any]]] = None,
        title: str = "Price Chart with Trade Markers"
    ) -> go.Figure:
        """
        Create candlestick chart with buy/sell markers and forecast overlay.
        
        Args:
            price_data: DataFrame with OHLC data (columns: timestamp, open, high, low, close)
            trades: List of trades with 'timestamp', 'action', 'price'
            forecasts: Optional list of forecasts with 'timestamp', 'forecast'
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if price_data.empty:
            logger.warning("Empty price data provided")
            return go.Figure()
        
        # Ensure date/timestamp column exists - handle both 'date' and 'timestamp'
        if 'date' in price_data.columns:
            price_data['timestamp'] = pd.to_datetime(price_data['date'])
        elif 'timestamp' in price_data.columns:
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        elif price_data.index.name in ['date', 'timestamp']:
            price_data = price_data.reset_index()
            price_data['timestamp'] = pd.to_datetime(price_data[price_data.index.name])
        else:
            # Try to use index if it's datetime
            if isinstance(price_data.index, pd.DatetimeIndex):
                price_data = price_data.reset_index()
                price_data['timestamp'] = price_data.index if 'index' not in price_data.columns else price_data.iloc[:, 0]
            else:
                raise ValueError(f"Could not find date/timestamp column. Available columns: {list(price_data.columns)}")
        
        # Normalize column names (handle both uppercase and lowercase)
        column_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        for old_col, new_col in column_map.items():
            if old_col in price_data.columns and new_col not in price_data.columns:
                price_data[new_col] = price_data[old_col]
        
        # Create subplot for price and volume (if available)
        has_volume = 'volume' in price_data.columns
        if has_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Volume")
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=price_data['timestamp'],
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price'
        )
        
        if has_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add buy markers (green triangles)
        buy_trades = [t for t in trades if t.get("action") == "BUY"]
        if buy_trades:
            buy_timestamps = [pd.to_datetime(t["timestamp"]) for t in buy_trades]
            buy_prices = [t.get("price", 0) for t in buy_trades]
            
            buy_marker = go.Scatter(
                x=buy_timestamps,
                y=buy_prices,
                mode='markers',
                name='BUY',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#10b981',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
            )
            
            if has_volume:
                fig.add_trace(buy_marker, row=1, col=1)
            else:
                fig.add_trace(buy_marker)
        
        # Add sell markers (red triangles)
        sell_trades = [t for t in trades if t.get("action") == "SELL"]
        if sell_trades:
            sell_timestamps = [pd.to_datetime(t["timestamp"]) for t in sell_trades]
            sell_prices = [t.get("price", 0) for t in sell_trades]
            
            sell_marker = go.Scatter(
                x=sell_timestamps,
                y=sell_prices,
                mode='markers',
                name='SELL',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#ef4444',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
            )
            
            if has_volume:
                fig.add_trace(sell_marker, row=1, col=1)
            else:
                fig.add_trace(sell_marker)
        
        # Add forecast overlay (if provided)
        if forecasts:
            forecast_timestamps = [pd.to_datetime(f["timestamp"]) for f in forecasts]
            forecast_values = [f.get("forecast", 0) for f in forecasts]
            
            forecast_line = go.Scatter(
                x=forecast_timestamps,
                y=forecast_values,
                mode='lines',
                name='Forecast',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                hovertemplate='<b>Forecast</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
            )
            
            if has_volume:
                fig.add_trace(forecast_line, row=1, col=1)
            else:
                fig.add_trace(forecast_line)
        
        # Add volume chart if available
        if has_volume:
            volume_bars = go.Bar(
                x=price_data['timestamp'],
                y=price_data['volume'],
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.5)'
            )
            fig.add_trace(volume_bars, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title if not has_volume else None,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=600 if has_volume else 400,
            showlegend=True
        )
        
        if has_volume:
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_metrics_summary(
        metrics: Dict[str, Any],
        title: str = "Performance Metrics Summary"
    ) -> go.Figure:
        """
        Create a summary visualization of key performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Extract key metrics
        metrics_to_show = {
            "Total Return (%)": metrics.get("total_return_pct", 0),
            "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
            "Volatility (%)": metrics.get("volatility_pct", 0),
            "Max Drawdown (%)": metrics.get("max_drawdown_pct", 0)
        }
        
        # Create bar chart
        fig = go.Figure()
        
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        
        fig.add_trace(go.Bar(
            x=list(metrics_to_show.keys()),
            y=list(metrics_to_show.values()),
            marker_color=colors,
            text=[f"{v:.2f}" for v in metrics_to_show.values()],
            textposition='outside',
            name='Metrics'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Value",
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_combined_dashboard(
        portfolio_history: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        forecasts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, go.Figure]:
        """
        Create a complete dashboard with all visualizations.
        
        Args:
            portfolio_history: Portfolio value history
            price_data: OHLC price data
            trades: Trade history
            metrics: Performance metrics
            forecasts: Optional forecast data
            
        Returns:
            Dictionary with chart names and Figure objects
        """
        charts = {
            "portfolio_growth": PortfolioDashboard.create_portfolio_growth_chart(
                portfolio_history,
                "Portfolio Growth Over Time"
            ),
            "price_chart": PortfolioDashboard.create_candlestick_with_trades(
                price_data,
                trades,
                forecasts,
                "Price Chart with Trade Markers"
            ),
            "metrics_summary": PortfolioDashboard.create_metrics_summary(
                metrics,
                "Performance Metrics Summary"
            )
        }
        
        return charts
    
    @staticmethod
    def charts_to_json(charts: Dict[str, go.Figure]) -> Dict[str, str]:
        """
        Convert Plotly figures to JSON for frontend consumption.
        
        Args:
            charts: Dictionary of chart names and Figure objects
            
        Returns:
            Dictionary of chart names and JSON strings
        """
        return {
            name: fig.to_json()
            for name, fig in charts.items()
        }

