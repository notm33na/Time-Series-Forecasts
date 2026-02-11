"""
Enhanced multi-view dashboard with Market, Performance Monitor, Learning Log, and Portfolio Manager.
"""

from datetime import datetime, timedelta
from typing import Optional

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import pandas as pd
import requests

from .config import get_settings

settings = get_settings()
API_BASE = "http://localhost:8000"


def create_dashboard():
    """Create enhanced multi-view dashboard."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://codepen.io/chriddyp/pen/bWLwgP.css",
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        ],
        url_base_pathname="/dashboard/",
    )

    app.layout = html.Div(
        [
            html.H1(
                "FinTech Forecasting Platform - Adaptive Learning System",
                style={"textAlign": "center", "marginBottom": "20px", "color": "#2c3e50"}
            ),
            
            # Control Panel
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Symbol:", style={"marginRight": "5px"}),
                            dcc.Dropdown(
                                id="symbol-select",
                                options=[
                                    {"label": "AAPL", "value": "AAPL"},
                                    {"label": "GOOGL", "value": "GOOGL"},
                                    {"label": "MSFT", "value": "MSFT"},
                                    {"label": "BTC-USD", "value": "BTC-USD"},
                                    {"label": "ETH-USD", "value": "ETH-USD"},
                                ],
                                value=settings.symbol,
                                style={"width": "150px", "display": "inline-block", "marginRight": "20px"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Model Type:", style={"marginRight": "5px"}),
                            dcc.Dropdown(
                                id="model-select",
                                options=[
                                    {"label": "Ensemble", "value": "Ensemble"},
                                    {"label": "ARIMA", "value": "ARIMA"},
                                    {"label": "LSTM", "value": "LSTM"},
                                    {"label": "GRU", "value": "GRU"},
                                    {"label": "Transformer", "value": "Transformer"},
                                ],
                                value="Ensemble",
                                style={"width": "150px", "display": "inline-block", "marginRight": "20px"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Horizon:", style={"marginRight": "5px"}),
                            dcc.Dropdown(
                                id="horizon-select",
                                options=[
                                    {"label": "1h", "value": "1h"},
                                    {"label": "3h", "value": "3h"},
                                    {"label": "24h", "value": "24h"},
                                    {"label": "72h", "value": "72h"},
                                ],
                                value="24h",
                                style={"width": "100px", "display": "inline-block", "marginRight": "20px"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Button("Refresh", id="refresh-button", n_clicks=0, style={"marginRight": "10px"}),
                    html.Button("Train Model", id="train-button", n_clicks=0),
                ],
                style={"padding": "20px", "textAlign": "center", "backgroundColor": "#ecf0f1", "borderRadius": "5px", "marginBottom": "20px"},
            ),
            
            # Tabs for different views
            dcc.Tabs(
                id="dashboard-tabs",
                value="market-tab",
                children=[
                    dcc.Tab(label="Market Dashboard", value="market-tab"),
                    dcc.Tab(label="Performance Monitor", value="performance-tab"),
                    dcc.Tab(label="Adaptive Learning Log", value="learning-tab"),
                    dcc.Tab(label="Portfolio Manager", value="portfolio-tab"),
                ],
                style={"marginBottom": "20px"},
            ),
            
            html.Div(id="tab-content"),
            
            dcc.Interval(
                id="interval-component",
                interval=settings.dashboard_refresh_seconds * 1000,
                n_intervals=0,
            ),
        ],
        style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
    )

    @callback(
        Output("tab-content", "children"),
        [
            Input("dashboard-tabs", "value"),
            Input("interval-component", "n_intervals"),
            Input("refresh-button", "n_clicks"),
            Input("symbol-select", "value"),
            Input("model-select", "value"),
            Input("horizon-select", "value"),
        ],
    )
    def update_tab_content(tab, n_intervals, n_clicks, symbol, model_type, horizon):
        """Update content based on selected tab."""
        try:
            if tab == "market-tab":
                return create_market_dashboard(symbol, model_type, horizon)
            elif tab == "performance-tab":
                return create_performance_monitor(symbol, model_type)
            elif tab == "learning-tab":
                return create_learning_log(symbol)
            elif tab == "portfolio-tab":
                return create_portfolio_manager(symbol)
        except Exception as e:
            return html.Div(f"Error loading dashboard: {str(e)}", style={"color": "red"})

    return app


def create_market_dashboard(symbol: str, model_type: str, horizon: str):
    """Market dashboard with candlestick and forecast overlay."""
    prices_data = fetch_prices(symbol)
    predictions_data = fetch_predictions(symbol, model_type, horizon)
    actuals_data = fetch_actuals(symbol)
    
    if not prices_data:
        return html.Div("No price data available. Please ingest data first.")
    
    fig = go.Figure()
    
    # Candlestick
    df = pd.DataFrame(prices_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    ))
    
    # Forecast overlay
    if predictions_data:
        last_timestamp = df["timestamp"].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=len(predictions_data),
            freq="H" if horizon.endswith("h") else "D",
        )
        
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=predictions_data,
            mode="lines+markers",
            name=f"Forecast ({model_type})",
            line=dict(color="orange", dash="dash", width=2),
            marker=dict(size=6),
        ))
    
    # Actual vs predicted (error overlay)
    if actuals_data and predictions_data:
        # Match actuals with predictions
        for actual in actuals_data:
            if actual.get("predicted_value") and actual.get("actual_value"):
                error = abs(actual["actual_value"] - actual["predicted_value"])
                error_pct = (error / actual["actual_value"]) * 100 if actual["actual_value"] != 0 else 0
                
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(actual["target_time"])],
                    y=[actual["actual_value"]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red" if error_pct > 5 else "yellow" if error_pct > 2 else "green",
                        symbol="x",
                    ),
                    name=f"Error: {error_pct:.1f}%",
                    showlegend=False,
                ))
    
    fig.update_layout(
        title=f"{symbol} - Price Chart with Forecast Overlay",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode="x unified",
    )
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div(id="forecast-stats", children=create_forecast_stats(symbol, model_type)),
    ])


def create_performance_monitor(symbol: str, model_type: str):
    """Performance monitoring with metrics trends and model drift."""
    metrics_data = fetch_metrics(symbol, model_type)
    model_versions = fetch_model_versions(symbol, model_type)
    
    fig = go.Figure()
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["mae"],
            mode="lines+markers",
            name="MAE",
            line=dict(color="blue", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["rmse"],
            mode="lines+markers",
            name="RMSE",
            line=dict(color="red", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["mape"],
            mode="lines+markers",
            name="MAPE (%)",
            line=dict(color="green", width=2),
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Metric Value",
        height=400,
        hovermode="x unified",
    )
    
    # Model drift visualization
    drift_fig = create_model_drift_chart(model_versions)
    
    return html.Div([
        html.H3("Performance Metrics Trends"),
        dcc.Graph(figure=fig),
        html.H3("Model Drift Visualization"),
        dcc.Graph(figure=drift_fig),
    ])


def create_learning_log(symbol: str):
    """Adaptive learning log showing retraining history."""
    model_versions = fetch_model_versions(symbol)
    
    if not model_versions:
        return html.Div("No model training history available.")
    
    df = pd.DataFrame(model_versions)
    df["trained_at"] = pd.to_datetime(df["trained_at"])
    
    # Create table
    table_rows = [
        html.Tr([
            html.Th("Version"),
            html.Th("Model Type"),
            html.Th("Horizon"),
            html.Th("Trained At"),
            html.Th("Status"),
        ])
    ]
    
    for _, row in df.iterrows():
        table_rows.append(html.Tr([
            html.Td(row["version_tag"][:30] + "..." if len(row["version_tag"]) > 30 else row["version_tag"]),
            html.Td(row.get("model_type", "Unknown")),
            html.Td(str(row.get("horizon", "N/A"))),
            html.Td(row["trained_at"].strftime("%Y-%m-%d %H:%M")),
            html.Td("✓" if row.get("fine_tuned") else "Initial"),
        ]))
    
    return html.Div([
        html.H3("Model Training History"),
        html.Table(
            table_rows,
            style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"},
        ),
        html.H3("Training Timeline"),
        dcc.Graph(figure=create_training_timeline(df)),
    ])


def create_portfolio_manager(symbol: str):
    """Portfolio management view with growth curve and metrics."""
    portfolio_data = fetch_portfolio(symbol)
    trades_data = fetch_trades(symbol)
    
    if not portfolio_data:
        return html.Div("No portfolio data available.")
    
    df = pd.DataFrame(portfolio_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Portfolio growth chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["total_value"],
        mode="lines+markers",
        name="Portfolio Value",
        line=dict(color="purple", width=3),
        fill="tozeroy",
    ))
    
    fig.update_layout(
        title="Portfolio Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
    )
    
    # Key metrics
    latest = df.iloc[-1] if len(df) > 0 else None
    metrics_html = html.Div([
        html.H3("Portfolio Metrics"),
        html.Div([
            html.Div([
                html.H4(f"${latest['total_value']:.2f}" if latest else "N/A"),
                html.P("Total Value"),
            ], style={"display": "inline-block", "margin": "20px", "textAlign": "center"}),
            html.Div([
                html.H4(f"{latest['sharpe_ratio']:.2f}" if latest else "N/A"),
                html.P("Sharpe Ratio"),
            ], style={"display": "inline-block", "margin": "20px", "textAlign": "center"}),
            html.Div([
                html.H4(f"{latest['daily_return']*100:.2f}%" if latest else "N/A"),
                html.P("Daily Return"),
            ], style={"display": "inline-block", "margin": "20px", "textAlign": "center"}),
        ]),
    ])
    
    return html.Div([
        metrics_html,
        dcc.Graph(figure=fig),
        html.H3("Recent Trades"),
        create_trades_table(trades_data),
    ])


# Helper functions
def fetch_prices(symbol: str, limit: int = 500):
    try:
        response = requests.get(f"{API_BASE}/api/data/prices/{symbol}?limit={limit}")
        if response.status_code == 200:
            return response.json().get("prices", [])
    except:
        pass
    return None


def fetch_predictions(symbol: str, model_type: str = "Ensemble", horizon: str = "24h"):
    try:
        response = requests.post(
            f"{API_BASE}/api/forecast/run",
            json={"symbol": symbol, "model_type": model_type, "horizon": horizon},
        )
        if response.status_code == 200:
            return response.json().get("predictions", [])
    except:
        pass
    return None


def fetch_metrics(symbol: str, model_type: str = None):
    try:
        url = f"{API_BASE}/api/evaluation/metrics/{symbol}?limit=50"
        if model_type:
            url += f"&model_type={model_type}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("trends", [])
    except:
        pass
    return None


def fetch_portfolio(symbol: str):
    try:
        response = requests.get(f"{API_BASE}/api/portfolio/history/{symbol}?limit=100")
        if response.status_code == 200:
            return response.json().get("history", [])
    except:
        pass
    return None


def fetch_model_versions(symbol: str, model_type: str = None):
    try:
        url = f"{API_BASE}/api/forecast/models/{symbol}"
        if model_type:
            url += f"?model_type={model_type}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("models", [])
    except:
        pass
    return None


def fetch_actuals(symbol: str):
    """Fetch actual vs predicted values from MongoDB."""
    try:
        from ..db.unified_db import UnifiedDataStore
        store = UnifiedDataStore()
        
        # Query forecasts from MongoDB
        forecasts = list(
            store.mongo_db.forecasts.find({
                "symbol": symbol,
                "actual_value": {"$ne": None}
            })
            .sort("target_time", -1)
            .limit(50)
        )
        
        return [
            {
                "target_time": f.get("target_time", ""),
                "predicted_value": f.get("predicted_value", 0),
                "actual_value": f.get("actual_value", 0),
                "absolute_error": f.get("absolute_error", 0),
            }
            for f in forecasts
        ]
    except Exception as e:
        print(f"Error fetching actuals: {e}")
        return []


def fetch_trades(symbol: str):
    """Fetch trades from MongoDB."""
    try:
        from ..db.unified_db import UnifiedDataStore
        store = UnifiedDataStore()
        
        # Query trades from MongoDB
        trades = list(
            store.mongo_db.trades.find({"symbol": symbol})
            .sort("timestamp", -1)
            .limit(20)
        )
        
        return [
            {
                "timestamp": t.get("timestamp", ""),
                "action": t.get("action", ""),
                "quantity": t.get("quantity", 0),
                "price": t.get("price", 0),
            }
            for t in trades
        ]
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return []


def create_forecast_stats(symbol: str, model_type: str):
    """Create forecast statistics display."""
    return html.Div([
        html.H4("Forecast Statistics"),
        html.P(f"Model: {model_type}"),
        html.P(f"Symbol: {symbol}"),
    ])


def create_model_drift_chart(model_versions):
    """Create model drift visualization."""
    fig = go.Figure()
    
    if model_versions:
        df = pd.DataFrame(model_versions)
        df["trained_at"] = pd.to_datetime(df["trained_at"])
        
        fig.add_trace(go.Scatter(
            x=df["trained_at"],
            y=range(len(df)),
            mode="lines+markers",
            name="Model Versions",
        ))
    
    fig.update_layout(
        title="Model Version Timeline",
        xaxis_title="Date",
        yaxis_title="Version Count",
        height=300,
    )
    return fig


def create_training_timeline(df: pd.DataFrame):
    """Create training timeline chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["trained_at"],
        y=[1] * len(df),
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Training Events",
    ))
    
    fig.update_layout(
        title="Model Training Timeline",
        xaxis_title="Date",
        yaxis_title="",
        height=200,
        showlegend=False,
    )
    return fig


def create_trades_table(trades_data):
    """Create trades table."""
    if not trades_data:
        return html.Div("No trades yet.")
    
    rows = [html.Tr([html.Th("Time"), html.Th("Action"), html.Th("Quantity"), html.Th("Price")])]
    
    for trade in trades_data[:10]:
        rows.append(html.Tr([
            html.Td(trade["timestamp"][:19]),
            html.Td(trade["action"], style={"color": "green" if trade["action"] == "BUY" else "red"}),
            html.Td(f"{trade['quantity']:.4f}"),
            html.Td(f"${trade['price']:.2f}"),
        ]))
    
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})

