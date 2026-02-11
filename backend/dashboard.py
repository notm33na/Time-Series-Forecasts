"""
Interactive Plotly Dash dashboard for monitoring forecasts, metrics, and portfolio.
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
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
        url_base_pathname="/dashboard/",
    )

    app.layout = html.Div(
        [
            html.H1("Adaptive Forecasting Dashboard", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Label("Symbol:"),
                    dcc.Input(
                        id="symbol-input",
                        type="text",
                        value=settings.symbol,
                        style={"marginRight": "10px"},
                    ),
                    html.Label("Refresh Interval (seconds):"),
                    dcc.Input(
                        id="refresh-input",
                        type="number",
                        value=settings.dashboard_refresh_seconds,
                        min=1,
                        style={"marginRight": "10px"},
                    ),
                    html.Button("Refresh", id="refresh-button", n_clicks=0),
                ],
                style={"padding": "20px", "textAlign": "center"},
            ),
            dcc.Interval(
                id="interval-component",
                interval=settings.dashboard_refresh_seconds * 1000,
                n_intervals=0,
            ),
            html.Div(id="dashboard-content"),
        ]
    )

    @callback(
        Output("dashboard-content", "children"),
        [Input("interval-component", "n_intervals"), Input("refresh-button", "n_clicks"), Input("symbol-input", "value")],
    )
    def update_dashboard(n_intervals, n_clicks, symbol):
        """Update dashboard content."""
        try:
            # Fetch data
            prices_data = fetch_prices(symbol)
            metrics_data = fetch_metrics(symbol)
            portfolio_data = fetch_portfolio(symbol)
            predictions_data = fetch_predictions(symbol)

            if not prices_data:
                return html.Div("No data available. Please ingest data first.")

            # Create charts
            candlestick_chart = create_candlestick_chart(prices_data, predictions_data)
            metrics_chart = create_metrics_chart(metrics_data)
            portfolio_chart = create_portfolio_chart(portfolio_data)

            return html.Div(
                [
                    html.Div(
                        [
                            html.H3("Price Forecast with Error Overlay"),
                            dcc.Graph(figure=candlestick_chart),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Metrics Trends"),
                                    dcc.Graph(figure=metrics_chart),
                                ],
                                style={"width": "50%", "display": "inline-block"},
                            ),
                            html.Div(
                                [
                                    html.H3("Portfolio Performance"),
                                    dcc.Graph(figure=portfolio_chart),
                                ],
                                style={"width": "50%", "display": "inline-block"},
                            ),
                        ],
                    ),
                ]
            )
        except Exception as e:
            return html.Div(f"Error loading dashboard: {str(e)}", style={"color": "red"})

    return app


def fetch_prices(symbol: str, limit: int = 500) -> Optional[list]:
    """Fetch price data from API."""
    try:
        response = requests.get(f"{API_BASE}/api/data/prices/{symbol}?limit={limit}")
        if response.status_code == 200:
            return response.json().get("prices", [])
    except Exception:
        pass
    return None


def fetch_metrics(symbol: str) -> Optional[list]:
    """Fetch metrics data from API."""
    try:
        response = requests.get(f"{API_BASE}/api/evaluation/metrics/{symbol}?limit=50")
        if response.status_code == 200:
            return response.json().get("trends", [])
    except Exception:
        pass
    return None


def fetch_portfolio(symbol: str) -> Optional[list]:
    """Fetch portfolio data from API."""
    try:
        response = requests.get(f"{API_BASE}/api/portfolio/history/{symbol}?limit=100")
        if response.status_code == 200:
            return response.json().get("history", [])
    except Exception:
        pass
    return None


def fetch_predictions(symbol: str) -> Optional[list]:
    """Fetch predictions from API."""
    try:
        response = requests.post(
            f"{API_BASE}/api/models/predict",
            json={"symbol": symbol, "horizon": settings.forecast_horizon},
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("predictions", [])
    except Exception:
        pass
    return None


def create_candlestick_chart(prices_data: list, predictions: Optional[list] = None) -> go.Figure:
    """Create candlestick chart with predictions and error overlay."""
    if not prices_data:
        return go.Figure()

    df = pd.DataFrame(prices_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Candlestick
    candlestick = go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    )

    traces = [candlestick]

    # Add predictions if available
    if predictions:
        last_timestamp = df["timestamp"].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(days=1),
            periods=len(predictions),
            freq="D",
        )

        predictions_trace = go.Scatter(
            x=future_timestamps,
            y=predictions,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="orange", dash="dash"),
        )
        traces.append(predictions_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Price Chart with Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    return fig


def create_metrics_chart(metrics_data: Optional[list]) -> go.Figure:
    """Create metrics trends chart."""
    if not metrics_data:
        return go.Figure()

    df = pd.DataFrame(metrics_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["mae"],
            mode="lines+markers",
            name="MAE",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["rmse"],
            mode="lines+markers",
            name="RMSE",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["mape"],
            mode="lines+markers",
            name="MAPE (%)",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title="Model Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Metric Value",
        height=400,
    )
    return fig


def create_portfolio_chart(portfolio_data: Optional[list]) -> go.Figure:
    """Create portfolio performance chart."""
    if not portfolio_data:
        return go.Figure()

    df = pd.DataFrame(portfolio_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["total_value"],
            mode="lines+markers",
            name="Portfolio Value",
            line=dict(color="purple"),
        )
    )

    fig.update_layout(
        title="Portfolio Growth",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
    )
    return fig

