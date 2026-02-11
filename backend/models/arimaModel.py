# backend/models/arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_arima(df, column='Close', order=(5,1,0), forecast_horizon=24):
    model = ARIMA(df[column], order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=forecast_horizon)
    return fitted, forecast

def evaluate_arima(df, column='Close', order=(5,1,0)):
    size = int(len(df) * 0.8)
    train, test = df[column][:size], df[column][size:]
    model = ARIMA(train, order=order)
    fitted = model.fit()
    preds = fitted.forecast(steps=len(test))

    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    return mae, rmse
