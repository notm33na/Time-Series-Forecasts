# backend/models/ensemble_model.py
import numpy as np

def ensemble_forecast(arima_forecast, lstm_forecast, weight_arima=0.4, weight_lstm=0.6):
    min_len = min(len(arima_forecast), len(lstm_forecast))
    arima_f = np.array(arima_forecast[:min_len])
    lstm_f = np.array(lstm_forecast[:min_len])
    combined = (weight_arima * arima_f) + (weight_lstm * lstm_f)
    return combined
