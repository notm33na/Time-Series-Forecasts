# backend/models/lstm_model.py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .utils import preprocess_data, create_sequences

def train_lstm(df, column='Close', seq_length=60, epochs=10, forecast_horizon=24):
    train, test, scaler = preprocess_data(df, column)
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_inv, preds)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds))

    # Forecast next N points
    last_seq = test[-seq_length:]
    input_seq = last_seq.reshape((1, seq_length, 1))
    forecast = []
    curr_seq = input_seq
    for _ in range(forecast_horizon):
        pred = model.predict(curr_seq)[0][0]
        forecast.append(pred)
        curr_seq = np.append(curr_seq[:,1:,:], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    return model, forecast, mae, rmse
