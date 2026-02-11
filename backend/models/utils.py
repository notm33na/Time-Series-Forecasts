# backend/models/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df

def preprocess_data(df, column='Close', train_split=0.8):
    data = df[[column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * train_split)
    train, test = scaled_data[:train_size], scaled_data[train_size:]
    return train, test, scaler

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
