
# ==============================
# 1. Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ==============================
# 2. Load NBK.KW Data (2015–2025)
# ==============================
ticker = "NBK.KW"
data = yf.download(ticker, start="2015-01-01", end="2025-11-25")
data.dropna(inplace=True)

# ==============================
# 3. Feature Engineering (OHLC + Volume only)
# ==============================
data['Volume'] = np.log1p(data['Volume'])  # log-transform volume
features = ['Open', 'High', 'Low', 'Close', 'Volume']
dataset = data[features].values

# Print the first 5 rows of the dataset
print("First 5 rows:")
print(dataset[:5])

# Print the total number of rows and columns
print("\nShape of dataset (rows, columns):")
print(dataset.shape)

# ==============================
# 4. Dataset Creation Function
# ==============================
window_size = 60
forecast_horizon = 5

def create_dataset(data, window_size=60, forecast_horizon=5):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 3])  # Close index
    return np.array(X), np.array(y)

X, y = create_dataset(dataset, window_size, forecast_horizon)

# Split the dataset into training and testing sets
split = int(0.8 * len(X))
print("Split:", split , len(X))
X_train_raw, X_test_raw = X[:split], X[split:]
y_train_raw, y_test_raw = y[:split], y[split:]


# ==============================
# 5. Scaling
# ==============================
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# ==============================
# 6. Build Models
# ==============================
def build_lstm(input_shape, horizon):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))
    model.add(Dense(horizon))
    model.compile(optimizer='adam', loss='mae')
    return model

def build_gru(input_shape, horizon):
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.3))
    model.add(Dense(horizon))
    model.compile(optimizer='adam', loss='mae')
    return model

lstm_model = build_lstm((window_size, X_train.shape[2]), forecast_horizon)
gru_model = build_gru((window_size, X_train.shape[2]), forecast_horizon)

# ==============================
# 7. Train Models
# ==============================
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training LSTM...")
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32,
               validation_data=(X_test, y_test),
               callbacks=[lr_scheduler, early_stop], verbose=1)

print("Training GRU...")
gru_model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[lr_scheduler, early_stop], verbose=1)

# ==============================
# 8. Predictions
# ==============================
lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

lstm_predictions_rescaled = y_scaler.inverse_transform(lstm_predictions)
gru_predictions_rescaled = y_scaler.inverse_transform(gru_predictions)
y_test_rescaled = y_scaler.inverse_transform(y_test)