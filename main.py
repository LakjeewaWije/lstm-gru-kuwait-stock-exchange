
# ==============================
# 1. Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.dates as mdates
import joblib
import config

# ==============================
# 2. Load NBK.KW Data (2015–2025)
# ==============================
ticker = "BOUBYAN.KW" # Replace with the desired stock ticker - NBK.KW , KFH.KW , ZAIN.KW , BOUBYAN.KW
data = yf.download(ticker, start=config.start, end=config.end)
data.dropna(inplace=True)

# ==============================
# 3. Feature Engineering (OHLC + Volume only)
# ==============================
data['Volume'] = np.log1p(data['Volume'])  # log-transform volume
features = ['Open', 'High', 'Low', 'Close', 'Volume']
dataset = data[features].values

# Print the first 5 rows with dates
print("First 5 rows with dates:")
print(data[features].head())

# Print the last 5 rows with dates
print("\nLast 5 rows with dates:")
print(data[features].tail())

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
# 8. Save trained models and scalers
# ==============================
safe_ticker = ticker.replace(".", "_")   # NBK_KW
gru_model.save(f"models/{safe_ticker}/gru_model.h5")
joblib.dump(scaler, f"models/{safe_ticker}/feature_scaler.pkl")
joblib.dump(y_scaler, f"models/{safe_ticker}/target_scaler.pkl")

print(f"Models and scalers saved successfully at models/{safe_ticker} folder.")


# ==============================
# 9. Predictions
# ==============================
lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

lstm_predictions_rescaled = y_scaler.inverse_transform(lstm_predictions)
gru_predictions_rescaled = y_scaler.inverse_transform(gru_predictions)
y_test_rescaled = y_scaler.inverse_transform(y_test)


# ==============================
# 10. Per-Horizon Evaluation
# ==============================
def evaluate_per_horizon(name, y_true, y_pred):
    print(f"\n{name} Per-Horizon Evaluation:")
    for h in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:,h], y_pred[:,h])
        rmse = np.sqrt(mean_squared_error(y_true[:,h], y_pred[:,h]))
        mape = np.mean(np.abs((y_true[:,h] - y_pred[:,h]) / y_true[:,h])) * 100
        r2 = r2_score(y_true[:,h], y_pred[:,h])
        print(f"Horizon {h+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")

evaluate_per_horizon("LSTM", y_test_rescaled, lstm_predictions_rescaled)
evaluate_per_horizon("GRU", y_test_rescaled, gru_predictions_rescaled)

# ==============================
# 11. Baseline Comparisons
# ==============================
print("\nNaive Baseline (Last Value Forward):")
naive_preds = np.repeat(y_test_rescaled[:,0].reshape(-1,1), forecast_horizon, axis=1)
evaluate_per_horizon("Naive", y_test_rescaled, naive_preds)


# ==============================
# 12. Plot Example Horizon-1 Results
# ==============================
test_dates = data.index[split + window_size : split + window_size + len(y_test)]
plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_rescaled[:,0], label="Actual (Horizon-1)", color='blue')
plt.plot(test_dates, lstm_predictions_rescaled[:,0], label="LSTM Horizon-1", color='orange')
plt.plot(test_dates, gru_predictions_rescaled[:,0], label="GRU Horizon-1", color='green')
plt.title("Horizon-1 Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Price (KWD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 13. Forecast Next 5 Trading Days (Sun–Thu)
# ==============================

# Prepare last input window
last_window = scaler.transform(dataset[-window_size:]).reshape(1, window_size, -1)

# Predict next 5 closes
lstm_forecast = y_scaler.inverse_transform(lstm_model.predict(last_window)).flatten()
gru_forecast  = y_scaler.inverse_transform(gru_model.predict(last_window)).flatten()

# Generate next 5 valid trading dates (Sun–Thu only)
last_date = data.index[-1]
forecast_dates = []
while len(forecast_dates) < forecast_horizon:
    last_date += pd.Timedelta(days=1)
    if last_date.weekday() in [6,0,1,2,3]:  # Sun=6, Mon=0 ... Thu=3
        forecast_dates.append(last_date)

# Build forecast DataFrame
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "LSTM Forecast": lstm_forecast,
    "GRU Forecast": gru_forecast
})

print("\nNext 5-Day Forecasts:")
print(forecast_df)

