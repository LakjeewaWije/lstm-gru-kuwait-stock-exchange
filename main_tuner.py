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
import keras_tuner as kt
import matplotlib.dates as mdates
import joblib
import config

# ==============================
# 2. Load NBK.KW Data (2015–2025)
# ==============================
ticker = "NBK.KW"
data = yf.download(ticker, start=config.start, end=config.end)
data.dropna(inplace=True)

# ==============================
# 3. Feature Engineering (OHLC + Volume)
# ==============================
data['Volume'] = np.log1p(data['Volume'])
features = ['Open', 'High', 'Low', 'Close', 'Volume']
dataset = data[features].values

# ==============================
# 4. Dataset Creation
# ==============================
window_size = 60
forecast_horizon = 5

def create_dataset(data, window_size=60, forecast_horizon=5):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 3])
    return np.array(X), np.array(y)

X, y = create_dataset(dataset, window_size, forecast_horizon)

# Split
split = int(0.8 * len(X))
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
# 6. Keras Tuner Model Builders
# ==============================

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def build_lstm_tuner(hp):
    model = Sequential()
    units = hp.Choice("units", [32, 64, 128, 256])
    num_layers = hp.Choice("num_layers", [1, 2, 3])
    dropout_rate = hp.Choice("dropout", [0.1, 0.2, 0.3, 0.5])

    model.add(Bidirectional(LSTM(units, return_sequences=(num_layers > 1)),
                            input_shape=(window_size, X_train.shape[2])))
    model.add(Dropout(dropout_rate))

    for i in range(num_layers - 1):
        return_seq = (i < num_layers - 2)
        model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(forecast_horizon))

    lr = hp.Choice("learning_rate", [0.01, 0.001, 0.0005, 0.0001])
    opt_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
    optimizer = Adam(lr) if opt_name == "adam" else RMSprop(lr) if opt_name == "rmsprop" else SGD(lr)

    loss_fn = hp.Choice("loss", ["mae", "mse", "huber"])
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model


def build_gru_tuner(hp):
    model = Sequential()
    units = hp.Choice("units", [32, 64, 128, 256])
    num_layers = hp.Choice("num_layers", [1, 2, 3])
    dropout_rate = hp.Choice("dropout", [0.1, 0.2, 0.3, 0.5])

    model.add(Bidirectional(GRU(units, return_sequences=(num_layers > 1)),
                            input_shape=(window_size, X_train.shape[2])))
    model.add(Dropout(dropout_rate))

    for i in range(num_layers - 1):
        return_seq = (i < num_layers - 2)
        model.add(Bidirectional(GRU(units, return_sequences=return_seq)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(forecast_horizon))

    lr = hp.Choice("learning_rate", [0.01, 0.001, 0.0005, 0.0001])
    opt_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
    optimizer = Adam(lr) if opt_name == "adam" else RMSprop(lr) if opt_name == "rmsprop" else SGD(lr)

    loss_fn = hp.Choice("loss", ["mae", "mse", "huber"])
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

# ==============================
# 7. Run Hyperparameter Search
# ==============================

lstm_tuner = kt.Hyperband(
    build_lstm_tuner,
    objective="val_loss",
    max_epochs=20,
    factor=3,
    directory="tuner_results",
    project_name="lstm_tuning"
)

gru_tuner = kt.Hyperband(
    build_gru_tuner,
    objective="val_loss",
    max_epochs=20,
    factor=3,
    directory="tuner_results",
    project_name="gru_tuning"
)

print("Tuning LSTM...")
lstm_tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

print("Tuning GRU...")
gru_tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

best_lstm_hp = lstm_tuner.get_best_hyperparameters(1)[0]
best_gru_hp = gru_tuner.get_best_hyperparameters(1)[0]

print("Best LSTM Hyperparameters:", best_lstm_hp.values)
print("Best GRU Hyperparameters:", best_gru_hp.values)

best_lstm_model = lstm_tuner.get_best_models(1)[0]
best_gru_model = gru_tuner.get_best_models(1)[0]

# ==============================
# 8. Train Best Models Fully
# ==============================

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training Best LSTM...")
best_lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=32, callbacks=[lr_scheduler, early_stop])

print("Training Best GRU...")
best_gru_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                   epochs=100, batch_size=32, callbacks=[lr_scheduler, early_stop])

# ==============================
# 9. Save Models
# ==============================
safe_ticker = ticker.replace(".", "_")
best_gru_model.save(f"models/{safe_ticker}/gru_model.h5")
joblib.dump(scaler, f"models/{safe_ticker}/feature_scaler.pkl")
joblib.dump(y_scaler, f"models/{safe_ticker}/target_scaler.pkl")

# ==============================
# 10. Predictions
# ==============================
lstm_predictions = best_lstm_model.predict(X_test)
gru_predictions = best_gru_model.predict(X_test)

lstm_predictions_rescaled = y_scaler.inverse_transform(lstm_predictions)
gru_predictions_rescaled = y_scaler.inverse_transform(gru_predictions)
y_test_rescaled = y_scaler.inverse_transform(y_test)

# ==============================
# 11. Per-Horizon Evaluation
# ==============================
def evaluate_per_horizon(name, y_true, y_pred):
    print(f"\n{name} Per-Horizon Evaluation:")
    for h in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:,h], y_pred[:,h])
        mse = mean_squared_error(y_true[:,h], y_pred[:,h])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true[:,h] - y_pred[:,h]) / y_true[:,h])) * 100
        print(f"Horizon {h+1}: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

evaluate_per_horizon("LSTM", y_test_rescaled, lstm_predictions_rescaled)
evaluate_per_horizon("GRU", y_test_rescaled, gru_predictions_rescaled)

# ==============================
# 12. Baseline
# ==============================
print("\nNaive Baseline (Last Value Forward):")
naive_preds = np.repeat(y_test_rescaled[:,0].reshape(-1,1), forecast_horizon, axis=1)
evaluate_per_horizon("Naive", y_test_rescaled, naive_preds)

# ==============================
# 13. Plot Horizon-1
# ==============================
test_dates = data.index[split + window_size : split + window_size + len(y_test)]
plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_rescaled[:,0], label="Actual", color='blue')
plt.plot(test_dates, lstm_predictions_rescaled[:,0], label="LSTM", color='orange')
plt.plot(test_dates, gru_predictions_rescaled[:,0], label="GRU", color='green')
plt.title("Horizon-1 Forecast Comparison")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 14. Forecast Next 5 Days
# ==============================
last_window = scaler.transform(dataset[-window_size:]).reshape(1, window_size, -1)

lstm_forecast = y_scaler.inverse_transform(best_lstm_model.predict(last_window)).flatten()
gru_forecast  = y_scaler.inverse_transform(best_gru_model.predict(last_window)).flatten()

last_date = data.index[-1]
forecast_dates = []
while len(forecast_dates) < forecast_horizon:
    last_date += pd.Timedelta(days=1)
    if last_date.weekday() in [6,0,1,2,3]:
        forecast_dates.append(last_date)

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "LSTM Forecast": lstm_forecast,
    "GRU Forecast": gru_forecast
})

print("\nNext 5-Day Forecasts:")
print(forecast_df)