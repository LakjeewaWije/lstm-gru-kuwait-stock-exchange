import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import config

def predict_next_5_closes(ticker,gru_model_path,scaler_path,y_scaler_path,start=config.START_DATE,end=config.END_DATE, window_size=60, forecast_horizon=5):
    # Load saved models and scalers
    gru_model  = load_model(gru_model_path)
    scaler     = joblib.load(scaler_path)
    y_scaler   = joblib.load(y_scaler_path)

    # Load latest data
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)

    # Feature engineering
    data['Volume'] = np.log1p(data['Volume'])
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset = data[features].values

    # Prepare last input window
    last_window = scaler.transform(dataset[-window_size:]).reshape(1, window_size, -1)

    # Predict next 5 closes
    gru_forecast  = y_scaler.inverse_transform(gru_model.predict(last_window)).flatten()

    # Generate next 5 valid trading dates (Sun–Thu only)
    last_date = data.index[-1]
    forecast_dates = []
    while len(forecast_dates) < forecast_horizon:
        last_date += pd.Timedelta(days=1)
        if last_date.weekday() in [6,0,1,2,3]:  # Sun=6, Mon=0 ... Thu=3
            forecast_dates.append(last_date)

    # Return forecast with dates
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "GRU Forecast": gru_forecast
    })

    print("\nNext 5-Day Forecasts:")
    print(forecast_df.to_string(index=False))
