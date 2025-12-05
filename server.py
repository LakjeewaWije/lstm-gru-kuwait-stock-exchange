from flask import Flask, render_template
from predict import predict_next_5_closes

app = Flask(__name__)

@app.route('/')
def dashboard():
    tickers = ["NBK.KW", "KFH.KW", "ZAIN.KW", "BOUBYAN.KW"]
    # tickers = ["NBK.KW"]
    forecasts = {}
    for ticker in tickers:
        safe_ticker = ticker.replace(".", "_")

        result = predict_next_5_closes(
            ticker=ticker,
            gru_model_path=f"models/{safe_ticker}/gru_model.h5",
            scaler_path=f"models/{safe_ticker}/feature_scaler.pkl",
            y_scaler_path=f"models/{safe_ticker}/target_scaler.pkl"
        )

        # result is already {ticker: [ {date, price}, ... ] }
        forecasts.update(result)
    print(forecasts)
    # Now render with actual forecasts
    return render_template("dashboard.html", forecasts=forecasts)

if __name__ == '__main__':
    app.run(debug=True)