from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Example forecast data for 4 tickers
    forecasts = {
        "NBK.KW": [
            {"date": "2025-12-01", "price": 120.5},
            {"date": "2025-12-02", "price": 121.2},
            {"date": "2025-12-03", "price": 119.8},
            {"date": "2025-12-04", "price": 122.0},
            {"date": "2025-12-05", "price": 123.1},
        ],
        "KFH.KW": [
            {"date": "2025-12-01", "price": 750.0},
            {"date": "2025-12-02", "price": 752.5},
            {"date": "2025-12-03", "price": 748.9},
            {"date": "2025-12-04", "price": 755.3},
            {"date": "2025-12-05", "price": 760.0},
        ],
        "ZAIN.KW": [
            {"date": "2025-12-01", "price": 620.0},
            {"date": "2025-12-02", "price": 618.5},
            {"date": "2025-12-03", "price": 622.0},
            {"date": "2025-12-04", "price": 625.0},
            {"date": "2025-12-05", "price": 627.5},
        ],
        "BOUBYAN.KW": [
            {"date": "2025-12-01", "price": 300.0},
            {"date": "2025-12-02", "price": 302.0},
            {"date": "2025-12-03", "price": 299.5},
            {"date": "2025-12-04", "price": 303.0},
            {"date": "2025-12-05", "price": 305.0},
        ],
    }
    return render_template("dashboard.html", forecasts=forecasts)

if __name__ == '__main__':
    app.run(debug=True)