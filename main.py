
# ==============================
# 1. Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

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