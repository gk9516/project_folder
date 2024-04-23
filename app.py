import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

model =load_model('stock_prediction.keras')
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol','AAPL')
start='1900-01-01'

link_data = yf.download(stock,start)

st.subheader('Stock Data')
st.write(link_data)

link_data.dropna(inplace=True)
link_data = link_data[['Close']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(link_data)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

def forecast_future_values(model, X, num_days):
    forecast = []
    current_sequence = X[-seq_length:]
    for _ in range(num_days):
        next_value = model.predict(current_sequence.reshape(1, seq_length, 1))[0][0]
        forecast.append(next_value)
        current_sequence = np.append(current_sequence[1:], next_value)
    return forecast


future_forecast = forecast_future_values(model, X_val[-1], 6)
actual_prices = scaler.inverse_transform(y_val[-6:].reshape(-1, 1))
predicted_prices = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_train_pred_inv = scaler.inverse_transform(y_train_pred)
y_val_pred_inv = scaler.inverse_transform(y_val_pred)

st.header('LSTM Model')

st.subheader('Actual Price Vs Predicted Price')
last_date = link_data.index[-1]
fig2=plt.figure(figsize=(12,6))
plt.plot(link_data.index[train_size:], link_data['Close'][train_size:], label='Actual Prices (Validation)', color='blue')
plt.plot(link_data.index[train_size:][-len(y_val_pred_inv):], y_val_pred_inv, label='Predicted Prices (Validation)', linestyle='dashed', color='orange')
predicted_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=6)
plt.plot(predicted_dates, predicted_prices, label='Predicted Prices (Next 6 Days)', linestyle='dashed', color='green')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
st.pyplot(fig2)

last_date = link_data.index[-1]

next_six_days = pd.date_range(last_date + pd.Timedelta(days=1), periods=6)
st.subheader('Predicted Prices')
print("Predicted Prices for the Next:")
for i, (date, price) in enumerate(zip(next_six_days, predicted_prices), start=1):
    st.text(f"{date.strftime('%Y-%m-%d')}: {price[0]}")

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print("Root Mean Square Error (RMSE):", rmse)
st.subheader('Root Mean Square Error (RMSE)')
st.text(rmse)

st.header('Conclusion')
st.write("The LSTM model provide predictions for future stock prices based on historical data. While these predictions offer valuable insights, it's important to note that they may not always accurately reflect actual market conditions. Factors such as unforeseen events, economic indicators, and investor sentiment can influence stock prices in ways that models may not capture. Therefore, these predictions should be used as tools to inform decision-making rather than as definitive forecasts. By combining these models with fundamental analysis and market expertise, investors can make more informed decisions in navigating the dynamic landscape of the stock market.")
