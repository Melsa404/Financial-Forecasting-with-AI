import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#Data Collection
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

#Data Preprocessing
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    train_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:train_data_len, :]
    
    x_train = []
    y_train = []
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = []
    y_test = stock_data['Close'][train_data_len:].values
    
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, y_test, scaler

#Model Building
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Training
def train_model(model, x_train, y_train, epochs=1, batch_size=1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

#Evaluation
def evaluate_model(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(f'Root Mean Squared Error: {rmse}')
    
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD')
    plt.plot(y_test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.legend(['Actual', 'Predicted'], loc='lower right')
    plt.show()

#Main Function
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2010-01-01'
end_date = '2022-01-01'
    
stock_data = get_stock_data(ticker, start_date, end_date)
x_train, y_train, x_test, y_test, scaler = preprocess_data(stock_data)
model = build_model()
train_model(model, x_train, y_train, epochs=1, batch_size=1)
evaluate_model(model, x_test, y_test, scaler)
