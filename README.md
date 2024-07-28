# AI-Driven Financial Forecasting Tool

## Description

This project is an AI-driven tool designed to predict future stock prices using historical stock price data. It leverages Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network (RNN) that excels in modeling time series data. The solution is implemented in Python and utilizes popular libraries such as `yfinance` for data collection, `numpy` and `pandas` for data preprocessing, `scikit-learn` for normalization, and `TensorFlow` for building and training the LSTM model.

Please also find a PPT `Financial Forecasting using AI` for your reference on this project.


## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Function Descriptions](#function-descriptions)

## Installation

To run this project, ensure you have Python installed on your machine. Then, install the necessary libraries using the following commands:

```bash
pip install yfinance
pip install numpy pandas scikit-learn matplotlib tensorflow
```

## Usage
To use the financial forecasting tool, follow these steps:

1. Clone the Repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Run the Script:
Modify the main function parameters as needed (e.g., ticker symbol, date range), then execute the script:

```bash
python financial_forecasting.py
```

## Project Structure

├── financial_forecasting.py         # Main script

└── README.md                        # Project documentation

└── Financial Forecasting using AI   # PPT Project Documentation


## Function Descriptions
1. get_stock_data(ticker, start_date, end_date):
Retrieves historical stock price data for a specified ticker symbol and date range using the yfinance library.

2. preprocess_data(stock_data):
Cleans and prepares the data for modeling, including normalization and splitting into training and test sets. Creates input sequences for the LSTM model.

3. build_model():
Builds an LSTM neural network model with specified architecture.

4. train_model(model, x_train, y_train, epochs=1, batch_size=1):
Trains the LSTM model using the training data for a specified number of epochs and batch size.

5. evaluate_model(model, x_test, y_test, scaler):
Evaluates the trained model using the test data, calculates performance metrics, and visualizes the actual vs. predicted stock prices.
