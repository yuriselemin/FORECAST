from datetime import timedelta
from decimal import Decimal
import os
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from torch.nn import Sequential
from tensorflow.keras.layers import Dense, LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .models import Stock
from django.http import HttpResponse
from django.shortcuts import render
from pandas_datareader import data as pdr
from .forms import AnalyzeForm
from django.shortcuts import redirect
import tempfile
import yfinance as yf
from tempfile import NamedTemporaryFile



def index(request):
    stocks = Stock.objects.all()
    context = {'stocks': stocks}
    return render(request, 'forecast_app/index.html', context)


def analyze(request):
    ticker_choices = [(ticker, ticker) for ticker in ['AAPL', 'TSLA', 'GOOGL']]

    if request.method == 'POST':
        form = AnalyzeForm(ticker_choices, request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            start_date_str = form.cleaned_data['start_date']
            end_date_str = form.cleaned_data['end_date']
            model_type = form.cleaned_data['model_type']

            # Проверяем, что передаются корректные значения
            print(f"Ticker: {ticker}, Start Date: {start_date_str}, End Date: {end_date_str}")

            try:
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str) + timedelta(days=1)
            except ValueError as e:
                print(e)
                return HttpResponse("Invalid date format. Please use YYYY-MM-DD.")

            df = get_stock_data(ticker, start_date, end_date)
            results = analyze_data(df, model_type)

            stock = Stock(
                ticker=ticker,
                start_date=start_date.date(),
                end_date=end_date.date() - timedelta(days=1),
                model_type=model_type,
                mse=Decimal(str(results['mse'])),
                mae=Decimal(str(results['mae']))
            )
            stock.save()

            context = {
                'stock': stock,
                'historical_dates': list(results['historical_dates']),
                'historical_prices': list(results['historical_prices']),
                'predicted_dates': list(results['predicted_dates']),
                'predicted_prices': list(results['predicted_prices'])
            }
            return render(request, 'forecast_app/analysis.html', context)
    else:
        form = AnalyzeForm(ticker_choices)
        return render(request, 'forecast_app/analyze.html', {'form': form})

def download_csv(request, stock_id):
    stock = Stock.objects.get(id=stock_id)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{stock.ticker}_{stock.start_date}_{stock.end_date}.csv"'

    writer = csv.writer(response)
    writer.writerow(['Date', 'Actual Price', 'Predicted Price'])
    for date, actual_price, predicted_price in zip(stock.historical_dates, stock.historical_prices, stock.predicted_prices):
        writer.writerow([date.strftime('%Y-%m-%d'), actual_price, predicted_price])

    return response


def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start_date, end_date)[['Adj Close']]

    with NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=True)

        temp_file.seek(0)
        df_from_csv = pd.read_csv(temp_file.name, parse_dates=[0], index_col=0)

    temp_file.close()
    os.remove(temp_file.name)

    return df_from_csv


def analyze_data(df, model_type):
    close_prices = df['Adj Close']
    dates = df.index

    # Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    # Split into training and testing sets
    split_index = int(len(scaled_close_prices) * 0.8)
    train_data = scaled_close_prices[:split_index]
    test_data = scaled_close_prices[split_index:]

    train_dates = dates[:split_index].strftime("%Y-%m-%d")
    test_dates = dates[split_index:].strftime("%Y-%m-%d")

    if model_type == 'sklearn':
        results = run_sklearn_model(train_data, test_data, train_dates, test_dates, scaler)
    elif model_type == 'tensorflow':
        results = run_tensorflow_model(train_data, test_data, train_dates, test_dates, scaler)
    elif model_type == 'pytorch':
        results = run_pytorch_model(train_data, test_data, train_dates, test_dates, scaler)

    return results


def run_sklearn_model(train_data, test_data, train_dates, test_dates, scaler):
    # Create features and targets
    X_train = np.array(range(len(train_data))).reshape(-1, 1)
    y_train = train_data

    X_test = np.array(range(len(test_data) + len(train_data))).reshape(-1, 1)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Calculate metrics
    original_test_data = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()
    original_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    mse = mean_squared_error(original_test_data, original_predictions)
    mae = mean_absolute_error(original_test_data, original_predictions)

    results = {
        'mse': mse,
        'mae': mae,
        'historical_dates': train_dates,
        'historical_prices': scaler.inverse_transform(train_data.reshape(-1, 1)).flatten(),
        'predicted_dates': test_dates,
        'predicted_prices': original_predictions
    }

    return results


def run_tensorflow_model(train_data, test_data, train_dates, test_dates, scaler):
    X_train, X_test, y_train, y_test = prepare_lstm_data(train_data, test_data)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    predictions = model.predict(X_test).flatten()

    # Rescale back to original values
    original_y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    original_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Metrics
    mse = mean_squared_error(original_y_test, original_predictions)
    mae = mean_absolute_error(original_y_test, original_predictions)

    results = {
        'mse': mse,
        'mae': mae,
        'historical_dates': train_dates,
        'historical_prices': scaler.inverse_transform(train_data.reshape(-1, 1)).flatten(),
        'predicted_dates': test_dates,
        'predicted_prices': original_predictions
    }

    return results


def prepare_lstm_data(train_data, test_data):
    time_steps = 1
    X_train = []
    y_train = []

    for i in range(time_steps, len(train_data)):
        X_train.append(train_data[i-time_steps:i])
        y_train.append(train_data[i])

    X_test = []
    y_test = []

    offset = len(train_data) - time_steps
    for j in range(offset, len(test_data) + offset):
        X_test.append(test_data[j-offset:j])
        y_test.append(test_data[j])

    X_train = np.array(X_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    y_train = np.array(y_train)

    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def run_pytorch_model(train_data, test_data, train_dates, test_dates, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
            return out

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = TensorDataset(torch.from_numpy(train_data).float().view(-1, 1).to(device), torch.from_numpy(np.arange(len(train_data))).long().view(-1, 1).to(device))
    test_dataset = TensorDataset(torch.from_numpy(test_data).float().view(-1, 1).to(device), torch.from_numpy(np.arange(len(test_data)) + len(train_data)).long().view(-1, 1).to(device))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_data), shuffle=False)

    for epoch in range(100):
        for i, (inputs, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), inputs)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        inputs = torch.from_numpy(test_data).float().unsqueeze(1).to(device)
        predictions = model(inputs).cpu().numpy().flatten()

    # Rescale back to original values
    original_test_data = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()
    original_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    mse = mean_squared_error(original_test_data, original_predictions)
    mae = mean_absolute_error(original_test_data, original_predictions)

    results = {
        'mse': mse,
        'mae': mae,
        'historical_dates': train_dates,
        'historical_prices': scaler.inverse_transform(train_data.reshape(-1, 1)).flatten(),
        'predicted_dates': test_dates,
        'predicted_prices': original_predictions
    }

    return results
