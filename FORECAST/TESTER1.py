from django.shortcuts import render
from .models import Stock
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


def home(request):
    context = {}
    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        try:
            df = get_stock_data(ticker, start_date, end_date)
        except Exception as e:
            context['error_message'] = str(e)
        else:
            for index, row in df.iterrows():
                Stock.objects.create(
                    ticker=ticker,
                    date=pd.to_datetime(index).date(),
                    close_price=row['close_price']
                )

            stocks = Stock.objects.filter(ticker=ticker)
            plot = create_plot(stocks)
            context['plot'] = plot

    return render(request, 'home.html', context)


def create_plot(stocks):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [s.date for s in stocks]
    y = [s.close_price for s in stocks]
    ax.plot(x, y)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Цена закрытия')
    ax.set_title(f'График цены закрытия {stocks.first().ticker}')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.urlsafe_b64encode(buf.getvalue()).decode('utf-8')

    return f'data:image/png;base64,{encoded_image}'


def get_stock_data(ticker, start_date, end_date):
    df = yf.download(tickers=ticker, start=start_date, end=end_date)[['Adj Close']]
    df.columns = ['close_price']
    df.index.names = ['date']
    return df