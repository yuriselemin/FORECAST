# import investpy
#
# df = investpy.get_stock_historical_data(stock='AAPL',
#                                         country='United States',
#                                         from_date='01/01/2010',
#                                         to_date='01/01/2020')
# print(df.head())


import yfinance as yf

ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2021-12-31')
print (data)
