# import yfinance as yf
# import pandas as pd
# import os
# from tempfile import NamedTemporaryFile
#
# def get_stock_data(ticker, start_date, end_date):
#     df = yf.download(ticker, start_date, end_date)[['Adj Close']]
#
#     with NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
#         df.to_csv(temp_file.name, index=True)
#
#         temp_file.seek(0)
#         df_from_csv = pd.read_csv(temp_file.name, parse_dates=[0], index_col=0)
#
#     # Delete the temporary file after reading it
#     temp_file.close()
#     os.remove(temp_file.name)
#
#     return df_from_csv
#
# df_from_csv = get_stock_data('AAPL', '2020-01-01', '2021-12-31').to_string()
# print(df_from_csv)
#
