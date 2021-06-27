import pandas as pd
import numpy as np
import requests

def gen_df(tickers, date_from, date_to):
    '''
    Function to generate dataframe from list of tickers
    '''

    # Take in API Key
    API_KEY = open('alphavantage_api_keys.txt').read()

    price_data = []

    # Loop through all tickers
    for ticker in tickers:

        # Get JSON output
        r = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}').json()

        # Create df
        df = pd.DataFrame(r['Time Series (Daily)'], dtype=float).transpose()
        df = df.reindex(index=df.index[::-1])
        df.index = pd.to_datetime(df.index)

        # Obtain subset of dataset
        df = df[ ( (df.index >= date_from) & (df.index <= date_to) ) ]

        # Append dataframe to list
        price_data.append(df[['5. adjusted close']])
    
    # Concatenate price data from all tickers
    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns = tickers
    print(df_stocks.tail())
    return df_stocks

# # Test
# tickers = ['AC.TO','ZSP.TO','XFN.TO','HEU.TO','XIT.TO']
# date_from = pd.to_datetime('2013-01-01')
# date_to = pd.to_datetime('2020-06-13')
# gen_df(tickers, date_from, date_to)