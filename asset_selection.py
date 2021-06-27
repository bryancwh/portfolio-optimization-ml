"""
We want assets from a diversified range of sectors
"""
import pandas as pd
import numpy as np
import requests


def build_dataset(n=50):
    df = pd.read_csv('data/constituents-financials_csv.csv')

    sp = df.sort_values(by='Market Cap', ascending=False).head(n)

    tickers = sp.Symbol.to_list()

    API_KEY = open('alphavantage_api_keys.txt').read()

    price_data = []
    for ticker in tickers:
        r = requests.get(
            f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}').json()

        df = pd.DataFrame(r['Time Series (Daily)'], dtype=float).transpose()

        df = df.reindex(index=df.index[::-1])

        df.index = pd.to_datetime(df.index)
        df = df[((df.index >= pd.to_datetime('2010-01-01')) &
                 (df.index <= pd.to_datetime('2020-12-31')))]

        price_data.append(df[['5. adjusted close']])

    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns = tickers

    return df_stocks


"""
Using autoencoders to come up with a good representation of the market:
- simple and powerful 
- learns the latenet relationships within the data
input data -> latent features -> recreated data

Universe of stocks: 500 stocks (from S&P 500)
Time series: last 10 years of price returns

x-axis: stocks, y-axis: time series
find compressed list of stocks that move the market
1. Recreate data
2. Find RMSE between original and recreated matrix
3. Top 50 with least and top 50 with most RMSE

x-axis: time series, y-axis: stocks
find compressed timeframe
1. Recreate data
2. Find RMSE between original and recreated matrix
3. 
"""
