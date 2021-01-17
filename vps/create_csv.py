import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import date, timedelta

today = str(date.today())
five_years_ago = str(date.today() - timedelta(1825))

#Create the DataFrame and fill with historical data
tickers = pd.read_csv('nasdaq.csv')['Symbol']

df = pd.DataFrame(columns=tickers)

for symbol in tickers:
    df[symbol] = web.DataReader(symbol, "yahoo", five_years_ago, today)["Adj Close"]

#Drop the rows where at least one element is missing.
df = df.dropna()

df.to_csv('stocks' + today + '.csv')
