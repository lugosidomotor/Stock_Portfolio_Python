import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import date, timedelta
import yfinance as yf

today = str(date.today())
five_years_ago = str(date.today() - timedelta(1825))
yf.pdr_override()

#Get list of companies in .csv from https://www.nasdaq.com/market-activity/stocks/screener
#Create the DataFrame and fill with historical data
tickers = pd.read_csv('nasdaq.csv')['Symbol']

df = pd.DataFrame(columns=tickers)

for symbol in tickers:
    if "." not in str(symbol):
        try:
            df[symbol] = pdr.get_data_yahoo(symbol, start=five_years_ago, end=today)["Adj Close"]
            print("Reading: " + symbol)
        except:
            print("ERROR TO GET DATA")

# Dropping the columns having NaN/NaT values 
df = df.dropna(axis=1)

df = df.clip(lower=1)
print("NEGATIVE: " + str(df.agg(lambda x: sum(x < 0)).sum()))

print(df)

df.to_csv('stocks-' + today + '.csv', index=False)
