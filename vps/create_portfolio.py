import pandas as pd
import numpy as np
import requests
import pandas_datareader as web
from datetime import date, timedelta

today = str(date.today())

#Create the DataFrame and fill with historical data
df = pd.read_csv('stocks-' + today + '.csv', header = 0)
print(df)

#df = df.iloc[:, 1500:1972]

assets = df.columns

#Optimize the portfolio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#Calculate the expected annualized returns and the annualized...
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize fot the maximal Sharpe ratio
ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(cleaned_weights)

ef.portfolio_performance(verbose=True)

#Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

portfolio_val = 5000
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds Rreaming $', leftover)

#Get the companies names
def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query=' + symbol + '&region=1&lang=en'
  result = requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol'] == symbol:
      return r['name']

company_name = []
discrete_allocation_list = []

for symbol in allocation:
  company_name.append(get_company_name(symbol))
  discrete_allocation_list.append(allocation.get(symbol))

portfolio_df = pd.DataFrame(columns=['Company_name', 'Company_Ticker', 'Discrete_val_'+ str(portfolio_val)])

portfolio_df['Company_name'] = company_name
portfolio_df['Company_Ticker'] = allocation
portfolio_df['Discrete_val_'+ str(portfolio_val)] = discrete_allocation_list

print(portfolio_df)
