import streamlit as st
import pandas as pd
from PIL import Image
####
import pandas as pd
import numpy as np
import requests
import pandas_datareader as web
from datetime import date, timedelta
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt

####### APPLICATION ######
st.set_page_config(
	page_title="Revolut Portólió",
	page_icon="💵",
	initial_sidebar_state="expanded")


st.title("💰🤑💵 Portfólió optimalizáció 💵🤑💰")
#st.subheader('Kizárólag Revoluton kereskedett részvényekből')

image = Image.open("header.jpg")

st.image(image, use_column_width=True)

#st.sidebar.header("User Input")

def get_input():
    #start_date = st.sidebar.text_input("Start Date", "2020-01-02")
    #end_date = st.sidebar.text_input("End Date", "2021-01-02")
    value = st.sidebar.text_input("Befektetni kívánt összeg:", "1000")
    return value

portfolio_val = int(get_input())


####### LOGIC ######

today = str(date.today())

#Create the DataFrame and fill with historical data
df = pd.read_csv('stocks-' + today + 'final.csv', header = 0)

df = df.clip(lower=0.1)
print("NEGATIVE: " + str(df.agg(lambda x: sum(x < 0)).sum()))

df = df.iloc[:, 0:10]

assets = df.columns

#Calculate the expected annualized returns and the annualized ...
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize fot the maximal Sharpe ratio
ef = EfficientFrontier(mu,S)

weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)


#Get the discrete allocation of each share per stock
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio()
#print('Discrete allocation: ', allocation)
#print('Funds Rreaming $', leftover)

#Get the companies name
def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query=' + symbol + '&region=1&lang=en'
  result = requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol'] == symbol:
      return r['name']

symbols = []
discrete_allocation_list = []
company_name = []
for symbol in allocation:
  company_name.append(get_company_name(symbol))
  discrete_allocation_list.append(allocation.get(symbol))
  #symbols.append(allocation.get(symbol))
  

portfolio_df = pd.DataFrame(columns=['Cég', 'Szimbólum', 'Részvények_száma'])

portfolio_df['Cég'] = company_name
portfolio_df['Szimbólum'] = allocation
portfolio_df['Részvények_száma'] = discrete_allocation_list


above_df = "Várható éves hozam: " +  str(round(ef.portfolio_performance(verbose=True)[0] * 100, 2)) + "%   |  " + "Volatiritás: " +  str(round(ef.portfolio_performance(verbose=True)[1] * 100, 2)) + "%   |  " + "Sharpe-ráta: " +  str(round(ef.portfolio_performance(verbose=True)[2], 3))





st.write(above_df , portfolio_df, 'A végén marad: ' + str(round(leftover, 2)) + "$")
st.write("Az elemzéshez használt részvények listája:")
st.write(assets)











