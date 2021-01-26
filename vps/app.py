import streamlit as st
import pandas as pd
from PIL import Image
####
import pandas as pd
import numpy as np
import requests
import pandas_datareader as pdr
from datetime import date, timedelta
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from currency_converter import CurrencyConverter

####Stock prediction######
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import yfinance as yf
plt.style.use("bmh")

####### APPLICATION ######
st.set_page_config(
	page_title="Trading Toolbox",
	page_icon="💵",
	initial_sidebar_state="expanded")


st.title("🔮 Gépi tanulás alapú tőzsdei eszközök")

image = Image.open("header.jpg")

st.sidebar.image(image, use_column_width=True)

#st.sidebar.header("User Input")

def get_input():
    #start_date = st.sidebar.text_input("Start Date", "2020-01-02")
    #end_date = st.sidebar.text_input("End Date", "2021-01-02")
    c = CurrencyConverter()
    value = st.sidebar.text_input("Befektetni kívánt összeg dollárban:", "1000")
	
    HUF = int(c.convert(value, 'USD', 'HUF'))
    st.sidebar.write("Összeg: ", HUF, 'Ft')
	
    if value == '1':
       st.sidebar.write("⚠️ Túl alacsony összeg")
    elif int(value) <= 0:
       st.sidebar.write("⚠️ Negatív összeg")

    return value


def get_csv():
    options =  values = ["Revolut", "NASDAQ", "NYSE"]
    default_ix = "Revolut"
    value = st.sidebar.selectbox("Válassz tőzsdét: ", options, key='1')

    if value == "Revolut":
       value = './data/revolut.csv'
       st.sidebar.write("📈 Részvények, amik a Revolut appon belül kereskedhetőek.")
    else:
       value = './data/revolut.csv'
       st.sidebar.write("Még nem működik... 🤡  A Revolut opció eredményét látod most.")
    return value

#Get the companies name
def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query=' + symbol + '&region=1&lang=en'
  result = requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol'] == symbol:
        return r['name']

####### LOGIC ######

def create_portfolio():
    today = str(date.today())

#Create the DataFrame and fill with historical data
    df = pd.read_csv(csv, header = 0)

    df = df.clip(lower=0.1)
    print("NEGATIVE: " + str(df.agg(lambda x: sum(x < 0)).sum()))

    df = df.iloc[:, 0:200]

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

    above_df = "Várható éves hozam: " +  str(round(ef.portfolio_performance(verbose=True)[0] * 100, 2)) + "%   |  " + "Volatilitás: " +  str(round(ef.portfolio_performance(verbose=True)[1] * 100, 2)) + "%   |  " + "Sharpe-ráta: " +  str(round(ef.portfolio_performance(verbose=True)[2], 3))

    st.write("Az első két oszlop a cég nevét és tőzsdei szimbólumát mutatja, a harmadik pedig azt, hogy hány darab részvényt kell belőle venned a kalkuláció szerint")
    st.success(str(above_df))
    st.write(portfolio_df, 'A végén marad: ' + str(round(leftover, 2)) + "$")
    st.write("Elemzéshez felhasznált részvények listája:")
    st.write(assets)

##############  STOCK PREDICTION ##############
today = str(date.today())
five_years_ago = str(date.today() - timedelta(3650))

def selected_company():
    tickers = pd.read_csv('comany_tickers.csv')['Symbols'].tolist()
    value = st.sidebar.selectbox('Válassz céget:',tickers)
    return value
    
def get_historical_data(selected_company):
    st.write("Historikus adatok")
    df = pdr.get_data_yahoo(selected_company, start=five_years_ago, end=today)
    st.write(df)
    
def plot_historical_data(selected_company):
    df = pdr.get_data_yahoo(selected_company, start=five_years_ago, end=today)
    st.write("Zárási ár grafikon")
    df_plot = df['Adj Close']
    st.line_chart(df_plot)
    
def predict_future_price(selected_company):
    st.write("Prediktált árfolyam")
    df = pdr.get_data_yahoo(selected_company, start=five_years_ago, end=today)
    all_data = df[["Close"]]
    #st.line_chart(all_data)
    df = df[["Close"]]
    future_days = 25
    df['Prediction'] = df[['Close']].shift(-future_days)
    X = np.array(df.drop(['Prediction'],1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]
    x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
    
    tree =  DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)
    kn = KNeighborsRegressor().fit(x_train, y_train)
    svr = SVR().fit(x_train, y_train)
    
    x_future = df.drop(['Prediction'],1)[-future_days:]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    
    tree_prediction = tree.predict(x_future)
    lr_prediction = lr.predict(x_future)
    kn_prediction = kn.predict(x_future)
    svr_prediction = svr.predict(x_future)
    
    valid = df[X.shape[0]:]
    valid = valid.drop(['Prediction'],1)
 
    valid['Prediction_tree'] = tree_prediction
    valid['Prediction_lr'] = lr_prediction
    valid['Prediction_kn'] = kn_prediction
    valid['Prediction_svr'] = svr_prediction
    st.write(valid)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("DecisionTreeRegressor alapú predikció")
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'].iloc[-90:])
    plt.plot(valid[['Close', 'Prediction_tree']])
    plt.legend(["Megelőző időszak", "Valódi ár", "Prediktált ár"])
    plt.ylabel("Ár ($)")
    st.pyplot()
    
    st.write("LinearRegression alapú predikció")
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'].iloc[-90:])
    plt.plot(valid[['Close', 'Prediction_lr']])
    plt.legend(["Megelőző időszak", "Valódi ár", "Prediktált ár"])
    plt.ylabel("Ár ($)")
    st.pyplot()

    st.write("KNeighborsRegressor alapú predikció")
    plt.figure(figsize=(20,10))
    plt.plot(df['Close'].iloc[-90:])
    plt.plot(valid[['Close', 'Prediction_kn']])
    plt.legend(["Megelőző időszak", "Valódi ár", "Prediktált ár"])
    plt.ylabel("Ár ($)")
    st.pyplot()
    
    st.write("SVR alapú predikció")
    plt.figure(figsize=(20,10))
    plt.plot(df['Close'].iloc[-90:])
    plt.plot(valid[['Close', 'Prediction_svr']])
    plt.legend(["Megelőző időszak", "Valódi ár", "Prediktált ár"])
    plt.ylabel("Ár ($)")
    st.pyplot()
	
#################### SIDEBAR ###################

st.sidebar.subheader("📈 Portfólió optimalizálása")
portfolio_val = int(get_input())
csv = str(get_csv())


if st.sidebar.button('🔍 Portfólió készítése!'):
    create_portfolio()

#----------

st.sidebar.subheader("🤖 Részvényár-előrejelzés")

selected_company = selected_company()

if st.sidebar.button('🔍 Árfolyam számítása!'):
    #get_historical_data(selected_company)
    #plot_historical_data(selected_company)
    predict_future_price(selected_company)

#----------

st.sidebar.subheader("📚 Fundamentális elemzés")

if st.sidebar.button('🔍 Kiválasztott cég elemzése!'):
    st.write("HAMAROSAN...")

if st.sidebar.button('🔍 Összesített kimutatások!'):
    st.write("HAMAROSAN...")

#----------

st.sidebar.subheader("🎢 Piaci hangulatelemzés")

if st.sidebar.button('🔍 Elemzés!'):
    st.write("HAMAROSAN...")
