import os
import random
import simfin as sf
import numpy as np
import pandas as pd
import urllib.request
import pycountry as pc
import datetime as dt
import sklearn.model_selection
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import openpyxl
from sklearn.linear_model import Ridge
import sklearn as sk
import matplotlib.pyplot as plt
import yfinance as yf
import scipy
import warnings

os.chdir('C:/Users/Nelson/PycharmProjects/pythonProject1')
if not os.path.exists('price_lists'):
    os.mkdir('price_lists')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# SimFin setup
sf.set_api_key('xTa2BSQTA6ralfGJu6dqA2xVMpKxhZTe')
sf.set_data_dir('simfin_data')
sf.load(dataset='income', variant='quarterly', market='us', refresh_days=31)
sf.load(dataset='balance', variant='quarterly', market='us', refresh_days=31)
sf.load(dataset='cashflow', variant='quarterly', market='us', refresh_days=31)
sf.load(dataset='companies', market='us')
df_inc = pd.read_csv('simfin_data/us-income-quarterly.csv', sep=';')
df_bs = pd.read_csv('simfin_data/us-balance-quarterly.csv', sep=';')
df_cs = pd.read_csv('simfin_data/us-cashflow-quarterly.csv', sep=';')
df_ind = pd.read_csv('simfin_data/us-companies.csv', sep=';')
data = pd.merge(df_bs, df_cs, on=['Ticker', 'Report Date'], how='inner')
data = data.merge(df_inc, on=['Ticker', 'Report Date'], how='inner')
data = data.merge(df_ind, on=['Ticker'], how='inner')

# Price concatenation function
prices = pd.DataFrame(columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
def concat_price(file, ticker, df_price):
    p = pd.read_csv(file, header=0)
    p = p.assign(Ticker=ticker)
    p['Date'] = pd.to_datetime(p['Date'])
    p.set_index('Date', inplace=True)
    p = p.resample('D').asfreq()
    p = p.interpolate(method='time')
    p['Close'] = p['Close'].interpolate(method='pad', limit=5)
    p['Ticker'] = p['Ticker'].interpolate(method='pad', limit=5)
    p.index = p.index.strftime('%Y-%m-%d')
    p = p.reset_index(drop=False)
    price_df = pd.concat([df_price, p])
    return price_df

# Grab stock prices
companies = sorted(set(data['Ticker']))
latest = max(data['Report Date'])
if not os.path.exists(f'price_lists/prices_{latest}.csv'):
    for ticker in companies:
        ticker_data = data[data['Ticker'] == ticker]['Report Date']
        if np.size(ticker_data) > 1:
            start = int((min(pd.to_datetime(ticker_data)) - pd.DateOffset(days=100)).timestamp())
            end = int((max(pd.to_datetime(ticker_data)) + pd.DateOffset(days=100)).timestamp())
            url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true'
            file = f'price_lists/{ticker}_{start}_{end}.csv'
            if not os.path.exists(file):
                try:
                    urllib.request.urlretrieve(url, file)
                    prices = concat_price(file, ticker, prices)
                    print(f'{ticker} downloaded.')
                except:
                    print(f'{ticker} failed to download.')
            else:
                prices = concat_price(file, ticker, prices)
                print(f'{ticker} already downloaded.')
        else:
            print(f'{ticker} has insufficient data.')
    prices.to_csv(f'price_lists/prices_{latest}.csv')
else:
    prices = pd.read_csv(f'price_lists/prices_{latest}.csv')

# S&P 500
start = int(dt.datetime.strptime(min(data['Report Date']), "%Y-%m-%d").timestamp())
end = int(dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
file = f'price_lists/^GSPC_{start}_{end}.csv'
if not os.path.exists(file):
    urllib.request.urlretrieve(f'https://query1.finance.yahoo.com/v7/finance/download/^GSPC?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true', file)
snp = pd.read_csv(file, header=0)
snp['Date'] = pd.to_datetime(snp['Date'])
snp.set_index('Date', inplace=True)
snp = snp.resample('D').asfreq()
snp = snp.interpolate(method='time')
snp.index = snp.index.strftime('%Y-%m-%d')
snp = snp.reset_index(drop=False)
snp = snp.rename(columns={'Close': 'S&P500', 'Date': 'Report Date'})
prices = prices.rename(columns={'Date': 'Report Date'})
prices = pd.merge(prices, snp, on=['Report Date'], how='outer')

# Merging Prices
lagged_days = 50
data['Report Date 2'] = pd.to_datetime(data['Report Date']) + pd.DateOffset(days=lagged_days)
data['Report Date 2'] = data['Report Date 2'].dt.strftime('%Y-%m-%d')
data['Report Date 3'] = pd.to_datetime(data['Report Date']) - pd.DateOffset(days=lagged_days)
data['Report Date 3'] = data['Report Date 3'].dt.strftime('%Y-%m-%d')
data['Report Date YM'] = pd.to_datetime(data['Report Date'])
data['Report Date YM'] = data['Report Date YM'].dt.strftime('%Y-%m')
data = data.merge(prices[['Ticker', 'Report Date', 'Close', 'S&P500']], on=['Ticker', 'Report Date'], how='inner')
data = data.merge(prices[['Ticker', 'Report Date', 'Close', 'S&P500']], left_on=['Ticker', 'Report Date 2'], right_on=['Ticker', 'Report Date'], suffixes=('', ' Leaded'), how='inner')
data = data.merge(prices[['Ticker', 'Report Date', 'Close', 'S&P500']], left_on=['Ticker', 'Report Date 3'], right_on=['Ticker', 'Report Date'], suffixes=('', ' Lagged'), how='inner')

# Calculating Fundamentals in New DataFrame
fields = ['Ticker', 'Report Date', 'Report Date Leaded', 'Report Date Lagged', 'Report Date YM', 'Close', 'Close Leaded', 'Close Lagged', 'S&P500', 'S&P500 Leaded', 'S&P500 Lagged', 'IndustryId']
data.replace(0, np.nan, inplace=True)
reg = pd.DataFrame()
reg[fields] = data[fields]
reg['Net Income'] = 4*data['Net Income']
reg['Retained Earnings'] = 4*data['Retained Earnings']
reg['Dividend Yield'] = (-data['Dividends Paid'] / data['Shares (Diluted)']) / data['Close']
reg['Gross Margin MRQ'] = 100*data['Gross Profit'] / data['Revenue']
reg['Net Profit Margin MRQ'] = 100*data['Net Income'] / data['Revenue']
reg['Operating Margin MRQ'] = 100*data['Operating Income (Loss)'] / data['Revenue']
reg['Rev Ratio'] = data['Revenue'] / data['Cost of Revenue']
reg['PB Ratio'] = data['Close'] * data['Shares (Diluted)'] / data['Total Equity']
reg['PCF Ratio'] = .25*data['Close'] * data['Shares (Diluted)'] / data['Net Cash from Operating Activities']
reg['PE Ratio'] = .25*data['Close'] * data['Shares (Diluted)'] / data['Net Income']
reg['PR Ratio'] = .25*data['Close'] * data['Shares (Diluted)'] / data['Revenue']
reg['Quick Ratio'] = (data['Total Current Assets'] - data['Inventories']) / data['Total Current Liabilities']
reg['Return on Assets'] = 4*100*data['Net Income'] / data['Total Assets']
reg['Asset Turnover'] = 100*data['Revenue'] / data['Total Assets']
reg['Equity Multiplier'] = data['Total Assets'] / data['Total Equity']
reg['Return on Equity'] = 4*100*data['Net Income'] / data['Total Equity']
reg['Total Debt to Equity'] = 100*(data['Short Term Debt'] + data['Long Term Debt']) / data['Total Equity']
reg['Dupont'] = reg['Net Profit Margin MRQ'] * reg['Asset Turnover'] * reg['Equity Multiplier']
reg = reg.dropna()

def wipe_outliers(df, column, threshold=3):
    z_scores = scipy.stats.zscore(df[column])
    return df[abs(z_scores) <= threshold]

# Industry growth rates + Dummies
exo = list(reg.columns[13:])
out = np.array([])
for var in exo:
    out = np.array([])
    for ticker in sorted(set(reg['Ticker'])):
        temp = reg[reg['Ticker'] == ticker]
        out = np.concatenate((out, pd.to_numeric(temp[var]).pct_change(periods=1).values))
    reg[f'{var} % change'] = out
reg = reg.dropna()

for time in set(reg['Report Date YM']):
    for industry in set(reg['IndustryId']):
        temp = data[(data['IndustryId'] == industry) & (data['Report Date YM'] == time)]
        reg.loc[(reg['IndustryId'] == industry) & (reg['Report Date YM'] == time), 'Ind Growth'] = np.average(np.log(temp['Close Lagged']) - np.log(temp['Close']))

exo = reg.columns[14:]
for var in exo:
    wipe_outliers(reg, var)

dummies = pd.get_dummies(reg['IndustryId'])
reg = pd.concat([reg, dummies], axis=1)
reg['Close Change'] = np.log(reg['Close'])-np.log(reg['Close Leaded'])

reg.to_excel('Regression Data.xlsx', index=False)

# Interactions + Standardization + X,Y
scaler = sk.preprocessing.StandardScaler()
poly = sk.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
#exo_matrix = poly.fit_transform(reg[exo].values)
#exo_matrix = scaler.fit_transform(reg[exo].values)
X = sm.add_constant(reg[exo].values)
Y = reg['Close Change']

# OLS Regression
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.25, random_state=random.randint(0, 1000))
model = sm.OLS(Y_train, X_train).fit()
Y_pred = model.predict(X_test)
R2 = sklearn.metrics.r2_score(Y_test, Y_pred)
print(model.summary())
print(R2)

print('ass.')