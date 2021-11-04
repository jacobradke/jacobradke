import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy
import datetime
end = datetime.datetime.today()

def RSIcalc(ticker, start, end):
    df = web.DataReader(ticker, 
                        "yahoo",
                        start = start, 
                        end = end)
    df["MA200"] = df["Adj Close"].rolling(window=200).mean()
    df["Price Change"] = df["Adj Close"].pct_change()
    df["Upmove"] = df["Price Change"].apply(lambda x: x if x > 0 else 0)
    df["Downmove"] = df["Price Change"].apply(lambda x: abs(x) if x < 0 else 0)
    df["Ave Up"] = df["Upmove"].ewm(span = 19).mean()
    df["Ave Down"] = df["Downmove"].ewm(span = 19).mean()
    df = df.dropna()
    df["RS"] = df["Ave Up"]/df["Ave Down"]
    df["RSI"] = df["RS"].apply(lambda x: 100-(100/(x+1)))
    df.loc[(df["RSI"] < 30), "Buy"] = "Yes"
    df.loc[(df["RSI"] > 30), "Buy"] = "No"
    df.loc[(df["RSI"] > 70), "Sell"] = "Yes"
    df.loc[(df["RSI"] < 70), "Sell"] = "No"
    cols_to_remove = ["High", "Low", "Open", "Close", "Volume"]
    df = df.drop(cols_to_remove, axis = 1)
    return df

def getSignals(df):
    Buying_dates = []
    Selling_dates = []
    
    for i in range(len(df)-11):
        if "Yes" in df["Buy"].iloc[i]:
            Buying_dates.append(df.iloc[i+1].name)
            for j in range(1,11): 
                if df["RSI"].iloc[i+j] > 40:
                    Selling_dates.append(df.iloc[i+j+1].name)
                    break
                elif j == 10:
                    Selling_dates.append(df.iloc[i+j+1].name)
    
    return Buying_dates,Selling_dates