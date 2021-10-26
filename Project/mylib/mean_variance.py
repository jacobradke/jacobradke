import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import datetime 
end = datetime.datetime.today()
def mean_variance(ticker_lst, start, end):
    ticker_dict = {}
    for ticker in ticker_lst:
        ticker_dict[ticker] = web.DataReader(ticker, "yahoo", start, end)["Adj Close"]
    df = pd.DataFrame(ticker_dict)
    pct_change = df.pct_change()*100
    cov = pct_change.cov()
    port_returns = []
    port_vol = []
    port_weights = []
    num_assets = len(df.columns)
    num_portfolios = 10000
    individual_rets = df.resample('Y').last().pct_change().mean()
    for port in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        port_weights.append(weights)
        returns = np.dot(weights, individual_rets)
        port_returns.append(returns)
        var = cov.mul(weights,axis = 0).mul(weights,axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd*np.sqrt(252)
        port_vol.append(ann_sd)
    data = {"Returns": port_returns, "Volatility": port_vol}
    for counter, symbol in enumerate(df.columns.to_list()):
        data[symbol+" Weight"]=[w[counter] for w in port_weights]
    portfolios = pd.DataFrame(data)
    return portfolios

def efficient_frontier(portfolios, s=15, color = "k",alpha = 0.5, marker = 'o', figsize = (24,18), rf = 0.02):
    portfolios.plot.scatter(x="Volatility",
                            y="Returns",
                            marker=marker,
                            color = color,
                            s=s,
                            alpha = alpha,
                            grid = True,
                            figsize = figsize)
    optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='forestgreen', marker='*', s=500) 
    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Expected Returns");

def optimal_portfolio(portfolios, rf = 0.02):
    optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
    sharpe = pd.DataFrame(optimal_risky_port).T
    return sharpe
    
