import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import statsmodels.api as sm
end = datetime.datetime.today()

def mean_variance(ticker_lst, num_ports, start, end):
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
    num_portfolios = num_ports
    individual_rets = df.resample("Y").last().pct_change().mean()
    for port in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        port_weights.append(weights)
        returns = np.dot(weights, individual_rets) *100
        port_returns.append(returns)
        var = cov.mul(weights,axis = 0).mul(weights,axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd*np.sqrt(252)
        port_vol.append(ann_sd)
    data = {"Returns": port_returns, "Volatility": port_vol}
    for counter, symbol in enumerate(df.columns.to_list()):
        data[symbol+" Weight"]=[w[counter] for w in port_weights]
    portfolios = pd.DataFrame(data)
    pct_change["Market"] = web.DataReader("^GSPC", "yahoo", start = start, end = end)["Adj Close"].pct_change()*100
    pct_change = pct_change.dropna()
    beta = {}
    for key in pct_change:
        Y = pct_change[key]
        X = pct_change["Market"]
        model = sm.OLS(Y,X)
        results = model.fit()
        beta[key + " Beta"] = results.params
    beta = pd.DataFrame(beta)
    beta = beta.drop("Market Beta", axis = 1).T
    ports = portfolios.drop(["Returns", "Volatility"], axis = 1).T
    port_beta_dct = {}
    for port in ports:
        port_beta_dct[port] = ports[port].values*beta["Market"].values
    port_beta_df = pd.DataFrame(port_beta_dct)
    port_beta_df.loc['Portfolio Beta',:] = port_beta_df.sum(axis=0)
    port_beta_df = port_beta_df.T
    portfolios["Portfolio Beta"] = port_beta_df["Portfolio Beta"]
    col_name = "Portfolio Beta"
    third_col = portfolios.pop(col_name)
    portfolios.insert(2, col_name, third_col)
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

def optimal_sharpe_portfolio(portfolios, rf = 0.02):
    optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
    sharpe = pd.DataFrame(optimal_risky_port).T
    return sharpe
    
def optimal_treynor_ratio(portfolios, rf = 0.02):
    optimal_treynor_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Portfolio Beta"]).idxmax()]
    optimal_treynor_port = pd.DataFrame(optimal_treynor_port).T
    return optimal_treynor_port