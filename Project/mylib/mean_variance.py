import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import statsmodels.api as sm
import quandl as ql
end = datetime.datetime.today()
start = datetime.datetime(2011,1,1)

def mean_variance(ticker_lst, num_ports, start, end, benchmark, rf):
    ticker_dict = {}
    for ticker in ticker_lst:
        ticker_dict[ticker] = web.DataReader(ticker, "yahoo", start, end)["Adj Close"]
    df = pd.DataFrame(ticker_dict)
    pct_change = df.pct_change().fillna(0)*100
    market = web.DataReader(benchmark, "yahoo", start = start, end = end)["Adj Close"]
    market_change = pd.DataFrame(market.pct_change()*100).fillna(0)
    def information_ratio(returns, benchmark, days = 1):
        active_returns = returns - benchmark
        tracking_error = active_returns.std() * np.sqrt(days)
        info_ratio = active_returns.mean() / tracking_error
        return info_ratio
    individual_info_ratios = {}
    for stock in pct_change:
        individual_info_ratios[stock] = information_ratio(returns = pct_change[stock], benchmark = market_change["Adj Close"])*10
    individual_info_ratios = pd.DataFrame(individual_info_ratios, index = [0]).T
    cov = pct_change.cov()
    port_returns = []
    port_info = []
    port_vol = []
    port_weights = []
    num_assets = len(df.columns)
    num_portfolios = num_ports
    individual_rets = df.resample("A").last().pct_change().mean()
    for port in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        port_weights.append(weights)
        returns = np.dot(weights, individual_rets) *100
        information_ratio = np.dot(weights, individual_info_ratios)
        port_info.append(float(information_ratio))
        port_returns.append(returns)
        var = cov.mul(weights,axis = 0).mul(weights,axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd*np.sqrt(252)
        port_vol.append(ann_sd)
    data = {"Returns": port_returns, "Volatility": port_vol, "Information Ratio": port_info}
    for counter, symbol in enumerate(df.columns.to_list()):
        data[symbol+" Weight"]=[w[counter] for w in port_weights]
    portfolios = pd.DataFrame(data)
    pct_change["Market"] = web.DataReader(benchmark, "yahoo", start = start, end = end)["Adj Close"].pct_change()*100
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
    ports = portfolios.drop(["Returns", "Volatility", "Information Ratio"], axis = 1).T
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
    market = web.DataReader(benchmark, "yahoo", start = start, end = end)["Adj Close"]
    market_mean = market.resample("A").last().pct_change().mean()*100
    jensens_alpha = portfolios["Returns"]-(rf + portfolios["Portfolio Beta"]*(market_mean-rf))
    portfolios["Jensen's Alpha"] = jensens_alpha
    jen_col = "Jensen's Alpha"
    fourth_col = portfolios.pop(jen_col)
    portfolios.insert(3, jen_col, fourth_col)
    
    portfolios["Sharpe Ratio"] = (portfolios["Returns"]-rf)/portfolios["Volatility"]
    shr_col = "Sharpe Ratio"
    col = portfolios.pop(shr_col)
    portfolios.insert(2, shr_col, col)
    
    return portfolios

def efficient_frontier(portfolios, benchmark, s=15, color = "k",alpha = 0.5, marker = 'o', figsize = (24,18), rf = 2, start = start, end = end):
    portfolios.plot.scatter(x="Volatility",
                            y="Returns",
                            marker=marker,
                            color = color,
                            s=s,
                            alpha = alpha,
                            grid = True,
                            figsize = figsize)
    optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
    optimal_treynor_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Portfolio Beta"]).idxmax()]
    plt.scatter(optimal_risky_port[1], 
                optimal_risky_port[0], 
                color='forestgreen', 
                marker='*', 
                s=500)
    plt.scatter(optimal_treynor_port[1], 
                optimal_treynor_port[0], 
                color='orange', 
                marker='*', 
                s=500)
    market = web.DataReader(benchmark, "yahoo", start = start, end = end)["Adj Close"]
    market = pd.DataFrame(market)
    market_mean = market.resample("Y").last().pct_change().mean()*100
    optimal_jensens_alpha = portfolios.iloc[(portfolios["Jensen's Alpha"]).idxmax()]
    plt.scatter(optimal_jensens_alpha[1], 
                optimal_jensens_alpha[0], 
                color = "purple", 
                marker = "*",
                s = 500)
    optimal_info_ratio = portfolios.iloc[(portfolios["Information Ratio"]).idxmax()]
    plt.scatter(optimal_info_ratio[1],
                optimal_info_ratio[0], 
                color = "black", 
                marker = "*", 
                s = 500)
    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Expected Returns");

def effects_on_volatility(portfolios, figsize, color, alpha, s):
    for key in portfolios: 
        x = portfolios[key]
        y = portfolios["Volatility"]
        m, b = np.polyfit(x, y, 1)
        portfolios.plot.scatter(x=key,
                                y="Volatility",
                                marker="o",
                                color = color,
                                s=s,
                                alpha = alpha,
                                grid = True,
                                figsize = figsize)
        plt.plot(x, 
                 m*x + b, 
                 linewidth = 5)
        plt.title(key+" Effect on Portfolio Volatility")
        print(key+": ", m)
        plt.show()
        plt.close();

def effects_on_returns(portfolios, figsize, color, alpha, s):
    for key in portfolios: 
        x = portfolios[key]
        y = portfolios["Returns"]
        m, b = np.polyfit(x, y, 1)
        portfolios.plot.scatter(x = key,
                                y ="Returns",
                                marker = "o",
                                color = color,
                                s = s,
                                alpha = alpha,
                                grid = True,
                                figsize = figsize)
        plt.plot(x, 
                 m*x + b, 
                 linewidth = 5)
        plt.title(key+" Effect on Portfolio Returns")
        print(key+": ", m)
        plt.show()
        plt.close();

def effects_on_sharpe_ratio(portfolios, figsize, color, alpha, s):
    for key in portfolios: 
        x = portfolios[key]
        y = portfolios["Sharpe Ratio"]
        m, b = np.polyfit(x, y, 1)
        portfolios.plot.scatter(x = key,
                                y ="Sharpe Ratio",
                                marker = "o",
                                color = color,
                                s = s,
                                alpha = alpha,
                                grid = True,
                                figsize = figsize)
        plt.plot(x, 
                 m*x + b, 
                 linewidth = 5)
        plt.title(key+" Effect on Portfolio Sharpe Ratio")
        print(key+": ", m)
        plt.show()
        plt.close();        

def effects_on_jensens_alpha(portfolios, figsize, color, alpha, s):
    for key in portfolios: 
        x = portfolios[key]
        y = portfolios["Jensen's Alpha"]
        m, b = np.polyfit(x, y, 1)
        portfolios.plot.scatter(x = key,
                                y ="Jensen's Alpha",
                                marker = "o",
                                color = color,
                                s = s,
                                alpha = alpha,
                                grid = True,
                                figsize = figsize)
        plt.plot(x, 
                 m*x + b, 
                 linewidth = 5)
        plt.title(key+" Effect on Portfolio Jensen's Alpha")
        print(key+": ", m)
        plt.show()
        plt.close();          
        
def effects_on_info_ratio(portfolios, figsize, color, alpha, s):
    for key in portfolios: 
        x = portfolios[key]
        y = portfolios["Information Ratio"]
        m, b = np.polyfit(x, y, 1)
        portfolios.plot.scatter(x = key,
                                y ="Information Ratio",
                                marker = "o",
                                color = color,
                                s = s,
                                alpha = alpha,
                                grid = True,
                                figsize = figsize)
        plt.plot(x, 
                 m*x + b, 
                 linewidth = 5)
        plt.title(key+" Effect on Portfolio Information Ratio")
        print(key+": ", m)
        plt.show()
        plt.close();          
        
def optimal_sharpe_portfolio(portfolios, rf = 0.02):
    optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]
    sharpe = pd.DataFrame(optimal_risky_port).T
    return sharpe
    
def optimal_treynor_ratio(portfolios, rf = 0.02):
    optimal_treynor_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Portfolio Beta"]).idxmax()]
    optimal_treynor_port = pd.DataFrame(optimal_treynor_port).T
    return optimal_treynor_port

def optimal_jensens_alpha(portfolios):
    optimal = portfolios.iloc[(portfolios["Jensen's Alpha"]).idxmax()]
    optimal = pd.DataFrame(optimal).T
    return optimal

def optimal_information_ratio(portfolios):
    optimal = portfolios.iloc[(portfolios["Information Ratio"]).idxmax()]
    optimal = pd.DataFrame(optimal).T
    return optimal

def get_yield_curve(figsize):
    data = ql.get("USTREASURY/YIELD")
    today = data.iloc[-1,:]
    month_ago = data.iloc[-21,:]
    three_month_ago = data.iloc[-63,:]
    one_year_ago = data.iloc[-252,:]
    df = pd.concat([today, 
                    month_ago, 
                    three_month_ago, 
                    one_year_ago], 
                   axis=1)
    df.columns = ['Today', 
                  'Month Ago', 
                  '3-Months Ago', 
                  "Year Ago"]

    df.plot(style={'Today': 'ro-', 
                   'Month Ago': 'yo--', 
                   '3-Months Ago': 'bx--', 
                   'Year Ago': 'go--'},
            title='Treasury Yield Curve, %', 
            figsize = figsize, 
            grid = True, 
            linewidth = 3);