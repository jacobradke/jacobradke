import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def market_beta(ind,dep,n):
    # ind = the independant variable (market)
    # dep = the dependant var (stock)
    # n = the length of the of the window
    obs = len(ind)
    betas = np.full(obs, np.nan)
    alphas = np.full(obs, np.nan)
    for i in range((obs-n)):
        regressor = LinearRegression()
        regressor.fit(ind.to_numpy()[i:i+n+1].reshape(-1,1),dep.to_numpy()[i:i+n+1])
        betas[i+n] = regressor.coef_[0]
        alphas[i+n] = regressor.intercept_
    return (alphas, betas)