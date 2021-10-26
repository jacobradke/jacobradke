import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_monte_carlo(mean, sigma, num_sims, sim_dict, index):
    for i in range(num_sims):
        sim_dict[i] = {}
        for ix in index:
            sim_dict[i][ix] = random.normalvariate(mean, sigma)
            
def plot_monte_carlo_sim(sim_data, observed_data = None, title = None, logy = True):
    sim_data["mean"] = sim_data.mean(axis=1)    
    index = sim_data.index
    fig, ax = plt.subplots(figsize = (40, 24))
    sim_data.drop(["mean"], inplace = False, axis = 1).plot.line(legend=False, 
                                                                 marker  =".", 
                                                                 markersize = .1, 
                                                                 color = "k", 
                                                                 alpha = .05, 
                                                                 logy = logy, 
                                                                 ax = ax)
    if observed_data is not None:
        observed_data.plot.line(legend = False, 
                                color = "C2", 
                                linewidth = 5, 
                                logy = logy, 
                                ax = ax)
        # find x coordinate of lowest value observed
        obs_text_x = observed_data[observed_data == observed_data.min()].index
        plt.text(obs_text_x, observed_data.loc[obs_text_x] * .7,
                 "Observed", fontsize = 70, color = "C2")
    sim_data["mean"].plot.line(legend = False, 
                               color = "C3", 
                               linewidth = 5, 
                               logy = logy, 
                               ax = ax)
    plt.text(index[-400], sim_data["mean"].iloc[-400] * 1.5, "Exp\nVal",
            fontsize = 70, color = "C3")
    plt.title(title, fontsize = 50)
    plt.show()
    plt.close()

