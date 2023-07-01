#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1">Introduction</a></span></li><li><span><a href="#VaR---Value-at-Risk" data-toc-modified-id="VaR---Value-at-Risk-2">VaR - Value at Risk</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3">Imports</a></span></li><li><span><a href="#Loading-the-data" data-toc-modified-id="Loading-the-data-4">Loading the data</a></span></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-5">Modelling</a></span><ul class="toc-item"><li><span><a href="#Variance-Covariance-Method" data-toc-modified-id="Variance-Covariance-Method-5.1">Variance-Covariance Method</a></span></li><li><span><a href="#Historical-Simulation-VaR" data-toc-modified-id="Historical-Simulation-VaR-5.2">Historical Simulation VaR</a></span></li><li><span><a href="#Monte-Carlo-VaR" data-toc-modified-id="Monte-Carlo-VaR-5.3">Monte Carlo VaR</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** VaR - Value at Risk modelling methods
# 
# </font>
# </div>

# # VaR - Value at Risk
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - VaR addresses one of the most common questions an investor has: what is the maximum expected loss of my investment? Suppose that a daily VaR of an investment is 1 million with 95% confidence interval. This would read as there being a 5% chance that an investor might incur a loss greater than $1 million in a day.
#     
# - The VaR  has two important characteristics: ot provides a common consistent measure of risk across different positions and it takes account of the correlations between different risk factors.
# 
# - In summary:
#     - VaR needs an estimation of the probability of loss.
#     - VaR concentrates on the potential losses not on realised losses
# 
# - VaR has three key ingredients:
#     1. Standard deviation that defines the level of loss
#     2. Fixed time horizon over which risk is assessed
#     3. Confidence interval.
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from scipy.stats import norm
import requests
from io import StringIO
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# # Loading the data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


def getDailyData(symbol, start, end):
    pd = pdr.get_data_yahoo(symbol, start, end)
    return pd


# In[3]:


getDailyData(["IBM", "MSFT", "INTC"], '2020-01-01', '2020-12-21')


# In[4]:


stocks = getDailyData(["IBM", "MSFT", "INTC"], '2020-01-01', '2020-12-31')["Close"]


# In[5]:


# Calculating logarithmic return; normalising data so they are comparable
stocks_returns = (np.log(stocks) - np.log(stocks.shift(1))).dropna()


# In[6]:


stocks


# In[7]:


stocks_returns


# # Modelling
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - VaR can be measured via three different approaches:
#     - Variance-covariance VaR
#     - Historical simulation VaR
#     - Monte Carlo VaR
# 
# </font>
# </div>

# ## Variance-Covariance Method

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - The variance-covariance method is also known as the parametric method, because observations are assumed to be normally distributed. 
# - The normality assumption makes things easier, but it is a strong assumption, as there is no guarantee that asset returns are normally distributed; rather, most asset returns do not follow a normal distribution.
# - This method changes depending on the time horizon in the sense that holding assets for a longer period makes an investor more susceptible to risk. The increase is poportional to squared-root of the holding period; we'll use 30 here. 
#     
# </font>
# </div>

# In[8]:


stocks_returns_mean = stocks_returns.mean()
# Drawing random numbers for weights
weights  = np.random.random(len(stocks_returns.columns))
# Generating weights
weights /= np.sum(weights)
# Calculating covariance matrix
cov_var = stocks_returns.cov()
# Finding the portfolio standard deviation
port_std = np.sqrt(weights.T.dot(cov_var).dot(weights))


# In[9]:


weights


# In[10]:


initial_investment = 1e6
conf_level = 0.95


# In[11]:


def VaR_parametric(initial_investment, conf_level):
    # Computing the Z-score for a specific value using the percent point function (ppf)
    alpha = norm.ppf(1 - conf_level, stocks_returns_mean, port_std)
    for i, j in zip(stocks.columns, range(len(stocks.columns))):
        VaR_param = (initial_investment - initial_investment *
                     (1 + alpha))[j]
        print("Parametric VaR result for {} is {} "
              .format(i, VaR_param))
    VaR_param = (initial_investment - initial_investment * (1 + alpha))
    print('--' * 25)
    return VaR_param


# In[12]:


VaR_param = VaR_parametric(initial_investment, conf_level)
VaR_param


# In[13]:


var_horizon = []
time_horizon = 30
for j in range(len(stocks_returns.columns)):
    for i in range(1, time_horizon + 1):
        var_horizon.append(VaR_param[j] * np.sqrt(i))
plt.plot(var_horizon[:time_horizon], "o",
         c='blue', marker='*', label='IBM')
plt.plot(var_horizon[time_horizon:time_horizon + 30], "o",
         c='green', marker='o', label='MSFT')
plt.plot(var_horizon[time_horizon + 30:time_horizon + 60], "o",
         c='red', marker='v', label='INTC')
plt.xlabel("Days")
plt.ylabel("USD")
plt.title("VaR over 30-day period")
plt.legend()
plt.show()


# ## Historical Simulation VaR

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - There is another method that **does not** have a normality assumption, namely the historical simulation VaR model.
# - We find the percentile, which is the Z-table equivalent of variance-covariance method. Suppose that the confidence interval is 95%; 5% will be used in lieu of the Z-table values, and all we need to do is to multiply this percentile by the initial investment.
# - The downside is that it requires a large sample.
#     
# </font>
# </div>

# In[14]:


def VaR_historical(initial_investment, conf_level):
    Hist_percentile95 = []
    for i, j in zip(stocks_returns.columns,
                    range(len(stocks_returns.columns))):
        Hist_percentile95.append(np.percentile(stocks_returns.loc[:, i],
                                               5))
        print("Based on historical values 95% of {}'s return is {:.4f}"
              .format(i, Hist_percentile95[j]))
        VaR_historical = (initial_investment - initial_investment *
                          (1 + Hist_percentile95[j]))
        print("Historical VaR result for {} is {:.2f} "
              .format(i, VaR_historical))
        print('--' * 35)


# In[15]:


VaR_historical(initial_investment,conf_level)


# ## Monte Carlo VaR

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - From the application standpoint, Monte Carlo is very similar to the historical simulation VaR, but it does not use historical observations. 
#     
# - Rather, it generates random samples from a given distribution.
# 
# </font>
# </div>

# In[16]:


sim_data = pd.DataFrame([])
num_reps = 1000
n = 100
for i in range(len(stocks.columns)):
    mean = np.random.randn(n).mean()
    std = np.random.randn(n).std()
    temp = pd.DataFrame(np.random.normal(mean, std, num_reps))
    sim_data = pd.concat([sim_data,   temp], axis=1)
sim_data.columns = ['Simulation 1', 'Simulation 2', 'Simulation 3']


# In[17]:


sim_data


# In[18]:


def MC_VaR(initial_investment, conf_level):
    MC_percentile95 = []
    for i, j in zip(sim_data.columns, range(len(sim_data.columns))):
        MC_percentile95.append(np.percentile(sim_data.loc[:, i], 5))
        print("Based on simulation 95% of {}'s return is {:.4f}"
              .format(i, MC_percentile95[j]))
        VaR_MC = (initial_investment - initial_investment * 
                  (1 + MC_percentile95[j]))
        print("Simulation VaR result for {} is {:.2f} "
              .format(i, VaR_MC))
        print('--' * 35)


# In[19]:


MC_VaR(initial_investment, conf_level)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_5.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# - https://www.quantstart.com/articles/Value-at-Risk-VaR-for-Algorithmic-Trading-Risk-Management-Part-I/
#     
# </font>
# </div>

# In[ ]:




