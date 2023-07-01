#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1">Introduction</a></span></li><li><span><a href="#Expected-shortfall" data-toc-modified-id="Expected-shortfall-2">Expected shortfall</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3">Imports</a></span></li><li><span><a href="#Loading-the-data" data-toc-modified-id="Loading-the-data-4">Loading the data</a></span></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-5">Modelling</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Expected shortfall
# 
# </font>
# </div>

# # Expected shortfall
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - Unlike VaR (Value at risk), ES (Expected Shortfall) focuses on the tail of the distribution. More specifically, ES enables us to take into account unexpected risks in the market. However, this doesn’t mean that ES and VaR are two entirely different concepts. Rather, they are related—that is, it is possible to express ES using VaR.
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


def ES_parametric(initial_investment, conf_level):

    alpha = - norm.ppf(1 - conf_level, stocks_returns_mean, port_std) 
    for i, j in zip(stocks.columns, range(len(stocks.columns))):
        VaR_param = (initial_investment * alpha)[j]
        ES_param = (1 / (1 - conf_level))             * initial_investment             * norm.expect(lambda x: x,
                          lb=norm.ppf(conf_level,
                                      stocks_returns_mean[j],
                                      port_std),
                          loc=stocks_returns_mean[j],
                          scale=port_std) 
        print(f"Parametric ES result for {i} is {ES_param}")


# In[12]:


ES_parametric(initial_investment, conf_level)


# <div class="alert alert-info">
# <font color=bla ck>
# 
# - ES can also be computed based on the historical observations. 
# - Like the historical simulation VaR method, parametric assumption can be relaxed.
# - To do that, the first return (or loss) corresponding to the 95% is found, and then the mean of the observations greater than the 95% gives us the result.
# 
# </font>
# </div>

# In[13]:


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


# In[14]:


VaR_historical(initial_investment, conf_level)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_5.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# 
# </font>
# </div>

# In[ ]:




