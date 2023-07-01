#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Monte Carlo Simulation and Efficient Frontier
# 
# </font>
# </div>

# # Action plan
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Introduction to Monte Carlo Simulation
# - Applying Monte Carlo Simulation on portfolios using Sharpe Ratio
# - Creating Efficient Frontier based on Sharpe Ratio
# 
# </font>
# </div>

# # Monte Carlo
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - It can be used to simulate risk and uncertainty that can affect the outcome of different decision options.
# 
# - If there are **too many** variables affecting the outcome, then it can simulate them and find the optimal based on the values.
# 
# - MCS are used to model the probability of different outcomes in a process that cannot easily be predicted **due to** the intervention of random variables. 
#     
# - It is a technique used to understand **the impact of** risk and uncertainty in prediction and forecasting models.
# 
# </font>
# </div>
# 
# 

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[57]:


import numpy as np
import pandas_datareader as pdr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


# # Load data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Let's pull some data time series of historic stock prices from YAHOO.
# - This can be easily done with Pandas Datareader.
# - This will read historic stock prices from Apple (`AAPL`) starting from 2020 and up until today. 
# 
# </font>
# </div>

# In[48]:


tickers = ['AAPL', 'MSFT', 'TWTR', 'IBM']
start = dt.datetime(2020, 1, 1)

data = pdr.get_data_yahoo(tickers, start)


# In[49]:


data.head()


# In[42]:


data = data['Adj Close']


# In[46]:


data.head()


# In[44]:


data.shape


# # Sharpe Ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The Sharpe Ratio combines Risk and Return in one number.
# 
# - The Sharpe Ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk. 
#     
# - **Volatility** is a measure of the price fluctuations of an asset or portfolio (source).
# 
# </font>
# </div>

# In[27]:


# To use it with Sharpe Ratio, we will calculate the log returns.
log_returns = np.log(data/data.shift())


# In[28]:


log_returns


# In[29]:


weight = np.random.random(4)
weight /= weight.sum()
weight


# In[30]:


exp_rtn = np.sum(log_returns.mean()*weight)*252


# In[31]:


exp_vol = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov()*252, weight)))


# In[32]:


sharpe_ratio = exp_rtn / exp_vol


# In[33]:


sharpe_ratio


# # Monte Carlo Simulation
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The code will run 5000 experiments. 
# - We will keep all the data from each run: 
#     - The weights of the portfolios (weights)
#     - The expected return (exp_rtns)
#     - The expected volatility (exp_vols)
#     - The Sharpe Ratio (sharpe_ratios)
# 
# </font>
# </div>

# In[51]:


n = 5000

weights = np.zeros((n, 4))
exp_rtns = np.zeros(n)
exp_vols = np.zeros(n)
sharpe_ratios = np.zeros(n)

for i in range(n):
    weight = np.random.random(4)
    weight /= weight.sum()
    weights[i] = weight

    exp_rtns[i] = np.sum(log_returns.mean()*weight)*252
    exp_vols[i] = np.sqrt(
        np.dot(weight.T, np.dot(log_returns.cov()*252, weight)))
    sharpe_ratios[i] = exp_rtns[i] / exp_vols[i]


# # Post-processing
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - We now want to know the maximum Sharpe Ratio is.
# - Once that is know we'd like to know the weights corresponding to this.
# - The weights are telling you the percentage of stocks you corresponding to the best Sharpe Ratio. 
# - For instance a portfoili could hold 45.7% to AAPL, 6.7% to MSFT, 47.5% to TWTR, and 0,03% to IBM is optimal.
# - **Pay attention** that due to the random component, everytime you ran the MCS, you will get some slightly different values.
# 
# </font>
# </div>

# In[52]:


sharpe_ratios.max()


# In[54]:


sharpe_ratios.argmax()


# In[55]:


weights[sharpe_ratios.argmax()]


# # Efficient Frontier
# <hr style = "border:2px solid black" ></hr>

# In[75]:


fig, ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(exp_vols, exp_rtns, c=sharpe_ratios, s=35, cmap=cm)

ax.scatter(exp_vols[sharpe_ratios.argmax()], exp_rtns[sharpe_ratios.argmax()], c='m')

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')

plt.colorbar(sc)
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [Monte Carlo Simulation Wikipedia article](https://en.wikipedia.org/wiki/Monte_Carlo_method)
# - [Source code on Github](https://github.com/LearnPythonWithRune/PythonForFinanceRiskAndReturn/blob/main/04%20-%20Monte%20Carlo%20Simulation%20and%20Efficient%20Frontier.ipynb)
# - [Blog article](https://www.learnpythonwithrune.org/monte-carlo-simulation-to-optimize-a-portfolio-using-pandas-and-numpy/)
# 
# </font>
# </div>

# In[ ]:




