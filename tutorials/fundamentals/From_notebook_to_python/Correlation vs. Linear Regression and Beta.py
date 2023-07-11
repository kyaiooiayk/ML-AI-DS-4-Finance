#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Correlation-vs-Linear-Regression" data-toc-modified-id="Correlation-vs-Linear-Regression-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Correlation vs Linear Regression</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Generate-uncorrelated-synthetic-data" data-toc-modified-id="Generate-uncorrelated-synthetic-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Generate uncorrelated synthetic data</a></span></li><li><span><a href="#Load-some-financial-real-data" data-toc-modified-id="Load-some-financial-real-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Load some financial real data</a></span></li><li><span><a href="#Linear-regression" data-toc-modified-id="Linear-regression-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Linear regression</a></span></li><li><span><a href="#Linear-regression-and-beta" data-toc-modified-id="Linear-regression-and-beta-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Linear regression and beta</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Correlation vs. Linear Regression
#
# </font>
# </div>

# # Correlation vs Linear Regression
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# **Similarities**
# - Quantify the direction and strength of the relationship
#
# **Differences**
# - Correlation is a single statistic
# - Linear regression produces an equation
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt


# # Generate uncorrelated synthetic data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - A great way to learn about relationships between variables is to compare it to random variables.
# - Rhe plot below shows how two non-correlated variables look like.
#
# </font>
# </div>

# In[12]:


X = np.random.randn(5000)
Y = np.random.randn(5000)

fig, ax = plt.subplots()
ax.scatter(X, Y, alpha=0.2)


# # Load some financial real data
# <hr style = "border:2px solid black" ></hr>

# In[3]:


tickers = ["AAPL", "TWTR", "IBM", "MSFT", "^GSPC"]
start = dt.datetime(2020, 1, 1)

data = pdr.get_data_yahoo(tickers, start)


# In[4]:


data = data["Adj Close"]


# In[5]:


data.head()


# In[6]:


log_returns = np.log(data / data.shift())


# In[7]:


log_returns


# # Linear regression
# <hr style = "border:2px solid black" ></hr>

# In[8]:


def linear_regression(ticker_a, ticker_b):
    X = log_returns[ticker_a].iloc[1:].to_numpy().reshape(-1, 1)
    Y = log_returns[ticker_b].iloc[1:].to_numpy().reshape(-1, 1)

    lin_regr = LinearRegression()
    lin_regr.fit(X, Y)

    Y_pred = lin_regr.predict(X)

    alpha = lin_regr.intercept_[0]
    beta = lin_regr.coef_[0, 0]

    fig, ax = plt.subplots()
    ax.set_title("Alpha: " + str(round(alpha, 5)) + ", Beta: " + str(round(beta, 3)))
    ax.scatter(X, Y)
    ax.plot(X, Y_pred, c="r")


# In[9]:


linear_regression("AAPL", "^GSPC")


# In[10]:


linear_regression("AAPL", "MSFT")


# In[11]:


linear_regression("AAPL", "TWTR")


# # Linear regression and beta
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Beta is a measure of a stock’s volatility in relation to the overall market (S&P 500).
#
# - **High-beta** stocks are supposed to be riskier but provide higher potential return.
# - **Low-beta** stocks pose less risk but also lower returns.
#
# - `Beta < 1` stock is more volatile than the market, but expects higher return.
# - `Beta > 1` stock with lower volatility, and expects less return.
#
# - Beta is `Covariance / variance` and is generally calculated on the monthly price.
#
# </font>
# </div>

# In[13]:


tickers = ["AAPL", "MSFT", "TWTR", "IBM", "^GSPC"]
start = dt.datetime(2015, 12, 1)
end = dt.datetime(2021, 1, 1)
# m stands for month
data = pdr.get_data_yahoo(tickers, start, end, interval="m")
data = data["Adj Close"]


# In[14]:


# Calculating  beta
log_returns = np.log(data / data.shift())
cov = log_returns.cov()
var = log_returns["^GSPC"].var()
cov.loc["AAPL", "^GSPC"] / var


# <div class="alert alert-info">
# <font color=black>
#
# - Is this beta related to the Beta value from Linear Regression?
# - Yes, it is and below is the proof.
#
# </font>
# </div>

# In[15]:


X = log_returns["^GSPC"].iloc[1:].to_numpy().reshape(-1, 1)
Y = log_returns["AAPL"].iloc[1:].to_numpy().reshape(-1, 1)
lin_regr = LinearRegression()
lin_regr.fit(X, Y)
lin_regr.coef_[0, 0]


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://github.com/LearnPythonWithRune/PythonForFinanceRiskAndReturn/blob/main/06%20-%20Linear%20Regression.ipynb
#
# </font>
# </div>

# In[ ]:
